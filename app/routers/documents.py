import asyncio
import hashlib
import uuid
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, Query, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.hardening import analytics_report_cache, sanitize_header_filename
from app.core.workspace import WorkspaceContext, get_verified_workspace_context
from app.db.supabase_client import supabase_db
from app.models.schemas import DocumentInfo, DocumentChunk, DocumentUpdateRequest, BoundingBox
from app.services.pdf_service import pdf_processor
from app.services.storage_service import storage_service
from app.worker import dispatch_ingestion_job

router = APIRouter(prefix="/documents", tags=["documents"])


def _row_to_document_info(row: Dict[str, Any], is_duplicate: bool = False) -> DocumentInfo:
    st = row.get("status") or row.get("processing_status") or "queued"
    proc_status = "completed" if st in ("ready", "partial_ready") else st
    if proc_status not in ("pending", "queued", "processing", "ready", "partial_ready", "completed", "failed"):
        proc_status = "completed"

    return DocumentInfo(
        id=str(row["id"]),
        workspace_id=str(row.get("workspace_id") or ""),
        title=row.get("title") or "Untitled Document",
        original_filename=row.get("original_filename") or row.get("title"),
        file_type=row.get("file_type") or "pdf",
        mime_type=row.get("mime_type") or "application/pdf",
        page_count=int(row.get("page_count") or 1),
        file_size_bytes=int(row.get("file_size_bytes") or 0),
        processing_status=proc_status,
        status=st,
        category=row.get("category") or "general_document",
        primary_domain=row.get("primary_domain"),
        language_code=row.get("language_code") or "id",
        summary=row.get("summary"),
        key_findings=row.get("key_findings") or [],
        main_topics=row.get("main_topics") or [],
        user_tags=row.get("user_tags") or [],
        has_tables=bool(row.get("has_tables", False)),
        table_count=int(row.get("table_count") or 0),
        metric_count=int(row.get("metric_count") or 0),
        is_duplicate=is_duplicate,
        error_message=row.get("error_message"),
        created_at=str(row.get("created_at")) if row.get("created_at") else None,
        processed_at=str(row.get("processed_at")) if row.get("processed_at") else None,
    )


@router.get("", response_model=List[DocumentInfo])
async def list_documents(
    search: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    tag: Optional[str] = Query(default=None),
    has_tables: Optional[bool] = Query(default=None),
    sort_by: str = Query(default="created_at"),
    sort_dir: str = Query(default="desc"),
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    rows = supabase_db.list_workspace_documents(
        ctx.workspace_id,
        search=search,
        status=status,
        category=category,
        tag=tag,
        has_tables=has_tables,
        sort_by=sort_by,
        sort_dir=sort_dir,
    )
    return [_row_to_document_info(r) for r in rows]


@router.get("/summary")
async def get_repository_summary(
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    rows = supabase_db.list_workspace_documents(ctx.workspace_id)
    total_bytes = sum(int(r.get("file_size_bytes") or 0) for r in rows)
    categories: Dict[str, int] = {}
    tags_set: set[str] = set()
    ready_count = 0
    processing_count = 0
    failed_count = 0

    for r in rows:
        st = r.get("status") or r.get("processing_status")
        if st in ("ready", "partial_ready", "completed"):
            ready_count += 1
        elif st in ("queued", "pending", "processing"):
            processing_count += 1
        elif st == "failed":
            failed_count += 1

        cat = r.get("category") or "general_document"
        categories[cat] = categories.get(cat, 0) + 1
        for t in (r.get("user_tags") or []) + (r.get("main_topics") or []):
            if t:
                tags_set.add(t)

    return {
        "workspace_id": ctx.workspace_id,
        "total_documents": len(rows),
        "ready_count": ready_count,
        "processing_count": processing_count,
        "failed_count": failed_count,
        "total_storage_bytes": total_bytes,
        "storage_limit_bytes": settings.default_workspace_storage_limit_mb * 1024 * 1024,
        "categories": categories,
        "available_tags": sorted(tags_set)[:50],
    }


async def _read_upload_bounded(file: UploadFile, max_bytes: int) -> bytes:
    """Stream upload in 1MB chunks and abort immediately if file exceeds max_bytes (anti-RAM-exhaustion)."""
    chunk_size = 1024 * 1024
    buf = bytearray()
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds the {settings.max_upload_size_mb} MB upload limit.",
            )
    return bytes(buf)


@router.post("/upload", response_model=DocumentInfo)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    force_replace: bool = Form(default=False),
    run_sync: bool = Form(default=False),
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    """
    Stages 1 & 2 Synchronous Upload & Validation + Asynchronous 15-Stage Pipeline Dispatch (DOC-10).
    Uses bounded chunked reads, storage quota checks, and non-blocking thread offload for fast uploads.
    """
    max_bytes = int(settings.max_upload_size_mb) * 1024 * 1024
    file_bytes = await _read_upload_bounded(file, max_bytes)
    try:
        validated = await asyncio.to_thread(
            pdf_processor.validate_upload_bytes, file.filename or "", file_bytes
        )
    except ValueError as exc:
        msg = str(exc)
        status_code = 422 if ("encrypted" in msg.lower() or "password" in msg.lower()) else 400
        raise HTTPException(status_code=status_code, detail=msg) from exc

    file_sha256 = hashlib.sha256(file_bytes).hexdigest()
    existing = supabase_db.find_document_by_sha256(ctx.workspace_id, file_sha256)
    if existing and not force_replace:
        return _row_to_document_info(existing, is_duplicate=True)

    # Enforce workspace storage quota (anti-spam / anti-boros storage protection)
    storage_limit_bytes = int(settings.default_workspace_storage_limit_mb) * 1024 * 1024
    existing_docs = supabase_db.list_workspace_documents(ctx.workspace_id)
    used_bytes = sum(int(d.get("file_size_bytes") or 0) for d in existing_docs)
    if existing and force_replace:
        used_bytes = max(0, used_bytes - int(existing.get("file_size_bytes") or 0))
    if used_bytes + len(file_bytes) > storage_limit_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Workspace storage limit ({settings.default_workspace_storage_limit_mb} MB) reached. "
                "Delete unused files before uploading."
            ),
        )

    if existing and force_replace:
        doc_id = str(existing["id"])
        next_version = int(existing.get("ingestion_version") or 1) + 1
    else:
        doc_id = str(uuid.uuid4())
        next_version = 1

    storage_path, local_path = await asyncio.to_thread(
        storage_service.save_document_bytes,
        workspace_id=ctx.workspace_id,
        doc_id=doc_id,
        filename=file.filename or "document.pdf",
        file_bytes=file_bytes,
        mime_type=validated["mime_type"],
    )

    doc_row, job_row = supabase_db.create_document_and_job(
        ctx.workspace_id,
        {
            "id": doc_id,
            "uploaded_by": ctx.user_id,
            "title": validated["inferred_title"],
            "original_filename": file.filename,
            "storage_path": storage_path,
            "file_type": validated["file_type"],
            "mime_type": validated["mime_type"],
            "file_sha256": file_sha256,
            "page_count": validated["page_count"],
            "file_size_bytes": validated["file_size_bytes"],
            "status": "queued",
            "processing_status": "queued",
            "ingestion_version": next_version,
        },
    )

    await dispatch_ingestion_job(
        background_tasks=None if run_sync else background_tasks,
        workspace_id=ctx.workspace_id,
        doc_id=doc_id,
        job_id=str(job_row["id"]),
        local_file_path=local_path,
        user_id=ctx.user_id,
        expected_version=next_version,
        run_inline_if_no_bg=True,
    )

    analytics_report_cache.invalidate_prefix(f"an:{ctx.workspace_id}:")
    latest_doc = supabase_db.get_workspace_document(ctx.workspace_id, doc_id) or doc_row
    return _row_to_document_info(latest_doc)


@router.get("/{document_id}")
async def get_document(
    document_id: str,
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    doc = supabase_db.get_workspace_document(ctx.workspace_id, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found in your workspace.")

    info = _row_to_document_info(doc)
    tables = supabase_db.get_workspace_tables(ctx.workspace_id, doc_ids=[document_id])
    metrics = supabase_db.get_workspace_metrics(ctx.workspace_id, doc_ids=[document_id])
    latest_job = supabase_db.get_latest_job_for_document(ctx.workspace_id, document_id)

    payload = info.model_dump()
    payload["extracted_tables"] = tables
    payload["extracted_metrics"] = metrics
    payload["processing_job"] = latest_job
    return payload


@router.get("/{document_id}/status")
async def get_document_processing_status(
    document_id: str,
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    doc = supabase_db.get_workspace_document(ctx.workspace_id, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found in your workspace.")
    job = supabase_db.get_latest_job_for_document(ctx.workspace_id, document_id)
    return {
        "document_id": document_id,
        "status": doc.get("status") or doc.get("processing_status") or "queued",
        "current_stage": (job or {}).get("current_stage", "completed"),
        "progress_pct": int((job or {}).get("progress_pct", 100)),
        "error_message": doc.get("error_message") or (job or {}).get("last_error_message"),
    }


@router.patch("/{document_id}", response_model=DocumentInfo)
async def update_document_metadata(
    document_id: str,
    req: DocumentUpdateRequest,
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    doc = supabase_db.get_workspace_document(ctx.workspace_id, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found in your workspace.")

    updates: Dict[str, Any] = {}
    if req.title is not None and req.title.strip():
        updates["title"] = req.title.strip()
    if req.category is not None:
        updates["category"] = req.category.strip()
    if req.user_tags is not None:
        updates["user_tags"] = [t.strip() for t in req.user_tags if t and t.strip()]

    updated = supabase_db.update_document(ctx.workspace_id, document_id, updates) or {**doc, **updates}
    analytics_report_cache.invalidate_prefix(f"an:{ctx.workspace_id}:")
    return _row_to_document_info(updated)


@router.post("/{document_id}/reprocess", response_model=DocumentInfo)
async def reprocess_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    run_sync: bool = Query(default=False),
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    doc = supabase_db.get_workspace_document(ctx.workspace_id, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found in your workspace.")

    storage_path = doc.get("storage_path") or ""
    local_path = storage_service.resolve_local_file_for_reading(ctx.workspace_id, storage_path)
    if not local_path:
        raise HTTPException(status_code=404, detail="Source binary file could not be resolved from storage.")

    next_version = int(doc.get("ingestion_version") or 1) + 1
    supabase_db.update_document(
        ctx.workspace_id,
        document_id,
        {"status": "queued", "processing_status": "queued", "ingestion_version": next_version, "error_message": None},
    )
    job = supabase_db.get_latest_job_for_document(ctx.workspace_id, document_id)
    job_id = str(job["id"]) if job else str(uuid.uuid4())

    await dispatch_ingestion_job(
        background_tasks=None if run_sync else background_tasks,
        workspace_id=ctx.workspace_id,
        doc_id=document_id,
        job_id=job_id,
        local_file_path=local_path,
        user_id=ctx.user_id,
        expected_version=next_version,
        run_inline_if_no_bg=True,
    )
    refreshed = supabase_db.get_workspace_document(ctx.workspace_id, document_id) or doc
    return _row_to_document_info(refreshed)


@router.get("/{document_id}/file")
async def get_document_file(
    document_id: str,
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    doc = supabase_db.get_workspace_document(ctx.workspace_id, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found in your workspace.")

    storage_path = doc.get("storage_path") or ""
    try:
        local_path = storage_service.resolve_local_file_for_reading(ctx.workspace_id, storage_path)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    if not local_path:
        raise HTTPException(status_code=404, detail="Document binary missing in storage.")

    media_type = doc.get("mime_type") or "application/pdf"
    safe_title = sanitize_header_filename(
        doc.get("original_filename") or doc.get("title") or "document.pdf"
    )
    headers = {
        "Content-Disposition": f'inline; filename="{safe_title}"',
        "Cache-Control": "private, max-age=300",
    }
    file_sha = doc.get("file_sha256")
    if file_sha:
        headers["ETag"] = f'"{file_sha[:32]}"'
    return FileResponse(
        path=local_path,
        media_type=media_type,
        headers=headers,
    )


@router.get("/{document_id}/chunks", response_model=List[DocumentChunk])
async def get_document_chunks(
    document_id: str,
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    doc = supabase_db.get_workspace_document(ctx.workspace_id, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found in your workspace.")

    raw_chunks = supabase_db.get_document_chunks(ctx.workspace_id, document_id)
    result: List[DocumentChunk] = []
    for c in raw_chunks:
        raw_boxes = c.get("bounding_boxes") or []
        boxes = [
            BoundingBox(**b) if isinstance(b, dict) else b
            for b in raw_boxes
        ]
        txt = c.get("text_content") or c.get("content") or ""
        result.append(
            DocumentChunk(
                id=str(c["id"]),
                document_id=document_id,
                doc_id=document_id,
                workspace_id=ctx.workspace_id,
                chunk_index=int(c.get("chunk_index", 0)),
                content=txt,
                text_content=txt,
                page_number=int(c.get("page_number", 1)),
                bounding_boxes=boxes,
                token_count=c.get("token_count"),
                has_table=bool(c.get("has_table", False)),
                metadata=c.get("metadata") or {},
            )
        )
    return result


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    doc = supabase_db.delete_workspace_document(ctx.workspace_id, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found in your workspace.")

    storage_path = doc.get("storage_path")
    if storage_path:
        try:
            storage_service.delete_document_file(ctx.workspace_id, storage_path)
        except Exception:
            pass

    analytics_report_cache.invalidate_prefix(f"an:{ctx.workspace_id}:")
    return {
        "deleted": True,
        "id": document_id,
        "workspace_id": ctx.workspace_id,
    }
