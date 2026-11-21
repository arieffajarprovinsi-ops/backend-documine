import math
import re
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from app.core.config import settings


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cosine_similarity(vec_a: Optional[List[float]], vec_b: Optional[List[float]]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1.0
    norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1.0
    return float(dot / (norm_a * norm_b))


class SupabaseManager:
    """
    Dual-Layer Multi-Tenant Data Access Layer for Supabase PostgreSQL + pgvector (ADR-002, ADR-003).
    Enforces mandatory workspace_id scoping on every query (never relying on service_role RLS alone)
    and provides a schema-faithful local in-memory tenant-isolated store when running offline or in tests.
    """

    def __init__(self):
        self.client = None
        self._local_tables: Dict[str, List[Dict[str, Any]]] = {
            "workspaces": [],
            "workspace_members": [],
            "documents": [],
            "document_pages": [],
            "chunks": [],
            "document_tables": [],
            "extracted_metrics": [],
            "topics": [],
            "chunk_topics": [],
            "entities": [],
            "document_entities": [],
            "document_relationships": [],
            "conversations": [],
            "messages": [],
            "message_citations": [],
            "dashboards": [],
            "dashboard_widgets": [],
            "processing_jobs": [],
            "ai_usage_logs": [],
        }
        self._init_client()

    def _init_client(self):
        service_key = settings.effective_supabase_service_key
        if settings.supabase_url and service_key:
            try:
                from supabase import create_client, Client

                self.client: Optional[Client] = create_client(
                    settings.supabase_url,
                    service_key,
                )
            except Exception as e:
                print(
                    f"[Supabase] Warning: Failed to connect to remote Supabase ({e}). Using local tenant-isolated store."
                )
                self.client = None
        else:
            self.client = None

    @property
    def is_connected(self) -> bool:
        return self.client is not None

    # -------------------------------------------------------------------------
    # 1. Workspace Provisioning & Membership Resolution (ADR-003, PR-006, PR-007)
    # -------------------------------------------------------------------------
    def ensure_personal_workspace(
        self,
        user_id: str,
        email: str = "",
        username: str = "",
        requested_workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not user_id:
            raise ValueError("user_id is required to resolve workspace context.")

        display_name = username or (email.split("@")[0] if email else "User")
        default_ws_name = f"{display_name}'s Workspace"

        if self.is_connected:
            try:
                if requested_workspace_id:
                    member_res = (
                        self.client.table("workspace_members")
                        .select("workspace_id, role, workspaces(id, name, owner_id, is_personal)")
                        .eq("workspace_id", requested_workspace_id)
                        .eq("user_id", user_id)
                        .limit(1)
                        .execute()
                    )
                    if member_res.data:
                        row = member_res.data[0]
                        ws_info = row.get("workspaces") or {}
                        return {
                            "workspace_id": row["workspace_id"],
                            "workspace_name": ws_info.get("name") or default_ws_name,
                            "role": row.get("role") or "member",
                        }
                    raise PermissionError("Access denied to the requested workspace.")

                member_res = (
                    self.client.table("workspace_members")
                    .select("workspace_id, role, workspaces(id, name, owner_id, is_personal)")
                    .eq("user_id", user_id)
                    .limit(1)
                    .execute()
                )
                if member_res.data:
                    row = member_res.data[0]
                    ws_info = row.get("workspaces") or {}
                    return {
                        "workspace_id": row["workspace_id"],
                        "workspace_name": ws_info.get("name") or default_ws_name,
                        "role": row.get("role") or "owner",
                    }

                ws_res = (
                    self.client.table("workspaces")
                    .select("id, name, owner_id")
                    .eq("owner_id", user_id)
                    .limit(1)
                    .execute()
                )
                if ws_res.data:
                    ws = ws_res.data[0]
                    self.client.table("workspace_members").upsert(
                        {
                            "workspace_id": ws["id"],
                            "user_id": user_id,
                            "role": "owner",
                        }
                    ).execute()
                    return {
                        "workspace_id": ws["id"],
                        "workspace_name": ws["name"],
                        "role": "owner",
                    }

                created_ws = (
                    self.client.table("workspaces")
                    .insert(
                        {
                            "owner_id": user_id,
                            "name": default_ws_name,
                            "is_personal": True,
                        }
                    )
                    .execute()
                )
                ws_id = created_ws.data[0]["id"]
                self.client.table("workspace_members").upsert(
                    {
                        "workspace_id": ws_id,
                        "user_id": user_id,
                        "role": "owner",
                    }
                ).execute()
                self.client.table("dashboards").upsert(
                    {
                        "workspace_id": ws_id,
                        "user_id": user_id,
                        "detected_corpus_profile": "empty_corpus",
                    },
                    on_conflict="workspace_id,user_id",
                ).execute()
                return {
                    "workspace_id": ws_id,
                    "workspace_name": default_ws_name,
                    "role": "owner",
                }
            except PermissionError:
                raise
            except Exception as e:
                print(f"[Supabase] Workspace lookup fallback to local store due to error: {e}")

        if requested_workspace_id:
            for wm in self._local_tables["workspace_members"]:
                if wm["workspace_id"] == requested_workspace_id and wm["user_id"] == user_id:
                    ws = next(
                        (w for w in self._local_tables["workspaces"] if w["id"] == requested_workspace_id),
                        {"name": default_ws_name},
                    )
                    return {
                        "workspace_id": requested_workspace_id,
                        "workspace_name": ws["name"],
                        "role": wm["role"],
                    }
            raise PermissionError("Access denied to the requested workspace.")

        for wm in self._local_tables["workspace_members"]:
            if wm["user_id"] == user_id:
                ws = next(
                    (w for w in self._local_tables["workspaces"] if w["id"] == wm["workspace_id"]),
                    {"name": default_ws_name},
                )
                return {
                    "workspace_id": wm["workspace_id"],
                    "workspace_name": ws["name"],
                    "role": wm["role"],
                }

        ws_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"documind:workspace:{user_id}"))
        ws_record = {
            "id": ws_id,
            "owner_id": user_id,
            "name": default_ws_name,
            "is_personal": True,
            "settings": {},
            "created_at": _utc_now_iso(),
        }
        self._local_tables["workspaces"].append(ws_record)
        self._local_tables["workspace_members"].append(
            {
                "workspace_id": ws_id,
                "user_id": user_id,
                "role": "owner",
                "joined_at": _utc_now_iso(),
            }
        )
        self._local_tables["dashboards"].append(
            {
                "id": str(uuid.uuid4()),
                "workspace_id": ws_id,
                "user_id": user_id,
                "detected_corpus_profile": "empty_corpus",
                "layout_preferences": {},
                "created_at": _utc_now_iso(),
            }
        )
        return {
            "workspace_id": ws_id,
            "workspace_name": default_ws_name,
            "role": "owner",
        }

    # -------------------------------------------------------------------------
    # 2. Documents & Processing Jobs Lifecycle (Pillar A - DOC-09, DOC-10, DOC-24)
    # -------------------------------------------------------------------------
    def find_document_by_sha256(self, workspace_id: str, file_sha256: str) -> Optional[Dict[str, Any]]:
        if not workspace_id or not file_sha256:
            return None
        if self.is_connected:
            try:
                res = (
                    self.client.table("documents")
                    .select("*")
                    .eq("workspace_id", workspace_id)
                    .eq("file_sha256", file_sha256)
                    .limit(1)
                    .execute()
                )
                if res.data:
                    return res.data[0]
            except Exception:
                pass
        for doc in self._local_tables["documents"]:
            if doc.get("workspace_id") == workspace_id and doc.get("file_sha256") == file_sha256:
                return dict(doc)
        return None

    def create_document_and_job(
        self,
        workspace_id: str,
        doc_data: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if not workspace_id:
            raise ValueError("workspace_id is required.")
        now = _utc_now_iso()
        doc_id = doc_data.get("id") or str(uuid.uuid4())
        doc_row: Dict[str, Any] = {
            "id": doc_id,
            "workspace_id": workspace_id,
            "uploaded_by": doc_data.get("uploaded_by"),
            "title": doc_data["title"],
            "original_filename": doc_data.get("original_filename") or doc_data["title"],
            "storage_path": doc_data["storage_path"],
            "file_type": doc_data.get("file_type", "pdf"),
            "mime_type": doc_data.get("mime_type", "application/pdf"),
            "file_sha256": doc_data.get("file_sha256"),
            "page_count": int(doc_data.get("page_count", 1)),
            "file_size_bytes": int(doc_data.get("file_size_bytes", 0)),
            "processing_status": doc_data.get("processing_status", "queued"),
            "status": doc_data.get("status", "queued"),
            "category": doc_data.get("category", "general_document"),
            "primary_domain": doc_data.get("primary_domain"),
            "language_code": doc_data.get("language_code", "id"),
            "summary": doc_data.get("summary"),
            "key_findings": doc_data.get("key_findings", []),
            "main_topics": doc_data.get("main_topics", []),
            "key_dates": doc_data.get("key_dates", []),
            "user_tags": doc_data.get("user_tags", []),
            "has_tables": bool(doc_data.get("has_tables", False)),
            "table_count": int(doc_data.get("table_count", 0)),
            "metric_count": int(doc_data.get("metric_count", 0)),
            "ingestion_version": int(doc_data.get("ingestion_version", 1)),
            "metadata": doc_data.get("metadata", {}),
            "created_at": now,
            "updated_at": now,
        }

        job_id = str(uuid.uuid4())
        job_row: Dict[str, Any] = {
            "id": job_id,
            "workspace_id": workspace_id,
            "doc_id": doc_id,
            "status": "queued",
            "current_stage": "stage_2_stored",
            "progress_pct": 10,
            "attempt_count": 1,
            "checkpoint_state": {},
            "created_at": now,
            "started_at": now,
            "updated_at": now,
        }

        if self.is_connected:
            try:
                res_doc = self.client.table("documents").insert(doc_row).execute()
                if res_doc.data:
                    doc_row = res_doc.data[0]
                res_job = self.client.table("processing_jobs").insert(job_row).execute()
                if res_job.data:
                    job_row = res_job.data[0]
            except Exception as e:
                print(f"[Supabase] Notice: Storing document/job in local tenant store ({e}).")

        self._local_tables["documents"] = [
            d for d in self._local_tables["documents"] if d.get("id") != doc_id
        ] + [dict(doc_row)]
        self._local_tables["processing_jobs"] = [
            j for j in self._local_tables["processing_jobs"] if j.get("id") != job_id
        ] + [dict(job_row)]

        return doc_row, job_row

    def update_document(
        self,
        workspace_id: str,
        doc_id: str,
        updates: Dict[str, Any],
        expected_version: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        if not workspace_id or not doc_id:
            raise ValueError("workspace_id and doc_id are mandatory.")

        payload = {**updates, "updated_at": _utc_now_iso()}
        # Keep status and processing_status synchronized
        if "status" in payload and "processing_status" not in payload:
            st = payload["status"]
            payload["processing_status"] = "completed" if st in ("ready", "partial_ready") else st

        if self.is_connected:
            try:
                q = self.client.table("documents").update(payload).eq("id", doc_id).eq("workspace_id", workspace_id)
                if expected_version is not None:
                    q = q.eq("ingestion_version", expected_version)
                res = q.execute()
                if res.data:
                    for i, d in enumerate(self._local_tables["documents"]):
                        if d.get("id") == doc_id and d.get("workspace_id") == workspace_id:
                            self._local_tables["documents"][i].update(payload)
                    return res.data[0]
            except Exception:
                pass

        for doc in self._local_tables["documents"]:
            if doc.get("id") == doc_id and doc.get("workspace_id") == workspace_id:
                if expected_version is not None and doc.get("ingestion_version", 1) != expected_version:
                    return None
                doc.update(payload)
                return dict(doc)
        return None

    def update_processing_job(
        self,
        workspace_id: str,
        job_id: str,
        updates: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        payload = {**updates, "updated_at": _utc_now_iso()}
        if self.is_connected:
            try:
                res = (
                    self.client.table("processing_jobs")
                    .update(payload)
                    .eq("id", job_id)
                    .eq("workspace_id", workspace_id)
                    .execute()
                )
                if res.data:
                    for j in self._local_tables["processing_jobs"]:
                        if j.get("id") == job_id and j.get("workspace_id") == workspace_id:
                            j.update(payload)
                    return res.data[0]
            except Exception:
                pass

        for job in self._local_tables["processing_jobs"]:
            if job.get("id") == job_id and job.get("workspace_id") == workspace_id:
                job.update(payload)
                return dict(job)
        return None

    def get_latest_job_for_document(self, workspace_id: str, doc_id: str) -> Optional[Dict[str, Any]]:
        if self.is_connected:
            try:
                res = (
                    self.client.table("processing_jobs")
                    .select("*")
                    .eq("workspace_id", workspace_id)
                    .eq("doc_id", doc_id)
                    .order("created_at", desc=True)
                    .limit(1)
                    .execute()
                )
                if res.data:
                    return res.data[0]
            except Exception:
                pass
        jobs = [
            j
            for j in self._local_tables["processing_jobs"]
            if j.get("workspace_id") == workspace_id and j.get("doc_id") == doc_id
        ]
        return dict(jobs[-1]) if jobs else None

    def save_document_pages(self, workspace_id: str, doc_id: str, pages: List[Dict[str, Any]]) -> None:
        rows = [
            {
                "id": p.get("id") or str(uuid.uuid4()),
                "workspace_id": workspace_id,
                "doc_id": doc_id,
                "page_number": int(p["page_number"]),
                "width_pt": float(p.get("width_pt", 612.0)),
                "height_pt": float(p.get("height_pt", 792.0)),
                "digital_word_count": int(p.get("digital_word_count", 0)),
                "image_area_ratio": float(p.get("image_area_ratio", 0.0)),
                "needs_ocr": bool(p.get("needs_ocr", False)),
                "ocr_status": p.get("ocr_status", "not_needed"),
                "created_at": _utc_now_iso(),
            }
            for p in pages
        ]
        if self.is_connected and rows:
            try:
                self.client.table("document_pages").upsert(rows, on_conflict="doc_id,page_number").execute()
            except Exception:
                pass
        self._local_tables["document_pages"] = [
            p
            for p in self._local_tables["document_pages"]
            if not (p.get("workspace_id") == workspace_id and p.get("doc_id") == doc_id)
        ] + rows

    def replace_document_chunks(
        self,
        workspace_id: str,
        doc_id: str,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        now = _utc_now_iso()
        rows: List[Dict[str, Any]] = []
        for c in chunks:
            cid = c.get("id") or str(uuid.uuid4())
            txt = c.get("text_content") or c.get("content") or ""
            rows.append(
                {
                    "id": cid,
                    "workspace_id": workspace_id,
                    "document_id": doc_id,
                    "doc_id": doc_id,
                    "chunk_index": int(c.get("chunk_index", 0)),
                    "content": txt,
                    "text_content": txt,
                    "page_number": int(c.get("page_number", 1)),
                    "bounding_boxes": c.get("bounding_boxes", []),
                    "embedding": c.get("embedding"),
                    "embedding_model": c.get("embedding_model", "openai:text-embedding-3-large:1536"),
                    "token_count": int(c.get("token_count") or max(1, len(txt.split()))),
                    "has_table": bool(c.get("has_table", False)),
                    "metadata": c.get("metadata", {}),
                    "created_at": now,
                }
            )

        if self.is_connected:
            try:
                self.client.table("chunks").delete().eq("workspace_id", workspace_id).eq("doc_id", doc_id).execute()
                batch_size = 25
                for idx in range(0, len(rows), batch_size):
                    self.client.table("chunks").insert(rows[idx : idx + batch_size]).execute()
            except Exception as e:
                print(f"[Supabase] Notice: Chunks stored in local tenant store ({e}).")

        self._local_tables["chunks"] = [
            c
            for c in self._local_tables["chunks"]
            if not (c.get("workspace_id") == workspace_id and (c.get("doc_id") == doc_id or c.get("document_id") == doc_id))
        ] + rows
        return rows

    def replace_document_tables(
        self,
        workspace_id: str,
        doc_id: str,
        tables: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        now = _utc_now_iso()
        rows = [
            {
                "id": t.get("id") or str(uuid.uuid4()),
                "workspace_id": workspace_id,
                "doc_id": doc_id,
                "table_name": t["table_name"],
                "table_title": t.get("table_title") or t["table_name"],
                "page_number": int(t.get("page_number", 1)),
                "bbox": t.get("bbox", {}),
                "columns_schema": t.get("columns_schema", []),
                "rows_data": t.get("rows_data", []),
                "row_count": int(t.get("row_count", len(t.get("rows_data", [])))),
                "extraction_method": (
                    t.get("extraction_method")
                    if t.get("extraction_method") in ("pymupdf", "vision_terra", "spreadsheet")
                    else "pymupdf"
                ),
                "created_at": now,
            }
            for t in tables
        ]
        if self.is_connected:
            try:
                self.client.table("document_tables").delete().eq("workspace_id", workspace_id).eq("doc_id", doc_id).execute()
                if rows:
                    self.client.table("document_tables").insert(rows).execute()
            except Exception:
                pass
        self._local_tables["document_tables"] = [
            t
            for t in self._local_tables["document_tables"]
            if not (t.get("workspace_id") == workspace_id and t.get("doc_id") == doc_id)
        ] + rows
        return rows

    def replace_extracted_metrics(
        self,
        workspace_id: str,
        doc_id: str,
        metrics: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        now = _utc_now_iso()
        rows = [
            {
                "id": m.get("id") or str(uuid.uuid4()),
                "workspace_id": workspace_id,
                "doc_id": doc_id,
                "source_chunk_id": m.get("source_chunk_id"),
                "metric_name": m["metric_name"],
                "canonical_metric_key": m.get("canonical_metric_key")
                or re.sub(r"[^a-z0-9_]", "_", m["metric_name"].lower()).strip("_"),
                "numeric_value": float(m["numeric_value"]),
                "unit_or_currency": m["unit_or_currency"],
                "period_label": m.get("period_label"),
                "verbatim_quote": m.get("verbatim_quote"),
                "created_at": now,
            }
            for m in metrics
        ]
        if self.is_connected:
            try:
                self.client.table("extracted_metrics").delete().eq("workspace_id", workspace_id).eq("doc_id", doc_id).execute()
                if rows:
                    self.client.table("extracted_metrics").insert(rows).execute()
            except Exception:
                pass
        self._local_tables["extracted_metrics"] = [
            m
            for m in self._local_tables["extracted_metrics"]
            if not (m.get("workspace_id") == workspace_id and m.get("doc_id") == doc_id)
        ] + rows
        return rows

    def upsert_topics_and_entities(
        self,
        workspace_id: str,
        doc_id: str,
        topics: List[str],
        entities: List[Dict[str, Any]],
        representative_chunk_id: Optional[str] = None,
    ) -> None:
        now = _utc_now_iso()
        for raw_topic in topics:
            name = (raw_topic or "").strip()
            if not name:
                continue
            slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
            if not slug:
                continue
            topic_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"topic:{workspace_id}:{slug}"))
            if self.is_connected:
                try:
                    self.client.table("topics").upsert(
                        {
                            "id": topic_id,
                            "workspace_id": workspace_id,
                            "name": name,
                            "slug": slug,
                            "category": "concept",
                        },
                        on_conflict="id",
                    ).execute()
                    if representative_chunk_id:
                        self.client.table("chunk_topics").upsert(
                            {
                                "chunk_id": representative_chunk_id,
                                "topic_id": topic_id,
                                "workspace_id": workspace_id,
                                "doc_id": doc_id,
                                "relevance_score": 1.0,
                            }
                        ).execute()
                except Exception:
                    pass

            existing_t = next(
                (t for t in self._local_tables["topics"] if t.get("workspace_id") == workspace_id and t.get("slug") == slug),
                None,
            )
            if not existing_t:
                self._local_tables["topics"].append(
                    {
                        "id": topic_id,
                        "workspace_id": workspace_id,
                        "name": name,
                        "slug": slug,
                        "category": "concept",
                        "doc_count": 1,
                        "created_at": now,
                    }
                )
            if representative_chunk_id:
                self._local_tables["chunk_topics"].append(
                    {
                        "chunk_id": representative_chunk_id,
                        "topic_id": topic_id,
                        "workspace_id": workspace_id,
                        "doc_id": doc_id,
                        "relevance_score": 1.0,
                    }
                )

        for ent in entities:
            ename = (ent.get("name") or "").strip()
            canonical = (ent.get("canonical_name") or ename.lower()).strip()
            etype = ent.get("entity_type") or "other"
            if not ename or not canonical:
                continue
            entity_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"entity:{workspace_id}:{canonical}:{etype}"))
            if self.is_connected:
                try:
                    self.client.table("entities").upsert(
                        {
                            "id": entity_id,
                            "workspace_id": workspace_id,
                            "name": ename,
                            "canonical_name": canonical,
                            "entity_type": etype,
                        },
                        on_conflict="workspace_id,canonical_name,entity_type",
                    ).execute()
                    self.client.table("document_entities").upsert(
                        {
                            "workspace_id": workspace_id,
                            "doc_id": doc_id,
                            "entity_id": entity_id,
                            "representative_chunk_id": representative_chunk_id,
                            "mention_count": 1,
                        }
                    ).execute()
                except Exception:
                    pass

            if not any(
                e.get("workspace_id") == workspace_id
                and e.get("canonical_name") == canonical
                and e.get("entity_type") == etype
                for e in self._local_tables["entities"]
            ):
                self._local_tables["entities"].append(
                    {
                        "id": entity_id,
                        "workspace_id": workspace_id,
                        "name": ename,
                        "canonical_name": canonical,
                        "entity_type": etype,
                        "created_at": now,
                    }
                )
            self._local_tables["document_entities"] = [
                de
                for de in self._local_tables["document_entities"]
                if not (de.get("doc_id") == doc_id and de.get("entity_id") == entity_id)
            ] + [
                {
                    "workspace_id": workspace_id,
                    "doc_id": doc_id,
                    "entity_id": entity_id,
                    "representative_chunk_id": representative_chunk_id,
                    "mention_count": 1,
                }
            ]

    # -------------------------------------------------------------------------
    # 3. Document Relationship Discovery Engine (Stage 13 - DOC-16, ADR-013)
    # -------------------------------------------------------------------------
    def _get_doc_entity_ids(self, workspace_id: str, doc_id: str) -> set[str]:
        entity_ids: set[str] = set()
        if self.is_connected:
            try:
                res = (
                    self.client.table("document_entities")
                    .select("entity_id")
                    .eq("workspace_id", workspace_id)
                    .eq("doc_id", doc_id)
                    .execute()
                )
                for row in res.data or []:
                    if row.get("entity_id"):
                        entity_ids.add(str(row["entity_id"]))
            except Exception:
                pass
        for de in self._local_tables["document_entities"]:
            if de.get("workspace_id") == workspace_id and de.get("doc_id") == doc_id and de.get("entity_id"):
                entity_ids.add(str(de["entity_id"]))
        return entity_ids

    def compute_and_save_relationships(self, workspace_id: str, doc_id: str) -> List[Dict[str, Any]]:
        docs = self.list_workspace_documents(workspace_id)
        target_doc = next((d for d in docs if d["id"] == doc_id), None)
        if not target_doc:
            return []

        target_topics = {t.lower() for t in (target_doc.get("main_topics") or []) if t}
        target_entities = self._get_doc_entity_ids(workspace_id, doc_id)
        target_vec = target_doc.get("centroid_embedding")
        created_edges: List[Dict[str, Any]] = []
        now = _utc_now_iso()

        for other in docs:
            other_id = other["id"]
            if other_id == doc_id:
                continue
            other_topics = {t.lower() for t in (other.get("main_topics") or []) if t}
            shared_topics = sorted(target_topics.intersection(other_topics))
            other_entities = self._get_doc_entity_ids(workspace_id, other_id)
            shared_entity_count = len(target_entities.intersection(other_entities))

            sim = _cosine_similarity(target_vec, other.get("centroid_embedding"))
            if len(shared_topics) >= 1 or shared_entity_count >= 1 or sim >= 0.72:
                if shared_topics:
                    rel_type = "shared_topics"
                elif shared_entity_count >= 1:
                    rel_type = "shared_entities"
                else:
                    rel_type = "semantic_similarity"

                confidence = round(
                    min(
                        0.99,
                        max(sim, 0.55 + 0.12 * len(shared_topics) + 0.10 * shared_entity_count),
                    ),
                    3,
                )
                if shared_topics:
                    explanation = f"Shares topics ({', '.join(shared_topics[:4])})"
                elif shared_entity_count >= 1:
                    explanation = f"Shares {shared_entity_count} canonical named entities"
                else:
                    explanation = f"High semantic content similarity ({int(sim * 100)}%)"

                pair_lo, pair_hi = min(doc_id, other_id), max(doc_id, other_id)
                edge = {
                    "id": str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"rel:{workspace_id}:{pair_lo}:{pair_hi}:{rel_type}",
                        )
                    ),
                    "workspace_id": workspace_id,
                    "source_doc_id": pair_lo,
                    "target_doc_id": pair_hi,
                    "relationship_type": rel_type,
                    "composite_confidence": confidence,
                    "explanation": explanation,
                    "evidence_summary": {
                        "shared_topics": shared_topics,
                        "shared_entity_count": shared_entity_count,
                        "semantic_similarity": round(sim, 4),
                    },
                    "is_dismissed": False,
                    "created_at": now,
                }
                created_edges.append(edge)

        if self.is_connected and created_edges:
            try:
                self.client.table("document_relationships").upsert(
                    created_edges,
                    on_conflict="id",
                ).execute()
            except Exception:
                pass

        for edge in created_edges:
            self._local_tables["document_relationships"] = [
                r
                for r in self._local_tables["document_relationships"]
                if r.get("id") != edge["id"]
            ] + [edge]

        return created_edges

    def list_workspace_relationships(self, workspace_id: str) -> List[Dict[str, Any]]:
        remote: List[Dict[str, Any]] = []
        if self.is_connected:
            try:
                res = (
                    self.client.table("document_relationships")
                    .select("*")
                    .eq("workspace_id", workspace_id)
                    .eq("is_dismissed", False)
                    .order("composite_confidence", desc=True)
                    .execute()
                )
                remote = res.data or []
            except Exception:
                pass
        local = [
            r
            for r in self._local_tables["document_relationships"]
            if r.get("workspace_id") == workspace_id and not r.get("is_dismissed", False)
        ]
        seen = {r.get("id") for r in remote if r.get("id")}
        return remote + [r for r in local if r.get("id") not in seen]

    def dismiss_relationship(self, workspace_id: str, relationship_id: str) -> Optional[Dict[str, Any]]:
        updated: Optional[Dict[str, Any]] = None
        if self.is_connected:
            try:
                res = (
                    self.client.table("document_relationships")
                    .update({"is_dismissed": True})
                    .eq("workspace_id", workspace_id)
                    .eq("id", relationship_id)
                    .execute()
                )
                if res.data:
                    updated = res.data[0]
            except Exception:
                pass
        for r in self._local_tables["document_relationships"]:
            if r.get("workspace_id") == workspace_id and r.get("id") == relationship_id:
                r["is_dismissed"] = True
                if not updated:
                    updated = dict(r)
        return updated

    VALID_RELATIONSHIP_TYPES = {
        "shared_topics",
        "shared_entities",
        "semantic_similarity",
        "explicit_reference",
        "shared_metric_lineage",
        "contradiction_detected",
        "composite_related",
    }

    def upsert_relationship(
        self,
        workspace_id: str,
        source_doc_id: str,
        target_doc_id: str,
        relationship_type: str,
        explanation: str,
        evidence_summary: Optional[Dict[str, Any]] = None,
        confidence: float = 0.78,
    ) -> Dict[str, Any]:
        rel_type = (
            relationship_type
            if relationship_type in self.VALID_RELATIONSHIP_TYPES
            else "composite_related"
        )
        pair_lo, pair_hi = min(source_doc_id, target_doc_id), max(source_doc_id, target_doc_id)
        rel_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"rel:{workspace_id}:{pair_lo}:{pair_hi}:{rel_type}",
            )
        )
        edge = {
            "id": rel_id,
            "workspace_id": workspace_id,
            "source_doc_id": pair_lo,
            "target_doc_id": pair_hi,
            "relationship_type": rel_type,
            "composite_confidence": round(max(0.1, min(0.99, float(confidence))), 3),
            "explanation": explanation,
            "evidence_summary": evidence_summary or {},
            "is_dismissed": False,
            "created_at": _utc_now_iso(),
        }
        if self.is_connected:
            try:
                self.client.table("document_relationships").upsert(
                    edge,
                    on_conflict="workspace_id,source_doc_id,target_doc_id,relationship_type",
                ).execute()
            except Exception:
                pass
        self._local_tables["document_relationships"] = [
            r for r in self._local_tables["document_relationships"] if r.get("id") != rel_id
        ] + [edge]
        return edge

    def save_contradiction_relationship(
        self,
        workspace_id: str,
        source_doc_id: str,
        target_doc_id: str,
        explanation: str,
        evidence_summary: Dict[str, Any],
        confidence: float = 0.88,
    ) -> Dict[str, Any]:
        return self.upsert_relationship(
            workspace_id=workspace_id,
            source_doc_id=source_doc_id,
            target_doc_id=target_doc_id,
            relationship_type="contradiction_detected",
            explanation=explanation,
            evidence_summary=evidence_summary,
            confidence=confidence,
        )

    # -------------------------------------------------------------------------
    # 4. Repository Querying, Filtering & Cascade Deletion (DOC-09, PR-014..PR-018)
    # -------------------------------------------------------------------------
    def list_workspace_documents(
        self,
        workspace_id: str,
        *,
        search: Optional[str] = None,
        status: Optional[str] = None,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        has_tables: Optional[bool] = None,
        sort_by: str = "created_at",
        sort_dir: str = "desc",
    ) -> List[Dict[str, Any]]:
        if not workspace_id:
            raise ValueError("workspace_id is mandatory.")

        remote_docs: List[Dict[str, Any]] = []
        if self.is_connected:
            try:
                res = (
                    self.client.table("documents")
                    .select("*")
                    .eq("workspace_id", workspace_id)
                    .order("created_at", desc=True)
                    .execute()
                )
                remote_docs = res.data or []
            except Exception:
                pass

        local_docs = [d for d in self._local_tables["documents"] if d.get("workspace_id") == workspace_id]
        seen_ids = {d.get("id") for d in remote_docs if d.get("id")}
        docs = remote_docs + [dict(d) for d in local_docs if d.get("id") not in seen_ids]

        if search:
            q = search.strip().lower()
            docs = [
                d
                for d in docs
                if q in (d.get("title") or "").lower()
                or q in (d.get("summary") or "").lower()
                or q in (d.get("primary_domain") or "").lower()
                or any(q in t.lower() for t in (d.get("main_topics") or []))
            ]
        if status:
            docs = [
                d
                for d in docs
                if d.get("status") == status or d.get("processing_status") == status
            ]
        if category:
            docs = [d for d in docs if d.get("category") == category]
        if tag:
            t_low = tag.strip().lower()
            docs = [
                d
                for d in docs
                if any(t_low == ut.lower() for ut in (d.get("user_tags") or []))
                or any(t_low == mt.lower() for mt in (d.get("main_topics") or []))
            ]
        if has_tables is not None:
            docs = [d for d in docs if bool(d.get("has_tables", False)) == has_tables]

        reverse = sort_dir.lower() != "asc"
        docs.sort(key=lambda d: str(d.get(sort_by) or ""), reverse=reverse)
        return docs

    def get_workspace_document(self, workspace_id: str, doc_id: str) -> Optional[Dict[str, Any]]:
        if not workspace_id or not doc_id:
            return None
        if self.is_connected:
            try:
                res = (
                    self.client.table("documents")
                    .select("*")
                    .eq("workspace_id", workspace_id)
                    .eq("id", doc_id)
                    .limit(1)
                    .execute()
                )
                if res.data:
                    return res.data[0]
            except Exception:
                pass
        for d in self._local_tables["documents"]:
            if d.get("workspace_id") == workspace_id and d.get("id") == doc_id:
                return dict(d)
        return None

    def get_document_chunks(self, workspace_id: str, doc_id: str) -> List[Dict[str, Any]]:
        if not workspace_id or not doc_id:
            return []
        if self.is_connected:
            try:
                res = (
                    self.client.table("chunks")
                    .select("*")
                    .eq("workspace_id", workspace_id)
                    .eq("doc_id", doc_id)
                    .order("chunk_index", desc=False)
                    .execute()
                )
                if res.data:
                    return res.data
            except Exception:
                pass
        chunks = [
            dict(c)
            for c in self._local_tables["chunks"]
            if c.get("workspace_id") == workspace_id
            and (c.get("doc_id") == doc_id or c.get("document_id") == doc_id)
        ]
        chunks.sort(key=lambda c: int(c.get("chunk_index", 0)))
        return chunks

    def get_workspace_tables(
        self,
        workspace_id: str,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        remote: List[Dict[str, Any]] = []
        if self.is_connected:
            try:
                q = self.client.table("document_tables").select("*").eq("workspace_id", workspace_id)
                if doc_ids:
                    q = q.in_("doc_id", doc_ids)
                res = q.execute()
                remote = res.data or []
            except Exception:
                pass
        local = [
            dict(t)
            for t in self._local_tables["document_tables"]
            if t.get("workspace_id") == workspace_id and (not doc_ids or t.get("doc_id") in doc_ids)
        ]
        seen = {t.get("id") for t in remote if t.get("id")}
        return remote + [t for t in local if t.get("id") not in seen]

    def get_workspace_metrics(
        self,
        workspace_id: str,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        remote: List[Dict[str, Any]] = []
        if self.is_connected:
            try:
                q = self.client.table("extracted_metrics").select("*").eq("workspace_id", workspace_id)
                if doc_ids:
                    q = q.in_("doc_id", doc_ids)
                res = q.execute()
                remote = res.data or []
            except Exception:
                pass
        local = [
            dict(m)
            for m in self._local_tables["extracted_metrics"]
            if m.get("workspace_id") == workspace_id and (not doc_ids or m.get("doc_id") in doc_ids)
        ]
        seen = {m.get("id") for m in remote if m.get("id")}
        return remote + [m for m in local if m.get("id") not in seen]

    def delete_workspace_document(self, workspace_id: str, doc_id: str) -> Optional[Dict[str, Any]]:
        doc = self.get_workspace_document(workspace_id, doc_id)
        if not doc:
            return None

        if self.is_connected:
            try:
                self.client.table("documents").delete().eq("workspace_id", workspace_id).eq("id", doc_id).execute()
            except Exception:
                pass

        self._local_tables["documents"] = [
            d for d in self._local_tables["documents"] if not (d.get("workspace_id") == workspace_id and d.get("id") == doc_id)
        ]
        for child_table in (
            "document_pages",
            "chunks",
            "document_tables",
            "extracted_metrics",
            "chunk_topics",
            "document_entities",
            "processing_jobs",
            "message_citations",
        ):
            self._local_tables[child_table] = [
                r
                for r in self._local_tables[child_table]
                if not (
                    r.get("workspace_id") == workspace_id
                    and (r.get("doc_id") == doc_id or r.get("document_id") == doc_id)
                )
            ]
        self._local_tables["document_relationships"] = [
            r
            for r in self._local_tables["document_relationships"]
            if not (
                r.get("workspace_id") == workspace_id
                and (r.get("source_doc_id") == doc_id or r.get("target_doc_id") == doc_id)
            )
        ]
        # Flag any dashboard widget whose sole source was this deleted document (DOC-14, OQ-005)
        for w in self._local_tables["dashboard_widgets"]:
            if w.get("workspace_id") == workspace_id and doc_id in (w.get("source_doc_ids") or []):
                remaining = [sid for sid in (w.get("source_doc_ids") or []) if sid != doc_id]
                w["source_doc_ids"] = remaining
                if not remaining:
                    w["status"] = "stale_source_deleted"

        return doc

    # -------------------------------------------------------------------------
    # 5. Hybrid RRF Search (match_document_chunks_hybrid + Local RRF Engine - DOC-12)
    # -------------------------------------------------------------------------
    def search_chunks_hybrid(
        self,
        *,
        workspace_id: str,
        query_embedding: List[float],
        query_text: str,
        match_count: int = 10,
        doc_ids: Optional[List[str]] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_vector_similarity: float = 0.28,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        if not workspace_id:
            raise ValueError("workspace_id is mandatory for tenant-isolated search.")

        if self.is_connected:
            try:
                rpc_params: Dict[str, Any] = {
                    "filter_workspace_id": workspace_id,
                    "query_embedding": query_embedding,
                    "query_text": query_text or "",
                    "match_count": match_count,
                    "filter_doc_ids": doc_ids if doc_ids else None,
                    "filter_category": category,
                    "filter_tags": tags if tags else None,
                    "min_vector_similarity": min_vector_similarity,
                    "rrf_k": rrf_k,
                }
                res = self.client.rpc("match_document_chunks_hybrid", rpc_params).execute()
                if res.data:
                    return res.data
            except Exception as e:
                print(f"[Supabase] Hybrid RPC fallback to in-process RRF ({e}).")

        # Schema-faithful in-process Hybrid RRF engine over workspace chunks
        eligible_docs = {
            d["id"]: d
            for d in self.list_workspace_documents(
                workspace_id,
                category=category,
            )
            if (not doc_ids or d["id"] in doc_ids)
            and (not tags or bool(set(d.get("user_tags") or []).intersection(set(tags))))
        }
        if not eligible_docs:
            return []

        all_chunks: List[Dict[str, Any]] = []
        for did in eligible_docs:
            all_chunks.extend(self.get_document_chunks(workspace_id, did))

        if not all_chunks:
            return []

        q_terms = [
            t.lower()
            for t in re.split(r"[^a-zA-Z0-9_]+", (query_text or "").strip())
            if len(t) >= 2
        ]

        vec_scored: List[tuple[str, float]] = []
        fts_scored: List[tuple[str, float]] = []
        chunk_map: Dict[str, Dict[str, Any]] = {}

        for c in all_chunks:
            cid = str(c["id"])
            chunk_map[cid] = c
            txt = (c.get("text_content") or c.get("content") or "").lower()

            v_sim = _cosine_similarity(query_embedding, c.get("embedding"))
            if v_sim >= min_vector_similarity:
                vec_scored.append((cid, v_sim))

            if q_terms:
                term_hits = sum(1 for term in q_terms if term in txt)
                if term_hits > 0:
                    f_score = round(term_hits / len(q_terms), 4)
                    fts_scored.append((cid, f_score))

        vec_scored.sort(key=lambda x: x[1], reverse=True)
        fts_scored.sort(key=lambda x: x[1], reverse=True)

        v_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(vec_scored[: match_count * 4])}
        v_sims = {cid: score for cid, score in vec_scored[: match_count * 4]}
        f_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(fts_scored[: match_count * 4])}
        f_scores = {cid: score for cid, score in fts_scored[: match_count * 4]}

        candidate_ids = set(v_ranks.keys()).union(set(f_ranks.keys()))
        results: List[Dict[str, Any]] = []
        for cid in candidate_ids:
            c = chunk_map[cid]
            did = str(c.get("doc_id") or c.get("document_id"))
            doc_meta = eligible_docs.get(did, {})
            vr = v_ranks.get(cid)
            fr = f_ranks.get(cid)
            rrf = (1.0 / (rrf_k + vr) if vr else 0.0) + (1.0 / (rrf_k + fr) if fr else 0.0)
            if vr and fr:
                explanation = "hybrid_semantic_and_keyword"
            elif fr:
                explanation = "exact_keyword_match"
            else:
                explanation = "semantic_conceptual_match"

            results.append(
                {
                    "chunk_id": cid,
                    "doc_id": did,
                    "doc_title": doc_meta.get("title", "Document"),
                    "doc_category": doc_meta.get("category", "general_document"),
                    "page_number": int(c.get("page_number", 1)),
                    "chunk_index": int(c.get("chunk_index", 0)),
                    "text_content": c.get("text_content") or c.get("content") or "",
                    "bounding_boxes": c.get("bounding_boxes", []),
                    "vector_similarity": round(v_sims.get(cid, 0.0), 4),
                    "fts_rank": round(f_scores.get(cid, 0.0), 4),
                    "rrf_score": round(rrf, 6),
                    "match_explanation": explanation,
                }
            )

        results.sort(key=lambda r: (r["rrf_score"], r["vector_similarity"]), reverse=True)
        return results[:match_count]

    # -------------------------------------------------------------------------
    # 6. Conversations, Messages & Citations Persistence (DOC-13, PR-030)
    # -------------------------------------------------------------------------
    def save_conversation_turn(
        self,
        *,
        workspace_id: str,
        user_id: str,
        conversation_id: Optional[str],
        scope_mode: str,
        scoped_doc_ids: List[str],
        user_message: str,
        assistant_answer: str,
        model_used: str,
        citations: List[Dict[str, Any]],
        suggested_followups: List[str],
    ) -> Dict[str, Any]:
        now = _utc_now_iso()
        conv_id = conversation_id or str(uuid.uuid4())
        title = (user_message or "Document Conversation").strip()[:80]

        conv_row = {
            "id": conv_id,
            "workspace_id": workspace_id,
            "user_id": user_id,
            "title": title,
            "scope_mode": scope_mode,
            "scoped_doc_ids": scoped_doc_ids,
            "created_at": now,
            "updated_at": now,
        }
        user_msg_row = {
            "id": str(uuid.uuid4()),
            "workspace_id": workspace_id,
            "conversation_id": conv_id,
            "role": "user",
            "content": user_message,
            "model_used": None,
            "suggested_followups": [],
            "created_at": now,
        }
        asst_msg_id = str(uuid.uuid4())
        asst_msg_row = {
            "id": asst_msg_id,
            "workspace_id": workspace_id,
            "conversation_id": conv_id,
            "role": "assistant",
            "content": assistant_answer,
            "model_used": model_used,
            "suggested_followups": suggested_followups,
            "created_at": now,
        }
        citation_rows = [
            {
                "id": str(uuid.uuid4()),
                "workspace_id": workspace_id,
                "message_id": asst_msg_id,
                "citation_index": idx + 1,
                "doc_id": cit["document_id"],
                "chunk_id": cit["chunk_id"],
                "page_number": int(cit["page_number"]),
                "bounding_boxes": cit.get("bounding_boxes", []),
                "verbatim_snippet": cit.get("quote", ""),
                "created_at": now,
            }
            for idx, cit in enumerate(citations)
        ]

        if self.is_connected:
            try:
                self.client.table("conversations").upsert(conv_row).execute()
                self.client.table("messages").insert([user_msg_row, asst_msg_row]).execute()
                if citation_rows:
                    self.client.table("message_citations").insert(citation_rows).execute()
            except Exception:
                pass

        self._local_tables["conversations"] = [
            c for c in self._local_tables["conversations"] if c.get("id") != conv_id
        ] + [conv_row]
        self._local_tables["messages"].extend([user_msg_row, asst_msg_row])
        self._local_tables["message_citations"].extend(citation_rows)

        return {
            "conversation_id": conv_id,
            "message_id": asst_msg_id,
        }

    def list_conversations(self, workspace_id: str) -> List[Dict[str, Any]]:
        remote: List[Dict[str, Any]] = []
        if self.is_connected:
            try:
                res = (
                    self.client.table("conversations")
                    .select("*")
                    .eq("workspace_id", workspace_id)
                    .order("updated_at", desc=True)
                    .limit(50)
                    .execute()
                )
                remote = res.data or []
            except Exception:
                pass
        local = [c for c in reversed(self._local_tables["conversations"]) if c.get("workspace_id") == workspace_id]
        seen = {c.get("id") for c in remote if c.get("id")}
        return remote + [c for c in local if c.get("id") not in seen]

    def get_conversation_messages(self, workspace_id: str, conversation_id: str) -> List[Dict[str, Any]]:
        if self.is_connected:
            try:
                res = (
                    self.client.table("messages")
                    .select("*")
                    .eq("workspace_id", workspace_id)
                    .eq("conversation_id", conversation_id)
                    .order("created_at", desc=False)
                    .execute()
                )
                if res.data:
                    return res.data
            except Exception:
                pass
        return [
            m
            for m in self._local_tables["messages"]
            if m.get("workspace_id") == workspace_id and m.get("conversation_id") == conversation_id
        ]

    # -------------------------------------------------------------------------
    # 7. Dashboard Widgets Persistence (Pillar B - DOC-14, DOC-15)
    # -------------------------------------------------------------------------
    def save_dashboard_widget(self, workspace_id: str, widget_data: Dict[str, Any]) -> Dict[str, Any]:
        now = _utc_now_iso()
        wid = widget_data.get("id") or str(uuid.uuid4())
        row = {
            "id": wid,
            "workspace_id": workspace_id,
            "title": widget_data["title"],
            "subtitle": widget_data.get("subtitle"),
            "description": widget_data.get("description") or widget_data.get("insight_summary") or "",
            "widget_type": widget_data.get("widget_type", "echarts_chart"),
            "chart_type": widget_data.get("chart_type", "bar"),
            "is_auto_generated": bool(widget_data.get("is_auto_generated", True)),
            "is_pinned": bool(widget_data.get("is_pinned", False)),
            "status": widget_data.get("status", "active"),
            "display_order": int(widget_data.get("display_order", 0)),
            "grid_span": int(widget_data.get("grid_span", 6)),
            "sql_query": widget_data.get("sql_query"),
            "unit_or_currency": widget_data.get("unit_or_currency"),
            "echarts_spec": widget_data.get("echarts_spec", {}),
            "echarts_config": widget_data.get("echarts_config") or widget_data.get("echarts_spec", {}),
            "kpi_payload": widget_data.get("kpi_payload", {}),
            "narrative_insight": widget_data.get("insight_summary") or widget_data.get("description") or "",
            "source_doc_ids": widget_data.get("source_doc_ids", []),
            "source_table_ids": widget_data.get("source_table_ids", []),
            "source_chunk_ids": widget_data.get("source_chunk_ids", []),
            "created_at": now,
            "updated_at": now,
        }
        if self.is_connected:
            try:
                res = self.client.table("dashboard_widgets").upsert(row).execute()
                if res.data:
                    row = res.data[0]
            except Exception:
                pass
        self._local_tables["dashboard_widgets"] = [
            w for w in self._local_tables["dashboard_widgets"] if w.get("id") != wid
        ] + [row]
        return row

    def list_dashboard_widgets(self, workspace_id: str) -> List[Dict[str, Any]]:
        remote: List[Dict[str, Any]] = []
        if self.is_connected:
            try:
                res = (
                    self.client.table("dashboard_widgets")
                    .select("*")
                    .eq("workspace_id", workspace_id)
                    .order("display_order", desc=False)
                    .execute()
                )
                remote = res.data or []
            except Exception:
                pass
        local = [w for w in self._local_tables["dashboard_widgets"] if w.get("workspace_id") == workspace_id]
        seen = {w.get("id") for w in remote if w.get("id")}
        return remote + [w for w in local if w.get("id") not in seen]

    def delete_dashboard_widget(self, workspace_id: str, widget_id: str) -> bool:
        if self.is_connected:
            try:
                self.client.table("dashboard_widgets").delete().eq("workspace_id", workspace_id).eq("id", widget_id).execute()
            except Exception:
                pass
        before = len(self._local_tables["dashboard_widgets"])
        self._local_tables["dashboard_widgets"] = [
            w
            for w in self._local_tables["dashboard_widgets"]
            if not (w.get("workspace_id") == workspace_id and w.get("id") == widget_id)
        ]
        return len(self._local_tables["dashboard_widgets"]) < before or self.is_connected

    # -------------------------------------------------------------------------
    # 8. AI Usage Telemetry Logging & Account Deletion (DOC-19, PR-004, PR-043)
    # -------------------------------------------------------------------------
    def log_ai_usage(self, usage_row: Dict[str, Any]) -> Dict[str, Any]:
        row = {
            "id": usage_row.get("id") or str(uuid.uuid4()),
            "workspace_id": usage_row["workspace_id"],
            "user_id": usage_row.get("user_id"),
            "task_type": usage_row["task_type"],
            "provider": usage_row.get("provider", "openai"),
            "model": usage_row["model"],
            "service_tier": usage_row.get("service_tier", "auto"),
            "input_tokens": int(usage_row.get("input_tokens", 0)),
            "cached_tokens": int(usage_row.get("cached_tokens", 0)),
            "output_tokens": int(usage_row.get("output_tokens", 0)),
            "reasoning_tokens": int(usage_row.get("reasoning_tokens", 0)),
            "estimated_cost_usd": round(float(usage_row.get("estimated_cost_usd", 0.0)), 6),
            "latency_ms": int(usage_row.get("latency_ms", 0)),
            "created_at": _utc_now_iso(),
        }
        if self.is_connected:
            try:
                res = self.client.table("ai_usage_logs").insert(row).execute()
                if res.data:
                    return res.data[0]
            except Exception as e:
                print(f"[Supabase] Warning: Failed to write ai_usage_logs ({e}). Stored locally.")
        self._local_tables["ai_usage_logs"].append(row)
        return row

    def list_ai_usage(self, workspace_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        if not workspace_id:
            raise ValueError("workspace_id is mandatory.")
        remote_rows: List[Dict[str, Any]] = []
        if self.is_connected:
            try:
                res = (
                    self.client.table("ai_usage_logs")
                    .select("*")
                    .eq("workspace_id", workspace_id)
                    .order("created_at", desc=True)
                    .limit(limit)
                    .execute()
                )
                remote_rows = res.data or []
            except Exception:
                pass
        local_rows = [
            r for r in reversed(self._local_tables["ai_usage_logs"]) if r.get("workspace_id") == workspace_id
        ]
        seen_ids = {r.get("id") for r in remote_rows if r.get("id")}
        merged = remote_rows + [r for r in local_rows if r.get("id") not in seen_ids]
        return merged[:limit]

    def delete_user_account_and_workspaces(self, user_id: str, workspace_id: str) -> Dict[str, Any]:
        if not user_id or not workspace_id:
            raise ValueError("Both user_id and workspace_id are required for account deletion.")

        deleted_storage_files = 0
        if self.is_connected:
            try:
                docs_res = (
                    self.client.table("documents")
                    .select("id, storage_path")
                    .eq("workspace_id", workspace_id)
                    .execute()
                )
                paths = [
                    d["storage_path"]
                    for d in (docs_res.data or [])
                    if d.get("storage_path") and not d["storage_path"].startswith("./")
                ]
                if paths:
                    try:
                        self.client.storage.from_(settings.supabase_storage_bucket).remove(paths)
                        deleted_storage_files = len(paths)
                    except Exception as storage_err:
                        print(f"[Supabase] Storage cleanup warning: {storage_err}")

                self.client.table("workspaces").delete().eq("id", workspace_id).eq("owner_id", user_id).execute()

                try:
                    self.client.auth.admin.delete_user(user_id)
                except Exception as auth_err:
                    print(f"[Supabase] Admin user delete warning: {auth_err}")
            except Exception as e:
                print(f"[Supabase] Account deletion error: {e}")

        for table_name, rows in self._local_tables.items():
            if table_name == "workspaces":
                self._local_tables[table_name] = [
                    r for r in rows if not (r.get("id") == workspace_id and r.get("owner_id") == user_id)
                ]
            else:
                self._local_tables[table_name] = [
                    r for r in rows if r.get("workspace_id") != workspace_id
                ]

        return {
            "deleted": True,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "deleted_storage_files": deleted_storage_files,
        }


supabase_db = SupabaseManager()
