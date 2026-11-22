import math
import os
import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from app.ai.contracts import (
    AITaskType,
    OCRPageExtraction,
    DocumentUnderstandingPayload,
    ExtractedEntityItem,
    ExtractedKPIItem,
    KeyFindingItem,
)
from app.ai.gateway import ai_gateway, format_untrusted_chunk
from app.core.config import settings
from app.db.supabase_client import supabase_db
from app.models.schemas import BoundingBox
from app.services.pdf_service import pdf_processor


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _verify_number_in_text(numeric_value: float, text: str) -> bool:
    """
    Stage 12 Deterministic Number Verification (PR-035, PR-036):
    Verifies that an extracted KPI number actually appears in its cited source chunk text.
    """
    if not text:
        return False
    normalized_text = text.replace(",", "")
    if float(numeric_value).is_integer():
        int_str = str(int(numeric_value))
        if re.search(rf"\b{re.escape(int_str)}\b", normalized_text):
            return True
    float_str = f"{numeric_value:.2f}".rstrip("0").rstrip(".")
    if float_str in normalized_text:
        return True
    raw_str = str(numeric_value)
    return raw_str in normalized_text


def _deterministic_document_understanding_fallback(
    title: str,
    chunks: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
) -> DocumentUnderstandingPayload:
    """
    Deterministic Document Understanding analyzer used when OPENAI_API_KEY is not configured
    or during transient API outages (EC-013). Extracts real sentences, topics, entities,
    and verifiable numbers directly from the uploaded document's chunks and tables.
    Never invents external or unrelated text.
    """
    combined_text = " ".join((c.get("text_content") or c.get("content") or "") for c in chunks[:8])
    lower_text = combined_text.lower()

    if tables and any(w in lower_text for w in ("revenue", "pendapatan", "profit", "laba", "quarter", "kuartal", "usd", "idr")):
        doc_type = "financial_report"
        domain = "Corporate Finance & Operations"
    elif any(w in lower_text for w in ("abstract", "methodology", "experiment", "efficiency", "hipotesis", "penelitian")):
        doc_type = "academic_paper"
        domain = "Academic & Scientific Research"
    elif any(w in lower_text for w in ("pasal", "clause", "agreement", "perjanjian", "contract")):
        doc_type = "contract_legal"
        domain = "Legal & Compliance"
    elif tables:
        doc_type = "spreadsheet_dataset"
        domain = "Quantitative Data Analysis"
    else:
        doc_type = "general_document"
        domain = "General Knowledge"

    id_markers = sum(1 for w in ("dan", "yang", "untuk", "pada", "dengan", "adalah", "dokumen") if f" {w} " in f" {lower_text} ")
    en_markers = sum(1 for w in ("the", "and", "for", "with", "that", "this", "from") if f" {w} " in f" {lower_text} ")
    lang_code = "id" if id_markers >= en_markers else "en"

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", combined_text)
        if len(s.strip()) > 25
    ]
    exec_summary = (
        " ".join(sentences[:3])[:750]
        if sentences
        else f"Document '{title}' processed with {len(chunks)} passages and {len(tables)} structured tables."
    )

    # Extract frequent meaningful capitalized/domain terms as topics
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{3,24}\b", combined_text)
    stop_words = {
        "yang", "untuk", "pada", "dengan", "dalam", "dari", "adalah", "akan", "atau", "juga",
        "this", "that", "with", "from", "have", "were", "their", "which", "about", "table", "page",
    }
    freq: Dict[str, int] = {}
    for w in words:
        wl = w.lower()
        if wl not in stop_words and not wl.isdigit():
            freq[wl] = freq.get(wl, 0) + 1
    top_words = [w.title() for w, _ in sorted(freq.items(), key=lambda item: item[1], reverse=True)[:7]]
    if not top_words:
        top_words = ["Document Analysis"]

    key_findings: List[KeyFindingItem] = []
    for c in chunks[:4]:
        cid = str(c["id"])
        txt = (c.get("text_content") or c.get("content") or "").strip()
        if txt:
            first_sent = re.split(r"(?<=[.!?])\s+", txt)[0][:220]
            key_findings.append(
                KeyFindingItem(
                    headline=f"Page {c.get('page_number', 1)} Key Evidence",
                    explanation=first_sent,
                    source_chunk_id=cid,
                )
            )

    # Extract verifiable standalone numbers/currencies/percentages from chunks
    extracted_kpis: List[ExtractedKPIItem] = []
    kpi_pattern = re.compile(
        r"(?:(?P<prefix>USD|IDR|Rp)\s*)?(?P<num>\d+(?:\.\d+)?)\s*(?P<suffix>%|USD|IDR|mAh/g|mA/cm2|million|juta|miliar)?",
        re.IGNORECASE,
    )
    for c in chunks[:10]:
        cid = str(c["id"])
        txt = (c.get("text_content") or c.get("content") or "").strip()
        for m in kpi_pattern.finditer(txt):
            prefix = (m.group("prefix") or "").upper()
            suffix = (m.group("suffix") or "").strip()
            if not prefix and not suffix:
                continue
            try:
                val = float(m.group("num"))
            except ValueError:
                continue
            unit = prefix or (suffix.upper() if suffix.upper() in ("USD", "IDR", "%") else suffix) or "count"
            if unit == "RP":
                unit = "IDR"
            start = max(0, m.start() - 45)
            end = min(len(txt), m.end() + 45)
            quote = txt[start:end].strip()
            label_words = [w for w in re.findall(r"[A-Za-z]{3,}", txt[max(0, m.start() - 35) : m.start()])]
            metric_label = " ".join(label_words[-3:]).title() if label_words else f"Page {c.get('page_number', 1)} Metric"
            extracted_kpis.append(
                ExtractedKPIItem(
                    metric_name=metric_label,
                    numeric_value=val,
                    unit_or_currency=unit,
                    period_or_context=f"Page {c.get('page_number', 1)}",
                    source_chunk_id=cid,
                    verbatim_quote=quote,
                )
            )
            if len(extracted_kpis) >= 10:
                break
        if len(extracted_kpis) >= 10:
            break

    entities = [
        ExtractedEntityItem(
            name=topic,
            entity_type="other",
            canonical_name=topic.lower(),
        )
        for topic in top_words[:5]
    ]

    return DocumentUnderstandingPayload(
        inferred_title=title,
        document_type=doc_type,
        primary_domain=domain,
        language_code=lang_code,
        executive_summary=exec_summary,
        main_topics=top_words,
        named_entities=entities,
        key_dates=[],
        key_findings=key_findings,
        extracted_kpis=extracted_kpis,
    )


def _compute_mean_centroid(vectors: List[List[float]]) -> Optional[List[float]]:
    valid = [v for v in vectors if v and len(v) == 1536]
    if not valid:
        return None
    n = len(valid)
    mean_vec = [sum(v[i] for v in valid) / n for i in range(1536)]
    norm = math.sqrt(sum(x * x for x in mean_vec)) or 1.0
    return [round(x / norm, 6) for x in mean_vec]


class DocumentIngestionService:
    """
    15-Stage Checkpointed Document Ingestion Pipeline (DOC-10, DOC-24, ADR-005, ADR-006, ADR-008, ADR-010, ADR-014).
    """

    async def execute_pipeline(
        self,
        *,
        workspace_id: str,
        doc_id: str,
        job_id: str,
        local_file_path: str,
        user_id: Optional[str] = None,
        expected_version: int = 1,
    ) -> Dict[str, Any]:
        try:
            doc_record = supabase_db.get_workspace_document(workspace_id, doc_id)
            if not doc_record:
                raise RuntimeError(f"Document {doc_id} no longer exists in workspace {workspace_id}.")

            title = doc_record.get("title") or os.path.basename(local_file_path)
            ext = os.path.splitext(local_file_path.lower())[1]

            # Update status to processing (Stage 3)
            if not supabase_db.update_document(
                workspace_id,
                doc_id,
                {"status": "processing", "processing_status": "processing"},
                expected_version=expected_version,
            ):
                raise RuntimeError("Stale ingestion job aborted: document version changed or was deleted.")

            supabase_db.update_processing_job(
                workspace_id,
                job_id,
                {"status": "running", "current_stage": "stage_3_5_layout_and_pages", "progress_pct": 20},
            )

            # -----------------------------------------------------------------
            # Stages 3, 4, 5 & 7 (Deterministic Layout, BBox & Table Extraction)
            # -----------------------------------------------------------------
            if ext in (".csv", ".xlsx"):
                page_profiles, page_blocks, extracted_tables = pdf_processor.extract_spreadsheet_tables_and_blocks(
                    local_file_path
                )
            else:
                page_profiles, page_blocks, extracted_tables = pdf_processor.analyze_pdf_pages_and_blocks(
                    local_file_path
                )

            supabase_db.save_document_pages(workspace_id, doc_id, page_profiles)

            # -----------------------------------------------------------------
            # Stage 6: Selective Vision OCR via gpt-6.1-terra (ONLY if needed)
            # -----------------------------------------------------------------
            ocr_pages = [
                p["page_number"]
                for p in page_profiles
                if p.get("needs_ocr")
            ][: settings.max_vision_ocr_pages]

            partial_ocr_failures = 0
            if ocr_pages and ext == ".pdf" and ai_gateway.is_openai_configured:
                supabase_db.update_processing_job(
                    workspace_id,
                    job_id,
                    {"current_stage": "stage_6_selective_vision_ocr", "progress_pct": 35},
                )
                total_ocr_pages = len(ocr_pages)
                for ocr_idx, p_num in enumerate(ocr_pages, start=1):
                    pct_tick = 35 + int((ocr_idx - 1) / max(1, total_ocr_pages) * 13)
                    supabase_db.update_processing_job(
                        workspace_id,
                        job_id,
                        {
                            "current_stage": f"stage_6_vision_ocr_page_{ocr_idx}_of_{total_ocr_pages}",
                            "progress_pct": pct_tick,
                        },
                    )
                    try:
                        b64_png = pdf_processor.render_page_png_base64(local_file_path, p_num, dpi=150)
                        if not b64_png:
                            continue
                        ocr_res = await ai_gateway.generate_structured(
                            task_type=AITaskType.VISION_OCR_AND_TABLES,
                            workspace_id=workspace_id,
                            user_id=user_id,
                            instructions=(
                                "Transcribe all readable text blocks, figure captions, and tables from this scanned "
                                "document page accurately. Preserve numbers and units verbatim."
                            ),
                            input_blocks=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "input_text", "text": f"Extract text and tables from page {p_num}."},
                                        {
                                            "type": "input_image",
                                            "image_url": f"data:image/png;base64,{b64_png}",
                                            "detail": "high",
                                        },
                                    ],
                                }
                            ],
                            response_model=OCRPageExtraction,
                            scope_key=f"ocr:{doc_id}:p{p_num}",
                        )
                        for blk in ocr_res.blocks:
                            if blk.text.strip():
                                y0 = max(0.0, min(0.95, float(blk.approx_y0)))
                                y1 = max(y0 + 0.04, min(1.0, float(blk.approx_y1)))
                                page_blocks.setdefault(p_num, []).append(
                                    {
                                        "text": blk.text.strip(),
                                        "word_count": len(blk.text.split()),
                                        "bbox": BoundingBox(
                                            left=0.06,
                                            top=round(y0, 4),
                                            width=0.88,
                                            height=round(y1 - y0, 4),
                                            page_number=p_num,
                                        ),
                                    }
                                )
                        for desc in ocr_res.visual_chart_descriptions:
                            if (desc or "").strip():
                                page_blocks.setdefault(p_num, []).append(
                                    {
                                        "text": f"[Visual / Diagram Description - Page {p_num}]: {desc.strip()}",
                                        "word_count": len(desc.split()),
                                        "bbox": BoundingBox(left=0.06, top=0.08, width=0.88, height=0.84, page_number=p_num),
                                    }
                                )
                        for tbl in ocr_res.extracted_tables:
                            col_names = [c.name for c in tbl.columns]
                            row_lines = [" | ".join(r) for r in tbl.rows if r]
                            tbl_text = f"Table: {tbl.table_title}\nColumns: {' | '.join(col_names)}\n" + "\n".join(row_lines)
                            if tbl_text.strip():
                                page_blocks.setdefault(p_num, []).append(
                                    {
                                        "text": tbl_text.strip(),
                                        "word_count": len(tbl_text.split()),
                                        "bbox": BoundingBox(left=0.06, top=0.1, width=0.88, height=0.8, page_number=p_num),
                                    }
                                )
                    except Exception as ocr_err:
                        print(f"[IngestionService] Vision OCR structured warning on page {p_num}: {ocr_err}")
                        # Fallback to plain-text Vision transcription if structured schema parse failed
                        try:
                            plain_ocr = await ai_gateway.generate_text(
                                task_type=AITaskType.VISION_OCR_AND_TABLES,
                                workspace_id=workspace_id,
                                user_id=user_id,
                                instructions=(
                                    "Transcribe all readable text, headings, labels, diagram annotations, and table data "
                                    "from this page image accurately in plain text."
                                ),
                                input_blocks=[
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "input_text", "text": f"Transcribe page {p_num} verbatim."},
                                            {
                                                "type": "input_image",
                                                "image_url": f"data:image/png;base64,{b64_png}",
                                                "detail": "high",
                                            },
                                        ],
                                    }
                                ],
                                scope_key=f"ocr_txt:{doc_id}:p{p_num}",
                            )
                            if (plain_ocr or "").strip():
                                page_blocks.setdefault(p_num, []).append(
                                    {
                                        "text": plain_ocr.strip(),
                                        "word_count": len(plain_ocr.split()),
                                        "bbox": BoundingBox(
                                            left=0.05,
                                            top=0.05,
                                            width=0.90,
                                            height=0.90,
                                            page_number=p_num,
                                        ),
                                    }
                                )
                            else:
                                partial_ocr_failures += 1
                        except Exception as txt_ocr_err:
                            print(f"[IngestionService] Vision OCR plain fallback warning on page {p_num}: {txt_ocr_err}")
                            partial_ocr_failures += 1

            # -----------------------------------------------------------------
            # Stage 7 & 8: Commit Tables & Build Layout-Aware Coordinate Chunks
            # -----------------------------------------------------------------
            supabase_db.update_processing_job(
                workspace_id,
                job_id,
                {"current_stage": "stage_7_8_tables_and_chunking", "progress_pct": 50},
            )
            saved_tables = supabase_db.replace_document_tables(workspace_id, doc_id, extracted_tables)
            table_pages = {int(t.get("page_number", 1)) for t in saved_tables}

            chunk_models = pdf_processor.build_layout_chunks(
                document_id=doc_id,
                workspace_id=workspace_id,
                page_blocks=page_blocks,
                pages_with_tables=table_pages,
            )

            # Fallback if a scanned PDF had 0 digital text and OPENAI_API_KEY was not set
            if not chunk_models:
                chunk_models = pdf_processor.build_layout_chunks(
                    document_id=doc_id,
                    workspace_id=workspace_id,
                    page_blocks={
                        1: [
                            {
                                "text": f"Scanned document '{title}' ({len(page_profiles)} pages). Configure OPENAI_API_KEY to enable gpt-6.1-terra Vision OCR.",
                                "word_count": 16,
                                "bbox": BoundingBox(left=0.05, top=0.05, width=0.9, height=0.2, page_number=1),
                            }
                        ]
                    },
                )

            # -----------------------------------------------------------------
            # Stage 9 & 10: Batch Vector Embeddings (text-embedding-3-large @ 1536d) & Indexing
            # -----------------------------------------------------------------
            supabase_db.update_processing_job(
                workspace_id,
                job_id,
                {"current_stage": "stage_9_10_embeddings_and_indexing", "progress_pct": 68},
            )
            chunk_texts = [c.content for c in chunk_models]
            vectors = await ai_gateway.embed_texts(
                texts=chunk_texts,
                workspace_id=workspace_id,
                user_id=user_id,
                is_query=False,
            )

            chunk_dicts: List[Dict[str, Any]] = []
            for idx, c_model in enumerate(chunk_models):
                vec = vectors[idx] if idx < len(vectors) else None
                chunk_dicts.append(
                    {
                        "id": c_model.id,
                        "workspace_id": workspace_id,
                        "document_id": doc_id,
                        "doc_id": doc_id,
                        "chunk_index": c_model.chunk_index,
                        "content": c_model.content,
                        "text_content": c_model.content,
                        "page_number": c_model.page_number,
                        "bounding_boxes": [b.model_dump() for b in c_model.bounding_boxes],
                        "embedding": vec,
                        "embedding_model": f"{settings.ai_default_provider}:{settings.openai_embedding_model}:{settings.openai_embedding_dimensions}",
                        "token_count": c_model.token_count,
                        "has_table": c_model.has_table,
                        "metadata": c_model.metadata,
                    }
                )

            saved_chunks = supabase_db.replace_document_chunks(workspace_id, doc_id, chunk_dicts)

            # -----------------------------------------------------------------
            # Stage 11: Single-Pass Document Understanding (gpt-6.1-terra)
            # -----------------------------------------------------------------
            supabase_db.update_processing_job(
                workspace_id,
                job_id,
                {"current_stage": "stage_11_document_understanding", "progress_pct": 82},
            )
            understanding: Optional[DocumentUnderstandingPayload] = None
            if ai_gateway.is_openai_configured and saved_chunks:
                head_chunks = saved_chunks[:4]
                table_or_mid_chunks = [
                    c for c in saved_chunks[4:-2] if c.get("has_table")
                ][:4]
                if not table_or_mid_chunks and len(saved_chunks) > 8:
                    mid = len(saved_chunks) // 3
                    table_or_mid_chunks = saved_chunks[mid : mid + 3]
                tail_chunks = saved_chunks[-2:] if len(saved_chunks) > 6 else []

                seen_sample_ids: set[str] = set()
                sampled: List[Dict[str, Any]] = []
                for cand in head_chunks + table_or_mid_chunks + tail_chunks:
                    cid_str = str(cand["id"])
                    if cid_str not in seen_sample_ids:
                        seen_sample_ids.add(cid_str)
                        sampled.append(cand)
                formatted_passages = "\n\n".join(
                    format_untrusted_chunk(
                        chunk_id=str(c["id"]),
                        text_content=c.get("text_content") or c.get("content") or "",
                        page_number=int(c.get("page_number", 1)),
                        doc_title=title,
                        doc_id=doc_id,
                    )
                    for c in sampled
                )
                try:
                    understanding = await ai_gateway.generate_structured(
                        task_type=AITaskType.DOCUMENT_UNDERSTANDING,
                        workspace_id=workspace_id,
                        user_id=user_id,
                        instructions=(
                            "You are the DocuMind Document Understanding Engine. Analyze the provided document "
                            "passages inside <untrusted_document_chunk> tags. Extract an accurate executive summary, "
                            "classification, main topics, named entities, key findings with exact source_chunk_id "
                            "references, and standalone quantitative KPIs whose numbers appear verbatim in the chunks."
                        ),
                        input_blocks=formatted_passages,
                        response_model=DocumentUnderstandingPayload,
                        scope_key=f"understanding:{doc_id}",
                    )
                except Exception as u_err:
                    print(f"[IngestionService] Stage 11 OpenAI fallback notice: {u_err}")

            if understanding is None:
                understanding = _deterministic_document_understanding_fallback(
                    title=title,
                    chunks=saved_chunks,
                    tables=saved_tables,
                )

            # -----------------------------------------------------------------
            # Stage 12: Deterministic KPI & Metric Verification (PR-035, PR-036)
            # -----------------------------------------------------------------
            chunk_by_id = {str(c["id"]): c for c in saved_chunks}
            verified_metrics: List[Dict[str, Any]] = []
            for kpi in understanding.extracted_kpis:
                src_chunk = chunk_by_id.get(kpi.source_chunk_id)
                if not src_chunk:
                    continue
                chunk_txt = src_chunk.get("text_content") or src_chunk.get("content") or ""
                if _verify_number_in_text(kpi.numeric_value, chunk_txt):
                    verified_metrics.append(
                        {
                            "source_chunk_id": kpi.source_chunk_id,
                            "metric_name": kpi.metric_name,
                            "numeric_value": kpi.numeric_value,
                            "unit_or_currency": kpi.unit_or_currency,
                            "period_label": kpi.period_or_context,
                            "verbatim_quote": kpi.verbatim_quote,
                        }
                    )

            saved_metrics = supabase_db.replace_extracted_metrics(workspace_id, doc_id, verified_metrics)

            # -----------------------------------------------------------------
            # Stage 13: Centroid Vector, Topics, Entities & Relationships (DOC-16)
            # -----------------------------------------------------------------
            supabase_db.update_processing_job(
                workspace_id,
                job_id,
                {"current_stage": "stage_13_relationships_and_graph", "progress_pct": 92},
            )
            centroid = _compute_mean_centroid(vectors)
            first_chunk_id = str(saved_chunks[0]["id"]) if saved_chunks else None
            supabase_db.upsert_topics_and_entities(
                workspace_id=workspace_id,
                doc_id=doc_id,
                topics=understanding.main_topics,
                entities=[e.model_dump() for e in understanding.named_entities],
                representative_chunk_id=first_chunk_id,
            )

            # -----------------------------------------------------------------
            # Stage 14 & 15: Commit Final Document State & Complete Job
            # -----------------------------------------------------------------
            final_status = "partial_ready" if partial_ocr_failures > 0 else "ready"
            updated_doc = supabase_db.update_document(
                workspace_id,
                doc_id,
                {
                    "page_count": len(page_profiles) or 1,
                    "status": final_status,
                    "processing_status": "completed",
                    "category": understanding.document_type,
                    "primary_domain": understanding.primary_domain,
                    "language_code": understanding.language_code,
                    "summary": understanding.executive_summary,
                    "key_findings": [kf.model_dump() for kf in understanding.key_findings],
                    "main_topics": understanding.main_topics,
                    "key_dates": understanding.key_dates,
                    "has_tables": len(saved_tables) > 0,
                    "table_count": len(saved_tables),
                    "metric_count": len(saved_metrics),
                    "centroid_embedding": centroid,
                    "processed_at": _utc_now_iso(),
                },
                expected_version=expected_version,
            )

            supabase_db.compute_and_save_relationships(workspace_id, doc_id)

            supabase_db.update_processing_job(
                workspace_id,
                job_id,
                {
                    "status": "completed",
                    "current_stage": "stage_15_completed",
                    "progress_pct": 100,
                    "completed_at": _utc_now_iso(),
                },
            )
            return updated_doc or {"id": doc_id, "status": final_status}

        except Exception as exc:
            supabase_db.update_document(
                workspace_id,
                doc_id,
                {
                    "status": "failed",
                    "processing_status": "failed",
                    "error_message": str(exc),
                },
            )
            supabase_db.update_processing_job(
                workspace_id,
                job_id,
                {
                    "status": "failed",
                    "last_error_code": "ERR_INGESTION_FAILED",
                    "last_error_message": str(exc),
                },
            )
            raise


ingestion_service = DocumentIngestionService()
