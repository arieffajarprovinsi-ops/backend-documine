import asyncio
import re
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Set
from app.core.workspace import WorkspaceContext, get_verified_workspace_context
from app.db.supabase_client import supabase_db
from app.ai.contracts import (
    AITaskType,
    CrossDocumentSynthesisResponse,
    DocumentFindingEntry,
    SynthesisComparisonRow,
    ContradictionAlertItem,
)
from app.ai.gateway import ai_gateway, format_untrusted_chunk
from app.services.retrieval_service import citation_resolver
from app.models.schemas import ChatCitation

router = APIRouter(prefix="/synthesis", tags=["synthesis"])

_CITE_TAG_RE = re.compile(r"\[cite:([a-zA-Z0-9_-]+)\]")
_SYNTH_STOPWORDS = {
    "yang", "dan", "atau", "dari", "untuk", "dengan", "pada", "dalam", "ini", "itu",
    "oleh", "sebagai", "adalah", "akan", "telah", "dapat", "juga", "tahun", "ajaran",
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "have",
    "dokumen", "document", "peserta", "kegiatan", "secara", "memuat", "merupakan",
}


def _clean_excerpt(text: str, max_chars: int = 280) -> str:
    """Truncate cleanly on a sentence or word boundary instead of mid-word."""
    normalized = " ".join((text or "").split()).strip()
    if not normalized or len(normalized) <= max_chars:
        return normalized
    window = normalized[:max_chars]
    # Prefer ending cleanly at a sentence period if near the end
    last_period = window.rfind(". ")
    if last_period >= int(max_chars * 0.55):
        return window[: last_period + 1].strip()
    last_space = window.rfind(" ")
    if last_space > 24:
        return window[:last_space].rstrip(",;:-") + "..."
    return window.rstrip(",;:-") + "..."


def _humanize_citations(text: str, chunks_by_id: Dict[str, Dict[str, Any]]) -> str:
    """
    Replace raw `[cite:UUID]` tokens with clean readable inline citations `[DocTitle · p. X]`
    so users never see raw database UUIDs in rendered synthesis prose or matrix cells.
    """
    if not text:
        return ""

    def _repl(match: re.Match[str]) -> str:
        cid = match.group(1)
        chunk = chunks_by_id.get(cid)
        if not chunk:
            return ""
        title = str(chunk.get("doc_title") or "Document").strip()
        short_title = title[:34] + ("..." if len(title) > 34 else "")
        page = int(chunk.get("page_number") or 1)
        return f"({short_title}, p. {page})"

    cleaned = _CITE_TAG_RE.sub(_repl, text)
    return re.sub(r"\s+([.,;:])", r"\1", re.sub(r" {2,}", " ", cleaned)).strip()


def _extract_meaningful_terms(text: str) -> Set[str]:
    words = re.findall(r"[A-Za-z0-9]{4,}", (text or "").lower())
    return {w for w in words if w not in _SYNTH_STOPWORDS}


class SynthesisItem(BaseModel):
    concept: str = Field(description="Key topic or dimension compared across documents")
    doc_a_stance: str = Field(description="Finding or metric in Document A")
    doc_b_stance: str = Field(description="Finding or metric in Document B")
    per_document_findings: Dict[str, str] = Field(default_factory=dict)
    comparison_analysis: str = Field(description="Consensus, divergence, or contradiction analysis")
    is_contradiction: bool = Field(default=False)


class SynthesisComparisonRowOut(BaseModel):
    dimension_name: str
    per_document_findings: Dict[str, str] = Field(default_factory=dict)
    consensus_or_divergence: str


class SynthesisMatrixResponse(BaseModel):
    title: str
    synthesis_headline: str
    document_ids: List[str]
    executive_summary: str
    executive_synthesis: str
    key_agreements: List[str] = Field(default_factory=list)
    matrix: List[SynthesisItem] = Field(default_factory=list)
    comparison_matrix: List[SynthesisComparisonRowOut] = Field(default_factory=list)
    contradiction_alerts: List[ContradictionAlertItem] = Field(default_factory=list)
    emergent_takeaways: List[str] = Field(default_factory=list)
    citations: List[ChatCitation] = Field(default_factory=list)


class SynthesisRequest(BaseModel):
    document_ids: List[str] = Field(
        min_length=2,
        max_length=12,
        description="Between 2 and 12 workspace document IDs to compare",
    )
    focus_query: Optional[str] = Field(
        default=None,
        description="Optional analytical angle or question to focus the cross-document comparison",
    )


def _deterministic_cross_doc_synthesis(
    docs: List[Dict[str, Any]],
    chunks_by_doc: Dict[str, List[Dict[str, Any]]],
    metrics_by_doc: Dict[str, List[Dict[str, Any]]],
    focus_query: Optional[str] = None,
) -> CrossDocumentSynthesisResponse:
    """
    Rich evidence-grounded cross-document synthesis derived strictly
    from actual summaries, chunk text, topics, dates, entities, and metrics.
    """
    doc_titles = [d.get("title", "Document") for d in docs]
    headline = f"Cross-Document Synthesis: {' vs. '.join(doc_titles[:3])}"

    scope_findings: Dict[str, str] = {}
    summary_parts: List[str] = []
    doc_terms: List[Set[str]] = []
    time_findings: Dict[str, str] = {}

    for d in docs:
        did = str(d["id"])
        d_title = d.get("title", "Document")
        d_chunks = chunks_by_doc.get(did, [])
        first_cid = str(d_chunks[0]["id"]) if d_chunks else ""
        cite_tag = f" [cite:{first_cid}]" if first_cid else ""

        chunk_blob = " ".join(
            (c.get("text_content") or c.get("content") or "").strip()
            for c in d_chunks[:4]
        )
        raw_summary = (d.get("summary") or "").strip()
        snippet_source = raw_summary or _clean_excerpt(chunk_blob, 260) or "Indexed workspace document."
        clean_scope = _clean_excerpt(snippet_source, 260)
        scope_findings[did] = f"{clean_scope}{cite_tag}"
        summary_parts.append(f"**{d_title}**: {_clean_excerpt(snippet_source, 220)}{cite_tag}")

        combined_text = f"{d_title} {raw_summary} {' '.join(d.get('main_topics') or [])} {chunk_blob}"
        doc_terms.append(_extract_meaningful_terms(combined_text))

        # Extract concrete schedule / time / year / page telemetry from the actual document
        dates_or_years = sorted(set(re.findall(r"\b(?:20\d{2}(?:[-/]\d{2,4})?|\d{1,2}[:.]\d{2}\s*(?:WIB|WITA|WIT)?|\d+\s*(?:jam|hours?|menit|minutes?))\b", combined_text, re.IGNORECASE)))
        pages = int(d.get("page_count") or 1)
        tbl_count = int(d.get("table_count") or 0)
        if dates_or_years:
            time_findings[did] = f"Timeline/markers: {', '.join(dates_or_years[:5])} ({pages}p, {len(d_chunks)} chunks){cite_tag}"
        else:
            time_findings[did] = f"{pages} pages, {len(d_chunks)} indexed passages, {tbl_count} structured tables{cite_tag}"

    topic_sets = [set(t.lower() for t in (d.get("main_topics") or []) if t) for d in docs]
    common_topics = sorted(set.intersection(*topic_sets)) if topic_sets else []
    all_topics = sorted(set.union(*topic_sets)) if topic_sets else []
    common_terms = sorted(set.intersection(*doc_terms)) if len(doc_terms) >= 2 else []

    topic_findings: Dict[str, str] = {}
    for d in docs:
        did = str(d["id"])
        d_chunks = chunks_by_doc.get(did, [])
        first_cid = str(d_chunks[0]["id"]) if d_chunks else ""
        cite_tag = f" [cite:{first_cid}]" if first_cid else ""
        topics_list = d.get("main_topics") or [d.get("primary_domain") or d.get("category") or "general"]
        topic_findings[did] = f"Key themes: {', '.join(topics_list[:5])}{cite_tag}"

    overlap_note = (
        f"Both files share institutional/thematic context around: {', '.join(common_terms[:6])}."
        if common_terms
        else (
            f"Shared focus areas: {', '.join(common_topics[:5])}."
            if common_topics
            else "Each document addresses a distinct operational workflow (no identical topics detected)."
        )
    )

    def _dict_to_entries(d_map: Dict[str, str]) -> List[DocumentFindingEntry]:
        return [DocumentFindingEntry(doc_id=k, finding=v) for k, v in d_map.items()]

    comparison_rows: List[SynthesisComparisonRow] = [
        SynthesisComparisonRow(
            dimension_name="Executive Scope & Purpose",
            findings_list=_dict_to_entries(scope_findings),
            consensus_or_divergence=(
                f"{overlap_note} {('Focused on: ' + focus_query) if focus_query else ''}".strip()
            ),
        ),
        SynthesisComparisonRow(
            dimension_name="Schedule, Period & Document Structure",
            findings_list=_dict_to_entries(time_findings),
            consensus_or_divergence=(
                f"Common timeline/context terms ({', '.join(common_terms[:4])}) align across the compared files."
                if common_terms
                else "Schedules, page depths, and structural layouts serve different operational purposes."
            ),
        ),
        SynthesisComparisonRow(
            dimension_name="Thematic & Conceptual Focus",
            findings_list=_dict_to_entries(topic_findings),
            consensus_or_divergence=(
                f"Shared topics: {', '.join(common_topics[:5])}."
                if common_topics
                else f"Complementary topic coverage spanning {', '.join(all_topics[:6]) or 'selected documents'}."
            ),
        ),
    ]

    # Quantitative Metrics Comparison & Contradiction Detection
    contradictions: List[ContradictionAlertItem] = []
    metric_findings: Dict[str, str] = {}
    for d in docs:
        did = str(d["id"])
        d_metrics = metrics_by_doc.get(did, [])
        d_chunks = chunks_by_doc.get(did, [])
        fallback_cid = str(d_chunks[0]["id"]) if d_chunks else ""
        if d_metrics:
            top_metrics = [
                f"{m.get('metric_name')}: {m.get('numeric_value')} {m.get('unit_or_currency', '')}".strip()
                for m in d_metrics[:3]
            ]
            mcid = str(d_metrics[0].get("source_chunk_id") or fallback_cid)
            cite_tag = f" [cite:{mcid}]" if mcid else ""
            metric_findings[did] = f"{' · '.join(top_metrics)}{cite_tag}"
        else:
            cite_tag = f" [cite:{fallback_cid}]" if fallback_cid else ""
            metric_findings[did] = f"Narrative/procedural content ({len(d_chunks)} indexed passages, 0 extracted numeric KPIs){cite_tag}"

    if len(docs) >= 2:
        for i_idx in range(len(docs)):
            for j_idx in range(i_idx + 1, len(docs)):
                d0, d1 = docs[i_idx], docs[j_idx]
                id0, id1 = str(d0["id"]), str(d1["id"])
                m_list0, m_list1 = metrics_by_doc.get(id0, []), metrics_by_doc.get(id1, [])
                for m0 in m_list0:
                    for m1 in m_list1:
                        val0 = float(m0.get("numeric_value", 0.0))
                        val1 = float(m1.get("numeric_value", 0.0))
                        u0 = (m0.get("unit_or_currency") or "").strip().upper()
                        u1 = (m1.get("unit_or_currency") or "").strip().upper()
                        if val0 != val1 and (u0 == u1 or m0.get("metric_name") == m1.get("metric_name")):
                            c0 = str(m0.get("source_chunk_id") or (chunks_by_doc[id0][0]["id"] if chunks_by_doc.get(id0) else ""))
                            c1 = str(m1.get("source_chunk_id") or (chunks_by_doc[id1][0]["id"] if chunks_by_doc.get(id1) else ""))
                            contradictions.append(
                                ContradictionAlertItem(
                                    topic_or_metric=str(m0.get("metric_name") or u0 or "Quantitative Metric"),
                                    doc_a_id=id0,
                                    doc_a_claim=f"{m0.get('metric_name')}: {val0} {u0}",
                                    doc_a_chunk_id=c0,
                                    doc_b_id=id1,
                                    doc_b_claim=f"{m1.get('metric_name')}: {val1} {u1}",
                                    doc_b_chunk_id=c1,
                                    severity="medium",
                                    explanation=(
                                        f"{d0.get('title')} reports {val0} {u0} whereas {d1.get('title')} reports {val1} {u1}."
                                    ),
                                )
                            )
                            break
                    if len(contradictions) >= 4:
                        break
                if len(contradictions) >= 4:
                    break
            if len(contradictions) >= 4:
                break

    has_any_metrics = any(bool(metrics_by_doc.get(str(d["id"]))) for d in docs)
    comparison_rows.append(
        SynthesisComparisonRow(
            dimension_name="Quantitative Indicators & Metrics",
            findings_list=_dict_to_entries(metric_findings),
            consensus_or_divergence=(
                contradictions[0].explanation
                if contradictions
                else (
                    "Extracted quantitative figures measure separate scopes without conflicting totals."
                    if has_any_metrics
                    else "Documents are primarily qualitative/procedural rather than numeric KPI sheets."
                )
            ),
        )
    )

    agreements: List[str] = []
    if common_terms:
        agreements.append(f"Shared context & terminology: {', '.join(common_terms[:6])}")
    if common_topics:
        agreements.append(f"Shared topic tags: {', '.join(common_topics[:5])}")
    if not agreements:
        agreements.append(
            f"Complementary coverage across {len(docs)} documents without overlapping duplicates."
        )

    takeaways = [
        f"Compared {len(docs)} documents ({' vs. '.join(doc_titles[:3])}) across scope, timeline markers, themes, and quantitative metrics.",
        (
            f"Detected {len(common_terms)} overlapping context terms ({', '.join(common_terms[:5])})."
            if common_terms
            else "Each document documents an independent activity or specification."
        ),
        (
            f"Flagged {len(contradictions)} metric variance(s) requiring review."
            if contradictions
            else "No factual contradictions or conflicting metrics were detected."
        ),
    ]

    return CrossDocumentSynthesisResponse(
        synthesis_headline=headline,
        executive_synthesis="\n\n".join(summary_parts),
        key_agreements=agreements,
        comparison_matrix=comparison_rows,
        contradiction_alerts=contradictions,
        emergent_takeaways=takeaways,
    )


async def _execute_cross_doc_synthesis(
    request: SynthesisRequest,
    ctx: WorkspaceContext,
) -> SynthesisMatrixResponse:
    docs: List[Dict[str, Any]] = []
    chunks_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    metrics_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    all_candidate_chunks: List[Dict[str, Any]] = []

    async def _fetch_one_doc(doc_id: str):
        doc, doc_chunks, doc_metrics = await asyncio.gather(
            asyncio.to_thread(supabase_db.get_workspace_document, ctx.workspace_id, doc_id),
            asyncio.to_thread(supabase_db.get_document_chunks, ctx.workspace_id, doc_id),
            asyncio.to_thread(supabase_db.get_workspace_metrics, ctx.workspace_id, doc_ids=[doc_id]),
        )
        return doc_id, doc, doc_chunks, doc_metrics

    fetched_items = await asyncio.gather(*[_fetch_one_doc(did) for did in request.document_ids])
    per_doc_chunk_cap = max(2, min(6, 24 // max(2, len(request.document_ids))))

    for doc_id, doc, doc_chunks, doc_metrics in fetched_items:
        if not doc:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "ERR_DOCUMENT_NOT_FOUND",
                    "message": f"Document '{doc_id}' not found in workspace.",
                },
            )
        docs.append(doc)
        for c in doc_chunks:
            c["doc_id"] = doc_id
            c["doc_title"] = doc.get("title", "Document")
        chunks_by_doc[doc_id] = doc_chunks
        all_candidate_chunks.extend(doc_chunks[:per_doc_chunk_cap])
        metrics_by_doc[doc_id] = doc_metrics

    chunks_by_id = {str(c["id"]): c for c in all_candidate_chunks if c.get("id")}
    synthesis_result: Optional[CrossDocumentSynthesisResponse] = None

    if ai_gateway.is_openai_configured and all_candidate_chunks:
        try:
            passages = "\n\n".join(
                format_untrusted_chunk(
                    chunk_id=str(c["id"]),
                    text_content=c.get("text_content") or c.get("content") or "",
                    page_number=int(c.get("page_number", 1)),
                    doc_title=c.get("doc_title", "Document"),
                    doc_id=str(c.get("doc_id") or c.get("document_id")),
                )
                for c in all_candidate_chunks
            )
            doc_overview = "\n".join(
                f"- doc_id={d['id']} | title={d.get('title')} | summary={d.get('summary')}"
                for d in docs
            )
            prompt_input = (
                f"Documents to Compare:\n{doc_overview}\n\n"
                f"Focus Query: {request.focus_query or 'Comprehensive comparative synthesis, overlap audit, and actionable takeaways'}\n\n"
                f"Evidence Passages:\n{passages}"
            )
            synthesis_result = await ai_gateway.generate_structured(
                task_type=AITaskType.CROSS_DOC_SYNTHESIS,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                instructions=(
                    "You are the DocuMind Cross-Document Synthesis Engine. Compare the provided documents strictly "
                    "using the <untrusted_document_chunk> passages. Write clear, complete, insightful sentences (never cut mid-word). "
                    "Include [cite:CHUNK_ID] citations naturally after concrete claims, compare 4 distinct dimensions "
                    "(Executive Scope, Key Schedules & Requirements, Shared Context & Concepts, Quantitative / Risk Differences), "
                    "and list concrete agreements, emergent takeaways, and any contradictions."
                ),
                input_blocks=prompt_input,
                response_model=CrossDocumentSynthesisResponse,
                scope_key=f"synthesis:{':'.join(sorted(request.document_ids))}",
            )
        except Exception as e:
            print(f"[SynthesisRouter] OpenAI synthesis notice, using deterministic synthesis: {e}")

    if synthesis_result is None:
        synthesis_result = _deterministic_cross_doc_synthesis(
            docs, chunks_by_doc, metrics_by_doc, focus_query=request.focus_query
        )

    for alert in synthesis_result.contradiction_alerts:
        if alert.doc_a_id and alert.doc_b_id and alert.doc_a_id != alert.doc_b_id:
            supabase_db.save_contradiction_relationship(
                workspace_id=ctx.workspace_id,
                source_doc_id=alert.doc_a_id,
                target_doc_id=alert.doc_b_id,
                explanation=alert.explanation,
                evidence_summary=alert.model_dump(),
            )

    combined_citation_text = synthesis_result.executive_synthesis
    for row in synthesis_result.comparison_matrix:
        for entry in row.findings_list:
            combined_citation_text += f" {entry.finding}"

    _, resolved_citations = citation_resolver.resolve_answer_citations(
        raw_answer=combined_citation_text,
        candidate_chunks=all_candidate_chunks,
    )

    # Humanize all `[cite:UUID]` tokens across rendered fields so raw UUIDs never appear
    clean_exec = _humanize_citations(synthesis_result.executive_synthesis, chunks_by_id)
    cleaned_rows: List[SynthesisComparisonRowOut] = []
    for row in synthesis_result.comparison_matrix:
        cleaned_findings = {
            entry.doc_id: _humanize_citations(entry.finding, chunks_by_id)
            for entry in row.findings_list
        }
        cleaned_rows.append(
            SynthesisComparisonRowOut(
                dimension_name=row.dimension_name,
                per_document_findings=cleaned_findings,
                consensus_or_divergence=_humanize_citations(row.consensus_or_divergence, chunks_by_id),
            )
        )

    doc_a_id = str(docs[0]["id"])
    doc_b_id = str(docs[1]["id"])
    contradiction_topics = {c.topic_or_metric.lower() for c in synthesis_result.contradiction_alerts}

    legacy_matrix: List[SynthesisItem] = []
    for row in cleaned_rows:
        is_contra = (
            row.dimension_name.lower() in contradiction_topics
            or "contradict" in row.consensus_or_divergence.lower()
            or ("quantitative" in row.dimension_name.lower() and len(synthesis_result.contradiction_alerts) > 0)
        )
        legacy_matrix.append(
            SynthesisItem(
                concept=row.dimension_name,
                doc_a_stance=row.per_document_findings.get(doc_a_id, "Not specified in Document A"),
                doc_b_stance=row.per_document_findings.get(doc_b_id, "Not specified in Document B"),
                per_document_findings=row.per_document_findings,
                comparison_analysis=row.consensus_or_divergence,
                is_contradiction=is_contra,
            )
        )

    return SynthesisMatrixResponse(
        title=synthesis_result.synthesis_headline,
        synthesis_headline=synthesis_result.synthesis_headline,
        document_ids=request.document_ids,
        executive_summary=clean_exec,
        executive_synthesis=clean_exec,
        key_agreements=[_humanize_citations(a, chunks_by_id) for a in synthesis_result.key_agreements],
        matrix=legacy_matrix,
        comparison_matrix=cleaned_rows,
        contradiction_alerts=synthesis_result.contradiction_alerts,
        emergent_takeaways=[_humanize_citations(t, chunks_by_id) for t in synthesis_result.emergent_takeaways],
        citations=resolved_citations,
    )


@router.post("/compare", response_model=SynthesisMatrixResponse)
async def compare_workspace_documents(
    request: SynthesisRequest,
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    """
    Stage 5 Cross-Document Synthesis & Contradiction Detection (DOC-16, PR-031, PR-032).
    """
    return await _execute_cross_doc_synthesis(request, ctx)


@router.post("/matrix", response_model=SynthesisMatrixResponse)
async def generate_synthesis_matrix(
    request: SynthesisRequest,
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    """
    Backward-compatible endpoint alias for `/api/synthesis/compare`.
    """
    return await _execute_cross_doc_synthesis(request, ctx)
