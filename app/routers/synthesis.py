from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from app.core.workspace import WorkspaceContext, get_verified_workspace_context
from app.db.supabase_client import supabase_db
from app.ai.contracts import (
    AITaskType,
    CrossDocumentSynthesisResponse,
    SynthesisComparisonRow,
    ContradictionAlertItem,
)
from app.ai.gateway import ai_gateway, format_untrusted_chunk
from app.services.retrieval_service import retrieval_service, citation_resolver
from app.models.schemas import ChatCitation

router = APIRouter(prefix="/synthesis", tags=["synthesis"])


class SynthesisItem(BaseModel):
    concept: str = Field(description="Key topic or dimension compared across documents")
    doc_a_stance: str = Field(description="Finding or metric in Document A")
    doc_b_stance: str = Field(description="Finding or metric in Document B")
    per_document_findings: Dict[str, str] = Field(default_factory=dict)
    comparison_analysis: str = Field(description="Consensus, divergence, or contradiction analysis")
    is_contradiction: bool = Field(default=False)


class SynthesisMatrixResponse(BaseModel):
    title: str
    synthesis_headline: str
    document_ids: List[str]
    executive_summary: str
    executive_synthesis: str
    key_agreements: List[str] = Field(default_factory=list)
    matrix: List[SynthesisItem] = Field(default_factory=list)
    comparison_matrix: List[SynthesisComparisonRow] = Field(default_factory=list)
    contradiction_alerts: List[ContradictionAlertItem] = Field(default_factory=list)
    emergent_takeaways: List[str] = Field(default_factory=list)
    citations: List[ChatCitation] = Field(default_factory=list)


class SynthesisRequest(BaseModel):
    document_ids: List[str] = Field(
        min_length=2,
        max_length=10,
        description="Between 2 and 10 workspace document IDs to compare",
    )
    focus_query: Optional[str] = Field(
        default=None,
        description="Optional analytical angle or question to focus the cross-document comparison",
    )


def _deterministic_cross_doc_synthesis(
    docs: List[Dict[str, Any]],
    chunks_by_doc: Dict[str, List[Dict[str, Any]]],
    metrics_by_doc: Dict[str, List[Dict[str, Any]]],
) -> CrossDocumentSynthesisResponse:
    """
    Deterministic, zero-hallucination cross-document synthesis derived strictly
    from the actual uploaded documents' summaries, topics, metrics, and chunks.
    """
    doc_titles = [d.get("title", "Document") for d in docs]
    headline = f"Cross-Document Comparative Synthesis: {' vs. '.join(doc_titles[:3])}"

    # 1. Compare Primary Domains & Executive Scope
    scope_findings: Dict[str, str] = {}
    summary_parts: List[str] = []
    for d in docs:
        did = str(d["id"])
        d_chunks = chunks_by_doc.get(did, [])
        first_cid = str(d_chunks[0]["id"]) if d_chunks else ""
        cite_tag = f" [cite:{first_cid}]" if first_cid else ""
        summary_snip = (d.get("summary") or "Document analyzed in workspace.").strip()
        scope_findings[did] = f"{summary_snip[:220]}{cite_tag}"
        summary_parts.append(f"**{d.get('title', did)}**: {summary_snip[:180]}{cite_tag}")

    # 2. Topic Overlap & Divergence
    topic_sets = [set(t.lower() for t in (d.get("main_topics") or []) if t) for d in docs]
    common_topics = sorted(set.intersection(*topic_sets)) if topic_sets else []
    all_topics = sorted(set.union(*topic_sets)) if topic_sets else []

    topic_findings: Dict[str, str] = {}
    for d in docs:
        did = str(d["id"])
        d_chunks = chunks_by_doc.get(did, [])
        first_cid = str(d_chunks[0]["id"]) if d_chunks else ""
        cite_tag = f" [cite:{first_cid}]" if first_cid else ""
        topics_list = d.get("main_topics") or [d.get("primary_domain") or d.get("category") or "general"]
        topic_findings[did] = f"Focuses on {', '.join(topics_list[:5])}{cite_tag}"

    comparison_rows: List[SynthesisComparisonRow] = [
        SynthesisComparisonRow(
            dimension_name="Core Scope & Findings",
            per_document_findings=scope_findings,
            consensus_or_divergence=(
                "Documents address complementary aspects of the workspace domain."
                if common_topics
                else "Documents cover distinct analytical scopes and perspectives."
            ),
        ),
        SynthesisComparisonRow(
            dimension_name="Thematic & Conceptual Focus",
            per_document_findings=topic_findings,
            consensus_or_divergence=(
                f"Shared thematic overlap on: {', '.join(common_topics[:5])}."
                if common_topics
                else f"Distinct topic distributions across {', '.join(all_topics[:6]) or 'uploaded files'}."
            ),
        ),
    ]

    # 3. Quantitative Metrics Comparison & Contradiction Detection
    contradictions: List[ContradictionAlertItem] = []
    metric_findings: Dict[str, str] = {}
    for d in docs:
        did = str(d["id"])
        d_metrics = metrics_by_doc.get(did, [])
        d_chunks = chunks_by_doc.get(did, [])
        fallback_cid = str(d_chunks[0]["id"]) if d_chunks else ""
        if d_metrics:
            m = d_metrics[0]
            mcid = str(m.get("source_chunk_id") or fallback_cid)
            cite_tag = f" [cite:{mcid}]" if mcid else ""
            metric_findings[did] = (
                f"{m.get('metric_name')}: {m.get('numeric_value')} {m.get('unit_or_currency', '')}{cite_tag}"
            )
        else:
            cite_tag = f" [cite:{fallback_cid}]" if fallback_cid else ""
            metric_findings[did] = f"Qualitative text passage ({len(d_chunks)} indexed chunks){cite_tag}"

    # Check if two documents report different values for the same metric name or unit
    if len(docs) >= 2:
        d0, d1 = docs[0], docs[1]
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
            if contradictions:
                break

    comparison_rows.append(
        SynthesisComparisonRow(
            dimension_name="Quantitative Indicators & Evidence",
            per_document_findings=metric_findings,
            consensus_or_divergence=(
                contradictions[0].explanation
                if contradictions
                else "Quantitative metrics across documents represent distinct measurements without direct contradiction."
            ),
        )
    )

    agreements = (
        [f"Shared domain concepts: {', '.join(common_topics[:5])}"]
        if common_topics
        else ["Each document provides primary source evidence within the workspace corpus."]
    )
    takeaways = [
        f"Synthesized {len(docs)} workspace documents ({', '.join(doc_titles)}).",
        f"Identified {len(common_topics)} shared topic areas and {len(contradictions)} quantitative variations.",
    ]

    return CrossDocumentSynthesisResponse(
        synthesis_headline=headline,
        executive_synthesis=" ".join(summary_parts),
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

    for doc_id in request.document_ids:
        doc = supabase_db.get_workspace_document(ctx.workspace_id, doc_id)
        if not doc:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "ERR_DOCUMENT_NOT_FOUND",
                    "message": f"Document '{doc_id}' not found in workspace.",
                },
            )
        docs.append(doc)
        doc_chunks = supabase_db.get_document_chunks(ctx.workspace_id, doc_id)
        for c in doc_chunks:
            c["doc_id"] = doc_id
            c["doc_title"] = doc.get("title", "Document")
        chunks_by_doc[doc_id] = doc_chunks
        all_candidate_chunks.extend(doc_chunks[:6])
        metrics_by_doc[doc_id] = supabase_db.get_workspace_metrics(ctx.workspace_id, doc_ids=[doc_id])

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
                f"Focus Query: {request.focus_query or 'Comprehensive comparative synthesis and contradiction audit'}\n\n"
                f"Evidence Passages:\n{passages}"
            )
            synthesis_result = await ai_gateway.generate_structured(
                task_type=AITaskType.CROSS_DOC_SYNTHESIS,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                instructions=(
                    "You are the DocuMind Cross-Document Synthesis Engine. Compare the provided documents strictly "
                    "using the <untrusted_document_chunk> passages. Every claim in executive_synthesis and "
                    "per_document_findings MUST include [cite:CHUNK_ID]. Flag any factual or quantitative contradictions."
                ),
                input_blocks=prompt_input,
                response_model=CrossDocumentSynthesisResponse,
                scope_key=f"synthesis:{':'.join(sorted(request.document_ids))}",
            )
        except Exception as e:
            print(f"[SynthesisRouter] OpenAI synthesis notice, using deterministic synthesis: {e}")

    if synthesis_result is None:
        synthesis_result = _deterministic_cross_doc_synthesis(docs, chunks_by_doc, metrics_by_doc)

    # Persist any discovered contradictions as relationship edges in public.document_relationships
    for alert in synthesis_result.contradiction_alerts:
        if alert.doc_a_id and alert.doc_b_id and alert.doc_a_id != alert.doc_b_id:
            supabase_db.save_contradiction_relationship(
                workspace_id=ctx.workspace_id,
                source_doc_id=alert.doc_a_id,
                target_doc_id=alert.doc_b_id,
                explanation=alert.explanation,
                evidence_summary=alert.model_dump(),
            )

    # Resolve [cite:CHUNK_ID] citations across the executive synthesis and matrix findings
    combined_citation_text = synthesis_result.executive_synthesis
    for row in synthesis_result.comparison_matrix:
        for finding in row.per_document_findings.values():
            combined_citation_text += f" {finding}"

    _, resolved_citations = citation_resolver.resolve_answer_citations(
        raw_answer=combined_citation_text,
        candidate_chunks=all_candidate_chunks,
    )

    # Build backward-compatible `matrix` items alongside `comparison_matrix`
    doc_a_id = str(docs[0]["id"])
    doc_b_id = str(docs[1]["id"])
    contradiction_topics = {c.topic_or_metric.lower() for c in synthesis_result.contradiction_alerts}

    legacy_matrix: List[SynthesisItem] = []
    for row in synthesis_result.comparison_matrix:
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
        executive_summary=synthesis_result.executive_synthesis,
        executive_synthesis=synthesis_result.executive_synthesis,
        key_agreements=synthesis_result.key_agreements,
        matrix=legacy_matrix,
        comparison_matrix=synthesis_result.comparison_matrix,
        contradiction_alerts=synthesis_result.contradiction_alerts,
        emergent_takeaways=synthesis_result.emergent_takeaways,
        citations=resolved_citations,
    )


@router.post("/compare", response_model=SynthesisMatrixResponse)
async def compare_workspace_documents(
    request: SynthesisRequest,
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    """
    Stage 5 Cross-Document Synthesis & Contradiction Detection (DOC-16, PR-031, PR-032).
    Compares 2 to 10 workspace documents over their actual extracted chunks, summaries, and metrics.
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
