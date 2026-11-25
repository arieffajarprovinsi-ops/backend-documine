import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple
from app.ai.gateway import ai_gateway
from app.db.supabase_client import supabase_db
from app.models.schemas import BoundingBox, ChatCitation

CITATION_TAG_REGEX = re.compile(r"\[cite:([a-zA-Z0-9_-]+)\]")


class CitationResolver:
    """
    Deterministic Citation & Coordinate Resolver (ADR-009, PR-028, Invariant 3).
    Never allows an LLM to invent page numbers or bounding boxes.
    Resolves [cite:CHUNK_ID] strictly against the retrieved workspace chunk map.
    """

    @staticmethod
    def resolve_citations(
        answer_text: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Tuple[str, List[ChatCitation]]:
        chunk_map: Dict[str, Dict[str, Any]] = {
            str(c.get("chunk_id") or c.get("id")): c for c in retrieved_chunks
        }
        cited_ids_ordered: List[str] = []
        for match in CITATION_TAG_REGEX.finditer(answer_text or ""):
            cid = match.group(1).strip()
            if cid in chunk_map and cid not in cited_ids_ordered:
                cited_ids_ordered.append(cid)

        citations: List[ChatCitation] = []
        for idx, cid in enumerate(cited_ids_ordered):
            c = chunk_map[cid]
            raw_boxes = c.get("bounding_boxes") or []
            boxes = [
                BoundingBox(**b) if isinstance(b, dict) else b
                for b in raw_boxes
            ]
            txt = (c.get("text_content") or c.get("content") or "").strip()
            citations.append(
                ChatCitation(
                    citation_id=f"cite_{idx + 1}",
                    chunk_id=cid,
                    document_id=str(c.get("doc_id") or c.get("document_id") or ""),
                    document_title=c.get("doc_title"),
                    page_number=int(c.get("page_number", 1)),
                    quote=txt[:280],
                    bounding_boxes=boxes,
                )
            )

        return answer_text, citations

    @classmethod
    def resolve_answer_citations(
        cls,
        raw_answer: str,
        candidate_chunks: List[Dict[str, Any]],
    ) -> Tuple[str, List[ChatCitation]]:
        return cls.resolve_citations(raw_answer, candidate_chunks)


def _diversify_hits_across_documents(
    candidates: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Maximal Marginal Document Coverage (Round-Robin Document Interleaving):
    Ensures that when 3, 4, or 25 PDFs contain relevant passages, verbose PDFs 1 & 2
    cannot monopolize every retrieval slot and starve PDF 3, PDF 4, etc.
    Round 1 selects the highest-scoring passage from each distinct matching document,
    Round 2 selects the second-best passage from each document, and so on up to `limit`.
    """
    if not candidates:
        return []

    def _score_key(row: Dict[str, Any]) -> Tuple[float, float]:
        return (
            float(row.get("rrf_score", 0.0)),
            float(row.get("vector_similarity", 0.0)),
        )

    sorted_all = sorted(candidates, key=_score_key, reverse=True)
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    doc_order: List[str] = []

    for h in sorted_all:
        did = str(h.get("doc_id") or h.get("document_id") or "")
        if not did:
            did = "__unknown__"
        if did not in buckets:
            buckets[did] = []
            doc_order.append(did)
        buckets[did].append(h)

    if len(doc_order) <= 1:
        return sorted_all[:limit]

    interleaved: List[Dict[str, Any]] = []
    seen_chunks: set[str] = set()
    round_idx = 0

    while len(interleaved) < limit:
        added_in_round = False
        for did in doc_order:
            bucket = buckets[did]
            if round_idx < len(bucket):
                item = bucket[round_idx]
                cid = str(item.get("chunk_id") or item.get("id") or "")
                if cid and cid not in seen_chunks:
                    seen_chunks.add(cid)
                    interleaved.append(item)
                    added_in_round = True
                    if len(interleaved) >= limit:
                        break
        if not added_in_round:
            break
        round_idx += 1

    return interleaved


class RetrievalService:
    """
    Hybrid RRF Search & Stratified Multi-Document Retrieval Service (DOC-12, ADR-006, PR-009, PR-025).
    Guarantees cross-document citation coverage across 2 to 25+ workspace PDFs.
    """

    async def search_hybrid(
        self,
        *,
        workspace_id: str,
        user_id: Optional[str],
        query: str,
        doc_ids: Optional[List[str]] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        match_count: int = 10,
        min_vector_similarity: float = 0.28,
    ) -> Dict[str, Any]:
        q_clean = (query or "").strip()
        if not q_clean:
            return {"results": [], "retrieval_status": "insufficient_evidence"}

        # Run query embedding and (when repository-scoped) cached workspace doc lookup in parallel
        if doc_ids:
            vectors = await ai_gateway.embed_texts(
                texts=[q_clean],
                workspace_id=workspace_id,
                user_id=user_id,
                is_query=True,
            )
            workspace_docs: List[Dict[str, Any]] = []
        else:
            vectors, workspace_docs = await asyncio.gather(
                ai_gateway.embed_texts(
                    texts=[q_clean],
                    workspace_id=workspace_id,
                    user_id=user_id,
                    is_query=True,
                ),
                asyncio.to_thread(
                    supabase_db.list_workspace_documents,
                    workspace_id,
                    category=category,
                ),
            )

        query_vec = vectors[0] if vectors else [0.0] * 1536

        # Determine effective multi-doc scope even when caller is in "All Repository Documents" (doc_ids=None)
        ready_ws_ids = [
            str(d["id"])
            for d in workspace_docs
            if d.get("id") and (d.get("status") or d.get("processing_status")) in ("ready", "partial_ready", "completed", None)
        ]
        scoped_ids = list(doc_ids) if doc_ids else ready_ws_ids

        # Case 1: Single document scope
        if len(scoped_ids) == 1:
            raw_hits = await asyncio.to_thread(
                supabase_db.search_chunks_hybrid,
                workspace_id=workspace_id,
                query_embedding=query_vec,
                query_text=q_clean,
                match_count=match_count,
                doc_ids=scoped_ids,
                category=category,
                tags=tags,
                min_vector_similarity=min_vector_similarity,
            )
            hits = raw_hits[:match_count]

        # Case 2: Small-to-Medium Multi-PDF Corpus (2..10 PDFs, e.g. 4 uploaded PDFs!)
        # Parallel per-PDF hybrid retrieval guarantees PDF 3 & PDF 4 are never starved by PDF 1 & 2.
        elif 2 <= len(scoped_ids) <= 10:
            per_doc_quota = 3 if len(scoped_ids) <= 5 else 2
            effective_limit = max(match_count, min(18, len(scoped_ids) * 2))
            merged_by_chunk: Dict[str, Dict[str, Any]] = {}

            per_doc_tasks = [
                asyncio.to_thread(
                    supabase_db.search_chunks_hybrid,
                    workspace_id=workspace_id,
                    query_embedding=query_vec,
                    query_text=q_clean,
                    match_count=per_doc_quota,
                    doc_ids=[did],
                    category=category,
                    tags=tags,
                    min_vector_similarity=min(min_vector_similarity, 0.24),
                )
                for did in scoped_ids
            ]
            global_task = asyncio.to_thread(
                supabase_db.search_chunks_hybrid,
                workspace_id=workspace_id,
                query_embedding=query_vec,
                query_text=q_clean,
                match_count=max(effective_limit, 16),
                doc_ids=doc_ids if doc_ids else None,
                category=category,
                tags=tags,
                min_vector_similarity=min(min_vector_similarity, 0.24),
            )
            gathered = await asyncio.gather(*per_doc_tasks, global_task)
            for doc_hits in gathered[:-1]:
                for h in doc_hits:
                    merged_by_chunk[str(h.get("chunk_id") or h.get("id"))] = h
            for h in gathered[-1]:
                merged_by_chunk.setdefault(str(h.get("chunk_id") or h.get("id")), h)

            hits = _diversify_hits_across_documents(
                list(merged_by_chunk.values()),
                limit=effective_limit,
            )

        # Case 3: Large Multi-PDF Repository (11..50+ PDFs, e.g. 25-PDF WAICY Jury Corpus!)
        # Uses wide candidate pool + lexical title/topic targeted probes + round-robin document diversity.
        else:
            effective_limit = max(match_count, 16)
            wide_pool_size = max(60, effective_limit * 4)

            # Identify up to 6 documents whose title/topics/summary directly match query terms
            q_terms = {
                w.lower()
                for w in re.findall(r"[A-Za-z0-9]{3,}", q_clean)
                if w.lower() not in {"the", "and", "for", "with", "from", "what", "how", "are", "yang", "dan", "dari", "apa"}
            }
            matched_doc_ids: List[str] = []
            if q_terms and workspace_docs:
                scored_docs: List[Tuple[int, str]] = []
                for d in workspace_docs:
                    did = str(d.get("id") or "")
                    if not did:
                        continue
                    meta_blob = (
                        f"{d.get('title') or ''} {' '.join(d.get('main_topics') or [])} "
                        f"{d.get('primary_domain') or ''} {d.get('summary') or ''}"
                    ).lower()
                    overlap = sum(1 for t in q_terms if t in meta_blob)
                    if overlap > 0:
                        scored_docs.append((overlap, did))
                scored_docs.sort(reverse=True)
                matched_doc_ids = [did for _, did in scored_docs[:6]]

            tasks = [
                asyncio.to_thread(
                    supabase_db.search_chunks_hybrid,
                    workspace_id=workspace_id,
                    query_embedding=query_vec,
                    query_text=q_clean,
                    match_count=wide_pool_size,
                    doc_ids=doc_ids if doc_ids else None,
                    category=category,
                    tags=tags,
                    min_vector_similarity=min(min_vector_similarity, 0.24),
                )
            ]
            for mdid in matched_doc_ids:
                tasks.append(
                    asyncio.to_thread(
                        supabase_db.search_chunks_hybrid,
                        workspace_id=workspace_id,
                        query_embedding=query_vec,
                        query_text=q_clean,
                        match_count=2,
                        doc_ids=[mdid],
                        category=category,
                        tags=tags,
                        min_vector_similarity=0.20,
                    )
                )

            gathered_large = await asyncio.gather(*tasks)
            merged_large: Dict[str, Dict[str, Any]] = {}
            for batch in gathered_large:
                for h in batch:
                    cid = str(h.get("chunk_id") or h.get("id"))
                    if cid and cid not in merged_large:
                        merged_large[cid] = h

            hits = _diversify_hits_across_documents(
                list(merged_large.values()),
                limit=effective_limit,
            )

        for h in hits:
            v_sim = float(h.get("vector_similarity", 0.0))
            expl = h.get("match_explanation", "")
            if v_sim >= 0.48 or expl == "hybrid_semantic_and_keyword":
                h["confidence_tier"] = "high"
            elif v_sim >= 0.30 or expl == "exact_keyword_match":
                h["confidence_tier"] = "medium"
            else:
                h["confidence_tier"] = "low"

        has_medium_or_high = any(h.get("confidence_tier") in ("high", "medium") for h in hits)
        status = "grounded" if (hits and has_medium_or_high) else "insufficient_evidence"
        return {
            "results": hits,
            "retrieval_status": status,
        }


citation_resolver = CitationResolver()
retrieval_service = RetrievalService()
