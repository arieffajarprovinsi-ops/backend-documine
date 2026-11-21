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

        # If the answer was generated via deterministic fallback without explicit [cite:...] tags
        # but retrieved_chunks has high-confidence evidence referenced in the answer, attach the top chunk only if grounded
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


class RetrievalService:
    """
    Hybrid RRF Search & Stratified Multi-Document Retrieval Service (DOC-12, ADR-006, PR-009, PR-025).
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

        vectors = await ai_gateway.embed_texts(
            texts=[q_clean],
            workspace_id=workspace_id,
            user_id=user_id,
            is_query=True,
        )
        query_vec = vectors[0] if vectors else [0.0] * 1536

        # Stratified Multi-Document Subset Scope (DOC-12 Section 3: 2 <= len(doc_ids) <= 15)
        if doc_ids and len(doc_ids) > 1:
            per_doc_quota = 2
            merged_by_chunk: Dict[str, Dict[str, Any]] = {}
            for did in doc_ids[:15]:
                doc_hits = supabase_db.search_chunks_hybrid(
                    workspace_id=workspace_id,
                    query_embedding=query_vec,
                    query_text=q_clean,
                    match_count=per_doc_quota,
                    doc_ids=[did],
                    category=category,
                    tags=tags,
                    min_vector_similarity=min_vector_similarity,
                )
                for h in doc_hits:
                    merged_by_chunk[str(h["chunk_id"])] = h

            global_hits = supabase_db.search_chunks_hybrid(
                workspace_id=workspace_id,
                query_embedding=query_vec,
                query_text=q_clean,
                match_count=match_count,
                doc_ids=doc_ids,
                category=category,
                tags=tags,
                min_vector_similarity=min_vector_similarity,
            )
            for h in global_hits:
                merged_by_chunk.setdefault(str(h["chunk_id"]), h)

            hits = sorted(
                merged_by_chunk.values(),
                key=lambda r: (float(r.get("rrf_score", 0.0)), float(r.get("vector_similarity", 0.0))),
                reverse=True,
            )[: max(match_count, len(doc_ids) * per_doc_quota)]
        else:
            hits = supabase_db.search_chunks_hybrid(
                workspace_id=workspace_id,
                query_embedding=query_vec,
                query_text=q_clean,
                match_count=match_count,
                doc_ids=doc_ids,
                category=category,
                tags=tags,
                min_vector_similarity=min_vector_similarity,
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
