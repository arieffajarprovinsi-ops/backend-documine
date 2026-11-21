from typing import Optional, List
from fastapi import APIRouter, Depends, Query
from app.core.workspace import WorkspaceContext, get_verified_workspace_context
from app.models.schemas import HybridSearchRequest
from app.services.retrieval_service import retrieval_service

router = APIRouter(prefix="/search", tags=["search"])


@router.post("")
async def execute_hybrid_search(
    req: HybridSearchRequest,
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    """
    Execute Hybrid Reciprocal Rank Fusion (RRF) search across the caller's workspace (DOC-12, PR-025).
    Combines pgvector 1536-d cosine similarity with PostgreSQL 'simple' GIN Full-Text Search.
    """
    outcome = await retrieval_service.search_hybrid(
        workspace_id=ctx.workspace_id,
        user_id=ctx.user_id,
        query=req.query,
        doc_ids=req.doc_ids,
        category=req.category,
        tags=req.tags,
        match_count=req.match_count,
        min_vector_similarity=req.min_vector_similarity,
    )
    return {
        "workspace_id": ctx.workspace_id,
        "query": req.query,
        "retrieval_status": outcome["retrieval_status"],
        "total_matches": len(outcome["results"]),
        "result_count": len(outcome["results"]),
        "results": outcome["results"],
    }


@router.get("")
async def execute_hybrid_search_get(
    q: str = Query(..., description="Search query string"),
    doc_id: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    doc_ids: Optional[List[str]] = [doc_id] if doc_id else None
    outcome = await retrieval_service.search_hybrid(
        workspace_id=ctx.workspace_id,
        user_id=ctx.user_id,
        query=q,
        doc_ids=doc_ids,
        category=category,
        match_count=limit,
    )
    return {
        "workspace_id": ctx.workspace_id,
        "query": q,
        "retrieval_status": outcome["retrieval_status"],
        "total_matches": len(outcome["results"]),
        "results": outcome["results"],
    }
