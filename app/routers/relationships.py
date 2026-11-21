from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from app.core.workspace import WorkspaceContext, get_verified_workspace_context
from app.db.supabase_client import supabase_db

router = APIRouter(prefix="/relationships", tags=["relationships"])


@router.get("")
async def get_workspace_relationship_graph(
    document_id: Optional[str] = Query(default=None, description="Optional filter for relationships touching a specific document"),
    min_confidence: float = Query(default=0.45, ge=0.0, le=1.0),
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
) -> Dict[str, Any]:
    """
    Returns the workspace Document Relationship Graph (DOC-16, ADR-013):
    - nodes: ready/partial_ready workspace documents with metadata & topics
    - edges: non-dismissed relationship edges in public.document_relationships
    """
    docs = supabase_db.list_workspace_documents(ctx.workspace_id)
    edges = supabase_db.list_workspace_relationships(ctx.workspace_id)

    # Filter by minimum confidence and optional focal document_id
    filtered_edges = [
        e
        for e in edges
        if float(e.get("composite_confidence", 0.0)) >= min_confidence
        and (
            not document_id
            or e.get("source_doc_id") == document_id
            or e.get("target_doc_id") == document_id
        )
    ]

    doc_map = {str(d["id"]): d for d in docs}
    nodes: List[Dict[str, Any]] = []
    for d in docs:
        did = str(d["id"])
        if document_id and did != document_id:
            if not any(
                e.get("source_doc_id") == did or e.get("target_doc_id") == did
                for e in filtered_edges
            ):
                continue
        nodes.append(
            {
                "id": did,
                "title": d.get("title", "Untitled Document"),
                "category": d.get("category", "general_document"),
                "primary_domain": d.get("primary_domain"),
                "status": d.get("status", "ready"),
                "main_topics": d.get("main_topics") or [],
                "page_count": int(d.get("page_count", 1)),
                "table_count": int(d.get("table_count", 0)),
                "metric_count": int(d.get("metric_count", 0)),
            }
        )

    enriched_edges: List[Dict[str, Any]] = []
    for e in filtered_edges:
        src_doc = doc_map.get(str(e.get("source_doc_id")), {})
        tgt_doc = doc_map.get(str(e.get("target_doc_id")), {})
        enriched_edges.append(
            {
                **e,
                "source_doc_title": src_doc.get("title", "Document"),
                "target_doc_title": tgt_doc.get("title", "Document"),
            }
        )

    return {
        "workspace_id": ctx.workspace_id,
        "node_count": len(nodes),
        "edge_count": len(enriched_edges),
        "nodes": nodes,
        "edges": enriched_edges,
    }


@router.post("/recompute")
async def recompute_workspace_relationships(
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
) -> Dict[str, Any]:
    """
    Recomputes pairwise topic, entity, and centroid semantic relationships
    across all documents in the workspace.
    """
    docs = supabase_db.list_workspace_documents(ctx.workspace_id)
    total_created = 0
    for doc in docs:
        edges = supabase_db.compute_and_save_relationships(ctx.workspace_id, str(doc["id"]))
        total_created += len(edges)

    all_edges = supabase_db.list_workspace_relationships(ctx.workspace_id)
    return {
        "workspace_id": ctx.workspace_id,
        "document_count": len(docs),
        "active_relationship_count": len(all_edges),
        "recomputed_pair_checks": total_created,
        "edges": all_edges,
    }


@router.patch("/{relationship_id}/dismiss")
async def dismiss_workspace_relationship(
    relationship_id: str,
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
) -> Dict[str, Any]:
    """
    Allows a user to dismiss a false-positive relationship edge (DOC-16).
    """
    updated = supabase_db.dismiss_relationship(ctx.workspace_id, relationship_id)
    if not updated:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "ERR_RELATIONSHIP_NOT_FOUND",
                "message": f"Relationship '{relationship_id}' not found in workspace.",
            },
        )
    return {
        "status": "dismissed",
        "relationship": updated,
    }
