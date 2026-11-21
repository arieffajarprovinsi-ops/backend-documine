from fastapi import APIRouter, Depends
from app.core.workspace import WorkspaceContext, get_verified_workspace_context
from app.db.supabase_client import supabase_db
from app.ai.gateway import ai_gateway

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/session")
async def verify_session(ctx: WorkspaceContext = Depends(get_verified_workspace_context)):
    """Verify the caller's Supabase Bearer token and return sanitized user + active workspace info."""
    return {
        "authenticated": True,
        "user": {
            "id": ctx.user_id,
            "email": ctx.email,
            "username": ctx.username,
            "email_verified": ctx.email_verified,
        },
        "workspace": {
            "id": ctx.workspace_id,
            "name": ctx.workspace_name,
            "role": ctx.role,
        },
    }


@router.get("/ai-usage")
async def get_workspace_ai_usage(ctx: WorkspaceContext = Depends(get_verified_workspace_context)):
    """Return AI token and cost telemetry for the caller's workspace (PR-043)."""
    logs = supabase_db.list_ai_usage(ctx.workspace_id, limit=100)
    total_cost_usd = round(sum(float(item.get("estimated_cost_usd") or 0.0) for item in logs), 6)
    total_input_tokens = sum(int(item.get("input_tokens") or 0) for item in logs)
    total_cached_tokens = sum(int(item.get("cached_tokens") or 0) for item in logs)
    total_output_tokens = sum(int(item.get("output_tokens") or 0) for item in logs)
    return {
        "workspace_id": ctx.workspace_id,
        "openai_configured": ai_gateway.is_openai_configured,
        "summary": {
            "total_calls": len(logs),
            "total_input_tokens": total_input_tokens,
            "total_cached_tokens": total_cached_tokens,
            "total_output_tokens": total_output_tokens,
            "total_estimated_cost_usd": total_cost_usd,
        },
        "recent_logs": logs[:25],
    }


@router.delete("/account")
async def delete_account(ctx: WorkspaceContext = Depends(get_verified_workspace_context)):
    """
    Self-service account & workspace cascade deletion (PR-004, AC-004).
    Deletes all storage objects, workspace rows, and the user's Supabase Auth account.
    """
    result = supabase_db.delete_user_account_and_workspaces(
        user_id=ctx.user_id,
        workspace_id=ctx.workspace_id,
    )
    return {
        "status": "account_deleted",
        **result,
    }
