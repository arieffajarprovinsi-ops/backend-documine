from typing import Optional, Dict, Any
from pydantic import BaseModel
from fastapi import Depends, Header, HTTPException
from app.core.security import require_authenticated_user
from app.db.supabase_client import supabase_db


class WorkspaceContext(BaseModel):
    """
    Verified multi-tenant identity and workspace context attached to every request (ADR-003).
    Never trust a raw workspace_id from request query/body without membership verification.
    """

    user_id: str
    email: str = ""
    username: str = "User"
    email_verified: bool = True
    workspace_id: str
    workspace_name: str = "Personal Workspace"
    role: str = "owner"
    access_token: Optional[str] = None


async def get_verified_workspace_context(
    user: Dict[str, Any] = Depends(require_authenticated_user),
    x_workspace_id: Optional[str] = Header(default=None, alias="X-Workspace-Id"),
) -> WorkspaceContext:
    """
    FastAPI dependency that:
    1. Requires a valid Supabase JWT (require_authenticated_user).
    2. Resolves or auto-provisions the caller's Personal Workspace (or verifies membership
       if X-Workspace-Id is explicitly requested).
    3. Returns a strongly-typed WorkspaceContext for downstream services.
    """
    user_id = str(user.get("id") or "").strip()
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user identity in token.")

    email = str(user.get("email") or "")
    meta = user.get("user_metadata") or {}
    username = str(meta.get("username") or meta.get("full_name") or (email.split("@")[0] if email else "User"))
    email_verified = bool(user.get("email_confirmed_at") or user.get("confirmed_at"))

    try:
        ws_info = supabase_db.ensure_personal_workspace(
            user_id=user_id,
            email=email,
            username=username,
            requested_workspace_id=x_workspace_id.strip() if x_workspace_id else None,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    return WorkspaceContext(
        user_id=user_id,
        email=email,
        username=username,
        email_verified=email_verified,
        workspace_id=ws_info["workspace_id"],
        workspace_name=ws_info["workspace_name"],
        role=ws_info["role"],
        access_token=user.get("_access_token"),
    )
