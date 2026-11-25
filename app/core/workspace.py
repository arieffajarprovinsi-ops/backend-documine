import asyncio
from typing import Optional, Dict, Any
from pydantic import BaseModel
from fastapi import Depends, Header, HTTPException
from app.core.hardening import workspace_context_cache
from app.core.security import require_authenticated_user
from app.db.supabase_client import supabase_db

_user_ws_locks: Dict[str, asyncio.Lock] = {}


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
    2. Resolves or auto-provisions the caller's Personal Workspace off the event loop,
       coalescing burst requests via per-user async lock and caching for 120s.
    3. Returns a strongly-typed WorkspaceContext for downstream services.
    """
    user_id = str(user.get("id") or "").strip()
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user identity in token.")

    email = str(user.get("email") or "")
    meta = user.get("user_metadata") or {}
    username = str(meta.get("username") or meta.get("full_name") or (email.split("@")[0] if email else "User"))
    email_verified = bool(user.get("email_confirmed_at") or user.get("confirmed_at"))
    req_ws = x_workspace_id.strip() if x_workspace_id else ""
    cache_key = f"wsctx:{user_id}:{req_ws or 'personal'}"

    ws_info = workspace_context_cache.get(cache_key)
    if ws_info is None:
        lock = _user_ws_locks.setdefault(user_id, asyncio.Lock())
        try:
            async with lock:
                ws_info = workspace_context_cache.get(cache_key)
                if ws_info is None:
                    ws_info = await asyncio.to_thread(
                        supabase_db.ensure_personal_workspace,
                        user_id=user_id,
                        email=email,
                        username=username,
                        requested_workspace_id=req_ws or None,
                    )
                    workspace_context_cache.set(cache_key, ws_info, ttl_seconds=120.0)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        finally:
            if len(_user_ws_locks) > 512:
                _user_ws_locks.pop(user_id, None)

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
