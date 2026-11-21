from typing import Optional, Dict, Any, Tuple
import asyncio
import base64
import hashlib
import time
import urllib.request
import urllib.error
import json
from fastapi import Header, HTTPException
from app.core.config import settings
from app.core.hardening import token_verification_cache


def _parse_valid_jwt_expiry(token: str) -> Tuple[bool, float]:
    """Ensure the bearer token carries a valid, bounded, unexpired `exp` claim. Returns (is_valid, remaining_sec)."""
    try:
        parts = token.split(".")
        if len(parts) != 3 or not parts[1]:
            return False, 0.0
        padded = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
        if not isinstance(payload, dict):
            return False, 0.0
        exp = payload.get("exp")
        if not isinstance(exp, (int, float)) or isinstance(exp, bool):
            return False, 0.0
        now = time.time()
        remaining = float(exp) - now
        max_ttl = max(300, int(getattr(settings, "max_access_token_ttl_seconds", 86400)))
        return (remaining > 0 and remaining <= max_ttl), max(0.0, remaining)
    except Exception:
        return False, 0.0


def _has_valid_jwt_expiry(token: str) -> bool:
    valid, _ = _parse_valid_jwt_expiry(token)
    return valid


def extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    """Extract raw JWT token from Authorization: Bearer <token> header."""
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header.")
    token = parts[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty bearer token.")
    return token


def verify_supabase_bearer_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify a Supabase JWT access token against Supabase GoTrue /auth/v1/user,
    backed by a short-TTL hash-keyed cache (<= 60s and never past token `exp`)
    to avoid redundant GoTrue round-trips on burst dashboard requests.
    """
    if not token or not settings.supabase_url:
        return None
    valid_exp, remaining_sec = _parse_valid_jwt_expiry(token)
    if not valid_exp:
        return None

    cache_key = "tok:" + hashlib.sha256(token.encode("utf-8")).hexdigest()
    cached_user = token_verification_cache.get(cache_key)
    if cached_user is not None:
        return dict(cached_user)

    api_key = settings.effective_supabase_service_key or settings.supabase_anon_key
    if not api_key:
        return None

    url = f"{settings.supabase_url.rstrip('/')}/auth/v1/user"
    req = urllib.request.Request(
        url,
        headers={
            "apikey": api_key,
            "Authorization": f"Bearer {token}",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            if resp.status != 200:
                return None
            user_obj = json.loads(resp.read().decode("utf-8"))
            if isinstance(user_obj, dict) and user_obj.get("id"):
                ttl = min(60.0, max(5.0, remaining_sec - 2.0))
                token_verification_cache.set(cache_key, dict(user_obj), ttl_seconds=ttl)
            return user_obj
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
        return None


async def get_optional_current_user(
    authorization: Optional[str] = Header(default=None),
) -> Optional[Dict[str, Any]]:
    """
    Validates Bearer token when provided without blocking the asyncio event loop;
    returns None when unauthenticated unless strict enforcement is required by a route.
    """
    token = extract_bearer_token(authorization)
    if not token:
        return None

    user = await asyncio.to_thread(verify_supabase_bearer_token, token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session token.")
    user["_access_token"] = token
    return user


async def require_authenticated_user(
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Strict dependency requiring a valid Supabase Bearer token."""
    user = await get_optional_current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    if settings.require_email_verification:
        is_verified = bool(user.get("email_confirmed_at") or user.get("confirmed_at"))
        if not is_verified:
            raise HTTPException(
                status_code=403,
                detail="Email verification is required before accessing workspace resources.",
            )
    return user
