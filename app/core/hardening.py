"""
Production Hardening Layer for DocuMind API:
- Bounded LRU+TTL in-memory cache (fast repeat reads & zero wasteful LLM recomputation)
- Sliding-window tiered rate limiter (anti-spam / anti-abuse per IP & user token)
- Strict request payload size guard (anti-DoS / memory exhaustion protection)
- HTTP security headers & opaque request ID propagation (anti-clickjacking / anti-sniffing)
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
import uuid
from collections import OrderedDict, deque
from typing import Any, Callable, Deque, Dict, Optional, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings


# ============================================================================
# 1. Thread-Safe Bounded TTL Cache (Anti-Boros & High Speed)
# ============================================================================

class TTLCache:
    """Thread-safe bounded LRU cache with per-entry TTL expiration."""

    def __init__(self, max_entries: int = 256, default_ttl_seconds: float = 120.0):
        self._max_entries = max(16, int(max_entries))
        self._default_ttl = max(1.0, float(default_ttl_seconds))
        self._store: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            expires_at, value = entry
            if expires_at <= now:
                self._store.pop(key, None)
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        ttl = self._default_ttl if ttl_seconds is None else max(0.5, float(ttl_seconds))
        expires_at = time.monotonic() + ttl
        with self._lock:
            self._store[key] = (expires_at, value)
            self._store.move_to_end(key)
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def invalidate_prefix(self, prefix: str) -> int:
        removed = 0
        with self._lock:
            keys = [k for k in self._store.keys() if k.startswith(prefix)]
            for k in keys:
                self._store.pop(k, None)
                removed += 1
        return removed

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._store),
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
            }


# Shared caches across backend services
token_verification_cache = TTLCache(max_entries=512, default_ttl_seconds=60.0)
analytics_report_cache = TTLCache(max_entries=192, default_ttl_seconds=180.0)
workspace_spend_cache = TTLCache(max_entries=256, default_ttl_seconds=30.0)


# ============================================================================
# 2. Tiered Sliding-Window Rate Limiter (Anti-Spam & Anti-Brute-Force)
# ============================================================================

class SlidingWindowRateLimiter:
    """In-memory sliding-window rate limiter with bounded memory pruning."""

    def __init__(self, max_buckets: int = 8192):
        self._max_buckets = max_buckets
        self._buckets: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()
        self._last_prune = time.monotonic()

    def check(self, key: str, limit: int, window_seconds: int = 60) -> Tuple[bool, int, int]:
        """
        Returns (allowed, remaining, retry_after_seconds).
        """
        now = time.monotonic()
        cutoff = now - window_seconds

        with self._lock:
            # Periodic cleanup to prevent unbounded RAM growth under spoofed-IP floods
            if len(self._buckets) > self._max_buckets or (now - self._last_prune) > 120.0:
                stale_keys = [
                    k for k, dq in self._buckets.items() if not dq or dq[-1] < (now - 120.0)
                ]
                for k in stale_keys:
                    self._buckets.pop(k, None)
                self._last_prune = now
                # Hard eviction if still above cap
                if len(self._buckets) > self._max_buckets:
                    overflow = len(self._buckets) - (self._max_buckets // 2)
                    for k in list(self._buckets.keys())[:overflow]:
                        self._buckets.pop(k, None)

            dq = self._buckets.get(key)
            if dq is None:
                dq = deque()
                self._buckets[key] = dq

            while dq and dq[0] <= cutoff:
                dq.popleft()

            if len(dq) >= limit:
                retry_after = max(1, int(math.ceil(window_seconds - (now - dq[0]))))
                return False, 0, retry_after

            dq.append(now)
            remaining = max(0, limit - len(dq))
            return True, remaining, 0

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()


rate_limiter = SlidingWindowRateLimiter()


def _extract_client_identity(request: Request) -> str:
    """Identify caller by IP + short hash of Bearer token (fair per-user limiting behind NAT)."""
    xff = request.headers.get("x-forwarded-for", "")
    ip = xff.split(",")[0].strip() if xff else (request.client.host if request.client else "unknown")
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer ") and len(auth_header) > 16:
        token_fp = hashlib.sha256(auth_header.encode("utf-8")).hexdigest()[:12]
        return f"u:{token_fp}"
    return f"ip:{ip}"


def _resolve_rate_tier(method: str, path: str) -> Tuple[str, int, int]:
    """Return (tier_name, max_requests, window_seconds) for a route."""
    p = path.lower()
    if p.startswith("/api/documents/upload") or p.endswith("/reprocess"):
        return ("upload", 15, 60)
    if (
        p.startswith("/api/chat")
        or p.startswith("/api/analytics/build-report")
        or p.startswith("/api/analytics/generate")
        or p.startswith("/api/synthesis")
        or p.startswith("/api/audio")
    ):
        return ("ai_heavy", 35, 60)
    if p.startswith("/api/auth"):
        return ("auth", 60, 60)
    if method.upper() in ("POST", "PUT", "PATCH", "DELETE"):
        return ("write", 90, 60)
    return ("read", 240, 60)


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        path = request.url.path
        if request.method == "OPTIONS" or path in ("/", "/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        tier, limit, window = _resolve_rate_tier(request.method, path)
        identity = _extract_client_identity(request)
        bucket_key = f"{tier}:{identity}"

        allowed, remaining, retry_after = rate_limiter.check(bucket_key, limit=limit, window_seconds=window)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Too many requests on the {tier} endpoint. "
                        f"Please try again in {retry_after} seconds."
                    ),
                    "retry_after_seconds": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


# ============================================================================
# 3. Request Payload Size Guard (Anti-Memory-Exhaustion / Anti-DoS)
# ============================================================================

class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Rejects oversized payloads before FastAPI buffers the body into RAM."""

    async def dispatch(self, request: Request, call_next: Callable):
        if request.method in ("POST", "PUT", "PATCH"):
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    size = int(content_length)
                except ValueError:
                    return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})

                is_upload = request.url.path.startswith("/api/documents/upload")
                max_bytes = (
                    (int(settings.max_upload_size_mb) + 2) * 1024 * 1024
                    if is_upload
                    else 2 * 1024 * 1024
                )
                if size > max_bytes:
                    limit_mb = max_bytes / (1024 * 1024)
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": f"Payload exceeds maximum allowed size ({limit_mb:.0f} MB)."
                        },
                    )
        return await call_next(request)


# ============================================================================
# 4. Security Headers & Request Correlation ID Middleware
# ============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        response = await call_next(request)

        response.headers["X-Request-Id"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(self), geolocation=()"
        response.headers["X-XSS-Protection"] = "0"

        if request.url.path.startswith("/api/auth"):
            response.headers["Cache-Control"] = "no-store, private"

        return response


def sanitize_header_filename(filename: str) -> str:
    """Prevent HTTP response header splitting / injection in Content-Disposition."""
    cleaned = (filename or "document.pdf").replace("\r", "").replace("\n", "").replace('"', "").replace(";", "_")
    cleaned = "".join(ch for ch in cleaned if 32 <= ord(ch) < 127).strip()
    return cleaned[:160] or "document.pdf"


# ============================================================================
# 5. Friendly Error Sanitizer (Full Detail in Server Logs, Sweet UX for Users)
# ============================================================================

_INTERNAL_LEAK_PATTERNS = (
    "openai",
    "api key",
    "api_key",
    "expired",
    "kadaluarsa",
    "invalid_api_key",
    "insufficient_quota",
    "rate_limit_exceeded",
    "error code: 401",
    "error code: 429",
    "error code: 500",
    "traceback",
    "supabase",
    "postgrest",
    "psycopg",
    "duckdb",
    "aigateway.",
    "httpstatuserror",
    "connection refused",
    "sk-",
    "bearer ",
)


def sanitize_client_error(
    raw_message: Any,
    *,
    status_code: int = 500,
    context: str = "api",
    request_id: str = "",
) -> str:
    """
    Logs raw technical / vendor error details on the backend server while ALWAYS
    returning a calm, polite, user-friendly message to the frontend.
    Never exposes OpenAI API key expiration, stack traces, SQL errors, or secrets.
    """
    raw_str = str(raw_message or "").strip()
    lower = raw_str.lower()

    has_internal_leak = any(tok in lower for tok in _INTERNAL_LEAK_PATTERNS) or len(raw_str) > 260
    if has_internal_leak or status_code >= 500:
        print(f"[ServerErrorLog context={context} status={status_code} req={request_id}] {raw_str}")

        if any(k in lower for k in ("budget", "quota", "insufficient_quota", "rate_limit", "429")):
            return "The AI assistant is currently at high capacity. Please try again in a moment."
        if any(k in lower for k in ("openai", "api key", "api_key", "invalid_api_key", "expired", "aigateway")):
            return "The AI analysis service is undergoing a brief configuration update. Your documents remain safe; please try again shortly."
        if any(k in lower for k in ("timeout", "timed out")):
            return "The request took longer than usual to process. Please resend your request."
        if status_code == 401:
            return "Your session has expired. Please sign in again to continue."
        if status_code == 403:
            return "You do not have permission to access this resource."
        if status_code == 404:
            return "The requested document or resource was not found."
        if status_code == 413:
            return "File size exceeds the allowed limit. Please upload a smaller file."
        if status_code == 429:
            return "Requests are too frequent. Please wait a few seconds and try again."
        return "A temporary issue occurred while processing your request. Please try again shortly."

    return raw_str or "A temporary issue occurred. Please try again."

