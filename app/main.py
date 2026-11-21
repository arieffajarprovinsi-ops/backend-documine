from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.core.config import settings
from app.core.hardening import (
    MaxBodySizeMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    analytics_report_cache,
    sanitize_client_error,
    token_verification_cache,
)
from app.db.supabase_client import supabase_db
from app.ai.gateway import ai_gateway
from app.routers import documents, chat, analytics, synthesis, audio, auth, search, relationships

app = FastAPI(
    title=settings.app_name,
    version="2.1.0",
    description="Backend API for DocuMind: Three-Pillar AI Document Intelligence Workspace",
)

# 1. Response Compression (60-80% smaller payloads for JSON & analytics charts)
app.add_middleware(GZipMiddleware, minimum_size=1024)

# 2. Anti-Hack & Anti-Spam Hardening Stack
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(MaxBodySizeMiddleware)
app.add_middleware(RateLimitMiddleware)

# 3. Hardened CORS (Never wildcard '*' when allow_credentials=True)
safe_origins = [o for o in settings.cors_origins if o and o != "*"] or [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://documine-tzlh.vercel.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=safe_origins,
    allow_origin_regex=r"^https://documine(-[a-z0-9-]+)?\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
        "Origin",
        "X-Workspace-Id",
        "X-Request-Id",
    ],
    expose_headers=["X-Request-Id", "X-RateLimit-Limit", "X-RateLimit-Remaining", "Retry-After"],
)


@app.exception_handler(StarletteHTTPException)
async def sanitized_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Sanitize any HTTPException detail so internal API/key/DB messages never reach users."""
    req_id = getattr(request.state, "request_id", "unknown")
    friendly = sanitize_client_error(
        exc.detail,
        status_code=exc.status_code,
        context=request.url.path,
        request_id=req_id,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": friendly, "request_id": req_id},
        headers=getattr(exc, "headers", None) or {},
    )


@app.exception_handler(Exception)
async def sanitized_unhandled_exception_handler(request: Request, exc: Exception):
    """Never leak internal stack traces, SQL errors, or secrets to clients."""
    req_id = getattr(request.state, "request_id", "unknown")
    friendly = sanitize_client_error(
        f"{type(exc).__name__}: {exc}",
        status_code=500,
        context=request.url.path,
        request_id=req_id,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": friendly,
            "request_id": req_id,
        },
    )


# Mount API Routers
app.include_router(auth.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(relationships.router, prefix="/api")
app.include_router(synthesis.router, prefix="/api")
app.include_router(audio.router, prefix="/api")


@app.get("/")
async def root():
    return {
        "app": settings.app_name,
        "version": "2.1.0",
        "status": "online",
        "ai_provider": settings.ai_default_provider,
        "docs_url": "/docs",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "supabase_connected": supabase_db.is_connected,
        "openai_configured": ai_gateway.is_openai_configured,
        "ai_default_provider": settings.ai_default_provider,
        "hardening": {
            "rate_limiting": True,
            "gzip": True,
            "security_headers": True,
            "token_cache": token_verification_cache.stats(),
            "analytics_cache": analytics_report_cache.stats(),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
