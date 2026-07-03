from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import logging
import os

from rate_limit import limiter

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tennis Analytics API",
    description="Backend API for tennis analytics application",
    version="1.0.0",
)

# Rate limiting (slowapi). Routers attach @limiter.limit(...) to sensitive
# endpoints; exceeding a limit returns HTTP 429.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware — explicit allowlist from env (no wildcard); empties stripped.
allowed_origins = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Internal-Secret"],
)


# Security headers on every response. This is a JSON API, so a deny-all CSP is
# safe; HTML is served separately by the Next.js frontend (see next.config.mjs).
_SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    for header, value in _SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all: log the full error server-side, return a generic message.

    Prevents stack traces, library names, SQL, or file paths from leaking to
    clients via unhandled exceptions. HTTPExceptions raised by handlers keep
    their own (already-generic) messages — only truly unhandled errors hit this.
    """
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Include routers
from api import teams, matches, videos, stats, activation, billing

app.include_router(teams.router)
app.include_router(matches.router)
app.include_router(videos.router)
app.include_router(stats.router)
app.include_router(activation.router)
app.include_router(billing.router)  # SCAFFOLD: returns 501 until Stripe is configured


@app.get("/")
async def root():
    return {"message": "Tennis Analytics API", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/health")
async def api_health():
    return {"status": "ok", "message": "API is running"}
