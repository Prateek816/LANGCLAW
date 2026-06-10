"""
FastAPI application factory with lifespan management.

Startup
───────
  • Launch the Playwright browser via BrowserManager.
  • Attach the manager to app.state so route handlers can reach it.

Shutdown
────────
  • Gracefully close all browser pages and the browser process.

Routes
──────
  • POST /v1/chat/completions   — OpenAI-compatible chat completions
  • GET  /health                — liveness probe
  • GET  /v1/models             — OpenAI-compatible model list
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from browser.manager import BrowserManager
from config import settings
from routes.chat import router as chat_router

# ── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logging.getLogger("watchfiles").setLevel(logging.WARNING)  # ← add this
logger = logging.getLogger(__name__)

# ── Lifespan ───────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage startup and shutdown of the browser."""
    browser_manager = BrowserManager()
    app.state.browser_manager = browser_manager
    app.state.start_time = time.time()

    try:
        await browser_manager.start()
        logger.info("Server is ready to accept requests.")
        yield
    finally:
        logger.info("Shutting down…")
        await browser_manager.stop()


# ── Application factory ────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(
        title="Gemini OpenAI Proxy",
        description=(
            "An OpenAI-compatible API proxy that uses Gemini Web "
            "via browser automation as the backend model."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS (permissive — tighten in production) ──────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ─────────────────────────────────────────────────────────────
    app.include_router(chat_router)

    # ── Health probe ───────────────────────────────────────────────────────
    @app.get("/health", include_in_schema=False)
    async def health() -> JSONResponse:
        bm: BrowserManager = app.state.browser_manager
        uptime = round(time.time() - app.state.start_time, 1)
        return JSONResponse(
            {
                "status": "ok",
                "uptime_seconds": uptime,
                "browser_running": bm.is_running,
                "active_pages": bm.active_pages,
                "max_concurrent_pages": settings.max_concurrent_pages,
            }
        )

    # ── OpenAI-compatible model list ───────────────────────────────────────
    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": settings.proxy_model_name,
                        "object": "model",
                        "created": 1_700_000_000,
                        "owned_by": "gemini-web-proxy",
                    }
                ],
            }
        )

    return app


# ── Module-level app instance (used by uvicorn) ────────────────────────────
app = create_app()


# ── Direct execution ───────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )