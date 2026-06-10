"""
app/routes/chat.py
──────────────────
OpenAI-compatible POST /v1/chat/completions endpoint.

Request flow
────────────
1. Validate the incoming OpenAI request.
2. Build a plain-text Gemini prompt from messages + tools.
3. Acquire a fresh Gemini browser tab via BrowserManager.
4. Send the prompt; collect the raw text response.
5. Parse the response into an OpenAI-compatible JSON body.
6. Return the response; the page is closed automatically.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.browser.gemini_page import GeminiPageError
from app.config import settings
from app.parser.response import parse_gemini_response
from app.prompt.builder import build_prompt
from app.schemas.openai_input import ChatCompletionRequest

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
) -> JSONResponse:
    """
    OpenAI-compatible Chat Completions endpoint.

    Accepts the standard OpenAI request schema and returns a response
    in the OpenAI format.  The underlying model is Gemini Web.
    """
    browser_manager = request.app.state.browser_manager

    if not browser_manager.is_running:
        raise HTTPException(
            status_code=503,
            detail="Browser manager is not running. The server may still be starting up.",
        )

    # ── 1. Build the Gemini prompt ─────────────────────────────────────────
    try:
        prompt = build_prompt(body)
    except Exception as exc:
        logger.exception("Failed to build prompt: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to build prompt: {exc}",
        ) from exc

    logger.info(
        "Incoming request: model=%s, messages=%d, tools=%d, stream=%s",
        body.model,
        len(body.messages),
        len(body.tools) if body.tools else 0,
        body.stream,
    )
    logger.debug("Built prompt (%d chars).", len(prompt))

    # ── 2. Send prompt to Gemini and collect response ──────────────────────
    try:
        async with browser_manager.acquire_page() as gemini_page:
            raw_text = await gemini_page.send_prompt(prompt)
    except GeminiPageError as exc:
        logger.error("Gemini page error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Gemini browser error: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during Gemini interaction: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {exc}",
        ) from exc

    logger.debug("Raw Gemini response (%d chars).", len(raw_text))

    # ── 3. Parse and return ────────────────────────────────────────────────
    response = parse_gemini_response(
        raw_text=raw_text,
        model=settings.proxy_model_name,
    )

    # Note: 'stream' is accepted in the schema for compatibility but
    # this proxy always returns a buffered (non-streaming) response.
    if body.stream:
        logger.warning(
            "Client requested streaming (stream=true) but this proxy "
            "always returns buffered responses."
        )

    return JSONResponse(content=response.model_dump(exclude_none=True))