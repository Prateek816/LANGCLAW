"""
app/browser/manager.py
──────────────────────
BrowserManager — owns the single persistent Chrome context and dishes
out pages on demand, matching the reference implementation's pattern:

  launch_persistent_context(channel="chrome", headless=False/True)
  new_page() per request → use → close()

Concurrency is capped by asyncio.Semaphore.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from playwright.async_api import BrowserContext, Playwright, async_playwright

from app.browser.gemini_page import GeminiPage
from app.config import settings

logger = logging.getLogger(__name__)


class BrowserManager:
    def __init__(self) -> None:
        self._playwright: Optional[Playwright] = None
        self._context: Optional[BrowserContext] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._active_pages: int = 0

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        logger.info("Launching Chrome…")
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_pages)
        self._playwright = await async_playwright().start()

        launch_kwargs: dict = {
            "headless": settings.browser_headless,
            "args": ["--no-sandbox", "--disable-dev-shm-usage"],
        }

        # Use channel="chrome" (installed Chrome) when a user-data dir is
        # provided, matching the reference implementation exactly.
        if settings.browser_user_data_dir:
            launch_kwargs["channel"] = "chrome"
            self._context = (
                await self._playwright.chromium.launch_persistent_context(
                    user_data_dir=settings.browser_user_data_dir,
                    **launch_kwargs,
                )
            )
            logger.info(
                "Persistent Chrome context ready (profile: %s, headless: %s).",
                settings.browser_user_data_dir,
                settings.browser_headless,
            )
        else:
            # Fallback: Playwright's bundled Chromium, ephemeral context
            logger.warning(
                "BROWSER_USER_DATA_DIR not set — using ephemeral Chromium context. "
                "Set headless=false and log in to Gemini on first launch."
            )
            browser = await self._playwright.chromium.launch(**launch_kwargs)
            self._context = await browser.new_context()

    async def stop(self) -> None:
        logger.info("Closing browser context…")
        try:
            if self._context:
                await self._context.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error during shutdown: %s", exc)

    # ── Page acquisition ───────────────────────────────────────────────────

    @asynccontextmanager
    async def acquire_page(self) -> AsyncIterator[GeminiPage]:
        """
        Async context manager:
          1. Wait for a free slot (semaphore).
          2. Open a new Chrome tab.
          3. Navigate to Gemini.
          4. Yield GeminiPage to caller.
          5. Close tab + release slot on exit (even on error).
        """
        if self._semaphore is None or self._context is None:
            raise RuntimeError("BrowserManager not started.")

        async with self._semaphore:
            self._active_pages += 1
            logger.debug(
                "Opening page (active=%d / max=%d).",
                self._active_pages,
                settings.max_concurrent_pages,
            )
            raw_page = await self._context.new_page()
            gemini_page = GeminiPage(raw_page)
            try:
                await gemini_page.navigate()
                yield gemini_page
            finally:
                await gemini_page.close()
                self._active_pages -= 1
                logger.debug("Closed page (active=%d).", self._active_pages)

    # ── Diagnostics ────────────────────────────────────────────────────────

    @property
    def active_pages(self) -> int:
        return self._active_pages

    @property
    def is_running(self) -> bool:
        return self._context is not None