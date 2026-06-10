"""
GeminiPage — one Playwright page, one request, then close.

Interaction pattern mirrors the reference implementation:
  • goto with networkidle
  • locate [role="textbox"], fill(), Enter
  • wait_for_timeout (fixed wait for generation)
  • read message-content nth(count-1)
"""
from __future__ import annotations

import logging

from playwright.async_api import Page, TimeoutError as PlaywrightTimeout

from config import settings

logger = logging.getLogger(__name__)


class GeminiPageError(Exception):
    """Raised when something goes wrong inside a Gemini page session."""


class GeminiPage:
    def __init__(self, page: Page) -> None:
        self._page = page

    async def navigate(self) -> None:
        """Navigate to a fresh Gemini chat and wait for the page to settle."""
        logger.debug("Navigating to %s", settings.gemini_url)
        try:
            await self._page.goto(
                settings.gemini_url,
                wait_until="networkidle",
                timeout=60_000,
            )
            logger.debug("Page ready.")
        except PlaywrightTimeout as exc:
            raise GeminiPageError(
                "Timed out loading Gemini. Is the browser logged in?"
            ) from exc

    async def send_prompt(self, prompt: str) -> str:
        """
        Fill the textbox, submit, wait for generation, return response text.
        """
        try:
            # ── Type and submit ───────────────────────────────────────────
            textbox = self._page.locator('[role="textbox"]').last
            await textbox.click()
            await textbox.fill(prompt)
            await self._page.keyboard.press("Enter")

            logger.debug("Prompt submitted (%d chars). Waiting for response…", len(prompt))

            # ── Fixed wait for generation to complete ─────────────────────
            await self._page.wait_for_timeout(settings.gemini_response_timeout_ms)

            # ── Read the last message-content element ─────────────────────
            response_locator = self._page.locator("message-content")
            count = await response_locator.count()

            if count == 0:
                logger.warning("No message-content elements found on page.")
                return ""

            text = await response_locator.nth(count - 1).inner_text()
            result = text.strip()
            logger.debug("Response received (%d chars).", len(result))
            return result

        except GeminiPageError:
            raise
        except Exception as exc:
            raise GeminiPageError(f"Error during Gemini interaction: {exc}") from exc

    async def close(self) -> None:
        try:
            await self._page.close()
            logger.debug("Page closed.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error closing page: %s", exc)