"""
app/config.py — Central configuration loaded from environment / .env
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Browser ───────────────────────────────────────────────────────────
    browser_user_data_dir: str = "./chrome_profile"
    """Path to Chrome profile that is already logged in to Gemini."""

    max_concurrent_pages: int = 10
    browser_headless: bool = False

    # ── Gemini interaction ────────────────────────────────────────────────
    gemini_url: str = "https://gemini.google.com/app"

    gemini_response_timeout_ms: int = 10_000
    """Fixed wait (ms) for Gemini to finish generating — matches reference."""

    # ── Selectors (update here if Gemini UI changes) ──────────────────────
    sel_input_box: str = '[role="textbox"]'
    sel_send_button: str = "button[aria-label='Send message']"
    sel_response_container: str = "message-content"
    sel_stop_button: str = "button[aria-label='Stop response']"

    # ── Response ──────────────────────────────────────────────────────────
    proxy_model_name: str = "gemini-web"

    # ── Tool-call sentinel ────────────────────────────────────────────────
    tool_call_sentinel: str = "TOOL_CALL_JSON:"


settings = Settings()