"""Telethon listener: append raw @wizzalert messages to subscription inbox only."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from app.services.kline_chart_service import _repo_root
from app.services.symbol_pipeline_store import inbox_append

logger = logging.getLogger(__name__)


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _session_path() -> str:
    custom = _env("TG_LISTEN_SESSION_PATH")
    if custom:
        return custom
    return str(_repo_root() / "data" / "telegram_listen")


async def telegram_listener_loop(stop_event: asyncio.Event) -> None:
    api_id = _env("TG_LISTEN_API_ID")
    api_hash = _env("TG_LISTEN_API_HASH")
    channel = _env("WIZZ_ALERT_CHANNEL", "wizzalert").lower()

    if not api_id or not api_hash:
        logger.warning("telegram listener disabled: missing TG_LISTEN_API_ID or TG_LISTEN_API_HASH")
        await stop_event.wait()
        return

    try:
        from telethon import TelegramClient, events
    except ImportError:
        logger.error("telethon not installed; telegram listener disabled")
        await stop_event.wait()
        return

    client = TelegramClient(_session_path(), int(api_id), api_hash)

    @client.on(events.NewMessage(chats=channel))
    async def handler(event):  # type: ignore[no-untyped-def]
        try:
            text = (event.message.message or "").strip()
            if not text:
                return
            inbox_append(channel_username=channel, raw_text=text)
            logger.info(
                "inbox append channel=%s message_id=%s len=%d",
                channel,
                int(event.message.id),
                len(text),
            )
        except Exception:
            logger.exception("telegram handler failed")

    await client.start()
    logger.info("telegram listener started channel=%s", channel)

    try:
        while not stop_event.is_set():
            await asyncio.sleep(1.0)
    finally:
        await client.disconnect()
        logger.info("telegram listener disconnected")
