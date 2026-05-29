"""Background tasks for symbol selection pipeline."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.services.futures_symbols import refresh_futures_symbols
from app.services.merger_analyzer import merger_loop
from app.services.symbol_pipeline_store import hot_board_purge_expired, pick_cooldown_purge_expired
from app.services.telegram_listener import telegram_listener_loop

logger = logging.getLogger(__name__)

_PURGE_INTERVAL_SEC = 600.0


async def _purge_loop(stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            removed = hot_board_purge_expired()
            if removed:
                logger.info("hot board purged expired=%d", removed)
            cooled = pick_cooldown_purge_expired()
            if cooled:
                logger.info("pick cooldown purged expired=%d", cooled)
        except Exception:
            logger.exception("hot board purge failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=_PURGE_INTERVAL_SEC)
        except asyncio.TimeoutError:
            pass


@asynccontextmanager
async def pipeline_lifespan(app: FastAPI) -> AsyncIterator[None]:
    stop_event = asyncio.Event()
    try:
        await refresh_futures_symbols(force=True)
    except Exception:
        logger.exception("initial futures symbol refresh failed")

    tasks = [
        asyncio.create_task(merger_loop(stop_event), name="merger_loop"),
        asyncio.create_task(_purge_loop(stop_event), name="hot_board_purge"),
        asyncio.create_task(telegram_listener_loop(stop_event), name="telegram_listener"),
    ]
    logger.info("symbol pipeline background tasks started count=%d", len(tasks))
    yield
    stop_event.set()
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("symbol pipeline background tasks stopped")
