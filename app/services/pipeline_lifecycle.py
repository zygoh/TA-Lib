"""App startup: refresh futures symbol cache and image-clean deps midnight loop."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import AsyncIterator
from zoneinfo import ZoneInfo

from fastapi import FastAPI

from app.services.futures_symbols import refresh_futures_symbols
from app.services.image_clean_service import refresh_image_clean_deps_cache

logger = logging.getLogger(__name__)
_DEPS_TZ = ZoneInfo("Asia/Shanghai")


async def _image_clean_deps_midnight_loop() -> None:
    while True:
        now = datetime.now(_DEPS_TZ)
        next_midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        await asyncio.sleep(max(1.0, (next_midnight - now).total_seconds()))
        try:
            refresh_image_clean_deps_cache(force=True)
            logger.info("image clean deps refreshed at GMT+8 midnight")
        except Exception:
            logger.exception("GMT+8 0 点刷新 image clean deps 失败")


@asynccontextmanager
async def pipeline_lifespan(app: FastAPI) -> AsyncIterator[None]:
    midnight_task = asyncio.create_task(_image_clean_deps_midnight_loop())
    try:
        await refresh_futures_symbols(force=True)
    except Exception:
        logger.exception("启动时刷新合约列表失败")
    try:
        yield
    finally:
        midnight_task.cancel()
        try:
            await midnight_task
        except asyncio.CancelledError:
            pass
