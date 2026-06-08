"""App startup: refresh futures symbol cache."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.services.futures_symbols import refresh_futures_symbols

logger = logging.getLogger(__name__)


@asynccontextmanager
async def pipeline_lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        await refresh_futures_symbols(force=True)
    except Exception:
        logger.exception("启动时刷新合约列表失败")
    yield
