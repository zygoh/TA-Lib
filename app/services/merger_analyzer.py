"""24h merger scanner: rule-based hot board writes."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from app.services.crypto_mcp_service import fetch_top_gainers
from app.services.futures_symbols import refresh_futures_symbols
from app.services.symbol_pipeline_store import hot_board_upsert

logger = logging.getLogger(__name__)

_SCAN_INTERVAL_SEC = 900.0
_MIN_QUOTE_VOLUME = 5_000_000.0
_MIN_ABS_24H_PCT = 8.0
_TOP_N = 15


async def _fetch_loser_candidates() -> List[Dict[str, Any]]:
    gainers_payload = await fetch_top_gainers(
        limit=500,
        min_quote_volume=_MIN_QUOTE_VOLUME,
        include_1000=True,
    )
    items = list(gainers_payload.get("gainers") or [])
    items.sort(key=lambda x: x.get("priceChangePercent", 0.0))
    losers = items[:_TOP_N]
    for idx, item in enumerate(losers, start=1):
        item["rank"] = idx
        item["loser_rank"] = idx
    return losers


def _item_qualifies(item: Dict[str, Any]) -> bool:
    pct = abs(float(item.get("priceChangePercent") or 0.0))
    vol = float(item.get("quoteVolume") or 0.0)
    return vol >= _MIN_QUOTE_VOLUME and pct >= _MIN_ABS_24H_PCT


async def run_merger_scan_once() -> int:
    await refresh_futures_symbols()
    gainers = (await fetch_top_gainers(
        limit=_TOP_N,
        min_quote_volume=_MIN_QUOTE_VOLUME,
        include_1000=True,
    )).get("gainers") or []
    losers = await _fetch_loser_candidates()

    seen: Dict[str, Dict[str, Any]] = {}
    for item in gainers[:_TOP_N] + losers[:_TOP_N]:
        sym = (item.get("symbol") or "").upper()
        if sym:
            seen[sym] = item
    broad = (await fetch_top_gainers(
        limit=100,
        min_quote_volume=_MIN_QUOTE_VOLUME,
        include_1000=True,
    )).get("gainers") or []
    for item in broad:
        if _item_qualifies(item):
            sym = (item.get("symbol") or "").upper()
            if sym:
                seen[sym] = item

    written = 0
    for symbol, item in seen.items():
        base = item.get("base_asset") or symbol.replace("USDT", "")
        merger = {
            "priceChangePercent": item.get("priceChangePercent"),
            "quoteVolume": item.get("quoteVolume"),
            "rank": item.get("rank"),
            "rule": "merger_24h_scan",
        }
        hot_board_upsert(
            {
                "symbol": symbol,
                "base_asset": base,
                "source": "merger_analyzer",
                "merger": merger,
            }
        )
        written += 1
    logger.info("merger scan done written=%d candidates=%d", written, len(seen))
    return written


async def merger_loop(stop_event: asyncio.Event) -> None:
    logger.info("merger loop started interval_sec=%.0f", _SCAN_INTERVAL_SEC)
    while not stop_event.is_set():
        try:
            await run_merger_scan_once()
        except Exception:
            logger.exception("merger scan failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=_SCAN_INTERVAL_SEC)
        except asyncio.TimeoutError:
            pass
    logger.info("merger loop stopped")
