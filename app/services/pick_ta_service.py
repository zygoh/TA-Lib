"""Batch TA summaries for hot-board-pick (direction-neutral scoring)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PICK_INTERVALS = ("1h", "2h", "4h")
_PICK_CONCURRENCY = 8


def _last_value(indicators: Dict[str, Any], key: str) -> Optional[float]:
    raw = indicators.get(key)
    if not isinstance(raw, list) or not raw:
        return None
    val = raw[-1]
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _interval_pick_metrics(interval_data: Dict[str, Any]) -> Dict[str, Any]:
    if interval_data.get("error"):
        return {"ok": False, "error": str(interval_data.get("error"))}
    indicators = interval_data.get("indicators")
    if not isinstance(indicators, dict) or not indicators:
        return {"ok": False, "error": "no indicators"}
    bb_u = _last_value(indicators, "bb_u")
    bb_l = _last_value(indicators, "bb_l")
    bb_m = _last_value(indicators, "bb_m")
    bb_width_pct: Optional[float] = None
    if bb_u is not None and bb_l is not None and bb_m and bb_m > 0:
        bb_width_pct = round((bb_u - bb_l) / bb_m * 100.0, 4)
    return {
        "ok": True,
        "adx": _last_value(indicators, "adx"),
        "atr": _last_value(indicators, "atr"),
        "rsi": _last_value(indicators, "rsi"),
        "bb_width_pct": bb_width_pct,
        "v": _last_value(indicators, "v"),
        "v_ma": _last_value(indicators, "v_ma"),
        "obv": _last_value(indicators, "obv"),
        "macd_h": _last_value(indicators, "macd_h"),
    }


def build_pick_ta_summary(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Slim TA + market fields for Agent pick-scoring (no RSS)."""
    ta = bundle.get("technical_analysis") or {}
    intervals: Dict[str, Any] = {}
    any_ok = False
    for iv in _PICK_INTERVALS:
        metrics = _interval_pick_metrics(ta.get(iv) or {})
        intervals[iv] = metrics
        if metrics.get("ok"):
            any_ok = True

    market = bundle.get("market_analysis") or {}
    stats = market.get("24h_stats") or {}
    pcp = stats.get("priceChangePercent")
    try:
        abs_24h_pct = abs(float(pcp)) if pcp is not None else None
    except (TypeError, ValueError):
        abs_24h_pct = None

    funding = market.get("funding_rate") or {}
    fr = funding.get("lastFundingRate")
    try:
        funding_rate = float(fr) if fr is not None else None
    except (TypeError, ValueError):
        funding_rate = None

    return {
        "ta_available": any_ok,
        "intervals": intervals,
        "market": {
            "priceChangePercent": pcp,
            "abs_priceChangePercent": abs_24h_pct,
            "lastPrice": stats.get("lastPrice"),
            "quoteVolume": stats.get("quoteVolume"),
            "funding_rate": funding_rate,
        },
    }


async def get_crypto_bundle_for_pick(symbol: str) -> Dict[str, Any]:
    """TA + market only (no RSS), fewer intervals than full bundle."""
    from app.services.crypto_mcp_service import fetch_market_snapshot, fetch_technical_data

    def _technical() -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for interval in _PICK_INTERVALS:
            out[interval] = fetch_technical_data(symbol, interval, for_chart=False)
        return out

    technical = await asyncio.to_thread(_technical)
    market = await fetch_market_snapshot(symbol)
    return {
        "target": symbol,
        "technical_analysis": technical,
        "market_analysis": market,
    }


async def fetch_pick_ta_map(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Concurrent pick TA summaries keyed by symbol."""
    sem = asyncio.Semaphore(_PICK_CONCURRENCY)

    async def _one(sym: str) -> Tuple[str, Dict[str, Any]]:
        async with sem:
            try:
                bundle = await get_crypto_bundle_for_pick(sym)
                return sym, build_pick_ta_summary(bundle)
            except Exception as exc:
                logger.warning("pick_ta failed symbol=%s err=%s", sym, exc)
                return sym, {"ta_available": False, "error": str(exc)}

    pairs = await asyncio.gather(*[_one(s) for s in symbols])
    return dict(pairs)
