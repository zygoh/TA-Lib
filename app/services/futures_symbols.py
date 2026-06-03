"""Binance USD-M futures symbol cache and base-asset resolution."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Dict, List, Optional, Set

import aiohttp

from app.services.crypto_mcp_service import BASE_URL, ensure_symbol_usdt

logger = logging.getLogger(__name__)

_TRADING_USDT: Set[str] = set()
_LOCK = asyncio.Lock()
_LAST_REFRESH: float = 0.0
_REFRESH_INTERVAL_SEC = 3600.0

_MULTIPLIER_PREFIXES = ("1000000", "1000")


async def refresh_futures_symbols(force: bool = False) -> Set[str]:
    global _TRADING_USDT, _LAST_REFRESH
    import time

    now = time.monotonic()
    if not force and _TRADING_USDT and (now - _LAST_REFRESH) < _REFRESH_INTERVAL_SEC:
        return _TRADING_USDT

    async with _LOCK:
        now = time.monotonic()
        if not force and _TRADING_USDT and (now - _LAST_REFRESH) < _REFRESH_INTERVAL_SEC:
            return _TRADING_USDT

        url = f"{BASE_URL}/fapi/v1/exchangeInfo"
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=True) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json()

        symbols: Set[str] = set()
        for item in data.get("symbols") or []:
            if not isinstance(item, dict):
                continue
            sym = (item.get("symbol") or "").upper()
            if not sym.endswith("USDT"):
                continue
            if (item.get("status") or "").upper() != "TRADING":
                continue
            if (item.get("quoteAsset") or "").upper() != "USDT":
                continue
            symbols.add(sym)

        _TRADING_USDT = symbols
        _LAST_REFRESH = time.monotonic()
        logger.info("U 本位合约列表缓存已刷新，数量=%d", len(symbols))
        return _TRADING_USDT


def get_trading_symbols_sync() -> Set[str]:
    return set(_TRADING_USDT)


async def list_trading_symbols() -> List[str]:
    symbols = await refresh_futures_symbols()
    return sorted(symbols)


async def is_trading_symbol(symbol: str) -> bool:
    sym = ensure_symbol_usdt(symbol)
    cache = await refresh_futures_symbols()
    return sym in cache


async def resolve_base_to_symbol(base_asset: str) -> Optional[str]:
    """Map Wizz base ticker to a TRADING USDT perpetual symbol."""
    base = (base_asset or "").strip().upper()
    if not base or not re.fullmatch(r"[A-Z0-9]{2,15}", base):
        return None

    cache = await refresh_futures_symbols()
    if not cache:
        return None

    candidates: List[str] = []
    try:
        candidates.append(ensure_symbol_usdt(base))
    except ValueError:
        pass

    for prefix in _MULTIPLIER_PREFIXES:
        candidates.append(f"{prefix}{base}USDT")

    seen: Set[str] = set()
    for cand in candidates:
        cand = cand.upper()
        if cand in seen:
            continue
        seen.add(cand)
        if cand in cache:
            return cand
    return None


async def validate_symbol_for_hot_board(symbol: str) -> str:
    sym = ensure_symbol_usdt(symbol)
    if not await is_trading_symbol(sym):
        raise ValueError(f"symbol 不是币安 U 本位 TRADING 合约: {sym}")
    return sym
