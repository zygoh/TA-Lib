"""Volatility Opportunity Score (VOS) scanner: direction-agnostic hot board writes."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
import pandas as pd
import talib

from app.services.crypto_mcp_service import BASE_URL
from app.services.futures_symbols import get_trading_symbols_sync, refresh_futures_symbols
from app.services.symbol_pipeline_store import hot_board_upsert

logger = logging.getLogger(__name__)

_SCAN_INTERVAL_SEC = 900.0
_TOP_N = 8
_COARSE_ABS_PCT = 12.0
_COARSE_POOL = 12
_KLINE_LIMIT = 100


# ---------------------------------------------------------------------------
# Stage 1: coarse filter (ticker only, no K-line)
# ---------------------------------------------------------------------------

async def _fetch_all_tickers() -> List[Dict[str, Any]]:
    """Fetch all USDT perpetual 24h tickers from Binance."""
    connector = aiohttp.TCPConnector(limit=20, limit_per_host=10)
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=True) as session:
        async with session.get(f"{BASE_URL}/fapi/v1/ticker/24hr") as resp:
            resp.raise_for_status()
            data = await resp.json()
    if not isinstance(data, list):
        return []
    results: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        symbol = (item.get("symbol") or "").upper()
        if not symbol.endswith("USDT"):
            continue
        try:
            pct = float(item.get("priceChangePercent") or 0)
            vol = float(item.get("quoteVolume") or 0)
            price = float(item.get("lastPrice") or 0)
            volume = float(item.get("volume") or 0)
        except (TypeError, ValueError):
            continue
        results.append({
            "symbol": symbol,
            "priceChangePercent": pct,
            "quoteVolume": vol,
            "lastPrice": price,
            "volume": volume,
        })
    return results


def _coarse_filter(
    tickers: List[Dict[str, Any]],
    trading_symbols: Set[str],
) -> List[Dict[str, Any]]:
    """粗筛：TRADING 合约且 |24h%| >= _COARSE_ABS_PCT，按 |24h%| 取前 _COARSE_POOL 只。"""
    candidates: List[Dict[str, Any]] = []
    for t in tickers:
        if t["symbol"] not in trading_symbols:
            continue
        if abs(t["priceChangePercent"]) < _COARSE_ABS_PCT:
            continue
        candidates.append(t)
    candidates.sort(key=lambda x: abs(x["priceChangePercent"]), reverse=True)
    return candidates[:_COARSE_POOL]


# ---------------------------------------------------------------------------
# Stage 2: VOS scoring (1h K-line + indicators)
# ---------------------------------------------------------------------------

async def _fetch_klines_1h(session: aiohttp.ClientSession, symbol: str) -> Optional[pd.DataFrame]:
    """Fetch 1h K-lines for a single symbol. Returns None on failure."""
    url = f"{BASE_URL}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": "1h", "limit": _KLINE_LIMIT}
    try:
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ])
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)
        if len(df) < 30:
            return None
        return df
    except Exception as e:
        logger.debug("kline fetch failed symbol=%s: %s", symbol, e)
        return None


def _score_dimension(value: float, low: float, high: float) -> float:
    """Map a raw value to 0-100 score with linear interpolation, clamped."""
    if high <= low:
        return 50.0
    return max(0.0, min(100.0, (value - low) / (high - low) * 100.0))


def _compute_vos(df: pd.DataFrame, ticker: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Compute Volatility Opportunity Score for one symbol. Returns None on failure."""
    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    volume = df["volume"].astype("float64")

    # ATR breakout: current ATR vs its own 20-bar moving average
    atr_arr = talib.ATR(high.values, low.values, close.values, timeperiod=14)
    atr_series = pd.Series(atr_arr)
    atr_current = atr_series.iloc[-1]
    atr_ma = atr_series.rolling(20).mean().iloc[-1]
    if pd.isna(atr_current) or pd.isna(atr_ma) or atr_ma <= 0:
        return None
    atr_ratio = atr_current / atr_ma

    # ADX trend strength
    adx_arr = talib.ADX(high.values, low.values, close.values, timeperiod=14)
    adx_val = float(adx_arr[-1]) if not pd.isna(adx_arr[-1]) else 0.0

    # Bollinger Band width expansion
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close.values, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
    if pd.isna(bb_upper[-1]) or pd.isna(bb_middle[-1]) or bb_middle[-1] <= 0:
        return None
    bb_width_current = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
    bb_width_series = pd.Series((bb_upper - bb_lower) / bb_middle)
    bb_width_ma = bb_width_series.rolling(20).mean().iloc[-1]
    bb_width_ratio = bb_width_current / bb_width_ma if (not pd.isna(bb_width_ma) and bb_width_ma > 0) else 1.0

    # Volume surge: current bar volume vs volume MA
    vol_ma_arr = talib.SMA(volume.values, timeperiod=40)
    vol_current = volume.iloc[-1]
    vol_ma = float(vol_ma_arr[-1]) if not pd.isna(vol_ma_arr[-1]) else 0.0
    volume_ratio = vol_current / vol_ma if vol_ma > 0 else 1.0

    # RSI extremity: distance from 50
    rsi_arr = talib.RSI(close.values, timeperiod=14)
    rsi_val = float(rsi_arr[-1]) if not pd.isna(rsi_arr[-1]) else 50.0
    rsi_extremity = abs(rsi_val - 50.0)

    # |24h%|
    abs_pct = abs(ticker["priceChangePercent"])

    # --- Dimension scores (0-100 each) ---
    s_atr = _score_dimension(atr_ratio, 0.8, 3.0)          # 0.8x = normal, 3x = extreme
    s_adx = _score_dimension(adx_val, 15.0, 55.0)          # 15 = no trend, 55 = very strong
    s_bb = _score_dimension(bb_width_ratio, 0.8, 3.0)      # same scale as ATR
    s_vol = _score_dimension(volume_ratio, 0.8, 5.0)       # 5x volume = extremely unusual
    s_pct = _score_dimension(abs_pct, 3.0, 40.0)           # 3% = mild, 40% = extreme
    s_rsi = _score_dimension(rsi_extremity, 5.0, 40.0)     # 5 = neutral, 40 = extreme

    # Weighted VOS
    vos = (
        s_atr * 0.25
        + s_adx * 0.20
        + s_bb * 0.15
        + s_vol * 0.20
        + s_pct * 0.10
        + s_rsi * 0.10
    )

    # Direction hint (informational only, does NOT affect ranking)
    if ticker["priceChangePercent"] > 2:
        hint = "bullish_momentum"
    elif ticker["priceChangePercent"] < -2:
        hint = "bearish_momentum"
    else:
        hint = "neutral"

    return {
        "vos_score": round(vos, 1),
        "scores": {
            "atr_breakout": round(s_atr, 1),
            "adx_strength": round(s_adx, 1),
            "bb_expansion": round(s_bb, 1),
            "volume_surge": round(s_vol, 1),
            "abs_24h_pct": round(s_pct, 1),
            "rsi_extremity": round(s_rsi, 1),
        },
        "priceChangePercent": round(ticker["priceChangePercent"], 2),
        "quoteVolume": round(ticker["quoteVolume"], 0),
        "atr_ratio": round(atr_ratio, 2),
        "adx": round(adx_val, 1),
        "bb_width_ratio": round(bb_width_ratio, 2),
        "volume_ratio": round(volume_ratio, 2),
        "rsi": round(rsi_val, 1),
        "direction_hint": hint,
    }


async def _fine_rank(candidates: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Fetch 1h K-lines for candidates, compute VOS, return top-N (ticker, vos_data) pairs."""
    connector = aiohttp.TCPConnector(limit=15, limit_per_host=10)
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=True) as session:
        tasks = [_fetch_klines_1h(session, c["symbol"]) for c in candidates]
        kline_results = await asyncio.gather(*tasks, return_exceptions=True)

    scored: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for ticker, kline_result in zip(candidates, kline_results):
        if isinstance(kline_result, Exception) or kline_result is None:
            continue
        vos_data = _compute_vos(kline_result, ticker)
        if vos_data is None:
            continue
        scored.append((ticker, vos_data))
        logger.debug(
            "VOS %s score=%.1f atr=%.2f adx=%.1f bb=%.2f vol=%.2f rsi=%.1f",
            ticker["symbol"], vos_data["vos_score"],
            vos_data["atr_ratio"], vos_data["adx"],
            vos_data["bb_width_ratio"], vos_data["volume_ratio"],
            vos_data["rsi"],
        )

    scored.sort(key=lambda x: x[1]["vos_score"], reverse=True)
    return scored[:_TOP_N]


# ---------------------------------------------------------------------------
# Public API (unchanged signatures)
# ---------------------------------------------------------------------------

async def run_merger_scan_once() -> int:
    await refresh_futures_symbols()
    trading = get_trading_symbols_sync()
    if not trading:
        logger.warning("merger scan skipped: empty TRADING symbol cache")
        return 0

    tickers = await _fetch_all_tickers()
    candidates = _coarse_filter(tickers, trading)
    if not candidates:
        logger.info("merger scan: no coarse candidates (tickers=%d)", len(tickers))
        return 0

    ranked = await _fine_rank(candidates)

    written = 0
    for ticker, vos_data in ranked:
        symbol = ticker["symbol"]
        hot_board_upsert({
            "symbol": symbol,
            "base_asset": symbol.replace("USDT", ""),
            "source": "merger_analyzer",
            "merger": {**vos_data, "rule": "vos_scan"},
        })
        written += 1

    logger.info(
        "merger scan done written=%d coarse=%d scored=%d",
        written,
        len(candidates),
        len(ranked),
    )
    return written


async def merger_loop(stop_event: asyncio.Event) -> None:
    logger.info(
        "merger loop started interval_sec=%.0f top_n=%d coarse_abs_pct=%.0f coarse_pool=%d",
        _SCAN_INTERVAL_SEC,
        _TOP_N,
        _COARSE_ABS_PCT,
        _COARSE_POOL,
    )
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
