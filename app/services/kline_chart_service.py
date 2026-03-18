from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import aiohttp
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from zoneinfo import ZoneInfo


BASE_URL = "https://fapi.binance.com"
_SH_TZ = ZoneInfo("Asia/Shanghai")

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

BG_COLOR = "#1a1a2e"
PANEL_COLOR = "#16213e"
GRID_COLOR = "#2a2a4a"
TEXT_COLOR = "#e0e0e0"
UP_COLOR = "#26a69a"
DOWN_COLOR = "#ef5350"

WIDTH_MAP = {"2h": 1.5, "4h": 3.0}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _to_forward_slash_relative_path(path: Path, root: Path) -> str:
    relative_path = path.relative_to(root)
    return relative_path.as_posix()


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    result = np.full_like(data, np.nan, dtype=float)
    if len(data) < period:
        return result
    result[period - 1] = np.mean(data[:period])
    multiplier = 2.0 / (period + 1)
    for i in range(period, len(data)):
        result[i] = data[i] * multiplier + result[i - 1] * (1 - multiplier)
    return result


def _sma(data: np.ndarray, period: int) -> np.ndarray:
    result = np.full_like(data, np.nan, dtype=float)
    for i in range(period - 1, len(data)):
        result[i] = np.mean(data[i - period + 1 : i + 1])
    return result


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(closes)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
    return _ema(tr, period)


LOCAL_PARAMS = {
    "2h": {"atr_period": 14, "vol_ma_period": 50},
    "4h": {"atr_period": 14, "vol_ma_period": 60},
}


def compute_local_indicators(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    interval: str,
) -> Dict[str, np.ndarray]:
    params = LOCAL_PARAMS.get(interval, LOCAL_PARAMS["2h"])
    return {"atr": _atr(highs, lows, closes, params["atr_period"]), "vol_sma": _sma(volumes, params["vol_ma_period"])}


async def fetch_klines(symbol: str, interval: str, limit: int = 200) -> dict:
    url = f"{BASE_URL}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        async with session.get(url, params=params) as resp:
            if resp.status >= 400:
                return {"error": f"获取K线失败: HTTP {resp.status}"}
            data = await resp.json()

    if not isinstance(data, list):
        return {"error": f"无效的K线数据格式: {type(data)}"}

    timestamps, opens, highs, lows, closes, volumes = [], [], [], [], [], []
    for kline in data:
        if len(kline) >= 6:
            timestamps.append(kline[0])
            opens.append(float(kline[1]))
            highs.append(float(kline[2]))
            lows.append(float(kline[3]))
            closes.append(float(kline[4]))
            volumes.append(float(kline[5]))

    if len(timestamps) > 1:
        timestamps = timestamps[:-1]
        opens = opens[:-1]
        highs = highs[:-1]
        lows = lows[:-1]
        closes = closes[:-1]
        volumes = volumes[:-1]

    return {"t": timestamps, "o": opens, "h": highs, "l": lows, "c": closes, "v": volumes}


def _ensure_list(value) -> list:
    return value if isinstance(value, list) and len(value) > 0 else []


def _align_data(data_list: list, dates: list, kline_ts: list, indicator_ts: list) -> Tuple[list, list]:
    if not data_list or not indicator_ts:
        return [], []
    ts_to_idx = {ts: i for i, ts in enumerate(kline_ts)}
    aligned_dates, aligned_values = [], []
    for i, ts in enumerate(indicator_ts):
        if i < len(data_list) and ts in ts_to_idx:
            idx = ts_to_idx[ts]
            aligned_dates.append(dates[idx])
            aligned_values.append(data_list[i])
    return aligned_dates, aligned_values


def _setup_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)


def _auto_bar_width(dates: list) -> float:
    if len(dates) >= 2:
        avg_gap = (mdates.date2num(dates[-1]) - mdates.date2num(dates[0])) / max(len(dates) - 1, 1)
        return avg_gap * 0.7
    return 0.003


def _draw_candlesticks(ax: plt.Axes, dates: list, opens: list, highs: list, lows: list, closes: list, interval: str) -> None:
    width_hours = WIDTH_MAP.get(interval, 2.0)
    width_days = width_hours / 24.0
    for i, date in enumerate(dates):
        is_up = closes[i] >= opens[i]
        color = UP_COLOR if is_up else DOWN_COLOR
        ax.plot([date, date], [lows[i], highs[i]], color=color, linewidth=0.8)
        body_height = abs(closes[i] - opens[i])
        body_bottom = min(opens[i], closes[i])
        rect = Rectangle(
            (mdates.date2num(date) - width_days * 0.4, body_bottom),
            width_days * 0.8,
            body_height if body_height > 0 else (highs[i] - lows[i]) * 0.05,
            facecolor=color,
            edgecolor=color,
            linewidth=0.5,
        )
        ax.add_patch(rect)


def _draw_main_indicators(
    ax: plt.Axes,
    dates: list,
    indicators: dict,
    local_ind: Dict[str, np.ndarray],
    closes: list,
    kline_ts: list,
    indicator_ts: list,
) -> None:
    ema = _ensure_list(indicators.get("ema", []))
    if ema:
        d_ema, v_ema = _align_data(ema, dates, kline_ts, indicator_ts)
        if d_ema:
            ax.plot(d_ema, v_ema, label="EMA", color="#ffb74d", linewidth=1.2)

    sma = _ensure_list(indicators.get("sma", []))
    if sma:
        d_sma, v_sma = _align_data(sma, dates, kline_ts, indicator_ts)
        if d_sma:
            ax.plot(d_sma, v_sma, label="SMA", color="#42a5f5", linewidth=1.2)

    vwap = _ensure_list(indicators.get("vwap", []))
    if vwap:
        d_vwap, v_vwap = _align_data(vwap, dates, kline_ts, indicator_ts)
        if d_vwap:
            ax.plot(d_vwap, v_vwap, label="VWAP", color="#ce93d8", linewidth=1.2, linestyle=":")

    bb_u = _ensure_list(indicators.get("bb_u", []))
    bb_m = _ensure_list(indicators.get("bb_m", []))
    bb_l = _ensure_list(indicators.get("bb_l", []))
    if bb_u and bb_m and bb_l:
        d_bu, v_bu = _align_data(bb_u, dates, kline_ts, indicator_ts)
        d_bl, v_bl = _align_data(bb_l, dates, kline_ts, indicator_ts)
        if d_bu and d_bl:
            bu = np.array(v_bu, dtype=float)
            bl = np.array(v_bl, dtype=float)
            valid_mask = ~(np.isnan(bu) | np.isnan(bl))
            if np.any(valid_mask):
                vd = [d_bu[i] for i in range(len(d_bu)) if valid_mask[i]]
                ax.fill_between(vd, bl[valid_mask], bu[valid_mask], alpha=0.06, color="#90caf9", label="Bollinger Bands")
                ax.plot(vd, bu[valid_mask], color="#64b5f6", linewidth=0.6, alpha=0.5)
                ax.plot(vd, bl[valid_mask], color="#64b5f6", linewidth=0.6, alpha=0.5)

    atr = local_ind.get("atr")
    if atr is not None:
        atr_val = atr[-1]
        if not np.isnan(atr_val) and closes[-1] > 0:
            atr_pct = atr_val / closes[-1] * 100
            ax.text(
                0.98,
                0.95,
                f"ATR(14): {atr_val:.4f} ({atr_pct:.2f}%)",
                transform=ax.transAxes,
                fontsize=9,
                color="#ffd54f",
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, alpha=0.8),
            )


def _draw_rsi_kdj_adx(ax: plt.Axes, dates: list, indicators: dict, kline_ts: list, indicator_ts: list) -> None:
    rsi = _ensure_list(indicators.get("rsi", []))
    if rsi:
        d_rsi, v_rsi = _align_data(rsi, dates, kline_ts, indicator_ts)
        if d_rsi:
            ax.plot(d_rsi, v_rsi, label="RSI", color="#f06292", linewidth=1.5)
            ax.axhline(y=70, color=DOWN_COLOR, linestyle="--", linewidth=0.7, alpha=0.6)
            ax.axhline(y=30, color=UP_COLOR, linestyle="--", linewidth=0.7, alpha=0.6)
            ax.axhspan(70, 100, alpha=0.05, color=DOWN_COLOR)
            ax.axhspan(0, 30, alpha=0.05, color=UP_COLOR)

    k_vals = _ensure_list(indicators.get("k", []))
    d_vals = _ensure_list(indicators.get("d", []))
    if k_vals and d_vals:
        dk, vk = _align_data(k_vals, dates, kline_ts, indicator_ts)
        dd, vd = _align_data(d_vals, dates, kline_ts, indicator_ts)
        if dk:
            ax.plot(dk, vk, label="K", color="#4dd0e1", linewidth=1, alpha=0.8)
        if dd:
            ax.plot(dd, vd, label="D", color="#ffb74d", linewidth=1, alpha=0.8)

    adx = _ensure_list(indicators.get("adx", []))
    if adx:
        d_adx, v_adx = _align_data(adx, dates, kline_ts, indicator_ts)
        if d_adx:
            ax_twin = ax.twinx()
            ax_twin.plot(d_adx, v_adx, label="ADX", color="#a1887f", linewidth=1.2, linestyle="-.")
            ax_twin.set_ylabel("ADX", fontsize=9, color="#a1887f")
            ax_twin.tick_params(axis="y", labelcolor="#a1887f", labelsize=8)
            ax_twin.set_ylim(0, 80)
            ax_twin.spines["right"].set_color(GRID_COLOR)

    ax.set_ylabel("RSI / KDJ", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=7, ncol=3, facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax.grid(True, alpha=0.15, color=GRID_COLOR)


def _draw_macd(ax: plt.Axes, dates: list, indicators: dict, kline_ts: list, indicator_ts: list) -> None:
    macd = _ensure_list(indicators.get("macd", []))
    macd_s = _ensure_list(indicators.get("macd_s", []))
    if macd and macd_s:
        dm, vm = _align_data(macd, dates, kline_ts, indicator_ts)
        ds, vs = _align_data(macd_s, dates, kline_ts, indicator_ts)
        if dm:
            ax.plot(dm, vm, label="MACD", color="#42a5f5", linewidth=1.5)
        if ds:
            ax.plot(ds, vs, label="Signal", color="#ffb74d", linewidth=1.5)

    macd_h = _ensure_list(indicators.get("macd_h", []))
    if macd_h:
        dh, vh = _align_data(macd_h, dates, kline_ts, indicator_ts)
        if dh:
            bar_width = _auto_bar_width(dh)
            vh_arr = np.array(vh, dtype=float)
            pos = np.where(vh_arr >= 0, vh_arr, 0)
            neg = np.where(vh_arr < 0, vh_arr, 0)
            ax.bar(dh, pos, color=UP_COLOR, alpha=0.7, width=bar_width, edgecolor="none")
            ax.bar(dh, neg, color=DOWN_COLOR, alpha=0.7, width=bar_width, edgecolor="none")

    ax.axhline(y=0, color="#555555", linestyle="-", linewidth=0.5)
    ax.set_ylabel("MACD", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax.grid(True, alpha=0.15, color=GRID_COLOR)


def _draw_volume(
    ax: plt.Axes,
    dates: list,
    opens: list,
    closes: list,
    volumes: list,
    indicators: dict,
    local_ind: Dict[str, np.ndarray],
    kline_ts: list,
    indicator_ts: list,
) -> None:
    n = len(dates)
    bar_width = _auto_bar_width(dates)
    up_mask = [closes[i] >= opens[i] for i in range(n)]
    down_mask = [not m for m in up_mask]
    ax.bar([d for d, m in zip(dates, up_mask) if m], [v for v, m in zip(volumes, up_mask) if m], color=UP_COLOR, alpha=0.8, width=bar_width, edgecolor="none")
    ax.bar([d for d, m in zip(dates, down_mask) if m], [v for v, m in zip(volumes, down_mask) if m], color=DOWN_COLOR, alpha=0.8, width=bar_width, edgecolor="none")

    obv = _ensure_list(indicators.get("obv", []))
    if obv:
        d_obv, v_obv = _align_data(obv, dates, kline_ts, indicator_ts)
        if d_obv:
            ax_obv = ax.twinx()
            ax_obv.plot(d_obv, v_obv, color="#ce93d8", linewidth=1, alpha=0.5)
            ax_obv.set_ylabel("OBV", fontsize=8, color="#ce93d8")
            ax_obv.tick_params(axis="y", labelcolor="#ce93d8", labelsize=7)
            ax_obv.spines["right"].set_color(GRID_COLOR)

    vol_sma = local_ind.get("vol_sma")
    if vol_sma is not None:
        vs = vol_sma[-n:]
        valid = ~np.isnan(vs)
        if np.any(valid):
            vd = [dates[i] for i in range(n) if valid[i]]
            ax.plot(vd, vs[valid], color="#ffd54f", linewidth=1.2, alpha=0.8, linestyle="--")

    def format_volume(value: float, pos: int) -> str:
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if value >= 1_000:
            return f"{value / 1_000:.1f}K"
        return f"{value:.0f}"

    ax.yaxis.set_major_formatter(FuncFormatter(format_volume))
    ax.set_ylabel("Volume", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.15, color=GRID_COLOR)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9, rotation=30, colors=TEXT_COLOR)


def create_candlestick_chart(symbol: str, interval: str, klines: dict, indicators: dict) -> str:
    if "error" in klines:
        raise ValueError(f"K线数据获取失败: {klines['error']}")
    timestamps = klines.get("t", [])
    opens = klines.get("o", [])
    highs = klines.get("h", [])
    lows = klines.get("l", [])
    closes = klines.get("c", [])
    volumes = klines.get("v", [])

    if len(timestamps) < 2:
        raise ValueError(f"K线数据不足，需要至少2根，当前 {len(timestamps)} 根")
    if not all([opens, highs, lows, closes]) or len(opens) != len(timestamps):
        raise ValueError("K线数据不完整")

    dates = [datetime.fromtimestamp(ts / 1000, tz=_SH_TZ).replace(tzinfo=None) for ts in timestamps]
    indicator_ts = _ensure_list(indicators.get("t", []))

    np_opens = np.array(opens, dtype=float)
    np_highs = np.array(highs, dtype=float)
    np_lows = np.array(lows, dtype=float)
    np_closes = np.array(closes, dtype=float)
    np_volumes = np.array(volumes, dtype=float)
    local_ind = compute_local_indicators(np_highs, np_lows, np_closes, np_volumes, interval)

    fig = plt.figure(figsize=(16, 14), facecolor=BG_COLOR)
    gs = fig.add_gridspec(4, 1, height_ratios=[4, 1.5, 1.5, 1.2], hspace=0.15)

    axes = []
    for i in range(4):
        ax = fig.add_subplot(gs[i])
        _setup_axis(ax)
        axes.append(ax)

    ax1, ax2, ax3, ax4 = axes
    ax2.sharex(ax1)
    ax3.sharex(ax1)
    ax4.sharex(ax1)

    if interval == "2h":
        ax1.xaxis.set_major_locator(mdates.HourLocator(byhour=[0]))
        ax1.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 2)))
    elif interval == "4h":
        ax1.xaxis.set_major_locator(mdates.HourLocator(byhour=[0]))
        ax1.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    else:
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())

    def _smart_date_formatter(x, pos=None):
        dt = mdates.num2date(x)
        if dt.hour == 0 and dt.minute == 0:
            return dt.strftime("%m-%d")
        return dt.strftime("%H:00")

    ax1.xaxis.set_major_formatter(plt.FuncFormatter(_smart_date_formatter))

    width_hours = WIDTH_MAP.get(interval, 2.0)
    margin = timedelta(hours=width_hours * 1.2)
    ax1.set_xlim(dates[0] - margin, dates[-1] + margin)

    _draw_candlesticks(ax1, dates, opens, highs, lows, closes, interval)
    _draw_main_indicators(ax1, dates, indicators, local_ind, closes, timestamps, indicator_ts)
    ax1.set_ylabel("Price (USDT)", fontsize=11, fontweight="bold")
    ax1.set_title(f"{symbol} - {interval.upper()} Chart", fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=10)
    ax1.legend(loc="upper left", fontsize=8, ncol=2, facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax1.grid(True, alpha=0.2, color=GRID_COLOR)
    ax1.tick_params(axis="x", labelbottom=False)

    _draw_rsi_kdj_adx(ax2, dates, indicators, timestamps, indicator_ts)
    ax2.tick_params(axis="x", labelbottom=False)

    _draw_macd(ax3, dates, indicators, timestamps, indicator_ts)
    ax3.tick_params(axis="x", labelbottom=False)

    _draw_volume(ax4, dates, opens, closes, volumes, indicators, local_ind, timestamps, indicator_ts)

    ax4.tick_params(axis="x", labelsize=8, rotation=45, colors=TEXT_COLOR)
    for label in ax4.get_xticklabels():
        label.set_ha("right")

    current_price = closes[-1]
    bars_24h = {"2h": 12, "4h": 6}
    n_bars = bars_24h.get(interval, len(opens))
    ref_idx = max(0, len(opens) - n_bars)
    ref_price = opens[ref_idx]
    price_change = current_price - ref_price
    price_change_pct = (price_change / ref_price * 100) if ref_price > 0 else 0
    change_color = UP_COLOR if price_change >= 0 else DOWN_COLOR
    info_text = f"Current: ${current_price:,.2f} | 24h: ${price_change:+,.2f} ({price_change_pct:+.2f}%)"
    fig.text(0.5, 0.02, info_text, fontsize=11, ha="center", color=change_color, fontweight="bold")

    fig.text(0.99, 0.01, "@TA-Lib", fontsize=9, color="gray", alpha=0.3, ha="right", va="bottom", fontstyle="italic")

    now = datetime.now(_SH_TZ)
    date_str = now.strftime("%Y-%m-%d")
    base_symbol = symbol.replace("USDT", "").upper()
    save_dir = _repo_root() / "image" / f"{base_symbol}_{date_str}"
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{base_symbol}_{interval}.png"
    filepath = save_dir / filename

    if filepath.exists():
        filepath.unlink()

    plt.savefig(str(filepath), dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    return _to_forward_slash_relative_path(filepath, _repo_root())


class KlineChartService:
    async def generate(
        self,
        symbol: str,
        fetch_technical_data_fn: Callable[[str, str, bool], Dict[str, Any]],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {"symbol": symbol, "charts": {}, "errors": []}
        for interval in ["2h", "4h"]:
            try:
                klines = await fetch_klines(symbol, interval, limit=200)
                if "error" in klines:
                    raise ValueError(klines["error"])
                indicators_data = fetch_technical_data_fn(symbol, interval, True)
                indicators = {}
                if "error" not in indicators_data:
                    indicators = indicators_data.get("indicators", {})
                filepath = create_candlestick_chart(symbol, interval, klines, indicators)
                result["charts"][interval] = {"filepath": filepath, "status": "success"}
            except Exception as e:
                err = str(e)
                result["errors"].append({"interval": interval, "error": err})
                result["charts"][interval] = {"status": "failed", "error": err}
        return result

