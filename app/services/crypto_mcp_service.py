from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import aiohttp
import feedparser
from zoneinfo import ZoneInfo

from app.services.indicator_service import calculate_indicators_sync
from app.services.grok_store import GrokStore
from app.services.kline_chart_service import KlineChartService


_SH_TZ = ZoneInfo("Asia/Shanghai")


def ensure_symbol_usdt(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    if not s:
        raise ValueError("symbol 不能为空")
    if s.endswith("USDT"):
        return s
    if len(s) <= 10 and re.fullmatch(r"[A-Z0-9]+", s):
        # 兼容 BTC -> BTCUSDT
        return f"{s}USDT"
    raise ValueError(f"无效的交易对符号: {symbol}")


def get_shanghai_time() -> Dict[str, Any]:
    now = datetime.now(_SH_TZ)
    return {
        "full": now.strftime("%Y-%m-%d %H:%M"),
        "short": now.strftime("%m-%d %H:%M"),
        "timestamp": int(now.timestamp()),
    }


# 复刻 crypto-mcp/tools/technical.py 的配置（n8n 对齐）
CONFIG_MAP: Dict[str, Dict[str, Any]] = {
    "15m": {
        "limit": 600,
        "rsi": 14,
        "macd": [12, 26, 9],
        "bb": [20, 2.0],
        "sma": 30,
        "ema": 21,
        "adx": 14,
        "volume_ma": 30,
        "stoch_rsi": {"period": 14, "kSmooth": 3, "dSmooth": 3},
        "atr": {"period": 14},
        "obv": True,
        "vwap": {"period": "day"},
        "fib": {"period": 100},
        "patterns": True,
        "turtle": {"entryPeriod": 30, "exitPeriod": 15, "atrPeriod": 14},
    },
    "1h": {
        "limit": 500,
        "rsi": 14,
        "macd": [12, 26, 9],
        "bb": [20, 2.0],
        "sma": 50,
        "ema": 21,
        "adx": 14,
        "volume_ma": 40,
        "stoch_rsi": {"period": 14, "kSmooth": 3, "dSmooth": 3},
        "atr": {"period": 14},
        "obv": True,
        "vwap": {"period": "day"},
        "fib": {"period": 150},
        "patterns": True,
        "turtle": {"entryPeriod": 40, "exitPeriod": 20, "atrPeriod": 14},
    },
    "2h": {
        "limit": 400,
        "rsi": 14,
        "macd": [12, 26, 9],
        "bb": [20, 2.2],
        "sma": 50,
        "ema": 30,
        "adx": 14,
        "volume_ma": 50,
        "stoch_rsi": {"period": 14, "kSmooth": 5, "dSmooth": 3},
        "atr": {"period": 14},
        "obv": True,
        "vwap": {"period": "day"},
        "fib": {"period": 200},
        "patterns": True,
        "turtle": {"entryPeriod": 55, "exitPeriod": 20, "atrPeriod": 20},
    },
    "4h": {
        "limit": 350,
        "rsi": 14,
        "macd": [12, 26, 9],
        "bb": [20, 2.2],
        "sma": 50,
        "ema": 50,
        "adx": 14,
        "volume_ma": 60,
        "stoch_rsi": {"period": 14, "kSmooth": 5, "dSmooth": 3},
        "atr": {"period": 14},
        "obv": True,
        "vwap": {"period": "session", "session_hours": 24},
        "fib": {"period": 300},
        "patterns": True,
        "turtle": {"entryPeriod": 55, "exitPeriod": 20, "atrPeriod": 20},
    },
    "1d": {
        "limit": 500,
        "rsi": 14,
        "macd": [12, 26, 9],
        "bb": [20, 2.5],
        "sma": 200,
        "ema": 100,
        "adx": 14,
        "volume_ma": 100,
        "stoch_rsi": {"period": 14, "kSmooth": 5, "dSmooth": 5},
        "atr": {"period": 14},
        "obv": True,
        "fib": {"period": 400},
        "patterns": True,
        "turtle": {"entryPeriod": 55, "exitPeriod": 20, "atrPeriod": 20},
    },
}


def _trim_indicators_for_bundle(result: Dict[str, Any], interval: str, trim_count: int = 1) -> Dict[str, Any]:
    if not isinstance(result, dict) or "error" in result:
        return result
    ind = result.get("indicators")
    if not isinstance(ind, dict):
        return result

    # 找一列长度
    length = 0
    for k, v in ind.items():
        if isinstance(v, list) and v:
            length = len(v)
            break
    if length <= trim_count:
        return result

    start = length - trim_count
    trimmed: Dict[str, Any] = {}
    for k, v in ind.items():
        trimmed[k] = v[start:] if isinstance(v, list) else v
    result["indicators"] = trimmed
    return result


def _trim_indicators_for_chart(result: Dict[str, Any], trim_count: int = 200) -> Dict[str, Any]:
    if not isinstance(result, dict) or "error" in result:
        return result
    ind = result.get("indicators")
    if not isinstance(ind, dict):
        return {"error": "API返回数据格式错误：缺少indicators字段"}

    # 优先用 K 线字段长度
    length = 0
    for key in ["t", "c", "o", "h", "l", "v"]:
        v = ind.get(key)
        if isinstance(v, list) and v:
            length = len(v)
            break
    if length == 0:
        for _, v in ind.items():
            if isinstance(v, list) and v:
                length = len(v)
                break
    if length == 0:
        return {"error": "API返回数据为空：没有可用的K线数据"}

    if length <= trim_count:
        return result

    start = length - trim_count
    trimmed: Dict[str, Any] = {}
    for k, v in ind.items():
        trimmed[k] = v[start:] if isinstance(v, list) else v
    result["indicators"] = trimmed
    return result


def fetch_technical_data(symbol: str, interval: str, for_chart: bool = False) -> Dict[str, Any]:
    config = CONFIG_MAP.get(interval, CONFIG_MAP["1h"])
    result = calculate_indicators_sync(symbol, interval, config)
    if for_chart:
        return _trim_indicators_for_chart(result, trim_count=200)
    return _trim_indicators_for_bundle(result, interval, trim_count=1)


BASE_URL = "https://fapi.binance.com"


async def _get_json(session: aiohttp.ClientSession, url: str, params: Dict[str, Any] | None = None) -> Any:
    async with session.get(url, params=params) as resp:
        resp.raise_for_status()
        return await resp.json()


async def fetch_market_snapshot(symbol: str) -> Dict[str, Any]:
    connector = aiohttp.TCPConnector(limit=50, limit_per_host=20)
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=True) as session:
        price_url = f"{BASE_URL}/fapi/v2/ticker/price"
        stats_url = f"{BASE_URL}/fapi/v1/ticker/24hr"
        premium_url = f"{BASE_URL}/fapi/v1/premiumIndex"
        depth_url = f"{BASE_URL}/fapi/v1/depth"

        tasks = [
            _get_json(session, price_url, {"symbol": symbol}),
            _get_json(session, stats_url, {"symbol": symbol}),
            _get_json(session, premium_url, {"symbol": symbol}),
            _get_json(session, depth_url, {"symbol": symbol, "limit": 20}),
        ]

        price, stats, premium, depth = await asyncio.gather(*tasks, return_exceptions=True)

    def _ok(v: Any) -> Any:
        if isinstance(v, Exception):
            return {"error": str(v)}
        return v

    depth_ok = _ok(depth)
    imbalance = "平衡"
    bids_top: List[Any] = []
    asks_top: List[Any] = []
    if isinstance(depth_ok, dict) and "bids" in depth_ok and "asks" in depth_ok:
        bids_top = depth_ok.get("bids", [])[:5]
        asks_top = depth_ok.get("asks", [])[:5]
        try:
            bids_volume = sum(float(b[1]) for b in bids_top)
            asks_volume = sum(float(a[1]) for a in asks_top)
            if bids_volume > 0 and asks_volume > 0:
                ratio = bids_volume / asks_volume
                if ratio > 1.2:
                    imbalance = f"买盘偏强 (比率 {ratio:.2f})"
                elif ratio < 0.8:
                    imbalance = f"卖盘偏强 (比率 {ratio:.2f})"
        except Exception:
            pass

    return {
        "price_info": _ok(price),
        "24h_stats": _ok(stats),
        "funding_rate": _ok(premium),
        "order_book": {"bids_top": bids_top, "asks_top": asks_top, "imbalance": imbalance},
    }


RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://cryptoslate.com/feed/",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://bitcoinist.com/feed/",
    "https://www.newsbtc.com/feed/",
    "https://blockcast.it/feed/",
    "https://99bitcoins.com/feed/",
    "https://cryptobriefing.com/feed/",
    "https://crypto.news/feed/",
]

SYMBOL_TO_KEYWORD = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "SOL": "Solana",
    "BNB": "Binance",
    "XRP": "Ripple",
    "ADA": "Cardano",
    "DOGE": "Dogecoin",
    "DOT": "Polkadot",
    "MATIC": "Polygon",
    "AVAX": "Avalanche",
}


def symbol_to_keyword(symbol: str) -> str:
    base = symbol.replace("USDT", "").replace("USDC", "").replace("BUSD", "").upper()
    return SYMBOL_TO_KEYWORD.get(base, base.capitalize())


async def _fetch_bytes(session: aiohttp.ClientSession, url: str) -> bytes:
    async with session.get(url) as resp:
        resp.raise_for_status()
        return await resp.read()


async def parse_rss_feed(url: str) -> List[Dict[str, str]]:
    try:
        connector = aiohttp.TCPConnector(limit=30, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=True) as session:
            content = await _fetch_bytes(session, url)
        feed = feedparser.parse(content)
        articles: List[Dict[str, str]] = []
        for entry in feed.entries[:20]:
            title = (entry.get("title", "") or "").strip()
            link = (entry.get("link", "") or "").strip()
            description = (entry.get("description", entry.get("summary", "")) or "").strip()
            pub_date = (entry.get("published", entry.get("updated", "")) or "").strip()
            if title:
                articles.append({"title": title, "link": link, "description": description, "pubDate": pub_date})
        return articles
    except Exception:
        return []


async def fetch_news_articles(keyword: str, max_articles: int = 50) -> List[Dict[str, str]]:
    keyword_lower = keyword.lower()
    all_articles: List[Dict[str, str]] = []

    # 并发拉取 RSS
    tasks = [parse_rss_feed(u) for u in RSS_FEEDS]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, list):
            all_articles.extend(r)

    filtered: List[Dict[str, str]] = []
    for a in all_articles:
        title = (a.get("title", "") or "").lower()
        desc = (a.get("description", "") or "").lower()
        if keyword_lower in title or keyword_lower in desc:
            filtered.append(a)
        if len(filtered) >= max_articles:
            break
    return filtered


_grok_store = GrokStore()


async def get_sentiment(symbol: str) -> Dict[str, Any]:
    grok_content = _grok_store.read()
    keyword = symbol_to_keyword(symbol)
    news_articles = await fetch_news_articles(keyword, max_articles=30)

    headlines: List[str] = []
    if news_articles:
        for article in news_articles[:5]:
            title = article.get("title", "")
            link = article.get("link", "")
            if title:
                headlines.append(f"- {title} ({link})" if link else f"- {title}")
        news_summary = f"从 {len(news_articles)} 篇相关文章中提取了 {len(headlines)} 条重要头条。"
    else:
        news_summary = f"未找到 {keyword} 相关的新闻文章。"

    return {
        "grok_analysis": grok_content,
        "news_articles": news_articles,
        "news_summary": news_summary,
        "news_headlines": headlines,
        "keyword": keyword,
        "symbol": symbol,
    }


async def get_crypto_bundle(symbol: str) -> Dict[str, Any]:
    indicators: Dict[str, Any] = {}
    for interval in ["15m", "1h", "2h", "4h", "1d"]:
        indicators[interval] = fetch_technical_data(symbol, interval, for_chart=False)
    market = await fetch_market_snapshot(symbol)
    sentiment = await get_sentiment(symbol)
    return {
        "target": symbol,
        "technical_analysis": indicators,
        "market_analysis": market,
        "sentiment_analysis": sentiment,
    }


_chart_service = KlineChartService()


async def generate_kline_charts(symbol: str) -> Dict[str, Any]:
    return await _chart_service.generate(symbol, fetch_technical_data_fn=fetch_technical_data)


update_grok_sentiment_file = _grok_store

