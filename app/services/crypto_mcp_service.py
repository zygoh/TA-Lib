from __future__ import annotations

import asyncio
import calendar
import re
import time
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import urlparse

import aiohttp
import feedparser
from zoneinfo import ZoneInfo

from app.services.indicator_service import calculate_indicators_sync
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


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _strip_quote_asset(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    for quote in ("USDT", "USDC", "BUSD", "FDUSD", "USD"):
        if s.endswith(quote):
            return s[: -len(quote)]
    return s


def _strip_contract_multiplier(base_asset: str) -> str:
    base = (base_asset or "").strip().upper()
    for prefix in ("1000000", "1000"):
        if base.startswith(prefix) and len(base) > len(prefix):
            return base[len(prefix) :]
    return base


def _base_asset_from_symbol(symbol: str, strip_multiplier: bool = False) -> str:
    base = _strip_quote_asset(symbol)
    return _strip_contract_multiplier(base) if strip_multiplier else base


async def fetch_top_gainers(
    limit: int = 20,
    min_quote_volume: float = 5_000_000,
    include_1000: bool = True,
) -> Dict[str, Any]:
    limit = max(1, min(int(limit), 100))
    min_quote_volume = max(0.0, float(min_quote_volume))

    connector = aiohttp.TCPConnector(limit=20, limit_per_host=10)
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=True) as session:
        data = await _get_json(session, f"{BASE_URL}/fapi/v1/ticker/24hr")

    if not isinstance(data, list):
        return {
            "source": "binance_futures_24hr",
            "min_quote_volume": min_quote_volume,
            "include_1000": include_1000,
            "gainers": [],
        }

    gainers: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        symbol = (item.get("symbol") or "").upper()
        if not symbol.endswith("USDT"):
            continue

        base_asset = _base_asset_from_symbol(symbol)
        normalized_base_asset = _strip_contract_multiplier(base_asset)
        if not include_1000 and base_asset != normalized_base_asset:
            continue

        quote_volume = _to_float(item.get("quoteVolume"))
        if quote_volume < min_quote_volume:
            continue

        gainers.append(
            {
                "symbol": symbol,
                "base_asset": base_asset,
                "normalized_base_asset": normalized_base_asset,
                "priceChangePercent": _to_float(item.get("priceChangePercent")),
                "lastPrice": _to_float(item.get("lastPrice")),
                "quoteVolume": quote_volume,
                "volume": _to_float(item.get("volume")),
            }
        )

    gainers.sort(key=lambda x: x["priceChangePercent"], reverse=True)
    for idx, item in enumerate(gainers[:limit], start=1):
        item["rank"] = idx

    return {
        "source": "binance_futures_24hr",
        "min_quote_volume": min_quote_volume,
        "include_1000": include_1000,
        "gainers": gainers[:limit],
    }


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
    "https://decrypt.co/feed/",
    "https://blockworks.com/feed/",
    "https://www.blocktempo.com/feed/",
    "https://techflowpost.substack.com/feed",
    "https://thedefiant.io/api/feed",
    "https://www.panewslab.com/rss.xml?lang=en&type=NEWS",
    "https://www.panewslab.com/rss.xml?lang=zh&type=NEWS",
    "https://rss.odaily.news/rss/newsflash",
    "https://rss.odaily.news/rss/post",
]

SHARPE_NEWS_URL = "https://www.sharpe.ai/api/news/feed"

SYMBOL_TO_KEYWORDS = {
    "BTC": ["Bitcoin", "BTC"],
    "ETH": ["Ethereum", "Ether", "ETH"],
    "SOL": ["Solana", "SOL"],
    "BNB": ["BNB Chain", "Binance Coin", "BNB"],
    "XRP": ["XRP", "Ripple"],
    "ADA": ["Cardano", "ADA"],
    "DOGE": ["Dogecoin", "DOGE"],
    "DOT": ["Polkadot", "DOT"],
    "POL": ["Polygon", "POL"],
    "MATIC": ["Polygon", "MATIC"],
    "AVAX": ["Avalanche", "AVAX"],
    "TRX": ["TRON", "TRX"],
    "LINK": ["Chainlink", "LINK"],
    "LTC": ["Litecoin", "LTC"],
    "BCH": ["Bitcoin Cash", "BCH"],
    "UNI": ["Uniswap", "UNI"],
    "AAVE": ["Aave", "AAVE"],
    "ATOM": ["Cosmos", "ATOM"],
    "FIL": ["Filecoin", "FIL"],
    "APT": ["Aptos", "APT"],
    "ARB": ["Arbitrum", "ARB"],
    "OP": ["Optimism"],
    "NEAR": ["NEAR Protocol"],
    "SUI": ["Sui Network", "Sui"],
    "SEI": ["Sei Network", "SEI"],
    "TIA": ["Celestia", "TIA"],
    "INJ": ["Injective", "INJ"],
    "JUP": ["Jupiter", "JUP"],
    "WLD": ["Worldcoin", "WLD"],
    "TAO": ["Bittensor", "TAO"],
    "FET": ["Fetch.ai", "Artificial Superintelligence Alliance", "FET"],
    "ENA": ["Ethena", "ENA"],
    "PENDLE": ["Pendle", "PENDLE"],
    "PEPE": ["Pepe", "PEPE"],
    "WIF": ["dogwifhat", "WIF"],
    "BONK": ["Bonk", "BONK"],
    "FLOKI": ["Floki", "FLOKI"],
    "SHIB": ["Shiba Inu", "SHIB"],
    "ORDI": ["Ordinals", "ORDI"],
    "SATS": ["SATS", "1000SATS"],
}


def symbol_to_keywords(symbol: str) -> List[str]:
    base = _base_asset_from_symbol(symbol, strip_multiplier=True)
    keywords = SYMBOL_TO_KEYWORDS.get(base, [base])
    seen = set()
    result: List[str] = []
    for keyword in keywords:
        clean = (keyword or "").strip()
        key = clean.lower()
        if clean and key not in seen:
            result.append(clean)
            seen.add(key)
    return result


def symbol_to_keyword(symbol: str) -> str:
    return symbol_to_keywords(symbol)[0]


def _source_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host or url


def _entry_timestamp(entry: Any) -> int:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        try:
            return int(calendar.timegm(parsed))
        except Exception:
            pass

    value = entry.get("published") or entry.get("updated") or ""
    if not value:
        return 0
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return 0


def _keyword_matches(text: str, keywords: List[str]) -> List[str]:
    matched: List[str] = []
    lowered = text.lower()
    for keyword in keywords:
        clean = keyword.strip()
        if not clean:
            continue
        escaped = re.escape(clean.lower())
        if re.fullmatch(r"[a-z0-9]+", clean.lower()):
            pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
            hit = re.search(pattern, lowered) is not None
        else:
            hit = re.search(escaped, lowered) is not None
        if hit:
            matched.append(clean)
    return matched


def _dedupe_and_sort_articles(articles: List[Dict[str, Any]], max_articles: int) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for article in articles:
        key = (article.get("link") or article.get("title") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(article)

    deduped.sort(key=lambda x: int(x.get("published_ts") or 0), reverse=True)
    return deduped[:max_articles]


async def _fetch_bytes(session: aiohttp.ClientSession, url: str) -> bytes:
    async with session.get(url) as resp:
        resp.raise_for_status()
        return await resp.read()


async def parse_rss_feed(url: str) -> List[Dict[str, Any]]:
    try:
        connector = aiohttp.TCPConnector(limit=30, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=True) as session:
            content = await _fetch_bytes(session, url)
        feed = feedparser.parse(content)
        source = (feed.feed.get("title", "") or "").strip() if getattr(feed, "feed", None) else ""
        if not source:
            source = _source_from_url(url)

        articles: List[Dict[str, Any]] = []
        for entry in feed.entries[:20]:
            title = (entry.get("title", "") or "").strip()
            link = (entry.get("link", "") or "").strip()
            description = (entry.get("description", entry.get("summary", "")) or "").strip()
            pub_date = (entry.get("published", entry.get("updated", "")) or "").strip()
            if title:
                articles.append(
                    {
                        "title": title,
                        "link": link,
                        "description": description,
                        "pubDate": pub_date,
                        "published_ts": _entry_timestamp(entry),
                        "source": source,
                        "source_feed": url,
                    }
                )
        return articles
    except Exception:
        return []


async def fetch_sharpe_news_articles(symbol: str, keywords: List[str], max_articles: int = 20) -> List[Dict[str, Any]]:
    base = _base_asset_from_symbol(symbol, strip_multiplier=True)
    connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
    timeout = aiohttp.ClientTimeout(total=15)
    try:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=True) as session:
            data = await _get_json(
                session,
                SHARPE_NEWS_URL,
                {"limit": max(1, min(max_articles, 100)), "category": "crypto", "coin": base},
            )
    except Exception:
        return []

    raw_articles = []
    if isinstance(data, dict):
        raw_articles = data.get("articles") or data.get("data", {}).get("articles") or []
    if not isinstance(raw_articles, list):
        return []

    articles: List[Dict[str, Any]] = []
    for item in raw_articles:
        if not isinstance(item, dict):
            continue
        title = (item.get("title", "") or "").strip()
        link = (item.get("link", "") or "").strip()
        description = (item.get("summary", "") or "").strip()
        published = (item.get("published", "") or "").strip()
        text = f"{title}\n{description}\n{' '.join(str(t) for t in item.get('coin_tags', []) if t)}"
        matched = _keyword_matches(text, keywords)
        if not matched and base.lower() not in {str(t).lower() for t in item.get("coin_tags", []) if t}:
            continue
        if title:
            articles.append(
                {
                    "title": title,
                    "link": link,
                    "description": description,
                    "pubDate": published,
                    "published_ts": _entry_timestamp({"published": published}),
                    "source": (item.get("source", "") or "Sharpe News").strip(),
                    "source_feed": SHARPE_NEWS_URL,
                    "matched_keywords": matched or [base],
                    "coin_tags": item.get("coin_tags", []),
                }
            )
    return articles


async def fetch_news_articles(symbol: str, max_articles: int = 50) -> List[Dict[str, Any]]:
    keywords = symbol_to_keywords(symbol)
    all_articles: List[Dict[str, Any]] = []

    # 并发拉取免费 RSS 与 Sharpe 公共新闻接口。
    tasks = [parse_rss_feed(u) for u in RSS_FEEDS]
    tasks.append(fetch_sharpe_news_articles(symbol, keywords, max_articles=20))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, list):
            all_articles.extend(r)

    filtered: List[Dict[str, Any]] = []
    for a in all_articles:
        text = f"{a.get('title', '')}\n{a.get('description', '')}"
        matched = a.get("matched_keywords") or _keyword_matches(text, keywords)
        if matched:
            a["matched_keywords"] = matched
            filtered.append(a)
    return _dedupe_and_sort_articles(filtered, max_articles=max_articles)


async def get_sentiment(symbol: str) -> Dict[str, Any]:
    keywords = symbol_to_keywords(symbol)
    keyword = keywords[0]
    news_articles = await fetch_news_articles(symbol, max_articles=30)

    headlines: List[str] = []
    if news_articles:
        for article in news_articles[:5]:
            title = article.get("title", "")
            link = article.get("link", "")
            source = article.get("source", "")
            if title:
                prefix = f"{source}: " if source else ""
                headlines.append(f"- {prefix}{title} ({link})" if link else f"- {prefix}{title}")
        news_summary = f"从 {len(news_articles)} 篇相关文章中提取了 {len(headlines)} 条重要头条。"
    else:
        news_summary = f"未找到 {keyword} 相关的免费新闻文章。"

    return {
        "grok_analysis": "",
        "news_articles": news_articles,
        "news_summary": news_summary,
        "news_headlines": headlines,
        "keyword": keyword,
        "keywords": keywords,
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

