"""
选币服务 - 两阶段筛选：流动性初筛 + K线趋势延续性确认

第一阶段：从 ticker 24hr 按成交额排名筛出 Top N 候选（确保流动性）
第二阶段：对候选币拉最近 3 根 4H K线，计算趋势延续性评分，选出趋势最明确的币

缓存策略：内存缓存 + 4小时定时更新（UTC 0:01, 4:01, 8:01, 12:01, 16:01, 20:01）
"""
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# ── 配置加载 ──────────────────────────────────────────────────────────────────

def _load_config() -> Dict[str, Any]:
    """复用项目配置加载机制"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_path = os.path.join(project_root, 'config', 'config.json')
    default_config: Dict[str, Any] = {
        "binance_api_url": "https://fapi.binance.com",
        "request_timeout": 30,
    }
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            return config
        return default_config
    except Exception as e:
        logger.error(f"❌ 加载配置文件失败: {e}，使用默认配置")
        return default_config


_config = _load_config()

# ── 常量 ──────────────────────────────────────────────────────────────────────

EXCLUDED_SYMBOLS: set = {
    # 稳定币
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "DAIUSDT",
    # 贵金属 / 外汇
    "XAGUSDT", "XAUUSDT", "EURUSDT", "GBPUSDT", "JPYUSDT",
}

UPDATE_INTERVAL_HOURS: int = 4
UPDATE_OFFSET_MINUTES: int = 1

# 第一阶段：流动性门槛（24小时成交额 >= 5000万 USDT）
MIN_QUOTE_VOLUME_USDT: float = 50_000_000.0

# 第二阶段：K线趋势确认参数
KLINE_INTERVAL: str = "2h"
KLINE_COUNT: int = 3


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class CoinScore:
    """单个币种的评分结果"""
    symbol: str
    score: float
    price: float
    change_24h: float
    updated_at: str


# ── 核心服务 ──────────────────────────────────────────────────────────────────

class CoinSelectorService:
    """选币服务核心类

    两阶段选币流程：
    1. ticker 成交额排名 → Top N 候选（流动性保障）
    2. 4H K线趋势延续性评分 → 选出趋势最明确的币
    """

    def __init__(self) -> None:
        self._cache: Optional[CoinScore] = None
        self._lock: asyncio.Lock = asyncio.Lock()
        self._session: Optional[aiohttp.ClientSession] = None
        self._background_task: Optional[asyncio.Task] = None
        self._active_perpetual_symbols: Optional[set] = None

    # ── Session 管理 ──────────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP 会话（复用连接池）"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=_config['request_timeout'])
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """关闭 HTTP 会话，释放资源"""
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        logger.info("✅ 选币服务已关闭")

    # ── 缓存读取 ──────────────────────────────────────────────────────────

    async def get_cached_result(self) -> Optional[CoinScore]:
        """获取缓存的选币结果"""
        return self._cache

    # ── 数据获取 ──────────────────────────────────────────────────────────

    async def _fetch_active_perpetual_symbols(self) -> set:
        """从 exchangeInfo 获取当前在线的 USDT 永续合约符号集合

        过滤条件：
        - contractType == "PERPETUAL"（永续合约）
        - status == "TRADING"（排除已下架币种）

        Returns:
            活跃永续合约符号集合
        """
        url = f"{_config['binance_api_url']}/fapi/v1/exchangeInfo"
        session = await self._get_session()
        async with session.get(url) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"获取 exchangeInfo 失败 {response.status}: {text}")
            data: Dict[str, Any] = await response.json()

        active_symbols: set = set()
        for symbol_info in data.get("symbols", []):
            if (
                symbol_info.get("contractType") == "PERPETUAL"
                and symbol_info.get("status") == "TRADING"
                and symbol_info.get("quoteAsset") == "USDT"
            ):
                active_symbols.add(symbol_info["symbol"])

        logger.info(f"📊 活跃永续合约: {len(active_symbols)} 个")
        return active_symbols

    async def _fetch_tickers(self) -> List[Dict[str, Any]]:
        """从币安获取所有交易对的24小时行情数据

        Returns:
            行情数据列表

        Raises:
            Exception: API 返回非 200 状态码
        """
        url = f"{_config['binance_api_url']}/fapi/v1/ticker/24hr"
        session = await self._get_session()
        async with session.get(url) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"币安 API 返回 {response.status}: {text}")
            data: List[Dict[str, Any]] = await response.json()
            return data

    async def _fetch_klines(self, symbol: str) -> List[List[Any]]:
        """获取指定交易对的最近 K 线数据

        Args:
            symbol: 交易对符号

        Returns:
            K线数据列表，每根K线格式: [open_time, open, high, low, close, volume, ...]

        Raises:
            Exception: API 请求失败
        """
        url = f"{_config['binance_api_url']}/fapi/v1/klines"
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": KLINE_INTERVAL,
            "limit": KLINE_COUNT + 1,  # 多取一根，最后一根可能未收盘
        }
        session = await self._get_session()
        async with session.get(url, params=params) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"获取 {symbol} K线失败 {response.status}: {text}")
            data: List[List[Any]] = await response.json()
            return data

    # ── 过滤逻辑 ──────────────────────────────────────────────────────────

    @staticmethod
    def _filter_symbols(
        tickers: List[Dict[str, Any]],
        active_perpetual_symbols: set,
    ) -> List[Dict[str, Any]]:
        """过滤排除币种，仅保留活跃的 USDT 永续合约

        过滤规则：
        1. 必须在 active_perpetual_symbols 白名单中（在线永续合约）
        2. 不在 EXCLUDED_SYMBOLS 黑名单中
        3. 基础数据有效（成交额 > 0）

        Args:
            tickers: 原始行情数据列表
            active_perpetual_symbols: 活跃永续合约符号集合

        Returns:
            过滤后的行情数据列表
        """
        filtered: List[Dict[str, Any]] = []
        for ticker in tickers:
            symbol: str = ticker.get("symbol", "")
            if (
                symbol in active_perpetual_symbols
                and symbol not in EXCLUDED_SYMBOLS
            ):
                try:
                    quote_volume = float(ticker.get("quoteVolume", 0))
                    if quote_volume > 0:
                        filtered.append(ticker)
                except (ValueError, TypeError):
                    continue
        return filtered

    # ── 第一阶段：流动性门槛过滤 ─────────────────────────────────────────

    @staticmethod
    def _filter_by_liquidity(
        tickers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """过滤掉24小时成交额低于门槛的交易对

        所有成交额 >= MIN_QUOTE_VOLUME_USDT 的币种均为候选，
        不做排名截断，避免 BTC/ETH 天然优势。

        Args:
            tickers: 过滤后的行情数据列表

        Returns:
            流动性合格的行情数据列表
        """
        return [
            t for t in tickers
            if float(t.get("quoteVolume", 0)) >= MIN_QUOTE_VOLUME_USDT
        ]

    # ── 第二阶段：K线趋势延续性评分 ──────────────────────────────────────

    @staticmethod
    def _calculate_trend_score(klines: List[List[Any]]) -> float:
        """基于最近已收盘的 K 线计算趋势延续性评分

        评分维度（满分 100）：
        1. 方向一致性（40分）：最近 N 根K线收盘方向是否一致（全阳/全阴）
        2. 收盘价递进（30分）：收盘价是否持续创新高/新低
        3. 实体占比（30分）：K线实体占整根K线的比例（实体大 = 趋势坚决，影线小）

        Args:
            klines: K线数据列表 [open_time, open, high, low, close, volume, ...]

        Returns:
            趋势延续性评分 0-100
        """
        if len(klines) < 2:
            return 0.0

        # 取最近 KLINE_COUNT 根已收盘的K线（排除最后一根可能未收盘的）
        # 判断最后一根是否已收盘：close_time < 当前时间
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        closed_klines: List[List[Any]] = []
        for k in klines:
            close_time = int(k[6])  # index 6 = close_time
            if close_time < now_ms:
                closed_klines.append(k)

        if len(closed_klines) < KLINE_COUNT:
            # 不够指定数量的已收盘K线，用所有已收盘的
            if len(closed_klines) < 2:
                return 0.0

        # 取最后 KLINE_COUNT 根
        recent = closed_klines[-KLINE_COUNT:]

        # 提取 open/close
        opens: List[float] = [float(k[1]) for k in recent]
        closes: List[float] = [float(k[4]) for k in recent]
        highs: List[float] = [float(k[2]) for k in recent]
        lows: List[float] = [float(k[3]) for k in recent]

        # ── 1. 方向一致性（40分）──
        directions: List[int] = []
        for o, c in zip(opens, closes):
            if c > o:
                directions.append(1)   # 阳线
            elif c < o:
                directions.append(-1)  # 阴线
            else:
                directions.append(0)   # 十字星

        # 全部同方向 = 满分，有一根不同 = 按比例扣分
        non_zero = [d for d in directions if d != 0]
        if not non_zero:
            direction_score = 0.0
        else:
            dominant = max(set(non_zero), key=non_zero.count)
            consistency = sum(1 for d in non_zero if d == dominant) / len(directions)
            direction_score = consistency * 40.0

        # ── 2. 收盘价递进（30分）──
        # 检查收盘价是否单调递增或单调递减
        increasing = all(closes[i] >= closes[i - 1] for i in range(1, len(closes)))
        decreasing = all(closes[i] <= closes[i - 1] for i in range(1, len(closes)))

        if increasing or decreasing:
            progression_score = 30.0
        else:
            # 部分递进：计算相邻K线中方向一致的比例
            consistent_pairs = 0
            for i in range(1, len(closes)):
                diff_curr = closes[i] - closes[i - 1]
                if non_zero and (
                    (non_zero[0] > 0 and diff_curr >= 0)
                    or (non_zero[0] < 0 and diff_curr <= 0)
                ):
                    consistent_pairs += 1
            progression_score = (consistent_pairs / (len(closes) - 1)) * 30.0

        # ── 3. 实体占比（30分）──
        body_ratios: List[float] = []
        for o, c, h, l in zip(opens, closes, highs, lows):
            full_range = h - l
            if full_range > 0:
                body = abs(c - o)
                body_ratios.append(body / full_range)
            else:
                body_ratios.append(0.0)

        avg_body_ratio = sum(body_ratios) / len(body_ratios) if body_ratios else 0.0
        body_score = avg_body_ratio * 30.0

        total = direction_score + progression_score + body_score
        return round(total, 2)

    # ── 选币主流程 ────────────────────────────────────────────────────────

    async def refresh(self) -> CoinScore:
        """执行两阶段选币流程

        第一阶段：ticker 成交额 Top N 初筛
        第二阶段：4H K线趋势延续性确认

        Returns:
            趋势延续性最强的币种评分结果

        Raises:
            Exception: 无法获取有效的选币结果
        """
        async with self._lock:
            try:
                # 获取活跃永续合约白名单
                self._active_perpetual_symbols = await self._fetch_active_perpetual_symbols()

                # 第一阶段：流动性门槛过滤
                tickers = await self._fetch_tickers()
                filtered = self._filter_symbols(tickers, self._active_perpetual_symbols)
                candidates = self._filter_by_liquidity(filtered)
                logger.info(
                    f"📊 过滤后候选: {len(filtered)} 个 → 流动性合格: {len(candidates)} 个"
                    f"（门槛 {MIN_QUOTE_VOLUME_USDT / 1_000_000:.0f}M USDT）"
                )

                # 第二阶段：K线趋势延续性评分
                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
                scored: List[CoinScore] = []

                for ticker in candidates:
                    symbol = ticker["symbol"]
                    try:
                        klines = await self._fetch_klines(symbol)
                        trend_score = self._calculate_trend_score(klines)
                        scored.append(CoinScore(
                            symbol=symbol,
                            score=trend_score,
                            price=round(float(ticker["lastPrice"]), 10),
                            change_24h=round(float(ticker["priceChangePercent"]), 2),
                            updated_at=now_str,
                        ))
                    except Exception as e:
                        logger.warning(f"⚠️ {symbol} K线获取失败，跳过: {e}")
                        continue

                if not scored:
                    raise Exception("所有候选交易对K线获取失败，无有效结果")

                # 选趋势延续性最强的
                scored.sort(key=lambda x: x.score, reverse=True)
                top = scored[0]
                self._cache = top
                logger.info(
                    f"✅ 选币完成: {top.symbol} | 趋势评分 {top.score} | "
                    f"价格 {top.price} | 24h变化 {top.change_24h}%"
                )
                return top

            except Exception as e:
                logger.error(f"❌ 选币流程失败: {e}")
                if self._cache is not None:
                    logger.warning(f"⚠️ 保留上一次缓存结果: {self._cache.symbol}")
                raise

    # ── 定时更新 ──────────────────────────────────────────────────────────

    @staticmethod
    def _seconds_until_next_update() -> float:
        """计算距离下一个更新时间点的秒数

        更新时间点（UTC）: 0:01, 4:01, 8:01, 12:01, 16:01, 20:01

        Returns:
            距下一个更新时间点的秒数（>= 0）
        """
        now = datetime.now(timezone.utc)
        current_hour = now.hour
        next_cycle_hour = ((current_hour // UPDATE_INTERVAL_HOURS) + 1) * UPDATE_INTERVAL_HOURS

        if next_cycle_hour >= 24:
            next_time = (
                now.replace(hour=0, minute=UPDATE_OFFSET_MINUTES, second=0, microsecond=0)
                + timedelta(days=1)
            )
        else:
            next_time = now.replace(
                hour=next_cycle_hour,
                minute=UPDATE_OFFSET_MINUTES,
                second=0,
                microsecond=0,
            )

        delta = (next_time - now).total_seconds()
        return max(delta, 0)

    async def _schedule_loop(self) -> None:
        """后台调度循环：按4小时周期定时刷新选币结果"""
        while True:
            seconds_until_next = self._seconds_until_next_update()
            logger.info(f"🚀 下次选币更新在 {seconds_until_next:.0f} 秒后")
            await asyncio.sleep(seconds_until_next)
            try:
                await self.refresh()
            except Exception as e:
                logger.error(f"❌ 定时选币失败: {e}")

    async def start_background_task(self) -> None:
        """启动后台定时更新任务

        启动时立即执行一次选币，确保缓存有数据可用。
        """
        try:
            await self.refresh()
            logger.info("🚀 启动选币完成，缓存已就绪")
        except Exception as e:
            logger.error(f"❌ 启动选币失败: {e}")

        self._background_task = asyncio.create_task(self._schedule_loop())
        logger.info("🚀 选币后台定时任务已启动")


# ── 模块级单例 ────────────────────────────────────────────────────────────────

_service = CoinSelectorService()


async def get_coin_selector_service() -> CoinSelectorService:
    """获取选币服务单例"""
    return _service
