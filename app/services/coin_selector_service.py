"""
选币服务 - 从币安期货USDT永续合约中筛选综合评分最高的交易对

评分模型：三维度加权
- 24小时成交量百分位排名 × 0.4
- 24小时价格变化率绝对值百分位排名 × 0.3
- 24小时成交额百分位排名 × 0.3

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

# 评分权重（方案A：趋势强度优先）
W_QUOTE_VOLUME: float = 0.35   # 成交额 — 确保流动性
W_TREND: float = 0.30          # 趋势强度 — AI 方向判断核心
W_VOLUME: float = 0.20         # 成交量 — 交易活跃度
W_VOLATILITY: float = 0.15     # 波动率 — 辅助参考


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

    职责：
    1. 从币安获取24小时行情数据
    2. 过滤排除币种
    3. 三维度加权评分
    4. 缓存最高分币种
    5. 后台定时更新
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
        """获取缓存的选币结果（线程安全）"""
        return self._cache

    # ── 数据获取 ──────────────────────────────────────────────────────────

    async def _fetch_active_perpetual_symbols(self) -> set:
        """从 exchangeInfo 获取当前在线的 USDT 永续合约符号集合

        过滤条件：
        - contractType == "PERPETUAL"（永续合约）
        - status == "TRADING"（排除已下架币种）

        Returns:
            活跃永续合约符号集合，如 {"BTCUSDT", "ETHUSDT", ...}
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
            aiohttp.ClientError: 网络请求失败
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

    # ── 过滤逻辑 ──────────────────────────────────────────────────────────

    @staticmethod
    def _filter_symbols(
        tickers: List[Dict[str, Any]],
        active_perpetual_symbols: set,
    ) -> List[Dict[str, Any]]:
        """过滤排除币种，仅保留活跃的 USDT 永续合约

        过滤规则：
        1. 必须在 active_perpetual_symbols 白名单中（确保是在线永续合约）
        2. 不在 EXCLUDED_SYMBOLS 黑名单中

        Args:
            tickers: 原始行情数据列表
            active_perpetual_symbols: 从 exchangeInfo 获取的活跃永续合约符号集合

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
                filtered.append(ticker)
        return filtered

    # ── 评分计算 ──────────────────────────────────────────────────────────

    @staticmethod
    def _percentile_rank(values: List[float]) -> List[float]:
        """计算百分位排名（0-100）

        Args:
            values: 数值列表

        Returns:
            对应的百分位排名列表
        """
        n = len(values)
        if n <= 1:
            return [50.0] * n
        sorted_indices = sorted(range(n), key=lambda i: values[i])
        ranks: List[float] = [0.0] * n
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = (rank / (n - 1)) * 100
        return ranks

    @staticmethod
    def _calculate_scores(tickers: List[Dict[str, Any]]) -> List[CoinScore]:
        """计算所有候选币种的综合评分（趋势强度优先）

        评分公式：
        Score = 0.35 × 成交额排名 + 0.30 × 趋势强度排名 + 0.20 × 成交量排名 + 0.15 × 波动率排名

        趋势强度：priceChangePercent 绝对值越大，趋势越明确，AI 越容易判断方向

        Args:
            tickers: 过滤后的行情数据列表

        Returns:
            评分结果列表（按 score 降序排列）
        """
        if not tickers:
            return []

        # 提取有效数据，跳过异常交易对
        valid_tickers: List[Dict[str, Any]] = []
        for t in tickers:
            try:
                volume = float(t.get("volume", 0))
                quote_volume = float(t.get("quoteVolume", 0))
                price = float(t.get("lastPrice", 0))
                _ = float(t.get("priceChangePercent", 0))
                if volume > 0 and quote_volume > 0 and price > 0:
                    valid_tickers.append(t)
            except (ValueError, TypeError):
                continue

        if not valid_tickers:
            return []

        # 提取四个维度的数值
        quote_volumes: List[float] = [float(t["quoteVolume"]) for t in valid_tickers]
        trend_strengths: List[float] = [abs(float(t["priceChangePercent"])) for t in valid_tickers]
        volumes: List[float] = [float(t["volume"]) for t in valid_tickers]
        # 波动率用 (high - low) / lastPrice 近似，ticker 接口有 highPrice/lowPrice
        volatilities: List[float] = []
        for t in valid_tickers:
            high = float(t.get("highPrice", 0))
            low = float(t.get("lowPrice", 0))
            last = float(t["lastPrice"])
            volatilities.append((high - low) / last * 100 if last > 0 else 0.0)

        # 计算百分位排名
        quote_volume_ranks = CoinSelectorService._percentile_rank(quote_volumes)
        trend_ranks = CoinSelectorService._percentile_rank(trend_strengths)
        volume_ranks = CoinSelectorService._percentile_rank(volumes)
        volatility_ranks = CoinSelectorService._percentile_rank(volatilities)

        # 加权评分
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        results: List[CoinScore] = []
        for i, t in enumerate(valid_tickers):
            score = (
                W_QUOTE_VOLUME * quote_volume_ranks[i]
                + W_TREND * trend_ranks[i]
                + W_VOLUME * volume_ranks[i]
                + W_VOLATILITY * volatility_ranks[i]
            )
            results.append(CoinScore(
                symbol=t["symbol"],
                score=round(score, 2),
                price=round(float(t["lastPrice"]), 10),
                change_24h=round(float(t["priceChangePercent"]), 2),
                updated_at=now_str,
            ))

        # 按 score 降序排列
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    # ── 选币主流程 ────────────────────────────────────────────────────────

    async def refresh(self) -> CoinScore:
        """执行一次完整的选币流程

        流程：获取行情 → 过滤 → 评分 → 缓存最高分

        Returns:
            最高分币种的评分结果

        Raises:
            Exception: 无法获取有效的选币结果
        """
        async with self._lock:
            try:
                # 获取活跃永续合约白名单
                self._active_perpetual_symbols = await self._fetch_active_perpetual_symbols()

                tickers = await self._fetch_tickers()
                filtered = self._filter_symbols(tickers, self._active_perpetual_symbols)
                logger.info(f"📊 选币候选: {len(filtered)} 个交易对（已过滤 {len(tickers) - len(filtered)} 个）")

                scores = self._calculate_scores(filtered)
                if not scores:
                    raise Exception("所有候选交易对评分失败，无有效结果")

                top = scores[0]
                self._cache = top
                logger.info(f"✅ 选币完成: {top.symbol} | 评分 {top.score} | 价格 {top.price} | 24h变化 {top.change_24h}%")
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
        # 启动时立即执行一次，确保接口可用
        try:
            await self.refresh()
            logger.info("🚀 启动选币完成，缓存已就绪")
        except Exception as e:
            logger.error(f"❌ 启动选币失败: {e}")

        # 启动后台定时循环
        self._background_task = asyncio.create_task(self._schedule_loop())
        logger.info("🚀 选币后台定时任务已启动")


# ── 模块级单例 ────────────────────────────────────────────────────────────────

_service = CoinSelectorService()


async def get_coin_selector_service() -> CoinSelectorService:
    """获取选币服务单例"""
    return _service
