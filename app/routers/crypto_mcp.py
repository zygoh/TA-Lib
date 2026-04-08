"""
兼容 crypto-mcp 的聚合能力路由

目标：
- 由 TA-Lib 直接提供 crypto-mcp 里用到的聚合、情绪、图表与 Telegram 投递能力
"""

from __future__ import annotations

import logging
import asyncio
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.models.crypto_mcp_schemas import (
    CryptoBundleResponse,
    GrokUpdateResponse,
    GrokStatusResponse,
    SentimentResponse,
    TimeResponse,
    ChartsResponse,
    TelegramSendRequest,
    TelegramSendResponse,
    CryptoMcpAllResponse,
)
from app.services.crypto_mcp_service import (
    ensure_symbol_usdt,
    get_shanghai_time,
    get_crypto_bundle,
    get_sentiment,
    generate_kline_charts,
    send_telegram_message,
)
from app.services.grok_api_client import GrokApiClient
from app.services.grok_store import GrokStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/crypto-mcp", tags=["crypto-mcp"])

grok_client: GrokApiClient = GrokApiClient()
grok_store: GrokStore = GrokStore()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_grok_base_symbol(symbol: str) -> str:
    """路径参数可为 BTC / ETH 或 BTCUSDT / ETHUSDT。"""
    s = (symbol or "").strip().upper()
    if s.endswith("USDT"):
        s = s[:-4]
    if s in ("BTC", "ETH"):
        return s
    raise HTTPException(
        status_code=400,
        detail="SYMBOL 仅支持 BTC 或 ETH（可带 USDT 后缀，如 BTCUSDT）",
    )


# Grok 任务状态：内存中维护，进程级别共享
_grok_task_status: dict[str, str | int | None] = {
    "status": "idle",       # idle / running / completed / failed
    "error": None,
    "updated_at": 0,
}

_grok_schedule: dict[str, tuple[int, int]] = {
    "ETH": (7, 55),   # 北京时间 07:55
    "BTC": (19, 55),  # 北京时间 19:55
}
_grok_scheduler_tasks: dict[str, asyncio.Task[None]] = {}
_shanghai_tz = ZoneInfo("Asia/Shanghai")


async def _run_grok_task(content: str) -> None:
    """后台执行 Grok AI 请求并更新状态。"""
    _grok_task_status["status"] = "running"
    _grok_task_status["error"] = None
    try:
        result: str = await grok_client.fetch_sentiment(content)
        grok_store.update(content=result)
        _grok_task_status["status"] = "completed"
        _grok_task_status["updated_at"] = int(time.time())
        logger.info("✅ Grok 后台任务完成，内容长度: %d", len(result))
    except Exception as e:
        _grok_task_status["status"] = "failed"
        _grok_task_status["error"] = str(e)
        logger.error("❌ Grok 后台任务失败: %s", e)


def _load_grok_prompt_content(base_symbol: str) -> str:
    """读取并校验 data/{SYMBOL}_grok_prompt.md。"""
    prompt_path = _repo_root() / "data" / f"{base_symbol}_grok_prompt.md"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"未找到提示文件: {prompt_path.name}")
    content = prompt_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"提示文件内容为空: {prompt_path.name}")
    return content


async def _run_grok_task_for_symbol(base_symbol: str) -> bool:
    """
    按 SYMBOL 读取提示词并执行 Grok 分析。
    返回 True 表示执行完成，False 表示因已有任务在跑而跳过。
    """
    content = _load_grok_prompt_content(base_symbol)
    if _grok_task_status["status"] == "running":
        logger.warning("⏭️ Grok 定时任务跳过：已有任务执行中 symbol=%s", base_symbol)
        return False
    await _run_grok_task(content)
    return True


def _seconds_until_next_run(hour: int, minute: int) -> tuple[float, str]:
    """计算距离下一次指定北京时间触发点的秒数与展示时间。"""
    now = datetime.now(_shanghai_tz)
    next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if next_run <= now:
        next_run = next_run + timedelta(days=1)
    return (next_run - now).total_seconds(), next_run.isoformat()


async def _scheduled_grok_runner(base_symbol: str, hour: int, minute: int) -> None:
    """循环执行 SYMBOL 的每日定时任务。"""
    logger.info(
        "🕒 启动 Grok 定时任务 symbol=%s, schedule=%02d:%02d Asia/Shanghai",
        base_symbol,
        hour,
        minute,
    )
    while True:
        sleep_seconds, next_run_str = _seconds_until_next_run(hour, minute)
        logger.info("⏱️ Grok 下次执行 symbol=%s at %s", base_symbol, next_run_str)
        await asyncio.sleep(sleep_seconds)
        try:
            await _run_grok_task_for_symbol(base_symbol)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("❌ Grok 定时任务执行异常 symbol=%s: %s", base_symbol, e)


def start_grok_scheduler() -> None:
    """启动 ETH/BTC 的 Grok 定时任务（幂等）。"""
    for base_symbol, (hour, minute) in _grok_schedule.items():
        existing = _grok_scheduler_tasks.get(base_symbol)
        if existing and not existing.done():
            continue
        _grok_scheduler_tasks[base_symbol] = asyncio.create_task(
            _scheduled_grok_runner(base_symbol, hour, minute)
        )


async def stop_grok_scheduler() -> None:
    """停止所有 Grok 定时任务。"""
    tasks = list(_grok_scheduler_tasks.values())
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _grok_scheduler_tasks.clear()


@router.get("/time", response_model=TimeResponse)
async def get_time():
    """北京时间（UTC+8）时间"""
    return get_shanghai_time()


@router.get("/bundle", response_model=CryptoBundleResponse)
async def bundle(symbol: str):
    """技术 + 市场 + 情绪 聚合数据"""
    target = ensure_symbol_usdt(symbol)
    return await get_crypto_bundle(target)


@router.get("/sentiment", response_model=SentimentResponse)
async def sentiment(symbol: str):
    """情绪聚合（本地 Grok 文件 + RSS 新闻）"""
    target = ensure_symbol_usdt(symbol)
    return await get_sentiment(target)


@router.get("/sentiment/grok/status", response_model=GrokStatusResponse)
async def grok_status():
    """查询 Grok AI 情绪分析任务的执行状态。"""
    return GrokStatusResponse(
        status=_grok_task_status["status"],  # type: ignore[arg-type]
        error=_grok_task_status["error"],  # type: ignore[arg-type]
        updated_at=_grok_task_status["updated_at"],  # type: ignore[arg-type]
    )


@router.get("/sentiment/grok/{symbol}", response_model=GrokUpdateResponse)
async def update_grok(symbol: str):
    """
    兼容接口：当前已改为定时任务模式，不再手动触发。
    - ETH: 每日 07:55（北京时间）
    - BTC: 每日 19:55（北京时间）
    """
    base = _normalize_grok_base_symbol(symbol)
    try:
        _load_grok_prompt_content(base)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return GrokUpdateResponse(ok=True)


@router.get("/charts", response_model=ChartsResponse)
async def charts(symbol: str):
    """生成 2h/4h K 线图，返回保存路径"""
    target = ensure_symbol_usdt(symbol)
    return await generate_kline_charts(target)


@router.get("/charts/image")
async def charts_image(symbol: str, interval: str = "2h"):
    """
    直接返回生成好的 K 线图 PNG（二进制流）。
    当前规则：
    - 只支持 interval=2h / 4h
    - 使用北京时间当天的图片文件
    """
    target = ensure_symbol_usdt(symbol)
    interval = interval.lower()
    if interval not in ("2h", "4h"):
        raise HTTPException(status_code=400, detail="interval 仅支持 2h 或 4h")

    from app.services.kline_chart_service import _repo_root

    tz = ZoneInfo("Asia/Shanghai")
    now = datetime.now(tz)
    date_str = now.strftime("%Y-%m-%d")
    base_symbol = target.replace("USDT", "").upper()

    root = _repo_root()
    filepath = root / "image" / f"{base_symbol}_{date_str}" / f"{base_symbol}_{interval}.png"

    if not filepath.is_file():
        raise HTTPException(status_code=404, detail=f"未找到图片文件: {filepath}")

    return FileResponse(str(filepath), media_type="image/png", filename=filepath.name)


@router.get("/all", response_model=CryptoMcpAllResponse)
async def all_in_one(symbol: str):
    """
    一次性汇总：
    - time
    - bundle
    - 后台触发图表生成（不阻塞、不返回）
    """
    t0 = time.perf_counter()
    target = ensure_symbol_usdt(symbol)
    logger.info("all_in_one 开始 symbol=%s -> %s", symbol, target)

    t_charts = time.perf_counter()
    await generate_kline_charts(target)
    logger.info(
        "all_in_one K 线图生成完成 symbol=%s 耗时 %.2fs",
        target,
        time.perf_counter() - t_charts,
    )

    time_data = get_shanghai_time()

    t_bundle = time.perf_counter()
    bundle_data = await get_crypto_bundle(target)
    logger.info(
        "all_in_one bundle 聚合完成 symbol=%s 耗时 %.2fs",
        target,
        time.perf_counter() - t_bundle,
    )

    logger.info(
        "all_in_one 结束 symbol=%s 总耗时 %.2fs",
        target,
        time.perf_counter() - t0,
    )
    return {
        "symbol": target,
        "time": time_data,
        "bundle": bundle_data,
    }


@router.post("/telegram", response_model=TelegramSendResponse)
async def telegram_send(body: TelegramSendRequest):
    """发送 Markdown 报告到 Telegram（需要在环境变量配置 bot token/chat id）"""
    return await send_telegram_message(body.message)

