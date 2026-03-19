"""
兼容 crypto-mcp 的聚合能力路由

目标：
- 由 TA-Lib 直接提供 crypto-mcp 里用到的聚合、情绪、图表与 Telegram 投递能力
"""

from __future__ import annotations

import logging
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.models.crypto_mcp_schemas import (
    CryptoBundleResponse,
    GrokUpdateRequest,
    GrokUpdateResponse,
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


@router.post("/sentiment/grok", response_model=GrokUpdateResponse)
async def update_grok(
    body: GrokUpdateRequest,
):
    """接收用户消息内容，调用 Grok 4.1 AI 获取情绪分析并写入本地文件。"""
    try:
        result: str = await grok_client.fetch_sentiment(body.content)
        grok_store.update(content=result)
        return GrokUpdateResponse(ok=True)
    except Exception as e:
        logger.error("❌ Grok 情绪分析失败: %s", e)
        return GrokUpdateResponse(ok=False, error=str(e))


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
    """
    target = ensure_symbol_usdt(symbol)
    time_data = get_shanghai_time()

    bundle_data = await get_crypto_bundle(target)

    return {
        "symbol": target,
        "time": time_data,
        "bundle": bundle_data,
    }


@router.post("/telegram", response_model=TelegramSendResponse)
async def telegram_send(body: TelegramSendRequest):
    """发送 Markdown 报告到 Telegram（需要在环境变量配置 bot token/chat id）"""
    return await send_telegram_message(body.message)

