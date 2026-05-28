"""
兼容 crypto-mcp 的聚合能力路由

目标：
- 由 TA-Lib 直接提供 crypto-mcp 里用到的聚合、情绪、图表与统一分发能力
"""

from __future__ import annotations

import mimetypes
import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.models.crypto_mcp_schemas import (
    CryptoBundleResponse,
    SentimentResponse,
    TimeResponse,
    ChartsResponse,
    CryptoMcpAllResponse,
    DistributeResponse,
    GainersResponse,
    InboxConsumeRequest,
    InboxConsumeResponse,
    InboxSeedRequest,
    InboxSeedResponse,
    InboxItem,
    HotBoardUpsertRequest,
    HotBoardUpsertResponse,
    HotBoardEntry,
    PickerSnapshotResponse,
    FuturesSymbolsResponse,
)
from app.services.futures_symbols import list_trading_symbols, validate_symbol_for_hot_board
from app.services.symbol_pipeline_store import (
    inbox_append,
    inbox_consume,
    inbox_delete,
    hot_board_upsert,
    hot_board_list_active,
)
from app.services.crypto_mcp_service import (
    ensure_symbol_usdt,
    get_shanghai_time,
    get_crypto_bundle,
    get_sentiment,
    fetch_top_gainers,
    generate_kline_charts,
)
from app.services.distribution_service import distribute_post

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/crypto-mcp", tags=["crypto-mcp"])


def _looks_like_image_upload(image: UploadFile) -> bool:
    content_type = (image.content_type or "").lower()
    if content_type.startswith("image/"):
        return True
    guessed, _ = mimetypes.guess_type(image.filename or "")
    return bool(guessed and guessed.startswith("image/"))


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
    """情绪聚合（RSS 新闻）"""
    target = ensure_symbol_usdt(symbol)
    return await get_sentiment(target)


@router.get("/news", response_model=SentimentResponse)
async def news(symbol: str):
    """指定币种免费新闻聚合（等价于 sentiment，便于按涨幅榜选币后直查）"""
    target = ensure_symbol_usdt(symbol)
    return await get_sentiment(target)


@router.post("/subscription-inbox/seed", response_model=InboxSeedResponse)
async def subscription_inbox_seed(body: InboxSeedRequest):
    """联调测试：写入 raw 收件箱（模拟 Telethon，不经过解析）。"""
    channel = (body.channel or "wizzalert").lower()
    ids: list[str] = []
    for item in body.items:
        if not (item.raw_text or "").strip():
            continue
        mid = item.message_id if item.message_id is not None else int(datetime.now().timestamp())
        row = inbox_append(
            channel_username=channel,
            message_id=mid,
            permalink=f"https://t.me/{channel}/{mid}",
            raw_text=item.raw_text.strip(),
        )
        ids.append(row["inbox_id"])
    return {"inserted": len(ids), "inbox_ids": ids}


@router.post("/subscription-inbox/consume", response_model=InboxConsumeResponse)
async def subscription_inbox_consume(body: InboxConsumeRequest | None = None):
    """取出待处理订阅消息并物理删除（consume 即删）。"""
    req = body or InboxConsumeRequest()
    rows = inbox_consume(channel=req.channel, limit=req.limit)
    return {"items": [InboxItem(**row) for row in rows]}


@router.delete("/subscription-inbox/{inbox_id}")
async def subscription_inbox_delete(inbox_id: str):
    """单条删除收件箱（consume 已批量删除时通常不需要）。"""
    if not inbox_delete(inbox_id):
        raise HTTPException(status_code=404, detail="inbox_id 不存在")
    return {"ok": True}


@router.post("/hot-board/upsert", response_model=HotBoardUpsertResponse)
async def hot_board_upsert_endpoint(body: HotBoardUpsertRequest):
    """清洗后的热榜写入（ingest skill / Merger）。"""
    if body.source not in ("wizz_alert", "merger_analyzer"):
        raise HTTPException(status_code=400, detail="source 必须是 wizz_alert 或 merger_analyzer")
    try:
        symbol = await validate_symbol_for_hot_board(body.symbol)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    entry = hot_board_upsert(
        {
            "symbol": symbol,
            "base_asset": body.base_asset,
            "source": body.source,
            "wizz": body.wizz,
            "merged_for_sentiment": body.merged_for_sentiment,
            "merger": body.merger,
        }
    )
    return {"ok": True, "entry": HotBoardEntry(**entry)}


@router.get("/hot-board/picker-snapshot", response_model=PickerSnapshotResponse)
async def hot_board_picker_snapshot(
    max_symbols: int = Query(10, ge=1, le=50),
    include_bundle: bool = Query(True),
):
    """供 hot-board-pick（crypto-post-flow Stage 0）读取有效热榜（可选附带 bundle）。"""
    time_data = get_shanghai_time()
    rows = hot_board_list_active(limit=max_symbols)
    entries: list[HotBoardEntry] = []
    for row in rows:
        bundle = await get_crypto_bundle(row["symbol"]) if include_bundle else None
        item = {**row, "bundle": bundle}
        entries.append(HotBoardEntry(**item))
    return {
        "board_ttl_hours": 12,
        "as_of": time_data["full"],
        "entries": entries,
    }


@router.get("/futures-symbols", response_model=FuturesSymbolsResponse)
async def futures_symbols():
    """TRADING 状态的币安 U 本位合约列表（供 ingest 校验）。"""
    symbols = await list_trading_symbols()
    return {"count": len(symbols), "symbols": symbols}


@router.get("/gainers", response_model=GainersResponse)
async def gainers(
    limit: int = Query(20, ge=1, le=100, description="返回涨幅榜数量"),
    min_quote_volume: float = Query(5_000_000, ge=0, description="最小 24h quoteVolume，默认过滤低流动性噪音"),
    include_1000: bool = Query(True, description="是否包含 1000PEPEUSDT 等乘数合约"),
):
    """币安 U 本位合约 24h 涨幅榜"""
    return await fetch_top_gainers(limit=limit, min_quote_volume=min_quote_volume, include_1000=include_1000)


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


@router.post("/distribute", response_model=DistributeResponse)
async def distribute(
    symbol: str = Form(..., min_length=1, description="BTC / ETH（可带USDT后缀）"),
    text: str = Form(..., min_length=1, description="待分发正文"),
    image: UploadFile | None = File(None, description="可选图片文件，支持常见 image/* 类型"),
    x_reply_to_previous: bool = Form(
        False,
        description="可选：发往 X 时用引用转帖方式引用上一次成功发送的帖子（读 X_LAST_POST_ID）。",
    ),
):
    """
    单接口统一分发到 Telegram / X / Binance Square。
    - 三渠道均支持图文：有图优先图文，缺图降级纯文本
    - Square：`#topic` 与 `$COIN` 标签须写在 `text` 正文内，由 Square OpenAPI 服务端解析（见 Binance square-post skill）
    """
    target = ensure_symbol_usdt(symbol)
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="text 不能为空")

    image_bytes: bytes | None = None
    image_filename: str | None = None
    image_content_type: str | None = None
    if image is not None:
        if not _looks_like_image_upload(image):
            raise HTTPException(status_code=400, detail="image 必须是图片文件")
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="image 不能为空")
        image_filename = image.filename or "upload.png"
        image_content_type = image.content_type

    result = await distribute_post(
        symbol_usdt=target,
        text=clean_text,
        image_bytes=image_bytes,
        image_filename=image_filename,
        image_content_type=image_content_type,
        x_reply_to_previous=x_reply_to_previous,
    )
    logger.info(
        "distribute api symbol=%s status=%s has_image=%s tg=%s x=%s square=%s",
        target,
        result.get("status"),
        bool(image_bytes),
        result.get("telegram_sent"),
        result.get("x_sent"),
        result.get("square_sent"),
    )
    return result

