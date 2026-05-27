"""
crypto-mcp 兼容接口的请求/响应模型
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class TimeResponse(BaseModel):
    full: str
    short: str
    timestamp: int


class SentimentResponse(BaseModel):
    grok_analysis: str = Field(
        default="",
        description="已停用：不再调用 xAI/Grok，此字段恒为空字符串（保留字段以兼容旧客户端）",
    )
    news_articles: List[Dict[str, Any]]
    news_summary: str
    news_headlines: List[str]
    keyword: str
    keywords: List[str] = Field(default_factory=list, description="用于匹配当前 symbol 的新闻关键词")
    symbol: str


class GainerItem(BaseModel):
    rank: int
    symbol: str
    base_asset: str
    normalized_base_asset: str
    priceChangePercent: float
    lastPrice: float
    quoteVolume: float
    volume: float


class GainersResponse(BaseModel):
    source: str
    min_quote_volume: float
    include_1000: bool
    gainers: List[GainerItem]


class CryptoBundleResponse(BaseModel):
    target: str
    technical_analysis: Dict[str, Any]
    market_analysis: Dict[str, Any]
    sentiment_analysis: Dict[str, Any]
    hot_board_supplement: Optional[Dict[str, Any]] = Field(
        default=None,
        description="当 symbol 在 12h 热榜时附带 Wizz/Merger 补充事实",
    )


class InboxItem(BaseModel):
    inbox_id: str
    received_at: str
    channel_username: str
    message_id: Optional[int] = None
    permalink: Optional[str] = None
    raw_text: str


class InboxConsumeRequest(BaseModel):
    channel: str = "wizzalert"
    limit: int = Field(default=50, ge=1, le=200)


class InboxConsumeResponse(BaseModel):
    items: List[InboxItem]


class InboxSeedItem(BaseModel):
    raw_text: str
    message_id: Optional[int] = None


class InboxSeedRequest(BaseModel):
    channel: str = "wizzalert"
    items: List[InboxSeedItem]


class InboxSeedResponse(BaseModel):
    inserted: int
    inbox_ids: List[str]


class HotBoardUpsertRequest(BaseModel):
    symbol: str
    source: str = Field(..., description="wizz_alert | merger_analyzer")
    base_asset: Optional[str] = None
    wizz: Optional[Dict[str, Any]] = None
    merged_for_sentiment: Optional[str] = None
    merger: Optional[Dict[str, Any]] = None


class HotBoardEntry(BaseModel):
    symbol: str
    base_asset: Optional[str] = None
    first_seen_at: str
    last_seen_at: str
    expires_at: str
    sources: List[str]
    hit_count: int
    wizz: Optional[Dict[str, Any]] = None
    merged_for_sentiment: Optional[str] = None
    merger: Optional[Dict[str, Any]] = None
    bundle: Optional[Dict[str, Any]] = None


class HotBoardUpsertResponse(BaseModel):
    ok: bool = True
    entry: HotBoardEntry


class PickerSnapshotResponse(BaseModel):
    board_ttl_hours: int = 12
    as_of: str
    entries: List[HotBoardEntry]


class FuturesSymbolsResponse(BaseModel):
    count: int
    symbols: List[str]


class ChartsResponse(BaseModel):
    symbol: str
    charts: Dict[str, Dict[str, Any]]
    errors: List[Dict[str, str]]


class CryptoMcpAllResponse(BaseModel):
    symbol: str
    time: TimeResponse
    bundle: CryptoBundleResponse


class DistributeResponse(BaseModel):
    status: str = Field(..., description="success / partial / failed")
    symbol: str
    telegram_sent: bool
    x_sent: bool
    square_sent: bool
    channels: Dict[str, Any]
    notes: List[str]

