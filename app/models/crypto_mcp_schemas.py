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


class MaternalDistributeResponse(BaseModel):
    status: str = Field(..., description="success / failed")
    telegram_sent: bool
    notes: List[str] = Field(default_factory=list)


class WechatDraftResponse(BaseModel):
    status: str = Field(..., description="success / failed")
    media_id: str = Field(default="", description="草稿 media_id（成功时非空）")
    notes: List[str] = Field(default_factory=list)

