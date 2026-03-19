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
    grok_analysis: str
    news_articles: List[Dict[str, Any]]
    news_summary: str
    news_headlines: List[str]
    keyword: str
    symbol: str


class CryptoBundleResponse(BaseModel):
    target: str
    technical_analysis: Dict[str, Any]
    market_analysis: Dict[str, Any]
    sentiment_analysis: Dict[str, Any]


class GrokUpdateRequest(BaseModel):
    content: str = Field(..., min_length=1, description="发送给 Grok AI 的用户消息内容")


class GrokUpdateResponse(BaseModel):
    ok: bool
    error: str | None = None


class GrokStatusResponse(BaseModel):
    status: str = Field(..., description="任务状态: pending, running, completed, failed")
    error: str | None = None
    updated_at: int = Field(0, description="最后更新时间戳（秒）")


class ChartsResponse(BaseModel):
    symbol: str
    charts: Dict[str, Dict[str, Any]]
    errors: List[Dict[str, str]]


class TelegramSendRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Markdown 文本")


class TelegramSendResponse(BaseModel):
    ok: bool
    result: Dict[str, Any]


class CryptoMcpAllResponse(BaseModel):
    symbol: str
    time: TimeResponse
    bundle: CryptoBundleResponse

