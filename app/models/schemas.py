"""
请求和响应数据模型
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# --- 之前的指标请求模型 (保持不变) ---
class IndicatorRequest(BaseModel):
    exchange: str = Field("binance", description="交易所名称")
    symbol: str = Field(..., description="交易对，例如 BTC/USDT")
    interval: str = Field(..., description="时间周期，例如 1h, 4h, 1d")
    limit: int = Field(100, description="获取K线数量")
    indicators: List[Dict[str, Any]] = Field(..., description="指标配置列表")

# --- 新增：交易信号请求模型 (添加这个类) ---
class TradingSignalsRequest(BaseModel):
    exchange: str = Field("binance", description="交易所名称，默认为 binance")
    symbol: str = Field(..., description="交易对，例如 ETH/USDT")
    interval: str = Field(..., description="时间周期，例如 15m, 1h, 4h")
    # 你可以在这里添加更多字段，比如策略参数等


class ImageGenerateRequest(BaseModel):
    """图片生成请求模型"""
    text: str = Field(..., description="要转换为图片的文本内容")
    width: Optional[int] = Field(750, description="图片宽度，范围100-2000像素")
    font_size: Optional[int] = Field(20, description="字体大小，范围8-72像素")
    margin: Optional[int] = Field(20, description="边距，范围0-200像素")
    text_color: Optional[str] = Field("#2D3748", description="文字颜色（十六进制）")
    bg_color: Optional[str] = Field("#FFFFFF", description="背景颜色（十六进制）")
    format: Optional[str] = Field("png", description="图片格式(png, jpeg, webp, bmp)")
    quality: Optional[int] = Field(98, description="图片质量，范围1-100")


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    timestamp: str
    version: str
