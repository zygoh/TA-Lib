"""
请求和响应数据模型
"""
from pydantic import BaseModel, Field
from typing import Optional,Dict, Any

# 技术指标计算请求模型
class IndicatorRequest(BaseModel):
    """技术指标计算请求模型"""
    symbol: str = Field(..., description="交易对符号，如BTCUSDT")
    interval: str = Field(..., description="时间间隔，如15m, 1h, 1d")
    config: Dict[str, Any] = Field(..., description="技术指标配置参数")

# --- 新增：交易信号请求模型 (添加这个类) ---
class TradingSignalsRequest(BaseModel):
    """交易信号请求模型"""
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
