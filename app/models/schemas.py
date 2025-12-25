"""
请求和响应数据模型
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

# 技术指标计算请求模型
class IndicatorRequest(BaseModel):
    """技术指标计算请求模型"""
    symbol: str = Field(..., description="交易对符号，如BTCUSDT")
    interval: str = Field(..., description="时间间隔，如15m, 1h, 1d")
    config: Dict[str, Any] = Field(..., description="技术指标配置参数")

# --- 交易信号请求模型 ---
class TradingSignal(BaseModel):
    """单个交易信号模型"""
    symbol: str = Field(..., description="交易对符号，如 BTCUSDT")
    action: str = Field(..., description="操作类型: open_long, open_short, close_long, close_short, adjust_stops, wait, hold")
    leverage: Optional[int] = Field(None, description="杠杆倍数（开仓时，但系统固定使用20x，此字段可忽略）")
    position_size_usd: Optional[float] = Field(None, description="仓位大小（USD，开仓时必填）")
    stop_loss: Optional[float] = Field(None, description="止损价格（开仓或调整时必填）")
    take_profit: Optional[float] = Field(None, description="止盈价格（开仓或调整时必填）")
    confidence: Optional[int] = Field(None, description="信心度 0-100（开仓时必填，需≥90）")
    risk_usd: Optional[float] = Field(None, description="风险金额（USD，开仓时必填）")
    reasoning: Optional[str] = Field(None, description="操作原因说明（可选）")

class TradingSignalsRequest(BaseModel):
    """交易信号请求模型"""
    signals: List[TradingSignal] = Field(..., description="交易信号列表")


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
