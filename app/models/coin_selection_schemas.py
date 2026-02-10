"""
选币功能数据模型
"""
from pydantic import BaseModel, Field


class CoinSelectionResponse(BaseModel):
    """选币接口响应模型"""
    symbol: str = Field(..., description="交易对符号，如 BTCUSDT")
    score: float = Field(..., description="综合评分 0-100")
    price: float = Field(..., description="当前价格")
    change_24h: float = Field(..., description="24小时涨跌幅（百分比）")
    updated_at: str = Field(..., description="更新时间 ISO 8601 格式")
