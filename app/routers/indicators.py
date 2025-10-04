"""
技术指标相关路由
"""
import logging
from fastapi import APIRouter, HTTPException
from app.models.schemas import IndicatorRequest
from app.services.indicator_service import calculate_indicators_sync, get_supported_intervals

logger = logging.getLogger(__name__)

router = APIRouter(tags=["技术指标"])


@router.get("/intervals")
def get_supported_intervals_endpoint():
    """获取支持的时间间隔"""
    return {
        "intervals": get_supported_intervals(),
        "description": "支持的时间间隔列表"
    }


@router.post("/calculate")
def calculate_indicators_endpoint(request: IndicatorRequest):
    """计算技术指标 - 支持并发"""
    try:
        logger.info(f"收到计算请求: {request.symbol} {request.interval}")
        result = calculate_indicators_sync(
            request.symbol, 
            request.interval, 
            request.config
        )
        
        if 'error' in result:
            logger.error(f"计算失败: {result['error']}")
            raise HTTPException(status_code=400, detail=result['error'])
        
        logger.info(f"计算成功: {request.symbol} {request.interval}")
        return result
        
    except Exception as e:
        logger.error(f"API错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
