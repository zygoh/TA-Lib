"""
选币路由 - GET /tro 端点
"""
import logging

from fastapi import APIRouter, HTTPException

from app.models.coin_selection_schemas import CoinSelectionResponse
from app.services.coin_selector_service import get_coin_selector_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["选币"])


@router.get("/tro", response_model=CoinSelectionResponse)
async def get_top_coin() -> CoinSelectionResponse:
    """获取当前推荐币种

    优先返回缓存数据；缓存为空时触发即时计算。
    """
    service = await get_coin_selector_service()
    cached = await service.get_cached_result()

    if cached is not None:
        return CoinSelectionResponse(
            symbol=cached.symbol,
            score=cached.score,
            price=cached.price,
            change_24h=cached.change_24h,
            updated_at=cached.updated_at,
        )

    # 缓存为空，触发即时计算
    logger.info("📊 缓存为空，触发即时选币计算")
    try:
        result = await service.refresh()
        return CoinSelectionResponse(
            symbol=result.symbol,
            score=result.score,
            price=result.price,
            change_24h=result.change_24h,
            updated_at=result.updated_at,
        )
    except Exception as e:
        logger.error(f"❌ 即时选币计算失败: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"选币服务暂不可用: {e}",
        )
