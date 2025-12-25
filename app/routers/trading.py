"""
交易相关路由
"""
import logging
from typing import List
from fastapi import APIRouter, HTTPException, Response
from app.models.schemas import TradingSignal
from app.services.trading_service import get_account_info, get_account_markdown, process_trading_signals

logger = logging.getLogger(__name__)

router = APIRouter(tags=["交易"])


@router.get("/account")
async def get_account_endpoint():
    """获取币安合约账户信息（固定19x杠杆）"""
    try:
        logger.info("收到获取账户信息请求")
        account_data = await get_account_info()
        return account_data
    except Exception as e:
        logger.error(f"获取账户信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/account/markdown")
async def get_account_markdown_endpoint():
    """获取币安合约账户信息的 Markdown 格式（供 Agent 使用）"""
    try:
        logger.info("收到获取账户 Markdown 请求")
        markdown = await get_account_markdown()
        return Response(content=markdown, media_type="text/markdown; charset=utf-8")
    except Exception as e:
        logger.error(f"获取账户 Markdown 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/signals")
async def receive_signals_endpoint(signals: List[TradingSignal]):
    """接收交易信号（直接接收数组）"""
    try:
        logger.info(f"收到交易信号请求，共{len(signals)}个信号")
        
        signals_data = [signal.dict() for signal in signals]
        result = await process_trading_signals(signals_data)
        
        logger.info(f"处理完成，共处理{result['processed_count']}个信号")
        return result
    except Exception as e:
        logger.error(f"处理交易信号失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
