"""
币安技术指标计算API服务
Web API服务
"""
import logging
from datetime import datetime
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import os

# 导入指标计算服务
from indicators import calculate_indicators_sync, get_supported_intervals

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="币安技术指标计算API",
    description="支持并发的币安期货技术指标计算服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class IndicatorRequest(BaseModel):
    symbol: str = Field(..., description="交易对符号，如BTCUSDT")
    interval: str = Field(..., description="时间间隔，如15m, 1h, 1d")
    config: Dict[str, Any] = Field(..., description="技术指标配置参数")


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

# 全局变量（配置已移至config.json）

# API路由
@app.get("/", response_model=HealthResponse)
async def root():
    """根路径健康检查"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.get("/intervals")
def get_supported_intervals():
    """获取支持的时间间隔"""
    from indicators import get_supported_intervals as get_intervals
    return {
        "intervals": get_intervals(),
        "description": "支持的时间间隔列表"
    }

@app.post("/calculate")
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


if __name__ == "__main__":
    # 生产环境配置
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info",
        access_log=True
    )
