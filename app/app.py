"""
币安技术指标计算API服务
主应用入口
"""
import logging
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import HealthResponse
from app.routers import indicators, images, trading, coin_selection
from app.services.coin_selector_service import get_coin_selector_service

# 配置日志 - 只输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
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

# 注册路由
app.include_router(indicators.router)
app.include_router(images.router)
app.include_router(trading.router)
app.include_router(coin_selection.router)


# 生命周期事件
@app.on_event("startup")
async def startup_event():
    """启动时初始化选币服务（立即执行一次选币 + 启动后台定时任务）"""
    service = await get_coin_selector_service()
    await service.start_background_task()


@app.on_event("shutdown")
async def shutdown_event():
    """关闭时释放选币服务资源"""
    service = await get_coin_selector_service()
    await service.close()

# 基础路由
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
