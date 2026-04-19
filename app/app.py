"""
币安技术指标计算API服务
主应用入口
"""
import logging
from datetime import datetime
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw

from app.models.schemas import HealthResponse
from app.routers import indicators, images, trading, crypto_mcp, oauth_x
from app.routers.crypto_mcp import start_grok_scheduler, stop_grok_scheduler
from app.services.cursor_agent_scheduler import cursor_agent_scheduler

# 配置日志 - 只输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """应用生命周期：启动/关闭时管理 Grok 定时任务。"""
    start_grok_scheduler()
    logger.info("Grok 定时任务已启动（ETH 07:55 / BTC 19:55，Asia/Shanghai）")
    cursor_agent_scheduler.start()
    try:
        yield
    finally:
        await cursor_agent_scheduler.stop()
        await stop_grok_scheduler()
        logger.info("Grok 定时任务已停止")


# 创建FastAPI应用
app = FastAPI(
    title="币安技术指标计算API",
    description="支持并发的币安期货技术指标计算服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
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
app.include_router(crypto_mcp.router)

# X OAuth2：同时挂载根路径与 /tail，适配反代（如 https://do2ge.com/tail → 本服务）
app.include_router(oauth_x.router)
app.include_router(oauth_x.router, prefix="/tail")

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


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    返回站点 favicon（避免浏览器请求 404）。
    运行时生成 32x32 ICO，无需额外静态文件。
    """
    size = 32
    img = Image.new("RGBA", (size, size), (22, 33, 62, 255))  # 深色底
    draw = ImageDraw.Draw(img)
    draw.rectangle([6, 6, size - 6, size - 6], fill=(38, 166, 154, 255))  # 亮色块

    buf = io.BytesIO()
    img.save(buf, format="ICO", sizes=[(size, size)])
    return Response(content=buf.getvalue(), media_type="image/x-icon")
