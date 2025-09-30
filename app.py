"""
币安技术指标计算API服务
Web API服务
"""
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import os

# 导入指标计算服务
from indicators import calculate_indicators_sync, get_supported_intervals
# 导入图片生成服务
from generate import TextToImageGenerator

# 配置日志 - 只输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 初始化图片生成器
generator = TextToImageGenerator()

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

# 图片生成请求模型
class ImageGenerateRequest(BaseModel):
    text: str = Field(..., description="要转换为图片的文本内容")
    width: Optional[int] = Field(750, description="图片宽度，范围100-2000像素")
    font_size: Optional[int] = Field(20, description="字体大小，范围8-72像素")
    margin: Optional[int] = Field(20, description="边距，范围0-200像素")
    text_color: Optional[str] = Field("#2D3748", description="文字颜色（十六进制）")
    bg_color: Optional[str] = Field("#FFFFFF", description="背景颜色（十六进制）")
    format: Optional[str] = Field("png", description="图片格式(png, jpeg, webp, bmp)")
    quality: Optional[int] = Field(98, description="图片质量，范围1-100")


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

@app.post("/generate")
def generate_image_endpoint(request: ImageGenerateRequest):
    """生成文本图片并直接返回文件流"""
    try:
        # 验证文本内容
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="文本内容不能为空")
        
        # 统一处理换行符：支持 \\n 和 \n 两种格式
        text = text.replace('\\n', '\n')
        
        # 验证参数范围
        if request.width < 100 or request.width > 2000:
            raise HTTPException(status_code=400, detail="宽度必须在100-2000像素之间")
        if request.font_size < 8 or request.font_size > 72:
            raise HTTPException(status_code=400, detail="字体大小必须在8-72像素之间")
        if request.margin < 0 or request.margin > 200:
            raise HTTPException(status_code=400, detail="边距必须在0-200像素之间")
        if request.format not in ['png', 'jpeg', 'webp', 'bmp']:
            raise HTTPException(status_code=400, detail="不支持的图片格式")
        if request.quality < 1 or request.quality > 100:
            raise HTTPException(status_code=400, detail="图片质量必须在1-100之间")
        
        # 生成图片
        logger.info(f"生成图片: text_length={len(text)}, width={request.width}, font_size={request.font_size}, format={request.format}")
        
        image_data, actual_height = generator.generate(
            text=text,
            width=request.width,
            font_size=request.font_size,
            margin=request.margin,
            text_color=request.text_color,
            bg_color=request.bg_color,
            format=request.format,
            quality=request.quality
        )
        
        # 根据格式设置MIME类型
        mime_types = {
            'png': 'image/png',
            'jpeg': 'image/jpeg', 
            'webp': 'image/webp',
            'bmp': 'image/bmp'
        }
        
        # 根据格式设置文件扩展名
        ext = request.format.lower()
        if ext == 'jpeg':
            ext = 'jpg'
            
        filename = f'text_image_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{ext}'
        
        logger.info(f"图片生成成功: {request.width}x{actual_height}, 格式:{request.format}")
        
        # 重置BytesIO指针到开始位置
        image_data.seek(0)
        
        # 返回文件流
        return StreamingResponse(
            image_data,
            media_type=mime_types.get(request.format, 'application/octet-stream'),
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(image_data.getvalue()))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成图片失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成图片失败: {str(e)}")


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
