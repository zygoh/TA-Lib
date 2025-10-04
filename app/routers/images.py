"""
图片生成相关路由
"""
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import ImageGenerateRequest
from app.services.image_service import ImageGeneratorService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["图片生成"])

# 初始化图片生成服务
image_service = ImageGeneratorService()


@router.post("/generate")
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
        
        image_data, actual_height = image_service.generate(
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
