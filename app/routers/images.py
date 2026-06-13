"""
图片生成与元数据清洗相关路由
"""
import json
import logging
import mimetypes
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from app.models.schemas import ImageGenerateRequest
from app.services.image_clean_service import ImageCleanError, clean_image_bytes, image_clean_health_payload
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


def _looks_like_image_upload(image: UploadFile) -> bool:
    content_type = (image.content_type or "").lower()
    if content_type.startswith("image/"):
        return True
    guessed, _ = mimetypes.guess_type(image.filename or "")
    return bool(guessed and guessed.startswith("image/"))


@router.get("/images/clean/health")
def image_clean_health():
    """返回当日缓存的 ffmpeg / exiftool 探测结果（GMT+8 每日 0 点刷新，非每次请求探测）。"""
    return image_clean_health_payload()


@router.post("/images/clean")
async def clean_image_endpoint(
    image: UploadFile = File(..., description="待清洗图片（GenerateImage 原图）"),
    mode: str = Form("standard", description="清洗模式：standard（4-pass，输出 JPEG）| strip-only"),
    quality: int = Form(92, ge=1, le=100, description="standard 模式 JPEG 质量"),
):
    """
    清洗 AI 生图元数据（C2PA / EXIF / XMP 等）。

    flow 契约：`GenerateImage` 成功后必须调用本接口，**严禁**将初始原图用于分发 / 落盘 / 校验交付。
    standard 模式固定 4-pass（Pass 1 含 gblur），输出横版 4:3 JPEG（约 1536×1024）。
    """
    if not _looks_like_image_upload(image):
        raise HTTPException(status_code=400, detail="image 必须是图片文件")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="image 不能为空")

    clean_mode = (mode or "standard").strip().lower()
    if clean_mode not in {"standard", "strip-only"}:
        raise HTTPException(status_code=400, detail=f"unsupported mode: {mode}")

    missing = image_clean_health_payload()["missing"]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"image clean unavailable, missing: {', '.join(missing)}",
        )

    filename = image.filename or "image.png"
    try:
        body, content_type, report = clean_image_bytes(
            raw,
            filename,
            mode=clean_mode,
            quality=quality,
        )
    except ImageCleanError as exc:
        logger.warning("image clean failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    out_name = Path(filename).stem + (".jpg" if clean_mode == "standard" else Path(filename).suffix or ".jpg")
    headers = {
        "Content-Disposition": f'attachment; filename="{out_name}"',
        "Content-Length": str(len(body)),
        "X-Image-Clean-Mode": report["mode"],
        "X-Image-Clean-Ok": "true",
        "X-Image-Clean-Width": str(report["width"]),
        "X-Image-Clean-Height": str(report["height"]),
        "X-Image-Clean-Report": json.dumps(report, ensure_ascii=False),
    }
    return Response(content=body, media_type=content_type, headers=headers)
