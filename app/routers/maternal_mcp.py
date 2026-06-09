"""
maternal-post-flow 专用：仅向 Telegram 分发（复用 distribution_service 与 TA-Lib/.env 中的 TG_* 配置）。
"""

from __future__ import annotations

import logging
import mimetypes

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.crypto_mcp_schemas import (
    MaternalDistributeResponse,
    WechatDraftResponse,
)
from app.services.distribution_service import _send_telegram
from app.services.wechat_draft_service import create_wechat_draft

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/maternal-mcp", tags=["maternal-mcp"])


def _looks_like_image_upload(image: UploadFile) -> bool:
    content_type = (image.content_type or "").lower()
    if content_type.startswith("image/"):
        return True
    guessed, _ = mimetypes.guess_type(image.filename or "")
    return bool(guessed and guessed.startswith("image/"))


@router.post("/distribute", response_model=MaternalDistributeResponse)
async def maternal_distribute(
    title: str = Form(..., min_length=1, description="标题（作封面图说明首行）"),
    text: str = Form(..., min_length=1, description="完整正稿正文"),
    digest: str = Form("", description="摘要（可选，作封面图说明第二行）"),
    image: UploadFile = File(..., description="封面图，必填"),
):
    """
    母婴 flow 专用：只发 Telegram，不碰 X / Square / 公众号。
    顺序：① sendPhoto（封面 + title/digest 说明）→ ② sendMessage（完整正稿）。
    凭据读 TA-Lib/.env：`TG_BOT_TOKEN` / `TG_CHAT_ID`（兼容 `TELEGRAM_*`）。
    """
    clean_title = title.strip()
    clean_text = text.strip()
    clean_digest = digest.strip()
    if not clean_title:
        raise HTTPException(status_code=400, detail="title 不能为空")
    if not clean_text:
        raise HTTPException(status_code=400, detail="text 不能为空")
    if not _looks_like_image_upload(image):
        raise HTTPException(status_code=400, detail="image 必须是图片文件")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="image 不能为空")

    caption = clean_title
    if clean_digest:
        caption = f"{caption}\n{clean_digest}"

    cover_result = await _send_telegram(
        caption,
        image_bytes,
        image.filename or "cover.png",
        image.content_type,
    )
    notes: list[str] = []
    if not cover_result.get("sent"):
        note = cover_result.get("note") or "cover sendPhoto failed"
        logger.warning("maternal distribute cover failed: %s", note)
        return MaternalDistributeResponse(
            status="failed",
            telegram_sent=False,
            notes=[f"cover_image: {note}"],
        )

    article_result = await _send_telegram(clean_text, None, None, None)
    if not article_result.get("sent"):
        note = article_result.get("note") or "article sendMessage failed"
        logger.warning("maternal distribute article failed: %s", note)
        notes.append(f"cover_image: ok")
        notes.append(f"article_text: {note}")
        return MaternalDistributeResponse(
            status="failed",
            telegram_sent=False,
            notes=notes,
        )

    logger.info("maternal distribute ok title_len=%s text_len=%s", len(clean_title), len(clean_text))
    return MaternalDistributeResponse(
        status="success",
        telegram_sent=True,
        notes=["cover_image: ok", "article_text: ok"],
    )


@router.post("/wechat-draft", response_model=WechatDraftResponse)
async def maternal_wechat_draft(
    title: str = Form(..., min_length=1, description="文章标题"),
    content_html: str = Form(..., min_length=1, description="内联 CSS HTML"),
    image: UploadFile = File(..., description="封面图，作 draft thumb"),
    digest: str = Form("", description="摘要（≤120 字）"),
    author: str = Form("", description="作者"),
    content_source_url: str = Form("", description="阅读原文"),
):
    """maternal-post-flow Stage 3.5：写入公众号草稿箱。"""
    clean_title = title.strip()
    clean_html = content_html.strip()
    if not clean_title:
        raise HTTPException(status_code=400, detail="title 不能为空")
    if not clean_html:
        raise HTTPException(status_code=400, detail="content_html 不能为空")
    if not _looks_like_image_upload(image):
        raise HTTPException(status_code=400, detail="image 必须是图片文件")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="image 不能为空")

    result = await create_wechat_draft(
        title=clean_title,
        content_html=clean_html,
        image_bytes=image_bytes,
        image_filename=image.filename or "cover.png",
        image_content_type=image.content_type,
        digest=digest.strip(),
        author=author.strip(),
        content_source_url=content_source_url.strip(),
    )
    return WechatDraftResponse(
        status=result.get("status", "failed"),
        media_id=result.get("media_id", ""),
        notes=result.get("notes", []),
    )
