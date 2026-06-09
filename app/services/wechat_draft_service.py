"""maternal-post-flow Stage 3.5：公众号草稿箱 draft/add。"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Tuple

import aiohttp

from app.services.distribution_service import (
    _guess_image_mime,
    _normalize_image_filename,
    _read_env_with_source,
)

logger = logging.getLogger(__name__)

_WECHAT_API_BASE = "https://api.weixin.qq.com"
_TOKEN_URL = f"{_WECHAT_API_BASE}/cgi-bin/token"
_ADD_MATERIAL_URL = f"{_WECHAT_API_BASE}/cgi-bin/material/add_material"
_DRAFT_ADD_URL = f"{_WECHAT_API_BASE}/cgi-bin/draft/add"

_TOKEN_CACHE: Dict[str, Any] = {"app_id": "", "token": "", "expires_at": 0.0}
_TOKEN_SAFETY_MARGIN_SEC = 300


class WechatDraftError(Exception):
    """微信草稿流程内的可预期失败，message 已脱敏（不含 token / secret）。"""


def _read_wechat_credentials() -> Tuple[str, str, Dict[str, str]]:
    app_id, app_id_source = _read_env_with_source("WECHAT_APP_ID")
    app_secret, app_secret_source = _read_env_with_source("WECHAT_APP_SECRET")
    sources = {"app_id_source": app_id_source, "app_secret_source": app_secret_source}
    return app_id, app_secret, sources


async def _fetch_access_token(session: aiohttp.ClientSession, app_id: str, app_secret: str) -> str:
    now = time.time()
    if (
        _TOKEN_CACHE["app_id"] == app_id
        and _TOKEN_CACHE["token"]
        and _TOKEN_CACHE["expires_at"] > now
    ):
        return str(_TOKEN_CACHE["token"])

    params = {"grant_type": "client_credential", "appid": app_id, "secret": app_secret}
    async with session.get(_TOKEN_URL, params=params) as resp:
        payload = await resp.json(content_type=None)

    token = str(payload.get("access_token") or "")
    if not token:
        errcode = payload.get("errcode")
        raise WechatDraftError(f"fetch access_token failed (errcode={errcode})")

    expires_in = int(payload.get("expires_in") or 7200)
    _TOKEN_CACHE.update(
        {
            "app_id": app_id,
            "token": token,
            "expires_at": now + max(expires_in - _TOKEN_SAFETY_MARGIN_SEC, 60),
        }
    )
    return token


async def _upload_thumb_material(
    session: aiohttp.ClientSession,
    token: str,
    image_bytes: bytes,
    image_filename: str | None,
    image_content_type: str | None,
) -> str:
    filename = _normalize_image_filename(image_filename, image_content_type)
    content_type = _guess_image_mime(filename, image_content_type)

    form = aiohttp.FormData()
    form.add_field("media", image_bytes, filename=filename, content_type=content_type)
    async with session.post(
        _ADD_MATERIAL_URL, params={"access_token": token, "type": "image"}, data=form
    ) as resp:
        payload = await resp.json(content_type=None)

    media_id = str(payload.get("media_id") or "")
    if not media_id:
        errcode = payload.get("errcode")
        raise WechatDraftError(f"upload thumb material failed (errcode={errcode})")
    return media_id


async def _add_draft(
    session: aiohttp.ClientSession,
    token: str,
    article: Dict[str, Any],
) -> str:
    body = json.dumps({"articles": [article]}, ensure_ascii=False).encode("utf-8")
    async with session.post(
        _DRAFT_ADD_URL,
        params={"access_token": token},
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
    ) as resp:
        payload = await resp.json(content_type=None)

    media_id = str(payload.get("media_id") or "")
    if not media_id:
        errcode = payload.get("errcode")
        raise WechatDraftError(f"draft/add failed (errcode={errcode})")
    return media_id


async def create_wechat_draft(
    *,
    title: str,
    content_html: str,
    image_bytes: bytes,
    image_filename: str | None = None,
    image_content_type: str | None = None,
    digest: str = "",
    author: str = "",
    content_source_url: str = "",
    need_open_comment: int = 1,
    only_fans_can_comment: int = 0,
) -> Dict[str, Any]:
    """把单篇文章写入公众号草稿箱，返回 {status, media_id, notes}。"""
    app_id, app_secret, sources = _read_wechat_credentials()
    if not app_id or not app_secret:
        return {
            "status": "failed",
            "media_id": "",
            "notes": ["missing WECHAT_APP_ID or WECHAT_APP_SECRET"],
            "credential_check": sources,
        }

    body_html = (content_html or "").strip()
    if not body_html:
        return {"status": "failed", "media_id": "", "notes": ["missing content_html"]}

    article: Dict[str, Any] = {
        "title": title,
        "content": body_html,
        "need_open_comment": int(need_open_comment),
        "only_fans_can_comment": int(only_fans_can_comment),
    }
    if author:
        article["author"] = author
    if digest:
        article["digest"] = digest[:120]
    if content_source_url:
        article["content_source_url"] = content_source_url

    timeout = aiohttp.ClientTimeout(total=60)
    try:
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            token = await _fetch_access_token(session, app_id, app_secret)
            thumb_media_id = await _upload_thumb_material(
                session, token, image_bytes, image_filename, image_content_type
            )
            article["thumb_media_id"] = thumb_media_id
            media_id = await _add_draft(session, token, article)
    except WechatDraftError as exc:
        logger.warning("wechat draft failed: %s", exc)
        return {"status": "failed", "media_id": "", "notes": [str(exc)]}
    except aiohttp.ClientError as exc:
        logger.warning("wechat draft network error: %s", exc)
        return {"status": "failed", "media_id": "", "notes": ["wechat api network error"]}

    logger.info("wechat draft ok media_id_len=%s title_len=%s", len(media_id), len(title))
    return {"status": "success", "media_id": media_id, "notes": ["draft_saved"]}
