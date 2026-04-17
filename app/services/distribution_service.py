from __future__ import annotations

import base64
import asyncio
import hashlib
import hmac
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote
from zoneinfo import ZoneInfo

import aiohttp

logger = logging.getLogger(__name__)

_SH_TZ = ZoneInfo("Asia/Shanghai")

_TG_API_BASE = "https://api.telegram.org"
_X_API_BASE = "https://api.x.com/2"
_X_UPLOAD_BASE = "https://upload.twitter.com/1.1"
_SQUARE_POST_URL = "https://www.binance.com/bapi/composite/v1/public/pgc/openApi/content/add"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_chart_4h_path(symbol_usdt: str) -> str | None:
    base_symbol = symbol_usdt.replace("USDT", "").upper()
    date_str = datetime.now(_SH_TZ).strftime("%Y-%m-%d")
    image_root = _repo_root() / "image"
    candidate = image_root / f"{base_symbol}_{date_str}" / f"{base_symbol}_4h.png"
    logger.info(
        "distribute chart lookup symbol=%s base_dir=%s candidate=%s",
        symbol_usdt,
        image_root,
        candidate,
    )
    if candidate.is_file():
        logger.info("distribute chart resolved symbol=%s path=%s", symbol_usdt, candidate)
        return str(candidate)
    logger.warning("distribute chart missing symbol=%s candidate=%s", symbol_usdt, candidate)
    return None


def _rfc3986(value: str) -> str:
    return quote(str(value), safe="~")


def _oauth_authorization_header(
    method: str,
    url: str,
    consumer_key: str,
    consumer_secret: str,
    access_token: str,
    access_token_secret: str,
    extra_params: Dict[str, str] | None = None,
) -> str:
    oauth_params: Dict[str, str] = {
        "oauth_consumer_key": consumer_key,
        "oauth_nonce": base64.b16encode(os.urandom(12)).decode("ascii").lower(),
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(int(time.time())),
        "oauth_token": access_token,
        "oauth_version": "1.0",
    }

    all_params = dict(oauth_params)
    if extra_params:
        all_params.update(extra_params)

    param_string = "&".join(
        f"{_rfc3986(k)}={_rfc3986(v)}" for k, v in sorted(all_params.items(), key=lambda item: item[0])
    )
    signature_base = "&".join([method.upper(), _rfc3986(url), _rfc3986(param_string)])
    signing_key = f"{_rfc3986(consumer_secret)}&{_rfc3986(access_token_secret)}"
    signature = base64.b64encode(hmac.new(signing_key.encode("utf-8"), signature_base.encode("utf-8"), hashlib.sha1).digest()).decode(
        "utf-8"
    )

    oauth_params["oauth_signature"] = signature
    return "OAuth " + ", ".join(
        f'{_rfc3986(k)}="{_rfc3986(v)}"' for k, v in sorted(oauth_params.items(), key=lambda item: item[0])
    )


async def _send_telegram(text: str, image_path: str | None) -> Dict[str, Any]:
    token = (os.getenv("TG_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("TG_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat_id:
        return {"sent": False, "mode": "none", "note": "missing TG_BOT_TOKEN or TG_CHAT_ID"}

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        if image_path and Path(image_path).is_file():
            form = aiohttp.FormData()
            form.add_field("chat_id", chat_id)
            form.add_field("caption", text)
            form.add_field("photo", Path(image_path).read_bytes(), filename=Path(image_path).name)
            async with session.post(f"{_TG_API_BASE}/bot{token}/sendPhoto", data=form) as resp:
                payload = await resp.json(content_type=None)
                if resp.status < 400:
                    return {"sent": True, "mode": "image+text", "payload": payload}
                return {"sent": False, "mode": "image+text", "note": "sendPhoto failed", "payload": payload}

        body = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
        async with session.post(
            f"{_TG_API_BASE}/bot{token}/sendMessage",
            json=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
        ) as resp:
            payload = await resp.json(content_type=None)
            if resp.status < 400:
                return {"sent": True, "mode": "text", "payload": payload}
            return {"sent": False, "mode": "text", "note": "sendMessage failed", "payload": payload}


async def _x_upload_media(
    session: aiohttp.ClientSession,
    image_path: str,
    consumer_key: str,
    consumer_secret: str,
    access_token: str,
    access_token_secret: str,
) -> str:
    media_url = f"{_X_UPLOAD_BASE}/media/upload.json"
    image_data = Path(image_path).read_bytes()
    media_b64 = base64.b64encode(image_data).decode("utf-8")

    auth = _oauth_authorization_header(
        method="POST",
        url=media_url,
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
    )

    form = aiohttp.FormData()
    form.add_field("media_data", media_b64)
    form.add_field("media_category", "tweet_image")
    async with session.post(media_url, data=form, headers={"Authorization": auth}) as resp:
        payload = await resp.json(content_type=None)
        if resp.status >= 400:
            raise RuntimeError(f"x media upload failed {resp.status}: {payload}")
        media_id = payload.get("media_id_string")
        if not media_id:
            raise RuntimeError(f"x media upload missing media_id_string: {payload}")
        return str(media_id)


async def _send_x(text: str, image_path: str | None) -> Dict[str, Any]:
    consumer_key = os.getenv("X_CONSUMER_KEY", "").strip()
    consumer_secret = os.getenv("X_CONSUMER_SECRET", "").strip()
    access_token = os.getenv("X_ACCESS_TOKEN", "").strip()
    access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET", "").strip()

    if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
        return {
            "sent": False,
            "mode": "none",
            "note": "missing X_CONSUMER_KEY/X_CONSUMER_SECRET/X_ACCESS_TOKEN/X_ACCESS_TOKEN_SECRET",
        }

    timeout = aiohttp.ClientTimeout(total=45)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        body: Dict[str, Any] = {"text": text}
        mode = "text"

        if image_path and Path(image_path).is_file():
            try:
                media_id = await _x_upload_media(
                    session,
                    image_path,
                    consumer_key,
                    consumer_secret,
                    access_token,
                    access_token_secret,
                )
                body["media"] = {"media_ids": [media_id]}
                mode = "image+text"
            except Exception as exc:  # noqa: BLE001
                return {"sent": False, "mode": "image+text", "note": f"x media upload failed: {exc}"}

        tweet_url = f"{_X_API_BASE}/tweets"
        auth = _oauth_authorization_header(
            method="POST",
            url=tweet_url,
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
        )
        async with session.post(
            tweet_url,
            json=body,
            headers={
                "Authorization": auth,
                "Content-Type": "application/json; charset=utf-8",
            },
        ) as resp:
            payload = await resp.json(content_type=None)
            if resp.status >= 400:
                return {"sent": False, "mode": mode, "note": f"x post failed http={resp.status}", "payload": payload}
            tweet_id = payload.get("data", {}).get("id")
            result: Dict[str, Any] = {"sent": True, "mode": mode, "payload": payload}
            if tweet_id:
                result["url"] = f"https://x.com/i/status/{tweet_id}"
            return result


async def _send_square(text: str) -> Dict[str, Any]:
    api_key = os.getenv("SQUARE_OPENAPI_KEY", "").strip()
    if not api_key:
        return {"sent": False, "mode": "none", "note": "missing SQUARE_OPENAPI_KEY"}

    timeout = aiohttp.ClientTimeout(total=30)
    body = {"bodyTextOnly": text}
    headers = {
        "X-Square-OpenAPI-Key": api_key,
        "Content-Type": "application/json; charset=utf-8",
        "clienttype": "binanceSkill",
    }
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        async with session.post(_SQUARE_POST_URL, json=body, headers=headers) as resp:
            payload = await resp.json(content_type=None)
            if resp.status >= 400:
                return {"sent": False, "mode": "text", "note": f"square http failed {resp.status}", "payload": payload}

            code = payload.get("code")
            post_id = payload.get("data", {}).get("id")
            if code == "000000" and post_id:
                return {
                    "sent": True,
                    "mode": "text",
                    "id": post_id,
                    "url": f"https://www.binance.com/en/square/post/{post_id}",
                    "payload": payload,
                }
            if code == "000000":
                return {"sent": True, "mode": "text", "note": "square success but id missing", "payload": payload}
            return {"sent": False, "mode": "text", "note": "square business failed", "payload": payload}


def _derive_status(telegram_sent: bool, x_sent: bool, square_sent: bool) -> str:
    flags = [telegram_sent, x_sent, square_sent]
    if all(flags):
        return "success"
    if any(flags):
        return "partial"
    return "failed"


async def distribute_post(
    symbol_usdt: str,
    text: str,
) -> Dict[str, Any]:
    if not text.strip():
        raise ValueError("text 不能为空")

    logger.info("distribute start symbol=%s text_len=%d", symbol_usdt, len(text))
    use_chart_4h = _resolve_chart_4h_path(symbol_usdt)
    telegram_result, x_result, square_result = await asyncio.gather(
        _send_telegram(text, use_chart_4h),
        _send_x(text, use_chart_4h),
        _send_square(text),
    )

    telegram_sent = bool(telegram_result.get("sent"))
    x_sent = bool(x_result.get("sent"))
    square_sent = bool(square_result.get("sent"))
    status = _derive_status(telegram_sent, x_sent, square_sent)

    notes: List[str] = []
    if use_chart_4h is None:
        notes.append("chart_4h missing under TA-Lib/image, TG/X may fallback to text")
    if status == "partial":
        notes.append("partial success")
    elif status == "failed":
        notes.append("all channels failed")

    result = {
        "status": status,
        "symbol": symbol_usdt,
        "telegram_sent": telegram_sent,
        "x_sent": x_sent,
        "square_sent": square_sent,
        "channels": {
            "telegram": telegram_result,
            "x": x_result,
            "square": square_result,
        },
        "notes": notes,
    }
    logger.info(
        "distribute status=%s symbol=%s tg=%s(%s) x=%s(%s) square=%s(%s)",
        status,
        symbol_usdt,
        telegram_sent,
        telegram_result.get("mode"),
        x_sent,
        x_result.get("mode"),
        square_sent,
        square_result.get("mode"),
    )
    if notes:
        logger.info("distribute notes symbol=%s notes=%s", symbol_usdt, notes)
    return result
