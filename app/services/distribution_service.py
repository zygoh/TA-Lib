from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import aiohttp
import requests
from requests import HTTPError
from xdk import Client as XdkClient
from xdk.media.models import AppendUploadRequest, InitializeUploadRequest
from xdk.oauth1_auth import OAuth1
from xdk.posts.models import CreateRequest, CreateRequestMedia

logger = logging.getLogger(__name__)

_SH_TZ = ZoneInfo("Asia/Shanghai")

_TG_API_BASE = "https://api.telegram.org"
_X_MEDIA_UPLOAD_V11 = "https://upload.twitter.com/1.1/media/upload.json"
_SQUARE_POST_URL = "https://www.binance.com/bapi/composite/v1/public/pgc/openApi/content/add"
_DOTENV_CACHE: Dict[str, str] | None = None

_X_OAUTH2_SCOPES = ["tweet.read", "tweet.write", "users.read", "media.write", "offline.access"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_repo_dotenv() -> Dict[str, str]:
    global _DOTENV_CACHE
    if _DOTENV_CACHE is not None:
        return _DOTENV_CACHE

    env_map: Dict[str, str] = {}
    env_path = _repo_root() / ".env"
    if env_path.is_file():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key:
                env_map[key] = value.strip()
    _DOTENV_CACHE = env_map
    return env_map


def _read_env_with_source(name: str) -> Tuple[str, str]:
    process_value = (os.getenv(name) or "").strip()
    if process_value:
        return process_value, "process_env"

    dotenv_value = _load_repo_dotenv().get(name, "").strip()
    if dotenv_value:
        return dotenv_value, "repo_dotenv"
    return "", "missing"


def _credential_snapshot(names: List[str]) -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    for name in names:
        value, source = _read_env_with_source(name)
        snapshot[name] = {"present": bool(value), "source": source, "length": len(value)}
    return snapshot


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


def _read_x_client_id() -> str:
    v, _ = _read_env_with_source("X_CLIENT_ID")
    if v:
        return v
    v, _ = _read_env_with_source("CLIENT_ID")
    return v


def _read_x_client_secret() -> str:
    v, _ = _read_env_with_source("X_CLIENT_SECRET")
    if v:
        return v
    v, _ = _read_env_with_source("CLIENT_SECRET")
    return v


def _read_x_redirect_uri() -> str:
    v, _ = _read_env_with_source("X_REDIRECT_URI")
    if v:
        return v
    v, _ = _read_env_with_source("OAUTH2_REDIRECT_URI")
    return v if v else "https://127.0.0.1/callback"


def _read_oauth2_user_token() -> Optional[Dict[str, Any]]:
    raw, _ = _read_env_with_source("X_OAUTH2_TOKEN")
    if raw:
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            logger.warning("X_OAUTH2_TOKEN present but not valid JSON")
            return None

    access, _ = _read_env_with_source("X_OAUTH2_ACCESS_TOKEN")
    if not access:
        return None
    token: Dict[str, Any] = {"access_token": access.strip()}
    refresh, _ = _read_env_with_source("X_OAUTH2_REFRESH_TOKEN")
    if refresh:
        token["refresh_token"] = refresh.strip()
    return token


def _guess_image_mime(image_path: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    if mime and mime.startswith("image/"):
        return mime
    return "image/png"


def _upload_image_oauth1_v11(client: XdkClient, image_path: str) -> str:
    """Legacy v1.1 media upload (OAuth 1.0a user context), signed via XDK OAuth1."""
    if not client.auth:
        raise RuntimeError("x oauth1 auth missing on client")
    media_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    auth_header = client.auth.build_request_header("POST", _X_MEDIA_UPLOAD_V11, "")
    resp = requests.post(
        _X_MEDIA_UPLOAD_V11,
        data={"media_data": media_b64, "media_category": "tweet_image"},
        headers={"Authorization": auth_header},
        timeout=45,
    )
    payload = resp.json() if resp.content else {}
    if resp.status_code >= 400:
        raise RuntimeError(f"x media upload failed {resp.status_code}: {payload}")
    media_id = payload.get("media_id_string")
    if not media_id:
        raise RuntimeError(f"x media upload missing media_id_string: {payload}")
    return str(media_id)


def _upload_image_oauth2_v2(client: XdkClient, image_path: str) -> str:
    """Media Upload v2 with OAuth 2.0 user token (Bearer), aligned with official samples."""
    raw = Path(image_path).read_bytes()
    total = len(raw)
    mime = _guess_image_mime(image_path)
    init_body = InitializeUploadRequest(
        total_bytes=total,
        media_type=mime,
        media_category="tweet_image",
    )
    init = client.media.initialize_upload(body=init_body)
    if init.errors:
        raise RuntimeError(f"x media initialize failed: {init.errors}")
    mid = init.data.id if init.data else None
    if not mid:
        raise RuntimeError("x media initialize missing id")

    segment_b64 = base64.b64encode(raw).decode("utf-8")
    append_body = AppendUploadRequest.model_validate({"segment_index": 0, "media": segment_b64})
    client.media.append_upload(id=mid, body=append_body)
    fin = client.media.finalize_upload(id=mid)
    if fin.errors:
        raise RuntimeError(f"x media finalize failed: {fin.errors}")
    return str(mid)


def _http_error_detail(exc: BaseException) -> Any:
    if isinstance(exc, HTTPError) and exc.response is not None:
        try:
            return exc.response.json()
        except Exception:
            return exc.response.text
    return str(exc)


def _send_x_sync(text: str, image_path: str | None) -> Dict[str, Any]:
    """Post to X using official xdk (OAuth 2.0 preferred; OAuth 1.0a fallback)."""
    oauth2_token = _read_oauth2_user_token()
    client_id = _read_x_client_id()
    client_secret = _read_x_client_secret()
    redirect_uri = _read_x_redirect_uri()

    consumer_key, _ = _read_env_with_source("X_CONSUMER_KEY")
    consumer_secret, _ = _read_env_with_source("X_CONSUMER_SECRET")
    access_token_o1, _ = _read_env_with_source("X_ACCESS_TOKEN")
    access_token_secret_o1, _ = _read_env_with_source("X_ACCESS_TOKEN_SECRET")

    oauth2_ready = bool(oauth2_token and client_id and client_secret)
    oauth1_ready = bool(consumer_key and consumer_secret and access_token_o1 and access_token_secret_o1)

    x_credential_check = _credential_snapshot(
        [
            "X_CLIENT_ID",
            "CLIENT_ID",
            "X_CLIENT_SECRET",
            "CLIENT_SECRET",
            "X_REDIRECT_URI",
            "X_OAUTH2_TOKEN",
            "X_OAUTH2_ACCESS_TOKEN",
            "X_OAUTH2_REFRESH_TOKEN",
            "X_CONSUMER_KEY",
            "X_CONSUMER_SECRET",
            "X_ACCESS_TOKEN",
            "X_ACCESS_TOKEN_SECRET",
        ]
    )
    logger.info("x credential check: %s", x_credential_check)

    if not oauth2_ready and not oauth1_ready:
        return {
            "sent": False,
            "mode": "none",
            "note": "configure OAuth2 (X_CLIENT_ID, X_CLIENT_SECRET, X_OAUTH2_ACCESS_TOKEN or X_OAUTH2_TOKEN) "
            "or OAuth1 (X_CONSUMER_KEY, X_CONSUMER_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET)",
            "credential_check": x_credential_check,
        }

    client: XdkClient
    mode_tag: str

    if oauth2_ready:
        mode_tag = "oauth2+xdk"
        client = XdkClient(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            token=oauth2_token,
            scope=_X_OAUTH2_SCOPES,
        )
        if client.oauth2_auth and oauth2_token.get("refresh_token") and client.is_token_expired():
            client.refresh_token()
    else:
        mode_tag = "oauth1+xdk"
        oauth1 = OAuth1(
            api_key=consumer_key,
            api_secret=consumer_secret,
            callback=(os.getenv("X_OAUTH1_CALLBACK") or "oob").strip(),
            access_token=access_token_o1,
            access_token_secret=access_token_secret_o1,
        )
        client = XdkClient(auth=oauth1)

    body: CreateRequest
    mode = "text"
    try:
        if image_path and Path(image_path).is_file():
            if oauth2_ready:
                media_id = _upload_image_oauth2_v2(client, image_path)
            else:
                media_id = _upload_image_oauth1_v11(client, image_path)
            body = CreateRequest(text=text, media=CreateRequestMedia(media_ids=[media_id]))
            mode = "image+text"
        else:
            body = CreateRequest(text=text)

        resp = client.posts.create(body=body)
        payload = resp.model_dump() if hasattr(resp, "model_dump") else resp  # type: ignore[assignment]
        tweet_id: Any = None
        if isinstance(payload, dict):
            tweet_id = (payload.get("data") or {}).get("id")
        result: Dict[str, Any] = {
            "sent": True,
            "mode": mode,
            "payload": payload,
            "credential_check": x_credential_check,
            "auth_mode": mode_tag,
        }
        if tweet_id:
            result["url"] = f"https://x.com/i/status/{tweet_id}"
        return result

    except (HTTPError, requests.RequestException, ValueError, RuntimeError) as exc:
        status_code: Optional[int] = None
        if isinstance(exc, HTTPError) and exc.response is not None:
            status_code = exc.response.status_code
        detail = _http_error_detail(exc)
        auth_note = None
        if status_code == 401:
            auth_note = "x auth failed (likely credential/permission mismatch)"
        return {
            "sent": False,
            "mode": mode,
            "note": f"x post failed http={status_code}" if status_code else f"x post failed: {exc}",
            "auth_note": auth_note,
            "payload": detail if not isinstance(detail, str) else {"detail": detail},
            "credential_check": x_credential_check,
            "auth_mode": mode_tag,
        }


async def _send_telegram(text: str, image_path: str | None) -> Dict[str, Any]:
    token, token_source = _read_env_with_source("TG_BOT_TOKEN")
    if not token:
        token, token_source = _read_env_with_source("TELEGRAM_BOT_TOKEN")

    chat_id, chat_id_source = _read_env_with_source("TG_CHAT_ID")
    if not chat_id:
        chat_id, chat_id_source = _read_env_with_source("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        return {
            "sent": False,
            "mode": "none",
            "note": "missing TG_BOT_TOKEN or TG_CHAT_ID",
            "credential_check": {
                "token_source": token_source,
                "chat_id_source": chat_id_source,
                "token_present": bool(token),
                "chat_id_present": bool(chat_id),
            },
        }

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


async def _send_x(text: str, image_path: str | None) -> Dict[str, Any]:
    """X API via xdk (blocking client runs in a thread pool)."""
    return await asyncio.to_thread(_send_x_sync, text, image_path)


async def _send_square(text: str) -> Dict[str, Any]:
    api_key, api_key_source = _read_env_with_source("SQUARE_OPENAPI_KEY")
    if not api_key:
        return {
            "sent": False,
            "mode": "none",
            "note": "missing SQUARE_OPENAPI_KEY",
            "credential_check": {
                "SQUARE_OPENAPI_KEY": {
                    "present": bool(api_key),
                    "source": api_key_source,
                    "length": len(api_key),
                }
            },
        }

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
