from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import requests
from requests import HTTPError
from xdk import Client as XdkClient
from xdk.media.models import AppendUploadRequest, InitializeUploadRequest
from xdk.oauth1_auth import OAuth1
from xdk.posts.models import CreateRequest, CreateRequestMedia

logger = logging.getLogger(__name__)

_TG_API_BASE = "https://api.telegram.org"
_X_MEDIA_UPLOAD_V11 = "https://upload.twitter.com/1.1/media/upload.json"
_SQUARE_BASE_V1 = "https://www.binance.com/bapi/composite/v1/public/pgc/openApi"
_SQUARE_BASE_V2 = "https://www.binance.com/bapi/composite/v2/public/pgc/openApi"
_SQUARE_POST_URL = f"{_SQUARE_BASE_V1}/content/add"
_SQUARE_POLL_INTERVAL_SEC = 3
_SQUARE_MAX_POLL_RETRIES = 10
_DOTENV_CACHE: Dict[str, str] | None = None

_X_OAUTH2_SCOPES = ["tweet.read", "tweet.write", "users.read", "media.write", "offline.access"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _btc_last_post_id_path() -> Path:
    """BTC X 引用链状态文件（硬编码路径，服务自动读写）。"""
    return _repo_root() / "data" / "x_btc_last_post_id.txt"


def _repo_env_path() -> Path:
    return _repo_root() / ".env"


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


def _upsert_repo_env(name: str, value: str) -> None:
    env_path = _repo_env_path()
    lines: List[str] = []
    if env_path.is_file():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    prefix = f"{name}="
    replaced = False
    new_lines: List[str] = []
    for line in lines:
        if line.startswith(prefix):
            new_lines.append(f"{name}={value}")
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.append(f"{name}={value}")
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _persist_oauth2_token(token: Dict[str, Any], reason: str) -> None:
    access_token = str(token.get("access_token") or "").strip()
    if not access_token:
        logger.warning("跳过持久化 OAuth2 令牌：缺少 access_token（原因=%s）", reason)
        return

    normalized: Dict[str, Any] = {
        "access_token": access_token,
        "token_type": token.get("token_type"),
        "expires_in": token.get("expires_in"),
        "refresh_token": token.get("refresh_token"),
        "scope": token.get("scope"),
        "expires_at": token.get("expires_at"),
    }
    token_json = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))

    os.environ["X_OAUTH2_TOKEN"] = token_json
    _upsert_repo_env("X_OAUTH2_TOKEN", token_json)

    refresh_token = str(token.get("refresh_token") or "").strip()
    os.environ["X_OAUTH2_ACCESS_TOKEN"] = access_token
    _upsert_repo_env("X_OAUTH2_ACCESS_TOKEN", access_token)
    if refresh_token:
        os.environ["X_OAUTH2_REFRESH_TOKEN"] = refresh_token
        _upsert_repo_env("X_OAUTH2_REFRESH_TOKEN", refresh_token)

    global _DOTENV_CACHE
    _DOTENV_CACHE = None
    logger.info("OAuth2 令牌已写入 .env 与进程环境（原因=%s）", reason)


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
            logger.warning("已配置 X_OAUTH2_TOKEN，但内容不是合法 JSON")
            return None

    access, _ = _read_env_with_source("X_OAUTH2_ACCESS_TOKEN")
    if not access:
        return None
    token: Dict[str, Any] = {"access_token": access.strip()}
    refresh, _ = _read_env_with_source("X_OAUTH2_REFRESH_TOKEN")
    if refresh:
        token["refresh_token"] = refresh.strip()
    return token


def _is_btc_symbol(symbol_usdt: str) -> bool:
    """Hard-coded: BTC / BTCUSDT posts use the BTC-only X quote chain."""
    s = (symbol_usdt or "").strip().upper()
    if not s:
        return False
    base = s[:-4] if s.endswith("USDT") else s
    return base == "BTC"


def _read_x_last_post_id() -> str:
    value, _ = _read_env_with_source("X_LAST_POST_ID")
    return value.strip()


def _read_x_btc_last_post_id() -> str:
    path = _btc_last_post_id_path()
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning("读取 BTC 上一条推文 ID 失败 path=%s 错误=%s", path, exc)
        return ""


def _read_x_status_url_handle() -> str:
    """Optional @handle (without @) for permalinks like https://x.com/handle/status/id."""
    for key in ("X_USERNAME", "X_SCREEN_NAME"):
        raw, _ = _read_env_with_source(key)
        h = raw.strip().lstrip("@")
        if h and re.fullmatch(r"[A-Za-z0-9_]{1,15}", h):
            return h
    return ""


def _persist_x_last_post_id(post_id: str) -> None:
    clean_id = (post_id or "").strip()
    if not clean_id:
        return
    os.environ["X_LAST_POST_ID"] = clean_id
    _upsert_repo_env("X_LAST_POST_ID", clean_id)
    global _DOTENV_CACHE
    _DOTENV_CACHE = None


def _persist_x_btc_last_post_id(post_id: str) -> None:
    clean_id = (post_id or "").strip()
    if not clean_id:
        return
    path = _btc_last_post_id_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(clean_id + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("保存 BTC 上一条推文 ID 失败 path=%s 错误=%s", path, exc)


def _x_status_permalink(tweet_id: str) -> str:
    tid = tweet_id.strip()
    handle = _read_x_status_url_handle()
    if handle:
        return f"https://x.com/{handle}/status/{tid}"
    return f"https://x.com/i/status/{tid}"


def _guess_image_mime(image_filename: str | None, image_content_type: str | None = None) -> str:
    if image_content_type and image_content_type.startswith("image/"):
        return image_content_type
    mime, _ = mimetypes.guess_type(image_filename or "")
    if mime and mime.startswith("image/"):
        return mime
    return "image/png"


def _normalize_image_filename(image_filename: str | None, image_content_type: str | None = None) -> str:
    filename = Path((image_filename or "").strip() or "upload.png").name
    if "." not in filename:
        guessed_ext = mimetypes.guess_extension(_guess_image_mime(filename, image_content_type)) or ".png"
        filename = f"{filename}{guessed_ext}"
    return filename


def _append_note(notes: List[str], note: str | None) -> None:
    clean_note = (note or "").strip()
    if clean_note and clean_note not in notes:
        notes.append(clean_note)


def _upload_image_oauth1_v11(client: XdkClient, image_bytes: bytes) -> str:
    """Legacy v1.1 media upload (OAuth 1.0a user context), signed via XDK OAuth1."""
    if not client.auth:
        raise RuntimeError("x oauth1 auth missing on client")
    media_b64 = base64.b64encode(image_bytes).decode("utf-8")
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


def _upload_image_oauth2_v2(client: XdkClient, image_bytes: bytes, image_content_type: str) -> str:
    """Media Upload v2 with OAuth 2.0 user token (Bearer), aligned with official samples."""
    total = len(image_bytes)
    init_body = InitializeUploadRequest(
        total_bytes=total,
        media_type=image_content_type,
        media_category="tweet_image",
    )
    init = client.media.initialize_upload(body=init_body)
    if init.errors:
        raise RuntimeError(f"x media initialize failed: {init.errors}")
    mid = init.data.id if init.data else None
    if not mid:
        raise RuntimeError("x media initialize missing id")

    segment_b64 = base64.b64encode(image_bytes).decode("utf-8")
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


def _send_x_sync(
    text: str,
    image_bytes: bytes | None,
    image_filename: str | None,
    image_content_type: str | None,
    reply_to_previous: bool = False,
    quote_tweet_id: str | None = None,
    persist_btc_chain: bool = False,
) -> Dict[str, Any]:
    """Post to X using official xdk (OAuth 2.0 preferred; OAuth 1.0a fallback).

    When reply_to_previous is True, quote_tweet_id (if not None) is used as quote_tweet_id;
    otherwise X_LAST_POST_ID is read. BTC distribute flow passes the prior BTC id explicitly.
    """
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
    logger.info("X 凭证检查：%s", x_credential_check)

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
    use_oauth2 = False

    if oauth2_ready:
        try:
            mode_tag = "oauth2+xdk"
            client = XdkClient(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                token=oauth2_token,
                scope=_X_OAUTH2_SCOPES,
            )
            use_oauth2 = True
            if client.oauth2_auth and oauth2_token.get("refresh_token") and client.is_token_expired():
                refreshed = client.refresh_token()
                if isinstance(refreshed, dict):
                    _persist_oauth2_token(refreshed, reason="refresh")
                elif isinstance(client.token, dict):
                    _persist_oauth2_token(client.token, reason="refresh")
        except Exception as exc:  # noqa: BLE001
            logger.warning("X OAuth2 初始化/刷新失败，回退 OAuth1=%s，错误=%s", oauth1_ready, exc)
            use_oauth2 = False
            if not oauth1_ready:
                return {
                    "sent": False,
                    "mode": "none",
                    "note": f"x oauth2 refresh/init failed and oauth1 unavailable: {exc}",
                    "credential_check": x_credential_check,
                    "auth_mode": "oauth2+xdk",
                    "payload": {"detail": str(exc)},
                }

    if not use_oauth2:
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
        quoted_previous_id = ""
        if reply_to_previous:
            if quote_tweet_id is not None:
                quoted_previous_id = (quote_tweet_id or "").strip()
            else:
                quoted_previous_id = _read_x_last_post_id()

        body_payload: Dict[str, Any] = {"text": text}
        if quoted_previous_id:
            body_payload["quote_tweet_id"] = quoted_previous_id

        if image_bytes:
            normalized_filename = _normalize_image_filename(image_filename, image_content_type)
            normalized_content_type = _guess_image_mime(normalized_filename, image_content_type)
            if use_oauth2:
                media_id = _upload_image_oauth2_v2(client, image_bytes, normalized_content_type)
            else:
                media_id = _upload_image_oauth1_v11(client, image_bytes)
            body_payload["media"] = CreateRequestMedia(media_ids=[media_id]).model_dump(exclude_none=True)
            mode = "image+text"

        body = CreateRequest.model_validate(body_payload)

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
            tweet_id_str = str(tweet_id)
            _persist_x_last_post_id(tweet_id_str)
            if persist_btc_chain:
                _persist_x_btc_last_post_id(tweet_id_str)
            result["id"] = tweet_id_str
            result["url"] = _x_status_permalink(str(tweet_id))
        if quoted_previous_id:
            result["quoted_previous_tweet_id"] = quoted_previous_id
            result["quote_tweet_id"] = quoted_previous_id
            result["previous_tweet_url"] = _x_status_permalink(quoted_previous_id)
        if reply_to_previous:
            result["reply_to_previous"] = True
            result["quote_previous"] = bool(quoted_previous_id)
            if not quoted_previous_id:
                if quote_tweet_id is not None:
                    result["quote_previous_note"] = "no previous BTC post id; posted without quote tweet"
                else:
                    result["quote_previous_note"] = "X_LAST_POST_ID missing; posted without quote tweet"
        if persist_btc_chain:
            result["btc_quote_chain"] = True
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
    except Exception as exc:  # noqa: BLE001
        return {
            "sent": False,
            "mode": mode,
            "note": f"x post failed: {exc}",
            "payload": {"detail": str(exc)},
            "credential_check": x_credential_check,
            "auth_mode": mode_tag,
        }


async def _send_telegram(
    text: str,
    image_bytes: bytes | None,
    image_filename: str | None,
    image_content_type: str | None,
) -> Dict[str, Any]:
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
        if image_bytes:
            normalized_filename = _normalize_image_filename(image_filename, image_content_type)
            normalized_content_type = _guess_image_mime(normalized_filename, image_content_type)
            form = aiohttp.FormData()
            form.add_field("chat_id", chat_id)
            form.add_field("caption", text)
            form.add_field(
                "photo",
                image_bytes,
                filename=normalized_filename,
                content_type=normalized_content_type,
            )
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


async def _send_x(
    text: str,
    image_bytes: bytes | None,
    image_filename: str | None,
    image_content_type: str | None,
    reply_to_previous: bool = False,
    quote_tweet_id: str | None = None,
    persist_btc_chain: bool = False,
) -> Dict[str, Any]:
    """X API via xdk (blocking client runs in a thread pool)."""
    return await asyncio.to_thread(
        _send_x_sync,
        text,
        image_bytes,
        image_filename,
        image_content_type,
        reply_to_previous,
        quote_tweet_id,
        persist_btc_chain,
    )


def _read_square_api_key() -> Tuple[str, str]:
    for name in ("SQUARE_OPENAPI_KEY", "BINANCE_SQUARE_OPENAPI_KEY"):
        value, source = _read_env_with_source(name)
        if value:
            return value, source
    return "", "missing"


def _square_credential_snapshot() -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    for name in ("SQUARE_OPENAPI_KEY", "BINANCE_SQUARE_OPENAPI_KEY"):
        value, source = _read_env_with_source(name)
        snapshot[name] = {"present": bool(value), "source": source, "length": len(value)}
    return snapshot


def _square_headers(api_key: str) -> Dict[str, str]:
    return {
        "X-Square-OpenAPI-Key": api_key,
        "Content-Type": "application/json; charset=utf-8",
        "clienttype": "binanceSkill",
    }


async def _square_api_json(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    endpoint: str,
    api_key: str,
    body: Dict[str, Any],
) -> Tuple[int, Any]:
    async with session.post(
        f"{base_url}{endpoint}",
        json=body,
        headers=_square_headers(api_key),
    ) as resp:
        raw = await resp.text()
        if endpoint == "/content/add" and resp.status == 504:
            return resp.status, {"code": "000000", "data": {"id": None, "shareLink": None}}
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            raise RuntimeError(f"square api non-json response {resp.status}: {raw[:200]}") from None
        if resp.status >= 400:
            raise RuntimeError(f"square api http {resp.status}: {payload}")
        code = payload.get("code")
        if code != "000000":
            message = payload.get("message") or payload
            raise RuntimeError(f"square api error [{code}]: {message}")
        return resp.status, payload.get("data") or {}


async def _upload_square_image(
    session: aiohttp.ClientSession,
    api_key: str,
    image_bytes: bytes,
    image_filename: str | None,
    image_content_type: str | None,
) -> str:
    normalized_filename = _normalize_image_filename(image_filename, image_content_type)
    normalized_content_type = _guess_image_mime(normalized_filename, image_content_type)

    _, presign_payload = await _square_api_json(
        session,
        base_url=_SQUARE_BASE_V2,
        endpoint="/image/presignedUrl",
        api_key=api_key,
        body={"imageName": normalized_filename},
    )
    presigned_url = presign_payload.get("presignedUrl")
    file_ticket = presign_payload.get("fileTicket")
    if not presigned_url or not file_ticket:
        raise RuntimeError(f"square presignedUrl missing fields: {presign_payload}")

    async with session.put(
        presigned_url,
        data=image_bytes,
        headers={"Content-Type": normalized_content_type},
    ) as put_resp:
        if put_resp.status >= 400:
            detail = await put_resp.text()
            raise RuntimeError(f"square s3 upload failed {put_resp.status}: {detail[:200]}")

    for attempt in range(_SQUARE_MAX_POLL_RETRIES):
        _, status_payload = await _square_api_json(
            session,
            base_url=_SQUARE_BASE_V2,
            endpoint="/image/imageStatus",
            api_key=api_key,
            body={"fileTicket": file_ticket},
        )
        status = status_payload.get("status")
        if status == 1:
            image_url = status_payload.get("imageUrl")
            if not image_url:
                raise RuntimeError(f"square image ready but imageUrl missing: {status_payload}")
            return str(image_url)
        if status == 2:
            raise RuntimeError(f"square image processing failed: {status_payload.get('failedReason')}")
        if attempt + 1 < _SQUARE_MAX_POLL_RETRIES:
            await asyncio.sleep(_SQUARE_POLL_INTERVAL_SEC)

    raise RuntimeError(f"square image poll timed out after {_SQUARE_MAX_POLL_RETRIES} retries")


    raise RuntimeError(f"square image poll timed out after {_SQUARE_MAX_POLL_RETRIES} retries")


def _square_publish_result(data: Dict[str, Any], *, mode: str, raw_payload: Any = None) -> Dict[str, Any]:
    post_id = data.get("id")
    share_link = data.get("shareLink")
    result: Dict[str, Any] = {"sent": True, "mode": mode}
    if raw_payload is not None:
        result["payload"] = raw_payload
    if post_id:
        result["id"] = post_id
        result["url"] = share_link or f"https://www.binance.com/en/square/post/{post_id}"
    elif share_link:
        result["url"] = share_link
        result["note"] = "square success but id missing"
    else:
        result["note"] = "square submitted; id/link unavailable (e.g. HTTP 504 after submit)"
    return result


async def _send_square(
    text: str,
    image_bytes: bytes | None = None,
    image_filename: str | None = None,
    image_content_type: str | None = None,
) -> Dict[str, Any]:
    api_key, api_key_source = _read_square_api_key()
    if not api_key:
        credential_check = _square_credential_snapshot()
        return {
            "sent": False,
            "mode": "none",
            "note": "missing SQUARE_OPENAPI_KEY or BINANCE_SQUARE_OPENAPI_KEY",
            "credential_check": credential_check,
        }

    mode = "text"
    publish_body: Dict[str, Any] = {"contentType": 1, "bodyTextOnly": text}
    timeout = aiohttp.ClientTimeout(total=120)
    try:
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            if image_bytes:
                image_url = await _upload_square_image(
                    session,
                    api_key,
                    image_bytes,
                    image_filename,
                    image_content_type,
                )
                publish_body["imageList"] = [image_url]
                mode = "image+text"

            async with session.post(
                _SQUARE_POST_URL,
                json=publish_body,
                headers=_square_headers(api_key),
            ) as resp:
                raw = await resp.text()
                if resp.status == 504:
                    return _square_publish_result(
                        {"id": None, "shareLink": None},
                        mode=mode,
                        raw_payload={"http_status": 504, "body": raw[:500]},
                    )
                try:
                    payload = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    return {
                        "sent": False,
                        "mode": mode,
                        "note": f"square publish non-json response {resp.status}",
                        "payload": {"detail": raw[:500]},
                    }
                if resp.status >= 400:
                    return {
                        "sent": False,
                        "mode": mode,
                        "note": f"square http failed {resp.status}",
                        "payload": payload,
                    }
                code = payload.get("code")
                if code != "000000":
                    return {
                        "sent": False,
                        "mode": mode,
                        "note": "square business failed",
                        "payload": payload,
                    }
                data = payload.get("data") or {}
                return _square_publish_result(data, mode=mode, raw_payload=payload)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Square 渠道发送失败：%s", exc)
        return {
            "sent": False,
            "mode": mode,
            "note": f"square failed: {exc}",
            "credential_check": {
                "resolved_key_source": api_key_source,
            },
        }


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
    image_bytes: bytes | None = None,
    image_filename: str | None = None,
    image_content_type: str | None = None,
    x_reply_to_previous: bool = False,
) -> Dict[str, Any]:
    if not text.strip():
        raise ValueError("text 不能为空")

    has_image = bool(image_bytes)
    is_btc = _is_btc_symbol(symbol_usdt)
    x_reply = bool(x_reply_to_previous or is_btc)
    btc_quote_id = _read_x_btc_last_post_id() if is_btc else None
    logger.info(
        "分发开始 symbol=%s 正文长度=%d 含图=%s BTC 引用链=%s",
        symbol_usdt,
        len(text),
        has_image,
        is_btc,
    )
    telegram_result, x_result, square_result = await asyncio.gather(
        _send_telegram(text, image_bytes, image_filename, image_content_type),
        _send_x(
            text,
            image_bytes,
            image_filename,
            image_content_type,
            x_reply,
            quote_tweet_id=btc_quote_id if is_btc else None,
            persist_btc_chain=is_btc,
        ),
        _send_square(text, image_bytes, image_filename, image_content_type),
        return_exceptions=True,
    )

    if isinstance(telegram_result, Exception):
        logger.exception("Telegram 渠道异常：%s", telegram_result)
        telegram_result = {"sent": False, "mode": "none", "note": f"telegram exception: {telegram_result}"}
    if isinstance(x_result, Exception):
        logger.exception("X 渠道异常：%s", x_result)
        x_result = {"sent": False, "mode": "none", "note": f"x exception: {x_result}"}
    if isinstance(square_result, Exception):
        logger.exception("Square 渠道异常：%s", square_result)
        square_result = {"sent": False, "mode": "none", "note": f"square exception: {square_result}"}

    telegram_sent = bool(telegram_result.get("sent"))
    x_sent = bool(x_result.get("sent"))
    square_sent = bool(square_result.get("sent"))
    status = _derive_status(telegram_sent, x_sent, square_sent)

    notes: List[str] = []
    if not has_image:
        _append_note(notes, "image not provided; channels sent as text-only where applicable")
    for channel_name, channel_result in (
        ("telegram", telegram_result),
        ("x", x_result),
        ("square", square_result),
    ):
        if not channel_result.get("sent"):
            _append_note(notes, f"{channel_name}: {channel_result.get('note') or 'failed'}")
    if status == "partial":
        _append_note(notes, "partial success")
    elif status == "failed":
        _append_note(notes, "all channels failed")

    result = {
        "status": status,
        "symbol": symbol_usdt,
        "x_btc_quote_chain": is_btc,
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
        "分发结束 status=%s symbol=%s Telegram=%s(%s) X=%s(%s) Square=%s(%s)",
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
        logger.info("分发备注 symbol=%s 备注=%s", symbol_usdt, notes)
    return result
