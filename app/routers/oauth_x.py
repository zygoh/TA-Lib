"""
X Developer Platform OAuth 2.0（PKCE）浏览器回调。

与 distribution_service 使用的 scope / 环境变量一致：
  X_CLIENT_ID、X_CLIENT_SECRET、X_REDIRECT_URI（须与本路由对外 URL 完全一致）。

对外示例（反代把 https://do2ge.com/tail 指到本服务时）：
  回调：https://do2ge.com/tail/oauth2/callback
  发起授权：浏览器打开 https://do2ge.com/tail/oauth2/start

若网关去掉 /tail 前缀再转发到 uvicorn，亦可使用 https://do2ge.com/oauth2/start。
"""

from __future__ import annotations

import html
import json
import logging
import secrets
import time
from typing import Dict, Tuple

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from xdk.oauth2_auth import OAuth2PKCEAuth

from app.services.distribution_service import (
    _read_x_client_id,
    _read_x_client_secret,
    _read_x_redirect_uri,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["X OAuth2"])

_X_OAUTH2_SCOPES = ["tweet.read", "tweet.write", "users.read", "media.write", "offline.access"]

# state -> (code_verifier, expiry_unix)；单 worker 内存存储；多 worker 需换 Redis 等
_PKCE_TTL_SEC = 600
_pkce_store: Dict[str, Tuple[str, float]] = {}


def _prune_pkce_store() -> None:
    now = time.time()
    dead = [s for s, (_, exp) in _pkce_store.items() if exp < now]
    for s in dead:
        del _pkce_store[s]


def _oauth_config_or_503() -> Tuple[str, str, str]:
    cid = _read_x_client_id().strip()
    secret = _read_x_client_secret().strip()
    redir = _read_x_redirect_uri().strip()
    if not cid or not secret or not redir:
        raise HTTPException(
            status_code=503,
            detail="缺少 X_CLIENT_ID / X_CLIENT_SECRET / X_REDIRECT_URI，无法启动 OAuth2。",
        )
    return cid, secret, redir


@router.get("/oauth2/start", summary="跳转 X 授权页（PKCE）")
async def x_oauth2_start() -> RedirectResponse:
    _prune_pkce_store()
    client_id, client_secret, redirect_uri = _oauth_config_or_503()
    auth = OAuth2PKCEAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=_X_OAUTH2_SCOPES,
    )
    state = secrets.token_urlsafe(32)
    url = auth.get_authorization_url(state=state)
    if not auth.code_verifier:
        raise HTTPException(status_code=500, detail="PKCE 未生成 code_verifier")
    _pkce_store[state] = (auth.code_verifier, time.time() + _PKCE_TTL_SEC)
    logger.info("x oauth2 start: redirecting to X authorize (state len=%d)", len(state))
    return RedirectResponse(url, status_code=302)


@router.get("/oauth2/callback", summary="X 授权完成后的回调")
async def x_oauth2_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
) -> HTMLResponse:
    if error:
        msg = error_description or error
        return HTMLResponse(
            f"<html><body><h1>授权失败</h1><pre>{html.escape(msg)}</pre></body></html>",
            status_code=400,
        )
    if not code or not state:
        raise HTTPException(status_code=400, detail="缺少 code 或 state 参数")

    _prune_pkce_store()
    item = _pkce_store.pop(state, None)
    if not item or time.time() > item[1]:
        raise HTTPException(status_code=400, detail="state 无效或已过期，请重新从 /oauth2/start 发起")
    code_verifier, _ = item

    client_id, client_secret, redirect_uri = _oauth_config_or_503()
    auth = OAuth2PKCEAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=_X_OAUTH2_SCOPES,
    )
    try:
        token = auth.exchange_code(code, code_verifier=code_verifier)
    except ValueError as exc:
        logger.warning("x oauth2 token exchange failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    token_json = json.dumps(token, ensure_ascii=False, separators=(",", ":"))
    safe_json = html.escape(token_json)
    logger.info("x oauth2 callback: token exchange ok")

    body = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"/><title>X OAuth2 成功</title></head>
<body>
<h1>X OAuth2 授权成功</h1>
<p>请将下面<strong>单行 JSON</strong>复制到 <code>TA-Lib/.env</code> 的 <code>X_OAUTH2_TOKEN=</code> 之后（不要换行）：</p>
<p><textarea readonly rows="8" cols="120">{safe_json}</textarea></p>
<p>或拆分填写：</p>
<ul>
<li><code>X_OAUTH2_ACCESS_TOKEN</code> = access_token</li>
<li><code>X_OAUTH2_REFRESH_TOKEN</code> = refresh_token（若有）</li>
</ul>
<p><a href="/docs">返回 API 文档</a></p>
</body>
</html>"""
    return HTMLResponse(body, status_code=200)
