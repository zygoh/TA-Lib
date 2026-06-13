"""LIB_API_KEY 网关鉴权：设置环境变量后，除健康检查与 OAuth 回调外均需密钥。"""

from __future__ import annotations

import logging
import os
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

_EXEMPT_EXACT = frozenset({"/", "/health", "/openapi.json", "/favicon.ico"})
_EXEMPT_PREFIXES = ("/docs", "/redoc")


def get_lib_api_key() -> str:
    return (os.getenv("LIB_API_KEY") or "").strip()


def _extract_api_key(request: Request) -> str:
    header_key = (request.headers.get("x-api-key") or "").strip()
    if header_key:
        return header_key
    auth = (request.headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def is_api_key_exempt_path(path: str) -> bool:
    normalized = (path or "/").rstrip("/") or "/"
    if normalized in _EXEMPT_EXACT:
        return True
    for prefix in _EXEMPT_PREFIXES:
        if normalized == prefix or normalized.startswith(f"{prefix}/"):
            return True
    if "/oauth2/" in normalized:
        return True
    return False


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        expected = get_lib_api_key()
        if not expected:
            return await call_next(request)

        if is_api_key_exempt_path(request.url.path):
            return await call_next(request)

        provided = _extract_api_key(request)
        if provided != expected:
            logger.warning("API key rejected path=%s", request.url.path)
            return JSONResponse(
                status_code=401,
                content={"detail": "invalid or missing API key"},
            )
        return await call_next(request)
