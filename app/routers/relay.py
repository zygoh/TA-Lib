#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSocket 和 HTTP 代理转发（使用 curl_cffi 模拟真实浏览器 TLS 指纹）
- WebSocket: ws://host:port/ws/relay/{path:path}
- HTTP: http://host:port/http/relay/{path:path}
"""

from fastapi import APIRouter, WebSocket, Request
from fastapi.responses import Response
import asyncio
import websockets
import ssl
from curl_cffi.requests import AsyncSession
from typing import Optional

router = APIRouter()

# 远端地址配置
WS_REMOTE_BASE = "wss://fstream.binance.com"
HTTP_REMOTE_BASE = "https://fapi.binance.com"

# 模拟浏览器指纹（推荐使用较新的 Chrome 版本）
IMPERSONATE_BROWSER = "chrome120"  # 可选: chrome110, chrome116, chrome119, edge110, safari15_3 等

# 模拟正常浏览器的请求头
BROWSER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
}

# 不应该转发的请求头（会暴露客户端身份）
SKIP_HEADERS = {
    'host', 'origin', 'referer', 'x-real-ip', 'x-forwarded-for', 
    'x-forwarded-proto', 'x-forwarded-host', 'forwarded',
    'cf-connecting-ip', 'cf-ipcountry', 'cf-ray', 'cf-visitor',
    'true-client-ip', 'x-client-ip'
}


async def relay(client_ws: WebSocket, remote_ws):
    """双向裸转发，任一端关闭则两端一起关闭"""
    
    async def client_to_remote():
        """客户端 -> 远端"""
        try:
            while True:
                data = await client_ws.receive_text()
                await remote_ws.send(data)
        except Exception:
            pass
    
    async def remote_to_client():
        """远端 -> 客户端"""
        try:
            async for msg in remote_ws:
                await client_ws.send_text(msg)
        except Exception:
            pass
    
    # 任一方向出错/断开，都会导致 gather 返回
    await asyncio.gather(
        client_to_remote(),
        remote_to_client(),
        return_exceptions=True
    )


async def websocket_proxy(websocket: WebSocket, path: str):
    """WebSocket 代理核心逻辑（支持自动重连）"""
    target_url = f"{WS_REMOTE_BASE}/{path}"
    if websocket.query_params:
        from urllib.parse import urlencode
        query = urlencode(websocket.query_params)
        target_url += f"?{query}"
    
    print(f"[WS] Client connecting to: {target_url}")
    
    accepted = False
    try:
        await websocket.accept()
        accepted = True
    except Exception as e:
        print(f"[WS] Failed to accept connection: {e}")
        return
    
    try:
        # 配置 SSL 上下文，模拟浏览器 TLS 指纹
        ssl_context = ssl.create_default_context()
        ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        
        headers = {
            'User-Agent': BROWSER_HEADERS['User-Agent'],
            'Origin': 'https://www.binance.com',
            'Host': 'fstream.binance.com',
            'Accept-Encoding': 'gzip, deflate, br',
        }
        
        async with websockets.connect(
            target_url,
            additional_headers=headers,
            ssl=ssl_context,
            ping_interval=20,
            ping_timeout=10
        ) as remote_ws:
            print(f"[WS] Connected to remote")
            await relay(websocket, remote_ws)
    except websockets.exceptions.WebSocketException as e:
        print(f"[WS] WebSocket error: {e}")
    except Exception as e:
        print(f"[WS] Unexpected error: {e}")
    finally:
        # 确保连接正常关闭，不影响后续重连
        if accepted:
            try:
                await websocket.close()
            except Exception:
                # 连接可能已经关闭，忽略异常
                pass
        print("[WS] Disconnected (ready for reconnection)")

async def _curl_request_async(
    method: str,
    url: str,
    headers: dict,
    data: Optional[bytes] = None,
    timeout: int = 30
) -> tuple[int, bytes, dict]:
    """使用 curl_cffi 异步版本发送请求"""
    try:
        async with AsyncSession() as session:
            resp = await session.request(
                method=method,
                url=url,
                headers=headers,
                content=data,
                timeout=timeout,
                allow_redirects=True,
                impersonate=IMPERSONATE_BROWSER
            )
            
            content = resp.content
            response_headers = {
                k: v for k, v in resp.headers.items()
                if k.lower() not in ['transfer-encoding', 'connection', 'content-encoding']
            }
            return resp.status_code, content, response_headers
    except Exception as e:
        print(f"[HTTP] curl_cffi request error: {e}")
        raise


async def http_relay(request: Request, path: str):
    """HTTP 代理转发端点"""
    target_url = f"{HTTP_REMOTE_BASE}/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"
    
    print(f"[HTTP] {request.method} {target_url}")
    
    try:
        headers = BROWSER_HEADERS.copy()
        
        for k, v in request.headers.items():
            k_lower = k.lower()
            if k_lower not in SKIP_HEADERS:
                if k_lower in ['content-type', 'authorization', 'x-mbx-apikey', 'accept']:
                    headers[k] = v
        
        # 添加 Referer 和 Origin 头，增强浏览器真实性
        headers['Referer'] = 'https://www.binance.com/'
        headers['Origin'] = 'https://www.binance.com'
        
        body = await request.body()
        
        status_code, content, response_headers = await _curl_request_async(
            method=request.method,
            url=target_url,
            headers=headers,
            data=body if body else None
        )
        
        print(f"[HTTP] Response {status_code}, Content-Length: {len(content)}")
        
        return Response(
            content=content,
            status_code=status_code,
            headers=response_headers
        )
    
    except Exception as e:
        print(f"[HTTP] Error: {e}")
        return Response(
            content=f"Relay error: {str(e)}",
            status_code=502,
            media_type="text/plain"
        )


@router.websocket("/ws/relay/{path:path}")
async def websocket_relay_endpoint(websocket: WebSocket, path: str):
    await websocket_proxy(websocket, path)


@router.api_route("/http/relay/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def http_relay_endpoint(request: Request, path: str):
    """HTTP 代理转发端点"""
    return await http_relay(request, path)
