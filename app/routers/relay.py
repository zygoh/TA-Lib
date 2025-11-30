#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSocket 和 HTTP 代理转发（使用 curl_cffi 模拟真实浏览器 TLS 指纹）
- Binance WebSocket: ws://host:port/ws/relay/{path:path}
- Binance HTTP: http://host:port/http/relay/{path:path}
- OKX WebSocket: ws://host:port/ws/relay/okx/{path:path}
- OKX HTTP: http://host:port/http/relay/okx/{path:path}
"""

from fastapi import APIRouter, WebSocket, Request
from fastapi.responses import Response
import asyncio
import websockets
import ssl
from curl_cffi.requests import AsyncSession
from typing import Optional

router = APIRouter()

# 交易所配置
EXCHANGE_CONFIG = {
    'binance': {
        'ws_base': 'wss://fstream.binance.com',
        'http_base': 'https://fapi.binance.com',
        'origin': 'https://www.binance.com',
        'host_ws': 'fstream.binance.com',
        'referer': 'https://www.binance.com/',
    },
    'okx': {
        'ws_base': 'wss://ws.okx.com:8443',
        'http_base': 'https://www.okx.com',
        'origin': 'https://www.okx.com',
        'host_ws': 'ws.okx.com',
        'referer': 'https://www.okx.com/',
    }
}

# 默认使用 Binance 配置（向后兼容）
WS_REMOTE_BASE = EXCHANGE_CONFIG['binance']['ws_base']
HTTP_REMOTE_BASE = EXCHANGE_CONFIG['binance']['http_base']

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


async def websocket_proxy(websocket: WebSocket, path: str, exchange: str = 'binance'):
    """WebSocket 代理核心逻辑（支持自动重连）"""
    config = EXCHANGE_CONFIG.get(exchange, EXCHANGE_CONFIG['binance'])
    target_url = f"{config['ws_base']}/{path}"
    if websocket.query_params:
        from urllib.parse import urlencode
        query = urlencode(websocket.query_params)
        target_url += f"?{query}"
    
    print(f"[WS-{exchange.upper()}] Client connecting to: {target_url}")
    
    accepted = False
    try:
        await websocket.accept()
        accepted = True
    except Exception as e:
        print(f"[WS-{exchange.upper()}] Failed to accept connection: {e}")
        return
    
    try:
        # 配置 SSL 上下文，模拟浏览器 TLS 指纹
        ssl_context = ssl.create_default_context()
        ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        
        headers = {
            'User-Agent': BROWSER_HEADERS['User-Agent'],
            'Origin': config['origin'],
            'Host': config['host_ws'],
            'Accept-Encoding': 'gzip, deflate, br',
        }
        
        async with websockets.connect(
            target_url,
            additional_headers=headers,
            ssl=ssl_context,
            ping_interval=20,
            ping_timeout=10
        ) as remote_ws:
            print(f"[WS-{exchange.upper()}] Connected to remote")
            await relay(websocket, remote_ws)
    except websockets.exceptions.WebSocketException as e:
        print(f"[WS-{exchange.upper()}] WebSocket error: {e}")
    except Exception as e:
        print(f"[WS-{exchange.upper()}] Unexpected error: {e}")
    finally:
        # 确保连接正常关闭，不影响后续重连
        if accepted:
            try:
                await websocket.close()
            except Exception:
                # 连接可能已经关闭，忽略异常
                pass
        print(f"[WS-{exchange.upper()}] Disconnected (ready for reconnection)")

async def _curl_request_async(
    method: str,
    url: str,
    headers: dict,
    data: Optional[bytes] = None,
    timeout: int = 30,
    impersonate: Optional[str] = None
) -> tuple[int, bytes, dict]:
    """使用 curl_cffi 异步版本发送请求
    
    Args:
        impersonate: 浏览器指纹伪装，None 表示使用标准 TLS
                    'chrome120' 等表示伪装成对应浏览器
    """
    try:
        async with AsyncSession() as session:
            resp = await session.request(
                method=method,
                url=url,
                headers=headers,
                data=data,
                timeout=timeout,
                allow_redirects=True,
                impersonate=impersonate
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


async def http_relay(request: Request, path: str, exchange: str = 'binance'):
    """HTTP 代理转发端点"""
    config = EXCHANGE_CONFIG.get(exchange, EXCHANGE_CONFIG['binance'])
    # 确保路径格式正确（移除开头的斜杠，避免双斜杠）
    path = path.lstrip('/')
    target_url = f"{config['http_base']}/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"
    
    print(f"[HTTP-{exchange.upper()}] {request.method} {target_url}")
    
    try:
        # OKX 使用程序化 API 请求头，Binance 使用浏览器请求头
        if exchange == 'okx':
            # OKX: 不使用浏览器头部，使用程序化的 API 客户端头部
            headers = {
                'Accept': 'application/json',
            }
        else:
            # Binance: 使用浏览器请求头（保持向后兼容）
            headers = BROWSER_HEADERS.copy()
        
        # OKX 需要特定的请求头
        okx_specific_headers = ['ok-access-key', 'ok-access-sign', 'ok-access-timestamp', 
                               'ok-access-passphrase', 'x-simulated-trading']
        # Binance 特定的请求头
        binance_specific_headers = ['x-mbx-apikey']
        
        for k, v in request.headers.items():
            k_lower = k.lower()
            # 允许透传 User-Agent（特别是对于 OKX）
            if k_lower not in SKIP_HEADERS or k_lower == 'user-agent':
                if k_lower in ['content-type', 'authorization', 'user-agent']:
                    headers[k] = v
                elif k_lower == 'accept':
                    # 如果客户端指定了 Accept，优先使用客户端的
                    if v:
                        headers['Accept'] = v
                elif exchange == 'okx' and k_lower in [h.lower() for h in okx_specific_headers]:
                    headers[k] = v
                elif exchange == 'binance' and k_lower in [h.lower() for h in binance_specific_headers]:
                    headers[k] = v
        
        # 只有 Binance 需要浏览器相关的头（Referer 和 Origin）
        if exchange == 'binance':
            headers['Referer'] = config['referer']
            headers['Origin'] = config['origin']
        
        # OKX: 如果没有 User-Agent，使用程序化的 UA（不是浏览器 UA）
        if exchange == 'okx' and 'User-Agent' not in headers:
            headers['User-Agent'] = 'py-proxy/1.0'
        
        # OKX: 如果没有 Accept，使用 JSON
        if exchange == 'okx' and 'Accept' not in headers:
            headers['Accept'] = 'application/json'
        
        # 决定是否使用浏览器指纹伪装
        # Binance: 开启 (chrome120)，用于绕过 Cloudflare
        # OKX: 关闭 (None)，使用标准 TLS，避免 CloudFront 403
        impersonate_browser = IMPERSONATE_BROWSER if exchange == 'binance' else None
        
        body = await request.body()
        
        status_code, content, response_headers = await _curl_request_async(
            method=request.method,
            url=target_url,
            headers=headers,
            data=body if body else None,
            impersonate=impersonate_browser
        )
        
        print(f"[HTTP-{exchange.upper()}] Response {status_code}, Content-Length: {len(content)}")
        if status_code != 200:
            print(f"[HTTP-{exchange.upper()}] Response content (first 500 chars): {content[:500]}")
        
        return Response(
            content=content,
            status_code=status_code,
            headers=response_headers
        )
    
    except Exception as e:
        print(f"[HTTP-{exchange.upper()}] Error: {e}")
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


@router.websocket("/ws/relay/okx/{path:path}")
async def websocket_okx_relay_endpoint(websocket: WebSocket, path: str):
    """OKX WebSocket 代理端点"""
    await websocket_proxy(websocket, path, exchange='okx')


@router.api_route("/http/relay/okx/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def http_okx_relay_endpoint(request: Request, path: str):
    """OKX HTTP 代理转发端点"""
    return await http_relay(request, path, exchange='okx')
