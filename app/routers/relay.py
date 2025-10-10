#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSocket 和 HTTP 代理转发
- WebSocket: ws://host:port/ws/relay/{path:path}
- HTTP: http://host:port/http/relay/{path:path}
"""

from fastapi import APIRouter, WebSocket, Request
from fastapi.responses import Response
import asyncio
import websockets
import aiohttp

router = APIRouter()

# 远端地址配置
WS_REMOTE_BASE = "wss://fstream.binance.com"
HTTP_REMOTE_BASE = "https://fapi.binance.com"

# 模拟正常浏览器的请求头（不包括压缩，让 aiohttp 自动处理）
BROWSER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
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
    """WebSocket 代理核心逻辑"""
    # 构建目标 URL
    target_url = f"{WS_REMOTE_BASE}/{path}"
    if websocket.query_params:
        from urllib.parse import urlencode
        query = urlencode(websocket.query_params)
        target_url += f"?{query}"
    
    print(f"[WS] Client connecting to: {target_url}")
    
    await websocket.accept()
    
    try:
        # 使用浏览器身份连接远端（模拟正常浏览器请求）
        async with websockets.connect(
            target_url,
            additional_headers={
                'User-Agent': BROWSER_HEADERS['User-Agent'],
                'Origin': 'https://www.binance.com',
                'Host': 'fstream.binance.com',
            }
        ) as remote_ws:
            print(f"[WS] Connected to remote")
            await relay(websocket, remote_ws)
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass
        print("[WS] Disconnected")

async def http_relay(request: Request, path: str):
    """HTTP 代理转发端点"""
    # 构建目标 URL
    target_url = f"{HTTP_REMOTE_BASE}/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"
    
    print(f"[HTTP] {request.method} {target_url}")
    
    try:
        async with aiohttp.ClientSession() as session:
            # 准备请求头：只保留业务相关的，过滤掉暴露身份的
            headers = BROWSER_HEADERS.copy()
            
            # 只保留业务必需的请求头
            for k, v in request.headers.items():
                k_lower = k.lower()
                if k_lower not in SKIP_HEADERS:
                    # 保留 content-type, authorization 等业务头
                    if k_lower in ['content-type', 'authorization', 'x-mbx-apikey']:
                        headers[k] = v
            
            # 读取请求体
            body = await request.body()
            
            # 发送请求到远端
            async with session.request(
                method=request.method,
                url=target_url,
                headers=headers,
                data=body if body else None,
                allow_redirects=False
            ) as resp:
                # aiohttp 会自动解压缩，读取文本或二进制内容
                content = await resp.read()
                
                # 转发响应头（排除压缩和连接相关的头）
                response_headers = {
                    k: v for k, v in resp.headers.items()
                    if k.lower() not in ['transfer-encoding', 'connection', 'content-encoding']
                }
                
                print(f"[HTTP] Response {resp.status}")
                
                # 返回响应（content 已经是解压后的）
                return Response(
                    content=content,
                    status_code=resp.status,
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
    """WebSocket 代理转发端点"""
    await websocket_proxy(websocket, path)


@router.api_route("/http/relay/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def http_relay_endpoint(request: Request, path: str):
    """HTTP 代理转发端点"""
    return await http_relay(request, path)
