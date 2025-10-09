#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSocket 和 HTTP 透明转发
- WebSocket: ws://host:port/ws/relay
- HTTP: http://host:port/http/relay/{path:path}
"""

from fastapi import APIRouter, WebSocket, Request
from fastapi.responses import Response
import asyncio
import websockets
import aiohttp

router = APIRouter()

# 远端地址配置
WS_REMOTE_URL = "wss://fstream.binance.com"
HTTP_REMOTE_URL = "https://fapi.binance.com"


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


@router.websocket("/ws/relay")
async def websocket_relay_endpoint(websocket: WebSocket):
    """WebSocket 转发端点"""
    await websocket.accept()
    print(f"[WS] Client connected, relaying to {WS_REMOTE_URL}")
    try:
        async with websockets.connect(WS_REMOTE_URL) as remote_ws:
            await relay(websocket, remote_ws)
    except Exception as e:
        print(f"[WS] Relay error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass
        print("[WS] Client disconnected")


@router.api_route("/http/relay/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def http_relay_endpoint(request: Request, path: str):
    """HTTP 透明转发端点"""
    # 构建目标 URL
    target_url = f"{HTTP_REMOTE_URL}/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"
    
    print(f"[HTTP] {request.method} {target_url}")
    
    try:
        async with aiohttp.ClientSession() as session:
            # 准备转发的请求头（排除 host）
            headers = {k: v for k, v in request.headers.items() if k.lower() != 'host'}
            
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
                # 读取响应内容
                content = await resp.read()
                
                # 转发响应头（排除某些不应该转发的）
                response_headers = {
                    k: v for k, v in resp.headers.items()
                    if k.lower() not in ['transfer-encoding', 'connection']
                }
                
                print(f"[HTTP] Response {resp.status}")
                
                # 返回响应
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
