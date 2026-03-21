"""X.AI（Responses API）AI API 客户端模块。

负责调用 X.AI 的 `/v1/responses` 接口；默认使用 **SSE 流式**（`stream: true`），
按官方说明持续接收数据，避免长任务时整包等待被中途断开。
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from typing import Any

import aiohttp

logger: logging.Logger = logging.getLogger(__name__)


class GrokApiClient:
    """X.AI Responses API 客户端：流式解析 `output_text` delta，并过滤推理类事件。"""

    # 流式长任务：总时间放宽；单段读不超时（避免工具调用阶段长时间无新字节）
    _TIMEOUT: aiohttp.ClientTimeout = aiohttp.ClientTimeout(
        total=3600,
        connect=30,
        sock_connect=30,
        sock_read=None,
    )

    def __init__(self) -> None:
        """从环境变量加载配置，缺失则抛出 ValueError。"""
        api_key: str | None = os.getenv("XAI_API_KEY")
        model: str | None = os.getenv("XAI_MODEL")
        base_url: str = os.getenv("XAI_API_BASE_URL") or "https://api.x.ai/v1"

        missing: list[str] = []
        if not api_key:
            missing.append("XAI_API_KEY")
        if not model:
            missing.append("XAI_MODEL")
        if missing:
            raise ValueError(f"缺少必需的环境变量: {', '.join(missing)}")

        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1") and "/v1" not in base_url:
            base_url = f"{base_url}/v1"

        self.base_url = base_url
        self.api_key = api_key
        self.model = model

        logger.info("🚀 XAI GrokApiClient 初始化完成，模型: %s", self.model)

    async def fetch_sentiment(self, content: str) -> str:
        """调用 X.AI 获取加密货币市场情绪分析（Responses API + SSE 流式）。"""
        url: str = f"{self.base_url}/responses"

        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        payload: dict[str, object] = {
            "model": self.model,
            "input": [
                {"role": "user", "content": content},
            ],
            "tools": [
                {"type": "x_search"},
                {"type": "web_search"},
            ],
            "stream": True,
        }

        prompt_len = len(content.encode("utf-8"))
        logger.info(
            "🚀 Grok SSE 请求: url=%s model=%s prompt_bytes=%d stream=True",
            url,
            self.model,
            prompt_len,
        )

        t0 = time.perf_counter()
        async with aiohttp.ClientSession(timeout=self._TIMEOUT) as session:
            async with session.post(
                url, headers=headers, json=payload
            ) as response:
                ct = response.headers.get("Content-Type", "")
                logger.info(
                    "📨 Grok 响应: status=%s Content-Type=%s",
                    response.status,
                    ct or "(none)",
                )
                response.raise_for_status()
                result, stream_stats = await self._read_sse_response_text(response)

        elapsed = time.perf_counter() - t0
        logger.info(
            "✅ Grok 情绪分析完成: result_chars=%d 用时=%.2fs 原始字节=%d "
            "data行=%d json解析失败=%d text_delta块=%d chat_delta块=%d "
            "literal[DONE]=%s response.completed=%s done全文兜底=%s 事件=%s",
            len(result),
            elapsed,
            stream_stats["raw_bytes"],
            stream_stats["sse_data_lines"],
            stream_stats["json_parse_errors"],
            stream_stats["text_delta_chunks"],
            stream_stats["chat_delta_chunks"],
            stream_stats["saw_done_marker"],
            stream_stats["saw_response_completed"],
            stream_stats["used_done_fallback"],
            dict(stream_stats["event_types"].most_common(20)),
        )
        return result

    async def _read_sse_response_text(
        self, response: aiohttp.ClientResponse
    ) -> tuple[str, dict[str, Any]]:
        """SSE 解析：标准事件边界（双换行），支持多行 data 与 event 行（xAI Responses）。"""
        parts: list[str] = []
        fallback_ref: list[str | None] = [None]
        stream_stats: dict[str, Any] = {
            "raw_bytes": 0,
            "sse_data_lines": 0,
            "text_delta_chunks": 0,
            "chat_delta_chunks": 0,
            "saw_done_marker": False,
            "saw_response_completed": False,
            "used_done_fallback": False,
            "event_types": Counter(),
            "json_parse_errors": 0,
            "first_data_logged": False,
        }

        buf = ""
        async for raw in response.content.iter_any():
            stream_stats["raw_bytes"] += len(raw)
            buf += raw.decode("utf-8", errors="replace")

            while "\n\n" in buf:
                event_block, buf = buf.split("\n\n", 1)
                self._process_sse_event_block(
                    event_block.strip(), parts, fallback_ref, stream_stats
                )

        if buf.strip():
            self._process_sse_event_block(buf.strip(), parts, fallback_ref, stream_stats)

        text = "".join(parts).strip()
        used_fallback = False
        if not text and fallback_ref[0]:
            text = fallback_ref[0].strip()
            used_fallback = bool(text)
        stream_stats["used_done_fallback"] = used_fallback

        if not text:
            logger.error(
                "❌ Grok SSE 无有效正文: event=%s data行=%d",
                dict(stream_stats["event_types"].most_common(15)),
                stream_stats["sse_data_lines"],
            )
            raise ValueError("❌ Grok API 流式返回为空，未获取到有效的情绪分析")
        return text, stream_stats

    def _process_sse_event_block(
        self,
        event_block: str,
        parts: list[str],
        fallback_ref: list[str | None],
        stream_stats: dict[str, Any],
    ) -> None:
        """处理单个 SSE 事件块（多行 data: 拼接 + 可选 event:）。"""
        lines = event_block.splitlines()
        event_type: str | None = None
        data_lines: list[str] = []

        for line in lines:
            line = line.rstrip("\r")
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())

        if not data_lines:
            return

        payload = "\n".join(data_lines)
        if payload == "[DONE]":
            stream_stats["saw_done_marker"] = True
            logger.debug("Grok SSE: data [DONE]")
            return

        try:
            obj: Any = json.loads(payload)
        except json.JSONDecodeError:
            stream_stats["json_parse_errors"] += 1
            logger.warning("Grok SSE: JSON 解析失败 payload前200字=%r", payload[:200])
            return

        if not isinstance(obj, dict):
            logger.warning("Grok SSE: data JSON 非 object，类型=%s", type(obj).__name__)
            return

        stream_stats["sse_data_lines"] += 1
        if not stream_stats["first_data_logged"]:
            stream_stats["first_data_logged"] = True
            logger.debug(
                "Grok SSE 首条 data（截断）: %s",
                payload[:400] + ("…" if len(payload) > 400 else ""),
            )

        # 错误处理
        err = obj.get("error")
        if err is not None:
            msg = err.get("message") if isinstance(err, dict) else str(err)
            logger.error("Grok SSE: error 事件 err=%r", err)
            raise RuntimeError(f"Grok API 返回错误: {msg}")

        # 类型优先级：JSON 中的 type > event: 头
        t: str | None = obj.get("type") if isinstance(obj.get("type"), str) else event_type
        if isinstance(t, str):
            stream_stats["event_types"][t] += 1

        # 1. 首选官方 delta
        if t == "response.output_text.delta" and isinstance(obj.get("delta"), str):
            stream_stats["text_delta_chunks"] += 1
            parts.append(obj["delta"])
            return

        if t == "response.completed":
            stream_stats["saw_response_completed"] = True
            return

        # 2. done 兜底
        if t == "response.output_text.done" and isinstance(obj.get("text"), str):
            fallback_ref[0] = obj["text"]
            logger.debug("Grok SSE: output_text.done 长度=%d", len(obj["text"]))
            return

        # 3. 兼容旧 Chat Completions 风格
        choices = obj.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                delta = c0.get("delta")
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    stream_stats["chat_delta_chunks"] += 1
                    parts.append(delta["content"])
                    return

        # 非正文事件安静处理
        if isinstance(t, str) and t not in ("response.output_text.delta", "response.output_text.done"):
            if t.startswith("response.") and (
                "reasoning" in t or "in_progress" in t or t.endswith(".created")
            ):
                logger.debug("Grok SSE: 忽略事件 type=%s", t)
            else:
                logger.debug(
                    "Grok SSE: 未拼接正文的 data 事件 type=%s keys=%s",
                    t,
                    list(obj.keys())[:12],
                )