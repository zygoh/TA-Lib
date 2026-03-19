"""Grok 4.1 AI API 客户端模块。

负责调用 Grok 4.1 API（OpenAI 兼容格式），处理 SSE 流式响应，
过滤 thinking/reasoning 内容，仅保留最终分析文本。
"""

from __future__ import annotations

import json
import logging
import os
import re

import aiohttp

logger: logging.Logger = logging.getLogger(__name__)


class GrokApiClient:
    """Grok 4.1 AI API 客户端，负责流式调用并过滤 thinking 内容。

    Attributes:
        base_url: Grok API 基础地址。
        api_key: Grok API 密钥。
        model: Grok 模型名称。
    """

    # SSE 流式请求的超时配置：总超时 300 秒，连接超时 30 秒
    _TIMEOUT: aiohttp.ClientTimeout = aiohttp.ClientTimeout(
        total=300,
        connect=30,
    )

    def __init__(self) -> None:
        """从环境变量加载配置，缺失则抛出 ValueError。

        Raises:
            ValueError: 当任一必需环境变量缺失时，错误信息中列出所有缺失变量名。
        """
        required_vars: dict[str, str | None] = {
            "GROK_API_BASE_URL": os.getenv("GROK_API_BASE_URL"),
            "GROK_API_KEY": os.getenv("GROK_API_KEY"),
            "GROK_MODEL": os.getenv("GROK_MODEL"),
        }

        missing: list[str] = [
            name for name, value in required_vars.items() if not value
        ]
        if missing:
            raise ValueError(
                f"缺少必需的环境变量: {', '.join(missing)}"
            )

        self.base_url: str = required_vars["GROK_API_BASE_URL"]  # type: ignore[assignment]
        self.api_key: str = required_vars["GROK_API_KEY"]  # type: ignore[assignment]
        self.model: str = required_vars["GROK_MODEL"]  # type: ignore[assignment]

        logger.info("🚀 GrokApiClient 初始化完成，模型: %s", self.model)

    async def fetch_sentiment(self, content: str) -> str:
        """调用 Grok 4.1 API 获取加密货币市场情绪分析。

        将用户传入的 content 作为 user message，连同预定义的 system prompt
        一起发送给 Grok 4.1 API，解析 SSE 流式响应并过滤 thinking 内容。

        Args:
            content: 用户传入的消息内容，将作为 messages 数组中的
                user message 发送给 Grok 4.1。

        Returns:
            过滤 thinking 后的最终分析文本。

        Raises:
            ValueError: 返回内容为空时抛出。
            aiohttp.ClientError: API 请求失败时抛出。
        """
        url: str = f"{self.base_url}/chat/completions"

        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": content},
            ],
            "stream": True,
        }

        logger.info("🚀 发送情绪分析请求到 Grok API: %s", url)

        async with aiohttp.ClientSession(timeout=self._TIMEOUT) as session:
            async with session.post(
                url, headers=headers, json=payload
            ) as response:
                response.raise_for_status()
                result: str = await self._parse_sse_stream(response)

        logger.info("✅ 情绪分析完成，内容长度: %d", len(result))
        return result

    async def _parse_sse_stream(
        self, response: aiohttp.ClientResponse
    ) -> str:
        """解析 SSE 流式响应，过滤 thinking/reasoning 内容。

        逐行读取 SSE 事件：
        - 遇到 ``data: [DONE]`` 终止读取
        - 解析 JSON，检查 delta 中的内容类型
        - 丢弃 ``reasoning_content`` 类型的 delta
        - 拼接 ``content`` 类型的 delta 文本
        - 单行 JSON 解析失败时跳过该行继续处理

        Args:
            response: aiohttp 的流式响应对象。

        Returns:
            拼接后的最终分析文本。

        Raises:
            ValueError: 最终拼接内容为空时抛出。
        """
        content_parts: list[str] = []

        async for raw_line in response.content:
            line: str = raw_line.decode("utf-8", errors="replace").strip()

            if not line:
                continue

            # SSE 协议：每行以 "data: " 开头
            if not line.startswith("data:"):
                continue

            data: str = line[len("data:"):].strip()

            # 终止标记
            if data == "[DONE]":
                logger.info("📊 SSE 流读取完成，收到 [DONE] 标记")
                break

            # 解析 JSON
            try:
                chunk: dict[str, object] = json.loads(data)
            except json.JSONDecodeError:
                logger.warning("⚠️ SSE 行 JSON 解析失败，跳过: %s", data[:100])
                continue

            # 提取 delta
            choices: list[dict[str, object]] = chunk.get("choices", [])  # type: ignore[assignment]
            if not choices:
                continue

            delta: dict[str, object] = choices[0].get("delta", {})  # type: ignore[assignment]

            # 丢弃 thinking/reasoning 内容
            # reasoning_content 存在时直接跳过，不拼接
            reasoning: object = delta.get("reasoning_content")
            if reasoning:
                continue

            # 拼接 content
            text: object = delta.get("content")
            if text and isinstance(text, str):
                content_parts.append(text)

        result: str = "".join(content_parts).strip()
        result = self._strip_thinking_lines(result)

        if not result:
            raise ValueError("❌ Grok API 返回内容为空，未获取到有效的情绪分析")

        return result

    @staticmethod
    def _strip_thinking_lines(text: str) -> str:
        """移除混在 content 中的 thinking/reasoning 行。

        某些 API 代理会将模型的思考过程以 markdown 引用格式
        （``> 🔍``、``> ***-``、``> 📖``、``> 🔧``、``> 📋``、``> 📝``）
        直接嵌入 content 字段。此方法按行过滤这些内容。

        Args:
            text: 拼接后的原始文本。

        Returns:
            过滤 thinking 行后的干净文本。
        """
        # 匹配以 > 开头，后跟空格和 thinking 标识符的行
        thinking_pattern: re.Pattern[str] = re.compile(
            r"^>\s*(?:"
            r"🔍|📖|🔧|📋|📝|\*\*\*-"
            r")",
        )
        lines: list[str] = text.splitlines()
        clean_lines: list[str] = [
            line for line in lines
            if not thinking_pattern.match(line.strip())
        ]
        return "\n".join(clean_lines).strip()
