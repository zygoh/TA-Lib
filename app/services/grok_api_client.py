"""X.AI（Responses API）AI API 客户端模块。

负责调用 X.AI 的 `/v1/responses` 接口，解析返回 JSON 中的最终文本（`output_text`）。
"""

from __future__ import annotations

import logging
from typing import Any
import os

import aiohttp

logger: logging.Logger = logging.getLogger(__name__)


class GrokApiClient:
    """X.AI Responses API 客户端，负责解析 output_text 并过滤 thinking 内容。

    Attributes:
        base_url: Grok API 基础地址。
        api_key: Grok API 密钥。
        model: Grok 模型名称。
    """

    # 请求超时配置：总超时 300 秒，连接超时 30 秒
    _TIMEOUT: aiohttp.ClientTimeout = aiohttp.ClientTimeout(
        total=300,
        connect=30,
    )

    def __init__(self) -> None:
        """从环境变量加载配置，缺失则抛出 ValueError。

        Raises:
            ValueError: 当任一必需环境变量缺失时，错误信息中列出所有缺失变量名。
        """
        # 只对接 xAI 官方 Responses API：不做任何变量名兼容/回退。
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

        # 规范化 base_url，确保后续拼接 `/responses` 不会出错
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1") and "/v1" not in base_url:
            base_url = f"{base_url}/v1"

        self.base_url = base_url
        self.api_key = api_key
        self.model = model

        logger.info("🚀 XAI GrokApiClient 初始化完成，模型: %s", self.model)

    async def fetch_sentiment(self, content: str) -> str:
        """调用 X.AI 获取加密货币市场情绪分析。

        将用户传入的 content 作为 input 发给 X.AI Responses API，
        从返回 JSON 中提取 `output_text`（并跳过 `type="reasoning"` 的推理块）。

        Args:
            content: 用户传入的消息内容，作为 Responses API 的 `input` 发送给模型。

        Returns:
            过滤 thinking 后的最终分析文本。

        Raises:
            ValueError: 返回内容为空时抛出。
            aiohttp.ClientError: API 请求失败时抛出。
        """
        url: str = f"{self.base_url}/responses"

        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, object] = {
            "model": self.model,
            "input": [
                {"role": "user", "content": content},
            ],
            # 让模型可调用内置搜索工具（由 xAI 服务器侧执行）
            "tools": [
                {"type": "x_search"},
                {"type": "web_search"},
            ],
        }

        logger.info("🚀 发送情绪分析请求到 Grok API: %s", url)

        async with aiohttp.ClientSession(timeout=self._TIMEOUT) as session:
            async with session.post(
                url, headers=headers, json=payload
            ) as response:
                response.raise_for_status()
                data: Any = await response.json()
                result: str = self._extract_output_text(data)

        logger.info("✅ 情绪分析完成，内容长度: %d", len(result))
        return result

    @staticmethod
    def _extract_output_text(data: Any) -> str:
        """从 Responses API 返回 JSON 中提取 `output_text` 文本。

        Responses API 的返回结构可能在不同模型/版本下略有差异，因此这里
        使用递归方式查找所有形如：
        - {"type": "output_text", "text": "..."} 的对象并拼接。
        """

        parts: list[str] = []

        def _walk(v: Any) -> None:
            if isinstance(v, dict):
                # 严格跳过推理过程块：官方示例中推理为 type="reasoning"
                if v.get("type") == "reasoning":
                    return
                if v.get("type") == "output_text":
                    text = v.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text)
                for vv in v.values():
                    _walk(vv)
            elif isinstance(v, list):
                for item in v:
                    _walk(item)

        _walk(data)
        result = "\n".join(parts).strip()
        if not result:
            raise ValueError("❌ Grok API 返回内容为空，未获取到有效的情绪分析")
        return result
