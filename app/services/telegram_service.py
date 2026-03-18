from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import aiohttp


class TelegramService:
    async def send_markdown(self, message: str) -> Tuple[bool, Dict[str, Any]]:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return False, {"error": "Telegram 配置缺失：需要 TELEGRAM_BOT_TOKEN 和 TELEGRAM_CHAT_ID"}

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}

        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()
                    if resp.status >= 400:
                        return False, {"status": resp.status, "response": data}
                    return True, data
            except Exception as e:
                return False, {"error": str(e)}

