from __future__ import annotations

import asyncio
import base64
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import aiohttp

logger = logging.getLogger(__name__)

CURSOR_API_URL = "https://api.cursor.com/v0/agents"


def _parse_hhmm(value: str, default: tuple[int, int]) -> tuple[int, int]:
    raw = (value or "").strip()
    if not raw:
        return default
    try:
        hour_str, minute_str = raw.split(":")
        hour = int(hour_str)
        minute = int(minute_str)
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour, minute
    except Exception:  # noqa: BLE001
        pass
    logger.warning("无效时间格式 '%s'，回退默认 %02d:%02d", value, default[0], default[1])
    return default


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no"}


def _build_prompt(symbol: str) -> str:
    skill_path = (
        "skills/btc-crypto-analyst/SKILL.md"
        if symbol == "BTC"
        else "skills/eth-crypto-analyst/SKILL.md"
    )
    lines = [
        f"Please execute {skill_path} from this repository directly.",
        "",
        "Requirements:",
        "1. Use only real market data. If data fails, report the exact failure point.",
        "2. After analysis, you must call distribute-post for publishing.",
        "3. Strictly follow the writing and behavior rules in the skill file.",
        "4. Output language must be Simplified Chinese.",
        "5. Final output should only be a distribution result summary (success/failed/partial).",
    ]
    return "\n".join(lines)


@dataclass(slots=True)
class CursorAgentConfig:
    enabled: bool
    api_key: str
    repository: str
    ref: str
    model: str
    timezone: str
    eth_time: tuple[int, int]
    btc_time: tuple[int, int]

    @classmethod
    def from_env(cls) -> "CursorAgentConfig":
        return cls(
            enabled=_bool_env("CURSOR_AGENT_SCHEDULER_ENABLED", True),
            api_key=os.getenv("CURSOR_API_KEY", "").strip(),
            repository=os.getenv("CURSOR_REPOSITORY", "").strip(),
            ref=os.getenv("CURSOR_REF", "main").strip() or "main",
            model=os.getenv("CURSOR_MODEL", "").strip(),
            timezone=os.getenv("CURSOR_SCHEDULE_TZ", "Asia/Shanghai").strip() or "Asia/Shanghai",
            eth_time=_parse_hhmm(os.getenv("CURSOR_ETH_TIME", "08:01"), (8, 1)),
            btc_time=_parse_hhmm(os.getenv("CURSOR_BTC_TIME", "20:01"), (20, 1)),
        )

    def is_ready(self) -> bool:
        return bool(self.enabled and self.api_key and self.repository)


async def _launch_cursor_agent(config: CursorAgentConfig, symbol: str) -> dict:
    body = {
        "prompt": {"text": _build_prompt(symbol)},
        "source": {"repository": config.repository, "ref": config.ref},
        "target": {"autoCreatePr": False},
    }
    if config.model:
        body["model"] = config.model

    basic = base64.b64encode(f"{config.api_key}:".encode("ascii")).decode("ascii")
    headers = {
        "Authorization": f"Basic {basic}",
        "Content-Type": "application/json",
    }

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        async with session.post(CURSOR_API_URL, json=body, headers=headers) as resp:
            payload = await resp.json(content_type=None)
            if resp.status >= 400:
                raise RuntimeError(f"Cursor API {resp.status}: {payload}")
            return payload


def _seconds_until_next_run(tz: ZoneInfo, hour: int, minute: int) -> tuple[float, str]:
    now = datetime.now(tz)
    next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if next_run <= now:
        next_run = next_run + timedelta(days=1)
    return (next_run - now).total_seconds(), next_run.isoformat()


class CursorAgentScheduler:
    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._config = CursorAgentConfig.from_env()
        try:
            self._tz = ZoneInfo(self._config.timezone)
        except Exception:  # noqa: BLE001
            logger.warning("无效时区 '%s'，回退 Asia/Shanghai", self._config.timezone)
            self._config.timezone = "Asia/Shanghai"
            self._tz = ZoneInfo("Asia/Shanghai")

    def start(self) -> None:
        if not self._config.enabled:
            logger.info("Cursor Agent 定时器已禁用（CURSOR_AGENT_SCHEDULER_ENABLED=false）")
            return
        if not self._config.api_key or not self._config.repository:
            logger.warning("Cursor Agent 定时器未启动：缺少 CURSOR_API_KEY 或 CURSOR_REPOSITORY")
            return

        self._start_symbol("ETH", *self._config.eth_time)
        self._start_symbol("BTC", *self._config.btc_time)
        logger.info(
            "Cursor Agent 定时器已启动（TZ=%s, ETH=%02d:%02d, BTC=%02d:%02d）",
            self._config.timezone,
            self._config.eth_time[0],
            self._config.eth_time[1],
            self._config.btc_time[0],
            self._config.btc_time[1],
        )

    def _start_symbol(self, symbol: str, hour: int, minute: int) -> None:
        existing = self._tasks.get(symbol)
        if existing and not existing.done():
            return
        self._tasks[symbol] = asyncio.create_task(self._runner(symbol, hour, minute))

    async def stop(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def _runner(self, symbol: str, hour: int, minute: int) -> None:
        logger.info("启动 Cursor Agent 定时任务 symbol=%s schedule=%02d:%02d", symbol, hour, minute)
        while True:
            wait_seconds, next_run = _seconds_until_next_run(self._tz, hour, minute)
            logger.info("Cursor Agent 下次执行 symbol=%s at %s", symbol, next_run)
            await asyncio.sleep(wait_seconds)
            try:
                result = await _launch_cursor_agent(self._config, symbol)
                logger.info(
                    "Cursor Agent 启动成功 symbol=%s id=%s status=%s",
                    symbol,
                    result.get("id"),
                    result.get("status"),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error("Cursor Agent 执行失败 symbol=%s: %s", symbol, exc)


cursor_agent_scheduler = CursorAgentScheduler()
