"""应用日志：控制台输出，级别名与消息面向中文阅读。"""

from __future__ import annotations

import logging
import sys

_LEVEL_ZH = {
    "DEBUG": "调试",
    "INFO": "信息",
    "WARNING": "警告",
    "ERROR": "错误",
    "CRITICAL": "严重",
}


class ChineseLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.levelname = _LEVEL_ZH.get(record.levelname, record.levelname)
        return super().format(record)


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        for h in root.handlers:
            root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ChineseLogFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root.setLevel(level)
    root.addHandler(handler)
