from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GrokStore:
    """
    管理 TA-Lib 内部的 grok_sentiment.txt

    - 默认路径：<repo_root>/data/grok_sentiment.txt
    - 更新时使用原子替换，避免写入一半导致读取到脏数据
    """

    filename: str = "grok_sentiment.txt"

    def _repo_root(self) -> Path:
        # app/services/.. -> app -> repo_root
        return Path(__file__).resolve().parents[2]

    def path(self) -> Path:
        return self._repo_root() / "data" / self.filename

    def get_required_token(self) -> str | None:
        return os.getenv("GROK_UPDATE_TOKEN")

    def read(self) -> str:
        p = self.path()
        try:
            content = p.read_text(encoding="utf-8").strip()
            return content or "Grok 文件为空"
        except FileNotFoundError:
            return f"Grok 文件不存在: {p}"
        except Exception as e:
            return f"读取 Grok 文件失败: {e}"

    def read_info(self) -> dict:
        """
        返回 Grok 文件的读取信息（内容 + 路径 + 最后修改时间）。
        """
        p = self.path()
        content = self.read()
        updated_at = 0
        try:
            updated_at = int(p.stat().st_mtime)
        except Exception:
            updated_at = 0
        return {"path": str(p), "content": content, "updated_at": updated_at}

    def update(self, content: str, max_bytes: int = 256_000) -> dict:
        if content is None:
            raise ValueError("content 不能为空")
        data = content.encode("utf-8")
        if len(data) == 0:
            raise ValueError("content 不能为空")
        if len(data) > max_bytes:
            raise ValueError(f"content 过大，最大 {max_bytes} bytes")

        p = self.path()
        p.parent.mkdir(parents=True, exist_ok=True)

        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)

        return {
            "ok": True,
            "bytes_written": len(data),
            "path": str(p),
            "updated_at": int(time.time()),
        }

