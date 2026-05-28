"""SQLite storage for subscription inbox and hot board."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.services.kline_chart_service import _repo_root

_DB_LOCK = threading.Lock()
_SCHEMA_READY = False

_SH_TZ = timezone(timedelta(hours=8))


def _db_path() -> Path:
    root = _repo_root()
    directory = root / "data" / "pipeline"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "pipeline.db"


@contextmanager
def _connection():
    conn = sqlite3.connect(_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now_iso() -> str:
    return datetime.now(_SH_TZ).isoformat()


def _migrate_inbox(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(subscription_inbox)").fetchall()}
    if not cols:
        return
    if "inbox_id" not in cols:
        return
    conn.execute(
        """
        CREATE TABLE subscription_inbox_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            received_at TEXT NOT NULL,
            channel_username TEXT NOT NULL,
            raw_text TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO subscription_inbox_new (received_at, channel_username, raw_text)
        SELECT received_at, channel_username, raw_text FROM subscription_inbox
        """
    )
    conn.execute("DROP TABLE subscription_inbox")
    conn.execute("ALTER TABLE subscription_inbox_new RENAME TO subscription_inbox")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_inbox_channel ON subscription_inbox(channel_username)"
    )


def _migrate_hot_board(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(hot_board)").fetchall()}
    if not cols:
        return
    if "alert_reason" not in cols:
        conn.execute("ALTER TABLE hot_board ADD COLUMN alert_reason TEXT")
    if "merged_for_sentiment" in cols:
        conn.execute(
            """
            UPDATE hot_board
            SET alert_reason = merged_for_sentiment
            WHERE alert_reason IS NULL AND merged_for_sentiment IS NOT NULL
              AND merged_for_sentiment != ''
            """
        )


def _ensure_schema() -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    with _DB_LOCK:
        if _SCHEMA_READY:
            return
        with _connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS subscription_inbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    received_at TEXT NOT NULL,
                    channel_username TEXT NOT NULL,
                    raw_text TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_inbox_channel
                    ON subscription_inbox(channel_username);

                CREATE TABLE IF NOT EXISTS hot_board (
                    symbol TEXT PRIMARY KEY,
                    base_asset TEXT,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    sources TEXT NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 1,
                    alert_reason TEXT,
                    merger_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_hot_board_expires
                    ON hot_board(expires_at);
                """
            )
            _migrate_inbox(conn)
            _migrate_hot_board(conn)
        _SCHEMA_READY = True


def inbox_append(*, channel_username: str, raw_text: str) -> None:
    """Telethon 只落原文；不存 message_id / permalink。"""
    _ensure_schema()
    text = (raw_text or "").strip()
    if not text:
        return
    with _DB_LOCK, _connection() as conn:
        conn.execute(
            """
            INSERT INTO subscription_inbox (received_at, channel_username, raw_text)
            VALUES (?, ?, ?)
            """,
            (_now_iso(), channel_username.lower(), text),
        )


def inbox_consume(*, channel: str = "wizzalert", limit: int = 50) -> List[Dict[str, str]]:
    """取出待处理原文并物理删除；对 skill 仅返回 raw_text。"""
    _ensure_schema()
    channel = channel.lower().strip()
    limit = max(1, min(int(limit), 200))
    with _DB_LOCK, _connection() as conn:
        rows = conn.execute(
            """
            SELECT id, raw_text FROM subscription_inbox
            WHERE channel_username = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (channel, limit),
        ).fetchall()
        items = [{"raw_text": row["raw_text"]} for row in rows]
        if rows:
            placeholders = ",".join("?" for _ in rows)
            ids = [row["id"] for row in rows]
            conn.execute(
                f"DELETE FROM subscription_inbox WHERE id IN ({placeholders})",
                ids,
            )
    return items


def inbox_delete(inbox_id: str) -> bool:
    """兼容旧 DELETE 路由；新表无 inbox_id 时恒为 false。"""
    _ensure_schema()
    return False


def hot_board_upsert(entry: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_schema()
    symbol = entry["symbol"].upper()
    now = _now_iso()
    expires = (datetime.now(_SH_TZ) + timedelta(hours=12)).isoformat()
    source = entry["source"]
    new_sources = [source]
    base_asset = entry.get("base_asset") or symbol.replace("USDT", "")
    alert_reason = (entry.get("alert_reason") or "").strip() or None
    merger_json = (
        json.dumps(entry["merger"], ensure_ascii=False) if entry.get("merger") is not None else None
    )

    with _DB_LOCK, _connection() as conn:
        row = conn.execute(
            "SELECT * FROM hot_board WHERE symbol = ?",
            (symbol,),
        ).fetchone()
        if row is None:
            sources = new_sources
            hit_count = 1
            first_seen_at = now
            conn.execute(
                """
                INSERT INTO hot_board
                (symbol, base_asset, first_seen_at, last_seen_at, expires_at, sources,
                 hit_count, alert_reason, merger_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    base_asset,
                    first_seen_at,
                    now,
                    expires,
                    json.dumps(sources, ensure_ascii=False),
                    hit_count,
                    alert_reason if source == "wizz_alert" else None,
                    merger_json,
                ),
            )
        else:
            old_sources = json.loads(row["sources"] or "[]")
            sources = list(dict.fromkeys(old_sources + new_sources))
            hit_count = int(row["hit_count"] or 0) + 1
            keep_alert = row["alert_reason"]
            if source == "wizz_alert" and alert_reason:
                keep_alert = alert_reason
            keep_merger = merger_json if source == "merger_analyzer" else row["merger_json"]
            conn.execute(
                """
                UPDATE hot_board SET
                    base_asset = ?,
                    last_seen_at = ?,
                    expires_at = ?,
                    sources = ?,
                    hit_count = ?,
                    alert_reason = ?,
                    merger_json = ?
                WHERE symbol = ?
                """,
                (
                    base_asset,
                    now,
                    expires,
                    json.dumps(sources, ensure_ascii=False),
                    hit_count,
                    keep_alert,
                    keep_merger,
                    symbol,
                ),
            )

    result = hot_board_get(symbol)
    assert result is not None
    return result


def hot_board_get(symbol: str) -> Optional[Dict[str, Any]]:
    _ensure_schema()
    symbol = symbol.upper()
    with _DB_LOCK, _connection() as conn:
        row = conn.execute(
            "SELECT * FROM hot_board WHERE symbol = ? AND expires_at > ?",
            (symbol, _now_iso()),
        ).fetchone()
    return _row_to_entry(row) if row else None


def hot_board_list_active(*, limit: int = 100) -> List[Dict[str, Any]]:
    _ensure_schema()
    limit = max(1, min(int(limit), 200))
    now = _now_iso()
    with _DB_LOCK, _connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM hot_board
            WHERE expires_at > ?
            ORDER BY last_seen_at DESC
            LIMIT ?
            """,
            (now, limit),
        ).fetchall()
    return [_row_to_entry(row) for row in rows]


def hot_board_purge_expired() -> int:
    _ensure_schema()
    now = _now_iso()
    with _DB_LOCK, _connection() as conn:
        cur = conn.execute("DELETE FROM hot_board WHERE expires_at <= ?", (now,))
        return cur.rowcount


def _row_to_entry(row: sqlite3.Row) -> Dict[str, Any]:
    sources = json.loads(row["sources"] or "[]")
    alert_reason = (row["alert_reason"] or "").strip() or None
    if not alert_reason:
        try:
            legacy = (row["merged_for_sentiment"] or "").strip()
        except (KeyError, IndexError):
            legacy = ""
        if legacy:
            alert_reason = legacy
    return {
        "symbol": row["symbol"],
        "base_asset": row["base_asset"],
        "first_seen_at": row["first_seen_at"],
        "last_seen_at": row["last_seen_at"],
        "expires_at": row["expires_at"],
        "sources": sources,
        "hit_count": row["hit_count"],
        "alert_reason": alert_reason,
    }


def build_hot_board_supplement(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sources": entry.get("sources") or [],
        "alert_reason": entry.get("alert_reason") or "",
    }
