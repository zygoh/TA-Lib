"""SQLite storage for subscription inbox and hot board."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
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
                    inbox_id TEXT PRIMARY KEY,
                    received_at TEXT NOT NULL,
                    channel_username TEXT NOT NULL,
                    message_id INTEGER,
                    permalink TEXT,
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
                    wizz_json TEXT,
                    merged_for_sentiment TEXT,
                    merger_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_hot_board_expires
                    ON hot_board(expires_at);
                """
            )
        _SCHEMA_READY = True


def inbox_append(
    *,
    channel_username: str,
    message_id: int | None,
    permalink: str,
    raw_text: str,
) -> Dict[str, Any]:
    _ensure_schema()
    inbox_id = str(uuid.uuid4())
    received_at = _now_iso()
    with _DB_LOCK, _connection() as conn:
        conn.execute(
            """
            INSERT INTO subscription_inbox
            (inbox_id, received_at, channel_username, message_id, permalink, raw_text)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (inbox_id, received_at, channel_username.lower(), message_id, permalink, raw_text),
        )
    return {
        "inbox_id": inbox_id,
        "received_at": received_at,
        "channel_username": channel_username.lower(),
        "message_id": message_id,
        "permalink": permalink,
        "raw_text": raw_text,
    }


def inbox_consume(*, channel: str = "wizzalert", limit: int = 50) -> List[Dict[str, Any]]:
    _ensure_schema()
    channel = channel.lower().strip()
    limit = max(1, min(int(limit), 200))
    with _DB_LOCK, _connection() as conn:
        rows = conn.execute(
            """
            SELECT inbox_id, received_at, channel_username, message_id, permalink, raw_text
            FROM subscription_inbox
            WHERE channel_username = ?
            ORDER BY received_at ASC
            LIMIT ?
            """,
            (channel, limit),
        ).fetchall()
        items = [dict(row) for row in rows]
        if items:
            placeholders = ",".join("?" for _ in items)
            ids = [item["inbox_id"] for item in items]
            conn.execute(
                f"DELETE FROM subscription_inbox WHERE inbox_id IN ({placeholders})",
                ids,
            )
    return items


def inbox_delete(inbox_id: str) -> bool:
    _ensure_schema()
    with _DB_LOCK, _connection() as conn:
        cur = conn.execute(
            "DELETE FROM subscription_inbox WHERE inbox_id = ?",
            (inbox_id,),
        )
        return cur.rowcount > 0


def hot_board_upsert(entry: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_schema()
    symbol = entry["symbol"].upper()
    now = _now_iso()
    expires = (datetime.now(_SH_TZ) + timedelta(hours=12)).isoformat()
    source = entry["source"]
    new_sources = [source]
    wizz_json = json.dumps(entry["wizz"], ensure_ascii=False) if entry.get("wizz") is not None else None
    merger_json = json.dumps(entry["merger"], ensure_ascii=False) if entry.get("merger") is not None else None
    merged = entry.get("merged_for_sentiment")
    base_asset = entry.get("base_asset") or symbol.replace("USDT", "")

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
                 hit_count, wizz_json, merged_for_sentiment, merger_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    base_asset,
                    first_seen_at,
                    now,
                    expires,
                    json.dumps(sources, ensure_ascii=False),
                    hit_count,
                    wizz_json,
                    merged,
                    merger_json,
                ),
            )
        else:
            old_sources = json.loads(row["sources"] or "[]")
            sources = list(dict.fromkeys(old_sources + new_sources))
            hit_count = int(row["hit_count"] or 0) + 1
            first_seen_at = row["first_seen_at"]
            keep_wizz = wizz_json if source == "wizz_alert" else row["wizz_json"]
            keep_merged = merged if source == "wizz_alert" and merged is not None else row["merged_for_sentiment"]
            if source == "wizz_alert":
                keep_wizz = wizz_json
                keep_merged = merged
            keep_merger = merger_json if source == "merger_analyzer" else row["merger_json"]
            conn.execute(
                """
                UPDATE hot_board SET
                    base_asset = ?,
                    last_seen_at = ?,
                    expires_at = ?,
                    sources = ?,
                    hit_count = ?,
                    wizz_json = ?,
                    merged_for_sentiment = ?,
                    merger_json = ?
                WHERE symbol = ?
                """,
                (
                    base_asset,
                    now,
                    expires,
                    json.dumps(sources, ensure_ascii=False),
                    hit_count,
                    keep_wizz,
                    keep_merged,
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
    wizz = json.loads(row["wizz_json"]) if row["wizz_json"] else None
    merger = json.loads(row["merger_json"]) if row["merger_json"] else None
    sources = json.loads(row["sources"] or "[]")
    return {
        "symbol": row["symbol"],
        "base_asset": row["base_asset"],
        "first_seen_at": row["first_seen_at"],
        "last_seen_at": row["last_seen_at"],
        "expires_at": row["expires_at"],
        "sources": sources,
        "hit_count": row["hit_count"],
        "wizz": wizz,
        "merged_for_sentiment": row["merged_for_sentiment"],
        "merger": merger,
    }


def build_hot_board_supplement(entry: Dict[str, Any]) -> Dict[str, Any]:
    wizz = entry.get("wizz") or {}
    subscription = wizz.get("subscription") or {
        "type": "telegram_channel",
        "username": "wizzalert",
        "title": "Wizz 异动警报",
    }
    return {
        "sources": entry.get("sources") or [],
        "subscription": subscription,
        "wizz": wizz,
        "merger": entry.get("merger"),
        "hit_count": entry.get("hit_count"),
        "last_seen_at": entry.get("last_seen_at"),
        "permalink": wizz.get("permalink"),
        "merged_for_sentiment": entry.get("merged_for_sentiment") or "",
    }
