#!/usr/bin/env python3
"""
选币管线 API 验收（仅 DESIGN §3 + skill 契约接口，真实 HTTPS）。

默认: https://do2ge.com/tail（与 hot-board-pick / crypto-subscription-ingest 一致）

  cd TA-Lib
  uv run python scripts/test_pipeline_api.py
  uv run python scripts/test_pipeline_api.py --log-only

验收接口（仅此四类，不测 /health、/gainers、seed、upsert）:
  - GET  /crypto-mcp/futures-symbols          （ingest 校验合约）
  - GET  /crypto-mcp/hot-board/picker-snapshot?max_symbols=10&include_bundle=false （hot-board-pick）
  - GET  /crypto-mcp/pick-slot                                              （crypto-post-flow Stage 0）
  - GET  /crypto-mcp/all?symbol=…             （crypto-analyst，验 hot_board_supplement）
  - POST /crypto-mcp/subscription-inbox/consume （仅 --consume-inbox；破坏性，默认跳过）

日志: logs/test_pipeline_api_<UTC>.log（gitignore）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
REAL_STDOUT = sys.__stdout__

DEFAULT_BASE = "https://do2ge.com/tail"
PICKER_MAX_SYMBOLS = 10  # hot-board-pick/SKILL.md
DEPRECATED_ENTRY_KEYS = ("wizz", "merged_for_sentiment", "merger")

PIPELINE_GET_PATHS = (
    "/crypto-mcp/futures-symbols",
    "/crypto-mcp/hot-board/picker-snapshot",
    "/crypto-mcp/pick-slot",
    "/crypto-mcp/all",
)
PIPELINE_POST_PATHS = ("/crypto-mcp/subscription-inbox/consume",)


def _pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


class _Tee(TextIO):
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


def _default_log_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return ROOT / "logs" / f"test_pipeline_api_{stamp}.log"


def _setup_logging(log_file: Path, *, log_only: bool) -> TextIO:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_file.open("w", encoding="utf-8")
    log_fp.write(
        f"# test_pipeline_api.py\n"
        f"# started: {datetime.now(timezone.utc).isoformat()}\n"
        f"# log: {log_file.resolve()}\n\n"
    )
    log_fp.flush()
    if log_only:
        sys.stdout = log_fp  # type: ignore[assignment]
    else:
        sys.stdout = _Tee(REAL_STDOUT, log_fp)  # type: ignore[assignment]
    return log_fp


def _step(name: str, ok: bool, detail: Any) -> Dict[str, Any]:
    row = {"step": name, "ok": ok, "detail": detail}
    mark = "PASS" if ok else "FAIL"
    print(f"\n=== [{mark}] {name} ===")
    if isinstance(detail, (dict, list)):
        print(_pretty(detail))
    else:
        print(detail)
    return row


class _Resp:
    def __init__(self, r: requests.Response):
        self.status_code = r.status_code
        self.text = r.text
        self._r = r

    def json(self) -> Any:
        return self._r.json()


def _request(session: requests.Session, base: str, method: str, path: str, **kwargs: Any) -> _Resp:
    if path not in PIPELINE_GET_PATHS and path not in PIPELINE_POST_PATHS:
        raise ValueError(f"非选币管线接口，禁止请求: {path}")
    url = f"{base.rstrip('/')}{path}"
    if "params" in kwargs:
        return _Resp(session.request(method, url, params=kwargs["params"], timeout=120))
    if "json" in kwargs:
        return _Resp(session.request(method, url, json=kwargs["json"], timeout=120))
    return _Resp(session.request(method, url, timeout=120))


def _hot_board_row_contract(entry: Dict[str, Any]) -> List[str]:
    """热榜行 §6 + HotBoardEntry 必填。"""
    issues: List[str] = []
    for key in ("symbol", "first_seen_at", "last_seen_at", "expires_at", "sources", "hit_count"):
        if entry.get(key) in (None, ""):
            issues.append(f"缺少 {key}")
    if not isinstance(entry.get("sources"), list):
        issues.append("sources 非数组")
    return issues


def _picker_entry_detail(entry: Dict[str, Any]) -> Dict[str, Any]:
    reason = (entry.get("alert_reason") or "").strip()
    bundle = entry.get("bundle") or {}
    sup = bundle.get("hot_board_supplement") or {}
    return {
        "symbol": entry.get("symbol"),
        "sources": entry.get("sources"),
        "hit_count": entry.get("hit_count"),
        "alert_reason_len": len(reason),
        "alert_reason_preview": reason[:240] + ("…" if len(reason) > 240 else ""),
        "bundle_present": bool(bundle),
        "bundle_has_technical": "technical_analysis" in bundle,
        "bundle_has_market": "market_analysis" in bundle,
        "bundle_has_sentiment": "sentiment_analysis" in bundle,
        "supplement_keys": sorted(sup.keys()) if isinstance(sup, dict) else [],
        "supplement_alert_reason_len": len((sup.get("alert_reason") or "").strip()),
        "contract_issues": _hot_board_row_contract(entry),
        "deprecated_fields_present": [k for k in DEPRECATED_ENTRY_KEYS if entry.get(k)],
    }


def _validate_picker_snapshot_body(snap: Dict[str, Any], *, expect_count: int) -> Dict[str, Any]:
    entries = snap.get("entries") or []
    issues: List[str] = []
    if snap.get("board_ttl_hours") != 12:
        issues.append(f"board_ttl_hours 应为 12，实际 {snap.get('board_ttl_hours')}")
    if len(entries) == 0:
        issues.append("entries 为空 → hot-board-pick 将 no_data")
    if len(entries) > expect_count:
        issues.append(f"entries 条数 {len(entries)} > max_symbols={expect_count}")

    bundle_present: List[str] = []
    wizz_rows: List[Dict[str, Any]] = []
    for e in entries:
        sym = e.get("symbol") or "?"
        issues.extend(f"{sym}: {x}" for x in _hot_board_row_contract(e))
        for bad in DEPRECATED_ENTRY_KEYS:
            if e.get(bad):
                issues.append(f"{sym}: 不应返回已废弃字段 {bad}")
        if e.get("bundle"):
            bundle_present.append(sym)
        if "wizz_alert" in (e.get("sources") or []):
            wizz_rows.append(_picker_entry_detail(e))
            if not (e.get("alert_reason") or "").strip():
                issues.append(f"{sym}: sources 含 wizz_alert 但 alert_reason 为空")

    return {
        "as_of": snap.get("as_of"),
        "entry_count": len(entries),
        "symbols": [e.get("symbol") for e in entries],
        "bundle_present_on_entries": bundle_present,
        "wizz_alert_entry_count": len(wizz_rows),
        "wizz_alert_samples": wizz_rows[:2],
        "first_entry": _picker_entry_detail(entries[0]) if entries else None,
        "validation_issues": issues,
        "params_used": {"max_symbols": expect_count, "include_bundle": False},
    }


def run_pipeline_checks(
    request_fn: Callable[..., _Resp],
    *,
    consume_inbox: bool,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    r = request_fn("GET", "/crypto-mcp/futures-symbols")
    fs_body = r.json() if r.status_code == 200 else {}
    symbols = fs_body.get("symbols") or []
    results.append(
        _step(
            "GET /crypto-mcp/futures-symbols（ingest 契约）",
            r.status_code == 200 and len(symbols) > 0,
            {
                "http_status": r.status_code,
                "count": fs_body.get("count", len(symbols)),
                "sample": symbols[:5],
            },
        )
    )

    r = request_fn(
        "GET",
        "/crypto-mcp/hot-board/picker-snapshot",
        params={"max_symbols": PICKER_MAX_SYMBOLS, "include_bundle": "false"},
    )
    snap = r.json() if r.status_code == 200 else {}
    entries = snap.get("entries") or []
    validation = _validate_picker_snapshot_body(snap, expect_count=PICKER_MAX_SYMBOLS)
    picker_ok = (
        r.status_code == 200
        and len(entries) > 0
        and not validation["validation_issues"]
        and snap.get("cooldown_filtered") is True
    )
    results.append(
        _step(
            "GET /crypto-mcp/hot-board/picker-snapshot（hot-board-pick 契约）",
            picker_ok,
            {"http_status": r.status_code, **validation},
        )
    )

    r_slot = request_fn("GET", "/crypto-mcp/pick-slot")
    slot = r_slot.json() if r_slot.status_code == 200 else {}
    slot_ok = r_slot.status_code == 200 and slot.get("status") in ("empty", "pending")
    results.append(
        _step(
            "GET /crypto-mcp/pick-slot（crypto-post-flow Stage 0）",
            slot_ok,
            {
                "http_status": r_slot.status_code,
                "status": slot.get("status"),
                "symbol": slot.get("symbol"),
                "note": "pending 时 flow 可 consume=true 认领；empty 须先跑 hot-board-pick",
            },
        )
    )

    if not entries:
        results.append(
            _step(
                "GET /crypto-mcp/all?symbol=…（crypto-analyst 契约）",
                False,
                "热榜为空，跳过 /all；请先 Merger 或 ingest 入榜",
            )
        )
    else:
        merger_probe = next((e for e in entries if "wizz_alert" not in (e.get("sources") or [])), entries[0])
        wizz_probe = next(
            (e for e in entries if "wizz_alert" in (e.get("sources") or [])),
            None,
        )

        for label, entry in (
            ("merger 样本", merger_probe),
            ("wizz 样本", wizz_probe),
        ):
            if entry is None:
                results.append(
                    _step(
                        f"GET /crypto-mcp/all?symbol=…（{label}）",
                        True,
                        "当前热榜无 wizz_alert 行（仅 merger）；ingest 入榜后应有 alert_reason",
                    )
                )
                continue
            symbol = (entry.get("symbol") or "").upper()
            r_all = request_fn("GET", "/crypto-mcp/all", params={"symbol": symbol})
            body = r_all.json() if r_all.status_code == 200 else {}
            bundle = body.get("bundle") or {}
            sup = bundle.get("hot_board_supplement") or {}
            reason_sup = (sup.get("alert_reason") or "").strip()
            reason_entry = (entry.get("alert_reason") or "").strip()
            has_wizz = "wizz_alert" in (entry.get("sources") or [])
            all_ok = r_all.status_code == 200 and isinstance(sup, dict)
            if has_wizz:
                all_ok = all_ok and bool(reason_sup) and reason_sup == reason_entry
            results.append(
                _step(
                    f"GET /crypto-mcp/all?symbol={symbol}（{label}）",
                    all_ok,
                    {
                        "http_status": r_all.status_code,
                        "symbol": symbol,
                        "picker_entry": _picker_entry_detail(entry),
                        "supplement_sources": sup.get("sources"),
                        "supplement_alert_reason_len": len(reason_sup),
                        "supplement_alert_reason_preview": reason_sup[:240]
                        + ("…" if len(reason_sup) > 240 else ""),
                    },
                )
            )

    if consume_inbox:
        r = request_fn(
            "POST",
            "/crypto-mcp/subscription-inbox/consume",
            json={"channel": "wizzalert", "limit": 50},
        )
        items = (r.json() or {}).get("items") if r.status_code == 200 else []
        sample = items[:2] if items else []
        item_ok = all(set(it.keys()) <= {"raw_text"} and it.get("raw_text") for it in sample) if sample else True
        extra_keys = [sorted(it.keys()) for it in sample if set(it.keys()) != {"raw_text"}]
        results.append(
            _step(
                "POST /crypto-mcp/subscription-inbox/consume（ingest 契约）",
                r.status_code == 200 and item_ok and not extra_keys,
                {
                    "http_status": r.status_code,
                    "count": len(items or []),
                    "extra_keys_on_items": extra_keys,
                    "sample_raw_text_len": [len((it.get("raw_text") or "")) for it in sample],
                    "note": "已物理删除 raw；count>0 须尽快跑 crypto-subscription-ingest",
                },
            )
        )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="选币管线 API 验收（仅 §3 接口，默认 https://do2ge.com/tail）",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE,
        help=f"TA-Lib 前缀（默认 {DEFAULT_BASE}）",
    )
    parser.add_argument(
        "--consume-inbox",
        action="store_true",
        help="额外测 POST consume（破坏性：取出并删除真实收件箱）",
    )
    parser.add_argument("--log-file", default="", help="日志路径")
    parser.add_argument("--log-only", action="store_true", help="详情只写日志")
    parser.add_argument("--no-log", action="store_true", help="不写日志")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    host = urlparse(base).netloc.lower()
    if host in ("127.0.0.1", "localhost") or host.startswith("127."):
        REAL_STDOUT.write("错误: 本脚本仅验收生产/部署 URL，请使用 https://do2ge.com/tail\n")
        return 2

    log_fp: Optional[TextIO] = None
    log_path: Optional[Path] = None
    if not args.no_log:
        log_path = Path(args.log_file).expanduser() if args.log_file else _default_log_path()
        if not log_path.is_absolute():
            log_path = ROOT / log_path
        log_fp = _setup_logging(log_path, log_only=args.log_only)
        REAL_STDOUT.write(f"日志: {log_path.resolve()}\n")
        REAL_STDOUT.flush()

    print("选币管线 API 验收")
    print(f"Base: {base}")
    print(f"接口范围: GET {', '.join(PIPELINE_GET_PATHS)}")
    if args.consume_inbox:
        print(f"         POST {PIPELINE_POST_PATHS[0]}（破坏性）")
    else:
        print("         POST consume 未测（加 --consume-inbox 可测）")

    session = requests.Session()
    try:

        def request_fn(method: str, path: str, **kwargs: Any) -> _Resp:
            return _request(session, base, method, path, **kwargs)

        results = run_pipeline_checks(request_fn, consume_inbox=args.consume_inbox)
    except requests.exceptions.ConnectionError as exc:
        print(f"\n无法连接 {base}: {exc}")
        exit_code = 1
        results = []
    else:
        passed = sum(1 for r in results if r["ok"])
        total = len(results)
        print(f"\n========== 汇总: {passed}/{total} 通过 ==========")
        print(_pretty(results))
        exit_code = 0 if passed == total else 1

    if log_fp is not None:
        passed = sum(1 for r in results if r["ok"])
        total = len(results)
        sys.stdout = REAL_STDOUT  # type: ignore[assignment]
        log_fp.flush()
        log_fp.close()
        REAL_STDOUT.write(f"\n汇总: {passed}/{total} 通过\n日志: {log_path.resolve()}\n")
        REAL_STDOUT.flush()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
