#!/usr/env python3
"""
选币管线 API 联调脚本（只读 / 真实数据，不向生产写入假数据）。

用法:
  cd TA-Lib
  uv run python scripts/test_pipeline_api.py --base-url https://do2ge.com/tail
  uv run python scripts/test_pipeline_api.py --base-url http://127.0.0.1:8000

可选（会破坏性，消费收件箱里真实的待处理消息，供 ingest 前人工确认用）:
  uv run python scripts/test_pipeline_api.py --base-url ... --consume-inbox

说明:
  - 不调用 subscription-inbox/seed、不写入假热榜、不测试假合约名。
  - 收件箱积压请用日志 inbox append 或加 --consume-inbox（consume 后物理删除）。
  - --inprocess 仅适合本机无服务时冒烟 /health；完整验收请对真实部署 URL 执行。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _step(name: str, ok: bool, detail: Any) -> Dict[str, Any]:
    row = {"step": name, "ok": ok, "detail": detail}
    mark = "PASS" if ok else "FAIL"
    print(f"\n=== [{mark}] {name} ===")
    if isinstance(detail, (dict, list)):
        print(_pretty(detail))
    else:
        print(detail)
    return row


def _entry_summary(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "symbol": entry.get("symbol"),
        "sources": entry.get("sources"),
        "hit_count": entry.get("hit_count"),
        "has_wizz": bool(entry.get("wizz")),
        "has_merger": bool(entry.get("merger")),
        "has_merged_for_sentiment": bool((entry.get("merged_for_sentiment") or "").strip()),
    }


def run_live_checks(
    request_fn: Callable[..., Any],
    *,
    inprocess: bool,
    consume_inbox: bool,
    max_symbols: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    r = request_fn("GET", "/health")
    results.append(
        _step("GET /health", r.status_code == 200, r.json() if r.status_code == 200 else r.text)
    )

    if inprocess:
        results.append(
            _step(
                "完整管线验收",
                False,
                "请对真实部署执行: --base-url https://你的域名/tail（不要用 --inprocess 验热榜/币安）",
            )
        )
        return results

    r = request_fn("GET", "/crypto-mcp/futures-symbols")
    fs_body = r.json() if r.status_code == 200 else {}
    symbols = fs_body.get("symbols") or []
    results.append(
        _step(
            "GET futures-symbols",
            r.status_code == 200 and len(symbols) > 0,
            {"count": fs_body.get("count", len(symbols)), "sample": symbols[:5]},
        )
    )

    max_symbols = max(1, min(int(max_symbols), 50))  # 与 API Query le=50 一致
    r = request_fn(
        "GET",
        "/crypto-mcp/hot-board/picker-snapshot",
        params={"max_symbols": max_symbols, "include_bundle": "false"},
    )
    snap = r.json() if r.status_code == 200 else {}
    entries = snap.get("entries") or []
    summaries = [_entry_summary(e) for e in entries]
    http_ok = r.status_code == 200
    has_entries = len(entries) > 0
    results.append(
        _step(
            "GET hot-board/picker-snapshot（真实热榜）",
            http_ok and has_entries,
            {
                "http_status": r.status_code,
                "as_of": snap.get("as_of"),
                "entry_count": len(entries),
                "symbols": [e.get("symbol") for e in entries],
                "entries": summaries,
                "max_symbols_requested": max_symbols,
                "note": "entry_count=0 且 status=200 表示热榜确实空；422 表示参数错误（曾用 max>50）",
            },
        )
    )

    if not entries:
        results.append(
            _step(
                "GET /all hot_board_supplement",
                False,
                "热榜为空，无法继续；请确认 merger 日志或等待 Wizz/ingest 写入",
            )
        )
    else:
        probe = entries[0]
        symbol = (probe.get("symbol") or "").upper()
        r = request_fn("GET", "/crypto-mcp/all", params={"symbol": symbol})
        if r.status_code == 200:
            bundle = (r.json() or {}).get("bundle") or {}
            sup = bundle.get("hot_board_supplement")
            results.append(
                _step(
                    f"GET /all?symbol={symbol}（热榜首条）",
                    sup is not None,
                    {
                        "symbol": symbol,
                        "has_supplement": sup is not None,
                        "sources": (sup or {}).get("sources"),
                        "merged_for_sentiment_len": len(
                            ((sup or {}).get("merged_for_sentiment") or "")
                        ),
                        "permalink": (sup or {}).get("permalink"),
                        "entry_on_board": _entry_summary(probe),
                    },
                )
            )
        else:
            results.append(
                _step(f"GET /all?symbol={symbol}", False, {"status": r.status_code, "text": r.text[:300]}),
            )

        wizz_entries = [e for e in entries if e.get("wizz")]
        if wizz_entries:
            wizz_sym = (wizz_entries[0].get("symbol") or "").upper()
            if wizz_sym and wizz_sym != symbol:
                r2 = request_fn("GET", "/crypto-mcp/all", params={"symbol": wizz_sym})
                sup2 = ((r2.json() or {}).get("bundle") or {}).get("hot_board_supplement") if r2.status_code == 200 else None
                results.append(
                    _step(
                        f"GET /all?symbol={wizz_sym}（含 wizz 来源的一条）",
                        r2.status_code == 200 and sup2 is not None,
                        {
                            "symbol": wizz_sym,
                            "has_supplement": sup2 is not None,
                            "permalink": (sup2 or {}).get("permalink"),
                            "entry_on_board": _entry_summary(wizz_entries[0]),
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
        results.append(
            _step(
                "POST consume（真实收件箱，已物理删除）",
                r.status_code == 200,
                {
                    "count": len(items or []),
                    "message_ids": [it.get("message_id") for it in (items or [])[:10]],
                    "note": "count=0 表示当前无积压；>0 时请尽快跑 ingest 处理",
                },
            )
        )
    else:
        results.append(
            _step(
                "收件箱（未 consume）",
                True,
                "未消费收件箱。积压请: docker logs ta-lib-api | grep 'inbox append'；"
                "或 ingest 前加 --consume-inbox（会删除 raw）",
            )
        )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="选币管线 API 联调（仅真实数据）")
    parser.add_argument("--base-url", default="", help="例如 https://do2ge.com/tail")
    parser.add_argument(
        "--inprocess",
        action="store_true",
        help="仅本机 /health 冒烟；完整验收请用 --base-url",
    )
    parser.add_argument(
        "--consume-inbox",
        action="store_true",
        help="消费真实 wizzalert 收件箱（破坏性，不 seed 假数据）",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=50,
        help="picker-snapshot 最多返回条数（API 上限 50，默认 50）",
    )
    args = parser.parse_args()

    print("选币管线 API 联调（真实数据，无 seed/假币）")
    print(f"模式: {'TestClient' if args.inprocess else args.base_url or 'http://127.0.0.1:8000'}")
    if args.consume_inbox:
        print("警告: --consume-inbox 将删除收件箱中待处理的真实消息")

    if args.inprocess:
        from fastapi.testclient import TestClient
        from app.app import app

        client = TestClient(app)

        def request_fn(method: str, path: str, **kwargs: Any):
            return client.request(method, path, **kwargs)

        results = run_live_checks(
            request_fn, inprocess=True, consume_inbox=False, max_symbols=args.max_symbols
        )
    else:
        import requests

        base = (args.base_url or "http://127.0.0.1:8000").rstrip("/")
        session = requests.Session()

        class Resp:
            def __init__(self, r: requests.Response):
                self.status_code = r.status_code
                self.text = r.text
                self._r = r

            def json(self) -> Any:
                return self._r.json()

        def request_fn(method: str, path: str, **kwargs: Any):
            url = f"{base}{path}"
            if "params" in kwargs:
                return Resp(session.request(method, url, params=kwargs["params"], timeout=120))
            if "json" in kwargs:
                return Resp(session.request(method, url, json=kwargs["json"], timeout=120))
            return Resp(session.request(method, url, timeout=120))

        try:
            results = run_live_checks(
                request_fn,
                inprocess=False,
                consume_inbox=args.consume_inbox,
                max_symbols=args.max_symbols,
            )
        except requests.exceptions.ConnectionError:
            print(f"\n无法连接 {base}")
            return 1

    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    print(f"\n========== 汇总: {passed}/{total} 通过 ==========")
    print(_pretty(results))
    print("\n请把完整终端输出发给助手验收（对照 docs/DESIGN-symbol-selection-pipeline.md）。")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
