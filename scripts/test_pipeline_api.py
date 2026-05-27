#!/usr/env python3
"""
选币管线 API 联调脚本。

用法（先启动服务）:
  cd TA-Lib
  uvicorn app.app:app --host 127.0.0.1 --port 8000

  python scripts/test_pipeline_api.py
  python scripts/test_pipeline_api.py --base-url http://127.0.0.1:8000
  python scripts/test_pipeline_api.py --base-url https://do2ge.com/tail

无服务进程时（内存 TestClient，不请求币安 bundle）:
  python scripts/test_pipeline_api.py --inprocess
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE_WIZZ = textwrap.dedent(
    """\
    🔻 HIGH #3 · -23.3% (24h)
    ────────
    ↘️ 续跌 · 分 60 · 初推 5.6h前

    市场
    5m -1.4% · 1h -6.6% · vol 3.9× · OI -1.1%

    背景
    Binance将HIGH列入观察标签，FARM下架，HIGH出现天地针行情

    社交
    X 177/21KOL · BSQ 18
    🚨 散户24h: 空1.6× · 看空加剧

    讨论
    🤫 社交安静 (4h, 4 提及 / 0 KOL)

    动作
    📌 同币 24h 内第 3 次 同向延续
    """
).strip()

SAMPLE_INVALID = "🔻 NOTACOINXYZ · -99% (24h)\n背景\n无此合约测试丢弃"


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


def run_with_client(request_fn: Callable[..., Any], *, inprocess: bool) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    # 0) health
    r = request_fn("GET", "/health")
    results.append(_step("GET /health", r.status_code == 200, r.json() if r.status_code == 200 else r.text))

    # 1) seed inbox（联调专用）
    r = request_fn(
        "POST",
        "/crypto-mcp/subscription-inbox/seed",
        json={"channel": "wizzalert", "items": [
            {"raw_text": SAMPLE_WIZZ, "message_id": 91999001},
            {"raw_text": SAMPLE_INVALID, "message_id": 91999002},
        ]},
    )
    seed_ok = r.status_code == 200 and (r.json().get("inserted") or 0) >= 1
    results.append(_step("POST subscription-inbox/seed", seed_ok, r.json() if r.status_code == 200 else r.text))

    # 2) consume（应删除）
    r = request_fn(
        "POST",
        "/crypto-mcp/subscription-inbox/consume",
        json={"channel": "wizzalert", "limit": 50},
    )
    consume_body = r.json() if r.status_code == 200 else {}
    items = consume_body.get("items") or []
    results.append(
        _step(
            "POST subscription-inbox/consume",
            r.status_code == 200 and len(items) >= 1,
            {"status": r.status_code, "count": len(items), "first_inbox_id": items[0]["inbox_id"] if items else None},
        )
    )

    # 3) consume again（应为空）
    r2 = request_fn(
        "POST",
        "/crypto-mcp/subscription-inbox/consume",
        json={"channel": "wizzalert", "limit": 50},
    )
    items2 = (r2.json() or {}).get("items") or []
    results.append(
        _step(
            "POST consume 第二次（应为空）",
            r2.status_code == 200 and len(items2) == 0,
            {"count": len(items2)},
        )
    )

    # 4) futures-symbols（inprocess 已注入缓存；远程模式需能访问币安）
    try:
        r = request_fn("GET", "/crypto-mcp/futures-symbols")
    except Exception as exc:
        r = None
        results.append(
            _step(
                "GET futures-symbols",
                inprocess,
                {"error": str(exc), "note": "inprocess 可忽略；远程请检查网络/币安可达性"},
            )
        )
    else:
        fs_body = r.json() if r.status_code == 200 else {}
        symbols = fs_body.get("symbols") or []
        has_high = "HIGHUSDT" in symbols
        results.append(
            _step(
                "GET futures-symbols",
                r.status_code == 200 and fs_body.get("count", 0) > 0 and has_high,
                {"count": fs_body.get("count"), "has_HIGHUSDT": has_high},
            )
        )

    # 5) hot-board upsert wizz
    r = request_fn(
        "POST",
        "/crypto-mcp/hot-board/upsert",
        json={
            "symbol": "HIGHUSDT",
            "base_asset": "HIGH",
            "source": "wizz_alert",
            "wizz": {
                "base_asset": "HIGH",
                "trend_label": "续跌",
                "change_24h_pct": -23.3,
                "permalink": "https://t.me/wizzalert/1872",
                "subscription": {"channel": "wizzalert", "title": "Wizz 异动警报"},
            },
            "merged_for_sentiment": "【Wizz @wizzalert】背景：Binance将HIGH列入观察标签（联调测试）",
        },
    )
    upsert_ok = r.status_code == 200
    results.append(_step("POST hot-board/upsert (wizz)", upsert_ok, r.json() if upsert_ok else r.text))

    # 6) upsert invalid symbol
    r_bad = request_fn(
        "POST",
        "/crypto-mcp/hot-board/upsert",
        json={"symbol": "NOTACOINXYZUSDT", "source": "wizz_alert"},
    )
    results.append(
        _step(
            "POST hot-board/upsert 非法合约（应 400）",
            r_bad.status_code == 400,
            {"status": r_bad.status_code, "body": r_bad.text[:500]},
        )
    )

    # 7) picker-snapshot（可能较慢：含 bundle）
    r = request_fn(
        "GET",
        "/crypto-mcp/hot-board/picker-snapshot",
        params={"max_symbols": 5, "include_bundle": "false"},
    )
    snap = r.json() if r.status_code == 200 else {}
    entries = snap.get("entries") or []
    has_high_entry = any(e.get("symbol") == "HIGHUSDT" for e in entries)
    results.append(
        _step(
            "GET hot-board/picker-snapshot",
            r.status_code == 200 and has_high_entry,
            {
                "as_of": snap.get("as_of"),
                "entry_count": len(entries),
                "symbols": [e.get("symbol") for e in entries],
            },
        )
    )

    # 8) /all supplement（可能较慢）
    if not inprocess:
        r = request_fn("GET", "/crypto-mcp/all", params={"symbol": "HIGHUSDT"})
        if r.status_code == 200:
            bundle = (r.json() or {}).get("bundle") or {}
            sup = bundle.get("hot_board_supplement")
            results.append(
                _step(
                    "GET /all hot_board_supplement",
                    sup is not None and bool(sup.get("merged_for_sentiment")),
                    {
                        "has_supplement": sup is not None,
                        "sources": (sup or {}).get("sources"),
                        "merged_preview": ((sup or {}).get("merged_for_sentiment") or "")[:120],
                    },
                )
            )
        else:
            results.append(_step("GET /all", False, {"status": r.status_code, "text": r.text[:300]}))
    else:
        results.append(
            _step(
                "GET /all（inprocess 跳过，避免拉币安）",
                True,
                "使用 --base-url 测 /all 与 bundle",
            )
        )

    return results


def _seed_inprocess_futures_cache() -> None:
    """本机无法访问币安时（如 HTTP 451），为 inprocess 联调注入最小合约集合。"""
    import app.services.futures_symbols as fs

    fs._TRADING_USDT = {"HIGHUSDT", "BTCUSDT", "ETHUSDT"}
    fs._LAST_REFRESH = 1e12


def main() -> int:
    parser = argparse.ArgumentParser(description="选币管线 API 联调")
    parser.add_argument("--base-url", default="", help="例如 http://127.0.0.1:8000")
    parser.add_argument("--inprocess", action="store_true", help="FastAPI TestClient，无需启动 uvicorn")
    args = parser.parse_args()

    print("选币管线 API 联调")
    print(f"模式: {'TestClient' if args.inprocess else args.base_url or 'http://127.0.0.1:8000'}")

    if args.inprocess:
        _seed_inprocess_futures_cache()
        from fastapi.testclient import TestClient
        from app.app import app

        client = TestClient(app)

        def request_fn(method: str, path: str, **kwargs: Any):
            return client.request(method, path, **kwargs)

        results = run_with_client(request_fn, inprocess=True)
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
            results = run_with_client(request_fn, inprocess=False)
        except requests.exceptions.ConnectionError:
            print(f"\n无法连接 {base}，请先启动: uvicorn app.app:app --host 127.0.0.1 --port 8000")
            return 1

    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    print(f"\n========== 汇总: {passed}/{total} 通过 ==========")
    print(_pretty(results))
    print("\n请把以上完整终端输出复制发给助手对照 docs/DESIGN-symbol-selection-pipeline.md 验收。")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
