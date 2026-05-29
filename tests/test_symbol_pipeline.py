"""Tests for subscription inbox consume and hot board upsert."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from app.services import symbol_pipeline_store as store


class SymbolPipelineStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp_root = __import__("pathlib").Path(self._tmp.name)

        def _tmp_db_path() -> __import__("pathlib").Path:
            directory = tmp_root / "data" / "pipeline"
            directory.mkdir(parents=True, exist_ok=True)
            return directory / "pipeline.db"

        self._db_patch = patch(
            "app.services.symbol_pipeline_store._db_path",
            _tmp_db_path,
        )
        self._db_patch.start()
        store._SCHEMA_READY = False

    def tearDown(self) -> None:
        store._SCHEMA_READY = False
        self._db_patch.stop()
        self._tmp.cleanup()

    def test_inbox_consume_returns_only_raw_text(self) -> None:
        store.inbox_append(channel_username="wizzalert", raw_text="🔻 HIGH · -10% (24h)")
        items = store.inbox_consume(channel="wizzalert", limit=10)
        self.assertEqual(items, [{"raw_text": "🔻 HIGH · -10% (24h)"}])
        again = store.inbox_consume(channel="wizzalert", limit=10)
        self.assertEqual(again, [])

    def test_hot_board_wizz_upsert_alert_reason(self) -> None:
        store.hot_board_upsert(
            {
                "symbol": "ZZTESTHIGHUSDT",
                "source": "wizz_alert",
                "alert_reason": "old reason",
            }
        )
        store.hot_board_upsert(
            {
                "symbol": "ZZTESTHIGHUSDT",
                "source": "wizz_alert",
                "alert_reason": "new reason",
            }
        )
        entry = store.hot_board_get("ZZTESTHIGHUSDT")
        assert entry is not None
        self.assertEqual(entry["hit_count"], 2)
        self.assertEqual(entry["alert_reason"], "new reason")
        self.assertNotIn("wizz", entry)
        self.assertNotIn("merger", entry)

    def test_hot_board_merge_sources(self) -> None:
        store.hot_board_upsert(
            {
                "symbol": "BTCUSDT",
                "source": "merger_analyzer",
                "merger": {"rule": "test"},
            }
        )
        store.hot_board_upsert(
            {
                "symbol": "BTCUSDT",
                "source": "wizz_alert",
                "alert_reason": "wizz cleaned text",
            }
        )
        entry = store.hot_board_get("BTCUSDT")
        assert entry is not None
        self.assertEqual(set(entry["sources"]), {"merger_analyzer", "wizz_alert"})
        self.assertEqual(entry["alert_reason"], "wizz cleaned text")
        sup = store.build_hot_board_supplement(entry)
        self.assertEqual(sup, {"sources": ["merger_analyzer", "wizz_alert"], "alert_reason": "wizz cleaned text"})

    def test_pick_slot_and_cooldown(self) -> None:
        for sym in ("AAAUSDT", "BBBUSDT", "CCCUSDT"):
            store.hot_board_upsert({"symbol": sym, "source": "merger_analyzer"})
        store.pick_slot_commit(
            symbol="AAAUSDT",
            selection_context={"source": "hot-board-pick", "symbol": "AAAUSDT"},
            candidate_symbols=["AAAUSDT", "BBBUSDT", "CCCUSDT"],
        )
        pending = store.pick_slot_get(consume=False)
        self.assertEqual(pending["status"], "pending")
        self.assertEqual(pending["symbol"], "AAAUSDT")

        picker_rows = store.hot_board_list_for_picker(limit=10)
        picker_syms = {r["symbol"] for r in picker_rows}
        self.assertIn("AAAUSDT", picker_syms)
        self.assertNotIn("BBBUSDT", picker_syms)
        self.assertNotIn("CCCUSDT", picker_syms)

        claimed = store.pick_slot_get(consume=True)
        self.assertTrue(claimed.get("consumed"))
        empty = store.pick_slot_get(consume=False)
        self.assertEqual(empty["status"], "empty")

    def test_pick_slot_symbol_must_be_in_candidates(self) -> None:
        with self.assertRaises(ValueError):
            store.pick_slot_commit(
                symbol="ZZZUSDT",
                selection_context={"source": "hot-board-pick"},
                candidate_symbols=["AAAUSDT"],
            )


if __name__ == "__main__":
    unittest.main()
