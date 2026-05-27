"""Tests for subscription inbox consume and hot board upsert."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from app.services import symbol_pipeline_store as store


class SymbolPipelineStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._db_patch = patch(
            "app.services.kline_chart_service._repo_root",
            return_value=__import__("pathlib").Path(self._tmp.name),
        )
        self._db_patch.start()
        store._SCHEMA_READY = False

    def tearDown(self) -> None:
        store._SCHEMA_READY = False
        self._db_patch.stop()
        self._tmp.cleanup()

    def test_inbox_consume_deletes_rows(self) -> None:
        store.inbox_append(
            channel_username="wizzalert",
            message_id=1,
            permalink="https://t.me/wizzalert/1",
            raw_text="🔻 HIGH · -10% (24h)",
        )
        items = store.inbox_consume(channel="wizzalert", limit=10)
        self.assertEqual(len(items), 1)
        again = store.inbox_consume(channel="wizzalert", limit=10)
        self.assertEqual(again, [])

    def test_hot_board_upsert_merge_latest(self) -> None:
        store.hot_board_upsert(
            {
                "symbol": "HIGHUSDT",
                "base_asset": "HIGH",
                "source": "wizz_alert",
                "wizz": {"score": 50},
                "merged_for_sentiment": "old",
            }
        )
        store.hot_board_upsert(
            {
                "symbol": "HIGHUSDT",
                "base_asset": "HIGH",
                "source": "wizz_alert",
                "wizz": {"score": 60},
                "merged_for_sentiment": "new",
            }
        )
        entry = store.hot_board_get("HIGHUSDT")
        assert entry is not None
        self.assertEqual(entry["hit_count"], 2)
        self.assertIn("wizz_alert", entry["sources"])
        self.assertEqual(entry["wizz"]["score"], 60)
        self.assertEqual(entry["merged_for_sentiment"], "new")

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
                "wizz": {"trend_label": "续涨"},
                "merged_for_sentiment": "wizz text",
            }
        )
        entry = store.hot_board_get("BTCUSDT")
        assert entry is not None
        self.assertEqual(set(entry["sources"]), {"merger_analyzer", "wizz_alert"})
        self.assertEqual(entry["merger"]["rule"], "test")
        self.assertEqual(entry["wizz"]["trend_label"], "续涨")


if __name__ == "__main__":
    unittest.main()
