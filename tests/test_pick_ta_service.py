"""Tests for pick_ta summary builder."""

from __future__ import annotations

import unittest

from app.services.pick_ta_service import build_pick_ta_summary


class PickTaSummaryTests(unittest.TestCase):
    def test_build_pick_ta_summary_ok(self) -> None:
        bundle = {
            "technical_analysis": {
                "1h": {
                    "indicators": {
                        "adx": [28.0],
                        "atr": [0.5],
                        "rsi": [62.0],
                        "bb_u": [110.0],
                        "bb_l": [90.0],
                        "bb_m": [100.0],
                        "v": [1000.0],
                        "v_ma": [800.0],
                        "obv": [1.0],
                    }
                },
                "2h": {"indicators": {"adx": [25.0]}},
                "4h": {"error": "skip"},
            },
            "market_analysis": {
                "24h_stats": {"priceChangePercent": "-15.2", "lastPrice": "1.0"},
                "funding_rate": {"lastFundingRate": "0.0001"},
            },
        }
        summary = build_pick_ta_summary(bundle)
        self.assertTrue(summary["ta_available"])
        self.assertEqual(summary["intervals"]["1h"]["ok"], True)
        self.assertEqual(summary["intervals"]["1h"]["adx"], 28.0)
        self.assertAlmostEqual(summary["intervals"]["1h"]["bb_width_pct"], 20.0)
        self.assertEqual(summary["market"]["abs_priceChangePercent"], 15.2)

    def test_build_pick_ta_summary_unavailable(self) -> None:
        bundle = {
            "technical_analysis": {
                "1h": {"error": "fail"},
                "2h": {"error": "fail"},
                "4h": {"error": "fail"},
            },
            "market_analysis": {},
        }
        summary = build_pick_ta_summary(bundle)
        self.assertFalse(summary["ta_available"])


if __name__ == "__main__":
    unittest.main()
