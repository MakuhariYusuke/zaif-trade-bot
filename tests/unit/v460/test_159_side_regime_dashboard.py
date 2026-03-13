"""159# P0-B/C: side_regime_dashboard 単体テスト."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.v460.analysis.side_regime_dashboard import (
    DashboardResult,
    _compute_side_metrics,
    run_dashboard,
)


class TestComputeSideMetrics:
    """_compute_side_metrics の計算正確性."""

    def test_basic_metrics(self) -> None:
        """基本的な3指標計算."""
        records = [
            {"filled": True, "post_fill_30s_pnl": 3.0, "adverse_selected": False, "side": "buy"},
            {"filled": True, "post_fill_30s_pnl": -1.0, "adverse_selected": True, "side": "buy"},
            {"filled": True, "post_fill_30s_pnl": 5.0, "adverse_selected": False, "side": "buy"},
            {"filled": True, "post_fill_30s_pnl": -2.0, "adverse_selected": True, "side": "buy"},
            {"filled": False, "side": "buy"},  # unfilled
        ]
        m = _compute_side_metrics(records)

        assert m["n_total"] == 5
        assert m["n_filled"] == 4
        assert m["fill_rate"] == pytest.approx(0.8, abs=0.001)
        # avg = (3 - 1 + 5 - 2) / 4 = 1.25
        assert m["avg_pnl30_bps"] == pytest.approx(1.25, abs=0.001)
        # profitable = 2/4 = 0.5
        assert m["profitable_rate"] == pytest.approx(0.5, abs=0.001)
        # AS rate = 2/4 = 0.5
        assert m["as_rate"] == pytest.approx(0.5, abs=0.001)

    def test_downside_p10(self) -> None:
        """p10 (worst decile) の計算."""
        # 10 records with PnL: -10, -5, -3, -1, 0, 1, 2, 3, 5, 10
        pnls = [-10.0, -5.0, -3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 10.0]
        records = [
            {"filled": True, "post_fill_30s_pnl": p, "adverse_selected": False}
            for p in pnls
        ]
        m = _compute_side_metrics(records)
        # p10 ≈ -8.5 (10th percentile of sorted array)
        assert m["downside_p10_bps"] < -5.0  # worst decile is very negative
        assert m["downside_p05_bps"] < m["downside_p10_bps"]  # p5 worse than p10

    def test_all_unfilled(self) -> None:
        """全レコード未約定."""
        records = [
            {"filled": False, "side": "sell"},
            {"filled": False, "side": "sell"},
        ]
        m = _compute_side_metrics(records)
        assert m["n_total"] == 2
        assert m["n_filled"] == 0
        assert m["fill_rate"] == pytest.approx(0.0, abs=0.001)
        assert m["avg_pnl30_bps"] == pytest.approx(0.0, abs=0.001)

    def test_no_as_records(self) -> None:
        """AS レコードなし."""
        records = [
            {"filled": True, "post_fill_30s_pnl": 2.0, "adverse_selected": False},
        ]
        m = _compute_side_metrics(records)
        assert m["as_rate"] == pytest.approx(0.0, abs=0.001)


class TestRunDashboard:
    """run_dashboard の統合テスト (tmp_path 利用)."""

    def test_with_sample_data(self, tmp_path: Path) -> None:
        """サンプルデータでダッシュボード生成."""
        records = [
            {"filled": True, "post_fill_30s_pnl": 2.0, "side": "buy",
             "adverse_selected": False, "regime": "ranging", "timestamp": 1771800000},
            {"filled": True, "post_fill_30s_pnl": -1.5, "side": "sell",
             "adverse_selected": True, "regime": "trending_down", "timestamp": 1771800100},
            {"filled": False, "side": "sell", "regime": "ranging", "timestamp": 1771800200},
        ]
        jsonl_path = tmp_path / "fill_records_20260223.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        result = run_dashboard(results_dir=str(tmp_path))

        assert result["total_records"] == 3
        assert result["total_filled"] == 2
        assert "buy" in result["side_summary"]
        assert "sell" in result["side_summary"]
        # buy: 1 filled / 1 total
        assert result["side_summary"]["buy"]["fill_rate"] == pytest.approx(1.0, abs=0.001)

    def test_trending_daily_populated(self, tmp_path: Path) -> None:
        """trending_down sell の日次集計が生成される."""
        records = [
            {"filled": True, "post_fill_30s_pnl": 5.0, "side": "sell",
             "adverse_selected": False, "regime": "trending_down", "timestamp": 1771800000},
            {"filled": True, "post_fill_30s_pnl": -2.0, "side": "sell",
             "adverse_selected": True, "regime": "trending_down", "timestamp": 1771800100},
        ]
        jsonl_path = tmp_path / "fill_records_20260223.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        result = run_dashboard(results_dir=str(tmp_path))
        assert len(result["trending_daily"]) >= 1
        assert result["trending_daily"][0]["n_filled"] == 2
