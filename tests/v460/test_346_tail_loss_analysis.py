"""346# S-7: tail_loss_analysis 単体テスト."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

# --- テスト対象のインポート ---
from scripts.v460.analysis.tail_loss_analysis import (
    _as_rate,
    _compute_hour_overrep,
    _compute_overrep,
    _derive_actionable_filters,
    _extract_filled,
    _flag_rate,
    _numeric_field_stats,
    _pnl_array,
    _record_to_utc_hour,
    analyze_tail_loss,
)


# ======================================================================
# Fixtures
# ======================================================================

def _make_record(
    *,
    side: str = "buy",
    filled: bool = True,
    pnl: float | None = 0.0,
    regime: str | None = "trending",
    timestamp: float = 1709251200.0,  # 2024-03-01 00:00 UTC
    adverse_selected: bool = False,
    decision_path: str = "primary_only",
    spread_at_order: float | None = 100.0,
    mid_price_trend_5s: float | None = 0.5,
    orderbook_imbalance: float | None = 0.1,
    skip_gate_score: float | None = 0.3,
    early_exit_triggered: bool | None = None,
    balance_forced_switch: bool | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """テスト用 fill record を生成."""
    return {
        "cycle_id": "test_cycle",
        "side": side,
        "filled": filled,
        "post_fill_30s_pnl": pnl,
        "regime": regime,
        "timestamp": timestamp,
        "adverse_selected": adverse_selected,
        "decision_path": decision_path,
        "spread_at_order": spread_at_order,
        "mid_price_trend_5s": mid_price_trend_5s,
        "orderbook_imbalance": orderbook_imbalance,
        "skip_gate_score": skip_gate_score,
        "early_exit_triggered": early_exit_triggered,
        "balance_forced_switch": balance_forced_switch,
        **extra,
    }


def _make_sample_records(n: int = 100, side: str = "buy") -> list[dict[str, Any]]:
    """n 件のサンプルレコードを生成 (p10 以下に悪い PnL を集中)."""
    rng = np.random.default_rng(42)
    records: list[dict[str, Any]] = []
    for i in range(n):
        pnl = float(rng.normal(0.5, 2.0))
        if i < n // 10:
            # テール: 悪い PnL
            pnl = float(rng.normal(-8.0, 2.0))
            regime = "high_vol"
            hour = 3  # UTC 03:00 に集中
            as_flag = True
        else:
            regime = rng.choice(["trending", "ranging", "high_vol"])
            hour = int(rng.integers(0, 24))
            as_flag = bool(rng.random() < 0.3)

        ts = 1709251200.0 + hour * 3600  # hour を timestamp にエンコード
        records.append(_make_record(
            side=side,
            pnl=pnl,
            regime=str(regime),
            timestamp=ts,
            adverse_selected=as_flag,
            spread_at_order=float(rng.uniform(50, 300)),
            mid_price_trend_5s=float(rng.normal(0, 1.5)),
            orderbook_imbalance=float(rng.uniform(-1, 1)),
        ))
    return records


# ======================================================================
# ヘルパー関数テスト
# ======================================================================

class TestExtractFilled:
    """_extract_filled のテスト."""

    def test_filters_unfilled(self) -> None:
        records = [
            _make_record(filled=True, side="buy"),
            _make_record(filled=False, side="buy"),
            _make_record(filled=True, side="sell"),
        ]
        result = _extract_filled(records)
        assert len(result) == 2

    def test_filters_by_side(self) -> None:
        records = [
            _make_record(filled=True, side="buy"),
            _make_record(filled=True, side="sell"),
            _make_record(filled=True, side="buy"),
        ]
        assert len(_extract_filled(records, side="buy")) == 2
        assert len(_extract_filled(records, side="sell")) == 1

    def test_empty_input(self) -> None:
        assert _extract_filled([]) == []


class TestPnlArray:
    """_pnl_array のテスト."""

    def test_extracts_values(self) -> None:
        records = [
            _make_record(pnl=1.0),
            _make_record(pnl=-2.0),
            _make_record(pnl=None),
        ]
        arr = _pnl_array(records)
        assert len(arr) == 2
        np.testing.assert_array_almost_equal(arr, [1.0, -2.0])

    def test_empty_returns_empty_array(self) -> None:
        arr = _pnl_array([])
        assert len(arr) == 0
        assert arr.dtype == float

    def test_all_none(self) -> None:
        records = [_make_record(pnl=None), _make_record(pnl=None)]
        arr = _pnl_array(records)
        assert len(arr) == 0


class TestAsRate:
    """_as_rate のテスト."""

    def test_basic(self) -> None:
        records = [
            _make_record(adverse_selected=True),
            _make_record(adverse_selected=False),
            _make_record(adverse_selected=True),
            _make_record(adverse_selected=False),
        ]
        assert _as_rate(records) == pytest.approx(0.5)

    def test_empty(self) -> None:
        assert _as_rate([]) == 0.0


class TestFlagRate:
    """_flag_rate のテスト."""

    def test_with_values(self) -> None:
        records = [
            _make_record(early_exit_triggered=True),
            _make_record(early_exit_triggered=False),
            _make_record(early_exit_triggered=True),
        ]
        assert _flag_rate(records, "early_exit_triggered") == pytest.approx(2 / 3)

    def test_all_none(self) -> None:
        records = [
            _make_record(early_exit_triggered=None),
            _make_record(early_exit_triggered=None),
        ]
        assert _flag_rate(records, "early_exit_triggered") is None

    def test_mixed_none(self) -> None:
        records = [
            _make_record(early_exit_triggered=True),
            _make_record(early_exit_triggered=None),
        ]
        assert _flag_rate(records, "early_exit_triggered") == pytest.approx(1.0)


class TestRecordToUtcHour:
    """_record_to_utc_hour のテスト."""

    def test_epoch_float(self) -> None:
        r = _make_record(timestamp=1709251200.0)  # 2024-03-01 00:00 UTC
        assert _record_to_utc_hour(r) == 0

    def test_iso_string(self) -> None:
        r: dict[str, Any] = {"timestamp": "2024-03-01T15:30:00+00:00"}
        assert _record_to_utc_hour(r) == 15

    def test_none(self) -> None:
        r: dict[str, Any] = {"timestamp": None}
        assert _record_to_utc_hour(r) is None

    def test_missing_key(self) -> None:
        assert _record_to_utc_hour({}) is None


# ======================================================================
# Over-representation テスト
# ======================================================================

class TestComputeOverrep:
    """_compute_overrep のテスト."""

    def test_basic_overrep(self) -> None:
        # tail に high_vol が集中
        tail = [_make_record(regime="high_vol")] * 5 + [_make_record(regime="trending")] * 1
        all_recs = (
            [_make_record(regime="high_vol")] * 10
            + [_make_record(regime="trending")] * 40
        )
        result = _compute_overrep(tail, all_recs, "regime")

        # high_vol: tail_share = 5/6 ≈ 0.833, total_share = 10/50 = 0.2
        # overrep = 0.833 / 0.2 ≈ 4.17
        assert result["high_vol"]["overrep_ratio"] > 3.0
        assert result["trending"]["overrep_ratio"] < 1.0

    def test_empty_tail(self) -> None:
        result = _compute_overrep([], [_make_record()], "regime")
        assert result == {}


class TestComputeHourOverrep:
    """_compute_hour_overrep のテスト."""

    def test_sorted_descending(self) -> None:
        # UTC 03:00 がテールに集中する設定
        tail = [_make_record(timestamp=1709262000.0)] * 3  # UTC 03:00
        all_recs = (
            [_make_record(timestamp=1709262000.0)] * 5  # UTC 03:00
            + [_make_record(timestamp=1709280000.0)] * 20  # UTC 08:00
        )
        result = _compute_hour_overrep(tail, all_recs)
        assert len(result) > 0
        # 降順ソート確認
        ratios = [e["overrep_ratio"] for e in result]
        assert ratios == sorted(ratios, reverse=True)

    def test_empty(self) -> None:
        assert _compute_hour_overrep([], []) == []


# ======================================================================
# 数値特徴量統計テスト
# ======================================================================

class TestNumericFieldStats:
    """_numeric_field_stats のテスト."""

    def test_basic_stats(self) -> None:
        tail = [_make_record(spread_at_order=200.0)] * 10
        total = [_make_record(spread_at_order=100.0)] * 50
        result = _numeric_field_stats(tail, total, "spread_at_order")
        assert result["tail_mean"] == pytest.approx(200.0)
        assert result["total_mean"] == pytest.approx(100.0)

    def test_none_values(self) -> None:
        tail = [_make_record(spread_at_order=None)]
        total = [_make_record(spread_at_order=None)]
        result = _numeric_field_stats(tail, total, "spread_at_order")
        assert result["tail_mean"] is None
        assert result["total_mean"] is None


# ======================================================================
# アクション可能フィルタテスト
# ======================================================================

class TestDeriveActionableFilters:
    """_derive_actionable_filters のテスト."""

    def test_detects_regime_skip(self) -> None:
        # high_vol が tail に集中: overrep > 1.5
        regime_overrep = {
            "high_vol": {
                "tail_n": 5,
                "total_n": 10,
                "tail_share": 0.83,
                "total_share": 0.2,
                "overrep_ratio": 4.15,
            },
        }
        hour_overrep: list[Any] = []
        tail = [_make_record(regime="high_vol")] * 5
        all_recs = [_make_record()] * 50

        proposals = _derive_actionable_filters(tail, all_recs, regime_overrep, hour_overrep)
        regime_proposals = [p for p in proposals if p["type"] == "regime_skip"]
        assert len(regime_proposals) >= 1
        assert regime_proposals[0]["tail_avoided"] == 5

    def test_empty_if_no_overrep(self) -> None:
        regime_overrep = {
            "trending": {
                "tail_n": 3,
                "total_n": 30,
                "tail_share": 0.3,
                "total_share": 0.3,
                "overrep_ratio": 1.0,
            },
        }
        proposals = _derive_actionable_filters([], [], regime_overrep, [])
        assert proposals == []

    def test_sorted_by_efficiency(self) -> None:
        records = _make_sample_records(200)
        tail = records[:20]  # worst 10%
        regime_overrep = _compute_overrep(tail, records, "regime")
        hour_overrep = _compute_hour_overrep(tail, records)
        proposals = _derive_actionable_filters(tail, records, regime_overrep, hour_overrep)
        if len(proposals) >= 2:
            efficiencies = [float(p["efficiency"]) for p in proposals]
            assert efficiencies == sorted(efficiencies, reverse=True)


# ======================================================================
# メイン分析関数テスト
# ======================================================================

class TestAnalyzeTailLoss:
    """analyze_tail_loss のテスト."""

    def test_basic_analysis(self) -> None:
        records = _make_sample_records(100, side="buy")
        result = analyze_tail_loss(records, percentile=10.0)
        assert "buy" in result
        assert "sell" in result
        buy = result["buy"]
        assert buy["n"] == 100
        assert buy["tail_n"] > 0
        assert buy["tail_threshold_bps"] < 0  # テールは負

    def test_insufficient_data(self) -> None:
        records = [_make_record(side="sell", pnl=-1.0)] * 5
        result = analyze_tail_loss(records, percentile=10.0)
        assert "insufficient" in result["sell"].get("message", "")

    def test_both_sides(self) -> None:
        buy_records = _make_sample_records(50, side="buy")
        sell_records = _make_sample_records(50, side="sell")
        records = buy_records + sell_records
        result = analyze_tail_loss(records, percentile=10.0)
        assert result["buy"]["n"] == 50
        assert result["sell"]["n"] == 50

    def test_tail_threshold_negative_for_losses(self) -> None:
        """テール閾値は損失側にあるため負値."""
        records = _make_sample_records(200, side="buy")
        result = analyze_tail_loss(records, percentile=10.0)
        # 上位10%のレコードは pnl=-8.0 ± 2.0 なので threshold は確実に負
        assert result["buy"]["tail_threshold_bps"] < 0

    def test_as_overrep_computed(self) -> None:
        """テールの AS rate が全体より高い場合 overrep > 1."""
        records = _make_sample_records(100, side="buy")
        result = analyze_tail_loss(records, percentile=10.0)
        buy = result["buy"]
        # テスト records は tail に AS=True を集中させている
        assert buy["as_rate_tail"] > buy["as_rate_total"]
        assert buy.get("as_overrep") is not None
        assert buy["as_overrep"] > 1.0

    def test_regime_overrep_present(self) -> None:
        """regime_overrep が正しく計算される."""
        records = _make_sample_records(100, side="buy")
        result = analyze_tail_loss(records, percentile=10.0)
        assert len(result["buy"].get("regime_overrep", {})) > 0

    def test_actionable_filters_included(self) -> None:
        """actionable_filters が結果に含まれる."""
        records = _make_sample_records(200, side="buy")
        result = analyze_tail_loss(records, percentile=10.0)
        # actionable_filters フィールドの存在確認
        assert "actionable_filters" in result["buy"]

    def test_custom_percentile(self) -> None:
        """p5 指定での分析."""
        records = _make_sample_records(200, side="buy")
        result_p10 = analyze_tail_loss(records, percentile=10.0)
        result_p5 = analyze_tail_loss(records, percentile=5.0)
        # p5 のテール閾値は p10 より厳しい (より負)
        assert result_p5["buy"]["tail_threshold_bps"] <= result_p10["buy"]["tail_threshold_bps"]


# ======================================================================
# CLI テスト (main のスモークテスト)
# ======================================================================

class TestMainCLI:
    """main() のスモークテスト."""

    def test_main_with_empty_dir(self, tmp_path: Path) -> None:
        """空ディレクトリでもクラッシュしない."""
        import argparse
        args = argparse.Namespace(
            results_dir=str(tmp_path),
            percentile=10.0,
            git_sha=None,
            date_from=None,
            date_to=None,
            output=str(tmp_path / "test_output.json"),
        )
        from scripts.v460.analysis.tail_loss_analysis import main
        result = main(args)
        assert "sell" in result
        assert "buy" in result
        # JSON 出力確認
        assert (tmp_path / "test_output.json").exists()
        data = json.loads((tmp_path / "test_output.json").read_text(encoding="utf-8"))
        assert data["script"] == "346# S-7 tail_loss_analysis"
