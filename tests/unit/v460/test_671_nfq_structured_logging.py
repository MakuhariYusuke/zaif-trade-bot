"""671# NFQ 構造化ログのテスト.

InfeasibleQuoteError の構造化属性、FillRecord のフィールド、
_make_price_error_skip の連携、分析スクリプトの NFQ セクションをテスト。
"""

from __future__ import annotations

import json

import numpy as np
import pytest


# ======================================================================
# InfeasibleQuoteError 構造化属性テスト
# ======================================================================


class TestInfeasibleQuoteErrorStructured:
    """InfeasibleQuoteError に 671# で追加した構造化属性のテスト."""

    def test_structured_fields_present(self) -> None:
        """構造化フィールドが正しく保持される."""
        from scripts.v460.lib.maker_price import InfeasibleQuoteError

        e = InfeasibleQuoteError(
            reason="spread_too_narrow",
            msg="Spread too narrow: 2000 JPY < min 3200",
            actual_spread=2000.0,
            min_spread_effective=3200.0,
            min_spread_abs=100.0,
            min_spread_atr=3200.0,
            sigma=0.000985,
        )
        assert e.reason == "spread_too_narrow"
        assert e.actual_spread == 2000.0
        assert e.min_spread_effective == 3200.0
        assert e.min_spread_abs == 100.0
        assert e.min_spread_atr == 3200.0
        assert e.sigma == 0.000985
        assert "2000" in str(e)

    def test_defaults_zero(self) -> None:
        """デフォルト値は旧来の呼び出しと後方互換."""
        from scripts.v460.lib.maker_price import InfeasibleQuoteError

        e = InfeasibleQuoteError(reason="spread_too_narrow", msg="test")
        assert e.actual_spread == 0.0
        assert e.min_spread_effective == 0.0
        assert e.min_spread_abs == 0.0
        assert e.min_spread_atr == 0.0
        assert e.sigma == 0.0

    def test_is_value_error_subclass(self) -> None:
        """ValueError サブクラス互換が維持されている."""
        from scripts.v460.lib.maker_price import InfeasibleQuoteError

        e = InfeasibleQuoteError(reason="test", msg="msg")
        assert isinstance(e, ValueError)


# ======================================================================
# FillRecord NFQ フィールドテスト
# ======================================================================


class TestFillRecordNfqFields:
    """FillRecord に 671# で追加した nfq_* フィールドのテスト."""

    def test_nfq_fields_exist(self) -> None:
        """nfq_* フィールドが FillRecord に存在する."""
        from ztb.metrics.fill_quality import FillRecord

        r = FillRecord(
            cycle_id="test",
            timestamp=1000000.0,
            side="buy",
            order_price=10000000.0,
            order_quantity=0.001,
        )
        assert r.nfq_actual_spread is None
        assert r.nfq_min_spread_effective is None
        assert r.nfq_min_spread_abs is None
        assert r.nfq_min_spread_atr is None
        assert r.nfq_sigma is None

    def test_nfq_fields_in_dict(self) -> None:
        """to_dict() / from_dict() で nfq_* がラウンドトリップする."""
        from ztb.metrics.fill_quality import FillRecord

        r = FillRecord(
            cycle_id="test",
            timestamp=1000000.0,
            side="buy",
            order_price=10000000.0,
            order_quantity=0.001,
            nfq_actual_spread=2000.0,
            nfq_min_spread_effective=3200.0,
            nfq_min_spread_abs=100.0,
            nfq_min_spread_atr=3100.0,
            nfq_sigma=0.000985,
        )
        d = r.to_dict()
        assert d["nfq_actual_spread"] == 2000.0
        assert d["nfq_min_spread_effective"] == 3200.0
        assert d["nfq_min_spread_atr"] == 3100.0
        assert d["nfq_sigma"] == 0.000985

        r2 = FillRecord.from_dict(d)
        assert r2.nfq_actual_spread == 2000.0
        assert r2.nfq_min_spread_effective == 3200.0

    def test_nfq_fields_json_serializable(self) -> None:
        """nfq_* フィールドが JSON シリアライズ可能."""
        from ztb.metrics.fill_quality import FillRecord

        r = FillRecord(
            cycle_id="test",
            timestamp=1000000.0,
            side="buy",
            order_price=10000000.0,
            order_quantity=0.001,
            nfq_actual_spread=2054.0,
            nfq_sigma=0.000985,
        )
        s = json.dumps(r.to_dict())
        d2 = json.loads(s)
        assert d2["nfq_actual_spread"] == 2054.0
        assert d2["nfq_sigma"] == 0.000985


# ======================================================================
# build_skip_fill_record NFQ 連携テスト
# ======================================================================


class TestBuildSkipRecordNfq:
    """build_skip_fill_record が nfq_* を extra 経由で受け取れる."""

    def test_nfq_fields_via_extra(self) -> None:
        from ztb.metrics.fill_quality import build_skip_fill_record

        r = build_skip_fill_record(
            cycle_id="test_nfq",
            timestamp=1000000.0,
            side="buy",
            order_price=10000000.0,
            order_quantity=0.001,
            cancel_reason="no_feasible_quote",
            run_id="test_run",
            git_sha="abc123",
            nfq_actual_spread=2054.0,
            nfq_min_spread_effective=3202.0,
            nfq_min_spread_abs=100.0,
            nfq_min_spread_atr=3202.0,
            nfq_sigma=0.000985,
        )
        assert r.nfq_actual_spread == 2054.0
        assert r.nfq_min_spread_effective == 3202.0
        assert r.nfq_min_spread_abs == 100.0
        assert r.nfq_min_spread_atr == 3202.0
        assert r.nfq_sigma == 0.000985
        assert r.cancel_reason == "no_feasible_quote"
        assert r.filled is False


# ======================================================================
# 分析スクリプト section_nfq_analysis テスト
# ======================================================================


class TestSectionNfqAnalysis:
    """section_nfq_analysis のテスト."""

    @staticmethod
    def _make_nfq_record(
        *,
        spread: float = 2000.0,
        min_spread: float = 3200.0,
        sigma: float = 0.000985,
        timestamp: float = 1774915200.0,
    ) -> dict:
        return {
            "cancel_reason": "no_feasible_quote",
            "cancelled": True,
            "filled": False,
            "timestamp": timestamp,
            "side": "buy",
            "nfq_actual_spread": spread,
            "nfq_min_spread_effective": min_spread,
            "nfq_min_spread_abs": 100.0,
            "nfq_min_spread_atr": min_spread,
            "nfq_sigma": sigma,
        }

    @staticmethod
    def _make_fill_record(*, timestamp: float = 1774915200.0) -> dict:
        return {
            "filled": True,
            "cancelled": False,
            "timestamp": timestamp,
            "side": "buy",
            "post_fill_30s_pnl": 1.0,
        }

    def test_basic_nfq_section(self) -> None:
        from scripts.v460.analysis.analyze_fill_logs import section_nfq_analysis

        records = [
            self._make_nfq_record(),
            self._make_nfq_record(spread=1800.0),
            self._make_fill_record(),
        ]
        lines = section_nfq_analysis(records)
        text = "\n".join(lines)
        assert "NFQ total: 2/3" in text
        assert "narrow_spread=2" in text
        assert "Spread gap" in text

    def test_no_nfq_records(self) -> None:
        from scripts.v460.analysis.analyze_fill_logs import section_nfq_analysis

        records = [self._make_fill_record()]
        lines = section_nfq_analysis(records)
        text = "\n".join(lines)
        assert "no NFQ records" in text

    def test_legacy_error_message_parsing(self) -> None:
        """671# 以前のレコード (error_message のみ) も解析できる."""
        from scripts.v460.analysis.analyze_fill_logs import section_nfq_analysis

        records = [
            {
                "cancel_reason": "no_feasible_quote",
                "cancelled": True,
                "filled": False,
                "timestamp": 1774915200.0,
                "side": "buy",
                "error_message": "Spread too narrow: 2054 JPY < min 3202 (abs=100, bps=406, atr=3202, σ=0.000985) [fallback_age=0.0s stale=False]",
            },
        ]
        lines = section_nfq_analysis(records)
        text = "\n".join(lines)
        assert "narrow_spread=1" in text
        assert "σ distribution" in text


# ======================================================================
# section_daily multi-metric テスト
# ======================================================================


class TestSectionDailyMultiMetric:
    """section_daily の 30s/EV/120s 三指標表示テスト."""

    def test_multi_metric_headers(self) -> None:
        from scripts.v460.analysis.analyze_fill_logs import section_daily

        records = [
            {
                "filled": True,
                "timestamp": 1774915200.0,
                "post_fill_30s_pnl": 1.0,
                "ev_weighted_pnl": 0.5,
                "post_fill_120s_pnl": 2.0,
            }
        ]
        lines = section_daily(records)
        text = "\n".join(lines)
        assert "avg30" in text
        assert "avgEV" in text
        assert "avg120" in text
        assert "multi-metric" in text

    def test_metrics_calculated_correctly(self) -> None:
        from scripts.v460.analysis.analyze_fill_logs import section_daily

        ts = 1774915200.0  # 固定日
        records = [
            {
                "filled": True,
                "timestamp": ts,
                "post_fill_30s_pnl": -1.0,
                "ev_weighted_pnl": 0.0,
                "post_fill_120s_pnl": 2.0,
            },
            {
                "filled": True,
                "timestamp": ts + 100,
                "post_fill_30s_pnl": 3.0,
                "ev_weighted_pnl": 2.0,
                "post_fill_120s_pnl": 4.0,
            },
            {
                "filled": False,
                "cancelled": True,
                "timestamp": ts + 200,
                "cancel_reason": "timeout",
            },
        ]
        lines = section_daily(records)
        text = "\n".join(lines)
        # 3 total, 2 filled, avg30 = 1.0, avgEV = 1.0, avg120 = 3.0
        assert "+1.00" in text  # avg30 and/or avgEV
        assert "+3.00" in text  # avg120
