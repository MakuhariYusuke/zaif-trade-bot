"""441# A/B Offset Comparison tool unit tests."""
from __future__ import annotations

import math

import pytest

from scripts.v460.analysis.ab_offset_comparison import (
    BucketMetrics,
    _compute_bucket_metrics,
    _std,
    _welch_t_test,
    compare_buckets,
)


class TestStd:
    def test_empty(self) -> None:
        assert _std([]) == 0.0

    def test_single(self) -> None:
        assert _std([5.0]) == 0.0

    def test_known(self) -> None:
        # std([1,2,3,4,5]) sample = sqrt(2.5) ≈ 1.5811
        result = _std([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(result - math.sqrt(2.5)) < 0.001


class TestWelchTTest:
    def test_too_few_samples(self) -> None:
        t, p = _welch_t_test([1.0, 2.0], [3.0, 4.0])
        assert t is None
        assert p is None

    def test_identical_groups(self) -> None:
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        t, p = _welch_t_test(a, a)
        assert t is not None
        assert abs(t) < 0.01
        assert p is not None
        assert p > 0.9

    def test_different_groups(self) -> None:
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [10.0, 11.0, 12.0, 13.0, 14.0]
        t, p = _welch_t_test(a, b)
        assert t is not None
        assert t > 0  # b > a
        assert p is not None
        assert p < 0.01  # highly significant


class TestComputeBucketMetrics:
    def test_basic_grouping(self) -> None:
        records = [
            {"regime": "ranging", "side": "buy", "filled": True, "post_fill_30s_pnl": -0.5, "adverse_selected": True},
            {"regime": "ranging", "side": "buy", "filled": True, "post_fill_30s_pnl": 0.5, "adverse_selected": False},
            {"regime": "ranging", "side": "buy", "filled": False, "post_fill_30s_pnl": None},
            {"regime": "ranging", "side": "sell", "filled": True, "post_fill_30s_pnl": 0.1, "adverse_selected": False},
        ]
        metrics = _compute_bucket_metrics(records)
        assert "ranging:buy" in metrics
        assert "ranging:sell" in metrics

        buy = metrics["ranging:buy"]
        assert buy["n_total"] == 3
        assert buy["n_filled"] == 2
        assert abs(buy["avg_pnl30_bps"] - 0.0) < 0.01
        assert abs(buy["fill_rate"] - 2 / 3) < 0.01
        assert abs(buy["as_rate"] - 0.5) < 0.01

    def test_empty_records(self) -> None:
        metrics = _compute_bucket_metrics([])
        assert len(metrics) == 0


class TestCompareBuckets:
    def _make_bucket(
        self,
        regime: str,
        side: str,
        pnl_values: list[float],
        n_total: int = 100,
    ) -> BucketMetrics:
        n_filled = len(pnl_values)
        avg = sum(pnl_values) / n_filled if n_filled else 0.0
        return BucketMetrics(
            regime=regime,
            side=side,
            n_total=n_total,
            n_filled=n_filled,
            fill_rate=n_filled / n_total if n_total else 0.0,
            avg_pnl30_bps=avg,
            std_pnl30_bps=_std(pnl_values),
            as_rate=0.0,
            downside_p10_bps=0.0,
            pnl_values=pnl_values,
        )

    def test_improvement_detected(self) -> None:
        import random
        rng = random.Random(42)
        before = {"ranging:buy": self._make_bucket("ranging", "buy", [-1.0 + rng.gauss(0, 0.1) for _ in range(50)])}
        after = {"ranging:buy": self._make_bucket("ranging", "buy", [1.0 + rng.gauss(0, 0.1) for _ in range(50)])}
        rows = compare_buckets(before, after, [("ranging", "buy")])
        assert len(rows) == 1
        row = rows[0]
        assert row["pnl_diff"] > 0
        assert row["significant"]

    def test_no_after_data(self) -> None:
        before = {"ranging:buy": self._make_bucket("ranging", "buy", [-0.5] * 10)}
        rows = compare_buckets(before, {}, [("ranging", "buy")])
        assert len(rows) == 1
        assert rows[0]["after_n"] == 0

    def test_missing_bucket(self) -> None:
        rows = compare_buckets({}, {}, [("ranging", "buy")])
        assert len(rows) == 0
