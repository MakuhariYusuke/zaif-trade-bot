"""160# P0-B/C: A/B判定基準 + trending_down sell 評価テスト.

Tests:
- P0-B: ABJudgmentCriteria 3指標判定 (fill_rate / avg_pnl30 / downside_p10)
- P0-C: TrendingEvalCriteria evaluate_trending_down_sell
- Dashboard judgment 統合
- YAML ロード
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pytest

from scripts.v460.lib.ab_judgment import (
    ABJudgmentCriteria,
    ABJudgmentResult,
    CriterionResult,
    PerRegimeResult,
    TrendingEvalCriteria,
    TrendingEvalResult,
    Verdict,
    evaluate_ab_variant,
    evaluate_per_regime,
    evaluate_trending_down_sell,
    _compute_metrics,
    _extract_pnl30_array,
    _safe_finite,
)


# ======================================================================
# Helpers
# ======================================================================


def _make_record(
    *,
    side: str = "sell",
    regime: str = "ranging",
    filled: bool = True,
    pnl30: float | None = 0.5,
    timestamp: float | None = None,
    adverse_selected: bool = False,
) -> dict[str, Any]:
    """テスト用 FillRecord 辞書."""
    ts = timestamp or time.time()
    r: dict[str, Any] = {
        "side": side,
        "regime": regime,
        "filled": filled,
        "timestamp": ts,
    }
    if filled and pnl30 is not None:
        r["post_fill_30s_pnl"] = pnl30
    if adverse_selected:
        r["adverse_selected"] = True
    return r


def _make_records(
    n: int,
    *,
    side: str = "sell",
    regime: str = "ranging",
    fill_rate: float = 0.5,
    pnl_mean: float = 0.0,
    pnl_std: float = 1.0,
    base_ts: float | None = None,
) -> list[dict[str, Any]]:
    """n 件のテスト用レコードを生成."""
    rng = np.random.default_rng(42)
    base = base_ts or time.time()
    records = []
    for i in range(n):
        filled = i < int(n * fill_rate)
        pnl = float(rng.normal(pnl_mean, pnl_std)) if filled else None
        records.append(_make_record(
            side=side,
            regime=regime,
            filled=filled,
            pnl30=pnl,
            timestamp=base + i * 120,
        ))
    return records


# ======================================================================
# P0-B: ABJudgmentCriteria テスト
# ======================================================================


class TestABJudgmentCriteria:
    """ABJudgmentCriteria dataclass テスト."""

    def test_defaults(self) -> None:
        c = ABJudgmentCriteria()
        assert c.min_filled_records == 50
        assert c.min_control_filled_records == 30
        assert c.fill_rate_min == 0.30
        assert c.avg_pnl30_min_bps == -1.0
        assert c.downside_p10_min_bps == -5.0

    def test_from_dict(self) -> None:
        d = {
            "min_filled_records": 100,
            "fill_rate_min": 0.40,
            "avg_pnl30_min_bps": -2.0,
            "unknown_key": "ignored",
        }
        c = ABJudgmentCriteria.from_dict(d)
        assert c.min_filled_records == 100
        assert c.fill_rate_min == 0.40
        assert c.avg_pnl30_min_bps == -2.0
        # デフォルト値が保持されること
        assert c.downside_p10_min_bps == -5.0

    def test_from_dict_empty(self) -> None:
        c = ABJudgmentCriteria.from_dict({})
        assert c.min_filled_records == 50


class TestSafeFinite:
    """_safe_finite ヘルパーテスト."""

    def test_normal_float(self) -> None:
        assert _safe_finite(1.5) == 1.5

    def test_int(self) -> None:
        assert _safe_finite(42) == 42.0

    def test_none(self) -> None:
        assert _safe_finite(None) is None

    def test_nan(self) -> None:
        assert _safe_finite(float("nan")) is None

    def test_inf(self) -> None:
        assert _safe_finite(float("inf")) is None

    def test_string_numeric(self) -> None:
        assert _safe_finite("3.14") == pytest.approx(3.14)

    def test_string_invalid(self) -> None:
        assert _safe_finite("hello") is None


class TestComputeMetrics:
    """_compute_metrics テスト."""

    def test_empty(self) -> None:
        m = _compute_metrics([])
        assert m["n_total"] == 0
        assert m["fill_rate"] == 0.0

    def test_all_filled(self) -> None:
        records = [_make_record(pnl30=1.0) for _ in range(10)]
        m = _compute_metrics(records)
        assert m["n_total"] == 10
        assert m["n_filled"] == 10
        assert m["fill_rate"] == pytest.approx(1.0)
        assert m["avg_pnl30_bps"] == pytest.approx(1.0)

    def test_mixed_fill(self) -> None:
        records = (
            [_make_record(filled=True, pnl30=2.0) for _ in range(3)]
            + [_make_record(filled=False) for _ in range(7)]
        )
        m = _compute_metrics(records)
        assert m["n_total"] == 10
        assert m["n_filled"] == 3
        assert m["fill_rate"] == pytest.approx(0.3)

    def test_downside_percentile(self) -> None:
        # 100件, 一様分布 [-10, 10]
        rng = np.random.default_rng(123)
        records = [
            _make_record(pnl30=float(rng.uniform(-10, 10)))
            for _ in range(100)
        ]
        m = _compute_metrics(records)
        # p10 は概ね -8 付近 (一様分布の10パーセンタイル)
        assert m["downside_p10_bps"] < -5.0


class TestExtractPnl30Array:
    """_extract_pnl30_array テスト."""

    def test_filled_only(self) -> None:
        records = [
            _make_record(filled=True, pnl30=1.0),
            _make_record(filled=False),
            _make_record(filled=True, pnl30=2.0),
        ]
        arr = _extract_pnl30_array(records)
        assert len(arr) == 2
        assert arr[0] == pytest.approx(1.0)
        assert arr[1] == pytest.approx(2.0)

    def test_empty(self) -> None:
        arr = _extract_pnl30_array([])
        assert len(arr) == 0


# ======================================================================
# P0-B: evaluate_ab_variant テスト
# ======================================================================


class TestEvaluateABVariant:
    """A/B variant 3指標判定テスト."""

    def test_pass_normal(self) -> None:
        """標準的な PASS ケース."""
        criteria = ABJudgmentCriteria(min_filled_records=5, min_control_filled_records=5, min_calendar_days=1)
        variant = _make_records(20, fill_rate=0.6, pnl_mean=1.0, pnl_std=0.5)
        control = _make_records(20, fill_rate=0.5, pnl_mean=0.5, pnl_std=0.5, side="buy")
        result = evaluate_ab_variant(variant, control, criteria)
        assert result.overall == Verdict.PASS
        assert len(result.criteria) == 3
        assert all(c.verdict == Verdict.PASS for c in result.criteria)

    def test_insufficient_sample(self) -> None:
        """サンプル不足で INSUFFICIENT."""
        criteria = ABJudgmentCriteria(min_filled_records=100)
        variant = _make_records(10, fill_rate=0.5)
        control = _make_records(10, fill_rate=0.5, side="buy")
        result = evaluate_ab_variant(variant, control, criteria)
        assert result.overall == Verdict.INSUFFICIENT

    def test_insufficient_calendar_days(self) -> None:
        """暦日不足で INSUFFICIENT."""
        criteria = ABJudgmentCriteria(min_filled_records=3, min_calendar_days=5)
        variant = _make_records(10, fill_rate=0.5)
        control = _make_records(10, fill_rate=0.5, side="buy")
        result = evaluate_ab_variant(variant, control, criteria)
        assert result.overall == Verdict.INSUFFICIENT

    def test_fail_fill_rate_absolute(self) -> None:
        """fill_rate が絶対下限未満で FAIL."""
        criteria = ABJudgmentCriteria(
            min_filled_records=5,
            min_control_filled_records=5,
            min_calendar_days=1,
            fill_rate_min=0.50,
        )
        # fill_rate = 20% < 50%
        variant = _make_records(50, fill_rate=0.20, pnl_mean=1.0)
        control = _make_records(50, fill_rate=0.50, pnl_mean=0.5, side="buy")
        result = evaluate_ab_variant(variant, control, criteria)
        assert result.overall == Verdict.FAIL
        fr_crit = next(c for c in result.criteria if c.name == "fill_rate")
        assert fr_crit.verdict == Verdict.FAIL

    def test_fail_fill_rate_degradation(self) -> None:
        """fill_rate の対 control 悪化で FAIL."""
        criteria = ABJudgmentCriteria(
            min_filled_records=5,
            min_control_filled_records=5,
            min_calendar_days=1,
            fill_rate_min=0.20,
            fill_rate_degradation_tolerance=0.05,
        )
        # variant 35% vs control 50% → 30% 悪化 > 5%
        variant = _make_records(100, fill_rate=0.35, pnl_mean=1.0)
        control = _make_records(100, fill_rate=0.50, pnl_mean=0.5, side="buy")
        result = evaluate_ab_variant(variant, control, criteria)
        fr_crit = next(c for c in result.criteria if c.name == "fill_rate")
        assert fr_crit.verdict == Verdict.FAIL

    def test_fail_avg_pnl30(self) -> None:
        """avg_pnl30 が下限未満で FAIL."""
        criteria = ABJudgmentCriteria(
            min_filled_records=5,
            min_control_filled_records=5,
            min_calendar_days=1,
            avg_pnl30_min_bps=-0.5,
        )
        # pnl_mean = -2.0 < -0.5
        variant = _make_records(20, fill_rate=0.6, pnl_mean=-2.0, pnl_std=0.3)
        control = _make_records(20, fill_rate=0.5, pnl_mean=0.5, side="buy")
        result = evaluate_ab_variant(variant, control, criteria)
        pnl_crit = next(c for c in result.criteria if c.name == "avg_pnl30")
        assert pnl_crit.verdict == Verdict.FAIL
        assert result.overall == Verdict.FAIL

    def test_fail_downside_p10(self) -> None:
        """downside_p10 が絶対下限未満で FAIL."""
        criteria = ABJudgmentCriteria(
            min_filled_records=5,
            min_control_filled_records=5,
            min_calendar_days=1,
            downside_p10_min_bps=-3.0,
        )
        # 低い PnL で p10 が -3.0 を割る
        variant = _make_records(50, fill_rate=0.6, pnl_mean=-1.0, pnl_std=3.0)
        control = _make_records(50, fill_rate=0.5, pnl_mean=0.5, pnl_std=0.5, side="buy")
        result = evaluate_ab_variant(variant, control, criteria)
        ds_crit = next(c for c in result.criteria if c.name == "downside_p10")
        assert ds_crit.verdict == Verdict.FAIL

    def test_statistical_test_integrated(self) -> None:
        """統計検定が実行されること (n>=10)."""
        criteria = ABJudgmentCriteria(min_filled_records=5, min_control_filled_records=5, min_calendar_days=1)
        variant = _make_records(30, fill_rate=0.8, pnl_mean=2.0, pnl_std=0.5)
        control = _make_records(30, fill_rate=0.8, pnl_mean=-2.0, pnl_std=0.5, side="buy")
        result = evaluate_ab_variant(variant, control, criteria)
        # 統計検定が実行されたか
        assert result.pnl30_p_value is not None
        assert result.pnl30_effect_size is not None
        # 明瞭な差 → p < 0.05
        assert result.pnl30_p_value < 0.05

    def test_summary_contains_verdict(self) -> None:
        """summary() がverdict を含むこと."""
        criteria = ABJudgmentCriteria(min_filled_records=5, min_control_filled_records=5, min_calendar_days=1)
        variant = _make_records(20, fill_rate=0.6, pnl_mean=1.0)
        control = _make_records(20, fill_rate=0.5, pnl_mean=0.5, side="buy")
        result = evaluate_ab_variant(variant, control, criteria)
        s = result.summary()
        assert "PASS" in s or "FAIL" in s or "INSUFFICIENT" in s

    def test_empty_variant(self) -> None:
        """variant が空 → INSUFFICIENT."""
        criteria = ABJudgmentCriteria(min_filled_records=1)
        result = evaluate_ab_variant([], [], criteria)
        assert result.overall == Verdict.INSUFFICIENT

    def test_avg_pnl30_must_improve(self) -> None:
        """avg_pnl30_must_improve=True でcontrol以下ならFAIL."""
        criteria = ABJudgmentCriteria(
            min_filled_records=5,
            min_control_filled_records=5,
            min_calendar_days=1,
            avg_pnl30_min_bps=-10.0,
            avg_pnl30_must_improve=True,
        )
        # variant pnl < control pnl
        variant = _make_records(20, fill_rate=0.6, pnl_mean=0.1, pnl_std=0.1)
        control = _make_records(20, fill_rate=0.5, pnl_mean=0.5, pnl_std=0.1, side="buy")
        result = evaluate_ab_variant(variant, control, criteria)
        pnl_crit = next(c for c in result.criteria if c.name == "avg_pnl30")
        assert pnl_crit.verdict == Verdict.FAIL


# ======================================================================
# P0-C: TrendingEvalCriteria テスト
# ======================================================================


class TestTrendingEvalCriteria:
    """TrendingEvalCriteria dataclass テスト."""

    def test_defaults(self) -> None:
        c = TrendingEvalCriteria()
        assert c.min_filled == 10
        assert c.target_filled == 30
        assert c.avg_pnl30_min_bps == -0.5
        assert c.counterfactual_pnl30_bps == pytest.approx(-0.66)

    def test_from_dict(self) -> None:
        d = {
            "min_filled": 20,
            "target_filled": 50,
            "avg_pnl30_min_bps": -1.0,
            "extra_key": "ignored",
        }
        c = TrendingEvalCriteria.from_dict(d)
        assert c.min_filled == 20
        assert c.target_filled == 50
        assert c.avg_pnl30_min_bps == -1.0


class TestEvaluateTrendingDownSell:
    """trending_down sell 実測評価テスト."""

    def test_pass_normal(self) -> None:
        """正常 PASS."""
        criteria = TrendingEvalCriteria(min_filled=5, target_filled=10)
        records = _make_records(
            30,
            fill_rate=0.5,
            pnl_mean=0.5,
            pnl_std=0.5,
            side="sell",
            regime="trending_down",
        )
        result = evaluate_trending_down_sell(records, criteria)
        assert result.verdict == Verdict.PASS
        assert result.n_filled >= 5
        assert result.avg_pnl30_bps > 0

    def test_insufficient_sample(self) -> None:
        """サンプル不足で INSUFFICIENT."""
        criteria = TrendingEvalCriteria(min_filled=100)
        records = _make_records(
            10,
            fill_rate=0.5,
            side="sell",
            regime="trending_down",
        )
        result = evaluate_trending_down_sell(records, criteria)
        assert result.verdict == Verdict.INSUFFICIENT

    def test_fail_avg_pnl30(self) -> None:
        """avg_pnl30 が閾値未満で FAIL."""
        criteria = TrendingEvalCriteria(
            min_filled=3,
            target_filled=5,
            avg_pnl30_min_bps=-0.3,
        )
        records = _make_records(
            30,
            fill_rate=0.5,
            pnl_mean=-2.0,
            pnl_std=0.5,
            side="sell",
            regime="trending_down",
        )
        result = evaluate_trending_down_sell(records, criteria)
        assert result.verdict == Verdict.FAIL

    def test_filters_regime_and_side(self) -> None:
        """trending_down sell のみ抽出されること."""
        criteria = TrendingEvalCriteria(min_filled=3, target_filled=5)
        records = (
            _make_records(20, fill_rate=0.5, side="sell", regime="trending_down", pnl_mean=1.0)
            + _make_records(20, fill_rate=0.5, side="buy", regime="trending_down", pnl_mean=-5.0)
            + _make_records(20, fill_rate=0.5, side="sell", regime="ranging", pnl_mean=-5.0)
        )
        result = evaluate_trending_down_sell(records, criteria)
        assert result.verdict == Verdict.PASS
        # buy と ranging のレコードは含まれない
        assert result.n_filled <= 20

    def test_counterfactual_gain(self) -> None:
        """CF gain = 実測 - カウンターファクチュアル."""
        criteria = TrendingEvalCriteria(
            min_filled=3,
            target_filled=5,
            counterfactual_pnl30_bps=-1.0,
        )
        records = _make_records(
            20,
            fill_rate=0.5,
            pnl_mean=0.5,
            pnl_std=0.1,
            side="sell",
            regime="trending_down",
        )
        result = evaluate_trending_down_sell(records, criteria)
        # CF gain ≈ 0.5 - (-1.0) = 1.5 (概算)
        assert result.counterfactual_gain_bps > 1.0

    def test_daily_breakdown_populated(self) -> None:
        """日次内訳が生成されること."""
        criteria = TrendingEvalCriteria(min_filled=3, target_filled=5)
        base_ts = time.time() - 86400 * 3  # 3日前から
        records = _make_records(
            30,
            fill_rate=0.5,
            pnl_mean=0.5,
            side="sell",
            regime="trending_down",
            base_ts=base_ts,
        )
        result = evaluate_trending_down_sell(records, criteria)
        assert len(result.daily_breakdown) >= 1
        assert "day" in result.daily_breakdown[0]
        assert "n_filled" in result.daily_breakdown[0]
        assert "avg_pnl30_bps" in result.daily_breakdown[0]

    def test_no_trending_down_sell(self) -> None:
        """trending_down sell が0件."""
        criteria = TrendingEvalCriteria(min_filled=1)
        records = _make_records(20, fill_rate=0.5, side="buy", regime="ranging")
        result = evaluate_trending_down_sell(records, criteria)
        assert result.verdict == Verdict.INSUFFICIENT
        assert result.n_filled == 0

    def test_provisional_pass(self) -> None:
        """min_filled 以上 target_filled 未満 → PROVISIONAL PASS."""
        criteria = TrendingEvalCriteria(
            min_filled=3,
            target_filled=100,
            avg_pnl30_min_bps=-5.0,
        )
        records = _make_records(
            20,
            fill_rate=0.5,
            pnl_mean=1.0,
            pnl_std=0.5,
            side="sell",
            regime="trending_down",
        )
        result = evaluate_trending_down_sell(records, criteria)
        assert result.verdict == Verdict.PASS
        assert "PROVISIONAL" in result.detail

    def test_summary_readable(self) -> None:
        """summary() が人間可読な出力を返すこと."""
        criteria = TrendingEvalCriteria(min_filled=3, target_filled=5)
        records = _make_records(
            20, fill_rate=0.5, pnl_mean=0.5, side="sell", regime="trending_down"
        )
        result = evaluate_trending_down_sell(records, criteria)
        s = result.summary()
        assert "Trending Down Sell Eval" in s
        assert "n_filled=" in s


# ======================================================================
# Dashboard 統合テスト
# ======================================================================


class TestDashboardJudgmentIntegration:
    """side_regime_dashboard の judgment 統合テスト."""

    def test_run_dashboard_without_judgment(self, tmp_path: Any) -> None:
        """with_judgment=False ではjudgment結果がNone."""
        from scripts.v460.analysis.side_regime_dashboard import run_dashboard

        # 空ディレクトリ
        result = run_dashboard(str(tmp_path), with_judgment=False)
        assert result.get("ab_judgment") is None
        assert result.get("trending_eval") is None

    def test_run_dashboard_with_judgment(self, tmp_path: Any) -> None:
        """with_judgment=True でjudgment結果が生成される."""
        import json as _json

        from scripts.v460.analysis.side_regime_dashboard import run_dashboard

        # テスト用 JSONL 作成
        jsonl_path = tmp_path / "fill_records_test.jsonl"
        records = (
            _make_records(50, fill_rate=0.6, pnl_mean=0.5, side="sell", regime="ranging")
            + _make_records(50, fill_rate=0.5, pnl_mean=0.3, side="buy", regime="ranging")
            + _make_records(30, fill_rate=0.5, pnl_mean=0.4, side="sell", regime="trending_down")
        )
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(_json.dumps(r) + "\n")

        criteria = ABJudgmentCriteria(min_filled_records=5, min_calendar_days=1)
        trending_criteria = TrendingEvalCriteria(min_filled=3, target_filled=10)

        result = run_dashboard(
            str(tmp_path),
            with_judgment=True,
            ab_criteria=criteria,
            trending_criteria=trending_criteria,
        )

        # A/B judgment が生成されること
        ab_j = result.get("ab_judgment")
        assert ab_j is not None
        assert "overall" in ab_j
        assert ab_j["overall"] in ("pass", "fail", "insufficient")

        # trending eval が生成されること
        te = result.get("trending_eval")
        assert te is not None
        assert "verdict" in te
        assert te["verdict"] in ("pass", "fail", "insufficient")


# ======================================================================
# YAML ロードテスト
# ======================================================================


class TestYAMLJudgmentLoad:
    """YAML judgment セクションのロードテスト."""

    def test_load_from_yaml(self, tmp_path: Any) -> None:
        """YAML から judgment 設定をロード."""
        yaml_content = """
judgment:
  ab_criteria:
    min_filled_records: 100
    fill_rate_min: 0.40
    avg_pnl30_min_bps: -2.0
    downside_p10_min_bps: -6.0
  trending_down_sell:
    min_filled: 20
    target_filled: 50
    avg_pnl30_min_bps: -1.0
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        from scripts.v460.analysis.side_regime_dashboard import _load_judgment_config
        ab, trending = _load_judgment_config(str(yaml_file))

        assert ab is not None
        assert ab.min_filled_records == 100
        assert ab.fill_rate_min == 0.40
        assert ab.avg_pnl30_min_bps == -2.0

        assert trending is not None
        assert trending.min_filled == 20
        assert trending.target_filled == 50

    def test_load_missing_file(self) -> None:
        """存在しないファイル → None."""
        from scripts.v460.analysis.side_regime_dashboard import _load_judgment_config
        ab, trending = _load_judgment_config("/nonexistent/path.yaml")
        assert ab is None
        assert trending is None

    def test_load_no_judgment_section(self, tmp_path: Any) -> None:
        """judgment セクションなし → None."""
        yaml_file = tmp_path / "minimal.yaml"
        yaml_file.write_text("some_key: value\n", encoding="utf-8")

        from scripts.v460.analysis.side_regime_dashboard import _load_judgment_config
        ab, trending = _load_judgment_config(str(yaml_file))
        assert ab is None
        assert trending is None


# ======================================================================
# exclude_regimes フィルタリングテスト
# ======================================================================


class TestExcludeRegimes:
    """exclude_regimes パラメータによる warmup/legacy ノイズ除外テスト."""

    def test_default_excludes_none(self) -> None:
        """デフォルトで regime=none を除外."""
        c = ABJudgmentCriteria()
        assert c.exclude_regimes == ["none"]

    def test_exclude_empty_means_all_included(self) -> None:
        """exclude_regimes=[] なら全 regime を含む."""
        c = ABJudgmentCriteria(exclude_regimes=[])
        assert c.exclude_regimes == []

    def test_from_dict_with_exclude_regimes(self) -> None:
        """YAML 辞書から exclude_regimes をロード."""
        c = ABJudgmentCriteria.from_dict({
            "exclude_regimes": ["none", "unknown"],
        })
        assert c.exclude_regimes == ["none", "unknown"]

    def test_from_dict_exclude_regimes_none_becomes_empty(self) -> None:
        """YAML で exclude_regimes: null → 空リスト."""
        c = ABJudgmentCriteria.from_dict({"exclude_regimes": None})
        assert c.exclude_regimes == []

    def test_evaluate_excludes_none_regime(self) -> None:
        """regime=none のレコードが判定から除外される."""
        # ranging records: 健全 (pnl30 > 0)
        sell_ranging = _make_records(
            100, side="sell", regime="ranging",
            fill_rate=0.8, pnl_mean=0.5, pnl_std=1.0,
        )
        buy_ranging = _make_records(
            100, side="buy", regime="ranging",
            fill_rate=0.8, pnl_mean=0.3, pnl_std=1.0,
        )
        # none records: 悪い (pnl30 << 0, AS 高い)
        sell_none = _make_records(
            80, side="sell", regime="none",
            fill_rate=0.5, pnl_mean=-3.0, pnl_std=2.0,
        )
        buy_none = _make_records(
            80, side="buy", regime="none",
            fill_rate=0.5, pnl_mean=-2.0, pnl_std=2.0,
        )

        all_sell = sell_ranging + sell_none
        all_buy = buy_ranging + buy_none

        # exclude_regimes=["none"] → none が除外されて ranging のみで判定
        criteria_exclude = ABJudgmentCriteria(
            exclude_regimes=["none"],
            min_filled_records=10,
        )
        result = evaluate_ab_variant(
            all_sell, all_buy, criteria=criteria_exclude,
            variant_label="sell", control_label="buy",
        )
        # ranging だけなら健全なはず
        assert result.n_variant > 0
        # none が含まれていないことを確認
        assert result.n_variant == len([r for r in sell_ranging if r.get("filled")])

    def test_evaluate_no_exclude_includes_all(self) -> None:
        """exclude_regimes=[] なら全レコード含む."""
        sell = _make_records(60, side="sell", regime="ranging", fill_rate=0.8)
        sell_none = _make_records(40, side="sell", regime="none", fill_rate=0.5)
        buy = _make_records(100, side="buy", regime="ranging", fill_rate=0.8)

        all_sell = sell + sell_none
        criteria = ABJudgmentCriteria(exclude_regimes=[], min_filled_records=10)
        result = evaluate_ab_variant(
            all_sell, buy, criteria=criteria,
            variant_label="sell", control_label="buy",
        )
        # 全レコードの filled が含まれるはず
        total_filled = len([r for r in all_sell if r.get("filled")])
        assert result.n_variant == total_filled

    def test_evaluate_excludes_multiple_regimes(self) -> None:
        """複数 regime を除外."""
        sell_r = _make_records(80, side="sell", regime="ranging", fill_rate=0.8, pnl_mean=0.5)
        sell_n = _make_records(30, side="sell", regime="none", fill_rate=0.5, pnl_mean=-5.0)
        sell_u = _make_records(20, side="sell", regime="unknown", fill_rate=0.5, pnl_mean=-4.0)
        buy = _make_records(100, side="buy", regime="ranging", fill_rate=0.8)

        all_sell = sell_r + sell_n + sell_u
        criteria = ABJudgmentCriteria(
            exclude_regimes=["none", "unknown"],
            min_filled_records=10,
        )
        result = evaluate_ab_variant(
            all_sell, buy, criteria=criteria,
            variant_label="sell", control_label="buy",
        )
        assert result.n_variant == len([r for r in sell_r if r.get("filled")])

    def test_exclude_regime_none_null(self) -> None:
        """regime=None (Python None) のレコードも 'none' として除外される."""
        sell = [_make_record(side="sell", regime="ranging", filled=True, pnl30=0.5)]
        sell_null = [
            {"side": "sell", "regime": None, "filled": True,
             "post_fill_30s_pnl": -5.0, "timestamp": time.time()},
        ]
        buy = _make_records(60, side="buy", regime="ranging", fill_rate=0.8)

        all_sell = sell * 60 + sell_null * 10
        criteria = ABJudgmentCriteria(exclude_regimes=["none"], min_filled_records=10)
        result = evaluate_ab_variant(
            all_sell, buy, criteria=criteria,
            variant_label="sell", control_label="buy",
        )
        # None regime が除外されて 60 レコードのみ
        assert result.n_variant == 60


# ======================================================================
# Per-regime A/B 判定テスト
# ======================================================================


class TestEvaluatePerRegime:
    """evaluate_per_regime テスト."""

    def test_basic_per_regime(self) -> None:
        """regime 別に分離して判定が行われる."""
        sell_ranging = _make_records(80, side="sell", regime="ranging", fill_rate=0.8, pnl_mean=0.5)
        sell_trending = _make_records(40, side="sell", regime="trending", fill_rate=0.4, pnl_mean=-2.0)
        buy_ranging = _make_records(80, side="buy", regime="ranging", fill_rate=0.8, pnl_mean=0.3)
        buy_trending = _make_records(40, side="buy", regime="trending", fill_rate=0.8, pnl_mean=0.2)

        all_sell = sell_ranging + sell_trending
        all_buy = buy_ranging + buy_trending

        criteria = ABJudgmentCriteria(min_filled_records=10)
        results = evaluate_per_regime(
            all_sell, all_buy,
            criteria=criteria,
            variant_label="sell", control_label="buy",
            target_regimes=["ranging", "trending"],
        )
        assert len(results) == 2
        regime_names = {r.regime for r in results}
        assert "ranging" in regime_names
        assert "trending" in regime_names

    def test_per_regime_label(self) -> None:
        """variant_label に regime 名が付加される."""
        sell = _make_records(60, side="sell", regime="ranging", fill_rate=0.8)
        buy = _make_records(60, side="buy", regime="ranging", fill_rate=0.8)

        results = evaluate_per_regime(
            sell, buy,
            criteria=ABJudgmentCriteria(min_filled_records=10),
            variant_label="sell", control_label="buy",
            target_regimes=["ranging"],
        )
        assert len(results) == 1
        assert results[0].result.variant_label == "sell[ranging]"
        assert results[0].result.control_label == "buy[ranging]"

    def test_per_regime_target_filter(self) -> None:
        """target_regimes で指定した regime のみ評価."""
        sell = (
            _make_records(60, side="sell", regime="ranging", fill_rate=0.8)
            + _make_records(20, side="sell", regime="none", fill_rate=0.5)
        )
        buy = (
            _make_records(60, side="buy", regime="ranging", fill_rate=0.8)
            + _make_records(20, side="buy", regime="none", fill_rate=0.5)
        )

        results = evaluate_per_regime(
            sell, buy,
            criteria=ABJudgmentCriteria(min_filled_records=5),
            target_regimes=["ranging"],
        )
        assert len(results) == 1
        assert results[0].regime == "ranging"

    def test_per_regime_no_target_shows_all(self) -> None:
        """target_regimes=None なら全 regime."""
        sell = (
            _make_records(40, side="sell", regime="ranging", fill_rate=0.8)
            + _make_records(40, side="sell", regime="trending", fill_rate=0.5)
        )
        buy = (
            _make_records(40, side="buy", regime="ranging", fill_rate=0.8)
            + _make_records(40, side="buy", regime="trending", fill_rate=0.8)
        )

        results = evaluate_per_regime(
            sell, buy,
            criteria=ABJudgmentCriteria(min_filled_records=5),
            target_regimes=None,
        )
        assert len(results) == 2

    def test_per_regime_insufficient_sample(self) -> None:
        """レコード数不足の regime は INSUFFICIENT."""
        sell = _make_records(5, side="sell", regime="trending_down", fill_rate=0.8)
        buy = _make_records(5, side="buy", regime="trending_down", fill_rate=0.8)

        results = evaluate_per_regime(
            sell, buy,
            criteria=ABJudgmentCriteria(min_filled_records=50),
            target_regimes=["trending_down"],
        )
        assert len(results) == 1
        assert results[0].result.overall == Verdict.INSUFFICIENT

    def test_per_regime_empty_regime_skipped(self) -> None:
        """データが存在しない target_regime はスキップ."""
        sell = _make_records(60, side="sell", regime="ranging", fill_rate=0.8)
        buy = _make_records(60, side="buy", regime="ranging", fill_rate=0.8)

        results = evaluate_per_regime(
            sell, buy,
            criteria=ABJudgmentCriteria(min_filled_records=5),
            target_regimes=["trending_down"],
        )
        assert len(results) == 0


# ======================================================================
# YAML exclude_regimes ロードテスト
# ======================================================================


class TestYAMLExcludeRegimes:
    """YAML 設定から exclude_regimes がロードされることをテスト."""

    def test_load_with_exclude_regimes(self, tmp_path: Any) -> None:
        yaml_content = """
judgment:
  ab_criteria:
    min_filled_records: 50
    exclude_regimes:
      - "none"
      - "unknown"
  trending_down_sell:
    min_filled: 10
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        from scripts.v460.analysis.side_regime_dashboard import _load_judgment_config
        ab, _ = _load_judgment_config(str(yaml_file))
        assert ab is not None
        assert ab.exclude_regimes == ["none", "unknown"]

    def test_load_without_exclude_regimes_uses_default(self, tmp_path: Any) -> None:
        yaml_content = """
judgment:
  ab_criteria:
    min_filled_records: 50
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        from scripts.v460.analysis.side_regime_dashboard import _load_judgment_config
        ab, _ = _load_judgment_config(str(yaml_file))
        assert ab is not None
        assert ab.exclude_regimes == ["none"]  # default


# ======================================================================
# control 側最小サンプル制約テスト (§10.4)
# ======================================================================


class TestControlMinFilled:
    """control 側の最小 filled 制約テスト."""

    def test_control_sufficient(self) -> None:
        """control が十分なら通常判定."""
        sell = _make_records(100, side="sell", fill_rate=0.8, pnl_mean=0.5)
        buy = _make_records(80, side="buy", fill_rate=0.8, pnl_mean=0.3)
        criteria = ABJudgmentCriteria(
            min_filled_records=10,
            min_control_filled_records=10,
            min_calendar_days=1,
            exclude_regimes=[],
        )
        result = evaluate_ab_variant(sell, buy, criteria=criteria)
        assert result.overall != Verdict.INSUFFICIENT

    def test_control_insufficient(self) -> None:
        """control が min_control_filled_records 未満 → INSUFFICIENT."""
        sell = _make_records(100, side="sell", fill_rate=0.8, pnl_mean=0.5)
        buy = _make_records(10, side="buy", fill_rate=0.5, pnl_mean=0.3)
        criteria = ABJudgmentCriteria(
            min_filled_records=10,
            min_control_filled_records=30,  # 10*0.5=5 < 30
            exclude_regimes=[],
        )
        result = evaluate_ab_variant(sell, buy, criteria=criteria)
        assert result.overall == Verdict.INSUFFICIENT
        assert any(c.name == "control_sample_size" for c in result.criteria)

    def test_control_default_threshold(self) -> None:
        """デフォルト min_control_filled_records=30."""
        c = ABJudgmentCriteria()
        assert c.min_control_filled_records == 30

    def test_control_from_dict(self) -> None:
        """YAML から min_control_filled_records をロード."""
        c = ABJudgmentCriteria.from_dict({"min_control_filled_records": 20})
        assert c.min_control_filled_records == 20


# ======================================================================
# 160# bugfix: 空 PnL データ誤 PASS 防止テスト
# ======================================================================


class TestEmptyPnlInsufficient:
    """PnL データなし時に INSUFFICIENT を返すことを確認."""

    def test_variant_no_pnl_is_insufficient(self) -> None:
        """variant に filled はあるが PnL が全て None → INSUFFICIENT."""
        base = time.time()
        # filled=True だが pnl30=None のレコード群
        variant = [
            _make_record(side="sell", regime="ranging", filled=True, pnl30=None, timestamp=base + i * 120)
            for i in range(60)
        ]
        control = _make_records(60, side="buy", fill_rate=0.8, pnl_mean=0.3)
        criteria = ABJudgmentCriteria(
            min_filled_records=10,
            min_control_filled_records=5,
            min_calendar_days=1,
            exclude_regimes=[],
        )
        result = evaluate_ab_variant(variant, control, criteria=criteria)
        assert result.overall == Verdict.INSUFFICIENT
        assert any(c.name == "pnl_data" for c in result.criteria)

    def test_variant_with_pnl_is_not_pnl_insufficient(self) -> None:
        """variant に PnL データあり → pnl_data INSUFFICIENT にならない."""
        sell = _make_records(100, side="sell", fill_rate=0.8, pnl_mean=0.5)
        buy = _make_records(80, side="buy", fill_rate=0.8, pnl_mean=0.3)
        criteria = ABJudgmentCriteria(
            min_filled_records=10,
            min_control_filled_records=5,
            min_calendar_days=1,
            exclude_regimes=[],
        )
        result = evaluate_ab_variant(sell, buy, criteria=criteria)
        assert not any(c.name == "pnl_data" for c in result.criteria)
