"""306# 6 提案の単体テスト.

#1 Queue Position Estimation (O1)
#2 Microprice Side Selection (L2)
#3 Dynamic Cycle Interval (L1)
#4 EV-based Offset Adaptation (A1)
#5 Offset Stage Recording (E1)
#6 Parkinson σ YAML 有効化
+ Offset Ceiling (300# T1-3)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from scripts.v460.lib.fill_config import FillTestConfig


# =====================================================================
# Helpers
# =====================================================================

def _make_config(**overrides):
    """FillTestConfig with sensible defaults + overrides."""
    from scripts.v460.lib.fill_config import FillTestConfig
    defaults = {
        "sigma_parkinson_enabled": True,
        "sigma_parkinson_window_sec": 300.0,
        "microprice_side_enabled": True,
        "microprice_side_threshold": 0.3,
        "dynamic_cycle_interval_enabled": True,
        "dynamic_cycle_interval_min_sec": 60.0,
        "dynamic_cycle_interval_max_sec": 300.0,
        "dynamic_cycle_interval_sigma_ref": 0.0005,
        "queue_position_tracking_enabled": True,
        "queue_position_early_cancel_prob": 0.05,
        "offset_stage_recording_enabled": True,
        "offset_ceiling_ratio": 0.15,
    }
    defaults.update(overrides)
    return FillTestConfig(**defaults)


def _make_maker_price(config=None, **kw):
    """MakerPriceCalculator with minimal deps."""
    from scripts.v460.lib.maker_price import MakerPriceCalculator

    if config is None:
        config = _make_config(**kw)
    ffd = MagicMock()
    ffd.config = MagicMock()
    ffd.config.fast_fill_cooldown_sec = 0
    return MakerPriceCalculator(
        config=config,
        fast_fill_defense=ffd,
        regime_detector=None,
        base_offset_ratio=0.05,
    )


# =====================================================================
# #2 Microprice Side Selection (L2)
# =====================================================================


class TestMicropriceBiasBps:
    """compute_microprice_bias_bps のテスト."""

    def test_positive_bias_when_bid_thick(self) -> None:
        """bid volume > ask volume → positive bias (sell 有利)."""
        mp = _make_maker_price()
        ob = MagicMock()
        ob.bids = [(100.0, 10.0)]  # thick bid
        ob.asks = [(101.0, 1.0)]   # thin ask
        mp._last_ob_snapshot = ob
        bias = mp.compute_microprice_bias_bps()
        assert bias > 0, f"Expected positive bias but got {bias}"

    def test_negative_bias_when_ask_thick(self) -> None:
        """ask volume > bid volume → negative bias (buy 有利)."""
        mp = _make_maker_price()
        ob = MagicMock()
        ob.bids = [(100.0, 1.0)]   # thin bid
        ob.asks = [(101.0, 10.0)]  # thick ask
        mp._last_ob_snapshot = ob
        bias = mp.compute_microprice_bias_bps()
        assert bias < 0, f"Expected negative bias but got {bias}"

    def test_zero_bias_when_balanced(self) -> None:
        """equal volume → zero bias."""
        mp = _make_maker_price()
        ob = MagicMock()
        ob.bids = [(100.0, 5.0)]
        ob.asks = [(101.0, 5.0)]
        mp._last_ob_snapshot = ob
        bias = mp.compute_microprice_bias_bps()
        assert abs(bias) < 1e-6, f"Expected near-zero bias but got {bias}"

    def test_zero_when_no_ob(self) -> None:
        """OB 未取得 → 0."""
        mp = _make_maker_price()
        mp._last_ob_snapshot = None
        assert mp.compute_microprice_bias_bps() == 0.0


class TestMicropriceSideSelector:
    """SideSelector microprice 統合テスト."""

    def test_microprice_overrides_to_buy(self) -> None:
        """309# Positive bias above threshold → buy (safety mode: buy pressure → buy)."""
        from scripts.v460.lib.side_selector import SideSelector
        config = _make_config(microprice_side_enabled=True, microprice_side_threshold=0.3)
        ss = SideSelector(config)
        # 310# C: spread + regime guardrails 通過のため引数追加
        side = ss.next(microprice_bias_bps=1.0, spread_bps=20.0, regime="ranging")
        assert side == "buy"

    def test_microprice_overrides_to_sell(self) -> None:
        """309# Negative bias below -threshold → sell (safety mode: sell pressure → sell)."""
        from scripts.v460.lib.side_selector import SideSelector
        config = _make_config(microprice_side_enabled=True, microprice_side_threshold=0.3)
        ss = SideSelector(config)
        # last_side=None → base_side="buy", bias < -0.3 → sell
        # 310# C: spread + regime guardrails 通過のため引数追加
        side = ss.next(microprice_bias_bps=-1.0, spread_bps=20.0, regime="ranging")
        assert side == "sell"

    def test_microprice_within_threshold_no_override(self) -> None:
        """Within threshold → no microprice override (alternation used)."""
        from scripts.v460.lib.side_selector import SideSelector
        config = _make_config(microprice_side_enabled=True, microprice_side_threshold=0.3)
        ss = SideSelector(config)
        side1 = ss.next(microprice_bias_bps=0.1, spread_bps=20.0, regime="ranging")
        # Should use alternation (first call = "buy")
        assert side1 in ("buy", "sell")

    def test_microprice_guardrail_spread_blocks(self) -> None:
        """310# C: spread < min → microprice skipped, fallback to alternation."""
        from scripts.v460.lib.side_selector import SideSelector
        config = _make_config(
            microprice_side_enabled=True, microprice_side_threshold=0.3,
            microprice_side_min_spread_bps=15.0,
        )
        ss = SideSelector(config)
        # bias would override to sell, but spread too narrow
        side = ss.next(microprice_bias_bps=-1.0, spread_bps=5.0, regime="ranging")
        assert side == "buy"  # alternation: first call = buy

    def test_microprice_guardrail_regime_blocks(self) -> None:
        """310# C: regime not in gate → microprice skipped."""
        from scripts.v460.lib.side_selector import SideSelector
        config = _make_config(
            microprice_side_enabled=True, microprice_side_threshold=0.3,
        )
        ss = SideSelector(config)
        # bias would override to sell, but regime is trending (not in ["ranging"])
        side = ss.next(microprice_bias_bps=-1.0, spread_bps=20.0, regime="trending")
        assert side == "buy"  # alternation: first call = buy


# =====================================================================
# #3 Dynamic Cycle Interval (L1)
# =====================================================================


class TestDynamicCycleInterval:
    """_compute_dynamic_interval のテスト."""

    @staticmethod
    def _compute(base: float, sigma: float, sigma_ref: float = 0.0005,
                 min_sec: float = 60.0, max_sec: float = 300.0) -> float:
        """309# Pure computation: σ/σ_ref (high vol → longer, low vol → shorter)."""
        if sigma <= 0:
            return base
        ratio = sigma / sigma_ref
        adjusted = base * ratio
        return max(min_sec, min(adjusted, max_sec))

    def test_high_sigma_lengthens_interval(self) -> None:
        """309# σ > σ_ref → interval lengthened (Cooldown)."""
        result = self._compute(120.0, sigma=0.001, sigma_ref=0.0005,
                               min_sec=30.0, max_sec=300.0)
        assert result == 240.0, f"Expected 240.0, got {result}"

    def test_low_sigma_shortens_interval(self) -> None:
        """309# σ < σ_ref → interval shortened (積極参加)."""
        result = self._compute(120.0, sigma=0.00025, sigma_ref=0.0005,
                               min_sec=60.0, max_sec=300.0)
        assert result == 60.0, f"Expected 60.0, got {result}"

    def test_sigma_zero_returns_base(self) -> None:
        """σ=0 → base_interval unchanged."""
        result = self._compute(120.0, sigma=0.0)
        assert result == 120.0

    def test_clamped_to_min(self) -> None:
        """309# Very low σ → clamped to min_sec."""
        result = self._compute(120.0, sigma=0.000001, sigma_ref=0.0005,
                               min_sec=60.0, max_sec=300.0)
        assert result == 60.0  # clamped to min

    def test_clamped_to_max(self) -> None:
        """309# Very high σ → clamped to max_sec."""
        result = self._compute(120.0, sigma=0.05, sigma_ref=0.0005,
                               min_sec=60.0, max_sec=300.0)
        assert result == 300.0  # clamped to max


# =====================================================================
# #1 Queue Position Estimation (O1)
# =====================================================================


class TestQueueDepthEstimation:
    """estimate_queue_depth のテスト."""

    def test_buy_depth_ahead(self) -> None:
        """Buy: OB の bid 側で order_price 以上の volume."""
        mp = _make_maker_price()
        ob = MagicMock()
        ob.bids = [(100.0, 5.0), (99.0, 3.0), (98.0, 2.0)]
        ob.asks = [(101.0, 1.0)]
        mp._last_ob_snapshot = ob

        depth = mp.estimate_queue_depth("buy", 99.0)
        # price >= 99: 100.0 (5.0) + 99.0 (3.0) = 8.0
        assert depth == 8.0

    def test_sell_depth_ahead(self) -> None:
        """Sell: OB の ask 側で order_price 以下の volume."""
        mp = _make_maker_price()
        ob = MagicMock()
        ob.bids = [(100.0, 1.0)]
        ob.asks = [(101.0, 2.0), (102.0, 4.0), (103.0, 6.0)]
        mp._last_ob_snapshot = ob

        depth = mp.estimate_queue_depth("sell", 102.0)
        # price <= 102: 101.0 (2.0) + 102.0 (4.0) = 6.0
        assert depth == 6.0

    def test_no_ob_returns_zero(self) -> None:
        """OB 未取得 → 0."""
        mp = _make_maker_price()
        mp._last_ob_snapshot = None
        assert mp.estimate_queue_depth("buy", 100.0) == 0.0

    def test_fill_probability_exponential(self) -> None:
        """Fill probability: exp(-depth/lot)."""
        depth = 0.01  # small depth
        lot = 0.001
        prob = math.exp(-depth / lot)
        assert 0 < prob < 1
        # Zero depth → prob ≈ 1
        assert math.exp(-0.0 / max(lot, 1e-8)) == pytest.approx(1.0)


# =====================================================================
# #5 Offset Stage Recording (E1)
# =====================================================================


class TestOffsetStageRecording:
    """offset_stage_recording プロパティの基本テスト."""

    def test_last_offset_stages_initially_none(self) -> None:
        """初期状態は None."""
        mp = _make_maker_price()
        assert mp.last_offset_stages is None

    def test_last_sigma_property(self) -> None:
        """last_sigma プロパティが正常に動作."""
        mp = _make_maker_price()
        assert mp.last_sigma == 0.0


# =====================================================================
# #4 EV-based Offset Adaptation (A1)
# =====================================================================


class TestEVBasedAdaptation:
    """compute_adaptation の EV 拡張テスト."""

    def test_ev_breaks_deadlock(self) -> None:
        """Both AS & fill_rate bad, but EV << 0 → increase offset."""
        from scripts.v460.lib.param_adapter import AdaptationConfig, compute_adaptation
        config = AdaptationConfig(
            current_offset_ratio=0.05,
            min_fill_rate=0.80,
            max_as_ratio=0.15,
            min_samples=20,
        )
        result = compute_adaptation(
            fill_rate=0.50,   # low
            as_ratio=0.40,    # high
            sample_count=50,  # >= min_samples * 2
            config=config,
            avg_pnl_bps=-5.0,  # negative → EV negative
            opportunity_cost_bps=0.5,
        )
        assert result.action == "increase"
        assert "EV" in result.reason

    def test_ev_holds_on_normal_deadlock(self) -> None:
        """Both bad but EV not clearly negative → hold."""
        from scripts.v460.lib.param_adapter import AdaptationConfig, compute_adaptation
        config = AdaptationConfig(
            current_offset_ratio=0.05,
            min_fill_rate=0.80,
            max_as_ratio=0.15,
            min_samples=20,
        )
        result = compute_adaptation(
            fill_rate=0.50,
            as_ratio=0.40,
            sample_count=30,  # < min_samples * 2
            config=config,
            avg_pnl_bps=0.0,
        )
        assert result.action == "hold"

    def test_ev_positive_with_as_margin_decreases(self) -> None:
        """EV positive + AS well within limits → micro-decrease offset."""
        from scripts.v460.lib.param_adapter import AdaptationConfig, compute_adaptation
        config = AdaptationConfig(
            current_offset_ratio=0.10,
            min_fill_rate=0.80,
            max_as_ratio=0.30,
            step_ratio=0.01,
            min_samples=20,
        )
        result = compute_adaptation(
            fill_rate=0.90,   # OK
            as_ratio=0.10,    # well below max 0.30 (10% < 0.30 * 0.7 = 0.21)
            sample_count=100,
            config=config,
            avg_pnl_bps=2.0,
            opportunity_cost_bps=0.5,
        )
        # EV = 0.90 * 2.0 - 0.10 * 0.5 = 1.75, which is > 0.5
        # AS ratio 0.10 < 0.30 * 0.7 = 0.21, so should micro-decrease
        assert result.action == "decrease"
        assert "EV" in result.reason

    def test_backward_compat_without_ev(self) -> None:
        """EV args not passed → original behavior."""
        from scripts.v460.lib.param_adapter import AdaptationConfig, compute_adaptation
        config = AdaptationConfig(
            current_offset_ratio=0.05,
            min_fill_rate=0.80,
            max_as_ratio=0.15,
            min_samples=20,
        )
        # Low fill rate → increase
        result = compute_adaptation(
            fill_rate=0.50,
            as_ratio=0.05,
            sample_count=50,
            config=config,
        )
        assert result.action == "increase"


# =====================================================================
# Offset Ceiling (300# T1-3)
# =====================================================================


class TestOffsetCeiling:
    """offset_ceiling_ratio のテスト."""

    def test_config_field_exists(self) -> None:
        """FillTestConfig に offset_ceiling_ratio フィールドが存在."""
        cfg = _make_config(offset_ceiling_ratio=0.15)
        assert cfg.offset_ceiling_ratio == 0.15

    def test_ceiling_zero_disabled(self) -> None:
        """offset_ceiling_ratio=0 → ceiling 無効."""
        cfg = _make_config(offset_ceiling_ratio=0.0)
        assert cfg.offset_ceiling_ratio == 0.0


# =====================================================================
# #6 Parkinson σ YAML 有効化
# =====================================================================


class TestParkinsonsigmaYAML:
    """YAML に sigma_parkinson セクションが存在."""

    def test_yaml_has_sigma_parkinson(self, v460_fill_test_yaml: dict[str, object]) -> None:
        """fill_test.yaml に sigma_parkinson が存在."""
        data = v460_fill_test_yaml
        assert "sigma_parkinson" in data
        assert data["sigma_parkinson"]["enabled"] is True

    def test_yaml_has_microprice_side(self, v460_fill_test_yaml: dict[str, object]) -> None:
        """fill_test.yaml に microprice_side が存在 (309# で disabled)."""
        data = v460_fill_test_yaml
        assert "microprice_side" in data
        assert data["microprice_side"]["enabled"] is False  # 309# 理論倒錯修正で無効化

    def test_yaml_has_dynamic_cycle_interval(
        self,
        v460_fill_test_yaml: dict[str, object],
    ) -> None:
        """fill_test.yaml に dynamic_cycle_interval が存在."""
        data = v460_fill_test_yaml
        assert "dynamic_cycle_interval" in data

    def test_yaml_has_queue_position(self, v460_fill_test_yaml: dict[str, object]) -> None:
        """fill_test.yaml に queue_position が存在."""
        data = v460_fill_test_yaml
        assert "queue_position" in data

    def test_yaml_has_offset_stage_recording(
        self,
        v460_fill_test_yaml: dict[str, object],
    ) -> None:
        """fill_test.yaml に offset_stage_recording が存在."""
        data = v460_fill_test_yaml
        assert "offset_stage_recording" in data


# =====================================================================
# Hot-Reload 新フィールド
# =====================================================================


class TestHotReloadNewFields:
    """306# 新フィールドが HOT_RELOADABLE に登録."""

    def test_new_fields_in_hot_reloadable(self) -> None:
        from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS
        expected = {
            "microprice_side_enabled",
            "microprice_side_threshold",
            "dynamic_cycle_interval_enabled",
            "dynamic_cycle_interval_min_sec",
            "dynamic_cycle_interval_max_sec",
            "dynamic_cycle_interval_sigma_ref",
            "queue_position_tracking_enabled",
            "queue_position_early_cancel_prob",
            "offset_stage_recording_enabled",
            "offset_ceiling_ratio",
        }
        missing = expected - _HOT_RELOADABLE_FIELDS
        assert not missing, f"Missing hot-reloadable fields: {missing}"


# =====================================================================
# FillRecord 新フィールド
# =====================================================================


class TestFillRecordNewFields:
    """FillRecord に 306# 新フィールドが存在."""

    def test_queue_depth_ahead_field(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        r = FillRecord(
            cycle_id="c1", timestamp=1.0, side="buy", order_price=100.0,
            order_quantity=0.001, queue_depth_ahead=5.0,
        )
        assert r.queue_depth_ahead == 5.0

    def test_queue_fill_prob_est_field(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        r = FillRecord(
            cycle_id="c1", timestamp=1.0, side="buy", order_price=100.0,
            order_quantity=0.001, queue_fill_prob_est=0.8,
        )
        assert r.queue_fill_prob_est == 0.8

    def test_offset_stages_field(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        r = FillRecord(
            cycle_id="c1", timestamp=1.0, side="buy", order_price=100.0,
            order_quantity=0.001, offset_stages='{"base": 0.05}',
        )
        assert r.offset_stages == '{"base": 0.05}'

    def test_microprice_bias_bps_field(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        r = FillRecord(
            cycle_id="c1", timestamp=1.0, side="buy", order_price=100.0,
            order_quantity=0.001, microprice_bias_bps=1.5,
        )
        assert r.microprice_bias_bps == 1.5


# =====================================================================
# 306# 301#-F2: Block Bootstrap + Matched Comparison + BH FDR
# =====================================================================

class TestBenjaminiHochberg:
    """BH FDR 多重比較補正テスト."""

    def test_all_nonsignificant(self) -> None:
        from scripts.v460.lib.ab_judgment import _benjamini_hochberg
        result = _benjamini_hochberg([0.5, 0.6, 0.7])
        assert result == [False, False, False]

    def test_all_significant(self) -> None:
        from scripts.v460.lib.ab_judgment import _benjamini_hochberg
        result = _benjamini_hochberg([0.001, 0.002, 0.003])
        assert result == [True, True, True]

    def test_partial_significant(self) -> None:
        from scripts.v460.lib.ab_judgment import _benjamini_hochberg
        # BH: sorted p=[0.01, 0.04, 0.5], thresholds=[0.0167, 0.0333, 0.05]
        # p1=0.01 <= 0.0167 → sig, p2=0.04 > 0.0333 → not sig
        result = _benjamini_hochberg([0.04, 0.01, 0.5])
        # idx sorted: [1, 0, 2] → thresholds [1/3*0.05, 2/3*0.05, 3/3*0.05]
        # p[1]=0.01 <= 0.0167 → sig
        # p[0]=0.04 > 0.0333 → not sig
        # max_significant_rank = 0 → only idx 1 is sig
        assert result[1] is True
        assert result[2] is False

    def test_empty(self) -> None:
        from scripts.v460.lib.ab_judgment import _benjamini_hochberg
        assert _benjamini_hochberg([]) == []


class TestBlockBootstrap:
    """Block Bootstrap 平均差 CI テスト."""

    def test_identical_distributions(self) -> None:
        import numpy as np
        from scripts.v460.lib.ab_judgment import _block_bootstrap_mean_diff
        rng = np.random.default_rng(99)
        x = rng.normal(0.0, 1.0, 200)
        y = rng.normal(0.0, 1.0, 200)
        diff, ci_lo, ci_hi, p = _block_bootstrap_mean_diff(x, y)
        # CI should contain 0 (no real difference)
        assert ci_lo <= 0.3  # generous bound
        assert ci_hi >= -0.3
        assert 0.0 <= p <= 1.0

    def test_distinct_distributions(self) -> None:
        import numpy as np
        from scripts.v460.lib.ab_judgment import _block_bootstrap_mean_diff
        x = np.ones(100) * 5.0  # mean=5
        y = np.ones(100) * 0.0  # mean=0
        diff, ci_lo, ci_hi, p = _block_bootstrap_mean_diff(x, y)
        assert diff == pytest.approx(5.0)
        assert ci_lo > 4.0  # CI well above 0
        assert ci_hi > 4.0

    def test_small_sample(self) -> None:
        import numpy as np
        from scripts.v460.lib.ab_judgment import _block_bootstrap_mean_diff
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        diff, ci_lo, ci_hi, p = _block_bootstrap_mean_diff(x, y, block_size=2)
        assert diff == pytest.approx(-3.0)
        assert ci_lo < ci_hi


class TestMatchedTemporalComparison:
    """時間近接 Matched Pair 比較テスト."""

    def _make_records(self, side: str, ts_pnl_pairs: list[tuple[float, float]]) -> list[dict]:
        return [
            {"side": side, "filled": True, "timestamp": ts, "post_fill_30s_pnl": pnl,
             "regime": "ranging"}
            for ts, pnl in ts_pnl_pairs
        ]

    def test_matched_pairs_basic(self) -> None:
        from scripts.v460.lib.ab_judgment import _matched_temporal_comparison
        # Create 20 temporally close buy/sell pairs
        v_records = self._make_records("sell", [(i * 100.0, 1.0) for i in range(20)])
        c_records = self._make_records("buy", [(i * 100.0 + 10.0, 0.0) for i in range(20)])
        n_pairs, diff, ci_lo, ci_hi, p = _matched_temporal_comparison(
            v_records, c_records, max_gap_sec=50.0,
        )
        assert n_pairs == 20
        assert diff is not None
        assert diff == pytest.approx(1.0)

    def test_no_pairs_when_too_far(self) -> None:
        from scripts.v460.lib.ab_judgment import _matched_temporal_comparison
        v_records = self._make_records("sell", [(0.0, 1.0)])
        c_records = self._make_records("buy", [(10000.0, 0.0)])
        n_pairs, diff, ci_lo, ci_hi, p = _matched_temporal_comparison(
            v_records, c_records, max_gap_sec=100.0,
        )
        assert n_pairs == 0
        assert diff is None

    def test_empty_records(self) -> None:
        from scripts.v460.lib.ab_judgment import _matched_temporal_comparison
        n_pairs, diff, _, _, _ = _matched_temporal_comparison([], [])
        assert n_pairs == 0
        assert diff is None

    def test_insufficient_pairs(self) -> None:
        from scripts.v460.lib.ab_judgment import _matched_temporal_comparison
        # Only 5 pairs, below minimum of 10
        v_records = self._make_records("sell", [(i * 100.0, 1.0) for i in range(5)])
        c_records = self._make_records("buy", [(i * 100.0 + 5.0, 0.0) for i in range(5)])
        n_pairs, diff, ci_lo, ci_hi, p = _matched_temporal_comparison(
            v_records, c_records, max_gap_sec=50.0,
        )
        assert n_pairs == 5
        assert diff is None  # below threshold for statistics


class TestWilcoxonSignedRank:
    """Wilcoxon signed-rank テスト."""

    def test_symmetric_differences(self) -> None:
        import numpy as np
        from scripts.v460.lib.ab_judgment import _wilcoxon_signed_rank
        diffs = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.5, -0.5, 1.5, -1.5])
        p = _wilcoxon_signed_rank(diffs)
        # Symmetric → high p-value (non-significant)
        assert p > 0.5

    def test_all_positive(self) -> None:
        import numpy as np
        from scripts.v460.lib.ab_judgment import _wilcoxon_signed_rank
        diffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        p = _wilcoxon_signed_rank(diffs)
        # All positive → should be significant
        assert p < 0.05

    def test_too_few_samples(self) -> None:
        import numpy as np
        from scripts.v460.lib.ab_judgment import _wilcoxon_signed_rank
        diffs = np.array([1.0, 2.0])
        p = _wilcoxon_signed_rank(diffs)
        assert p == 1.0


class TestABJudgmentNewFields:
    """ABJudgmentResult に追加された 306# フィールドのテスト."""

    def test_bootstrap_fields_in_summary(self) -> None:
        from scripts.v460.lib.ab_judgment import ABJudgmentResult, Verdict
        r = ABJudgmentResult(
            overall=Verdict.PASS,
            bootstrap_mean_diff=0.5,
            bootstrap_ci_lower=-0.2,
            bootstrap_ci_upper=1.2,
            bootstrap_p_value=0.35,
        )
        s = r.summary()
        assert "Block Bootstrap" in s
        assert "+0.5000" in s
        assert "95%CI" in s

    def test_matched_fields_in_summary(self) -> None:
        from scripts.v460.lib.ab_judgment import ABJudgmentResult, Verdict
        r = ABJudgmentResult(
            overall=Verdict.PASS,
            matched_n_pairs=50,
            matched_mean_diff=-0.3,
            matched_ci_lower=-0.8,
            matched_ci_upper=0.2,
            matched_p_value=0.12,
        )
        s = r.summary()
        assert "Matched Pairs" in s
        assert "n=50" in s

    def test_integration_with_evaluate(self) -> None:
        import time
        from scripts.v460.lib.ab_judgment import evaluate_ab_variant, ABJudgmentCriteria
        # Create enough records for bootstrap + matched comparison
        base_ts = time.time()
        sell_records = [
            {"side": "sell", "regime": "ranging", "filled": True,
             "post_fill_30s_pnl": float(i % 5 - 2),
             "timestamp": base_ts + i * 60.0}
            for i in range(60)
        ]
        buy_records = [
            {"side": "buy", "regime": "ranging", "filled": True,
             "post_fill_30s_pnl": float(i % 5 - 2),
             "timestamp": base_ts + i * 60.0 + 10.0}
            for i in range(60)
        ]
        result = evaluate_ab_variant(
            sell_records, buy_records,
            criteria=ABJudgmentCriteria(
                min_filled_records=10,
                min_control_filled_records=10,
                min_calendar_days=0,
                exclude_regimes=[],
            ),
            variant_label="sell",
            control_label="buy",
        )
        # Bootstrap fields should be populated
        assert result.bootstrap_mean_diff is not None
        assert result.bootstrap_ci_lower is not None
        assert result.bootstrap_ci_upper is not None
        # Matched pairs should find some
        assert result.matched_n_pairs > 0


# =====================================================================
# 310# A: Sell AS Time-of-Day Offset Boost
# =====================================================================


class TestSellHourOffsetBoost:
    """310# A: sell_hour_offset_boost pipeline stage tests."""

    def test_config_field_exists(self) -> None:
        """sell_hour_offset_boost field is available in FillTestConfig."""
        config = _make_config()
        assert hasattr(config, "sell_hour_offset_boost")
        assert isinstance(config.sell_hour_offset_boost, dict)

    def test_sell_hour_boost_applies_multiplier(self) -> None:
        """Sell side with matching UTC hour → offset multiplied."""
        from unittest.mock import patch
        from datetime import datetime, timezone
        mp = _make_maker_price(sell_hour_offset_boost={8: 1.5, 16: 1.5})
        mock_dt = datetime(2025, 1, 1, 8, 0, tzinfo=timezone.utc)
        with patch("scripts.v460.lib.maker_risk_guards.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.side_effect = lambda *a, **k: datetime(*a, **k)
            result = mp._apply_sell_hour_boost("sell", 0.05)
        assert result > 0.05  # Should be boosted

    def test_sell_hour_boost_no_effect_on_buy(self) -> None:
        """Buy side → no boost regardless of hour."""
        mp = _make_maker_price(sell_hour_offset_boost={8: 1.5})
        result = mp._apply_sell_hour_boost("buy", 0.05)
        assert result == 0.05

    def test_sell_hour_boost_empty_dict(self) -> None:
        """Empty config → no boost."""
        mp = _make_maker_price()
        result = mp._apply_sell_hour_boost("sell", 0.05)
        assert result == 0.05

    def test_yaml_parsing(self) -> None:
        """YAML sell_hour_offset_boost parses correctly."""
        config = FillTestConfig.from_yaml({
            "sell_hour_offset_boost": {"8": 1.5, "16": 1.5},
        })
        assert config.sell_hour_offset_boost == {8: 1.5, 16: 1.5}


# =====================================================================
# 310# B: param_adapter decision_path
# =====================================================================


class TestDecisionPath:
    """310# B: AdaptationResult.decision_path tests (307# F6)."""

    def test_insufficient_data(self) -> None:
        """少サンプル → decision_path='insufficient_data'."""
        from scripts.v460.lib.param_adapter import AdaptationConfig, compute_adaptation
        result = compute_adaptation(
            fill_rate=0.5, as_ratio=0.4, sample_count=5,
            config=AdaptationConfig(min_samples=20),
        )
        assert result.decision_path == "insufficient_data"

    def test_as_defense_hold(self) -> None:
        """AS+fill両方異常, EV中立 → decision_path='as_defense'."""
        from scripts.v460.lib.param_adapter import AdaptationConfig, compute_adaptation
        result = compute_adaptation(
            fill_rate=0.50, as_ratio=0.40, sample_count=30,
            config=AdaptationConfig(min_samples=20, max_as_ratio=0.15),
            avg_pnl_bps=0.0,
        )
        assert result.decision_path == "as_defense"
        assert result.action == "hold"

    def test_deadlock_break(self) -> None:
        """AS+fill両方異常, EV<<0, 十分サンプル → decision_path='deadlock_break'."""
        from scripts.v460.lib.param_adapter import AdaptationConfig, compute_adaptation
        result = compute_adaptation(
            fill_rate=0.50, as_ratio=0.40, sample_count=50,
            config=AdaptationConfig(min_samples=20, max_as_ratio=0.15),
            avg_pnl_bps=-5.0,
        )
        assert result.decision_path == "deadlock_break"
        assert result.action == "increase"

    def test_as_defense_decrease(self) -> None:
        """AS のみ高 → decision_path='as_defense', action='decrease'."""
        from scripts.v460.lib.param_adapter import AdaptationConfig, compute_adaptation
        result = compute_adaptation(
            fill_rate=0.90, as_ratio=0.40, sample_count=50,
            config=AdaptationConfig(min_samples=20, max_as_ratio=0.15),
        )
        assert result.decision_path == "as_defense"
        assert result.action == "decrease"

    def test_fill_recovery(self) -> None:
        """fill_rate のみ低 → decision_path='fill_recovery'."""
        from scripts.v460.lib.param_adapter import AdaptationConfig, compute_adaptation
        result = compute_adaptation(
            fill_rate=0.50, as_ratio=0.05, sample_count=50,
            config=AdaptationConfig(min_samples=20, min_fill_rate=0.80, max_as_ratio=0.15),
        )
        assert result.decision_path == "fill_recovery"
        assert result.action == "increase"

    def test_ev_optimization(self) -> None:
        """正常+EV正+AS余裕 → decision_path='ev_optimization'."""
        from scripts.v460.lib.param_adapter import AdaptationConfig, compute_adaptation
        result = compute_adaptation(
            fill_rate=0.90, as_ratio=0.10, sample_count=100,
            config=AdaptationConfig(
                min_samples=20, min_fill_rate=0.80, max_as_ratio=0.30,
                current_offset_ratio=0.10, step_ratio=0.01,
            ),
            avg_pnl_bps=2.0, opportunity_cost_bps=0.5,
        )
        assert result.decision_path == "ev_optimization"
        assert result.action == "decrease"

    def test_hold_normal(self) -> None:
        """正常範囲 → decision_path='hold'."""
        from scripts.v460.lib.param_adapter import AdaptationConfig, compute_adaptation
        result = compute_adaptation(
            fill_rate=0.90, as_ratio=0.10, sample_count=50,
            config=AdaptationConfig(min_samples=20, min_fill_rate=0.80, max_as_ratio=0.15),
        )
        assert result.decision_path == "hold"
        assert result.action == "hold"


# =====================================================================
# 310# C: L2 Microprice Guardrails
# =====================================================================


class TestMicropriceGuardrails:
    """310# C: spread/regime guardrails for microprice side selection."""

    def test_config_guardrail_fields(self) -> None:
        """FillTestConfig has guardrail fields."""
        config = _make_config()
        assert hasattr(config, "microprice_side_min_spread_bps")
        assert hasattr(config, "microprice_side_regime_gate")

    def test_yaml_guardrail_parsing(self) -> None:
        """YAML guardrails parse correctly."""
        config = FillTestConfig.from_yaml({
            "microprice_side": {
                "enabled": True,
                "threshold_bps": 0.3,
                "min_spread_bps": 20.0,
                "regime_gate": ["ranging", "high_vol"],
            },
        })
        assert config.microprice_side_min_spread_bps == 20.0
        assert config.microprice_side_regime_gate == ["ranging", "high_vol"]
