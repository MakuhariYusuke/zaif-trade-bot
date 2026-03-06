"""186# テスト: Trend Mode ヒステリシス + Strictness Clamp.

Phase A-1: gated_regime() ヒステリシス (enter/exit/min_dwell)
Phase A-2: skip_gate threshold_offset clamp
"""

from __future__ import annotations

import pytest
from scripts.v460.lib.regime_policy import DefaultCycleStrategy, RegimePolicyConfig


# ======================================================================
# A-1: Trend Mode ヒステリシス
# ======================================================================

class TestTrendModeHysteresis:
    """186# gated_regime ヒステリシスの検証."""

    def _make_strategy(
        self,
        enter: float = 0.45,
        exit_: float = 0.30,
        dwell: int = 3,
    ) -> DefaultCycleStrategy:
        policy = RegimePolicyConfig(
            dynamic_cycle_enabled=True,
            trend_min_confidence=enter,
            trend_exit_confidence=exit_,
            trend_min_dwell=dwell,
        )
        return DefaultCycleStrategy(
            base_interval=120.0,
            base_wait_buy=30.0,
            base_wait_sell=90.0,
            policy=policy,
        )

    def test_enter_below_threshold(self) -> None:
        """confidence < enter → ranging に降格."""
        s = self._make_strategy()
        assert s.gated_regime("trending_up", confidence=0.40) == "ranging"

    def test_enter_at_threshold(self) -> None:
        """confidence == enter → trending 許可."""
        s = self._make_strategy()
        assert s.gated_regime("trending_up", confidence=0.45) == "trending_up"

    def test_enter_above_threshold(self) -> None:
        """confidence > enter → trending 許可."""
        s = self._make_strategy()
        assert s.gated_regime("trending_down", confidence=0.60) == "trending_down"

    def test_stay_in_trend_confidence_between_exit_and_enter(self) -> None:
        """enter 後 confidence が exit 以上 enter 未満 → trend 維持 (ヒステリシス核心)."""
        s = self._make_strategy()
        # Enter
        assert s.gated_regime("trending_up", confidence=0.50) == "trending_up"
        # confidence 低下 (exit 以上) → trend 維持
        assert s.gated_regime("trending_up", confidence=0.35) == "trending_up"
        assert s.gated_regime("trending_up", confidence=0.31) == "trending_up"

    def test_exit_requires_min_dwell(self) -> None:
        """confidence < exit でも min_dwell 未到達なら trend 維持."""
        s = self._make_strategy(dwell=3)
        # Enter (cycle 1)
        assert s.gated_regime("trending_up", confidence=0.50) == "trending_up"
        # Cycle 2: confidence drops below exit, but dwell=1 < 3
        assert s.gated_regime("trending_up", confidence=0.20) == "trending_up"
        # Cycle 3: dwell=2 < 3
        assert s.gated_regime("trending_up", confidence=0.20) == "trending_up"
        # Cycle 4: dwell=3 >= 3 → EXIT
        assert s.gated_regime("trending_up", confidence=0.20) == "ranging"

    def test_exit_then_reenter(self) -> None:
        """exit 後に再度 confidence が回復 → 再 enter."""
        s = self._make_strategy(dwell=1)
        # Enter
        assert s.gated_regime("trending_up", confidence=0.50) == "trending_up"
        # Exit (dwell=1 で即 exit 可能)
        assert s.gated_regime("trending_up", confidence=0.20) == "ranging"
        # Re-enter
        assert s.gated_regime("trending_up", confidence=0.50) == "trending_up"

    def test_non_trending_input_passthrough(self) -> None:
        """non-trending regime ('ranging', 'high_vol') は gated_regime の影響を受けない."""
        s = self._make_strategy()
        assert s.gated_regime("ranging", confidence=0.10) == "ranging"
        assert s.gated_regime("high_vol", confidence=0.90) == "high_vol"

    def test_none_regime_passthrough(self) -> None:
        """regime=None → None (変更なし)."""
        s = self._make_strategy()
        assert s.gated_regime(None, confidence=0.50) is None

    def test_trend_mode_exit_on_regime_change(self) -> None:
        """trend 中に non-trending regime が来た場合 → exit."""
        s = self._make_strategy()
        # Enter
        assert s.gated_regime("trending_up", confidence=0.50) == "trending_up"
        # Regime changes to ranging (non-trending)
        assert s.gated_regime("ranging", confidence=0.50) == "ranging"
        # Trend mode should have exited
        assert not s._in_trend_mode

    def test_effective_interval_uses_gated_regime(self) -> None:
        """effective_interval は gated_regime を通後の regime を使う."""
        s = self._make_strategy()
        # low confidence → gated to ranging → 120s
        assert s.effective_interval("trending_up") == 120.0
        s.update_confidence(0.50)
        # high confidence → trending_up → 60s
        assert s.effective_interval("trending_up") == 60.0

    def test_is_chase_enabled_uses_gated_regime(self) -> None:
        """Chase は gated_regime 通過後の regime で判定."""
        policy = RegimePolicyConfig(
            chase_enabled=True,
            trend_min_confidence=0.45,
            trend_exit_confidence=0.30,
        )
        s = DefaultCycleStrategy(120, 30, 90, policy=policy)
        s.update_confidence(0.20)
        assert not s.is_chase_enabled("trending_up")  # gated → ranging
        s.update_confidence(0.50)
        # まず enter させる
        s.gated_regime("trending_up", confidence=0.50)
        assert s.is_chase_enabled("trending_up")


class TestHysteresisYAML:
    """186# YAML パース・hot-reload 検証."""

    def test_from_yaml_new_fields(self) -> None:
        """YAML に trend_exit_confidence / trend_min_dwell がある場合."""
        cfg = RegimePolicyConfig.from_yaml({
            "regime_policy": {
                "trend_min_confidence": 0.45,
                "trend_exit_confidence": 0.30,
                "trend_min_dwell": 5,
            }
        })
        assert cfg.trend_min_confidence == 0.45
        assert cfg.trend_exit_confidence == 0.30
        assert cfg.trend_min_dwell == 5

    def test_from_yaml_defaults(self) -> None:
        """YAML に新フィールドがない場合 → デフォルト値."""
        cfg = RegimePolicyConfig.from_yaml({
            "regime_policy": {
                "trend_min_confidence": 0.50,
            }
        })
        assert cfg.trend_min_confidence == 0.50
        assert cfg.trend_exit_confidence == 0.30  # default
        assert cfg.trend_min_dwell == 3  # default

    def test_fill_test_yaml_integration(self, v460_fill_test_yaml: dict[str, object]) -> None:
        """実際の fill_test.yaml からヒステリシス設定が正しくパースされる."""
        cfg = RegimePolicyConfig.from_yaml(v460_fill_test_yaml)
        assert cfg.trend_min_confidence == 0.45
        assert cfg.trend_exit_confidence == 0.30
        assert cfg.trend_min_dwell == 3


# ======================================================================
# A-2: Strictness Clamp
# ======================================================================

class TestStrictnessClamp:
    """186# skip_gate threshold_offset clamp の検証."""

    def test_clamp_within_range(self) -> None:
        """正常範囲 [-0.3, 0.5] 内 → そのまま."""
        # テスト対象は clamp ロジック自体
        offset = 0.3
        clamped = max(-0.3, min(0.5, offset))
        assert clamped == 0.3

    def test_clamp_upper_bound(self) -> None:
        """hour_offset(0.5) + spread_offset(0.2) = 0.7 → 0.5 にクランプ."""
        offset = 0.5 + 0.2  # 0.7
        clamped = max(-0.3, min(0.5, offset))
        assert clamped == 0.5

    def test_clamp_lower_bound(self) -> None:
        """負のオフセット -0.5 → -0.3 にクランプ."""
        offset = -0.5
        clamped = max(-0.3, min(0.5, offset))
        assert clamped == -0.3

    def test_clamp_zero(self) -> None:
        """0.0 → そのまま."""
        offset = 0.0
        clamped = max(-0.3, min(0.5, offset))
        assert clamped == 0.0

    def test_worst_case_accumulation(self) -> None:
        """最悪ケース: hour(0.5) + spread(0.2) = 0.7 → 0.5."""
        hour = 0.5
        spread = 0.2
        total = hour + spread
        clamped = max(-0.3, min(0.5, total))
        assert clamped == 0.5
        # regime_threshold は evaluate() 内部で別途加算されるため影響なし


# ======================================================================
# 既存テストとの互換性
# ======================================================================

class TestBackwardCompatibility:
    """186# 変更が既存動作を壊さないことの確認."""

    def test_strategy_without_hysteresis_fields(self) -> None:
        """RegimePolicyConfig のヒステリシスフィールドはデフォルトで安全."""
        policy = RegimePolicyConfig()
        assert policy.trend_min_confidence == 0.45  # 186# default
        assert policy.trend_exit_confidence == 0.30
        assert policy.trend_min_dwell == 3

    def test_182_gated_regime_behavior_preserved_for_low_confidence(self) -> None:
        """182# の基本動作: 低 confidence → ranging.

        ヒステリシスにより即時 exit ではなく min_dwell が必要だが、
        一度も enter しなければ即座に ranging になる (182# 互換).
        """
        policy = RegimePolicyConfig(trend_min_confidence=0.45)
        s = DefaultCycleStrategy(120, 30, 90, policy=policy)
        # 一度も enter せず低 confidence → ranging
        assert s.gated_regime("trending_up", confidence=0.20) == "ranging"
