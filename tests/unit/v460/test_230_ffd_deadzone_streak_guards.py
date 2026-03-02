"""230# テスト: FFD deadzone/streak + MCB/SAD None guard + hasattr 排除.

変更概要:
  H-1: FFD Layer 2 deadzone — 正常 spread cost で L2 誤発火しない
  H-2: FFD boost gradual release — Kyle 1985 連続正常 fill streak 要求
  H-3: MCB/SAD None guard — _mcb/_sad is None で AttributeError 回避
  H-4: regime_detector hasattr→init 変換
  M-1: fill_cycle_executor hasattr 排除 (8/10, 2 legitimate 残留)
  Config: ffd_l2_deadzone_bps / ffd_boost_release_streak 新規バリデーション
"""

from __future__ import annotations

import inspect

import pytest

from scripts.v460.lib.fast_fill_defense import (
    FastFillDefense,
    FastFillDefenseConfig,
)
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.regime_detector import FillTestRegimeDetector, RegimeConfig


# ======================================================================
# Helpers
# ======================================================================


def _make_ffd(
    *,
    deadzone_bps: float = 3.0,
    streak: int = 3,
    threshold_sec: float = 5.0,
    offset_boost: float = 2.0,
) -> FastFillDefense:
    """テスト用 FFD を生成."""
    cfg = FastFillDefenseConfig(
        enabled=True,
        threshold_sec=threshold_sec,
        offset_boost=offset_boost,
        l2_deadzone_bps=deadzone_bps,
        boost_release_streak=streak,
    )
    return FastFillDefense(cfg, base_offset_ratio=0.005)


def _activate_boost(ffd: FastFillDefense, side: str = "buy") -> None:
    """fast fill + L1 negative edge で boost を確実に有効化."""
    fp = 101_000 if side == "buy" else 99_000
    ffd.evaluate_fill(
        side,
        queue_wait_sec=1.0,
        fill_price=fp,
        mid_at_fill=100_000,
    )
    assert ffd.is_boost_active(side)


def _normal_fill(ffd: FastFillDefense, side: str = "buy") -> None:
    """正常 fill (not fast, no negative edge)."""
    ffd.evaluate_fill(
        side,
        queue_wait_sec=60.0,
        fill_price=99_500 if side == "buy" else 100_500,
        mid_at_fill=100_000,
    )


# ======================================================================
# H-1: FFD Layer 2 deadzone (AS theory)
# ======================================================================


class TestFFDL2Deadzone:
    """H-1: 正常スプレッドコスト (~2-3bps) による L2 誤発火を防止."""

    def test_within_deadzone_no_trigger(self) -> None:
        """pnl = -2.5bps, deadzone = 3.0bps → |pnl| < deadzone → L2 不発動."""
        ffd = _make_ffd(deadzone_bps=3.0)
        ffd.evaluate_fill(
            "sell",
            queue_wait_sec=3.0,
            fill_price=100_500,
            mid_at_fill=100_000,
            post_fill_pnl_bps=-2.5,  # -2.5 > -3.0 → within deadzone
        )
        assert not ffd.is_boost_active("sell")

    def test_beyond_deadzone_triggers(self) -> None:
        """pnl = -5.0bps, deadzone = 3.0bps → |pnl| > deadzone → L2 発動."""
        ffd = _make_ffd(deadzone_bps=3.0)
        ffd.evaluate_fill(
            "sell",
            queue_wait_sec=3.0,
            fill_price=100_500,
            mid_at_fill=100_000,
            post_fill_pnl_bps=-5.0,  # -5.0 < -3.0 → beyond deadzone
        )
        assert ffd.is_boost_active("sell")

    def test_exactly_at_deadzone_boundary_no_trigger(self) -> None:
        """pnl = -3.0bps, deadzone = 3.0bps → not strictly less → no trigger."""
        ffd = _make_ffd(deadzone_bps=3.0)
        ffd.evaluate_fill(
            "sell",
            queue_wait_sec=3.0,
            fill_price=100_500,
            mid_at_fill=100_000,
            post_fill_pnl_bps=-3.0,  # -3.0 is NOT < -3.0
        )
        assert not ffd.is_boost_active("sell")

    def test_deadzone_zero_behaves_like_old(self) -> None:
        """deadzone=0 → 旧挙動 (pnl<0 で即発動)."""
        ffd = _make_ffd(deadzone_bps=0.0)
        ffd.evaluate_fill(
            "buy",
            queue_wait_sec=2.0,
            fill_price=99_900,  # L1: no negative edge (buy: price < mid)
            mid_at_fill=100_000,
            post_fill_pnl_bps=-0.5,  # small negative, but deadzone=0
        )
        assert ffd.is_boost_active("buy")

    def test_positive_pnl_never_triggers_l2(self) -> None:
        """pnl>0 → deadzone 関係なく L2 は不発動."""
        ffd = _make_ffd(deadzone_bps=0.0)
        ffd.evaluate_fill(
            "buy",
            queue_wait_sec=2.0,
            fill_price=99_500,
            mid_at_fill=100_000,
            post_fill_pnl_bps=1.0,
        )
        assert not ffd.is_boost_active("buy")

    def test_l1_still_works_regardless_of_deadzone(self) -> None:
        """L1 (fill_price vs mid) は deadzone に影響されない."""
        ffd = _make_ffd(deadzone_bps=100.0)  # 大きな deadzone
        # buy: fill_price > mid → L1 negative edge → triggers
        ffd.evaluate_fill(
            "buy",
            queue_wait_sec=2.0,
            fill_price=101_000,
            mid_at_fill=100_000,
        )
        assert ffd.is_boost_active("buy")


# ======================================================================
# H-2: FFD boost gradual release (Kyle 1985)
# ======================================================================


class TestFFDBoostGradualRelease:
    """H-2: 情報漸次伝播 — N回連続正常fillで解除."""

    def test_single_normal_fill_not_enough(self) -> None:
        """streak=3 + 1 normal fill → boost 維持."""
        ffd = _make_ffd(streak=3)
        _activate_boost(ffd, "buy")
        _normal_fill(ffd, "buy")
        assert ffd.is_boost_active("buy")  # まだ解除されない

    def test_two_normal_fills_not_enough(self) -> None:
        """streak=3 + 2 normal fills → boost 維持."""
        ffd = _make_ffd(streak=3)
        _activate_boost(ffd, "buy")
        _normal_fill(ffd, "buy")
        _normal_fill(ffd, "buy")
        assert ffd.is_boost_active("buy")

    def test_three_normal_fills_deactivates(self) -> None:
        """streak=3 + 3 normal fills → boost 解除."""
        ffd = _make_ffd(streak=3)
        _activate_boost(ffd, "buy")
        for _ in range(3):
            _normal_fill(ffd, "buy")
        assert not ffd.is_boost_active("buy")
        assert ffd.get_boost_multiplier("buy") == 1.0

    def test_streak_resets_on_new_adverse(self) -> None:
        """途中で再度 adverse fill → streak リセット."""
        ffd = _make_ffd(streak=3)
        _activate_boost(ffd, "buy")

        # 2 normal fills
        _normal_fill(ffd, "buy")
        _normal_fill(ffd, "buy")
        assert ffd.is_boost_active("buy")

        # new adverse fill (fast + negative L1) → streak resets
        ffd.evaluate_fill(
            "buy",
            queue_wait_sec=1.0,
            fill_price=101_000,
            mid_at_fill=100_000,
        )
        assert ffd.is_boost_active("buy")
        assert ffd._get_state("buy").normal_fill_streak == 0

        # need 3 more normal fills now
        _normal_fill(ffd, "buy")
        _normal_fill(ffd, "buy")
        assert ffd.is_boost_active("buy")
        _normal_fill(ffd, "buy")
        assert not ffd.is_boost_active("buy")

    def test_streak_one_behaves_like_old(self) -> None:
        """streak=1 → 旧挙動 (1 normal fill で即解除)."""
        ffd = _make_ffd(streak=1)
        _activate_boost(ffd, "buy")
        _normal_fill(ffd, "buy")
        assert not ffd.is_boost_active("buy")

    def test_sell_side_streak(self) -> None:
        """sell 側でも streak logic が動作."""
        ffd = _make_ffd(streak=2)
        _activate_boost(ffd, "sell")
        _normal_fill(ffd, "sell")
        assert ffd.is_boost_active("sell")  # 1/2
        _normal_fill(ffd, "sell")
        assert not ffd.is_boost_active("sell")  # 2/2 → deactivated

    def test_side_isolation_with_streak(self) -> None:
        """buy streak は sell に影響しない."""
        ffd = _make_ffd(streak=2)
        _activate_boost(ffd, "buy")
        _activate_boost(ffd, "sell")

        _normal_fill(ffd, "buy")
        _normal_fill(ffd, "buy")  # buy streak=2 → deactivated
        assert not ffd.is_boost_active("buy")
        assert ffd.is_boost_active("sell")  # sell は未変更

    def test_reset_on_unfilled_clears_streak(self) -> None:
        """未約定リセットで streak もクリア."""
        ffd = _make_ffd(streak=3)
        _activate_boost(ffd, "buy")
        _normal_fill(ffd, "buy")
        _normal_fill(ffd, "buy")
        assert ffd._get_state("buy").normal_fill_streak == 2

        ffd.reset_on_unfilled("buy")
        assert not ffd.is_boost_active("buy")
        assert ffd._get_state("buy").normal_fill_streak == 0


# ======================================================================
# H-2 state persistence
# ======================================================================


class TestFFDStreakStatePersistence:
    """export/import で normal_fill_streak が保存・復元される."""

    def test_export_includes_streak(self) -> None:
        ffd = _make_ffd(streak=3)
        _activate_boost(ffd, "buy")
        _normal_fill(ffd, "buy")

        state = ffd.export_state()
        assert state["buy_normal_fill_streak"] == 1
        assert state["sell_normal_fill_streak"] == 0

    def test_import_restores_streak(self) -> None:
        ffd = _make_ffd(streak=3)
        state = {
            "buy_boost_active": True,
            "buy_boost_multiplier": 2.0,
            "buy_boost_activated_at": 12345.0,
            "buy_normal_fill_streak": 2,
            "sell_boost_active": False,
            "sell_boost_multiplier": 1.0,
            "sell_boost_activated_at": 0.0,
            "sell_normal_fill_streak": 0,
        }
        ffd.import_state(state)
        assert ffd._state_buy.normal_fill_streak == 2
        assert ffd._state_sell.normal_fill_streak == 0

    def test_import_missing_streak_defaults_zero(self) -> None:
        """旧バージョン state (streak なし) → 0 にフォールバック."""
        ffd = _make_ffd(streak=3)
        state = {
            "buy_boost_active": True,
            "buy_boost_multiplier": 2.0,
            "buy_boost_activated_at": 12345.0,
            "sell_boost_active": False,
            "sell_boost_multiplier": 1.0,
            "sell_boost_activated_at": 0.0,
        }
        ffd.import_state(state)
        assert ffd._state_buy.normal_fill_streak == 0
        assert ffd._state_sell.normal_fill_streak == 0


# ======================================================================
# H-3: MCB/SAD None guard
# ======================================================================


class TestMCBSADNoneGuard:
    """H-3: _mcb/_sad is None 時に AttributeError しない."""

    def test_orchestrator_mcb_none_attribute(self) -> None:
        """fill_loop_orchestrator に 'self._mcb is not None and' パターンがある."""
        from scripts.v460.lib import fill_loop_orchestrator as mod

        src = inspect.getsource(mod)
        # _mcb.config.enabled の前に None check がある
        assert "self._mcb is not None and self._mcb.config.enabled" in src

    def test_orchestrator_sad_none_attribute(self) -> None:
        """fill_loop_orchestrator に 'self._sad is not None and' パターンがある."""
        from scripts.v460.lib import fill_loop_orchestrator as mod

        src = inspect.getsource(mod)
        assert "self._sad is not None and self._sad.config.enabled" in src

    def test_no_bare_mcb_config_access(self) -> None:
        """None guard なしの self._mcb.config.enabled が残っていない."""
        from scripts.v460.lib import fill_loop_orchestrator as mod

        src = inspect.getsource(mod)
        lines = src.split("\n")
        for line in lines:
            stripped = line.strip()
            if "self._mcb.config.enabled" in stripped:
                assert "self._mcb is not None" in stripped, (
                    f"Unguarded _mcb access: {stripped}"
                )

    def test_no_bare_sad_config_access(self) -> None:
        """None guard なしの self._sad.config.enabled が残っていない."""
        from scripts.v460.lib import fill_loop_orchestrator as mod

        src = inspect.getsource(mod)
        lines = src.split("\n")
        for line in lines:
            stripped = line.strip()
            if "self._sad.config.enabled" in stripped:
                assert "self._sad is not None" in stripped, (
                    f"Unguarded _sad access: {stripped}"
                )


# ======================================================================
# H-4: regime_detector hasattr→init
# ======================================================================


class TestRegimeDetectorInit:
    """H-4: _last_result / _last_velocity_pct が __init__ で初期化."""

    def test_no_hasattr_in_source(self) -> None:
        """regime_detector.py のソースに hasattr() 呼び出しが含まれない."""
        from scripts.v460.lib import regime_detector as mod

        src = inspect.getsource(mod)
        # コメント中の "hasattr" は許容、実際の呼び出しのみ検出
        lines = src.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "hasattr(" not in stripped, (
                f"hasattr() call found: {stripped}"
            )

    def test_no_getattr_fallback_in_source(self) -> None:
        """regime_detector.py に getattr(..., default) のフォールバックがない."""
        from scripts.v460.lib import regime_detector as mod

        src = inspect.getsource(mod)
        # _last_velocity_pct の getattr パターンが除去されている
        assert 'getattr(self, "_last_velocity_pct"' not in src

    def test_last_volatility_ratio_before_update(self) -> None:
        """update() 呼び出し前に last_volatility_ratio が 1.0 を返す."""
        rd = FillTestRegimeDetector(RegimeConfig())
        assert rd.last_volatility_ratio == 1.0

    def test_current_confidence_before_update(self) -> None:
        """update() 呼び出し前に current_confidence が 0.0 を返す."""
        rd = FillTestRegimeDetector(RegimeConfig())
        assert rd.current_confidence == 0.0

    def test_init_attributes_exist(self) -> None:
        """__init__ で _last_result と _last_velocity_pct が宣言されている."""
        rd = FillTestRegimeDetector(RegimeConfig())
        assert hasattr(rd, "_last_result")
        assert hasattr(rd, "_last_velocity_pct")
        assert rd._last_result is None
        assert rd._last_velocity_pct == 0.0


# ======================================================================
# M-1: fill_cycle_executor hasattr 排除
# ======================================================================


class TestFillCycleExecutorHasattr:
    """M-1: fill_cycle_executor の hasattr を is not None に変換."""

    def test_no_hasattr_cycle_strategy(self) -> None:
        """hasattr(self, '_cycle_strategy') が排除されている."""
        from scripts.v460.lib import fill_cycle_executor as mod

        src = inspect.getsource(mod)
        assert 'hasattr(self, "_cycle_strategy")' not in src

    def test_no_hasattr_regime_detector(self) -> None:
        """hasattr(self, '_regime_detector') が排除されている."""
        from scripts.v460.lib import fill_cycle_executor as mod

        src = inspect.getsource(mod)
        assert 'hasattr(self, "_regime_detector")' not in src

    def test_no_hasattr_macro_regime_detector(self) -> None:
        """hasattr(self, '_macro_regime_detector') が排除されている."""
        from scripts.v460.lib import fill_cycle_executor as mod

        src = inspect.getsource(mod)
        assert 'hasattr(self, "_macro_regime_detector")' not in src

    def test_legitimate_hasattr_current_regime_value_remains(self) -> None:
        """hasattr(self, '_current_regime_value') は mixin 確認で正当."""
        from scripts.v460.lib import fill_cycle_executor as mod

        src = inspect.getsource(mod)
        # 正当な mixin method 存在確認として残っている
        assert 'hasattr(self, "_current_regime_value")' in src


# ======================================================================
# Config validation
# ======================================================================


class TestConfigValidation230:
    """230# 新規フィールドのバリデーション."""

    def test_ffd_l2_deadzone_bps_default(self) -> None:
        cfg = FillTestConfig()
        assert cfg.ffd_l2_deadzone_bps == 3.0

    def test_ffd_boost_release_streak_default(self) -> None:
        cfg = FillTestConfig()
        assert cfg.ffd_boost_release_streak == 3

    def test_ffd_l2_deadzone_bps_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="ffd_l2_deadzone_bps"):
            FillTestConfig(ffd_l2_deadzone_bps=-1.0)

    def test_ffd_l2_deadzone_bps_zero_ok(self) -> None:
        cfg = FillTestConfig(ffd_l2_deadzone_bps=0.0)
        assert cfg.ffd_l2_deadzone_bps == 0.0

    def test_ffd_boost_release_streak_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="ffd_boost_release_streak"):
            FillTestConfig(ffd_boost_release_streak=0)

    def test_ffd_boost_release_streak_one_ok(self) -> None:
        cfg = FillTestConfig(ffd_boost_release_streak=1)
        assert cfg.ffd_boost_release_streak == 1


# ======================================================================
# FastFillDefenseConfig defaults
# ======================================================================


class TestFFDConfigDefaults:
    """230# FFDConfig の新規デフォルト値."""

    def test_l2_deadzone_default(self) -> None:
        cfg = FastFillDefenseConfig()
        assert cfg.l2_deadzone_bps == 3.0

    def test_boost_release_streak_default(self) -> None:
        cfg = FastFillDefenseConfig()
        assert cfg.boost_release_streak == 3

    def test_side_state_normal_fill_streak_default(self) -> None:
        from scripts.v460.lib.fast_fill_defense import _SideState

        s = _SideState()
        assert s.normal_fill_streak == 0
