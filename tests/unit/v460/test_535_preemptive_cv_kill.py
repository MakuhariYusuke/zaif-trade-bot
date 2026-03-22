"""535# Pre-emptive CV kill + min_spread_jpy 引き下げテスト.

532# §4: sell_dynamic_kill は事後反応（損失発生後に kill 発動）→
CV adverse velocity 持続時に事前ブロックする仕組みを追加。
532# §5 P1-6: min_spread_jpy 700→500 (buy NFQ 94% 集中改善)。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.orchestrator_guards import OrchestratorGuardsMixin


def _make_cv_hint(
    *,
    adverse_side: str = "sell",
    velocity_bps: float = 3.0,
    confidence: float = 0.8,
    direction: str = "up",
    spread_bps: float = 5.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        adverse_side=adverse_side,
        reference_velocity_bps=velocity_bps,
        confidence=confidence,
        direction=direction,
        spread_bps=spread_bps,
        reference_exchange="bitflyer",
        age_sec=0.5,
    )


def _build_mixin(
    *,
    preemptive_enabled: bool = True,
    velocity_threshold: float = 2.0,
    confidence_floor: float = 0.5,
    consecutive_threshold: int = 3,
    cooldown_cycles: int = 5,
) -> OrchestratorGuardsMixin:
    mixin = OrchestratorGuardsMixin.__new__(OrchestratorGuardsMixin)
    mixin._sell_kill_mgr = MagicMock()
    mixin._buy_kill_mgr = MagicMock()
    cfg = FillTestConfig(
        sell_preemptive_cv_kill_enabled=preemptive_enabled,
        sell_preemptive_cv_velocity_threshold=velocity_threshold,
        sell_preemptive_cv_confidence_floor=confidence_floor,
        sell_preemptive_cv_consecutive_threshold=consecutive_threshold,
        sell_preemptive_cv_cooldown_cycles=cooldown_cycles,
    )
    mixin.config = cfg
    mixin._regime_detector = None
    mixin._maker_price = MagicMock()
    mixin._maker_price.inv_net_imbalance = 0.0
    mixin._maker_price.cross_venue_lead_lag_hint = None
    # kill release tracking
    mixin._kill_was_active_buy = False
    mixin._kill_was_active_sell = False
    mixin._kill_released_at_cycle_buy = None
    mixin._kill_released_at_cycle_sell = None
    mixin._cycle_count = 0
    mixin._guard_fire_counts = {}
    # pre-emptive CV kill state
    mixin._preemptive_cv_sell_adverse_count = 0
    mixin._preemptive_cv_sell_cooldown = 0
    return mixin


def _setup_standard_kill_not_active(mixin: OrchestratorGuardsMixin) -> None:
    """Standard dynamic kill returns False (not killed)."""
    for mgr in (mixin._sell_kill_mgr, mixin._buy_kill_mgr):
        mgr.check_kill.return_value = (
            False,
            SimpleNamespace(
                threshold_used=-0.5,
                cooldown_remaining=0,
                probe_fired=False,
                force_release_fired=False,
            ),
        )


class TestPreemptiveCvKillActivation:
    """535# CV adverse velocity 連続 → pre-emptive kill 発動."""

    def test_single_adverse_no_kill(self) -> None:
        """1 回の adverse signal では kill しない (consecutive < threshold)."""
        mixin = _build_mixin(consecutive_threshold=3)
        _setup_standard_kill_not_active(mixin)
        mixin._maker_price.cross_venue_lead_lag_hint = _make_cv_hint(
            velocity_bps=3.0, confidence=0.8,
        )
        result = mixin._is_side_killed("sell")
        assert result is False
        assert mixin._preemptive_cv_sell_adverse_count == 1

    def test_consecutive_adverse_triggers_kill(self) -> None:
        """連続 adverse signal が consecutive_threshold に到達すると kill."""
        mixin = _build_mixin(consecutive_threshold=3, cooldown_cycles=5)
        _setup_standard_kill_not_active(mixin)
        hint = _make_cv_hint(velocity_bps=3.0, confidence=0.8)
        mixin._maker_price.cross_venue_lead_lag_hint = hint

        # 1 回目
        assert mixin._is_side_killed("sell") is False
        # 2 回目
        assert mixin._is_side_killed("sell") is False
        # 3 回目 → kill 発動
        assert mixin._is_side_killed("sell") is True
        assert mixin._preemptive_cv_sell_cooldown == 5

    def test_cooldown_sustains_kill(self) -> None:
        """cooldown 中は hint がなくても kill 継続."""
        mixin = _build_mixin(consecutive_threshold=1, cooldown_cycles=2)
        _setup_standard_kill_not_active(mixin)
        hint = _make_cv_hint(velocity_bps=3.0, confidence=0.8)
        mixin._maker_price.cross_venue_lead_lag_hint = hint

        # 1 回目 → kill 発動 (cooldown=2)
        assert mixin._is_side_killed("sell") is True
        # hint 消失
        mixin._maker_price.cross_venue_lead_lag_hint = None
        # cooldown=2 → 1 → kill 継続
        assert mixin._is_side_killed("sell") is True
        # cooldown=1 → 0 → kill 継続
        assert mixin._is_side_killed("sell") is True
        # cooldown=0 → fall through → kill 解除
        assert mixin._is_side_killed("sell") is False

    def test_non_adverse_resets_counter(self) -> None:
        """adverse でない signal で consecutive カウンタリセット."""
        mixin = _build_mixin(consecutive_threshold=3)
        _setup_standard_kill_not_active(mixin)

        # 2 回 adverse
        mixin._maker_price.cross_venue_lead_lag_hint = _make_cv_hint(
            velocity_bps=3.0, confidence=0.8,
        )
        mixin._is_side_killed("sell")
        mixin._is_side_killed("sell")
        assert mixin._preemptive_cv_sell_adverse_count == 2

        # buy-adverse (sell にとっては non-adverse) → リセット
        mixin._maker_price.cross_venue_lead_lag_hint = _make_cv_hint(
            adverse_side="buy", velocity_bps=3.0, confidence=0.8,
        )
        mixin._is_side_killed("sell")
        assert mixin._preemptive_cv_sell_adverse_count == 0


class TestPreemptiveCvKillFilters:
    """535# velocity / confidence フィルターのテスト."""

    def test_low_velocity_no_count(self) -> None:
        """velocity < threshold では adverse カウントされない."""
        mixin = _build_mixin(velocity_threshold=2.0)
        _setup_standard_kill_not_active(mixin)
        mixin._maker_price.cross_venue_lead_lag_hint = _make_cv_hint(
            velocity_bps=1.5, confidence=0.8,
        )
        mixin._is_side_killed("sell")
        assert mixin._preemptive_cv_sell_adverse_count == 0

    def test_low_confidence_no_count(self) -> None:
        """confidence < floor では adverse カウントされない."""
        mixin = _build_mixin(confidence_floor=0.5)
        _setup_standard_kill_not_active(mixin)
        mixin._maker_price.cross_venue_lead_lag_hint = _make_cv_hint(
            velocity_bps=3.0, confidence=0.3,
        )
        mixin._is_side_killed("sell")
        assert mixin._preemptive_cv_sell_adverse_count == 0

    def test_disabled_no_kill(self) -> None:
        """preemptive_enabled=False なら kill しない."""
        mixin = _build_mixin(preemptive_enabled=False, consecutive_threshold=1)
        _setup_standard_kill_not_active(mixin)
        mixin._maker_price.cross_venue_lead_lag_hint = _make_cv_hint(
            velocity_bps=5.0, confidence=0.9,
        )
        # 何回呼んでも pre-emptive kill は発動しない
        for _ in range(5):
            result = mixin._is_side_killed("sell")
        assert result is False

    def test_buy_side_not_affected(self) -> None:
        """buy 側は pre-emptive CV kill の対象外."""
        mixin = _build_mixin(consecutive_threshold=1)
        _setup_standard_kill_not_active(mixin)
        mixin._maker_price.cross_venue_lead_lag_hint = _make_cv_hint(
            adverse_side="buy", velocity_bps=5.0, confidence=0.9,
        )
        result = mixin._is_side_killed("buy")
        # pre-emptive kill は sell のみなので buy には影響しない
        assert result is False


class TestPreemptiveCvKillGuardFire:
    """535# guard fire カウンタが正しく記録される."""

    def test_guard_fire_incremented(self) -> None:
        """pre-emptive kill 発動時に guard fire カウンタが加算される."""
        mixin = _build_mixin(consecutive_threshold=1, cooldown_cycles=1)
        _setup_standard_kill_not_active(mixin)
        mixin._maker_price.cross_venue_lead_lag_hint = _make_cv_hint(
            velocity_bps=3.0, confidence=0.8,
        )
        mixin._is_side_killed("sell")
        assert mixin._guard_fire_counts.get("preemptive_cv_sell_kill", 0) == 1


class TestKillReleaseTrackingWithPreemptive:
    """535# pre-emptive kill のリリース追跡が既存 343# 機構と連携する."""

    def test_preemptive_kill_release_tracked(self) -> None:
        """pre-emptive kill → 解除で _kill_released_at_cycle_sell が記録される."""
        mixin = _build_mixin(consecutive_threshold=1, cooldown_cycles=1)
        _setup_standard_kill_not_active(mixin)
        mixin._cycle_count = 10
        hint = _make_cv_hint(velocity_bps=3.0, confidence=0.8)
        mixin._maker_price.cross_venue_lead_lag_hint = hint

        # kill 発動 (cooldown=1)
        assert mixin._is_side_killed("sell") is True
        assert mixin._kill_was_active_sell is True

        # hint 消失
        mixin._maker_price.cross_venue_lead_lag_hint = None
        mixin._cycle_count = 11
        # cooldown=1→0 → kill 継続
        assert mixin._is_side_killed("sell") is True

        # cooldown=0 → fall through → released
        mixin._cycle_count = 12
        assert mixin._is_side_killed("sell") is False
        assert mixin._kill_released_at_cycle_sell == 12


class TestMinSpreadJpyConfig:
    """535# min_spread_jpy 700→500 の設定テスト."""

    def test_yaml_parser_min_spread(self) -> None:
        """fill_config_parser が min_spread_jpy を正しくパース."""
        from scripts.v460.lib.fill_config_parser import parse_fill_config_yaml

        yaml_cfg = {"min_spread_jpy": 500}
        config = parse_fill_config_yaml(yaml_cfg)
        assert config.min_spread_jpy == 500

    def test_default_value(self) -> None:
        """FillTestConfig のデフォルト値が 0 (フィルタなし) であること."""
        cfg = FillTestConfig()
        assert cfg.min_spread_jpy == 0.0


class TestPreemptiveCvKillConfig:
    """535# pre-emptive CV kill config のテスト."""

    def test_config_fields_exist(self) -> None:
        """FillTestConfig に pre-emptive CV kill フィールドが存在."""
        cfg = FillTestConfig()
        assert hasattr(cfg, "sell_preemptive_cv_kill_enabled")
        assert hasattr(cfg, "sell_preemptive_cv_velocity_threshold")
        assert hasattr(cfg, "sell_preemptive_cv_confidence_floor")
        assert hasattr(cfg, "sell_preemptive_cv_consecutive_threshold")
        assert hasattr(cfg, "sell_preemptive_cv_cooldown_cycles")

    def test_default_disabled(self) -> None:
        """デフォルトでは無効."""
        cfg = FillTestConfig()
        assert cfg.sell_preemptive_cv_kill_enabled is False

    def test_yaml_parser_preemptive_cv_kill(self) -> None:
        """fill_config_parser が pre-emptive CV kill パラメータをパース."""
        from scripts.v460.lib.fill_config_parser import parse_fill_config_yaml

        yaml_cfg = {
            "cross_venue_lead_lag": {
                "preemptive_sell_kill_enabled": True,
                "preemptive_sell_kill_velocity_threshold": 3.0,
                "preemptive_sell_kill_confidence_floor": 0.6,
                "preemptive_sell_kill_consecutive_threshold": 5,
                "preemptive_sell_kill_cooldown_cycles": 10,
            }
        }
        config = parse_fill_config_yaml(yaml_cfg)
        assert config.sell_preemptive_cv_kill_enabled is True
        assert config.sell_preemptive_cv_velocity_threshold == 3.0
        assert config.sell_preemptive_cv_confidence_floor == 0.6
        assert config.sell_preemptive_cv_consecutive_threshold == 5
        assert config.sell_preemptive_cv_cooldown_cycles == 10

    def test_hot_reload_fields_present(self) -> None:
        """config_hot_reload に pre-emptive CV kill フィールドが含まれる."""
        from scripts.v460.lib import config_hot_reload

        src = open(config_hot_reload.__file__).read()
        assert "sell_preemptive_cv_kill_enabled" in src
        assert "sell_preemptive_cv_velocity_threshold" in src
