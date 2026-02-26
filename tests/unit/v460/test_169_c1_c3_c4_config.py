"""169# C1/C3/C4 設定テスト — time_filter + trending_up_sell 閾値 + DailyDrawdownGuard.

C1: JST23 (UTC14) / JST02 (UTC17) の buy スキップ追加
C3: sell_dynamic_kill trending_up 閾値 -0.3→-0.1 + 安全弁 10→20
C4: DailyDrawdownGuard enabled: true

YAML パース経由で設定値が正しく反映されることを検証。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.v460.lib.fill_config import FillTestConfig as FillConfig

YAML_PATH = Path("configs/v460/fill_test.yaml")


@pytest.fixture()
def config_from_yaml() -> FillConfig:
    """fill_test.yaml から dict ロード → FillConfig 構築."""
    assert YAML_PATH.exists(), f"YAML not found: {YAML_PATH}"
    with open(YAML_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return FillConfig.from_yaml(raw)


# ================================================================
# C1: time_filter JST23/JST02 buy skip
# ================================================================


class TestC1TimeFilterBuySkip:
    """C1: skip_utc_hours_buy に UTC14/17 が追加されたことを検証."""

    def test_utc14_in_buy_skip(self, config_from_yaml: FillConfig) -> None:
        """UTC14 (JST23) が buy スキップリストに含まれる."""
        assert 14 in config_from_yaml.skip_utc_hours_buy

    def test_utc17_in_buy_skip(self, config_from_yaml: FillConfig) -> None:
        """UTC17 (JST02) が buy スキップリストに含まれる."""
        assert 17 in config_from_yaml.skip_utc_hours_buy

    def test_utc16_still_in_buy_skip(self, config_from_yaml: FillConfig) -> None:
        """既存の UTC16 (JST01) が維持されている."""
        assert 16 in config_from_yaml.skip_utc_hours_buy

    def test_buy_skip_count(self, config_from_yaml: FillConfig) -> None:
        """buy スキップは 3 時間帯."""
        assert len(config_from_yaml.skip_utc_hours_buy) == 3

    def test_sell_skip_unchanged(self, config_from_yaml: FillConfig) -> None:
        """sell スキップリストは変更なし [8, 21]."""
        assert sorted(config_from_yaml.skip_utc_hours_sell) == [8, 21]


# ================================================================
# C3: sell_dynamic_kill trending_up threshold + safety valve
# ================================================================


class TestC3TrendingUpSellThreshold:
    """C3: trending_up 閾値強化 + 安全弁拡大."""

    def test_regime_threshold_trending_up(self, config_from_yaml: FillConfig) -> None:
        """trending_up の sell_dynamic_kill 閾値が -0.1 に強化されている."""
        thresholds = config_from_yaml.sell_dynamic_kill_regime_thresholds
        assert thresholds is not None
        assert thresholds.get("trending_up") == pytest.approx(-0.1)

    def test_regime_threshold_trending_down_unchanged(
        self, config_from_yaml: FillConfig
    ) -> None:
        """trending_down の閾値は -1.0 のまま."""
        thresholds = config_from_yaml.sell_dynamic_kill_regime_thresholds
        assert thresholds is not None
        assert thresholds.get("trending_down") == pytest.approx(-1.0)

    def test_regime_threshold_ranging_unchanged(
        self, config_from_yaml: FillConfig
    ) -> None:
        """ranging の閾値は -0.5 のまま."""
        thresholds = config_from_yaml.sell_dynamic_kill_regime_thresholds
        assert thresholds is not None
        assert thresholds.get("ranging") == pytest.approx(-0.5)

    def test_max_consecutive_trending_sell_skip(
        self, config_from_yaml: FillConfig
    ) -> None:
        """安全弁が 20 に拡大されている."""
        assert config_from_yaml.max_consecutive_trending_sell_skip == 20


# ================================================================
# C4: DailyDrawdownGuard
# ================================================================


class TestC4DailyDrawdownGuard:
    """C4: DailyDrawdownGuard が有効化されていることを検証."""

    def test_enabled(self, config_from_yaml: FillConfig) -> None:
        """daily_drawdown_enabled が True."""
        assert config_from_yaml.daily_drawdown_enabled is True

    def test_hard_limit(self, config_from_yaml: FillConfig) -> None:
        """hard limit が -50.0bps."""
        assert config_from_yaml.daily_drawdown_hard_limit_bps == pytest.approx(-50.0)

    def test_soft_limit(self, config_from_yaml: FillConfig) -> None:
        """soft limit が -30.0bps."""
        assert config_from_yaml.daily_drawdown_soft_limit_bps == pytest.approx(-30.0)

    def test_guard_instantiation(self, config_from_yaml: FillConfig) -> None:
        """DailyDrawdownGuard がインスタンス化できること."""
        from scripts.v460.lib.daily_drawdown_guard import DailyDrawdownGuard

        guard = DailyDrawdownGuard(
            enabled=config_from_yaml.daily_drawdown_enabled,
            hard_limit_bps=config_from_yaml.daily_drawdown_hard_limit_bps,
            soft_limit_bps=config_from_yaml.daily_drawdown_soft_limit_bps,
        )
        assert guard.enabled is True

    def test_guard_halts_on_hard_limit(self) -> None:
        """hard limit 超過で halted=True."""
        from scripts.v460.lib.daily_drawdown_guard import DailyDrawdownGuard

        guard = DailyDrawdownGuard(
            enabled=True,
            hard_limit_bps=-50.0,
            soft_limit_bps=-30.0,
        )
        # 大きな損失を一括投入
        result = guard.update_pnl(-60.0)
        assert result["halted"] is True

    def test_guard_soft_trigger(self) -> None:
        """soft limit 超過で soft_triggered=True."""
        from scripts.v460.lib.daily_drawdown_guard import DailyDrawdownGuard

        guard = DailyDrawdownGuard(
            enabled=True,
            hard_limit_bps=-50.0,
            soft_limit_bps=-30.0,
        )
        result = guard.update_pnl(-35.0)
        assert result["soft_triggered"] is True
        assert result["halted"] is False
