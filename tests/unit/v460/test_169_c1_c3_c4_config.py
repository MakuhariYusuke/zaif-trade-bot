"""169# C1/C3/C4 設定テスト — time_filter 全廃 + trending_up_sell 閾値 + DailyDrawdownGuard.

C1: time_filter 全廃 — 条件ベースフィルタ (B1', SkipGate, VG, sell_dynamic_kill) が根本対策
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
# C1: time_filter 全廃 — 条件ベースフィルタが根本対策
# ================================================================


class TestC1TimeFilterFullAbolition:
    """169# time_filter 全廃: 全ての静的時間帯遮断を撤廃.

    根拠 (107# Phase 3 Step 3 完了):
    - 全ての時間帯遮断は「市場状態の時間帯相関」を因果と混同した弥縫策
    - 条件ベースフィルタが全ての根本原因を直接処理:
      * B1': ranging_buy at low_vol → hard skip (損失源の 69%)
      * SkipGate: ML + hour_sin/cos + regime → AS 確率予測
      * VG: velocity/VPIN → offset_boost (リアルタイム変動)
      * sell_dynamic_kill: rolling PnL → sell 停止
      * trending_sell_skip: trending_up → sell skip
      * DailyDrawdownGuard: 日次 cumPnL → hard/soft limit
    - 原則: 条件ベースフィルタ > 時間ベースフィルタ
    """

    def test_buy_skip_empty(self, config_from_yaml: FillConfig) -> None:
        """buy スキップリストが空 — 全時間帯開放."""
        assert config_from_yaml.skip_utc_hours_buy == []

    def test_sell_skip_empty(self, config_from_yaml: FillConfig) -> None:
        """sell スキップリストが空 — 全時間帯開放."""
        assert config_from_yaml.skip_utc_hours_sell == []

    def test_global_skip_empty(self, config_from_yaml: FillConfig) -> None:
        """グローバルスキップリストが空."""
        assert config_from_yaml.skip_utc_hours == []

    def test_regime_adaptive_buy_empty(self, config_from_yaml: FillConfig) -> None:
        """regime_adaptive_extra_buy が空 — VG が high_vol を直接処理."""
        assert config_from_yaml.regime_adaptive_extra_buy == []

    def test_regime_adaptive_sell_empty(self, config_from_yaml: FillConfig) -> None:
        """regime_adaptive_extra_sell が空 — sell_dynamic_kill + VG が担当."""
        assert config_from_yaml.regime_adaptive_extra_sell == []

    def test_b1_prime_handles_root_cause(self, config_from_yaml: FillConfig) -> None:
        """B1' (ranging_buy low_vol skip) が有効 — 時間帯遮断は不要."""
        assert config_from_yaml.skip_ranging_buy_low_vol is True

    def test_time_filter_enabled_for_framework(self, config_from_yaml: FillConfig) -> None:
        """time_filter 機構自体は維持 (コードパス健全性, 即時復帰可能)."""
        assert config_from_yaml.enable_time_filter is True

    def test_regime_adaptive_mechanism_preserved(self, config_from_yaml: FillConfig) -> None:
        """regime_adaptive 機構自体は維持 (将来の再有効化に備える)."""
        assert config_from_yaml.regime_adaptive_enabled is True


# ================================================================
# C3: sell_dynamic_kill trending_up threshold + safety valve
# ================================================================


class TestC3TrendingUpSellThreshold:
    """C3: trending_up 閾値強化 + 安全弁調整.

    171# Guard Paradox 対策で閾値を緩和 (-0.1→-0.3), 安全弁を早期化 (20→10).
    """

    def test_regime_threshold_trending_up(self, config_from_yaml: FillConfig) -> None:
        """trending_up の sell_dynamic_kill 閾値が -0.3 (171# Guard Paradox 対策)."""
        thresholds = config_from_yaml.sell_dynamic_kill_regime_thresholds
        assert thresholds is not None
        assert thresholds.get("trending_up") == pytest.approx(-0.3)

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
        """171# Guard Paradox: 安全弁が 10 に短縮 (sell 機会確保)."""
        assert config_from_yaml.max_consecutive_trending_sell_skip == 10

    def test_sell_guard_inv_bypass_threshold(
        self, config_from_yaml: FillConfig
    ) -> None:
        """171# Guard Paradox: 在庫偏重バイパス閾値が 0.3."""
        assert config_from_yaml.sell_guard_inv_bypass_threshold == pytest.approx(0.3)


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
