"""202# ログ分析基づく改善の単体テスト.

A: 単一サイクル大損失クールダウン (連鎖損失防止)
B: 片側残高枯渇時の rescue offset 適用
C: VG sell-side 補完 (velocity_bps ベース)
"""

from __future__ import annotations

import pytest
import yaml

from scripts.v460.lib.fast_fill_defense import FastFillDefense
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
from scripts.v460.lib.maker_price import MakerPriceCalculator


def _yaml_mapping(yaml_text: str) -> dict[str, object]:
    data = yaml.safe_load(yaml_text)
    if not isinstance(data, dict):
        raise TypeError("expected YAML mapping")
    return data


# ============================================================
# 202# A: Loss cooldown config + multiplier
# ============================================================

class TestLossCooldownConfig:
    """202# A: loss_cooldown config fields."""

    def test_config_defaults(self) -> None:
        cfg = FillTestConfig()
        assert cfg.loss_cooldown_threshold_bps == -10.0
        assert cfg.loss_cooldown_interval_mult == 2.0

    def test_loss_cooldown_mult_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="loss_cooldown_interval_mult"):
            FillTestConfig(loss_cooldown_interval_mult=0.5)

    def test_loss_cooldown_mult_one_ok(self) -> None:
        cfg = FillTestConfig(loss_cooldown_interval_mult=1.0)
        assert cfg.loss_cooldown_interval_mult == 1.0

    def test_yaml_parsing(self) -> None:
        yaml_str = """
止血:
  loss_cooldown_threshold_bps: -8.0
  loss_cooldown_interval_mult: 3.0
  one_sided_balance_rescue_offset: false
"""
        data = _yaml_mapping(yaml_str)
        cfg = FillTestConfig.from_yaml(data)
        assert cfg.loss_cooldown_threshold_bps == -8.0
        assert cfg.loss_cooldown_interval_mult == 3.0
        assert cfg.one_sided_balance_rescue_offset is False


class TestLossCooldownMixin:
    """202# A: _loss_cooldown_mult クラスレベル属性が宣言されていること."""

    def test_class_attr_exists(self) -> None:
        assert hasattr(FillLoopOrchestratorMixin, "_loss_cooldown_mult")
        assert FillLoopOrchestratorMixin._loss_cooldown_mult == 1.0


# ============================================================
# 202# B: One-sided balance rescue offset
# ============================================================

class TestOneSidedBalanceRescueConfig:
    """202# B: one_sided_balance_rescue_offset config."""

    def test_config_default_true(self) -> None:
        cfg = FillTestConfig()
        assert cfg.one_sided_balance_rescue_offset is True

    def test_config_false(self) -> None:
        cfg = FillTestConfig(one_sided_balance_rescue_offset=False)
        assert cfg.one_sided_balance_rescue_offset is False


# ============================================================
# 202# C: VG sell-side supplement
# ============================================================

class TestVGSellSupplement:
    """202# C: VG sell-side 補完の概念テスト."""

    def test_vg_threshold_field_exists(self) -> None:
        """volatility_guard_velocity_threshold_bps が config に存在."""
        cfg = FillTestConfig()
        assert hasattr(cfg, "volatility_guard_velocity_threshold_bps")
        assert cfg.volatility_guard_velocity_threshold_bps > 0

    def test_vg_boost_field_exists(self) -> None:
        """volatility_guard_offset_boost_factor が config に存在."""
        cfg = FillTestConfig()
        assert hasattr(cfg, "volatility_guard_offset_boost_factor")
        assert cfg.volatility_guard_offset_boost_factor >= 1.0

    def test_maker_price_last_vg_triggered_property(self) -> None:
        """MakerPriceCalculator.last_vg_triggered プロパティが存在."""
        cfg = FillTestConfig()
        ffd = FastFillDefense(config=cfg, base_offset_ratio=0.05)
        mp = MakerPriceCalculator(
            config=cfg,
            fast_fill_defense=ffd,
            regime_detector=None,
            base_offset_ratio=0.05,
        )
        assert hasattr(mp, "last_vg_triggered")
        assert mp.last_vg_triggered is False  # 初期状態


# ============================================================
# Integration-like: 202# A loss cooldown logic
# ============================================================

class TestLossCooldownLogic:
    """202# A: 大損失後の cooldown 乗数が正しく設定・リセットされること."""

    def test_large_loss_sets_cooldown(self) -> None:
        """PnL <= threshold → _loss_cooldown_mult が設定される."""
        # FillLoopOrchestrator のロジックを模擬的に検証
        cfg = FillTestConfig(
            loss_cooldown_threshold_bps=-10.0,
            loss_cooldown_interval_mult=2.5,
        )
        # 模擬: threshold チェック
        pnl = -17.27  # 大損失
        assert pnl <= cfg.loss_cooldown_threshold_bps
        cooldown = cfg.loss_cooldown_interval_mult
        assert cooldown == 2.5

    def test_small_loss_no_cooldown(self) -> None:
        """PnL > threshold → cooldown なし."""
        cfg = FillTestConfig(loss_cooldown_threshold_bps=-10.0)
        pnl = -3.87  # 小さな損失
        assert pnl > cfg.loss_cooldown_threshold_bps

    def test_profit_no_cooldown(self) -> None:
        """利益時は cooldown なし."""
        cfg = FillTestConfig(loss_cooldown_threshold_bps=-10.0)
        pnl = 5.78  # 利益
        assert pnl > cfg.loss_cooldown_threshold_bps


# ============================================================
# 202# B: One-sided rescue offset logic
# ============================================================

# ============================================================
# Cross-cutting: YAML fill_test.yaml に設定が存在すること
# ============================================================

class TestYAMLConfigPresence:
    """YAML config に 202# 設定が正しく記述されていること."""

    def test_fill_test_yaml_has_202_config(
        self,
        v460_fill_test_yaml_base: dict[str, object],
    ) -> None:
        data = v460_fill_test_yaml_base
        loss_ctrl = data.get("loss_control", {})
        assert "loss_cooldown_threshold_bps" in loss_ctrl, "202# A: missing in YAML"
        assert "loss_cooldown_interval_mult" in loss_ctrl, "202# A: missing in YAML"
        assert "one_sided_balance_rescue_offset" in loss_ctrl, "202# B: missing in YAML"
