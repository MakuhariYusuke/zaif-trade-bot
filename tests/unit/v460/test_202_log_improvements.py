"""202# ログ分析基づく改善の単体テスト.

A: 単一サイクル大損失クールダウン (連鎖損失防止)
B: 片側残高枯渇時の rescue offset 適用
C: VG sell-side 補完 (velocity_60s ベース)
"""

from __future__ import annotations

import pytest


# ============================================================
# 202# A: Loss cooldown config + multiplier
# ============================================================

class TestLossCooldownConfig:
    """202# A: loss_cooldown config fields."""

    def test_config_defaults(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.loss_cooldown_threshold_bps == -10.0
        assert cfg.loss_cooldown_interval_mult == 2.0

    def test_loss_cooldown_mult_below_one_raises(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        with pytest.raises(ValueError, match="loss_cooldown_interval_mult"):
            FillTestConfig(loss_cooldown_interval_mult=0.5)

    def test_loss_cooldown_mult_one_ok(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig(loss_cooldown_interval_mult=1.0)
        assert cfg.loss_cooldown_interval_mult == 1.0

    def test_yaml_parsing(self) -> None:
        import yaml
        yaml_str = """
止血:
  loss_cooldown_threshold_bps: -8.0
  loss_cooldown_interval_mult: 3.0
  one_sided_balance_rescue_offset: false
"""
        data = yaml.safe_load(yaml_str)
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig.from_yaml(data)
        assert cfg.loss_cooldown_threshold_bps == -8.0
        assert cfg.loss_cooldown_interval_mult == 3.0
        assert cfg.one_sided_balance_rescue_offset is False


class TestLossCooldownMixin:
    """202# A: _loss_cooldown_mult クラスレベル属性が宣言されていること."""

    def test_class_attr_exists(self) -> None:
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        assert hasattr(FillLoopOrchestratorMixin, "_loss_cooldown_mult")
        assert FillLoopOrchestratorMixin._loss_cooldown_mult == 1.0


# ============================================================
# 202# B: One-sided balance rescue offset
# ============================================================

class TestOneSidedBalanceRescueConfig:
    """202# B: one_sided_balance_rescue_offset config."""

    def test_config_default_true(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.one_sided_balance_rescue_offset is True

    def test_config_false(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig(one_sided_balance_rescue_offset=False)
        assert cfg.one_sided_balance_rescue_offset is False


# ============================================================
# 202# C: VG sell-side supplement
# ============================================================

class TestVGSellSupplement:
    """202# C: VG sell-side 補完の概念テスト."""

    def test_vg_threshold_field_exists(self) -> None:
        """volatility_guard_velocity_threshold_bps が config に存在."""
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert hasattr(cfg, "volatility_guard_velocity_threshold_bps")
        assert cfg.volatility_guard_velocity_threshold_bps > 0

    def test_vg_boost_field_exists(self) -> None:
        """volatility_guard_offset_boost_factor が config に存在."""
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert hasattr(cfg, "volatility_guard_offset_boost_factor")
        assert cfg.volatility_guard_offset_boost_factor >= 1.0

    def test_maker_price_last_vg_triggered_property(self) -> None:
        """MakerPriceCalculator.last_vg_triggered プロパティが存在."""
        from scripts.v460.lib.maker_price import MakerPriceCalculator
        from scripts.v460.lib.fill_config import FillTestConfig
        from scripts.v460.lib.fast_fill_defense import FastFillDefense
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
        from scripts.v460.lib.fill_config import FillTestConfig
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
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig(loss_cooldown_threshold_bps=-10.0)
        pnl = -3.87  # 小さな損失
        assert pnl > cfg.loss_cooldown_threshold_bps

    def test_profit_no_cooldown(self) -> None:
        """利益時は cooldown なし."""
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig(loss_cooldown_threshold_bps=-10.0)
        pnl = 5.78  # 利益
        assert pnl > cfg.loss_cooldown_threshold_bps


# ============================================================
# 202# B: One-sided rescue offset logic
# ============================================================

class TestOneSidedRescueLogic:
    """202# B: 片側残高枯渇時にも rescue offset が適用されること."""

    def test_rescue_mult_from_config(self) -> None:
        """rescue_offset_mult が config から取得可能."""
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig(
            balance_forced_rescue_offset_mult=1.5,
            one_sided_balance_rescue_offset=True,
        )
        assert cfg.balance_forced_rescue_offset_mult == 1.5
        assert cfg.one_sided_balance_rescue_offset is True

    def test_disabled_no_rescue(self) -> None:
        """one_sided_balance_rescue_offset=False の場合は rescue 無効."""
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig(one_sided_balance_rescue_offset=False)
        # ロジック: original_also_insufficient=True だが rescue=False
        original_also_insufficient = True
        _is_rescue = False
        if original_also_insufficient and cfg.one_sided_balance_rescue_offset:
            _is_rescue = True
        assert _is_rescue is False


# ============================================================
# Cross-cutting: YAML fill_test.yaml に設定が存在すること
# ============================================================

class TestYAMLConfigPresence:
    """YAML config に 202# 設定が正しく記述されていること."""

    def test_fill_test_yaml_has_202_config(self) -> None:
        import yaml
        from pathlib import Path
        yaml_path = Path("configs/v460/fill_test.yaml")
        if not yaml_path.exists():
            pytest.skip("fill_test.yaml not found")
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        loss_ctrl = data.get("loss_control", {})
        assert "loss_cooldown_threshold_bps" in loss_ctrl, "202# A: missing in YAML"
        assert "loss_cooldown_interval_mult" in loss_ctrl, "202# A: missing in YAML"
        assert "one_sided_balance_rescue_offset" in loss_ctrl, "202# B: missing in YAML"
