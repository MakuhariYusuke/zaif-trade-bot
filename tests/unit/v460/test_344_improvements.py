"""344# 改善テスト.

対象:
  A: velocity_ema_alpha 1.0→0.3 有効化
  B: ranging_obi_asymmetry_factor 0.0→0.3 有効化
  C: inv_decay_tau_sec 0→1800 有効化
  D: 342#B inv_bypass ステップ関数→gradual 化 (inv_relaxation max_bps 拡大)
  E: 342#D EWMA モード (DynamicKillManager)
"""

from __future__ import annotations

import pytest

from ztb.risk.sell_dynamic_kill import DynamicKillConfig, DynamicKillManager


# ============================================================
# A: velocity_ema_alpha コードデフォルト同期
# ============================================================

class TestVelocityEmaAlphaDefault:
    """velocity_ema_alpha のコードデフォルトが 0.3 であること."""

    def test_code_default_is_0_3(self):
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.velocity_ema_alpha == 0.3


# ============================================================
# B: ranging_obi_asymmetry_factor コードデフォルト同期
# ============================================================

class TestRangingObiAsymmetryDefault:
    """ranging_obi_asymmetry_factor のコードデフォルトが 0.3 であること."""

    def test_code_default_is_0_3(self):
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.ranging_obi_asymmetry_factor == 0.3


# ============================================================
# C: inv_decay_tau_sec コードデフォルト同期
# ============================================================

class TestInvDecayTauDefault:
    """inv_decay_tau_sec のコードデフォルトが 1800.0 であること."""

    def test_code_default_is_1800(self):
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.inv_decay_tau_sec == 1800.0


# ============================================================
# D: 342#B inv_bypass → gradual (inv_relaxation max_bps 拡大)
# ============================================================

class TestInvBypassGradual:
    """inv_bypass ステップ関数廃止 + inv_relaxation max_bps 拡大."""

    def test_inv_bypass_threshold_default_is_zero(self):
        """sell_guard_inv_bypass_threshold がデフォルト 0.0 (無効) であること."""
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.sell_guard_inv_bypass_threshold == 0.0

    def test_sell_inv_relaxation_max_bps_default(self):
        """sell inv_relaxation max_bps が 0.5 (拡大済) であること."""
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.sell_dynamic_kill_inv_relaxation_max_bps == 0.5

    def test_gate_no_bypass_when_threshold_zero(self):
        """inv_bypass_threshold=0 時は bypass が発動しないこと."""
        from scripts.v460.lib.fill_config import FillTestConfig
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator

        cfg = FillTestConfig(
            sell_guard_inv_bypass_threshold=0.0,
            skip_sell_trending=True,
            skip_sell_trending_up_only=False,
            trending_sell_as_offset_enabled=False,  # hard skip mode
        )
        gate = CycleGateAggregator(cfg)
        result = gate._check_trending_sell(
            side="sell",
            regime="trending_up",
            inv_net_imbalance=0.5,  # 高い在庫偏重
        )
        # bypass=0 なので blocked (offset mode off)
        assert result.blocked is True

    def test_gate_bypass_still_works_if_threshold_positive(self):
        """inv_bypass_threshold > 0 なら従来通り bypass が発動."""
        from scripts.v460.lib.fill_config import FillTestConfig
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator

        cfg = FillTestConfig(
            sell_guard_inv_bypass_threshold=0.3,
            skip_sell_trending=True,
            skip_sell_trending_up_only=False,
            trending_sell_as_offset_enabled=False,
        )
        gate = CycleGateAggregator(cfg)
        result = gate._check_trending_sell(
            side="sell",
            regime="trending_up",
            inv_net_imbalance=0.5,
        )
        assert result.blocked is False

    def test_sell_dynamic_kill_no_bypass_when_threshold_zero(self):
        """inv_bypass_threshold=0 → sell_dynamic_kill は inv_bypass しない."""
        from scripts.v460.lib.fill_config import FillTestConfig
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator

        cfg = FillTestConfig(
            sell_guard_inv_bypass_threshold=0.0,
            sell_dynamic_kill_enabled=True,
        )
        gate = CycleGateAggregator(cfg)
        result = gate._check_sell_dynamic_kill(
            side="sell",
            is_sell_killed=True,
            inv_net_imbalance=0.5,  # 高い在庫偏重
        )
        # bypass 無効 → killed のまま
        assert result.blocked is True


# ============================================================
# E: 342#D EWMA モード (DynamicKillManager)
# ============================================================

class TestEwmaMode:
    """DynamicKillManager の EWMA モード."""

    def test_ewma_alpha_default_is_zero(self):
        """DynamicKillConfig のデフォルト ewma_alpha は 0.0 (無効)."""
        cfg = DynamicKillConfig()
        assert cfg.ewma_alpha == 0.0

    def test_ewma_alpha_validation_negative(self):
        """ewma_alpha < 0 は ValueError."""
        with pytest.raises(ValueError, match="ewma_alpha"):
            DynamicKillConfig(ewma_alpha=-0.1)

    def test_ewma_alpha_validation_over_1(self):
        """ewma_alpha > 1 は ValueError."""
        with pytest.raises(ValueError, match="ewma_alpha"):
            DynamicKillConfig(ewma_alpha=1.5)

    def test_ewma_alpha_zero_uses_count_based(self):
        """ewma_alpha=0 → 従来の count-based rolling mean."""
        cfg = DynamicKillConfig(ewma_alpha=0.0, window=3, threshold_bps=-1.0)
        mgr = DynamicKillManager(cfg)
        mgr.track(0.5)
        mgr.track(0.5)
        # window=3 未達 → rolling mean = None
        assert mgr._get_rolling_mean() is None
        mgr.track(0.5)
        # window=3 達成 → mean = 0.5
        assert mgr._get_rolling_mean() == pytest.approx(0.5)

    def test_ewma_alpha_positive_uses_ewma(self):
        """ewma_alpha > 0 → EWMA を使用 (window サイズ制約なし)."""
        cfg = DynamicKillConfig(ewma_alpha=0.5, window=100, threshold_bps=-1.0)
        mgr = DynamicKillManager(cfg)
        # 1 fill でも EWMA は値を持つ
        mgr.track(1.0)
        assert mgr._get_rolling_mean() == pytest.approx(1.0)
        # 2 fill 目: EWMA = 0.5 * 2.0 + 0.5 * 1.0 = 1.5
        mgr.track(2.0)
        assert mgr._get_rolling_mean() == pytest.approx(1.5)

    def test_ewma_seed_is_first_value(self):
        """EWMA の初回は seed (最初の PnL 値)."""
        cfg = DynamicKillConfig(ewma_alpha=0.3, window=50, threshold_bps=-1.0)
        mgr = DynamicKillManager(cfg)
        mgr.track(-0.5)
        assert mgr._ewma_value == pytest.approx(-0.5)

    def test_ewma_kill_detection(self):
        """EWMA モードで kill 発動が正常に動作すること."""
        cfg = DynamicKillConfig(
            ewma_alpha=0.5, window=50, threshold_bps=-1.0, resume_window=1,
        )
        mgr = DynamicKillManager(cfg)
        # 悪い PnL を連続投入
        for _ in range(5):
            mgr.track(-2.0)
        killed, telem = mgr.check_kill()
        assert killed is True
        assert telem.rolling_mean is not None
        assert telem.rolling_mean < -1.0

    def test_ewma_no_kill_with_good_data(self):
        """EWMA モードでデータが良好なら kill されないこと."""
        cfg = DynamicKillConfig(
            ewma_alpha=0.5, window=50, threshold_bps=-1.0,
        )
        mgr = DynamicKillManager(cfg)
        for _ in range(10):
            mgr.track(0.5)
        killed, telem = mgr.check_kill()
        assert killed is False

    def test_ewma_reacts_faster_to_regime_change(self):
        """EWMA は新しいデータにより速く反応する (count-based との比較)."""
        # EWMA: α=0.5
        cfg_ewma = DynamicKillConfig(
            ewma_alpha=0.5, window=5, threshold_bps=-1.0,
        )
        mgr_ewma = DynamicKillManager(cfg_ewma)

        # Count-based: window=5
        cfg_count = DynamicKillConfig(
            ewma_alpha=0.0, window=5, threshold_bps=-1.0,
        )
        mgr_count = DynamicKillManager(cfg_count)

        # 5 個の良い PnL
        for _ in range(5):
            mgr_ewma.track(1.0)
            mgr_count.track(1.0)

        # 急激に悪化 (3 個の悪い PnL)
        for _ in range(3):
            mgr_ewma.track(-3.0)
            mgr_count.track(-3.0)

        ewma_mean = mgr_ewma._get_rolling_mean()
        count_mean = mgr_count._get_rolling_mean()

        # EWMA は新しいデータに重みをかけるので、count-based よりもネガティブ
        assert ewma_mean is not None
        assert count_mean is not None
        assert ewma_mean < count_mean

    def test_ewma_is_kill_active_uses_ewma(self):
        """is_kill_active() もEWMA 値を使用すること."""
        cfg = DynamicKillConfig(
            ewma_alpha=0.5, window=50, threshold_bps=-1.0,
        )
        mgr = DynamicKillManager(cfg)
        for _ in range(3):
            mgr.track(-2.0)
        is_active, rolling_mean, count = mgr.is_kill_active()
        assert is_active is True
        assert rolling_mean is not None
        assert rolling_mean < -1.0

    def test_ewma_assess_toxicity_uses_ewma(self):
        """assess_toxicity() も EWMA 値を使用すること."""
        cfg = DynamicKillConfig(
            ewma_alpha=0.5,
            window=50,
            threshold_bps=-1.0,
            toxicity_budget_enabled=True,
            toxicity_warn_level=0.3,
            toxicity_caution_level=0.7,
        )
        mgr = DynamicKillManager(cfg)
        for _ in range(3):
            mgr.track(-0.5)
        assessment = mgr.assess_toxicity()
        # score = rolling_mean / threshold = -0.5 / -1.0 = 0.5
        # EWMA(-0.5, -0.5, -0.5) = -0.5 → score ≈ 0.5
        assert assessment.rolling_mean is not None
        assert assessment.score > 0

    def test_fill_config_ewma_alpha_defaults(self):
        """FillTestConfig の EWMA alpha コードデフォルトが 0.05."""
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.sell_dynamic_kill_ewma_alpha == 0.05
        assert cfg.buy_dynamic_kill_ewma_alpha == 0.05


# ============================================================
# YAML パース統合テスト
# ============================================================

class TestYamlParsing:
    """YAML から全新規パラメータが正しくパースされること."""

    def test_yaml_parses_all_new_params(self):
        from scripts.v460.lib.fill_config_parser import parse_fill_config_yaml
        import os
        import yaml

        yaml_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "configs", "v460", "fill_test.yaml",
        )
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        cfg = parse_fill_config_yaml(raw)
        # A/B/C: パラメータ有効化
        assert cfg.velocity_ema_alpha == 0.3
        assert cfg.ranging_obi_asymmetry_factor == 0.3
        assert cfg.inv_decay_tau_sec == 1800.0
        # D: inv_bypass → gradual
        assert cfg.sell_guard_inv_bypass_threshold == 0.0
        assert cfg.sell_dynamic_kill_inv_relaxation_max_bps == 0.5
        # E: EWMA α
        assert cfg.sell_dynamic_kill_ewma_alpha == 0.05
        assert cfg.buy_dynamic_kill_ewma_alpha == 0.05
