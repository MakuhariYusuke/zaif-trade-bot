"""385# 設定監査: 閾値整合性テスト.

P0-1: continuous_to_discrete_threshold が訓練とライブで乖離するリスクを検証。
P0-2: reward_scaling デッドコードの検証 (386# 修正済み。修正後の正常動作を確認)。
P0-5: reward_settings YAML→env 伝播バグ検証 (386# 修正)。
"""

from __future__ import annotations

import pytest


class TestThresholdConsistency:
    """continuous_to_discrete_threshold の整合性チェック."""

    def test_yaml_threshold_value(self) -> None:
        """g2_sac_train.yaml の threshold が 0.10 であること."""
        from scripts.v460.lib.config_loader import load_config

        cfg = load_config("configs/v460/experiments/g2_sac_train.yaml")
        env_cfg = cfg.get("environment", {})
        threshold = float(env_cfg.get("continuous_to_discrete_threshold", 0.0))
        assert threshold == pytest.approx(0.10), (
            f"Expected threshold=0.10, got {threshold}"
        )

    def test_yaml_neg_threshold_value(self) -> None:
        """negative threshold も 0.10 であること."""
        from scripts.v460.lib.config_loader import load_config

        cfg = load_config("configs/v460/experiments/g2_sac_train.yaml")
        env_cfg = cfg.get("environment", {})
        neg = float(env_cfg.get("continuous_to_discrete_threshold_neg", 0.0))
        assert neg == pytest.approx(-0.10)

    def test_live_default_threshold_documented(self) -> None:
        """386# FIX: ライブ・訓練・定数の閾値が統一されていること."""
        from ztb.trading.constants import SAC_CONTINUOUS_THRESHOLD

        # SAC_CONTINUOUS_THRESHOLD が 0.10 であること (386# 修正)
        assert SAC_CONTINUOUS_THRESHOLD == pytest.approx(0.10), (
            f"SAC_CONTINUOUS_THRESHOLD should be 0.10, got {SAC_CONTINUOUS_THRESHOLD}"
        )

    def test_training_threshold_range(self) -> None:
        """threshold が SAC の tanh 出力範囲 [-1, 1] 内で妥当であること."""
        threshold = 0.10
        # HOLD zone ratio
        hold_ratio = (threshold * 2) / 2.0  # [-t, t] / [-1, 1]
        assert 0.05 <= hold_ratio <= 0.50, (
            f"HOLD zone ratio {hold_ratio:.2f} is outside reasonable range [0.05, 0.50]"
        )


class TestRewardScalingFixed:
    """reward_scaling が _calculate_default_reward に正しく流れることの検証 (386# 修正)."""

    def test_default_reward_accepts_scaling(self) -> None:
        """_calculate_default_reward が reward_scaling を受け取ること (386# 修正)."""
        import inspect
        from ztb.trading.environment.components.calculators.reward_calculator import (
            RewardCalculator,
        )

        sig = inspect.signature(RewardCalculator._calculate_default_reward)
        params = set(sig.parameters.keys())
        assert "reward_scaling" in params, (
            "386# FIX: reward_scaling が _calculate_default_reward に存在すること"
        )

    def test_pnl_focused_accepts_scaling(self) -> None:
        """_calculate_pnl_focused_reward は reward_scaling を受け取ること."""
        import inspect
        from ztb.trading.environment.components.calculators.reward_calculator import (
            RewardCalculator,
        )

        sig = inspect.signature(RewardCalculator._calculate_pnl_focused_reward)
        params = set(sig.parameters.keys())
        assert "reward_scaling" in params, (
            "pnl_focused は reward_scaling を使用すべき"
        )

    def test_reward_scaling_default_is_sac_value(self) -> None:
        """EnvironmentConfig の reward_scaling デフォルトが 1.0 (386# 修正後)."""
        from ztb.trading.environment.utils.config import EnvironmentConfig

        config = EnvironmentConfig()
        # 386# FIX: PPO 値 6.0 → SAC 値 1.0
        assert config.reward_scaling == pytest.approx(1.0), (
            f"Expected SAC default 1.0, got {config.reward_scaling}"
        )

    def test_pnl_reward_with_unit_scaling(self) -> None:
        """_calculate_pnl_reward(pnl, 1.0) は生の PnL × 1.0 を返す."""
        from ztb.trading.environment.components.calculators.reward_calculator import (
            RewardCalculator,
        )
        from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings

        config = EnvironmentConfig()
        settings = RewardSettings()
        calc = RewardCalculator(config, settings, initial_portfolio_value=10_000_000.0)
        result = calc._calculate_pnl_reward(100.0, 1.0)
        # pnl × reward_scaling × pnl_reward_multiplier(=1.0)
        assert result == pytest.approx(100.0), f"Expected 100.0, got {result}"

    def test_reward_scaling_flows_through_default_reward(self) -> None:
        """386# FIX: reward_scaling が default_reward に正しく伝搬すること."""
        import numpy as np

        from ztb.trading.environment.components.calculators.reward_calculator import (
            RewardCalculator,
        )
        from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings

        config = EnvironmentConfig()
        settings = RewardSettings()
        calc = RewardCalculator(config, settings, initial_portfolio_value=10_000_000.0)

        # reward_scaling=1.0 で呼出
        r1 = calc._calculate_default_reward(
            action=0, atr_normalised=0.01, portfolio_return=0.0,
            position=0.0, effective_max_position=0.01, current_price=15_000_000.0,
            atr=50_000.0, pnl=100.0, reward_scaling=1.0,
        )
        # reward_scaling=2.0 で呼出 — PnL 部分のみ 2 倍
        r2 = calc._calculate_default_reward(
            action=0, atr_normalised=0.01, portfolio_return=0.0,
            position=0.0, effective_max_position=0.01, current_price=15_000_000.0,
            atr=50_000.0, pnl=100.0, reward_scaling=2.0,
        )
        # PnL以外の penalty は同一なので差分 = pnl × (2.0 - 1.0) = 100.0
        diff = r2 - r1
        assert diff == pytest.approx(100.0, abs=0.01), (
            f"Expected reward diff 100.0 (pnl × scaling diff), got {diff:.4f}"
        )

    def test_yaml_reward_scaling_explicit(self) -> None:
        """386# FIX: SAC YAML に reward_scaling=1.0 が明示されていること."""
        from scripts.v460.lib.config_loader import load_config

        cfg = load_config("configs/v460/experiments/g2_sac_train.yaml")
        env_cfg = cfg.get("environment", {})
        scaling = env_cfg.get("reward_scaling")
        assert scaling is not None, "reward_scaling が YAML に明示されていない"
        assert float(scaling) == pytest.approx(1.0), (
            f"SAC YAML reward_scaling should be 1.0, got {scaling}"
        )


class TestGammaConfigModelDir:
    """gamma 実験用 YAML が別 model_dir を使用していること."""

    def test_gamma095_separate_model_dir(self) -> None:
        from scripts.v460.lib.config_loader import load_config

        cfg = load_config("configs/v460/experiments/g2_sac_gamma095.yaml")
        model_dir = cfg.get("output", {}).get("model_dir", "")
        assert "gamma095" in str(model_dir), (
            f"gamma095 config should use separate model dir, got: {model_dir}"
        )

    def test_gamma099_separate_model_dir(self) -> None:
        from scripts.v460.lib.config_loader import load_config

        cfg = load_config("configs/v460/experiments/g2_sac_gamma099.yaml")
        model_dir = cfg.get("output", {}).get("model_dir", "")
        assert "gamma099" in str(model_dir), (
            f"gamma099 config should use separate model dir, got: {model_dir}"
        )

    def test_baseline_and_gamma_model_dirs_differ(self) -> None:
        from scripts.v460.lib.config_loader import load_config

        base = load_config("configs/v460/experiments/g2_sac_train.yaml")
        g095 = load_config("configs/v460/experiments/g2_sac_gamma095.yaml")
        g099 = load_config("configs/v460/experiments/g2_sac_gamma099.yaml")

        base_dir = str(base.get("output", {}).get("model_dir", ""))
        g095_dir = str(g095.get("output", {}).get("model_dir", ""))
        g099_dir = str(g099.get("output", {}).get("model_dir", ""))

        assert base_dir != g095_dir, "Baseline and gamma095 should have different model dirs"
        assert base_dir != g099_dir, "Baseline and gamma099 should have different model dirs"
        assert g095_dir != g099_dir, "gamma095 and gamma099 should have different model dirs"


class TestRewardSettingsPropagation:
    """386# FIX: reward_settings YAML→env 伝播の検証."""

    def test_top_level_reward_settings_merged_into_env_config(self) -> None:
        """トップレベル reward_settings が actual_env_config にマージされること."""
        from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer

        # Simulate a config with top-level reward_settings
        config = {
            "environment": {
                "transaction_cost": 0.0,
                "reward_scaling": 1.0,
            },
            "reward_settings": {
                "balance_penalty_value": 0.1,
                "hold_penalty_weight": 0.001,
            },
        }

        trainer = SACTrainer.__new__(SACTrainer)
        trainer.config = config
        expected = trainer._extract_expected_reward_params(config)
        # Top-level reward_settings should be picked up
        assert "balance_penalty_value" in expected, (
            "Top-level reward_settings should be extracted"
        )
        assert expected["balance_penalty_value"] == 0.1

    def test_env_nested_reward_settings_takes_priority(self) -> None:
        """environment 内の reward_settings がトップレベルより優先されること."""
        from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer

        config = {
            "environment": {
                "transaction_cost": 0.0,
                "reward_settings": {
                    "balance_penalty_value": 0.2,
                },
            },
            "reward_settings": {
                "balance_penalty_value": 0.1,
            },
        }

        trainer = SACTrainer.__new__(SACTrainer)
        trainer.config = config
        expected = trainer._extract_expected_reward_params(config)
        # env-nested reward_settings has balance_penalty_value
        assert expected.get("balance_penalty_value") == 0.2, (
            "environment-nested reward_settings should take priority"
        )

    def test_reward_tuned_yaml_has_reward_settings(self) -> None:
        """reward-tuned YAML に reward_settings が存在すること."""
        from scripts.v460.lib.config_loader import load_config

        cfg = load_config(
            "configs/v460/experiments/g2_sac_gamma095_reward_tuned.yaml"
        )
        rs = cfg.get("reward_settings")
        assert rs is not None, "reward_settings section missing from reward-tuned YAML"
        assert isinstance(rs, dict)
        # hold_penalty_weight は直接キーなので reward_settings に存在
        assert rs.get("hold_penalty_weight") == 0.001, (
            f"Expected hold_penalty_weight=0.001, got {rs.get('hold_penalty_weight')}"
        )

    def test_reward_tuned_yaml_behavior_optimization(self) -> None:
        """reward-tuned YAML の behavior_optimization が正しく構成されていること."""
        from scripts.v460.lib.config_loader import load_config

        cfg = load_config(
            "configs/v460/experiments/g2_sac_gamma095_reward_tuned.yaml"
        )
        env = cfg.get("environment", {})
        bo = env.get("behavior_optimization")
        assert bo is not None, "behavior_optimization section missing from environment"
        assert bo.get("balance_penalty") == 0.1, (
            f"Expected balance_penalty=0.1, got {bo.get('balance_penalty')}"
        )
        assert bo.get("consistency_penalty") == 0.01, (
            f"Expected consistency_penalty=0.01, got {bo.get('consistency_penalty')}"
        )

    def test_behavior_optimization_flows_to_env_config(self) -> None:
        """behavior_optimization が EnvironmentConfig に正しくマッピングされること."""
        from ztb.trading.environment.utils.config import EnvironmentConfig

        config_dict = {
            "behavior_optimization": {
                "balance_penalty": 0.1,
                "consistency_penalty": 0.01,
            },
        }
        env_config = EnvironmentConfig.from_dict(config_dict)
        assert env_config.behavior_optimization.get("balance_penalty") == 0.1
        # consistency_penalty は reward_settings にもマッピングされる
        assert env_config.reward_settings is not None
        assert env_config.reward_settings.consistency_penalty == pytest.approx(0.01)

    def test_e2e_reward_tuned_penalty_values(self) -> None:
        """E2E: reward-tuned 設定でペナルティ値が正しく伝播すること."""
        from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings
        from ztb.trading.environment.components.calculators.reward_calculator import (
            RewardCalculator,
        )
        from ztb.trading.constants import ACTION_HOLD

        # Simulate the env config from reward-tuned YAML
        config_dict = {
            "behavior_optimization": {
                "balance_penalty": 0.1,
                "consistency_penalty": 0.01,
            },
            "reward_settings": {
                "hold_penalty_weight": 0.001,
                "confidence_penalty_threshold": 0.2,
                "position_penalty_weight": 0.01,
            },
        }
        env_config = EnvironmentConfig.from_dict(config_dict)

        # Verify settings propagated to reward_settings
        assert env_config.reward_settings.balance_penalty == pytest.approx(0.1)
        assert env_config.reward_settings.consistency_penalty == pytest.approx(0.01)

        # Verify behavior_optimization dict stored
        assert env_config.behavior_optimization["balance_penalty"] == 0.1

        # Create RewardCalculator and verify penalty values
        calc = RewardCalculator(
            env_config, env_config.reward_settings,
            initial_portfolio_value=10_000_000.0,
        )
        assert calc.balance_penalty == pytest.approx(0.1), (
            f"Expected balance_penalty=0.1, got {calc.balance_penalty}"
        )

        # Verify hold penalty
        hp = calc._calculate_hold_penalty(ACTION_HOLD)
        assert hp == pytest.approx(-0.001), (
            f"Expected hold_penalty=-0.001 for HOLD, got {hp}"
        )
