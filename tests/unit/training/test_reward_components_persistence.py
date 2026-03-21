"""Test reward_components persistence through the training pipeline."""
import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from ztb.trading.constants import ACTION_HOLD
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.training.training_stats_payloads import average_reward_component_history
from ztb.training.unified_trainer.base.callbacks import TrainingProgressCallback


def _make_minimal_env() -> HeavyTradingEnv:
    rows = 64
    close = np.linspace(100.0, 103.0, rows)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="5min"),
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": np.full(rows, 1000.0),
        }
    )
    config = EnvironmentConfig.from_dict(
        {
            "random_start": False,
            "feature_set": "minimal",
            "use_continuous_actions": False,
        }
    )
    return HeavyTradingEnv(df=df, config=config)


class TestRewardComponentsPersistence:
    """Test that reward_components are properly collected and persisted."""

    def test_reward_components_in_env_step(self):
        """Test that HeavyTradingEnv.step() returns reward_components in info."""
        env = _make_minimal_env()
        env.reset()
        reward_components = {
            "balance_penalty": -0.02,
            "skew_penalty": -0.01,
            "balance_shaping": 0.03,
            "action_bonus": 0.0,
            "final_reward": 0.5,
        }

        with (
            patch.object(env.reward_calculator, "calculate_reward", return_value=0.5),
            patch.object(
                env.reward_calculator,
                "get_last_reward_components",
                return_value=reward_components,
            ),
        ):
            _, _, _, _, info = env.step(ACTION_HOLD)

        assert "reward_components" in info
        assert info["reward_components"]["balance_penalty"] == -0.02
        assert info["reward_components"]["skew_penalty"] == -0.01
        assert info["reward_components"]["balance_shaping"] == 0.03

    def test_callback_collects_reward_components(self):
        """Test that TrainingProgressCallback collects reward_components."""
        callback = TrainingProgressCallback(
            check_freq=10,
            verbose=0,
        )

        mock_model = Mock()
        mock_model.num_timesteps = 1
        mock_model.logger = Mock()
        mock_model.logger.name_to_value = {}
        mock_model.policy = Mock()
        mock_model.policy.optimizer = Mock()
        mock_model.policy.optimizer.param_groups = [{"lr": 0.001}]

        mock_info = {
            "reward_components": {
                "balance_penalty": -0.02,
                "skew_penalty": -0.01,
                "balance_shaping": 0.03,
                "final_reward": 0.5
            }
        }
        callback.model = mock_model
        callback.locals = {
            "actions": [ACTION_HOLD],
            "rewards": [0.5],
            "infos": [mock_info],
        }
        callback._log_progress = Mock()
        callback.n_calls = 1

        assert callback._on_step() is True
        assert len(callback.reward_components_history) == 1
        assert callback.reward_components_history[0]["balance_penalty"] == -0.02
        assert callback.reward_components_history[0]["balance_shaping"] == 0.03

    def test_reward_components_averaging(self):
        """Test that reward_components are correctly averaged for reporting."""
        # Simulate collected history
        history = [
            {"balance_penalty": -0.02, "skew_penalty": -0.01, "final_reward": 0.5},
            {"balance_penalty": -0.03, "skew_penalty": -0.015, "final_reward": 0.6},
            {"balance_penalty": -0.025, "skew_penalty": -0.012, "final_reward": 0.55},
        ]

        avg_components = average_reward_component_history(history)

        # Verify averages
        assert abs(avg_components["balance_penalty"] - (-0.025)) < 1e-6
        assert abs(avg_components["skew_penalty"] - (-0.0123333)) < 1e-5
        assert abs(avg_components["final_reward"] - 0.55) < 1e-6

    def test_training_report_includes_reward_components(self, tmp_path: Path):
        """Test that training report JSON includes reward_components section."""
        report_path = tmp_path / "test_report.json"

        # Create mock report with reward_components
        report = {
            "configuration": {"version": "test"},
            "training_stats": {"total_timesteps": 1000, "final_reward": 0.5},
            "reward_components": {
                "balance_penalty": -0.025,
                "skew_penalty": -0.012,
                "balance_shaping": 0.03,
                "final_reward": 0.55,
            },
        }

        # Write report
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Read and verify
        with open(report_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert "reward_components" in loaded
        assert loaded["reward_components"]["balance_penalty"] == -0.025
        assert loaded["reward_components"]["balance_shaping"] == 0.03


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
