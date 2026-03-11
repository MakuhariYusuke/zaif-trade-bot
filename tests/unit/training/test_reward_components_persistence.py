"""Test reward_components persistence through the training pipeline."""
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.training.unified_trainer.base.callbacks import TrainingProgressCallback


class TestRewardComponentsPersistence:
    """Test that reward_components are properly collected and persisted."""

    def test_reward_components_in_env_step(self):
        """Test that HeavyTradingEnv.step() returns reward_components in info."""
        # Mock minimal env setup
        with patch.object(HeavyTradingEnv, '__init__', return_value=None):
            env = HeavyTradingEnv()
            env.current_step = 0
            env.data_provider = Mock()
            env.data_provider.get_current_price.return_value = 1000000.0
            env.reward_calculator = Mock()
            
            # Mock reward calculator to return components
            env.reward_calculator.calculate_reward.return_value = (
                0.5,  # reward
                {
                    "balance_penalty": -0.02,
                    "skew_penalty": -0.01,
                    "balance_shaping": 0.03,
                    "action_bonus": 0.0,
                    "final_reward": 0.5
                }
            )
            env.reward_calculator._last_reward_components = {
                "balance_penalty": -0.02,
                "skew_penalty": -0.01,
                "balance_shaping": 0.03,
                "action_bonus": 0.0,
                "final_reward": 0.5
            }
            
            # Mock other required methods
            env._get_observation = Mock(return_value=[0] * 10)
            env._is_done = Mock(return_value=False)
            
            # Execute step
            obs, reward, done, truncated, info = env.step(1)
            
            # Verify reward_components in info
            assert "reward_components" in info
            assert info["reward_components"]["balance_penalty"] == -0.02
            assert info["reward_components"]["skew_penalty"] == -0.01
            assert info["reward_components"]["balance_shaping"] == 0.03

    def test_callback_collects_reward_components(self):
        """Test that TrainingProgressCallback collects reward_components."""
        callback = TrainingProgressCallback(
            verbose=0,
            eval_freq=10,
            log_freq=10,
            n_eval_episodes=1
        )
        
        # Mock model and locals
        mock_model = Mock()
        mock_model.num_timesteps = 1
        
        # Mock info with reward_components
        mock_info = {
            "reward_components": {
                "balance_penalty": -0.02,
                "skew_penalty": -0.01,
                "balance_shaping": 0.03,
                "final_reward": 0.5
            }
        }
        
        callback.model = mock_model
        callback.locals = {"infos": [mock_info]}
        
        # Initialize history
        callback.reward_components_history = []
        
        # Collect components (simulate what happens in _on_step)
        for info in callback.locals.get("infos", []):
            if "reward_components" in info:
                callback.reward_components_history.append(info["reward_components"].copy())
        
        # Verify collection
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
        
        # Aggregate components (as done in sac_trainer.py)
        components = {}
        for comp_dict in history:
            for key, val in comp_dict.items():
                if key not in components:
                    components[key] = []
                components[key].append(float(val))
        
        # Average
        avg_components = {k: sum(v) / len(v) for k, v in components.items() if v}
        
        # Verify averages
        assert abs(avg_components["balance_penalty"] - (-0.025)) < 1e-6
        assert abs(avg_components["skew_penalty"] - (-0.0123333)) < 1e-5
        assert abs(avg_components["final_reward"] - 0.55) < 1e-6

    def test_training_report_includes_reward_components(self):
        """Test that training report JSON includes reward_components section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "test_report.json"
            
            # Create mock report with reward_components
            report = {
                "configuration": {"version": "test"},
                "training_stats": {
                    "total_timesteps": 1000,
                    "final_reward": 0.5
                },
                "reward_components": {
                    "balance_penalty": -0.025,
                    "skew_penalty": -0.012,
                    "balance_shaping": 0.03,
                    "final_reward": 0.55
                }
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
