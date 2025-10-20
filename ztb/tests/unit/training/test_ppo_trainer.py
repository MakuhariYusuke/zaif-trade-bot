"""Unit tests for PPO Trainer with auto-halt functionality."""


class TestPPOConfig:
    """Test cases for PPOConfig dataclass."""

    def test_default_config(self):
        """Test default PPOConfig values."""
        from ztb.training.config.ppo_config import DEFAULT_PPO_CONFIG

        config = DEFAULT_PPO_CONFIG

        assert config["total_timesteps"] == 1000000
        assert config["n_steps"] == 2048
        assert config["gamma"] == 0.99
        assert config["learning_rate"] == 3e-4
        assert config["clip_range"] == 0.2
        assert config["gae_lambda"] == 0.95
        assert config["max_grad_norm"] == 0.5
        assert config["vf_coef"] == 0.5
        assert config["ent_coef"] == 0.0
        assert config["batch_size"] == 64
        assert config["reward_scaling"] == 6.0
        assert config["transaction_cost"] == 0.001
        assert config["max_position_size"] == 1.0
