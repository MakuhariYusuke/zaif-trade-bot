"""Unit tests for PPO Trainer with auto-halt functionality."""

from ztb.training.ppo_trainer import PPOConfig


class TestPPOConfig:
    """Test cases for PPOConfig dataclass."""

    def test_default_config(self):
        """Test default PPOConfig values."""
        config = PPOConfig()

        assert config.algorithm.value == "ppo"
        assert config.total_timesteps == 1000000
        assert config.n_steps == 2048
        assert config.gamma == 0.99
        assert config.learning_rate == 3e-4
        assert config.clip_range == 0.2
        assert config.gae_lambda == 0.95
        assert config.max_grad_norm == 0.5
        assert config.vf_coef == 0.5
        assert config.ent_coef == 0.0
        assert config.tensorboard_log == ""
        assert config.seed == 42
        assert config.batch_size == 64
        assert config.reward_scaling == 1.0
        assert config.transaction_cost == 0.0
        assert config.max_position_size == 1.0
