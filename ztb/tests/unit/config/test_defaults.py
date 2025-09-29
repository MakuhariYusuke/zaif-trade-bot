"""
Unit tests for default configuration values.
"""

from ztb.schema import (
    DEFAULT_RISK_PROFILE,
    DEFAULT_TRAINING_CONFIG,
    RiskProfileConfig,
    TrainingConfig,
)


class TestDefaultConfigurations:
    """Test default configuration values."""

    def test_default_risk_profile_is_aggressive(self):
        """Test that default risk profile is aggressive."""
        assert DEFAULT_RISK_PROFILE.name == "aggressive"
        assert DEFAULT_RISK_PROFILE.max_position_size == 1.0
        assert DEFAULT_RISK_PROFILE.stop_loss_pct == 0.05
        assert DEFAULT_RISK_PROFILE.take_profit_pct == 0.10

    def test_default_training_config_btc_jpy(self):
        """Test that default training config uses BTC/JPY."""
        assert DEFAULT_TRAINING_CONFIG.symbol == "BTC_JPY"
        assert DEFAULT_TRAINING_CONFIG.venue == "coincheck"
        assert DEFAULT_TRAINING_CONFIG.risk_profile.name == "aggressive"

    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        assert config.symbol == "BTC_JPY"
        assert config.venue == "coincheck"
        assert config.total_timesteps == 1_000_000
        assert config.n_envs == 4
        assert config.risk_profile.name == "aggressive"

    def test_risk_profile_config_defaults(self):
        """Test RiskProfileConfig default values."""
        profile = RiskProfileConfig()
        assert profile.name == "aggressive"
        assert profile.max_position_size == 1.0
        assert profile.stop_loss_pct == 0.05
        assert profile.take_profit_pct == 0.10
        assert profile.max_daily_loss_pct == 0.20
        assert profile.circuit_breaker_threshold == 0.15

    def test_explicit_config_overrides_defaults(self):
        """Test that explicit configuration overrides defaults."""
        config = TrainingConfig(symbol="ETH_JPY", venue="other_venue")
        assert config.symbol == "ETH_JPY"
        assert config.venue == "other_venue"
        # Other defaults should remain
        assert config.total_timesteps == 1_000_000
        assert config.risk_profile.name == "aggressive"
