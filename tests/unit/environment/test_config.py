from ztb.trading.environment.utils.config import EnvironmentConfig


class TestEnvironmentConfig:
    """Test EnvironmentConfig.from_dict with various config layouts."""

    def test_from_dict_top_level_direct(self):
        """Test config with use_continuous_actions at top level."""
        config_dict = {"use_continuous_actions": True}
        config = EnvironmentConfig.from_dict(config_dict)
        assert config.use_continuous_actions is True

    def test_from_dict_training_environment(self):
        """Test config with use_continuous_actions under training.environment."""
        config_dict = {"training": {"environment": {"use_continuous_actions": True}}}
        config = EnvironmentConfig.from_dict(config_dict)
        assert config.use_continuous_actions is True

    def test_from_dict_training_environment_config(self):
        """Test config with use_continuous_actions under training.environment.config."""
        config_dict = {
            "training": {"environment": {"config": {"use_continuous_actions": True}}}
        }
        config = EnvironmentConfig.from_dict(config_dict)
        assert config.use_continuous_actions is True

    def test_from_dict_default_false(self):
        """Test empty config defaults use_continuous_actions to False."""
        config_dict = {}
        config = EnvironmentConfig.from_dict(config_dict)
        assert config.use_continuous_actions is False
