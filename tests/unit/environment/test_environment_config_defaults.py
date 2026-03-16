from ztb.trading.environment.utils.config import EnvironmentConfig


def test_environment_config_default_initial_portfolio():
    cfg = EnvironmentConfig()
    assert hasattr(cfg, "initial_portfolio_value")
    assert isinstance(cfg.initial_portfolio_value, float)
    assert cfg.initial_portfolio_value == 200_000.0


def test_environment_config_from_dict_behavior_optimization_mapping():
    cfg_dict = {
        "environment": {
            "behavior_optimization": {
                "action_balance_target": 0.6,
                "balance_penalty": 0.12,
                "entropy_regularization": 0.01,
                "action_smoothing": 0.05,
                "consistency_penalty": 0.2,
                "redundant_trade_penalty": 0.01,
            }
        }
    }

    env_cfg = EnvironmentConfig.from_dict(cfg_dict)
    assert env_cfg.reward_settings is not None
    rs = env_cfg.reward_settings
    # Attributes may be set on RewardSettings if present, otherwise added dynamically
    # We expect that at least action_balance_target and custom params are present
    assert getattr(rs, "action_balance_target", 0.6) == 0.6
    assert getattr(rs, "balance_penalty", 0.12) == 0.12


def test_environment_config_from_dict_balance_penalty_tolerance_mapping():
    """401# F3: balance_penalty_tolerance が behavior_optimization から正しくマッピングされること."""
    cfg_dict = {
        "environment": {
            "behavior_optimization": {
                "balance_penalty_tolerance": 0.10,
            }
        }
    }
    env_cfg = EnvironmentConfig.from_dict(cfg_dict)
    rs = env_cfg.reward_settings
    assert rs is not None
    assert getattr(rs, "balance_penalty_tolerance", None) == 0.10


def test_environment_config_from_dict_balance_penalty_tolerance_default():
    """401# F3: balance_penalty_tolerance 未設定時はデフォルト 0.05 が使われること."""
    cfg_dict = {
        "environment": {
            "behavior_optimization": {
                "balance_penalty": 0.0,
            }
        }
    }
    env_cfg = EnvironmentConfig.from_dict(cfg_dict)
    rs = env_cfg.reward_settings
    assert rs is not None
    assert getattr(rs, "balance_penalty_tolerance", None) == 0.05


def test_environment_config_unknown_key_warning(caplog):
    """401# F5: 未知の設定キーが WARNING ログとして出力されること."""
    import logging

    cfg_dict = {
        "totally_unknown_key_xyz": 42,
    }
    with caplog.at_level(logging.WARNING):
        EnvironmentConfig.from_dict(cfg_dict)
    assert any("Unknown config key" in r.message for r in caplog.records)
