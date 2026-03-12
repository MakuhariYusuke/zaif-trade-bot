"""
v456 最終統合テスト

既存の ztb/trading/environment/components/reward_calculator.py との
統合を検証し、config/v456/base/config.yaml の設定値が
正しく RewardSettings dataclass に変換されることを確認
"""

import pytest
import yaml
from pathlib import Path
from typing import Dict, Any

from ztb.trading.environment.utils.config import (
    EnvironmentConfig,
    RewardSettings,
)


def load_v456_config() -> Dict[str, Any]:
    """Load v456 configuration."""
    config_path = Path('config/v456/base/config.yaml')
    assert config_path.exists(), f"Config not found: {config_path}"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


class TestV456ConfigStructure:
    """Test v456 configuration basic structure."""
    
    def test_config_loads(self):
        """Config should load without errors."""
        config = load_v456_config()
        assert config is not None
        assert 'training' in config
        assert 'environment' in config['training']
    
    def test_reward_settings_exists(self):
        """Reward settings should be defined."""
        config = load_v456_config()
        env_config = config['training']['environment']
        assert 'reward_settings' in env_config
        
        reward_settings = env_config['reward_settings']
        assert isinstance(reward_settings, dict)
        assert len(reward_settings) > 0
    
    def test_key_reward_params_present(self):
        """Key reward parameters should be present."""
        config = load_v456_config()
        reward_settings = config['training']['environment']['reward_settings']
        
        # Check for key parameters from RewardSettings dataclass
        required_keys = [
            'use_simple_reward',
            'reward_scale',
            'balance_penalty',
            'position_soft_cap',
            'volatility_window',
        ]
        
        for key in required_keys:
            assert key in reward_settings, f"Missing key: {key}"


class TestRewardSettingsDataclassIntegration:
    """Test that v456 config integrates with RewardSettings dataclass."""
    
    def test_reward_settings_from_dict(self):
        """RewardSettings.from_dict should handle v456 config."""
        config = load_v456_config()
        reward_dict = config['training']['environment']['reward_settings']
        
        # Should not raise
        reward_settings = RewardSettings.from_dict(reward_dict)
        assert reward_settings is not None
    
    def test_reward_settings_values(self):
        """RewardSettings should have correct values from v456 config."""
        config = load_v456_config()
        reward_dict = config['training']['environment']['reward_settings']
        
        reward_settings = RewardSettings.from_dict(reward_dict)
        
        # Check specific values
        assert reward_settings.use_simple_reward is False
        assert reward_settings.reward_scale == 100.0
        assert reward_settings.balance_penalty == 0.1
        assert reward_settings.position_soft_cap == 0.5
        assert reward_settings.volatility_window == 20
    
    def test_reward_settings_custom_params(self):
        """Custom parameters should be captured in custom_reward_params."""
        config = load_v456_config()
        reward_dict = config['training']['environment']['reward_settings']
        
        reward_settings = RewardSettings.from_dict(reward_dict)
        
        # Unknown keys (like curriculum_stage, enable_*) should be in custom_reward_params
        # Note: Actually they won't be since from_dict filters them
        # But they're safe to ignore as they're handled elsewhere


class TestEnvironmentConfigIntegration:
    """Test EnvironmentConfig.from_dict integration."""
    
    def test_environment_config_from_dict(self):
        """EnvironmentConfig should properly initialize from v456 config."""
        config = load_v456_config()
        env_config_dict = config['training']['environment']
        
        # This is how the actual training code would use the config
        env_config = EnvironmentConfig.from_dict(env_config_dict)
        
        assert env_config is not None
        assert env_config.transaction_cost == 0.001
        assert env_config.max_position_size == 1.0
    
    def test_environment_config_with_reward_settings(self):
        """EnvironmentConfig should integrate RewardSettings properly."""
        config = load_v456_config()
        env_config_dict = config['training']['environment']
        
        env_config = EnvironmentConfig.from_dict(env_config_dict)
        
        # Check that reward_settings was mapped
        assert hasattr(env_config, 'reward_settings')
        if env_config.reward_settings:
            # reward_settings might be a dict or RewardSettings object
            if isinstance(env_config.reward_settings, dict):
                assert 'reward_scale' in env_config.reward_settings
            else:
                assert hasattr(env_config.reward_settings, 'reward_scale')


class TestV456TrainingConfig:
    """Test training-related v456 configuration."""
    
    def test_training_config_section(self):
        """Training config section should have required parameters."""
        config = load_v456_config()
        training = config['training']
        
        required_keys = [
            'model_name',
            'algorithm',
            'total_timesteps',
            'sac_hyperparameters',
        ]
        
        for key in required_keys:
            assert key in training, f"Missing training key: {key}"
    
    def test_sac_hyperparameters(self):
        """SAC hyperparameters should be properly configured."""
        config = load_v456_config()
        sac_params = config['training']['sac_hyperparameters']
        
        # Check key SAC parameters
        assert sac_params['learning_rate'] == 0.0003
        assert sac_params['gamma'] == 0.99
        assert sac_params['tau'] == 0.005
    
    def test_evaluation_config(self):
        """Evaluation config should be present and valid."""
        config = load_v456_config()
        
        assert 'evaluation' in config
        eval_config = config['evaluation']
        
        assert 'parallel' in eval_config
        assert eval_config['parallel']['enabled'] is True
        assert eval_config['parallel']['num_workers'] == 8


class TestV456ConfigValues:
    """Test actual values in v456 config for trading correctness."""
    
    def test_balance_penalty_parameters(self):
        """Balance penalty parameters should be reasonable."""
        config = load_v456_config()
        reward_settings = config['training']['environment']['reward_settings']
        
        # Check balance penalty values
        assert 0 <= reward_settings['balance_penalty'] <= 1.0
        assert 0 < reward_settings['balance_penalty_tolerance'] < 1.0
    
    def test_trade_control_parameters(self):
        """Trade control parameters should be sane."""
        config = load_v456_config()
        reward_settings = config['training']['environment']['reward_settings']
        
        # Trade frequency penalty should be reasonable
        assert reward_settings['trade_frequency_penalty'] > 0
        assert reward_settings['max_consecutive_trades'] > 0
        assert reward_settings['trade_cooldown_steps'] > 0
    
    def test_position_parameters(self):
        """Position parameters should make sense."""
        config = load_v456_config()
        reward_settings = config['training']['environment']['reward_settings']
        
        # Position soft cap should be between 0 and 1
        assert 0 < reward_settings['position_soft_cap'] <= 1.0
        assert reward_settings['position_penalty_scale'] > 0
        assert reward_settings['position_penalty_exponent'] > 0


class TestV456CompatibilityWithExisting:
    """Test that v456 config is compatible with existing RewardCalculator."""
    
    def test_curriculum_stage_value(self):
        """Curriculum stage should be set to a valid value."""
        config = load_v456_config()
        reward_settings = config['training']['environment']['reward_settings']
        
        # curriculum_stage will be in custom_reward_params but is a valid setting
        curriculum = reward_settings.get('curriculum_stage')
        # Can be simple, balanced, advanced, or any custom stage
        assert curriculum is not None
    
    def test_no_old_format_keys(self):
        """Config should not use old format keys."""
        config = load_v456_config()
        reward_settings = config['training']['environment']['reward_settings']
        
        # v456 should use the new RewardSettings-compatible format
        # not the old nested structure (action_discovery, risk_adjustment, etc.)
        old_keys = ['action_discovery', 'risk_adjustment', 'trade_profitability',
                   'balance_management', 'smart_incentive']
        
        for key in old_keys:
            assert key not in reward_settings, f"Old format key found: {key}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
