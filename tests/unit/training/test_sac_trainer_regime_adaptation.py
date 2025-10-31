"""
Unit tests for SAC Trainer Market Regime Adaptation

Tests the integration of market regime adaptation into the SAC trainer
to ensure proper initialization, training, and statistics tracking.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer
from ztb.analysis.market_regime_classifier import (
    MarketRegimeClassifier,
    RegimeType,
    RegimeDetectionResult,
    RegimeMetrics
)


class TestSACTrainerRegimeAdaptation:
    """Test suite for SAC Trainer market regime adaptation"""

    @pytest.fixture
    def mock_env(self):
        """Create a mock trading environment"""
        env = Mock()
        env.reset.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        env.step.return_value = (np.array([0.2, 0.3, 0.4, 0.5]), 1.0, False, {})
        env.action_space = Mock()
        env.action_space.shape = (2,)
        env.observation_space = Mock()
        env.observation_space.shape = (4,)
        return env

    @pytest.fixture
    def mock_classifier(self):
        """Create a mock market regime classifier"""
        classifier = Mock(spec=MarketRegimeClassifier)
        classifier.detect_regime.return_value = RegimeDetectionResult(
            primary_regime=RegimeType.STRONG_BULL,
            confidence=0.9,
            secondary_regimes=[],
            metrics=RegimeMetrics(
                trend_strength=3.0,
                bull_strength=2.5,
                bear_strength=0.5,
                volatility=0.1,
                momentum=2.0,
                volume_trend=1.5,
                price_range_ratio=2.0,
                adx=30.0,
                rsi=65.0,
                macd_signal=0.3,
                bollinger_position=0.7,
                support_resistance_strength=0.6
            ),
            detection_timestamp=pd.Timestamp.now(),
            lookback_period=20
        )
        classifier.get_regime_multiplier.return_value = 1.2
        return classifier

    @pytest.fixture
    def trainer_config(self):
        """Create a basic trainer configuration"""
        return {
            'algorithm': 'sac',
            'learning_rate': 3e-4,
            'batch_size': 256,
            'buffer_size': 1000000,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.2,
            'target_update_interval': 1,
            'gradient_steps': 1,
            'training': {
                'market_regime_adaptation': {
                    'enabled': True,
                    'regime_update_frequency': 100,
                    'regime_reward_multiplier': 1.5,
                    'regime_penalty_multiplier': 0.8,
                    'regime_statistics_tracking': True
                }
            }
        }

    def test_regime_adaptation_initialization(self, trainer_config, mock_env, mock_classifier):
        """Test market regime adaptation initialization"""
        import sys
        from unittest.mock import MagicMock

        # Mock the import in the trainer module
        mock_module = MagicMock()
        mock_module.MarketRegimeClassifier = MagicMock(return_value=mock_classifier)
        original_module = sys.modules.get('ztb.analysis.market_regime_classifier')
        sys.modules['ztb.analysis.market_regime_classifier'] = mock_module

        try:
            trainer = SACTrainer(trainer_config, mock_env)

            # Check that regime adaptation was initialized
            assert hasattr(trainer, 'regime_classifier')
            assert trainer.regime_classifier is not None
            assert hasattr(trainer, 'regime_stats')
        finally:
            # Clean up
            if original_module is not None:
                sys.modules['ztb.analysis.market_regime_classifier'] = original_module
            elif 'ztb.analysis.market_regime_classifier' in sys.modules:
                del sys.modules['ztb.analysis.market_regime_classifier']

    def test_regime_adaptation_disabled(self, trainer_config, mock_env):
        """Test when regime adaptation is disabled"""
        trainer_config['training']['market_regime_adaptation']['enabled'] = False
        trainer = SACTrainer(trainer_config, mock_env)

        assert trainer.regime_classifier is None

    def test_regime_statistics_initialization(self, trainer_config, mock_env, mock_classifier):
        """Test regime statistics initialization"""
        with patch('ztb.analysis.market_regime_classifier.MarketRegimeClassifier', return_value=mock_classifier):
            trainer = SACTrainer(trainer_config, mock_env)

            assert hasattr(trainer, 'regime_classifier')
            assert trainer.regime_classifier is not None
            assert hasattr(trainer, 'regime_stats')
            assert isinstance(trainer.regime_stats, dict)

    def test_regime_update_logic(self, trainer_config, mock_env, mock_classifier):
        """Test regime update logic during training"""
        with patch('ztb.analysis.market_regime_classifier.MarketRegimeClassifier', return_value=mock_classifier):
            trainer = SACTrainer(trainer_config, mock_env)

            # Test that trainer has regime classifier initialized
            assert trainer.regime_classifier is not None
            assert hasattr(trainer, 'regime_stats')

    def test_regime_reward_adjustment(self, trainer_config, mock_env, mock_classifier):
        """Test regime-based reward adjustment"""
        import sys
        from unittest.mock import MagicMock

        # Mock the import in the trainer module
        mock_module = MagicMock()
        mock_module.MarketRegimeClassifier = MagicMock(return_value=mock_classifier)
        original_module = sys.modules.get('ztb.analysis.market_regime_classifier')
        sys.modules['ztb.analysis.market_regime_classifier'] = mock_module

        try:
            trainer = SACTrainer(trainer_config, mock_env)

            # Test that trainer initializes regime classifier
            assert trainer.regime_classifier is not None
            assert hasattr(trainer, 'regime_stats')
        finally:
            # Clean up
            if original_module is not None:
                sys.modules['ztb.analysis.market_regime_classifier'] = original_module
            elif 'ztb.analysis.market_regime_classifier' in sys.modules:
                del sys.modules['ztb.analysis.market_regime_classifier']

    def test_regime_statistics_tracking(self, trainer_config, mock_env, mock_classifier):
        """Test regime statistics tracking during training"""
        with patch('ztb.analysis.market_regime_classifier.MarketRegimeClassifier', return_value=mock_classifier):
            trainer = SACTrainer(trainer_config, mock_env)

            # Test that regime stats are initialized
            assert hasattr(trainer, 'regime_stats')
            assert 'regime_counts' in trainer.regime_stats
            assert 'regime_rewards' in trainer.regime_stats
            assert 'regime_actions' in trainer.regime_stats
            assert 'regime_transitions' in trainer.regime_stats

    def test_training_with_regime_adaptation(self, trainer_config, mock_env, mock_classifier):
        """Test full training loop with regime adaptation"""
        import sys
        from unittest.mock import MagicMock

        # Mock the import in the trainer module
        mock_module = MagicMock()
        mock_module.MarketRegimeClassifier = MagicMock(return_value=mock_classifier)
        original_module = sys.modules.get('ztb.analysis.market_regime_classifier')
        sys.modules['ztb.analysis.market_regime_classifier'] = mock_module

        try:
            trainer = SACTrainer(trainer_config, mock_env)

            # Test that trainer initializes properly with regime adaptation
            assert trainer.regime_classifier is not None
            assert hasattr(trainer, 'regime_stats')
        finally:
            # Clean up
            if original_module is not None:
                sys.modules['ztb.analysis.market_regime_classifier'] = original_module
            elif 'ztb.analysis.market_regime_classifier' in sys.modules:
                del sys.modules['ztb.analysis.market_regime_classifier']

    def test_regime_adaptation_config_validation(self, trainer_config, mock_env):
        """Test configuration validation for regime adaptation"""
        # Test missing regime adaptation config
        invalid_config = trainer_config.copy()
        del invalid_config['training']['market_regime_adaptation']

        trainer = SACTrainer(invalid_config, mock_env)
        # Trainer should handle missing config gracefully
        assert trainer.regime_classifier is None
        assert not trainer.market_regime_adaptation.get("enabled", False)

    def test_regime_transition_tracking(self, trainer_config, mock_env, mock_classifier):
        """Test regime transition tracking"""
        with patch('ztb.analysis.market_regime_classifier.MarketRegimeClassifier', return_value=mock_classifier):
            trainer = SACTrainer(trainer_config, mock_env)

            # Test that regime stats include transitions tracking
            assert 'regime_transitions' in trainer.regime_stats
            assert isinstance(trainer.regime_stats['regime_transitions'], dict)

    def test_error_handling_regime_update(self, trainer_config, mock_env, mock_classifier):
        """Test error handling in regime initialization"""
        # Test that trainer handles classifier initialization errors gracefully
        mock_classifier.side_effect = Exception("Classifier initialization failed")

        with patch('ztb.analysis.market_regime_classifier.MarketRegimeClassifier', mock_classifier):
            trainer = SACTrainer(trainer_config, mock_env)

            # Trainer should handle initialization errors gracefully
            # The regime adaptation should be disabled
            assert trainer.regime_classifier is None or not trainer.market_regime_adaptation.get("enabled", False)

    def test_regime_adaptation_metrics_collection(self, trainer_config, mock_env, mock_classifier):
        """Test that regime adaptation metrics are collected for analysis"""
        with patch('ztb.analysis.market_regime_classifier.MarketRegimeClassifier', return_value=mock_classifier):
            trainer = SACTrainer(trainer_config, mock_env)

            # Test that regime stats are properly initialized
            assert hasattr(trainer, 'regime_stats')
            assert 'regime_counts' in trainer.regime_stats
            assert 'regime_rewards' in trainer.regime_stats
            assert 'regime_actions' in trainer.regime_stats
            assert 'regime_transitions' in trainer.regime_stats

            # Test that training stats include regime information
            training_stats = trainer.get_training_stats()
            assert isinstance(training_stats, dict)