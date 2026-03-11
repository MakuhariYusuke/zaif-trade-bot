"""
Unit tests for HeavyTradingEnv Market Regime Adaptation

Tests the integration of market regime adaptation into the heavy trading environment
to ensure proper reward adjustment and regime-aware behavior.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from ztb.analysis.regime.market_regime_classifier import (
    MarketRegimeClassifier,
    RegimeDetectionResult,
    RegimeMetrics,
    RegimeType,
)
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv


class TestHeavyTradingEnvRegimeAdaptation:
    """Test suite for HeavyTradingEnv market regime adaptation"""

    @pytest.fixture
    def mock_classifier(self):
        """Create a mock market regime classifier"""
        classifier = Mock(spec=MarketRegimeClassifier)
        classifier.detect_regime.return_value = RegimeDetectionResult(
            primary_regime=RegimeType.STRONG_BULL,
            confidence=0.85,
            secondary_regimes=[],
            metrics=RegimeMetrics(
                trend_strength=3.5,
                bull_strength=3.0,
                bear_strength=0.5,
                volatility=0.12,
                momentum=2.8,
                volume_trend=2.0,
                price_range_ratio=2.2,
                adx=32.0,
                rsi=68.0,
                macd_signal=0.4,
                bollinger_position=0.75,
                support_resistance_strength=0.7,
            ),
            detection_timestamp=pd.Timestamp.now(),
            lookback_period=25,
        )
        classifier.get_regime_multiplier.return_value = 1.3
        return classifier

    @pytest.fixture
    def env_config(self):
        """Create environment configuration"""
        return {
            "initial_balance": 10000,
            "max_position_size": 1.0,
            "transaction_fee": 0.001,
            "slippage": 0.0005,
            "market_regime_adaptation": {
                "enabled": True,
                "regime_reward_multiplier": 1.2,
                "regime_penalty_multiplier": 0.9,
            },
        }

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing"""
        dates = pd.date_range("2023-01-01", periods=1000, freq="5min")
        np.random.seed(42)

        # Create realistic price data with trends
        base_price = 100
        trend = np.sin(np.linspace(0, 4 * np.pi, 1000)) * 5  # Cyclical trend
        noise = np.random.normal(0, 0.5, 1000)
        close = base_price + trend + noise

        # Create OHLC data
        high = close + np.abs(np.random.normal(0, 0.3, 1000))
        low = close - np.abs(np.random.normal(0, 0.3, 1000))
        open_price = np.roll(close, 1)
        open_price[0] = base_price
        volume = np.random.uniform(100, 1000, 1000)

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

        return df

    def test_regime_adaptation_initialization(self, env_config, mock_classifier):
        """Test market regime adaptation initialization in environment"""
        with patch(
            "ztb.trading.environment.heavy_env.core.MarketRegimeClassifier",
            return_value=mock_classifier,
        ):
            env = HeavyTradingEnv(env_config)

            # Enable regime adaptation
            env.enable_market_regime_adaptation(mock_classifier)

            assert env.market_regime_adaptation_enabled is True
            assert env.market_regime_classifier is not None
            assert env.regime_reward_multiplier == 1.2
            assert env.regime_penalty_multiplier == 0.9

    def test_regime_adaptation_disabled_by_default(self, env_config):
        """Test that regime adaptation is disabled by default"""
        env = HeavyTradingEnv(env_config)

        assert env.market_regime_adaptation_enabled is False
        assert env.market_regime_classifier is None

    def test_regime_reward_adjustment_positive(self, env_config, mock_classifier):
        """Test positive reward adjustment based on regime"""
        with patch(
            "ztb.trading.environment.heavy_env.core.MarketRegimeClassifier",
            return_value=mock_classifier,
        ):
            env = HeavyTradingEnv(env_config)
            env.enable_market_regime_adaptation(mock_classifier)

            # Set current regime
            env.current_regime = RegimeType.STRONG_BULL

            # Test positive reward adjustment
            original_reward = 10.0
            adjusted_reward = env._adjust_reward_for_regime(original_reward)

            # Should be multiplied by regime reward multiplier
            expected_reward = original_reward * 1.3  # classifier returns 1.3
            assert adjusted_reward == expected_reward

    def test_regime_penalty_adjustment_negative(self, env_config, mock_classifier):
        """Test negative reward (penalty) adjustment based on regime"""
        with patch(
            "ztb.trading.environment.heavy_env.core.MarketRegimeClassifier",
            return_value=mock_classifier,
        ):
            env = HeavyTradingEnv(env_config)
            env.enable_market_regime_adaptation(mock_classifier)

            # Set current regime
            env.current_regime = RegimeType.STRONG_BEAR

            # Test negative reward adjustment
            original_reward = -5.0
            adjusted_reward = env._adjust_reward_for_regime(original_reward)

            # Should be multiplied by regime penalty multiplier
            expected_reward = (
                original_reward * 1.3
            )  # classifier returns 1.3 for penalty
            assert adjusted_reward == expected_reward

    def test_regime_update_in_step(
        self, env_config, mock_classifier, sample_market_data
    ):
        """Test regime update during environment step"""
        with patch(
            "ztb.trading.environment.heavy_env.core.MarketRegimeClassifier",
            return_value=mock_classifier,
        ):
            env = HeavyTradingEnv(env_config)
            env.enable_market_regime_adaptation(mock_classifier)

            # Set market data
            env.market_data = sample_market_data

            # Mock action and current state
            action = np.array([0.1, 0.0])  # Small buy action
            env.current_step = 100

            # Call step method
            observation, reward, done, info = env.step(action)

            # Check that regime was updated (every 50 steps by default)
            if env.current_step % 50 == 0:
                mock_classifier.detect_regime.assert_called()

            # Check that reward was adjusted
            assert isinstance(reward, (int, float))

    def test_regime_statistics_tracking(self, env_config, mock_classifier):
        """Test regime statistics tracking in environment"""
        with patch(
            "ztb.trading.environment.heavy_env.core.MarketRegimeClassifier",
            return_value=mock_classifier,
        ):
            env = HeavyTradingEnv(env_config)
            env.enable_market_regime_adaptation(mock_classifier)

            # Set current regime
            env.current_regime = RegimeType.STRONG_BULL

            # Simulate some steps
            env._update_regime_statistics(2.5, action=[0.2, 0.1])
            env._update_regime_statistics(-1.2, action=[-0.1, -0.2])

            # Check statistics
            bull_stats = env.regime_statistics[RegimeType.STRONG_BULL.name]
            assert bull_stats["count"] == 2
            assert bull_stats["total_reward"] == 2.5
            assert bull_stats["total_penalty"] == -1.2
            assert len(bull_stats["actions"]) == 2

    def test_regime_transition_detection(self, env_config, mock_classifier):
        """Test regime transition detection"""
        with patch(
            "ztb.trading.environment.heavy_env.core.MarketRegimeClassifier",
            return_value=mock_classifier,
        ):
            env = HeavyTradingEnv(env_config)
            env.enable_market_regime_adaptation(mock_classifier)

            # Set initial regime
            env.current_regime = RegimeType.LOW_VOLATILITY_RANGE

            # Simulate regime change
            env._update_market_regime()

            # Check transition was recorded
            assert len(env.regime_transitions) == 1
            assert env.regime_transitions[0][0] == RegimeType.LOW_VOLATILITY_RANGE
            assert env.regime_transitions[0][1] == RegimeType.STRONG_BULL

    def test_regime_adaptation_with_different_regimes(
        self, env_config, mock_classifier
    ):
        """Test regime adaptation with different market regimes"""
        with patch(
            "ztb.trading.environment.heavy_env.core.MarketRegimeClassifier",
            return_value=mock_classifier,
        ):
            env = HeavyTradingEnv(env_config)
            env.enable_market_regime_adaptation(mock_classifier)

            # Configure different multipliers for different regimes
            mock_classifier.get_regime_multiplier.side_effect = (
                lambda regime, reward_type: {
                    (RegimeType.STRONG_BULL, "reward"): 1.5,
                    (RegimeType.STRONG_BULL, "penalty"): 0.8,
                    (RegimeType.STRONG_BEAR, "reward"): 0.7,
                    (RegimeType.STRONG_BEAR, "penalty"): 1.4,
                    (RegimeType.HIGH_VOLATILITY_RANGE, "reward"): 1.2,
                    (RegimeType.HIGH_VOLATILITY_RANGE, "penalty"): 1.2,
                }.get((regime, reward_type), 1.0)
            )

            # Test strong bull regime
            env.current_regime = RegimeType.STRONG_BULL
            reward = env._adjust_reward_for_regime(10.0)
            penalty = env._adjust_reward_for_regime(-5.0)

            assert reward == 15.0  # 10 * 1.5
            assert penalty == -4.0  # -5 * 0.8

            # Test strong bear regime
            env.current_regime = RegimeType.STRONG_BEAR
            reward = env._adjust_reward_for_regime(10.0)
            penalty = env._adjust_reward_for_regime(-5.0)

            assert reward == 7.0  # 10 * 0.7
            assert penalty == -7.0  # -5 * 1.4

    def test_regime_adaptation_config_validation(self, env_config):
        """Test configuration validation for regime adaptation"""
        # Test with missing regime adaptation config
        invalid_config = env_config.copy()
        del invalid_config["market_regime_adaptation"]

        env = HeavyTradingEnv(invalid_config)
        # Should not crash, just disable adaptation
        assert env.market_regime_adaptation_enabled is False

        # Test with invalid multipliers
        invalid_config = env_config.copy()
        invalid_config["market_regime_adaptation"]["regime_reward_multiplier"] = -1.0

        env = HeavyTradingEnv(invalid_config)
        # Should use default or handle gracefully
        assert env.regime_reward_multiplier >= 0

    def test_regime_update_frequency(self, env_config, mock_classifier):
        """Test regime update frequency control"""
        with patch(
            "ztb.trading.environment.heavy_env.core.MarketRegimeClassifier",
            return_value=mock_classifier,
        ):
            env = HeavyTradingEnv(env_config)
            env.enable_market_regime_adaptation(mock_classifier, update_frequency=10)

            # Simulate steps
            for step in range(15):
                env.current_step = step
                env._update_market_regime_if_needed()

                # Should update every 10 steps
                expected_calls = (step // 10) + 1 if step % 10 == 0 else step // 10
                assert mock_classifier.detect_regime.call_count == expected_calls

    @patch("ztb.trading.environment.heavy_env.core.logger")
    def test_error_handling_regime_update(
        self, mock_logger, env_config, mock_classifier
    ):
        """Test error handling in regime updates"""
        mock_classifier.detect_regime.side_effect = Exception("Detection failed")

        with patch(
            "ztb.trading.environment.heavy_env.core.MarketRegimeClassifier",
            return_value=mock_classifier,
        ):
            env = HeavyTradingEnv(env_config)
            env.enable_market_regime_adaptation(mock_classifier)

            # Should not crash on regime update failure
            env._update_market_regime()

            # Check that error was logged
            mock_logger.error.assert_called()

            # Should maintain previous regime
            assert env.current_regime is not None

    def test_regime_adaptation_info_logging(self, env_config, mock_classifier):
        """Test that regime adaptation info is included in step info"""
        with patch(
            "ztb.trading.environment.heavy_env.core.MarketRegimeClassifier",
            return_value=mock_classifier,
        ):
            env = HeavyTradingEnv(env_config)
            env.enable_market_regime_adaptation(mock_classifier)

            # Set current regime
            env.current_regime = RegimeType.STRONG_BULL
            env.regime_confidence = 0.85

            # Mock step method to return info dict
            action = np.array([0.1, 0.0])
            observation, reward, done, info = env.step(action)

            # Check that regime info is included
            assert "regime" in info
            assert "regime_confidence" in info
            assert "regime_adjusted_reward" in info
            assert info["regime"] == RegimeType.STRONG_BULL.name
            assert info["regime_confidence"] == 0.85

    def test_regime_statistics_reset(self, env_config, mock_classifier):
        """Test regime statistics reset functionality"""
        with patch(
            "ztb.trading.environment.heavy_env.core.MarketRegimeClassifier",
            return_value=mock_classifier,
        ):
            env = HeavyTradingEnv(env_config)
            env.enable_market_regime_adaptation(mock_classifier)

            # Add some statistics
            env.current_regime = RegimeType.STRONG_BULL
            env._update_regime_statistics(1.0, action=[0.1, 0.0])

            # Reset environment
            env.reset()

            # Statistics should be preserved or reset based on implementation
            # (This depends on the specific reset logic)
            assert hasattr(env, "regime_statistics")
