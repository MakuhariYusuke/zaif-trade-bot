"""
Integration tests for market regime adaptation across env and trainer wiring.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.helpers.market_data import make_multi_regime_ohlcv_data
from ztb.analysis.regime.market_regime_classifier import RegimeType
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
]


class _StubStructuredLogger:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def info(self, *args, **kwargs) -> None:
        pass

    def debug(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass


class _StubTrainingStateManager:
    def __init__(self, *args, **kwargs) -> None:
        pass


class _StubCheckpointManager:
    def __init__(self, *args, **kwargs) -> None:
        pass


class _StubRegimeClassifier:
    def detect_regime(self, price_data):
        return SimpleNamespace(primary_regime=RegimeType.CONSOLIDATION)

    def get_regime_multiplier(self, regime, reward_type: str) -> float:
        if reward_type == "reward" and getattr(regime, "value", regime) == RegimeType.STRONG_BULL.value:
            return 1.4
        if reward_type == "penalty" and getattr(regime, "value", regime) == RegimeType.STRONG_BEAR.value:
            return 1.3
        return 1.0


class TestMarketRegimeAdaptationIntegration:
    """Integration tests with fast-path fixtures for regime adaptation wiring."""

    @pytest.fixture(scope="class", autouse=True)
    def patch_trainer_side_effects(self):
        with (
            patch(
                "ztb.training.unified_trainer.algorithms.sac_trainer.StructuredLogger",
                _StubStructuredLogger,
            ),
            patch(
                "ztb.training.unified_trainer.algorithms.sac_trainer.TrainingStateManager",
                _StubTrainingStateManager,
            ),
            patch(
                "ztb.training.unified_trainer.algorithms.sac_trainer.TrainingCheckpointManager",
                _StubCheckpointManager,
            ),
        ):
            yield

    @pytest.fixture(scope="class")
    def sample_market_data(self):
        return make_multi_regime_ohlcv_data(rows_per_regime=32, seed=42)

    @pytest.fixture(scope="class")
    def trainer_config(self):
        return {
            "algorithm": "sac",
            "learning_rate": 3e-4,
            "batch_size": 64,
            "buffer_size": 1024,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
            "target_update_interval": 1,
            "gradient_steps": 1,
            "training": {
                "market_regime_adaptation": {
                    "enabled": True,
                    "regime_update_frequency": 10,
                    "regime_statistics_tracking": True,
                    "lookback_periods": {"short": 4, "medium": 8, "long": 16},
                    "regime_scheme": "comprehensive",
                }
            },
        }

    @pytest.fixture(scope="class")
    def env_config(self):
        return {
            "initial_balance": 10000,
            "max_position_size": 1.0,
            "transaction_fee": 0.001,
            "slippage": 0.0005,
            "random_start": False,
            "use_continuous_actions": False,
            "feature_set": "minimal",
            "market_regime_adaptation": {"enabled": True},
        }

    def _make_env(self, market_data, env_config):
        return HeavyTradingEnv(df=market_data.copy(), config=copy.deepcopy(env_config))

    def _make_trainer(self, trainer_config, env):
        return SACTrainer(copy.deepcopy(trainer_config), env)

    @pytest.fixture(scope="class")
    def initialized_system(self, sample_market_data, trainer_config, env_config):
        env = self._make_env(sample_market_data.head(80), env_config)
        trainer = self._make_trainer(trainer_config, env)
        return env, trainer

    @pytest.fixture(scope="class")
    def adapted_env_pair(self, sample_market_data, env_config):
        market_data = sample_market_data.head(64)
        env_with_adaptation = self._make_env(market_data, env_config)
        env_with_adaptation.enable_market_regime_adaptation(
            _StubRegimeClassifier(), {"enabled": True}
        )
        env_without_adaptation = self._make_env(market_data, env_config)
        return env_with_adaptation, env_without_adaptation

    def test_complete_regime_adaptation_workflow(self, initialized_system):
        env, trainer = initialized_system

        assert trainer.regime_adaptation_enabled is True
        assert env.market_regime_adaptation_enabled is True
        assert trainer.regime_classifier is not None
        assert env.regime_classifier is not None

        state = env.reset()
        assert state is not None
        assert hasattr(trainer, "regime_statistics")
        assert "regime_counts" in trainer.regime_statistics
        assert "regime_counts" in env.regime_statistics

    def test_regime_adaptation_changes_reward_path(self, adapted_env_pair):
        env_with_adaptation, env_without_adaptation = adapted_env_pair
        env_with_adaptation.reset()
        env_without_adaptation.reset()

        with (
            patch.object(
                env_with_adaptation.reward_calculator,
                "calculate_reward",
                return_value=1.0,
            ),
            patch.object(
                env_without_adaptation.reward_calculator,
                "calculate_reward",
                return_value=1.0,
            ),
            patch.object(
                env_with_adaptation,
                "_get_current_market_regime",
                return_value=RegimeType.STRONG_BULL,
            ),
            patch.object(
                env_without_adaptation,
                "_get_current_market_regime",
                return_value=RegimeType.STRONG_BULL,
            ),
        ):
            _, reward_with, _, _, _ = env_with_adaptation.step(0)
            _, reward_without, _, _, _ = env_without_adaptation.step(0)

        assert env_with_adaptation.market_regime_adaptation_enabled is True
        assert (
            getattr(env_without_adaptation, "market_regime_adaptation_enabled", False)
            is False
        )
        assert reward_with > reward_without

    def test_regime_transition_handling(self, adapted_env_pair):
        env_with_adaptation, _ = adapted_env_pair
        env_with_adaptation.reset()
        env_with_adaptation.regime_stats["regime_transitions"].clear()
        env_with_adaptation.regime_stats["current_regime"] = None

        for regime in (
            RegimeType.STRONG_BULL,
            RegimeType.STRONG_BEAR,
            RegimeType.CONSOLIDATION,
        ):
            with (
                patch.object(
                    env_with_adaptation.reward_calculator,
                    "calculate_reward",
                    return_value=0.5,
                ),
                patch.object(
                    env_with_adaptation,
                    "_get_current_market_regime",
                    return_value=regime,
                ),
            ):
                env_with_adaptation.step(0)

        transitions = list(env_with_adaptation.regime_stats["regime_transitions"])
        assert len(transitions) >= 2
        assert transitions[-1]["from"] == RegimeType.STRONG_BEAR
        assert transitions[-1]["to"] == RegimeType.CONSOLIDATION

    def test_regime_adaptation_stability(self, initialized_system):
        env, trainer = initialized_system
        assert env.market_regime_adaptation_enabled is True
        assert trainer.regime_adaptation_enabled is True
        assert "regime_counts" in env.regime_statistics
        assert "regime_counts" in trainer.regime_statistics

    def test_regime_statistics_accuracy(self, initialized_system):
        env, trainer = initialized_system
        assert isinstance(env.regime_statistics, dict)
        assert isinstance(trainer.regime_statistics, dict)
        assert "regime_counts" in env.regime_statistics
        assert "regime_counts" in trainer.regime_statistics

    def test_error_handling_integration(self, adapted_env_pair):
        env_with_adaptation, _ = adapted_env_pair
        env_with_adaptation.reset()
        if hasattr(env_with_adaptation, "_market_regime_cache"):
            env_with_adaptation._market_regime_cache = [None] * len(env_with_adaptation.df)

        with (
            patch.object(
                env_with_adaptation.reward_calculator,
                "calculate_reward",
                return_value=0.25,
            ),
            patch.object(
                env_with_adaptation.regime_classifier,
                "detect_regime",
                side_effect=Exception("Detection failed"),
            ),
            patch("ztb.trading.environment.heavy_env.core.logger"),
        ):
            next_state, reward, done, truncated, info = env_with_adaptation.step(0)

        assert next_state is not None
        assert isinstance(reward, (int, float))
