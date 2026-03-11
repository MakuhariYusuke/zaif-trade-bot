"""Unit tests for HeavyTradingEnv market regime adaptation."""

from __future__ import annotations

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
from ztb.trading.constants import ACTION_HOLD
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig


@pytest.fixture
def mock_classifier() -> Mock:
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
        detection_timestamp=pd.Timestamp("2023-01-01T00:00:00Z"),
        lookback_period=25,
    )
    classifier.get_regime_multiplier.return_value = 1.3
    return classifier


@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    rows = 256
    rng = np.random.default_rng(42)
    base = 100 + np.linspace(0, 4, rows)
    close = base + np.sin(np.linspace(0, 8 * np.pi, rows)) * 2 + rng.normal(0, 0.2, rows)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "open": close + rng.normal(0, 0.1, rows),
            "high": close + np.abs(rng.normal(0, 0.15, rows)),
            "low": close - np.abs(rng.normal(0, 0.15, rows)),
            "close": close,
            "volume": rng.uniform(100.0, 1000.0, rows),
        }
    )


def _make_env(
    df: pd.DataFrame,
    **overrides: object,
) -> HeavyTradingEnv:
    config = EnvironmentConfig.from_dict(
        {
            "random_start": False,
            "feature_set": "minimal",
            "use_continuous_actions": False,
            **overrides,
        }
    )
    return HeavyTradingEnv(df=df, config=config)


class TestHeavyTradingEnvRegimeAdaptation:
    def test_regime_adaptation_initialization(
        self,
        sample_market_data: pd.DataFrame,
        mock_classifier: Mock,
    ) -> None:
        env = _make_env(sample_market_data)

        env.enable_market_regime_adaptation(
            regime_classifier=mock_classifier,
            adaptation_config={
                "enabled": True,
                "regime_reward_multiplier": 1.2,
                "regime_penalty_multiplier": 0.9,
            },
        )

        assert env.market_regime_adaptation_enabled is True
        assert env.regime_classifier is mock_classifier
        assert env.regime_statistics is env.regime_stats
        assert set(env.regime_stats.keys()) == {
            "regime_counts",
            "regime_rewards",
            "regime_actions",
            "current_regime",
            "regime_transitions",
        }

    def test_regime_adaptation_is_opt_in(
        self,
        sample_market_data: pd.DataFrame,
    ) -> None:
        env = _make_env(sample_market_data)

        assert not getattr(env, "market_regime_adaptation_enabled", False)
        assert env.regime_classifier is None

    def test_advanced_market_regime_config_initializes_classifier(
        self,
        sample_market_data: pd.DataFrame,
    ) -> None:
        env = _make_env(
            sample_market_data,
            advanced_market_regime={
                "enabled": True,
                "regime_classifier_config": {
                    "lookback_periods": {"short": 5, "medium": 10, "long": 20}
                },
            },
        )

        assert isinstance(env.regime_classifier, MarketRegimeClassifier)

    def test_positive_reward_adjustment_applied_in_step(
        self,
        sample_market_data: pd.DataFrame,
        mock_classifier: Mock,
    ) -> None:
        env = _make_env(sample_market_data)
        env.enable_market_regime_adaptation(mock_classifier, {"enabled": True})
        env.reset()

        with (
            patch.object(env.reward_calculator, "calculate_reward", return_value=10.0),
            patch.object(env, "_get_current_market_regime", return_value=RegimeType.STRONG_BULL),
        ):
            _, reward, _, _, _ = env.step(ACTION_HOLD)

        assert reward == pytest.approx(13.0)
        mock_classifier.get_regime_multiplier.assert_any_call(
            RegimeType.STRONG_BULL,
            "reward",
        )

    def test_negative_reward_adjustment_applied_in_step(
        self,
        sample_market_data: pd.DataFrame,
        mock_classifier: Mock,
    ) -> None:
        env = _make_env(sample_market_data)
        env.enable_market_regime_adaptation(mock_classifier, {"enabled": True})
        env.reset()
        mock_classifier.get_regime_multiplier.side_effect = (
            lambda regime, reward_type: 1.0 if reward_type == "reward" else 1.4
        )

        with (
            patch.object(env.reward_calculator, "calculate_reward", return_value=-5.0),
            patch.object(env, "_get_current_market_regime", return_value=RegimeType.STRONG_BEAR),
        ):
            _, reward, _, _, _ = env.step(ACTION_HOLD)

        assert reward == pytest.approx(-7.0)

    def test_step_tracks_regime_statistics_and_transitions(
        self,
        sample_market_data: pd.DataFrame,
        mock_classifier: Mock,
    ) -> None:
        env = _make_env(sample_market_data)
        env.enable_market_regime_adaptation(mock_classifier, {"enabled": True})
        env.reset()

        with (
            patch.object(env.reward_calculator, "calculate_reward", return_value=2.0),
            patch.object(env, "_get_current_market_regime", return_value=RegimeType.STRONG_BULL),
        ):
            env.step(ACTION_HOLD)

        with (
            patch.object(env.reward_calculator, "calculate_reward", return_value=1.0),
            patch.object(env, "_get_current_market_regime", return_value=RegimeType.STRONG_BEAR),
        ):
            env.step(ACTION_HOLD)

        assert env.regime_stats["regime_counts"][RegimeType.STRONG_BULL] == 1
        assert env.regime_stats["regime_counts"][RegimeType.STRONG_BEAR] == 1
        assert len(env.regime_stats["regime_rewards"][RegimeType.STRONG_BULL]) == 1
        assert len(env.regime_stats["regime_actions"][RegimeType.STRONG_BEAR]) == 1
        transition = env.regime_stats["regime_transitions"][-1]
        assert transition["from"] == RegimeType.STRONG_BULL
        assert transition["to"] == RegimeType.STRONG_BEAR

    def test_step_info_includes_market_regime(
        self,
        sample_market_data: pd.DataFrame,
        mock_classifier: Mock,
    ) -> None:
        env = _make_env(sample_market_data)
        env.enable_market_regime_adaptation(mock_classifier, {"enabled": True})
        env.reset()

        with patch.object(env, "_get_current_market_regime", return_value=RegimeType.STRONG_BULL):
            _, _, _, _, info = env.step(ACTION_HOLD)

        assert "market_regime" in info
        assert info["market_regime"] == RegimeType.STRONG_BULL
        assert "reward_components" in info
        assert "trend_signal" in info

    def test_regime_adaptation_error_is_logged(
        self,
        caplog: pytest.LogCaptureFixture,
        sample_market_data: pd.DataFrame,
        mock_classifier: Mock,
    ) -> None:
        env = _make_env(sample_market_data)
        env.enable_market_regime_adaptation(mock_classifier, {"enabled": True})
        env.reset()
        mock_classifier.get_regime_multiplier.side_effect = RuntimeError("boom")

        with caplog.at_level(
            "WARNING", logger="ztb.trading.environment.heavy_env.core"
        ), patch.object(
            env.reward_calculator, "calculate_reward", return_value=4.0
        ), patch.object(
            env, "_get_current_market_regime", return_value=RegimeType.STRONG_BULL
        ):
            _, reward, _, _, _ = env.step(ACTION_HOLD)

        assert reward == pytest.approx(4.0)
        assert "Failed to apply regime adaptation: boom" in caplog.text

    def test_regime_statistics_structure_survives_reset(
        self,
        sample_market_data: pd.DataFrame,
        mock_classifier: Mock,
    ) -> None:
        env = _make_env(sample_market_data)
        env.enable_market_regime_adaptation(mock_classifier, {"enabled": True})
        env.reset()

        with (
            patch.object(env.reward_calculator, "calculate_reward", return_value=1.0),
            patch.object(env, "_get_current_market_regime", return_value=RegimeType.STRONG_BULL),
        ):
            env.step(ACTION_HOLD)

        env.reset()

        assert hasattr(env, "regime_statistics")
        assert isinstance(env.regime_stats["regime_transitions"], type(env.regime_stats["regime_transitions"]))
