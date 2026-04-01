from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import ztb.features.scalping  # noqa: F401
from tests.unit.v460._yaml_test_helpers import load_yaml_mapping
from ztb.features.core.registry import FeatureRegistry
from ztb.features.scalping import compute_signed_obi_values
from ztb.trading.constants import ACTION_BUY, ACTION_SELL
from ztb.trading.environment.components.calculators.reward_calculator import RewardCalculator
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings


def _make_calculator(
    *,
    reward_scaling: float = 1.0,
    reward_clip_value: float = 10.0,
    sell_as_penalty_mult: float = 1.5,
) -> RewardCalculator:
    config = EnvironmentConfig()
    config.dynamic_reward_shaping = {"enabled": False}
    config.signal_guidance_enabled = False
    reward_settings = RewardSettings(
        reward_scaling=reward_scaling,
        reward_clip_value=reward_clip_value,
        sell_as_penalty_mult=sell_as_penalty_mult,
    )
    calculator = RewardCalculator(
        config,
        reward_settings,
        initial_portfolio_value=10_000.0,
    )
    calculator.dynamic_reward_shaper.enabled = False
    calculator.signal_integrator.enabled = False
    calculator.asymmetric_reward_scaler.enabled = False
    return calculator


class TestSellAwareReward:
    def test_sell_as_penalty_applied(self) -> None:
        calculator = _make_calculator(reward_scaling=2.0, reward_clip_value=100.0)
        reward = calculator.calculate_reward_simple(
            pnl=-2.0,
            action=ACTION_SELL,
            portfolio_value=10_000.0,
            adverse_selected=True,
        )
        assert reward == pytest.approx(-6.0)

    def test_buy_as_penalty_not_applied(self) -> None:
        calculator = _make_calculator(reward_scaling=2.0, reward_clip_value=100.0)
        reward = calculator.calculate_reward_simple(
            pnl=-2.0,
            action=ACTION_BUY,
            portfolio_value=10_000.0,
            adverse_selected=True,
        )
        assert reward == pytest.approx(-4.0)

    def test_sell_no_as_no_penalty(self) -> None:
        calculator = _make_calculator(reward_scaling=2.0, reward_clip_value=100.0)
        reward = calculator.calculate_reward_simple(
            pnl=-2.0,
            action=ACTION_SELL,
            portfolio_value=10_000.0,
            adverse_selected=False,
        )
        assert reward == pytest.approx(-4.0)

    def test_penalty_before_clip(self) -> None:
        calculator = _make_calculator(reward_scaling=1.0, reward_clip_value=10.0)
        reward = calculator.calculate_reward_simple(
            pnl=-8.0,
            action=ACTION_SELL,
            portfolio_value=10_000.0,
            adverse_selected=True,
        )
        assert reward == pytest.approx(-10.0)


class TestNewFeatures:
    def test_mid_price_trend_5s_registered(self) -> None:
        assert "mid_price_trend_5s" in FeatureRegistry.list()

    def test_signed_obi_registered(self) -> None:
        assert "signed_obi" in FeatureRegistry.list()

    def test_signed_obi_sell_positive_obi(self) -> None:
        signed = compute_signed_obi_values(
            np.array([0.4], dtype=np.float64),
            np.array([-1.0], dtype=np.float64),
        )
        assert signed[0] < 0.0

    def test_signed_obi_buy_positive_obi(self) -> None:
        signed = compute_signed_obi_values(
            np.array([0.4], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        )
        assert signed[0] > 0.0

    def test_sac_train_yaml_has_new_features(self) -> None:
        cfg = load_yaml_mapping(Path("configs/v460/experiments/g2_sac_train.yaml"))
        selected = cfg["features"]["selected"]
        assert "mid_price_trend_5s" in selected
        assert "signed_obi" in selected
        assert cfg["environment"]["reward_settings"]["sell_as_penalty_mult"] == pytest.approx(1.5)

