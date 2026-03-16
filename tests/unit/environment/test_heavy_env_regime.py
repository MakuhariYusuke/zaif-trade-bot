"""Quick test of HeavyTradingEnv market regime adaptation."""

from tests.helpers import make_schema_feature_env_config, make_trending_ohlcv_data
from ztb.analysis.regime.market_regime_classifier import MarketRegimeClassifier
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv


def test_env_regime_adaptation() -> None:
    market_data = make_trending_ohlcv_data(
        rows=64,
        seed=42,
        freq="1h",
        include_timestamp=True,
    )

    env = HeavyTradingEnv(
        df=market_data,
        config=make_schema_feature_env_config(
            market_data,
            initial_portfolio_value=10000.0,
            max_position_size=1.0,
            slippage=0.0005,
            transaction_cost=0.001,
        ),
    )

    classifier = MarketRegimeClassifier(
        {
            "adaptation": {
                "enabled": True,
                "regime_reward_multipliers": {"STRONG_BULL": 1.5},
            }
        }
    )

    env.enable_market_regime_adaptation(classifier)

    observation, reward, terminated, truncated, info = env.step(0)

    assert observation is not None
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert env.market_regime_adaptation_enabled is True
    assert env.regime_classifier is classifier
    assert "regime" in info or "market_regime" in info
