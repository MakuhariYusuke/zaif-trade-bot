import pandas as pd
import pytest

from tests.helpers.market_data import make_trending_ohlcv_data
from ztb.trading.environment import EnvironmentConfig, HeavyTradingEnv

if HeavyTradingEnv is None:
    pytest.skip(
        "HeavyTradingEnv not available (torch missing or import failed)",
        allow_module_level=True,
    )


def create_synthetic_df(rows: int = 200) -> pd.DataFrame:
    return make_trending_ohlcv_data(
        rows=rows,
        seed=7,
        start="2024-01-01",
        freq="5min",
        start_price=100.0,
        end_price=120.0,
        noise_scale=0.0,
        volume_low=1000.0,
        volume_high=1000.0,
        include_timestamp=True,
    )


@pytest.mark.integration
def test_trend_and_curriculum_info_present() -> None:
    df = create_synthetic_df(rows=96)
    env = HeavyTradingEnv(
        df=df,
        config=EnvironmentConfig.from_dict(
            {
                "random_start": False,
                "use_continuous_actions": False,
                "feature_set": "minimal",
                "curriculum_stage": "forced_balance",
                "curriculum_learning": {
                    "enabled": True,
                    "auto_progression": False,
                },
            }
        ),
    )

    _, reset_info = env.reset()
    assert "current_step" in reset_info

    for _ in range(3):
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)
        assert "trend_signal" in info
        assert "trend_detector_stats" in info
        assert "curriculum_stage" in info
        assert info["curriculum_stage"] == "forced_balance"
        if terminated or truncated:
            break
