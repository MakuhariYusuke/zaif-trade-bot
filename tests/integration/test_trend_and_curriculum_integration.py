import numpy as np
import pandas as pd

import pytest
from ztb.trading.environment import HeavyTradingEnv

if HeavyTradingEnv is None:
    pytest.skip("HeavyTradingEnv not available (torch missing or import failed)", allow_module_level=True)


def create_synthetic_df(rows=200):
    rng = np.random.default_rng(42)
    price_trend = np.linspace(100, 110, rows) + rng.normal(0, 0.5, rows)
    return pd.DataFrame(
        {
            "open": price_trend + rng.normal(0, 0.1, rows),
            "high": price_trend + rng.normal(0, 0.2, rows),
            "low": price_trend - rng.normal(0, 0.2, rows),
            "close": price_trend + rng.normal(0, 0.05, rows),
            "volume": rng.normal(1000, 50, rows),
        }
    )


def test_trend_and_curriculum_info_present():
    df = create_synthetic_df(rows=200)
    env = HeavyTradingEnv(
        df=df,
        config={
            "feature_set": "minimal",
            "curriculum_stage": "forced_balance",
            "curriculum_learning": {"enabled": True, "auto_progression": False},
        },
    )

    obs, info = env.reset()
    assert "trend_signal" in info
    assert "trend_detector_stats" in info
    assert "curriculum_stage" in info
    assert info["curriculum_stage"] == "forced_balance"

    # take a few steps and check info is present in step outputs
    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert "trend_signal" in info
        assert "curriculum_stage" in info
