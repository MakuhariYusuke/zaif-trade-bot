import numpy as np
import pandas as pd

from backtest.signal_guidance_backtest import SignalGuidanceBacktestEnv

mock_df = pd.DataFrame(
    {
        "close": [50000, 50100, 49900, 50200],
        "high": [50100, 50200, 50000, 50300],
        "low": [49900, 50000, 49800, 50100],
        "volume": [100, 110, 90, 120],
    }
)
config = {
    "transaction_cost": 0.001,
    "max_position_size": 0.1,
    "feature_names": list(mock_df.columns),
    "reward_scaling": 1.0,
    "max_steps": 100,
}

env = SignalGuidanceBacktestEnv(mock_df, config)

# Test the case that failed: observation = [1.0, -1.0, 0.1]
obs = np.array([1.0, -1.0, 0.1])
action, score = env._get_signal_guidance_score(obs, 0.0)

print("Action:", action)
print("Score:", score)

# Also print extracted signals
signals = env._extract_technical_signals(obs)
print("Signals:", signals)
print("BB:", env._calculate_bollinger_score_simple(signals))
print("Supertrend:", env._calculate_supertrend_score_simple(signals))
print("OBV:", env._calculate_obv_score_simple(signals))
print("MA:", env._calculate_ma_cross_score_simple(signals))

# Print the calculation components as well (weights)
weights = {"bollinger": 0.25, "supertrend": 0.5, "obv": 0.15, "ma_cross": 0.1}
bb = env._calculate_bollinger_score_simple(signals)
st = env._calculate_supertrend_score_simple(signals)
obv = env._calculate_obv_score_simple(signals)
ma = env._calculate_ma_cross_score_simple(signals)

print(
    "Total calc:",
    bb * weights["bollinger"]
    + st * weights["supertrend"]
    + obv * weights["obv"]
    + ma * weights["ma_cross"],
)
