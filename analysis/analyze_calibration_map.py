import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from ztb.trading.environment.constants import EPSILON
from ztb.trading.signal.calibration_map import CalibrationMap
from ztb.utils.data_utils import load_csv_data


def load_data(csv_path: Path) -> pd.DataFrame:
    df = load_csv_data(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df


def compute_regime(df: pd.DataFrame, window: int = 100) -> pd.Series:
    # Simple regime classification based on spec
    # Volatility Normalization
    returns = df["close"].pct_change()
    vol_raw = returns.rolling(window=20).std()

    # Rolling Median of Vol
    vol_median = vol_raw.rolling(window=window).median()
    vol_norm = vol_raw / (vol_median + EPSILON)

    # Trend Normalization
    # Use EMA(20) slope
    ema20 = df["close"].ewm(span=20).mean()
    trend_raw = ema20.diff()
    # Normalize by price * vol to get dimensionless trend strength relative to volatility
    trend_norm = trend_raw / (vol_raw * df["close"] + EPSILON)

    regimes = []
    for i in range(len(df)):
        v = vol_norm.iloc[i]
        t = trend_norm.iloc[i]

        if pd.isna(v) or pd.isna(t):
            regimes.append("Unknown")
            continue

        if v > 1.5:
            regimes.append("High_Volatility")
        elif t > 0.05:  # Threshold tuned for normalized trend
            regimes.append("Trend_Bull")
        elif t < -0.05:
            regimes.append("Trend_Bear")
        else:
            regimes.append("Ranging")

    return pd.Series(regimes, index=df.index)


def simulate_actions(df: pd.DataFrame) -> pd.Series:
    # Mock actions using RSI heuristic
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + EPSILON)
    rsi = 100 - (100 / (1 + rs))

    actions = []
    for r in rsi:
        if pd.isna(r):
            actions.append(0.0)
        elif r < 30:
            actions.append(0.8)  # Strong Buy
        elif r > 70:
            actions.append(-0.8)  # Strong Sell
        else:
            actions.append(0.0)  # Neutral

    return pd.Series(actions, index=df.index)


def run_analysis():
    csv_path = project_root / "data" / "btc_jpy_1m_dataset.csv"
    if not csv_path.exists():
        print(f"Data not found at {csv_path}")
        return

    print("Loading data...")
    df = load_data(csv_path)
    print(f"Data loaded: {len(df)} rows")

    print("Computing regimes...")
    df["regime"] = compute_regime(df)

    print("Simulating actions...")
    df["action"] = simulate_actions(df)

    # Initialize Map
    config = {"ewma_tau": 1000.0, "n_min": 30.0}  # Larger tau for longer history
    cmap = CalibrationMap(config)

    print("Populating Calibration Map...")

    future_steps = 5
    df["future_close"] = df["close"].shift(-future_steps)

    count = 0
    for i in range(len(df) - future_steps):
        row = df.iloc[i]
        action = row["action"]
        regime = row["regime"]

        if regime == "Unknown":
            continue
        if abs(action) < 0.1:
            continue

        side = 1 if action > 0 else -1
        pnl = (row["future_close"] - row["close"]) * side

        cmap.update(regime, action, pnl, i)
        count += 1

    print(f"Processed {count} trades.")

    # Visualize
    print("\nAnalysis Complete. Stats (Level 1):")

    regimes = ["Trend_Bull", "Trend_Bear", "Ranging", "High_Volatility"]
    actions = [0.8, -0.8]  # Check Strong Buy/Sell

    for r in regimes:
        print(f"\nRegime: {r}")
        for a in actions:
            stats = cmap.get_stats(r, a)
            l1 = stats["l1"]
            print(
                f"  Action {a}: WinRate(LCB)={l1['p_win_lcb']:.2f}, AvgWin={l1['avg_win']:.0f}, AvgLoss={l1['avg_loss']:.0f}, N_eff={l1['n_eff']:.1f}"
            )


if __name__ == "__main__":
    run_analysis()
