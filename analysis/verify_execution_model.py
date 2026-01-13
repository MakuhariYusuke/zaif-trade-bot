import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from ztb.trading.types import MarketState

from ztb.trading.execution.pseudo_hft import PseudoHFTExecutionModel
from ztb.utils.data_utils import load_csv_data


def load_data(csv_path: Path) -> pd.DataFrame:
    df = load_csv_data(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Ensure ATR is present (simple calculation if missing)
    if "atr" not in df.columns:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=14).mean()

    # Convert Volume to BTC if it looks like JPY (Price > 1M, Volume > 1M usually)
    # Heuristic: If mean volume > 10000 and mean price > 100000, assume JPY volume.
    if df["volume"].mean() > 10000 and df["close"].mean() > 100000:
        print("Detected JPY Volume. Converting to BTC...")
        df["volume"] = df["volume"] / df["close"]

    return df


def run_verification():
    csv_path = project_root / "data" / "btc_jpy_1m_dataset.csv"
    if not csv_path.exists():
        print("Data not found.")
        return

    print("Loading data...")
    df = load_data(csv_path)
    print(f"Data loaded: {len(df)} rows")

    config = {
        "c_spread": 0.3,
        "c_vol": 0.2,
        "c_imp": 0.5,
        "gamma": 0.5,
        "min_volume": 0.01,
        "latency_sec": 1.0,
    }

    model = PseudoHFTExecutionModel(config)

    print("Simulating Execution...")

    slippages = []
    order_size = 0.01  # BTC

    for i in range(len(df)):
        row = df.iloc[i]
        if pd.isna(row["atr"]):
            continue

        market_data = row.to_dict()

        # Calculate One-Way Slippage
        slip = model.calculate_slippage_one_way(market_data, order_size)  # type: ignore[arg-type]
        slippages.append(slip)

    slippages = np.array(slippages)

    print("\nVerification Results (JPY/BTC):")
    print(f"Mean One-Way Slippage: {np.mean(slippages):.2f}")
    print(f"Mean Round-Trip Slippage: {np.mean(slippages) * 2:.2f}")
    print(f"Max One-Way Slippage: {np.max(slippages):.2f}")
    print(f"Min One-Way Slippage: {np.min(slippages):.2f}")

    # Breakdown
    # We can't easily breakdown inside the loop without modifying the class or duplicating logic.
    # But we can do a quick check on components for the mean case.
    mean_atr = df["atr"].mean()
    mean_spread = (df["high"] - df["low"]).mean()
    mean_vol = df["volume"].mean()

    print("\nComponent Analysis (Mean Data):")
    spread_comp = config["c_spread"] * mean_spread
    vol_comp = config["c_vol"] * mean_atr * (1.0 / 60.0) ** 0.5
    imp_comp = config["c_imp"] * mean_atr * ((order_size / max(mean_vol, 0.01)) ** 0.5)

    print(f"Spread Component: {spread_comp:.2f}")
    print(f"Volatility Component: {vol_comp:.2f}")
    print(f"Impact Component: {imp_comp:.2f}")
    print(f"Total Estimated: {spread_comp + vol_comp + imp_comp:.2f}")


if __name__ == "__main__":
    run_verification()
