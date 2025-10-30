#!/usr/bin/env python3
"""
Simple backtest for SAC v435.5 model
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.utils.analysis_formatters import print_formatted_metrics


def main():
    print("🚀 SAC v435.5 Backtest")
    print("=" * 40)

    # Load model
    model_path = "models/sac_v435.5.zip"
    try:
        model = SAC.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Load metadata to get feature columns
    metadata_path = "models/schemas/sac_v435.5/metadata.json"
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        feature_columns = metadata["feature_names"]
        obs_dim = (
            len(feature_columns) + 3
        )  # features + balance + position + unrealized_pnl
        print(
            f"✅ Metadata loaded: {len(feature_columns)} features, observation space: {obs_dim}"
        )
    except Exception as e:
        print(f"❌ Failed to load metadata: {e}")
        return

    # Load data
    data_path = "data/btc_jpy_real_dataset.csv"
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded: {len(df)} rows")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Simple evaluation
    print("📊 Running simple evaluation...")

    # Initialize variables
    balance = 10000.0
    position = 0.0
    trades = 0
    wins = 0

    # Simple feature engineering (minimal - just for testing)
    if "close" in df.columns:
        # Calculate basic features that match metadata
        df["returns"] = df["close"].pct_change()
        df["sma_5"] = df["close"].rolling(5).mean()
        df["sma_20"] = df["close"].rolling(20).mean()
        df["rsi_14"] = 50  # Placeholder
        df["macd"] = 0  # Placeholder
        df["macd_signal"] = 0  # Placeholder
        df["macd_hist"] = 0  # Placeholder
        df["bb_upper"] = df["close"] * 1.02  # Placeholder
        df["bb_middle"] = df["close"]  # Placeholder
        df["bb_lower"] = df["close"] * 0.98  # Placeholder
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df[
            "bb_middle"
        ]  # Placeholder
        df["stoch_k"] = 50  # Placeholder
        df["stoch_d"] = 50  # Placeholder
        df["williams_r"] = -50  # Placeholder
        df["ichimoku_tenkan"] = df["close"]  # Placeholder
        df["ichimoku_kijun"] = df["close"]  # Placeholder
        df["ichimoku_senkou_a"] = df["close"]  # Placeholder
        df["ichimoku_senkou_b"] = df["close"]  # Placeholder
        df["atr_14"] = df["close"] * 0.01  # Placeholder
        df["cci_14"] = 0  # Placeholder
        df["mfi_14"] = 50  # Placeholder
        df["roc_12"] = 0  # Placeholder
        df["mom_10"] = 0  # Placeholder
        df["price_change"] = df["close"].pct_change()  # Placeholder
        df["volume_change"] = 0  # Placeholder (no volume data)
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))  # Placeholder
        df["sma_10"] = df["close"].rolling(10).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["ema_5"] = df["close"].ewm(span=5).mean()
        df["ema_10"] = df["close"].ewm(span=10).mean()
        df["ema_20"] = df["close"].ewm(span=20).mean()
        df["ema_50"] = df["close"].ewm(span=50).mean()
        df["vwap"] = df["close"]  # Placeholder (no volume data)
        df["price_volume_trend"] = 0  # Placeholder
        df["volatility_5"] = df["close"].rolling(5).std()
        df["volatility_10"] = df["close"].rolling(10).std()
        df["volatility_20"] = df["close"].rolling(20).std()
        df["atr_5"] = df["close"] * 0.005  # Placeholder
        df["atr_10"] = df["close"] * 0.007  # Placeholder
        df["atr_20"] = df["close"] * 0.01  # Placeholder
        df["bollinger_volatility"] = df["bb_width"]  # Placeholder
        df["close_to_bb_ratio"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )  # Placeholder
        df["momentum_5"] = df["close"] / df["close"].shift(5) - 1  # Placeholder
        df["momentum_10"] = df["close"] / df["close"].shift(10) - 1  # Placeholder
        df["momentum_20"] = df["close"] / df["close"].shift(20) - 1  # Placeholder
        df["roc_5"] = df["close"].pct_change(5)  # Placeholder
        df["roc_10"] = df["close"].pct_change(10)  # Placeholder
        df["roc_20"] = df["close"].pct_change(20)  # Placeholder
        df["williams_r_5"] = -50  # Placeholder
        df["williams_r_10"] = -50  # Placeholder
        df = df.dropna()

    # Run through data
    for i in range(len(df)):
        if i < 50:  # Skip initial data for sufficient lookback
            continue

        # Create observation with 3 features (matching trained model)
        # Based on typical trading environments: normalized_price, position, balance
        obs = np.array(
            [
                df.iloc[i]["close"] / 1000000,  # Normalized price (same as before)
                position,  # Current position
                balance / 10000,  # Normalized balance
            ],
            dtype=np.float32,
        )

        # Get action
        try:
            action, _ = model.predict(obs, deterministic=True)
            new_position = float(action[0])

            # Debug output for first few steps
            if i < 55:
                print(
                    f"Step {i}: obs={obs}, action={action[0]:.4f}, new_position={new_position:.4f}"
                )

            # Execute trade (simplified)
            if abs(new_position - position) > 0.1:  # Threshold
                price = df.iloc[i]["close"]
                if position == 0 and new_position > 0:  # Open long
                    position = new_position
                    trades += 1
                    print(
                        f"Opened long position at {price:.2f}, size: {new_position:.4f}"
                    )
                elif position > 0 and new_position == 0:  # Close long
                    pnl = position * (price - df.iloc[i - 1]["close"])
                    balance += pnl
                    if pnl > 0:
                        wins += 1
                    print(f"Closed long position at {price:.2f}, PnL: {pnl:.2f}")
                    position = 0

        except Exception as e:
            print(f"Warning: Prediction failed at step {i}: {e}")
            continue

    # Calculate results
    total_return = (balance - 10000) / 10000 * 100
    win_rate = wins / trades * 100 if trades > 0 else 0

    results = {
        "model": "sac_v435.5",
        "total_return_pct": total_return,
        "total_trades": trades,
        "win_rate_pct": win_rate,
        "final_balance": balance,
    }

    print_formatted_metrics(results, "SAC v435.5 Backtest Results")

    # Save results
    with open("results/sac_v435_5_backtest.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Results saved to results/sac_v435_5_backtest.json")


if __name__ == "__main__":
    main()
