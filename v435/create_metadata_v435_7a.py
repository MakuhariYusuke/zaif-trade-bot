#!/usr/bin/env python3
"""
Create metadata.json for SAC v435.7a model.
"""

import json
from pathlib import Path


def create_metadata_v435_7a():
    """Create metadata for SAC v435.7a model."""

    # Feature names for v435.7a
    feature_names = [
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_width",
        "stoch_k",
        "stoch_d",
        "williams_r",
        "sma_5",
        "sma_10",
        "sma_20",
        "sma_50",
        "ema_5",
        "ema_10",
        "ema_20",
        "ema_50",
        "atr_14",
        "cci_14",
        "mfi_14",
        "roc_12",
        "mom_10",
        "vwap",
        "price_volume_trend",
        "volatility_20",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
    ]

    # Metadata structure
    metadata = {
        "model_name": "sac_v435.7a",
        "version": "4.3.5.7a",
        "description": "SAC v435.7a with ultra-micro frequency penalty and symmetric thresholds",
        "algorithm": "SAC",
        "features": {
            "count": len(feature_names),
            "names": feature_names,
            "scaling": {"method": "standard", "feature_means": {}, "feature_stds": {}},
        },
        "symmetric_thresholds": {
            "enabled": True,
            "buy_threshold": -0.3333,
            "sell_threshold": 0.3333,
            "hold_range": [-0.3333, 0.3333],
        },
        "reward_function": {
            "base_profit_bonus": 5.0,
            "action_penalty": 0.15,
            "frequency_penalty": 0.0001,
            "symmetric_rewards": True,
        },
        "hyperparameters": {
            "learning_rate": 3e-4,
            "batch_size": 256,
            "buffer_size": 1000000,
            "learning_starts": 1000,
            "tau": 0.005,
            "gamma": 0.99,
            "ent_coef": "auto_1.0",
        },
    }

    # Create directory if it doesn't exist
    models_dir = Path("models/schemas/sac_v435.7a")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata_path = models_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Created metadata.json for sac_v435.7a")
    print(f"Features: {len(feature_names)}")
    print(f"Symmetric thresholds: {metadata['symmetric_thresholds']['enabled']}")


if __name__ == "__main__":
    create_metadata_v435_7a()
