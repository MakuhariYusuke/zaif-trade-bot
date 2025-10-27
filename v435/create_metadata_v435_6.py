#!/usr/bin/env python3
"""
Create metadata.json for SAC v435.6 ensemble model.
"""

import json
from pathlib import Path


def create_metadata_v435_6():
    """Create metadata for SAC v435.6 ensemble model."""

    # Feature names for ensemble model (same as v435.5 but for ensemble)
    feature_names = [
        # Price features
        "close",
        "high",
        "low",
        "open",
        "volume",
        # Technical indicators
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
        # Moving averages
        "sma_5",
        "sma_10",
        "sma_20",
        "sma_50",
        "ema_5",
        "ema_10",
        "ema_20",
        "ema_50",
        # Volatility indicators
        "atr_14",
        "std_20",
        "volatility_20",
        # Momentum indicators
        "roc_10",
        "mom_10",
        "trix_15",
        # Volume indicators
        "volume_sma_20",
        "volume_ratio",
        "obv",
        # Ichimoku Cloud
        "tenkan_sen",
        "kijun_sen",
        "senkou_span_a",
        "senkou_span_b",
        "chikou_span",
        "ichimoku_cloud_top",
        "ichimoku_cloud_bottom",
        "ichimoku_cloud_thickness",
        # Advanced Ichimoku features
        "ichimoku_tenkan_kijun_cross",
        "ichimoku_price_cloud_relation",
        "ichimoku_span_cross",
        "ichimoku_cloud_breakout",
        # Multi-timeframe Ichimoku
        "ichimoku_1h_tenkan",
        "ichimoku_1h_kijun",
        "ichimoku_1h_span_a",
        "ichimoku_1h_span_b",
        "ichimoku_4h_tenkan",
        "ichimoku_4h_kijun",
        "ichimoku_4h_span_a",
        "ichimoku_4h_span_b",
        "ichimoku_1d_tenkan",
        "ichimoku_1d_kijun",
        "ichimoku_1d_span_a",
        "ichimoku_1d_span_b",
        # Ta-Lib enhanced indicators
        "adx_14",
        "cci_14",
        "mfi_14",
        "ultimate_oscillator",
        "stoch_rsi_k",
        "stoch_rsi_d",
        # High priority indicators
        "vwap",
        "pivot_point",
        "support_resistance_distance",
        "trend_strength",
        # Time features
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
    ]

    # Metadata structure
    metadata = {
        "model_name": "sac_v435.6",
        "version": "4.3.5.6",
        "description": "SAC v435.6 with ensemble majority voting system",
        "algorithm": "SAC",
        "features": {
            "count": len(feature_names),
            "names": feature_names,
            "scaling": {"method": "standard", "feature_means": {}, "feature_stds": {}},
        },
        "ensemble": {
            "models": ["sac_v435.3", "sac_v435.4", "sac_v435.5"],
            "voting_method": "majority",
            "consensus_threshold": 0.6,
            "diversity_penalty": 0.1,
        },
        "reward_function": {
            "base_profit_bonus": 5.0,
            "action_penalty": 0.15,
            "frequency_penalty": 0.001,
            "consensus_bonus": 2.0,
            "diversity_penalty": 0.1,
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
    models_dir = Path("models/schemas/sac_v435.6")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata_path = models_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Created metadata.json for sac_v435.6")
    print(f"Features: {len(feature_names)}")


if __name__ == "__main__":
    create_metadata_v435_6()
