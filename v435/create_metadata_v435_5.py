#!/usr/bin/env python3
"""
Create metadata.json for sac_v435.5
"""

import json
from datetime import datetime

# Feature names from config
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
    "ichimoku_tenkan",
    "ichimoku_kijun",
    "ichimoku_senkou_a",
    "ichimoku_senkou_b",
    "atr_14",
    "cci_14",
    "mfi_14",
    "roc_12",
    "mom_10",
    "price_change",
    "volume_change",
    "returns",
    "log_returns",
    "sma_5",
    "sma_10",
    "sma_20",
    "sma_50",
    "ema_5",
    "ema_10",
    "ema_20",
    "ema_50",
    "vwap",
    "price_volume_trend",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "atr_5",
    "atr_10",
    "atr_20",
    "bollinger_volatility",
    "close_to_bb_ratio",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "roc_5",
    "roc_10",
    "roc_20",
    "williams_r_5",
    "williams_r_10",
]

metadata = {
    "model_name": "sac_v435.5",
    "num_features": len(feature_names),
    "feature_names": feature_names,
    "schema_hash": "v435_5_micro_penalty_" + str(hash(str(feature_names))),
    "created_at": datetime.now().isoformat(),
    "training_config": {
        "model_name": "sac_v435.5",
        "version": "4.3.5.5",
        "frequency_penalty": 0.001,
    },
    "curated_features_spec": None,
    "feature_filtering_enabled": False,
    "feature_filter_mode": None,
}

with open("models/schemas/sac_v435.5/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Created metadata.json for sac_v435.5")
print(f"Features: {len(feature_names)}")
