# Fix for Windows DLL error
try:
    import torch
except ImportError:
    pass

import numpy as np
import pandas as pd

from ztb.features.unified_feature import UnifiedFeatureEngineer


def test_global_feature_integration():
    print("Testing Global Feature Integration...")

    # 1. Create Mock Data
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1min")

    # Main Exchange (Zaif) - Lags behind
    main_prices = np.sin(np.linspace(0, 10, 100)) * 100 + 50000
    # Shift main prices to simulate lag (e.g., 2 minutes lag)
    main_prices = np.roll(main_prices, 2)

    main_df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": main_prices,
            "high": main_prices + 10,
            "low": main_prices - 10,
            "close": main_prices,
            "volume": 1000,
        }
    )

    # External Exchange (Binance) - Leads
    ext_prices = np.sin(np.linspace(0, 10, 100)) * 100 + 50000

    ext_df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": ext_prices,
            "high": ext_prices + 10,
            "low": ext_prices - 10,
            "close": ext_prices,
            "volume": 5000,
        }
    )

    # 2. Initialize Engineer
    engineer = UnifiedFeatureEngineer()

    # 3. Generate Features with External Data
    print("Generating features with external data...")
    features = engineer.generate_features(
        main_df, model_type="sac", external_data=ext_df, external_suffix="_binance"
    )

    # 4. Verify Results
    print(f"Generated {len(features.columns)} features")

    # Check for global columns
    global_cols = [c for c in features.columns if "_binance" in c]
    print(f"Global columns found: {len(global_cols)}")
    print(global_cols[:5])

    # Check specific Lead-Lag features
    if "price_divergence_close_binance" in features.columns:
        print("✅ price_divergence_close_binance found")
    else:
        print("❌ price_divergence_close_binance NOT found")

    if "return_spread_close_binance" in features.columns:
        print("✅ return_spread_close_binance found")
    else:
        print("❌ return_spread_close_binance NOT found")

    # Check correlation (should be high since it's the same sine wave just shifted)
    # Note: rolling correlation needs window, so first few will be NaN
    corr_col = "corr_close_binance_15"
    if corr_col in features.columns:
        last_corr = features[corr_col].iloc[-1]
        print(f"Correlation (15m): {last_corr:.4f}")
        if last_corr > 0.8:
            print("✅ High correlation detected (as expected)")
        else:
            print(f"⚠️ Correlation might be low due to lag: {last_corr}")

    print("Test Complete")


if __name__ == "__main__":
    test_global_feature_integration()
