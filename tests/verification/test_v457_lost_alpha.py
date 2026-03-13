
import unittest
import pandas as pd
import numpy as np
import logging
from ztb.trading.environment.factory_v456 import EnvironmentFactory
from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456

# Setup logging
logging.basicConfig(level=logging.INFO)

class TestV457LostAlpha(unittest.TestCase):
    def setUp(self):
        # 1. Create dummy data
        n_steps = 1000
        dates = pd.date_range('2025-01-01', periods=n_steps, freq='1min', tz='UTC')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(n_steps)) # Random walk
        
        # Base cols placeholders
        base_cols = ["open", "high", "low", "close", "volume"] + \
                    ["sma_5", "sma_20", "sma_50", "ema_5", "ema_20", "ema_50", "rsi_14", "rsi_20", "atr_14", "atr_20", 
                     "bb_upper_20", "bb_lower_20", "bb_pct_b_20", "macd_line", "macd_signal", "adx_14", "plus_di_14", "minus_di_14",
                     "obv", "vpt", "sma_5_close_ratio", "atr_pct_close", "hl_ratio", "hml_ratio", "trend_direction"]
        
        # Fill required base cols
        data = {c: np.random.randn(n_steps) for c in base_cols}
        data['close'] = prices
        data['open'] = prices + np.random.randn(n_steps)*0.1
        data['high'] = prices + np.abs(np.random.randn(n_steps)*0.5)
        data['low'] = prices - np.abs(np.random.randn(n_steps)*0.5)
        data['volume'] = np.abs(np.random.randn(n_steps)*1000)
        
        self.df = pd.DataFrame(data, index=dates)
        self.df['timestamp'] = dates
        self.n_steps = n_steps
        self.dates = dates
        
    def test_mtf_resampling(self):
        """Test if MTF features are correctly resampled (not just 1m duplicates)"""
        print("\n--- Testing Factory MTF Generation ---")
        factory = EnvironmentFactory(self.df)
        df_prepared, feature_cols = factory.prepare_features()
        
        mtf_cols = feature_cols['mtf']
        self.assertEqual(len(mtf_cols), 27)
        
        # Check if MTF values are different (proof of resampling)
        # mtf_rsi_5m vs mtf_rsi_15m
        idx_5m = [col for col in mtf_cols if 'rsi_5m' in col][0] 
        idx_15m = [col for col in mtf_cols if 'rsi_15m' in col][0]
        
        rsi_5m = df_prepared[idx_5m].values
        rsi_15m = df_prepared[idx_15m].values
        
        # Ignore first 100 steps (warmup)
        diff = np.abs(rsi_5m[100:] - rsi_15m[100:]).mean()
        print(f"Mean Difference between RSI 5m and 15m: {diff:.4f}")
        
        # If resampling works, 5m RSI and 15m RSI should be significantly different
        self.assertGreater(diff, 1.0, "MTF features appear to be identical/duplicated (Resampling failed)")
        
    def test_cyclical_features(self):
        """Test if Cyclical features are populated in the Environment observation"""
        print("\n--- Testing Env Cyclical Features ---")
        factory = EnvironmentFactory(self.df)
        df_prepared, feature_cols = factory.prepare_features()
        
        env = FastIntradayEnvV456(
            df=df_prepared,
            base_feature_columns=feature_cols['base'],
            mtf_feature_columns=feature_cols['mtf'],
            regime_feature_columns=feature_cols['regime']
        )
        
        obs, _ = env.reset(seed=42)
        # obs structure: base(30) -> mtf(27) -> cyclical(6)
        cyclical_start = 30 + 27
        cyclical_end = cyclical_start + 6
        
        cyclical_slice = obs[cyclical_start:cyclical_end]
        print(f"Cyclical Features at step {env.current_step}: {cyclical_slice}")
        
        # Check if vectors are non-zero (checking absolute sum)
        self.assertGreater(np.sum(np.abs(cyclical_slice)), 0.01, "Cyclical features are all zeros!")
        
    def test_ichimoku_signals(self):
        """Test if Ichimoku signals are calculated in Env"""
        print("\n--- Testing Ichimoku Signal Calculation ---")
        factory = EnvironmentFactory(self.df)
        df_prepared, feature_cols = factory.prepare_features()
        
        env = FastIntradayEnvV456(
            df=df_prepared,
            base_feature_columns=feature_cols['base'],
            mtf_feature_columns=feature_cols['mtf'],
            regime_feature_columns=feature_cols['regime']
        )
        
        self.assertTrue(hasattr(env, "ichimoku_signals"), "Env missing ichimoku_signals attribute")
        
        signals = env.ichimoku_signals
        non_zeros = np.count_nonzero(signals)
        print(f"Ichimoku Signals Non-Zero Count: {non_zeros}/{len(signals)}")
        
        # With random walk, we expect some signals (1 or -1)
        self.assertGreater(non_zeros, 0, "No Ichimoku signals generated")

if __name__ == '__main__':
    unittest.main()
