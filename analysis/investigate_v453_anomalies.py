import pandas as pd
import os

def investigate_anomalies():
    print("=== Starting Deep Dive Investigation for v453 ===")
    
    # 1. Load Data
    data_path = 'data/btc_jpy_1m_merged.csv'
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return
    
    print(f"Loading market data from {data_path}...")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Use last 20% to match backtest
    start_idx = int(len(df) * 0.8)
    df = df.iloc[start_idx:]
    print(f"Analyzing last 20% of data ({len(df)} rows)")

    # Calculate Volatility (1h rolling std of returns)
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=60).std()
    
    # Calculate Volume MA for relative volume
    if 'volume' in df.columns:
        df['vol_ma'] = df['volume'].rolling(window=60).mean()
        df['rel_volume'] = df['volume'] / df['vol_ma']

    # 2. Hourly Analysis (Focus on 14, 17)
    df['hour'] = df.index.hour
    
    hourly_stats = df.groupby('hour').agg({
        'volatility': ['mean', 'std', 'max'],
        'volume': ['mean', 'std'] if 'volume' in df.columns else [],
        'returns': lambda x: x.abs().mean() # Mean absolute return (activity)
    })
    
    print("\n--- Hourly Market Statistics ---")
    # Highlight 14 and 17
    print(hourly_stats.loc[[1, 14, 17]])
    print("\nComparison with Average:")
    print(hourly_stats.mean())

    # 3. Regime Analysis from Backtest Results
    results_path = 'backtest_results/v453_hybrid/backtest_results.csv'
    if os.path.exists(results_path):
        print(f"\nLoading backtest results from {results_path}...")
        res_df = pd.read_csv(results_path)
        res_df['timestamp'] = pd.to_datetime(res_df['timestamp'])
        res_df['hour'] = res_df['timestamp'].dt.hour
        
        # Check Regime Stability per Hour
        # Count how many times regime changes within an hour
        res_df['regime_shift'] = res_df['regime'] != res_df['regime'].shift(1)
        
        hourly_regime_shifts = res_df.groupby('hour')['regime_shift'].sum()
        hourly_counts = res_df.groupby('hour')['regime'].count()
        shift_ratio = hourly_regime_shifts / hourly_counts
        
        print("\n--- Hourly Regime Instability (Shift Ratio) ---")
        print("Higher ratio means regime flickers more often.")
        print(shift_ratio.sort_values(ascending=False).head(5))
        print(f"\nRatio for Hour 14: {shift_ratio.get(14, 0):.4f}")
        print(f"Ratio for Hour 17: {shift_ratio.get(17, 0):.4f}")
        
        # Check Regime Distribution for Hour 14/17
        print("\n--- Regime Distribution in Hour 14 ---")
        print(res_df[res_df['hour'] == 14]['regime'].value_counts(normalize=True))
        
        print("\n--- Regime Distribution in Hour 17 ---")
        print(res_df[res_df['hour'] == 17]['regime'].value_counts(normalize=True))

    else:
        print("Backtest results not found, skipping regime analysis.")

if __name__ == "__main__":
    investigate_anomalies()
