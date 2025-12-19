import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_v453_results():
    results_path = 'backtest_results/v453_hybrid/backtest_results.csv'
    if not os.path.exists(results_path):
        print(f'File not found: {results_path}')
        return

    df = pd.read_csv(results_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Calculate step PnL (change in portfolio value)
    df['step_pnl'] = df['portfolio_value'].diff().fillna(0)
    
    # Calculate returns
    df['returns'] = df['price'].pct_change()
    
    # Calculate volatility (1-hour rolling std dev of returns)
    df['volatility'] = df['returns'].rolling(window=60).std()

    # Filter Analysis
    # 1. Time Filter Analysis
    df['hour'] = df.index.hour
    hourly_pnl = df.groupby('hour')['step_pnl'].sum()
    hourly_trades = df[df['action_type'] != 'HOLD'].groupby('hour')['action_type'].count()
    
    # 2. Volatility Filter Analysis
    # Define bins for volatility
    vol_bins = [0, 0.0005, 0.0015, 0.0025, 0.0050, 0.0100, 1.0]
    vol_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Extreme']
    df['vol_category'] = pd.cut(df['volatility'], bins=vol_bins, labels=vol_labels)
    
    vol_pnl = df.groupby('vol_category')['step_pnl'].sum()
    vol_count = df.groupby('vol_category')['action_type'].count()

    print('=== v453 Hybrid Strategy Analysis ===')
    total_pnl = df['portfolio_value'].iloc[-1] - df['portfolio_value'].iloc[0]
    print(f'Total PnL (from Portfolio Value): {total_pnl:.2f}')
    
    print('\n--- Hourly Performance (Check if 14, 17, 01 are avoided/low) ---')
    print('PnL per Hour:')
    print(hourly_pnl)
    print('\nTrades per Hour:')
    print(hourly_trades)

    print('\n--- Volatility Performance (Check \'Medium\' range 0.005-0.015) ---')
    # Config Danger Zone: 0.005 to 0.015
    
    df['in_danger_zone'] = (df['volatility'] >= 0.005) & (df['volatility'] <= 0.015)
    danger_zone_pnl = df[df['in_danger_zone']]['step_pnl'].sum()
    
    # Count trades in danger zone (where action is NOT HOLD)
    danger_zone_trades = df[df['in_danger_zone'] & (df['action_type'] != 'HOLD')].shape[0]
    
    print(f'PnL in Danger Zone (0.005-0.015): {danger_zone_pnl:.2f}')
    print(f'Trades in Danger Zone: {danger_zone_trades}')
    
    print('\n--- Volatility Distribution of PnL ---')
    print(vol_pnl)

    # Trade Statistics
    trades = df[df['action_type'] != 'HOLD']
    print(f'\nTotal Trades: {len(trades)}')
    
    # Save plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['portfolio_value'], label='Portfolio Value')
    plt.title('v453 Hybrid Strategy Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('backtest_results/v453_hybrid/equity_curve.png')
    print('\nEquity curve saved to backtest_results/v453_hybrid/equity_curve.png')

if __name__ == '__main__':
    analyze_v453_results()
