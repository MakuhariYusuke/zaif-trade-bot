import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_hft_potential(trades_file):
    print(f"Analyzing {trades_file}...")
    df = pd.read_csv(trades_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    trades = []
    current_trade = None
    
    for index, row in df.iterrows():
        if row['type'] == 'BUY' or row['type'] == 'SELL':
            if current_trade is None:
                current_trade = {
                    'entry_time': row['timestamp'],
                    'entry_price': row['price'],
                    'type': row['type'],
                    'entry_step': row['step']
                }
        elif row['type'] == 'CLOSE':
            if current_trade is not None:
                current_trade['exit_time'] = row['timestamp']
                current_trade['exit_price'] = row['price']
                current_trade['exit_step'] = row['step']
                
                # Calculate duration
                duration = current_trade['exit_time'] - current_trade['entry_time']
                current_trade['duration_minutes'] = duration.total_seconds() / 60
                
                # Calculate PnL
                if current_trade['type'] == 'BUY':
                    pnl = (current_trade['exit_price'] - current_trade['entry_price']) / current_trade['entry_price']
                else:
                    pnl = (current_trade['entry_price'] - current_trade['exit_price']) / current_trade['entry_price']
                
                current_trade['pnl'] = pnl
                trades.append(current_trade)
                current_trade = None

    trades_df = pd.DataFrame(trades)
    
    if trades_df.empty:
        print("No completed trades found.")
        return

    print("\n=== Trade Analysis ===")
    print(f"Total Trades: {len(trades_df)}")
    print(f"Win Rate: {(trades_df['pnl'] > 0).mean() * 100:.2f}%")
    print(f"Average PnL per Trade: {trades_df['pnl'].mean() * 100:.4f}%")
    print(f"Average Holding Time: {trades_df['duration_minutes'].mean():.2f} minutes")
    print(f"Median Holding Time: {trades_df['duration_minutes'].median():.2f} minutes")
    print(f"Min Holding Time: {trades_df['duration_minutes'].min():.2f} minutes")
    print(f"Max Holding Time: {trades_df['duration_minutes'].max():.2f} minutes")
    
    # Calculate time between trades
    trades_df['entry_time_diff'] = trades_df['entry_time'].diff().dt.total_seconds() / 60  # type: ignore[attr-defined]
    print(f"\nAverage Time Between Trades: {trades_df['entry_time_diff'].mean():.2f} minutes")
    print(f"Median Time Between Trades: {trades_df['entry_time_diff'].median():.2f} minutes")

    # HFT Suitability Assessment
    print("\n=== HFT Suitability Assessment ===")
    if trades_df['duration_minutes'].median() < 5:
        print("✅ Holding time is suitable for HFT (< 5 mins).")
    else:
        print(f"❌ Holding time is too long for HFT ({trades_df['duration_minutes'].median():.2f} mins).")
        
    if len(trades_df) > 1000: # Arbitrary threshold for "High Frequency" in this context
        print("✅ Trade frequency is high.")
    else:
        print(f"❌ Trade frequency is too low ({len(trades_df)} trades).")

if __name__ == "__main__":
    analyze_hft_potential("backtest_trades_v455.csv")
