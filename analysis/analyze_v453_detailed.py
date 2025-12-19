import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def analyze_performance(df):
    initial_value = df['portfolio_value'].iloc[0]
    final_value = df['portfolio_value'].iloc[-1]
    total_return = (final_value - initial_value) / initial_value
    
    # Calculate daily returns for Sharpe
    df['returns'] = df['portfolio_value'].pct_change().fillna(0)
    sharpe_ratio = df['returns'].mean() / df['returns'].std() * np.sqrt(24 * 60 * 365) # Assuming 1m data
    
    max_drawdown = (df['portfolio_value'] / df['portfolio_value'].cummax() - 1).min()
    
    print("=== Performance Summary ===")
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Final Portfolio Value: {final_value:,.0f}")
    print("-" * 30)

def analyze_regimes(df):
    print("\n=== Regime Analysis ===")
    
    # Calculate step PnL from portfolio value change
    df['step_pnl'] = df['portfolio_value'].diff().fillna(0)
    
    # Group by regime
    regime_stats = df.groupby('regime').agg({
        'step_pnl': 'sum',
        'filter_active': 'mean',
        'blocked_entry': 'sum',
        'effective_action': lambda x: (x != 0).sum()
    }).sort_values('step_pnl', ascending=False)
    
    regime_stats['filter_active_pct'] = regime_stats['filter_active'] * 100
    
    print(regime_stats[['step_pnl', 'effective_action', 'blocked_entry', 'filter_active_pct']])
    
    # Plot PnL by Regime
    plt.figure(figsize=(12, 6))
    regime_stats['step_pnl'].plot(kind='bar')
    plt.title('PnL by Market Regime')
    plt.ylabel('Total PnL')
    plt.tight_layout()
    plt.savefig('analysis_results/v453_pnl_by_regime.png')
    print("\nSaved regime PnL plot to analysis_results/v453_pnl_by_regime.png")

def analyze_trades(df):
    print("\n=== Trade Analysis ===")
    
    # Use step_pnl for trade analysis
    if 'step_pnl' not in df.columns:
        df['step_pnl'] = df['portfolio_value'].diff().fillna(0)
    
    # Filter out small noise (floating point errors)
    trade_steps = df[abs(df['step_pnl']) > 1e-5]
    
    win_trades = trade_steps[trade_steps['step_pnl'] > 0]
    loss_trades = trade_steps[trade_steps['step_pnl'] < 0]
    
    n_wins = len(win_trades)
    n_losses = len(loss_trades)
    total_trades = n_wins + n_losses
    
    if total_trades > 0:
        win_rate = n_wins / total_trades
        avg_win = win_trades['step_pnl'].mean()
        avg_loss = loss_trades['step_pnl'].mean()
        profit_factor = abs(win_trades['step_pnl'].sum() / loss_trades['step_pnl'].sum()) if n_losses > 0 else float('inf')
        
        print(f"Total PnL Events: {total_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Avg Win: {avg_win:.2f}")
        print(f"Avg Loss: {avg_loss:.2f}")
        print(f"Profit Factor (based on step PnL): {profit_factor:.2f}")
    else:
        print("No trades found.")

def analyze_filters(df):
    print("\n=== Filter Analysis ===")
    
    blocked = df[df['blocked_entry'] == True]
    print(f"Total Blocked Entries: {len(blocked)}")
    
    if len(blocked) > 0:
        print("\nBlocked Entries by Regime:")
        print(blocked['regime'].value_counts().head())
        
        # Check what the model wanted to do
        # attempted_discrete_action might show what was blocked
        print("\nAttempted Actions in Blocked Entries:")
        print(blocked['attempted_discrete_action'].value_counts())

def main():
    results_path = "backtest_results/v453_hybrid_final_solution/backtest_results.csv"
    if not Path(results_path).exists():
        print(f"File not found: {results_path}")
        return
        
    df = load_data(results_path)
    
    analyze_performance(df)
    analyze_regimes(df)
    analyze_trades(df)
    analyze_filters(df)

if __name__ == "__main__":
    main()
