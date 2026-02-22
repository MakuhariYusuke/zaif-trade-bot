#!/usr/bin/env python3
"""
Analyze training data to verify SELL-lock fix
"""

from pathlib import Path

from ztb.io.data_loader import DataLoader
# プロジェクトルートを取得
project_root = Path(__file__).parent.parent.parent.parent

# Load training data
data_path = project_root / 'data' / 'quick_training_data.csv'
df = DataLoader.load_csv_strict(data_path)

print("=" * 80)
print("SELL-LOCK FIX VERIFICATION REPORT")
print("=" * 80)
print(f"\nTotal training steps: {len(df)}")
print(f"\nDataFrame columns: {df.columns.tolist()}")
print(f"\nDataFrame shape: {df.shape}")

# Display first few rows
print("\nFirst 20 rows:")
print(df.head(20).to_string())

# Analyze action distribution
if 'discrete_action' in df.columns:
    action_names = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
    action_counts = df['discrete_action'].value_counts().sort_index()
    total_actions = len(df)

    print("\n" + "=" * 80)
    print("ACTION DISTRIBUTION (CRITICAL FOR SELL-LOCK VERIFICATION)")
    print("=" * 80)
    for action_id, count in action_counts.items():
        action_name = action_names.get(action_id, f'Unknown({action_id})')
        percentage = (count / total_actions) * 100
        print(f"{action_name:8s}: {count:4d} steps ({percentage:5.1f}%)")

    print(f"\nTotal: {total_actions} steps")

    # Check SELL percentage
    sell_count = action_counts.get(-1, 0)
    sell_pct = (sell_count / total_actions) * 100

    print("\n" + "=" * 80)
    print("SELL-LOCK STATUS")
    print("=" * 80)
    if sell_pct > 80:
        print(f"🔴 FAIL: SELL at {sell_pct:.1f}% - SELL-LOCK STILL ACTIVE")
    elif sell_pct < 30:
        print(f"🟡 WARNING: SELL at {sell_pct:.1f}% - May indicate BUY/HOLD bias")
    else:
        print(f"✅ PASS: SELL at {sell_pct:.1f}% - SELL-LOCK FIXED!")

    # Expected distribution after fix: SELL ~30-40%, BUY ~30-40%, HOLD ~20-40%
    print("\nExpected after fix:")
    print("  SELL: ~30-40%")
    print("  BUY:  ~30-40%")
    print("  HOLD: ~20-40%")

# Analyze position changes
if 'position' in df.columns:
    print("\n" + "=" * 80)
    print("POSITION ANALYSIS")
    print("=" * 80)
    print(f"Min position (max short): {df['position'].min():.4f}")
    print(f"Max position (max long):  {df['position'].max():.4f}")
    print(f"Mean position: {df['position'].mean():.4f}")

    # Count position transitions
    df['position_change'] = df['position'].diff()
    increasing = (df['position_change'] > 0.0001).sum()
    decreasing = (df['position_change'] < -0.0001).sum()

    print(f"Position increases (BUY): {increasing}")
    print(f"Position decreases (SELL): {decreasing}")

# Analyze rewards
if 'reward' in df.columns:
    print("\n" + "=" * 80)
    print("REWARD ANALYSIS")
    print("=" * 80)
    print(f"Total reward: {df['reward'].sum():.2f}")
    print(f"Mean reward: {df['reward'].mean():.4f}")
    print(f"Median reward: {df['reward'].median():.4f}")
    print(f"Max reward: {df['reward'].max():.2f}")
    print(f"Min reward: {df['reward'].min():.2f}")

# Portfolio return
if 'portfolio_return' in df.columns:
    print("\n" + "=" * 80)
    print("PORTFOLIO PERFORMANCE")
    print("=" * 80)
    print(f"Final portfolio return: {df['portfolio_return'].iloc[-1]:.4f}%")
    print(f"Max portfolio return: {df['portfolio_return'].max():.4f}%")
    print(f"Min portfolio return: {df['portfolio_return'].min():.4f}%")

print("\n" + "=" * 80)
