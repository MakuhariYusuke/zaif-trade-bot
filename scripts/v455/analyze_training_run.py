import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def analyze_monitor_log(log_path):
    print(f"Analyzing log: {log_path}")
    
    # Read the monitor file (skip first line which is metadata)
    try:
        df = pd.read_csv(log_path, skiprows=1)
    except Exception as e:
        print(f"Error reading log file: {e}")
        return

    # Columns are usually r (reward), l (length), t (time)
    if 'r' not in df.columns or 'l' not in df.columns:
        print("Error: Unexpected columns in monitor.csv")
        print(df.head())
        return

    print("\n--- Training Summary ---")
    print(f"Total Episodes: {len(df)}")
    print(f"Total Steps: {df['l'].sum()}")
    print(f"Average Reward: {df['r'].mean():.2f}")
    print(f"Average Episode Length: {df['l'].mean():.2f}")
    print(f"Median Episode Length: {df['l'].median():.2f}")
    print(f"Min Episode Length: {df['l'].min()}")
    print(f"Max Episode Length: {df['l'].max()}")

    # Analyze "Instant Death" (short episodes with negative reward)
    # Assuming max_steps is 1000 (from train_hft.py)
    # If length is significantly shorter than max_steps, it likely hit a stop condition (drawdown or bankruptcy)
    
    short_episodes = df[df['l'] < 100]
    print(f"\n--- Instant Death Analysis (< 100 steps) ---")
    print(f"Count: {len(short_episodes)} ({len(short_episodes)/len(df)*100:.1f}%)")
    print(f"Avg Reward in Short Episodes: {short_episodes['r'].mean():.2f}")
    
    very_short = df[df['l'] < 20]
    print(f"\n--- Immediate Failure (< 20 steps) ---")
    print(f"Count: {len(very_short)} ({len(very_short)/len(df)*100:.1f}%)")
    
    # Rolling average of length to see if it improved
    df['l_rolling'] = df['l'].rolling(window=100).mean()
    
    print("\n--- Trend Analysis ---")
    print(f"First 100 Ep Avg Length: {df['l'].iloc[:100].mean():.2f}")
    print(f"Last 100 Ep Avg Length: {df['l'].iloc[-100:].mean():.2f}")
    
    if df['l'].iloc[-100:].mean() < 50:
        print("\n[CRITICAL] Agent is consistently failing early even at the end of training.")
    else:
        print("\n[INFO] Agent seems to be surviving longer towards the end.")

if __name__ == "__main__":
    log_file = "logs/v455_hft_reward_tuned/monitor.csv"
    if os.path.exists(log_file):
        analyze_monitor_log(log_file)
    else:
        print(f"File not found: {log_file}")
