import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_learning_curve():
    LOG_DIR = "logs/v455_hft_main"
    OUTPUT_DIR = "docs/v455/plots"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    monitor_path = os.path.join(LOG_DIR, "monitor.csv")
    if not os.path.exists(monitor_path):
        print(f"Monitor file not found: {monitor_path}")
        return
        
    # Read monitor.csv (skip first 2 lines usually)
    # SB3 monitor files have 2 header lines.
    df = pd.read_csv(monitor_path, skiprows=1)
    
    # Columns: r (reward), l (length), t (time), ... custom metrics
    # Custom metrics added: edge_shortfall, vol_ratio, trade_cost, balance, drawdown
    
    # Rolling means
    window = 50
    df["r_roll"] = df["r"].rolling(window).mean()
    df["l_roll"] = df["l"].rolling(window).mean()
    
    if "balance" in df.columns:
        df["balance_roll"] = df["balance"].rolling(window).mean()
    
    if "trade_cost" in df.columns:
        df["cost_roll"] = df["trade_cost"].rolling(window).mean()
        
    if "edge_shortfall" in df.columns:
        df["edge_roll"] = df["edge_shortfall"].rolling(window).mean()

    # Plot 1: Reward & Balance
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward", color="tab:blue")
    ax1.plot(df["r_roll"], color="tab:blue", label="Reward (MA50)")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    
    if "balance" in df.columns:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Final Balance", color="tab:green")
        ax2.plot(df["balance_roll"], color="tab:green", label="Balance (MA50)")
        ax2.tick_params(axis="y", labelcolor="tab:green")
        
    plt.title("Learning Curve: Reward & Balance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "learning_curve_reward_balance.png"))
    
    # Plot 2: Cost & Edge Shortfall
    plt.figure(figsize=(12, 6))
    if "trade_cost" in df.columns:
        plt.plot(df["cost_roll"], label="Trade Cost (MA50)")
    if "edge_shortfall" in df.columns:
        plt.plot(df["edge_roll"], label="Edge Shortfall (MA50)")
        
    plt.title("Cost Analysis")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "learning_curve_costs.png"))
    
    # Plot 3: Episode Length
    plt.figure(figsize=(12, 6))
    plt.plot(df["l_roll"], label="Episode Length (MA50)", color="purple")
    plt.title("Episode Survival Length")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "learning_curve_length.png"))
    
    print(f"Plots saved to {OUTPUT_DIR}")
    
    # Print Stats
    print("Final 50 Episodes Stats:")
    print(df.tail(50).mean())

if __name__ == "__main__":
    analyze_learning_curve()
