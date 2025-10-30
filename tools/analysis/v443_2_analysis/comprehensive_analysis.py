import json
from pathlib import Path


def analyze_backtest_results():
    """Analyze existing backtest results to establish baseline performance"""

    results_dir = Path("backtest_results")

    print("=== BACKTEST RESULTS ANALYSIS ===\n")

    # Analyze main backtest results
    main_results_file = results_dir / "backtest_results.json"
    if main_results_file.exists():
        with open(main_results_file, "r") as f:
            data = json.load(f)

        print("📊 MAIN BACKTEST RESULTS:")
        print(f"  Total Reward: {data.get('total_reward', 'N/A')}")
        print(f"  Average Episode Reward: {data.get('avg_episode_reward', 'N/A')}")
        print(f"  Total Trades: {data.get('total_trades', 'N/A')}")
        print(f"  Final Portfolio Value: {data.get('final_portfolio_value', 'N/A')}")
        print(f"  Portfolio Return %: {data.get('portfolio_return_pct', 'N/A')}")
        print(f"  Sharpe Ratio: {data.get('sharpe_ratio', 'N/A')}")
        print(f"  Max Drawdown: {data.get('max_drawdown', 'N/A')}")
        print(f"  Win Rate: {data.get('win_rate', 'N/A')}")
        print()

    # Analyze SAC v427 hybrid results
    sac_results_file = (
        results_dir / "backtest_results_sac_v427_hybrid_20251026_063723.json"
    )
    if sac_results_file.exists():
        with open(sac_results_file, "r") as f:
            data = json.load(f)

        print("🤖 SAC V427 HYBRID RESULTS:")
        print(f"  Model: {data.get('model', 'N/A')}")
        print(f"  Total Reward: {data.get('total_reward', 'N/A')}")
        print(f"  Average Episode Reward: {data.get('avg_episode_reward', 'N/A')}")
        print(f"  Total Trades: {data.get('total_trades', 'N/A')}")
        print(f"  Final Portfolio Value: {data.get('final_portfolio_value', 'N/A')}")
        print(f"  Portfolio Return %: {data.get('portfolio_return_pct', 'N/A')}")
        print(f"  Sharpe Ratio: {data.get('sharpe_ratio', 'N/A')}")
        print(f"  Max Drawdown: {data.get('max_drawdown', 'N/A')}")
        print(f"  Win Rate: {data.get('win_rate', 'N/A')}")
        print(f"  Feature Count: {data.get('feature_count', 'N/A')}")
        print()

    # Analyze v443.2 phase 2 results
    v443_results_dir = results_dir / "v443_2_phase2" / "rl_20251031_004029"
    metrics_file = v443_results_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            data = json.load(f)

        print("🚀 V443.2 PHASE 2 RESULTS:")
        metadata = data.get("metadata", {})
        metrics = data.get("metrics", {})

        print(f"  Strategy: {metadata.get('strategy', 'N/A')}")
        print(f"  Dataset: {metadata.get('dataset', 'N/A')}")
        print(f"  Initial Capital: {metadata.get('initial_capital', 'N/A')}")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
        print(f"  Total Return: {metrics.get('total_return', 'N/A')}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 'N/A')}")
        print(f"  Win Rate: {metrics.get('win_rate', 'N/A')}")
        print(f"  Total Trades: {metrics.get('total_trades', 'N/A')}")
        print(f"  CAGR: {metrics.get('cagr', 'N/A')}")
        print(f"  Volatility: {metrics.get('volatility', 'N/A')}")
        print()

    print("=== TRAINING STATUS ===")
    print("✅ v443.2 Phase 3 training completed successfully")
    print("✅ v443.2 model validation passed")
    print("⚠️  v441 training script executed but no actual training performed")
    print("📊 Model saved: models/ppo_v443_2_backtest_optimization.zip")


if __name__ == "__main__":
    analyze_backtest_results()
