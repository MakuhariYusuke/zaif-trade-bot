with open('scripts/backtest/backtest_sac_v438_quick.py', 'a') as f:
    f.write('''


def calculate_backtest_summary(
    results_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    trades_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Calculate backtest summary statistics.

    Args:
        results_df: DataFrame with episode results
        portfolio_df: DataFrame with portfolio values over time
        trades_df: DataFrame with trade history

    Returns:
        Dict containing summary statistics
    """
    summary = {
        "total_episodes": len(results_df),
        "avg_total_reward": results_df["total_reward"].mean(),
        "std_total_reward": results_df["total_reward"].std(),
        "avg_final_portfolio_value": results_df["final_portfolio_value"].mean(),
        "std_final_portfolio_value": results_df["final_portfolio_value"].std(),
        "avg_total_trades": results_df["total_trades"].mean(),
        "avg_trades_per_step": results_df["trades_per_step"].mean(),
        "total_trades_all_episodes": trades_df.shape[0],
        "best_episode_reward": results_df["total_reward"].max(),
        "worst_episode_reward": results_df["total_reward"].min(),
        "reward_positive_ratio": (results_df["total_reward"] > 0).mean(),
        "portfolio_value_positive_ratio": (
            results_df["final_portfolio_value"] > 200000
        ).mean(),
    }

    # Calculate Sharpe-like ratio
    if len(results_df) > 1:
        returns = results_df["total_reward"]
        summary["sharpe_ratio"] = returns.mean() / (returns.std() + 1e-8)

    # Calculate max drawdown from portfolio values
    if not portfolio_df.empty:
        portfolio_values = portfolio_df.groupby("step")["portfolio_value"].mean()
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        summary["max_drawdown"] = drawdown.min()

    return summary


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Quick backtest SAC v438.1 model")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument("--data-path", type=str, default=None, help="Path to test data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="backtest_experiments/v438.1",
        help="Output directory for results",
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="Number of backtest episodes"
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic actions"
    )

    args = parser.parse_args()

    # Run backtest
    summary = backtest_sac_v438_quick(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_episodes=args.episodes,
        deterministic=args.deterministic,
    )

    if summary:
        print("\nBacktest Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    else:
        print("❌ Backtest failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
''')