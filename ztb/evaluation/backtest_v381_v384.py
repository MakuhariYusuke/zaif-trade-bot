#!/usr/bin/env python3
"""
Backtest Comparison: v381 (110 features) vs v384 (68 curated features)

Tests both models on historical BTC/JPY market data to compare:
1. Trading performance (PnL, Sharpe ratio)
2. Action distribution (HOLD/BUY/SELL balance)
3. Risk metrics (max drawdown, win rate)
"""

import sys
from pathlib import Path
from typing import Any, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime

from ztb.utils.data_utils import load_csv_data_optimized
from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.training.policies.policy_utils import predict_with_masks
from ztb.utils.logging_utils import get_logger
from ztb.utils.config import TypedConfig

logger = get_logger(__name__)


def load_model(model_path: str) -> Tuple[Any, str]:
    """Load PPO model (try MaskablePPO first, fallback to standard PPO)."""
    try:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(model_path)
        logger.info(f"Loaded {model_path} with MaskablePPO")
        return model, "MaskablePPO"
    except Exception as e1:
        logger.warning(f"MaskablePPO load failed: {e1}")
        try:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
            logger.info(f"Loaded {model_path} with standard PPO")
            return model, "PPO"
        except Exception as e2:
            logger.error(f"Failed to load model: {e2}")
            raise


def run_backtest(
    model_path: str,
    data_path: str,
    episodes: int = 10,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Run backtest for a single model.
    
    Args:
        model_path: Path to model .zip file
        data_path: Path to CSV dataset
        episodes: Number of episodes to run
        model_name: Name for logging
        
    Returns:
        Dictionary with backtest results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Backtesting {model_name}")
    logger.info(f"{'='*80}")
    
    # Load data
    df = load_csv_data_optimized(data_path)
    logger.info(f"Loaded {len(df):,} rows from {data_path}")
    logger.info(f"Data columns: {len(df.columns)}")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Create environment with optimized config
    config = {
        "reward_scaling": 0.01,
        "transaction_cost": 0.00505,
        "max_position_size": 1.05,
        "risk_free_rate": 0.05,
    }
    
    env = HeavyTradingEnv(df=df, config=config)
    logger.info(f"Environment created: {env.observation_space.shape[0]} features expected")
    
    # Load model
    model, model_type = load_model(model_path)
    logger.info(f"Model type: {model_type}")
    
    # Run episodes
    episode_rewards = []
    episode_pnls = []
    episode_returns = []
    action_counts = {"HOLD": 0, "BUY": 0, "SELL": 0}
    all_pnls = []
    trades = []
    
    for episode in range(episodes):
        try:
            obs, info = env.reset()
            done = False
            truncated = False
            steps = 0
            episode_reward = 0.0
            episode_pnl = 0.0
            episode_actions = []
            
            while not (done or truncated):
                # Predict action
                action, _states = predict_with_masks(model, obs, env, deterministic=True)
                if isinstance(action, np.ndarray):
                    action = action.item()
                
                # Step environment
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                step_pnl = info.get('pnl', 0.0)
                episode_pnl += step_pnl
                all_pnls.append(step_pnl)
                episode_actions.append(action)
                
                # Count actions
                if action == 0:
                    action_counts["HOLD"] += 1
                elif action == 1:
                    action_counts["BUY"] += 1
                    trades.append(("BUY", steps, step_pnl))
                else:
                    action_counts["SELL"] += 1
                    trades.append(("SELL", steps, step_pnl))
                
                steps += 1
            
            # Episode statistics
            episode_rewards.append(episode_reward)
            episode_pnls.append(episode_pnl)
            
            # Calculate return
            initial_balance = info.get('initial_balance', 10000000)
            episode_return = (episode_pnl / initial_balance) * 100 if initial_balance > 0 else 0
            episode_returns.append(episode_return)
            
            logger.info(
                f"Episode {episode+1}/{episodes}: "
                f"Reward={episode_reward:.2f}, "
                f"PnL={episode_pnl:,.0f} JPY, "
                f"Return={episode_return:.2f}%, "
                f"Steps={steps}"
            )
            
        except Exception as e:
            logger.error(f"Episode {episode+1} failed: {e}", exc_info=True)
            continue
    
    # Calculate statistics
    total_actions = sum(action_counts.values())
    
    results = {
        "model_name": model_name,
        "model_path": model_path,
        "data_path": data_path,
        "episodes": len(episode_rewards),
        "avg_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "avg_pnl": float(np.mean(episode_pnls)),
        "std_pnl": float(np.std(episode_pnls)),
        "total_pnl": float(np.sum(episode_pnls)),
        "avg_return_pct": float(np.mean(episode_returns)),
        "std_return_pct": float(np.std(episode_returns)),
        "action_distribution": {
            "HOLD": action_counts["HOLD"],
            "BUY": action_counts["BUY"],
            "SELL": action_counts["SELL"],
            "HOLD_pct": (action_counts["HOLD"] / total_actions * 100) if total_actions > 0 else 0,
            "BUY_pct": (action_counts["BUY"] / total_actions * 100) if total_actions > 0 else 0,
            "SELL_pct": (action_counts["SELL"] / total_actions * 100) if total_actions > 0 else 0,
        },
        "total_actions": total_actions,
        "num_trades": len(trades),
    }
    
    # Calculate risk metrics
    if len(all_pnls) > 0:
        cumulative_pnl = np.cumsum(all_pnls)
        max_drawdown = float(np.min(cumulative_pnl - np.maximum.accumulate(cumulative_pnl)))
        
        # Sharpe ratio (annualized, assuming 15s intervals)
        if np.std(all_pnls) > 0:
            periods_per_year = 365 * 24 * 60 * 4  # 15-second intervals
            sharpe_ratio = (np.mean(all_pnls) / np.std(all_pnls)) * np.sqrt(periods_per_year)
        else:
            sharpe_ratio = 0.0
        
        # Win rate
        winning_trades = [t for t in trades if t[2] > 0]
        win_rate = (len(winning_trades) / len(trades) * 100) if len(trades) > 0 else 0
        
        results.update({
            "max_drawdown": max_drawdown,
            "sharpe_ratio": float(sharpe_ratio),
            "win_rate_pct": float(win_rate),
            "num_winning_trades": len(winning_trades),
        })
    
    return results


def print_comparison(results_v381: Dict[str, Any], results_v384: Dict[str, Any]) -> None:
    """Print side-by-side comparison of results."""
    
    print("\n" + "="*100)
    print("BACKTEST COMPARISON: v381 (110 features) vs v384 (68 curated features)")
    print("="*100)
    
    print(f"\n{'Metric':<30} {'v381':>20} {'v384':>20} {'Difference':>20}")
    print("-"*100)
    
    # Performance metrics
    print(f"\n{'PERFORMANCE METRICS':^100}")
    print("-"*100)
    
    metrics = [
        ("Episodes", "episodes", "{:.0f}"),
        ("Avg Reward", "avg_reward", "{:.2f}"),
        ("Std Reward", "std_reward", "{:.2f}"),
        ("Avg PnL (JPY)", "avg_pnl", "{:,.0f}"),
        ("Total PnL (JPY)", "total_pnl", "{:,.0f}"),
        ("Avg Return (%)", "avg_return_pct", "{:.2f}%"),
        ("Sharpe Ratio", "sharpe_ratio", "{:.3f}"),
        ("Max Drawdown (JPY)", "max_drawdown", "{:,.0f}"),
        ("Win Rate (%)", "win_rate_pct", "{:.1f}%"),
    ]
    
    for label, key, fmt in metrics:
        v381_val = results_v381.get(key, 0)
        v384_val = results_v384.get(key, 0)
        diff = v384_val - v381_val
        
        if "%" in fmt:
            print(f"{label:<30} {fmt.format(v381_val):>20} {fmt.format(v384_val):>20} {fmt.format(diff):>20}")
        else:
            print(f"{label:<30} {fmt.format(v381_val):>20} {fmt.format(v384_val):>20} {fmt.format(diff):>20}")
    
    # Action distribution
    print(f"\n{'ACTION DISTRIBUTION':^100}")
    print("-"*100)
    
    actions = ["HOLD", "BUY", "SELL"]
    for action in actions:
        v381_pct = results_v381["action_distribution"][f"{action}_pct"]
        v384_pct = results_v384["action_distribution"][f"{action}_pct"]
        diff_pct = v384_pct - v381_pct
        
        print(f"{action:<30} {v381_pct:>19.1f}% {v384_pct:>19.1f}% {diff_pct:>19.1f}%")
    
    # Trading activity
    print(f"\n{'TRADING ACTIVITY':^100}")
    print("-"*100)
    
    print(f"{'Total Actions':<30} {results_v381['total_actions']:>20,} {results_v384['total_actions']:>20,}")
    print(f"{'Number of Trades':<30} {results_v381['num_trades']:>20,} {results_v384['num_trades']:>20,}")
    
    # Winner determination
    print(f"\n{'OVERALL ASSESSMENT':^100}")
    print("="*100)
    
    score_v381 = 0
    score_v384 = 0
    
    # Higher is better
    if results_v381["avg_reward"] > results_v384["avg_reward"]:
        score_v381 += 1
    else:
        score_v384 += 1
    
    if results_v381["total_pnl"] > results_v384["total_pnl"]:
        score_v381 += 1
    else:
        score_v384 += 1
    
    if results_v381["sharpe_ratio"] > results_v384["sharpe_ratio"]:
        score_v381 += 1
    else:
        score_v384 += 1
    
    if results_v381["win_rate_pct"] > results_v384["win_rate_pct"]:
        score_v381 += 1
    else:
        score_v384 += 1
    
    # Lower is better (more negative = worse)
    if results_v381["max_drawdown"] > results_v384["max_drawdown"]:  # Less negative = better
        score_v381 += 1
    else:
        score_v384 += 1
    
    print(f"v381 Score: {score_v381}/5")
    print(f"v384 Score: {score_v384}/5")
    
    if score_v384 > score_v381:
        print("\n🏆 WINNER: v384 (68 curated features)")
        print("   → Feature curation improved performance!")
    elif score_v381 > score_v384:
        print("\n🏆 WINNER: v381 (110 features)")
        print("   → More features provided better performance")
    else:
        print("\n🤝 TIE: Both models performed similarly")
    
    print("="*100)


def main() -> int:
    """Main backtest comparison."""
    
    # Configuration - use config-based model paths
    config = TypedConfig()
    v381_model = config.get_model_path("ppo_reward_v381_revised_profit_focused.zip")
    v384_model = config.get_model_path("ppo_reward_v384_curated_60.zip")
    
    # Try both datasets
    datasets = [
        "btc_jpy_real_dataset.csv",
        "btc_jpy_yahoo_real_dataset.csv",
        "ml-dataset-enhanced.csv",
    ]
    
    # Find available dataset
    data_path = None
    for ds in datasets:
        if Path(ds).exists():
            data_path = ds
            logger.info(f"Using dataset: {data_path}")
            break
    
    if data_path is None:
        logger.error("No dataset found! Please ensure one of the following exists:")
        for ds in datasets:
            logger.error(f"  - {ds}")
        return 1
    
    episodes = 10  # Number of backtest episodes
    
    try:
        # Run backtests
        logger.info("\n" + "="*80)
        logger.info("STARTING BACKTEST COMPARISON")
        logger.info("="*80)
        
        results_v381 = run_backtest(v381_model, data_path, episodes, "v381_revised_profit_focused")
        results_v384 = run_backtest(v384_model, data_path, episodes, "v384_curated_60")
        
        # Print comparison
        print_comparison(results_v381, results_v384)
        
        # Save results to JSON
        output_file = f"backtest_comparison_v381_v384_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results = {
            "timestamp": datetime.now().isoformat(),
            "dataset": data_path,
            "episodes": episodes,
            "v381": results_v381,
            "v384": results_v384,
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✅ Results saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
