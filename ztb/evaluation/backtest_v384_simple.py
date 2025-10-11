#!/usr/bin/env python3
"""
Simple backtest for v384 (68 curated features) on historical BTC data.

This script tests the v384 model which uses curated features.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
from typing import Dict, Any
from datetime import datetime

from ztb.utils.data_utils import load_csv_data_optimized
from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.training.policies.policy_utils import predict_with_masks
from ztb.utils.logging_utils import get_logger
from ztb.utils.config import TypedConfig

logger = get_logger(__name__)


def main() -> None:
    """Run backtest for v384 model."""
    
    config = TypedConfig()
    model_path = config.get_model_path("ppo_reward_v384_curated_60.zip")
    
    # Try multiple datasets
    datasets = [
        "ml-dataset-enhanced.csv",
        "btc_jpy_real_dataset.csv",
        "btc_jpy_yahoo_real_dataset.csv",
    ]
    
    data_path = None
    for ds in datasets:
        if Path(ds).exists():
            data_path = ds
            break
    
    if not data_path:
        logger.error("No dataset found!")
        return 1
    
    logger.info(f"{'='*80}")
    logger.info(f"v384 Backtest - 68 Curated Features")
    logger.info(f"{'='*80}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Dataset: {data_path}")
    
    # Load data
    df = load_csv_data_optimized(data_path)
    logger.info(f"Loaded {len(df):,} rows")
    logger.info(f"Columns: {len(df.columns)}")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Create environment
    config = {
        "reward_scaling": 0.01,
        "transaction_cost": 0.00505,
        "max_position_size": 1.05,
        "risk_free_rate": 0.05,
    }
    
    env = HeavyTradingEnv(df=df, config=config)
    logger.info(f"Environment observation space: {env.observation_space.shape[0]} features")
    
    # Load model
    try:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Run multiple episodes
    episodes = 20
    logger.info(f"\nRunning {episodes} episodes...")
    logger.info("-"*80)
    
    all_rewards = []
    all_pnls = []
    all_returns = []
    action_counts = {"HOLD": 0, "BUY": 0, "SELL": 0}
    all_step_pnls = []
    trades = []
    
    for ep in range(episodes):
        try:
            obs, info = env.reset()
            done = False
            truncated = False
            steps = 0
            ep_reward = 0.0
            ep_pnl = 0.0
            
            while not (done or truncated) and steps < 1000:  # Limit steps per episode
                action, _ = predict_with_masks(model, obs, env, deterministic=True)
                if isinstance(action, np.ndarray):
                    action = action.item()
                
                obs, reward, done, truncated, info = env.step(action)
                
                ep_reward += reward
                step_pnl = info.get('pnl', 0.0)
                ep_pnl += step_pnl
                all_step_pnls.append(step_pnl)
                
                # Count actions
                if action == 0:
                    action_counts["HOLD"] += 1
                elif action == 1:
                    action_counts["BUY"] += 1
                    if step_pnl != 0:
                        trades.append(("BUY", steps, step_pnl))
                else:
                    action_counts["SELL"] += 1
                    if step_pnl != 0:
                        trades.append(("SELL", steps, step_pnl))
                
                steps += 1
            
            all_rewards.append(ep_reward)
            all_pnls.append(ep_pnl)
            
            initial_balance = info.get('initial_balance', 10000000)
            ep_return = (ep_pnl / initial_balance) * 100 if initial_balance > 0 else 0
            all_returns.append(ep_return)
            
            logger.info(
                f"Ep {ep+1:2d}: Reward={ep_reward:7.2f}, "
                f"PnL={ep_pnl:10,.0f} JPY, "
                f"Return={ep_return:6.2f}%, "
                f"Steps={steps:4d}"
            )
            
        except Exception as e:
            logger.error(f"Episode {ep+1} failed: {e}")
            continue
    
    # Results
    logger.info("\n" + "="*80)
    logger.info("BACKTEST RESULTS")
    logger.info("="*80)
    
    total_actions = sum(action_counts.values())
    
    logger.info(f"\nPerformance Metrics:")
    logger.info(f"  Episodes completed: {len(all_rewards)}")
    logger.info(f"  Average reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    logger.info(f"  Average PnL: {np.mean(all_pnls):,.0f} ± {np.std(all_pnls):,.0f} JPY")
    logger.info(f"  Total PnL: {np.sum(all_pnls):,.0f} JPY")
    logger.info(f"  Average return: {np.mean(all_returns):.2f}% ± {np.std(all_returns):.2f}%")
    logger.info(f"  Total return: {np.sum(all_returns):.2f}%")
    
    logger.info(f"\nAction Distribution:")
    logger.info(f"  Total actions: {total_actions:,}")
    for action, count in action_counts.items():
        pct = (count / total_actions * 100) if total_actions > 0 else 0
        logger.info(f"  {action}: {count:,} ({pct:.1f}%)")
    
    logger.info(f"\nTrading Activity:")
    logger.info(f"  Total trades: {len(trades)}")
    
    if len(trades) > 0:
        winning = [t for t in trades if t[2] > 0]
        losing = [t for t in trades if t[2] < 0]
        logger.info(f"  Winning trades: {len(winning)} ({len(winning)/len(trades)*100:.1f}%)")
        logger.info(f"  Losing trades: {len(losing)} ({len(losing)/len(trades)*100:.1f}%)")
        if winning:
            logger.info(f"  Avg win: {np.mean([t[2] for t in winning]):,.0f} JPY")
        if losing:
            logger.info(f"  Avg loss: {np.mean([t[2] for t in losing]):,.0f} JPY")
    
    # Risk metrics
    if len(all_step_pnls) > 0:
        cumulative = np.cumsum(all_step_pnls)
        max_dd = np.min(cumulative - np.maximum.accumulate(cumulative))
        
        if np.std(all_step_pnls) > 0:
            periods_per_year = 365 * 24 * 60 * 4
            sharpe = (np.mean(all_step_pnls) / np.std(all_step_pnls)) * np.sqrt(periods_per_year)
        else:
            sharpe = 0.0
        
        logger.info(f"\nRisk Metrics:")
        logger.info(f"  Max drawdown: {max_dd:,.0f} JPY")
        logger.info(f"  Sharpe ratio: {sharpe:.3f}")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "dataset": data_path,
        "episodes": len(all_rewards),
        "avg_reward": float(np.mean(all_rewards)),
        "avg_pnl": float(np.mean(all_pnls)),
        "total_pnl": float(np.sum(all_pnls)),
        "avg_return_pct": float(np.mean(all_returns)),
        "action_distribution": {
            k: {"count": v, "pct": (v/total_actions*100 if total_actions > 0 else 0)}
            for k, v in action_counts.items()
        },
        "total_trades": len(trades),
    }
    
    output_file = f"backtest_v384_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {output_file}")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
