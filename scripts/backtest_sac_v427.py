#!/usr/bin/env python3
"""
SAC v427 Complete Backtest Analysis

Execute comprehensive backtest for SAC v427 market-adaptive ensemble system.
Includes detailed performance analysis, regime-specific evaluation, and comparison.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.analysis.analyze_backtest import BacktestAnalyzer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_sac_v427_backtest(model_path: str, data_path: str, output_path: str) -> Dict[str, Any]:
    """
    Run comprehensive backtest for SAC v427 model.

    Args:
        model_path: Path to trained SAC v427 model
        data_path: Path to test data
        output_path: Path to save backtest results

    Returns:
        Backtest results dictionary
    """
    logger.info("Starting SAC v427 comprehensive backtest...")

    try:
        from stable_baselines3 import SAC
        import gymnasium as gym
        from ztb.trading.environment.constants import continuous_to_discrete_action
        from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
        from ztb.trading.environment.utils.config import EnvironmentConfig

        # Load test data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded test data: {len(df)} rows")

        # Create environment config for v427
        env_config_dict = {
            "initial_portfolio_value": 200000.0,  # Match v424 for fair comparison
            "transaction_cost": 1e-05,  # Match v424 transaction cost
            "max_position_size": 1.0,
            "enable_action_masking": True,
            "use_continuous_actions": True,
            "use_standardized_observations": True,
            "random_start": False,
            "curriculum_stage": "strong_penalty_trading",
            "continuous_to_discrete_threshold": 0.1,
            "feature_set": "v427_adaptive",
            "adaptive_feature_selection": {
                "enabled": True,
                "regime_classifier": {
                    "adx_threshold": 25.0,
                    "volatility_percentile": 70.0
                },
                "attention_layer": {
                    "enabled": True,
                    "hidden_dim": 128
                },
                "min_weight_threshold": 0.3
            }
        }
        env_config = EnvironmentConfig.from_dict(env_config_dict)

        # Create HeavyTradingEnv for backtesting
        env = HeavyTradingEnv(df=df, config=env_config)
        logger.info(f"Created HeavyTradingEnv with {len(env.features)} features")

        # Load SAC v427 model
        model = SAC.load(model_path)
        logger.info(f"Loaded model from {model_path}")

        # Run backtest
        obs, info = env.reset()
        done = False
        truncated = False

        backtest_results = {
            'portfolio_values': [],
            'actions': [],
            'rewards': [],
            'timestamps': [],
            'market_data': [],
            'regime_info': []
        }

        step = 0
        while not done and not truncated:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            discrete_action = continuous_to_discrete_action(action)

            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)

            # Record results
            backtest_results['portfolio_values'].append(env.portfolio_value)
            backtest_results['actions'].append(discrete_action)
            backtest_results['rewards'].append(reward)

            # Get timestamp from data
            if hasattr(env, 'current_step') and hasattr(env, 'df') and len(env.df) > env.current_step:
                current_row = env.df.iloc[env.current_step]
                if 'timestamp' in current_row:
                    current_timestamp = pd.to_datetime(current_row['timestamp'])
                else:
                    current_timestamp = pd.Timestamp.now()
            else:
                current_timestamp = pd.Timestamp.now()

            backtest_results['timestamps'].append(current_timestamp)

            # Get market data
            if hasattr(env, 'current_step') and hasattr(env, 'df') and len(env.df) > env.current_step:
                current_row = env.df.iloc[env.current_step]
                backtest_results['market_data'].append({
                    'open': float(current_row.get('open', 0)),
                    'high': float(current_row.get('high', 0)),
                    'low': float(current_row.get('low', 0)),
                    'close': float(current_row.get('close', 0)),
                    'volume': float(current_row.get('volume', 0)) if 'volume' in current_row else 0
                })
            else:
                backtest_results['market_data'].append({
                    'open': 0, 'high': 0, 'low': 0, 'close': 0, 'volume': 0
                })

            # Detect market regime (simplified)
            regime = 'unknown'  # Placeholder for regime detection
            backtest_results['regime_info'].append(regime)

            obs = next_obs
            step += 1

            if step % 1000 == 0:
                logger.info(f"Backtest progress: {step} steps completed")

        # Calculate comprehensive metrics
        results = calculate_backtest_metrics(backtest_results, df)

        # Save results
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        serializable_results = convert_numpy_types(results)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        logger.info(f"Backtest completed. Results saved to {output_path}")
        return results

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'error': str(e)}


def calculate_backtest_metrics(backtest_results: Dict[str, Any], original_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive backtest metrics.

    Args:
        backtest_results: Raw backtest results
        original_data: Original market data

    Returns:
        Comprehensive metrics dictionary
    """
    portfolio_values = np.array(backtest_results['portfolio_values'])
    actions = np.array(backtest_results['actions'])
    rewards = np.array(backtest_results['rewards'])
    timestamps = pd.to_datetime(backtest_results['timestamps'])

    # Basic return metrics
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_value) / initial_value

    # Annualized return (assuming daily data)
    days = len(portfolio_values)
    annual_return = (1 + total_return) ** (365 / days) - 1

    # Drawdown analysis
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown)

    # Sharpe ratio (assuming daily returns)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    if len(returns) > 1:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365) if np.std(returns) > 0 else 0
    else:
        sharpe_ratio = 0

    # Win rate
    winning_trades = np.sum(returns > 0)
    total_trades = len(returns)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # Action distribution
    unique_actions, action_counts = np.unique(actions, return_counts=True)
    action_distribution = dict(zip(unique_actions, action_counts / len(actions)))

    # Market correlation
    market_returns = original_data['close'].pct_change().iloc[:len(returns)]
    if len(market_returns) == len(returns):
        market_correlation = np.corrcoef(returns, market_returns)[0, 1] if len(returns) > 1 else 0
    else:
        market_correlation = 0

    # Regime-specific analysis
    regime_performance = analyze_regime_performance(backtest_results)

    # Risk metrics
    volatility = np.std(returns) * np.sqrt(365) if len(returns) > 1 else 0
    var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0

    # Performance by time periods
    temporal_analysis = analyze_temporal_performance(backtest_results)

    return {
        'summary': {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'volatility': volatility,
            'var_95': var_95,
            'market_correlation': market_correlation,
            'total_steps': len(portfolio_values),
            'initial_value': initial_value,
            'final_value': final_value
        },
        'action_distribution': action_distribution,
        'regime_performance': regime_performance,
        'temporal_analysis': temporal_analysis,
        'raw_data': {
            'portfolio_values': portfolio_values.tolist(),
            'actions': actions.tolist(),
            'rewards': rewards.tolist(),
            'timestamps': [t.isoformat() for t in timestamps]
        }
    }


def analyze_regime_performance(backtest_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze performance by market regime."""
    regimes = backtest_results['regime_info']
    portfolio_values = backtest_results['portfolio_values']

    regime_performance = {}
    unique_regimes = set(regimes)

    for regime in unique_regimes:
        regime_indices = [i for i, r in enumerate(regimes) if r == regime]
        if len(regime_indices) > 1:
            regime_values = [portfolio_values[i] for i in regime_indices]
            regime_returns = np.diff(regime_values) / regime_values[:-1]

            regime_performance[regime] = {
                'steps': len(regime_indices),
                'avg_return': np.mean(regime_returns) if len(regime_returns) > 0 else 0,
                'volatility': np.std(regime_returns) if len(regime_returns) > 0 else 0,
                'sharpe_ratio': np.mean(regime_returns) / np.std(regime_returns) * np.sqrt(365)
                              if len(regime_returns) > 1 and np.std(regime_returns) > 0 else 0,
                'win_rate': np.sum(regime_returns > 0) / len(regime_returns) if len(regime_returns) > 0 else 0
            }

    return regime_performance


def analyze_temporal_performance(backtest_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze performance by time periods."""
    timestamps = pd.to_datetime(backtest_results['timestamps'])
    portfolio_values = backtest_results['portfolio_values']

    # Monthly performance
    monthly_returns = []
    for month in pd.date_range(timestamps.min(), timestamps.max(), freq='M'):
        month_mask = (timestamps >= month) & (timestamps < month + pd.offsets.MonthEnd(1))
        if month_mask.sum() > 1:
            month_values = np.array(portfolio_values)[month_mask]
            month_return = (month_values[-1] - month_values[0]) / month_values[0]
            monthly_returns.append({
                'month': month.strftime('%Y-%m'),
                'return': month_return
            })

    # Quarterly performance
    quarterly_returns = []
    for quarter in pd.date_range(timestamps.min(), timestamps.max(), freq='Q'):
        quarter_mask = (timestamps >= quarter) & (timestamps < quarter + pd.offsets.QuarterEnd(1))
        if quarter_mask.sum() > 1:
            quarter_values = np.array(portfolio_values)[quarter_mask]
            quarter_return = (quarter_values[-1] - quarter_values[0]) / quarter_values[0]
            quarterly_returns.append({
                'quarter': quarter.strftime('%Y-Q%q'),
                'return': quarter_return
            })

    return {
        'monthly_returns': monthly_returns,
        'quarterly_returns': quarterly_returns,
        'best_month': max(monthly_returns, key=lambda x: x['return']) if monthly_returns else None,
        'worst_month': min(monthly_returns, key=lambda x: x['return']) if monthly_returns else None
    }


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description='SAC v427 Complete Backtest')
    parser.add_argument('--model', required=True, help='Path to SAC v427 model')
    parser.add_argument('--data', required=True, help='Path to test data')
    parser.add_argument('--output', required=True, help='Output path for results')

    args = parser.parse_args()

    # Run backtest
    results = run_sac_v427_backtest(args.model, args.data, args.output)

    # Print summary
    if 'error' not in results:
        summary = results['summary']
        print("\n" + "="*60)
        print("SAC v427 BACKTEST RESULTS")
        print("="*60)
        print(".2%")
        print(".2%")
        print(".2f")
        print(".2%")
        print(".3f")
        print(".3f")
        print("="*60)

    return 0 if 'error' not in results else 1


if __name__ == "__main__":
    sys.exit(main())