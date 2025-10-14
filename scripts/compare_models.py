#!/usr/bin/env python3
"""
Model Comparison Script for SAC Trading Models
Performs statistical comparison between v413, v411, and v412 models
"""

import json
import numpy as np
from scipy import stats
import sys
from pathlib import Path

def load_backtest_results(file_path):
    """Load backtest results from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_trade_returns(trades):
    """Extract individual trade returns from trades data"""
    returns = []
    for trade in trades:
        pnl = float(trade['pnl'])
        # Calculate return percentage based on position size and entry price
        position = abs(float(trade['position']))
        entry_price = float(trade['entry_price'])
        trade_return = (pnl / (position * entry_price)) * 100
        returns.append(trade_return)
    return returns

def perform_statistical_tests(model_results):
    """Perform t-tests and other statistical comparisons"""
    results = {}

    # Extract trade returns for each model
    trade_returns = {}
    for model_name, data in model_results.items():
        trade_returns[model_name] = extract_trade_returns(data['trades'])

    # Perform pairwise t-tests
    models = list(model_results.keys())
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model1, model2 = models[i], models[j]
            returns1 = trade_returns[model1]
            returns2 = trade_returns[model2]

            # Perform t-test
            t_stat, p_value = stats.ttest_ind(returns1, returns2, equal_var=False)

            results[f'{model1}_vs_{model2}'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05),
                'model1_mean': float(np.mean(returns1)),
                'model2_mean': float(np.mean(returns2)),
                'model1_std': float(np.std(returns1)),
                'model2_std': float(np.std(returns2))
            }

    return results

def p_mean_method(model_results):
    """Implement p-mean method for model comparison"""
    p_mean_results = {}

    for model_name, data in model_results.items():
        metrics = data['metrics']
        total_return = float(metrics['total_return'])
        sharpe_ratio = float(metrics['sharpe_ratio'])
        win_rate = float(metrics['win_rate']) / 100  # Convert to decimal
        max_drawdown = abs(float(metrics['max_drawdown']))

        # Calculate p-mean score (higher is better)
        # Weight: Return (40%), Sharpe (30%), Win Rate (20%), Drawdown penalty (10%)
        p_mean_score = (
            0.4 * total_return +
            0.3 * sharpe_ratio +
            0.2 * win_rate * 100 +  # Scale win rate back
            0.1 * (1 / (1 + max_drawdown)) * 100  # Penalize drawdown
        )

        p_mean_results[model_name] = {
            'p_mean_score': p_mean_score,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate * 100,
            'max_drawdown': max_drawdown
        }

    return p_mean_results

def main():
    # Load backtest results
    results_dir = Path('results')
    model_files = {
        'v413_ultra_profit': results_dir / 'v413_backtest_results.json',
        'v411_trading_focused': results_dir / 'v411_backtest_results.json',
        'v412_profit_focused': results_dir / 'v412_backtest_results.json'
    }

    model_results = {}
    for model_name, file_path in model_files.items():
        if file_path.exists():
            model_results[model_name] = load_backtest_results(file_path)
            print(f"Loaded {model_name} results")
        else:
            print(f"Warning: {file_path} not found")

    if len(model_results) < 2:
        print("Error: Need at least 2 model results for comparison")
        sys.exit(1)

    # Perform statistical tests
    print("\nPerforming statistical tests...")
    stat_results = perform_statistical_tests(model_results)

    # Perform p-mean method comparison
    print("Performing p-mean method comparison...")
    p_mean_results = p_mean_method(model_results)

    # Determine best model
    best_model = max(p_mean_results.items(), key=lambda x: x[1]['p_mean_score'])[0]

    # Prepare final results
    comparison_results = {
        'statistical_tests': stat_results,
        'p_mean_comparison': p_mean_results,
        'best_model': best_model,
        'summary': {
            'models_compared': list(model_results.keys()),
            'total_models': len(model_results),
            'best_performing_model': best_model
        }
    }

    # Save results
    output_file = results_dir / 'model_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)

    print(f"\nComparison completed. Results saved to {output_file}")
    print(f"\nBest performing model: {best_model}")

    # Print summary
    print("\n=== MODEL COMPARISON SUMMARY ===")
    print(f"Models compared: {', '.join(model_results.keys())}")
    print(f"Best model: {best_model}")
    print("\nP-Mean Scores:")
    for model, results in p_mean_results.items():
        print(".2f")

    print("\nStatistical Significance (t-tests):")
    for test_name, results in stat_results.items():
        sig = "SIGNIFICANT" if results['significant'] else "NOT SIGNIFICANT"
        print(".4f")

if __name__ == '__main__':
    main()