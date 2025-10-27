#!/usr/bin/env python3
"""
Optimal Configuration Analysis for Action Signal Guide

This script analyzes the strength analysis results to determine optimal
configuration settings for the Action Signal Guide pattern recognition system.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

from ztb.trading.strategies.action_signal_guide.action_signal_guide import ActionSignalGuide, ActionSignalGuideConfig


def generate_comprehensive_test_data():
    """Generate comprehensive test data with various market conditions."""
    # Generate multiple market scenarios
    scenarios = [
        {'name': 'trending_bull', 'trend': 0.001, 'volatility': 0.02},
        {'name': 'trending_bear', 'trend': -0.001, 'volatility': 0.02},
        {'name': 'sideways', 'trend': 0.0001, 'volatility': 0.015},
        {'name': 'volatile', 'trend': 0.0005, 'volatility': 0.04},
        {'name': 'calm', 'trend': 0.0002, 'volatility': 0.01},
    ]

    all_data = {}

    for scenario in scenarios:
        dates = pd.date_range(start='2024-01-01', end='2024-02-01', freq='30min')
        np.random.seed(42)

        n_points = len(dates)
        base_price = 100.0

        # Add trend and noise
        trend_component = np.linspace(0, scenario['trend'] * n_points, n_points)
        noise = np.random.normal(0, scenario['volatility'], n_points)
        prices = base_price + trend_component + noise

        # Ensure no negative prices
        prices = np.maximum(prices, 1.0)

        # Create OHLCV data
        high_prices = prices * (1 + np.random.uniform(0, scenario['volatility']*2, n_points))
        low_prices = prices * (1 - np.random.uniform(0, scenario['volatility']*2, n_points))
        open_prices = prices + np.random.normal(0, scenario['volatility']*0.5, n_points)
        close_prices = prices + np.random.normal(0, scenario['volatility']*0.5, n_points)
        volumes = np.random.uniform(1000, 10000, n_points)

        data = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=dates)

        all_data[scenario['name']] = data

    return all_data


def analyze_strength_thresholds(signal_guide, data, trading_results):
    """Analyze optimal strength thresholds for different patterns."""
    print("=== Analyzing Optimal Strength Thresholds ===")

    # Test different strength thresholds
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_results = {}

    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}")

        # Count signals above threshold
        total_signals = 0
        pattern_counts = {}

        for i in range(100, len(data), 20):  # Sample every 20 points
            signals = signal_guide.generate_signals(data, current_index=i)
            filtered_signals = [s for s in signals if s.strength >= threshold]
            total_signals += len(filtered_signals)

            for signal in filtered_signals:
                pattern = signal.source_patterns[0] if signal.source_patterns else 'unknown'
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        threshold_results[threshold] = {
            'total_signals': total_signals,
            'pattern_distribution': pattern_counts
        }

        print(f"  Signals >= {threshold}: {total_signals}")

    return threshold_results


def analyze_pattern_correlations(signal_guide, trading_results):
    """Analyze correlations between pattern strengths and trading performance."""
    print("\n=== Analyzing Pattern-Strength Correlations ===")

    # Group results by pattern and strength ranges
    strength_ranges = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
    pattern_performance = {}

    for result in trading_results:
        for signal in result['signals']:
            pattern = signal['signal_type']
            strength = signal['strength']

            if pattern not in pattern_performance:
                pattern_performance[pattern] = {range_key: [] for range_key in strength_ranges}

            # Find appropriate strength range
            for range_min, range_max in strength_ranges:
                if range_min <= strength < range_max:
                    pattern_performance[pattern][(range_min, range_max)].append(result['profit'])
                    break

    # Calculate average performance per strength range
    correlations = {}
    for pattern, ranges in pattern_performance.items():
        correlations[pattern] = {}
        for range_key, profits in ranges.items():
            if profits:
                avg_profit = np.mean(profits)
                win_rate = np.mean([1 if p > 0 else 0 for p in profits])
                correlations[pattern][f"{range_key[0]}-{range_key[1]}"] = {
                    'avg_profit': avg_profit,
                    'win_rate': win_rate,
                    'sample_size': len(profits)
                }

    return correlations


def generate_optimal_configuration(analysis_results, threshold_analysis, correlations):
    """Generate optimal configuration based on analysis results."""
    print("\n=== Generating Optimal Configuration ===")

    optimal_config = {
        'strength_thresholds': {},
        'pattern_weights': {},
        'recommendations': []
    }

    # Analyze strength thresholds
    threshold_data = threshold_analysis
    thresholds = list(threshold_data.keys())

    # Find optimal threshold (balance between signal quality and quantity)
    signal_counts = [data['total_signals'] for data in threshold_data.values()]

    # Use elbow method to find optimal threshold
    if len(signal_counts) > 2:
        # Calculate rate of change
        rates = []
        for i in range(1, len(signal_counts)):
            if signal_counts[i-1] > 0:
                rate = (signal_counts[i] - signal_counts[i-1]) / signal_counts[i-1]
                rates.append(abs(rate))

        if rates:
            optimal_idx = np.argmin(rates) + 1  # +1 because rates start from index 1
            optimal_threshold = thresholds[optimal_idx]
        else:
            optimal_threshold = 0.3
    else:
        optimal_threshold = 0.3

    optimal_config['strength_thresholds']['global'] = optimal_threshold
    optimal_config['recommendations'].append(f"Global strength threshold: {optimal_threshold}")

    # Analyze pattern-specific thresholds based on correlations
    for pattern, ranges in correlations.items():
        best_range = None
        best_performance = -float('inf')

        for range_key, stats in ranges.items():
            if stats['sample_size'] > 10:  # Minimum sample size
                performance_score = stats['avg_profit'] * stats['win_rate']
                if performance_score > best_performance:
                    best_performance = performance_score
                    best_range = range_key

        if best_range:
            range_min, range_max = map(float, best_range.split('-'))
            optimal_config['strength_thresholds'][pattern] = (range_min + range_max) / 2
            optimal_config['recommendations'].append(
                f"{pattern} optimal strength: {(range_min + range_max) / 2:.2f} "
                f"(profit: {correlations[pattern][best_range]['avg_profit']:.4f}, "
                f"win_rate: {correlations[pattern][best_range]['win_rate']:.2f})"
            )

    # Pattern weights based on consistency and performance
    pattern_stats = analysis_results.get('pattern_stats', {})
    for pattern, stats in pattern_stats.items():
        strength_stats = stats.get('strength_stats', {})
        if strength_stats:
            consistency = 1 / (strength_stats.get('std', 1) + 0.1)  # Lower std = higher consistency
            signal_count = stats.get('signals_generated', 0)
            weight = consistency * min(signal_count / 100, 1)  # Balance consistency and volume
            optimal_config['pattern_weights'][pattern] = weight

    return optimal_config


def run_comprehensive_analysis():
    """Run comprehensive analysis to determine optimal configuration."""
    print("=== Action Signal Guide Optimal Configuration Analysis ===")
    print()

    # Create signal guide with all patterns enabled
    config = ActionSignalGuideConfig(
        enable_candlestick_patterns=True,
        enable_fibonacci_patterns=True,
        enable_gann_patterns=True,
        enable_wave_patterns=True,
        enable_harmonic_patterns=True,
        enable_oscillator_patterns=True,
        enable_volume_patterns=True,
        enable_bollinger_patterns=True,
        enable_adx_patterns=True,
        enable_granville_patterns=True,
        enable_heikin_ashi_patterns=True,
        enable_dow_theory_patterns=True
    )

    signal_guide = ActionSignalGuide(config=config)

    # Generate comprehensive test data
    print("Generating comprehensive test data...")
    test_datasets = generate_comprehensive_test_data()
    print(f"Generated {len(test_datasets)} market scenarios")
    print()

    # Analyze each scenario
    all_results = {}
    all_trading_results = []

    for scenario_name, data in test_datasets.items():
        print(f"Analyzing scenario: {scenario_name}")

        # Generate signals and build statistics
        signals_data = []
        for i in range(100, len(data), 10):
            signals = signal_guide.generate_signals(data, current_index=i)
            signals_data.extend(signals)

        # Generate mock trading results
        trading_results = []
        for i in range(10, len(data)):
            current_data = data.iloc[:i+1]

            # Get signals for this period
            period_signals = [s for s in signals_data if s.timestamp >= data.index[max(0, i-10)] and s.timestamp <= data.index[i]]

            # Mock trading performance
            profit = np.random.normal(0, 0.01)
            win_rate = 0.5 + np.random.normal(0, 0.1)
            sharpe_ratio = np.random.normal(0.5, 0.3)
            max_drawdown = abs(np.random.normal(0.05, 0.02))

            # Adjust performance based on signals
            if period_signals:
                avg_strength = np.mean([s.strength for s in period_signals])
                profit += avg_strength * 0.005
                win_rate += avg_strength * 0.1

            trading_results.append({
                'timestamp': data.index[i],
                'signals': [
                    {
                        'direction': s.direction,
                        'strength': s.strength,
                        'signal_type': s.signal_type,
                        'source_patterns': s.source_patterns
                    } for s in period_signals
                ],
                'profit': profit,
                'win_rate': max(0, min(1, win_rate)),
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            })

        all_trading_results.extend(trading_results)

        # Run strength analysis
        analysis = signal_guide.analyze_pattern_effectiveness(trading_results)
        all_results[scenario_name] = analysis

        print(f"  Completed analysis for {scenario_name}")
        print()

    # Aggregate results across scenarios
    print("Aggregating results across all scenarios...")
    aggregated_analysis = signal_guide.analyze_pattern_effectiveness(all_trading_results)

    # Analyze strength thresholds
    primary_data = test_datasets['trending_bull']  # Use trending scenario as primary
    threshold_analysis = analyze_strength_thresholds(signal_guide, primary_data, all_trading_results)

    # Analyze correlations
    correlations = analyze_pattern_correlations(signal_guide, all_trading_results)

    # Generate optimal configuration
    optimal_config = generate_optimal_configuration(aggregated_analysis, threshold_analysis, correlations)

    # Display final results
    print("\n" + "="*60)
    print("OPTIMAL CONFIGURATION RESULTS")
    print("="*60)

    print("\nRecommended Strength Thresholds:")
    for key, value in optimal_config['strength_thresholds'].items():
        print(f"  {key}: {value}")

    print("\nRecommended Pattern Weights:")
    for pattern, weight in optimal_config['pattern_weights'].items():
        print(f"  {pattern}: {weight:.3f}")

    print("\nConfiguration Recommendations:")
    for rec in optimal_config['recommendations']:
        print(f"  • {rec}")

    print("\nDetailed Analysis Results:")
    print("Pattern Statistics:")
    for pattern, stats in aggregated_analysis['pattern_stats'].items():
        print(f"  {pattern.upper()}: {stats['signals_generated']} signals")
        strength_stats = stats.get('strength_stats', {})
        if strength_stats:
            print(f"    Strength: {strength_stats.get('mean', 0):.3f} ± {strength_stats.get('std', 0):.3f}")

    print("\n=== Analysis completed successfully! ===")

    return optimal_config


if __name__ == "__main__":
    optimal_config = run_comprehensive_analysis()