#!/usr/bin/env python3
"""
Test script for Action Signal Guide strength analysis functionality.
Tests the new strength distribution tracking, correlation analysis, and recommendations.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuide,
    ActionSignalGuideConfig
)
from ztb.utils.logging_utils import setup_logging
from ztb.utils.errors import safe_operation, TradingBotError
import logging


class StrengthAnalysisTestError(TradingBotError):
    """Custom error for strength analysis testing."""
    pass


def generate_test_data() -> pd.DataFrame:
    """Generate test market data and trading results for analysis."""
    logger = logging.getLogger(__name__)

    try:
        logger.info("Generating test market data...")

        # Generate sample OHLCV data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
        np.random.seed(42)

        # Generate realistic price data with trends and volatility
        n_points = len(dates)
        base_price = 100.0

        # Add trend and noise
        trend = np.linspace(0, 10, n_points)
        noise = np.random.normal(0, 2, n_points)
        prices = base_price + trend + noise

        # Ensure no negative prices
        prices = np.maximum(prices, 1.0)

        # Create OHLCV data
        high_prices = prices * (1 + np.random.uniform(0, 0.02, n_points))
        low_prices = prices * (1 - np.random.uniform(0, 0.02, n_points))
        open_prices = prices + np.random.normal(0, 0.5, n_points)
        close_prices = prices + np.random.normal(0, 0.5, n_points)
        volumes = np.random.uniform(1000, 10000, n_points)

        data = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=dates)

        logger.info(f"Generated {len(data)} data points")
        return data

    except Exception as e:
        logger.error(f"Failed to generate test data: {e}")
        raise StrengthAnalysisTestError(f"Data generation failed: {e}") from e


def generate_trading_results(signal_guide: ActionSignalGuide, data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generate mock trading results with signals."""
    logger = logging.getLogger(__name__)

    try:
        logger.info("Generating mock trading results...")
        trading_results = []

        for i in range(10, len(data)):  # Start from index 10 to have enough history
            current_data = data.iloc[:i+1]

            # Generate signals
            signals = safe_operation(
                signal_guide.generate_signals,
                logger=logger,
                context="Signal generation",
                default_result=[],
                data=current_data,
                current_index=i
            )

            # Mock trading performance based on signals
            profit = np.random.normal(0, 0.01)  # Base random profit
            win_rate = 0.5 + np.random.normal(0, 0.1)  # Base win rate around 50%
            sharpe_ratio = np.random.normal(0.5, 0.3)  # Base Sharpe ratio
            max_drawdown = abs(np.random.normal(0.05, 0.02))  # Base drawdown

            # Adjust performance based on signal characteristics
            if signals:
                # Stronger signals tend to improve performance
                avg_strength = np.mean([s.strength for s in signals])
                profit += avg_strength * 0.005  # Small positive effect from strong signals
                win_rate += avg_strength * 0.1   # Better win rate with strong signals

                # Some patterns might be more effective than others
                pattern_bonus = len(set([p for s in signals for p in s.source_patterns])) * 0.002
                profit += pattern_bonus

            trading_results.append({
                'timestamp': data.index[i],
                'signals': [
                    {
                        'direction': s.direction,
                        'strength': s.strength,
                        'source_patterns': s.source_patterns,
                        'signal_type': s.signal_type
                    } for s in signals
                ],
                'profit': profit,
                'win_rate': max(0, min(1, win_rate)),  # Clamp to [0,1]
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            })

        logger.info(f"Generated {len(trading_results)} trading periods")
        return trading_results

    except Exception as e:
        logger.error(f"Failed to generate trading results: {e}")
        raise StrengthAnalysisTestError(f"Trading results generation failed: {e}") from e


def run_strength_analysis_test() -> None:
    """Run the complete strength analysis test."""
    logger = logging.getLogger(__name__)

    try:
        logger.info("=== Testing Action Signal Guide Strength Analysis ===")

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

        signal_guide = safe_operation(
            ActionSignalGuide,
            logger=logger,
            context="Signal guide creation",
            default_result=None,
            config=config
        )

        if signal_guide is None:
            raise StrengthAnalysisTestError("Failed to create ActionSignalGuide")

        # Generate test data
        data = generate_test_data()

        # Generate signals to build statistics
        logger.info("Generating signals and building statistics...")
        total_signals = safe_operation(
            lambda: sum(len(signal_guide.generate_signals(data, current_index=i))
                       for i in range(50, len(data), 10)),
            logger=logger,
            context="Signal statistics generation",
            default_result=0
        )

        logger.info(f"Generated {total_signals} total signals")

        # Generate trading results
        trading_results = generate_trading_results(signal_guide, data)

        # Run strength analysis
        logger.info("Running strength analysis...")
        analysis = safe_operation(
            signal_guide.analyze_pattern_effectiveness,
            logger=logger,
            context="Strength analysis",
            default_result=None,
            trading_results=trading_results
        )

        if analysis is None:
            raise StrengthAnalysisTestError("Strength analysis failed")

        logger.info("Analysis complete!")

        # Print results
        print("\n=== STRENGTH ANALYSIS RESULTS ===")
        print(analysis)

        logger.info("=== Test completed successfully! ===")

    except StrengthAnalysisTestError:
        raise  # Re-raise our custom errors
    except Exception as e:
        logger.error(f"Unexpected error in strength analysis test: {e}")
        raise StrengthAnalysisTestError(f"Test execution failed: {e}") from e


def main() -> None:
    """Main entry point."""
    # Set up logging
    setup_logging(
        level=logging.INFO,
        format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)

    try:
        run_strength_analysis_test()
    except StrengthAnalysisTestError as e:
        logger.error(f"Strength analysis test failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()