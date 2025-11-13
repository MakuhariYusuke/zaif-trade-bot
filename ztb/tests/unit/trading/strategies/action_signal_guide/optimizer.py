#!/usr/bin/env python3
"""
Test script for Action Signal Guide strength analysis functionality.
Tests the new strength distribution tracking, correlation analysis, and recommendations.
"""

import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add the project root to the path
sys.path.insert(0, os.path.abspath("."))

import logging

from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuide,
    ActionSignalGuideConfig,
    GuidanceLevel,
)
from ztb.utils.errors import TradingBotError, safe_operation
from ztb.utils.logging_utils import setup_logging


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""

    config: Dict[str, Any]
    score: float
    metrics: Dict[str, float]
    trading_results: List[Dict[str, Any]]


@dataclass
class OptimizationConfig:
    """Configuration for optimization process."""

    max_iterations: int = 50
    population_size: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    early_stopping_rounds: int = 10
    random_seed: int = 42


class StrengthAnalysisTestError(TradingBotError):
    """Custom error for strength analysis testing."""

    pass


def generate_test_data() -> pd.DataFrame:
    """Generate test market data and trading results for analysis."""
    logger = logging.getLogger(__name__)

    try:
        logger.info("Generating test market data...")

        # Generate sample OHLCV data
        dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="1H")
        np.random.seed(42)

        # Generate realistic price data with trends and volatility
        n_points = len(dates)
        base_price = 5000000.0  # JPY-based price

        # Add trend and noise
        trend = np.linspace(0, 500000, n_points)  # Adjusted for JPY scale
        noise = np.random.normal(0, 100000, n_points)  # Adjusted volatility
        prices = base_price + trend + noise

        # Ensure no negative prices
        prices = np.maximum(prices, 1000000.0)

        # Create OHLCV data
        high_prices = prices * (1 + np.random.uniform(0, 0.02, n_points))
        low_prices = prices * (1 - np.random.uniform(0, 0.02, n_points))
        open_prices = prices + np.random.normal(0, 0.5, n_points)
        close_prices = prices + np.random.normal(0, 0.5, n_points)
        volumes = np.random.uniform(1000, 10000, n_points)

        data = pd.DataFrame(
            {
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volumes,
            },
            index=dates,
        )

        logger.info(f"Generated {len(data)} data points")
        return data

    except Exception as e:
        logger.error(f"Failed to generate test data: {e}")
        raise StrengthAnalysisTestError(f"Data generation failed: {e}") from e


def generate_trading_results(
    signal_guide: ActionSignalGuide, data: pd.DataFrame
) -> List[Dict[str, Any]]:
    """Generate mock trading results with signals."""
    logger = logging.getLogger(__name__)

    try:
        logger.info("Generating mock trading results...")
        trading_results = []

        for i in range(10, len(data)):  # Start from index 10 to have enough history
            current_data = data.iloc[: i + 1]

            # Generate signals
            signals = safe_operation(
                signal_guide.generate_signals,
                logger=logger,
                context="Signal generation",
                default_result=[],
                data=current_data,
                current_index=i,
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
                profit += (
                    avg_strength * 0.005
                )  # Small positive effect from strong signals
                win_rate += avg_strength * 0.1  # Better win rate with strong signals

                # Some patterns might be more effective than others
                pattern_bonus = (
                    len(set([p for s in signals for p in s.source_patterns])) * 0.002
                )
                profit += pattern_bonus

            trading_results.append(
                {
                    "timestamp": data.index[i],
                    "signals": [
                        {
                            "direction": s.direction,
                            "strength": s.strength,
                            "source_patterns": s.source_patterns,
                            "signal_type": s.signal_type,
                        }
                        for s in signals
                    ],
                    "profit": profit,
                    "win_rate": max(0, min(1, win_rate)),  # Clamp to [0,1]
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                }
            )

        logger.info(f"Generated {len(trading_results)} trading periods")
        return trading_results

    except Exception as e:
        logger.error(f"Failed to generate trading results: {e}")
        raise StrengthAnalysisTestError(
            f"Trading results generation failed: {e}"
        ) from e


def optimize_signal_guide_parameters(
    data: pd.DataFrame,
    base_config: ActionSignalGuideConfig,
    optimization_config: OptimizationConfig,
) -> Optional[OptimizationResult]:
    """
    Optimize ActionSignalGuide parameters using evolutionary algorithm.

    Args:
        data: Market data for testing
        base_config: Base configuration to optimize from
        optimization_config: Optimization algorithm settings

    Returns:
        Best optimization result found, or None if optimization failed
    """
    logger = logging.getLogger(__name__)
    random.seed(optimization_config.random_seed)
    np.random.seed(optimization_config.random_seed)

    logger.info("Starting parameter optimization...")

    # Parameter ranges to optimize (based on actual ActionSignalGuideConfig)
    param_ranges = {
        "max_signals_per_bar": (1, 10),
        "guidance_level": ["weak", "strong", "full"],  # Map to GuidanceLevel enum
        "enable_parallel_processing": [True, False],
        "enable_caching": [True, False],
    }

    def create_random_config() -> Dict[str, Any]:
        """Create a random parameter configuration."""
        return {
            "max_signals_per_bar": random.randint(*param_ranges["max_signals_per_bar"]),
            "guidance_level": random.choice(param_ranges["guidance_level"]),
            "enable_parallel_processing": random.choice(
                param_ranges["enable_parallel_processing"]
            ),
            "enable_caching": random.choice(param_ranges["enable_caching"]),
        }

    def evaluate_config(config_params: Dict[str, Any]) -> OptimizationResult:
        """Evaluate a parameter configuration."""
        try:
            # Map guidance level string to enum
            guidance_level_map = {
                "weak": GuidanceLevel.WEAK,
                "strong": GuidanceLevel.STRONG,
                "full": GuidanceLevel.FULL,
            }

            # Create config with optimized parameters
            config = ActionSignalGuideConfig(
                max_signals_per_bar=config_params["max_signals_per_bar"],
                guidance_level=guidance_level_map[config_params["guidance_level"]],
                enable_parallel_processing=config_params["enable_parallel_processing"],
                enable_caching=config_params["enable_caching"],
                # Keep other settings from base config
                enable_candlestick_patterns=base_config.enable_candlestick_patterns,
                enable_fibonacci_patterns=base_config.enable_fibonacci_patterns,
                enable_gann_patterns=base_config.enable_gann_patterns,
                enable_wave_patterns=base_config.enable_wave_patterns,
                enable_harmonic_patterns=base_config.enable_harmonic_patterns,
                enable_oscillator_patterns=base_config.enable_oscillator_patterns,
                enable_volume_patterns=base_config.enable_volume_patterns,
                enable_bollinger_patterns=base_config.enable_bollinger_patterns,
                enable_adx_patterns=base_config.enable_adx_patterns,
                enable_granville_patterns=base_config.enable_granville_patterns,
                enable_heikin_ashi_patterns=base_config.enable_heikin_ashi_patterns,
                enable_dow_theory_patterns=base_config.enable_dow_theory_patterns,
            )

            # Create signal guide
            signal_guide = safe_operation(
                ActionSignalGuide,
                logger=logger,
                context="Signal guide creation for optimization",
                default_result=None,
                config=config,
            )

            if signal_guide is None:
                return OptimizationResult(config_params, -float("inf"), {}, [])

            # Generate trading results
            trading_results = generate_trading_results(signal_guide, data)

            # Calculate performance score
            if not trading_results:
                return OptimizationResult(config_params, -float("inf"), {}, [])

            total_profit = sum(r["profit"] for r in trading_results)
            avg_win_rate = np.mean([r["win_rate"] for r in trading_results])
            avg_sharpe = np.mean([r["sharpe_ratio"] for r in trading_results])
            avg_drawdown = np.mean([r["max_drawdown"] for r in trading_results])

            # Composite score: profit + win_rate bonus - drawdown penalty
            score = total_profit + (avg_win_rate * 10) - (avg_drawdown * 20)

            metrics = {
                "total_profit": total_profit,
                "avg_win_rate": avg_win_rate,
                "avg_sharpe_ratio": avg_sharpe,
                "avg_max_drawdown": avg_drawdown,
                "total_trades": len(trading_results),
            }

            return OptimizationResult(config_params, score, metrics, trading_results)

        except Exception as e:
            logger.warning(f"Configuration evaluation failed: {e}")
            return OptimizationResult(config_params, -float("inf"), {}, [])

    # Evolutionary optimization
    population = [
        create_random_config() for _ in range(optimization_config.population_size)
    ]
    best_result = None
    best_score = -float("inf")
    no_improvement_count = 0

    for iteration in range(optimization_config.max_iterations):
        logger.info(
            f"Optimization iteration {iteration + 1}/{optimization_config.max_iterations}"
        )

        # Evaluate population
        results = []
        for config in population:
            result = evaluate_config(config)
            results.append(result)

            if result.score > best_score:
                best_score = result.score
                best_result = result
                no_improvement_count = 0
                logger.info(f"New best score: {best_score:.4f}")
            else:
                no_improvement_count += 1

        # Early stopping
        if no_improvement_count >= optimization_config.early_stopping_rounds:
            logger.info(f"Early stopping at iteration {iteration + 1}")
            break

        # Create next generation
        new_population = []

        # Elitism: keep best individual
        best_config = max(results, key=lambda r: r.score).config
        new_population.append(best_config)

        # Tournament selection and crossover
        while len(new_population) < optimization_config.population_size:
            # Tournament selection
            tournament = random.sample(results, min(3, len(results)))
            parent1 = max(tournament, key=lambda r: r.score).config
            tournament = random.sample(results, min(3, len(results)))
            parent2 = max(tournament, key=lambda r: r.score).config

            # Crossover
            if random.random() < optimization_config.crossover_rate:
                child = {}
                for key in parent1.keys():
                    if key == "max_signals_per_bar":
                        # Average integer values
                        child[key] = int((parent1[key] + parent2[key]) / 2)
                    elif key in ["enable_parallel_processing", "enable_caching"]:
                        # Random choice for boolean
                        child[key] = random.choice([parent1[key], parent2[key]])
                    elif key == "guidance_level":
                        # Random choice for categorical
                        child[key] = random.choice([parent1[key], parent2[key]])
                    else:
                        child[key] = parent1[key]  # Keep as is
            else:
                child = parent1.copy()

            # Mutation
            if random.random() < optimization_config.mutation_rate:
                mutation_type = random.choice(
                    ["max_signals", "guidance", "parallel", "caching"]
                )
                if mutation_type == "max_signals":
                    child["max_signals_per_bar"] = random.randint(
                        *param_ranges["max_signals_per_bar"]
                    )
                elif mutation_type == "guidance":
                    child["guidance_level"] = random.choice(
                        param_ranges["guidance_level"]
                    )
                elif mutation_type == "parallel":
                    child["enable_parallel_processing"] = random.choice(
                        param_ranges["enable_parallel_processing"]
                    )
                elif mutation_type == "caching":
                    child["enable_caching"] = random.choice(
                        param_ranges["enable_caching"]
                    )

            new_population.append(child)

        population = new_population

    logger.info("Parameter optimization completed")
    return best_result


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
            enable_dow_theory_patterns=True,
            # Enable short mode for faster testing and reduced log spam
            debug_short_mode=True,
            short_mode_recognizer_limit=5,
            error_suppression_threshold=2,
        )

        signal_guide = safe_operation(
            ActionSignalGuide,
            logger=logger,
            context="Signal guide creation",
            default_result=None,
            config=config,
        )

        if signal_guide is None:
            raise StrengthAnalysisTestError("Failed to create ActionSignalGuide")

        # Generate test data
        data = generate_test_data()

        # Generate signals to build statistics
        logger.info("Generating signals and building statistics...")
        total_signals = safe_operation(
            lambda: sum(
                len(signal_guide.generate_signals(data, current_index=i))
                for i in range(50, len(data), 10)
            ),
            logger=logger,
            context="Signal statistics generation",
            default_result=0,
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
            trading_results=trading_results,
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


def run_optimization_test() -> None:
    """Run parameter optimization test."""
    logger = logging.getLogger(__name__)

    try:
        logger.info("=== Testing Action Signal Guide Parameter Optimization ===")

        # Create base configuration
        base_config = ActionSignalGuideConfig(
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
            enable_dow_theory_patterns=True,
        )

        # Generate test data
        data = generate_test_data()

        # Run optimization
        optimization_config = OptimizationConfig(
            max_iterations=20,
            population_size=10,
            early_stopping_rounds=5,  # Reduced for testing
        )

        best_result = optimize_signal_guide_parameters(
            data, base_config, optimization_config
        )

        if best_result is None:
            raise StrengthAnalysisTestError(
                "Optimization failed to find any valid configuration"
            )

        # Print results
        print("\n=== OPTIMIZATION RESULTS ===")
        print(f"Best Score: {best_result.score:.4f}")
        print(f"Configuration: {best_result.config}")
        print(f"Metrics: {best_result.metrics}")

        logger.info("=== Optimization test completed successfully! ===")

    except StrengthAnalysisTestError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in optimization test: {e}")
        raise StrengthAnalysisTestError(f"Optimization test failed: {e}") from e


def main() -> None:
    """Main entry point."""
    # Set up logging
    setup_logging(
        level=logging.INFO,
        format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    try:
        run_strength_analysis_test()
        run_optimization_test()
    except StrengthAnalysisTestError as e:
        logger.error(f"Strength analysis test failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
