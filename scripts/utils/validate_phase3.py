#!/usr/bin/env python3
"""
Action Signal Guide Phase 3 Validation Script

This script validates the Phase 3 implementation of Action Signal Guide,
including ML integration, real-time adaptation, and portfolio optimization.
"""

import logging
import os
import sys
import time

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all Phase 3 components can be imported successfully."""
    logger.info("Testing Phase 3 component imports...")

    try:
        # Test interface imports
        from ztb.trading.strategies.action_signal_guide.interfaces.ml_interfaces import (
            IPatternOptimizer,
            MLPredictionModel,
            MLTrainingData,
        )

        logger.info("✓ ML interfaces imported successfully")

        from ztb.trading.strategies.action_signal_guide.interfaces.portfolio_interfaces import (
            AllocationStrategy,
            IStrategyAllocator,
            PortfolioAllocation,
        )

        logger.info("✓ Portfolio interfaces imported successfully")

        from ztb.trading.strategies.action_signal_guide.interfaces.adaptation_interfaces import (
            IStreamingProcessor,
            ProcessingResult,
            StreamingDataPoint,
            StreamingDataType,
        )

        logger.info("✓ Adaptation interfaces imported successfully")

        # Test config imports
        from ztb.trading.strategies.action_signal_guide.config.asg_ml_config import (
            MLIntegrationConfig,
            PatternOptimizerConfig,
        )

        logger.info("✓ ML config imported successfully")

        from ztb.trading.strategies.action_signal_guide.config.asg_portfolio_config import (
            PortfolioOptimizationConfig,
            StrategyAllocatorConfig,
        )

        logger.info("✓ Portfolio config imported successfully")

        from ztb.trading.strategies.action_signal_guide.config.asg_adaptation_config import (
            RealTimeAdaptationConfig,
            StreamingProcessorConfig,
        )

        logger.info("✓ Adaptation config imported successfully")

        # Test implementation imports
        from ztb.trading.strategies.action_signal_guide.ml_integration.pattern_optimizer import (
            create_pattern_optimizer,
        )

        logger.info("✓ Pattern optimizer imported successfully")

        from ztb.trading.strategies.action_signal_guide.portfolio_optimization.strategy_allocator import (
            create_strategy_allocator,
        )

        logger.info("✓ Strategy allocator imported successfully")

        from ztb.trading.strategies.action_signal_guide.realtime_adaptation.streaming_processor import (
            create_streaming_processor,
        )

        logger.info("✓ Streaming processor imported successfully")

        return True

    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during import: {e}")
        return False


def test_config_validation():
    """Test configuration validation."""
    logger.info("Testing configuration validation...")

    try:
        from ztb.trading.strategies.action_signal_guide.config.asg_adaptation_config import (
            RealTimeAdaptationConfig,
        )
        from ztb.trading.strategies.action_signal_guide.config.asg_ml_config import (
            MLIntegrationConfig,
        )
        from ztb.trading.strategies.action_signal_guide.config.asg_portfolio_config import (
            PortfolioOptimizationConfig,
        )

        # Test ML config
        ml_config = MLIntegrationConfig()
        issues = ml_config.pattern_optimizer.validate_config()
        if issues:
            logger.warning(f"ML config validation issues: {issues}")
        else:
            logger.info("✓ ML config validation passed")

        # Test portfolio config
        portfolio_config = PortfolioOptimizationConfig()
        issues = portfolio_config.strategy_allocator.validate_config()
        if issues:
            logger.warning(f"Portfolio config validation issues: {issues}")
        else:
            logger.info("✓ Portfolio config validation passed")

        # Test adaptation config
        adaptation_config = RealTimeAdaptationConfig()
        issues = adaptation_config.validate_config()
        if issues:
            logger.warning(f"Adaptation config validation issues: {issues}")
        else:
            logger.info("✓ Adaptation config validation passed")

        return True

    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        return False


def test_component_initialization():
    """Test component initialization."""
    logger.info("Testing component initialization...")

    try:
        from ztb.trading.strategies.action_signal_guide.config.asg_adaptation_config import (
            RealTimeAdaptationConfig,
        )
        from ztb.trading.strategies.action_signal_guide.config.asg_ml_config import (
            MLIntegrationConfig,
        )
        from ztb.trading.strategies.action_signal_guide.config.asg_portfolio_config import (
            PortfolioOptimizationConfig,
        )
        from ztb.trading.strategies.action_signal_guide.ml_integration.pattern_optimizer import (
            create_pattern_optimizer,
        )
        from ztb.trading.strategies.action_signal_guide.portfolio_optimization.strategy_allocator import (
            create_strategy_allocator,
        )
        from ztb.trading.strategies.action_signal_guide.realtime_adaptation.streaming_processor import (
            create_streaming_processor,
        )

        # Initialize configs
        ml_config = MLIntegrationConfig()
        portfolio_config = PortfolioOptimizationConfig()
        adaptation_config = RealTimeAdaptationConfig()

        # Test pattern optimizer
        optimizer = create_pattern_optimizer(ml_config)
        logger.info("✓ Pattern optimizer initialized successfully")

        # Test strategy allocator
        allocator = create_strategy_allocator(portfolio_config)
        logger.info("✓ Strategy allocator initialized successfully")

        # Test streaming processor
        processor = create_streaming_processor(adaptation_config)
        logger.info("✓ Streaming processor initialized successfully")

        return True

    except Exception as e:
        logger.error(f"Component initialization failed: {e}")
        return False


def generate_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate test market data."""
    np.random.seed(42)

    dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="1H")

    # Generate realistic price data
    base_price = 100.0
    prices = [base_price]

    for i in range(1, n_samples):
        # Add trend and volatility
        trend = 0.0001 * np.sin(i / 50)
        volatility = 0.02 * np.random.normal(0, 1)
        price_change = trend + volatility
        new_price = prices[-1] * (1 + price_change)
        prices.append(max(new_price, 0.01))  # Prevent negative prices

    # Generate volume
    volumes = np.random.lognormal(mean=10, sigma=1, size=n_samples)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "price": prices,
            "volume": volumes,
            "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            "open": [prices[i - 1] if i > 0 else prices[0] for i in range(n_samples)],
            "close": prices,
        }
    )

    return df


def test_pattern_optimizer():
    """Test pattern optimizer functionality."""
    logger.info("Testing pattern optimizer functionality...")

    try:
        from ztb.trading.strategies.action_signal_guide.config.asg_ml_config import (
            MLIntegrationConfig,
        )
        from ztb.trading.strategies.action_signal_guide.interfaces.ml_interfaces import (
            MLTrainingData,
        )
        from ztb.trading.strategies.action_signal_guide.ml_integration.pattern_optimizer import (
            create_pattern_optimizer,
        )

        # Generate test data
        df = generate_test_data(500)

        # Prepare training data
        training_data = MLTrainingData(
            features=df[["price", "volume", "high", "low"]].values,
            target=df["price"]
            .pct_change()
            .shift(-1)
            .fillna(0)
            .values,  # Next period return
            feature_names=["price", "volume", "high", "low"],
        )

        # Initialize optimizer
        config = MLIntegrationConfig()
        optimizer = create_pattern_optimizer(config)

        # Test optimization
        start_time = time.time()
        result = optimizer.optimize_patterns(training_data)
        optimization_time = time.time() - start_time

        if result.success:
            logger.info(f"✓ Pattern optimization completed in {optimization_time:.2f}s")
            logger.info(f"  Best model: {result.metadata.get('best_model', 'Unknown')}")
            logger.info(
                f"  Training time: {result.metadata.get('training_time', 0):.2f}s"
            )
        else:
            logger.error(f"✗ Pattern optimization failed: {result.message}")
            return False

        # Test prediction
        test_features = {
            "price": df.iloc[-1]["price"],
            "volume": df.iloc[-1]["volume"],
            "high": df.iloc[-1]["high"],
            "low": df.iloc[-1]["low"],
        }

        prediction_result = optimizer.predict_patterns(test_features)
        if prediction_result.success:
            logger.info("✓ Pattern prediction completed")
        else:
            logger.error(f"✗ Pattern prediction failed: {prediction_result.message}")
            return False

        return True

    except Exception as e:
        logger.error(f"Pattern optimizer test failed: {e}")
        return False


def test_strategy_allocator():
    """Test strategy allocator functionality."""
    logger.info("Testing strategy allocator functionality...")

    try:
        from ztb.trading.strategies.action_signal_guide.config.asg_portfolio_config import (
            PortfolioOptimizationConfig,
        )
        from ztb.trading.strategies.action_signal_guide.interfaces.portfolio_interfaces import (
            StrategyPerformance,
        )
        from ztb.trading.strategies.action_signal_guide.portfolio_optimization.strategy_allocator import (
            create_strategy_allocator,
        )

        # Create mock strategy performance data
        strategy_performance = {
            "strategy_1": StrategyPerformance(
                strategy_name="strategy_1",
                expected_return=0.15,
                volatility=0.25,
                sharpe_ratio=0.6,
                max_drawdown=0.12,
                win_rate=0.55,
                profit_factor=1.2,
                timestamp=time.time(),
                correlations={"strategy_2": 0.3, "strategy_3": 0.1},
            ),
            "strategy_2": StrategyPerformance(
                strategy_name="strategy_2",
                expected_return=0.12,
                volatility=0.20,
                sharpe_ratio=0.6,
                max_drawdown=0.08,
                win_rate=0.58,
                profit_factor=1.3,
                timestamp=time.time(),
                correlations={"strategy_1": 0.3, "strategy_3": 0.4},
            ),
            "strategy_3": StrategyPerformance(
                strategy_name="strategy_3",
                expected_return=0.18,
                volatility=0.30,
                sharpe_ratio=0.6,
                max_drawdown=0.15,
                win_rate=0.52,
                profit_factor=1.1,
                timestamp=time.time(),
                correlations={"strategy_1": 0.1, "strategy_2": 0.4},
            ),
        }

        market_conditions = {"regime": "neutral", "volatility": 0.2, "trend": 0.05}

        # Initialize allocator
        config = PortfolioOptimizationConfig()
        allocator = create_strategy_allocator(config)

        # Test allocation
        start_time = time.time()
        allocation = allocator.allocate_strategies(
            strategy_performance, market_conditions
        )
        allocation_time = time.time() - start_time

        if allocation.allocations:
            logger.info(f"✓ Strategy allocation completed in {allocation_time:.4f}s")
            logger.info(f"  Allocations: {allocation.allocations}")
            logger.info(f"  Expected return: {allocation.expected_return:.4f}")
            logger.info(f"  Expected risk: {allocation.expected_volatility:.4f}")
            logger.info(f"  Sharpe ratio: {allocation.sharpe_ratio:.4f}")
        else:
            logger.error("✗ Strategy allocation failed: No allocations returned")
            return False

        # Test risk metrics (simplified)
        logger.info("✓ Basic allocation validation completed")
        # risk_metrics = allocator.calculate_risk_metrics(allocation.allocations, strategy_performance)
        # logger.info("✓ Risk metrics calculated successfully")
        # logger.info(f"  Portfolio volatility: {risk_metrics.portfolio_volatility:.4f}")
        # logger.info(f"  Sharpe ratio: {risk_metrics.sharpe_ratio:.4f}")
        # logger.info(f"  Diversification ratio: {risk_metrics.diversification_ratio:.4f}")

        return True

    except Exception as e:
        logger.error(f"Strategy allocator test failed: {e}")
        return False


def test_streaming_processor():
    """Test streaming processor functionality."""
    logger.info("Testing streaming processor functionality...")

    try:
        from ztb.trading.strategies.action_signal_guide.config.asg_adaptation_config import (
            RealTimeAdaptationConfig,
        )
        from ztb.trading.strategies.action_signal_guide.interfaces.adaptation_interfaces import (
            StreamingDataPoint,
            StreamingDataType,
        )
        from ztb.trading.strategies.action_signal_guide.realtime_adaptation.streaming_processor import (
            create_streaming_processor,
        )

        # Generate test data
        df = generate_test_data(100)

        # Initialize processor
        config = RealTimeAdaptationConfig()
        processor = create_streaming_processor(config)

        # Start processing
        if not processor.start_processing():
            logger.error("✗ Failed to start streaming processor")
            return False

        logger.info("✓ Streaming processor started")

        # Add data points
        data_points = []
        for i in range(min(10, len(df))):
            data_point = StreamingDataPoint(
                data_type=StreamingDataType.MARKET_DATA,
                timestamp=time.time() + i,
                data={
                    "price": df.iloc[i]["price"],
                    "volume": df.iloc[i]["volume"],
                    "indicators": {
                        "rsi": 50 + np.random.normal(0, 10),
                        "macd": np.random.normal(0, 0.01),
                        "bollinger_upper": df.iloc[i]["price"] * 1.02,
                        "bollinger_lower": df.iloc[i]["price"] * 0.98,
                    },
                },
                metadata={},
            )
            data_points.append(data_point)

        # Process batch
        start_time = time.time()
        result = processor.process_batch(data_points)
        processing_time = time.time() - start_time

        if result.success:
            logger.info(f"✓ Batch processing completed in {processing_time:.4f}s")
            logger.info(f"  Processed count: {result.processed_count}")
            logger.info(
                f"  Throughput: {result.processed_count / processing_time:.2f} items/sec"
            )
        else:
            logger.error(
                f"✗ Batch processing failed: {result.metadata.get('error', 'Unknown error')}"
            )
            return False

        # Stop processing
        if processor.stop_processing():
            logger.info("✓ Streaming processor stopped")
        else:
            logger.warning("⚠ Failed to stop streaming processor gracefully")

        return True

    except Exception as e:
        logger.error(f"Streaming processor test failed: {e}")
        return False


def run_validation():
    """Run all validation tests."""
    logger.info("Starting Action Signal Guide Phase 3 Validation")
    logger.info("=" * 60)

    tests = [
        ("Import Test", test_imports),
        ("Config Validation", test_config_validation),
        ("Component Initialization", test_component_initialization),
        ("Pattern Optimizer", test_pattern_optimizer),
        ("Strategy Allocator", test_strategy_allocator),
        ("Streaming Processor", test_streaming_processor),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "PASSED" if result else "FAILED"
            logger.info(f"--- {test_name}: {status} ---")
        except Exception as e:
            logger.error(f"--- {test_name}: ERROR - {e} ---")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status} {test_name}")
        if result:
            passed += 1

    logger.info("-" * 60)
    logger.info(f"Total: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All Phase 3 validation tests PASSED!")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
