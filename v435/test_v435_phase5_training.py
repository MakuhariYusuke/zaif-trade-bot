#!/usr/bin/env python3
"""
SAC v435 Phase 5 Testing Script
Test training and evaluation with risk management integration
"""

import json
import logging
from pathlib import Path

from ztb.training.v435.evaluate_sac_v435 import SACv435Evaluator
from ztb.training.v435.train_sac_v435 import SACv435Trainer

logger = logging.getLogger(__name__)


def test_phase5_training():
    """Test Phase 5 training setup with risk management (skip actual training)"""
    logger.info("🧪 Testing SAC v435 Phase 5 Training Setup with Risk Management")

    # Create trainer with short training for testing
    trainer = SACv435Trainer()

    # Modify config for short test training
    trainer.config["training"]["total_timesteps"] = 100  # Very short for testing

    # Test setup only (don't actually train due to environment issues)
    try:
        # Test that trainer initializes correctly
        assert trainer.risk_manager is not None, "Risk manager not initialized"
        assert trainer.config is not None, "Config not loaded"

        # Test risk management setup
        risk_result = trainer.risk_manager.calculate_risk_adjusted_position(
            base_position=0.1,
            current_price=100000,
            portfolio_value=100000,
            atr=1000,
            df=pd.DataFrame([{"close": 100000, "atr": 1000}]),
        )
        assert "adjusted_position" in risk_result, "Risk calculation failed"

        logger.info("✅ Phase 5 training setup test passed")
        return {"status": "success", "risk_test": risk_result}

    except Exception as e:
        logger.error(f"Training setup test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_phase5_evaluation():
    """Test Phase 5 evaluation setup with risk management"""
    logger.info("🧪 Testing SAC v435 Phase 5 Evaluation Setup")

    # Create evaluator
    evaluator = SACv435Evaluator()

    # Test configuration loading
    assert evaluator.config is not None, "Config not loaded"
    assert "risk_management" in evaluator.config, "Risk management config missing"

    # Test risk management setup
    assert evaluator.risk_manager is not None, "Risk manager not initialized"

    # Test that evaluation methods exist (don't actually run evaluation without model)
    assert hasattr(evaluator, "evaluate_model"), "evaluate_model method missing"
    assert hasattr(
        evaluator, "_calculate_risk_metrics"
    ), "risk metrics calculation method missing"

    logger.info("✅ Phase 5 evaluation setup test passed")
    return True


def test_risk_management_integration():
    """Test that risk management is properly integrated"""
    logger.info("🧪 Testing Risk Management Integration")

    trainer = SACv435Trainer()

    # Check that risk manager is initialized
    assert trainer.risk_manager is not None, "Risk manager not initialized in trainer"

    # Check risk manager has required components
    assert hasattr(
        trainer.risk_manager, "calculate_risk_adjusted_position"
    ), "Risk manager missing position calculation method"

    # Test risk calculation with sample data
    sample_data = {"close": 100000, "atr": 1000}

    risk_result = trainer.risk_manager.calculate_risk_adjusted_position(
        base_position=0.1,
        current_price=sample_data["close"],
        portfolio_value=100000,
        atr=sample_data["atr"],
        df=pd.DataFrame([sample_data]),
    )

    assert "adjusted_position" in risk_result, "Risk result missing adjusted position"
    assert isinstance(
        risk_result["adjusted_position"], (int, float)
    ), "Adjusted position not numeric"

    logger.info(
        f"Risk adjustment test: base=0.1, adjusted={risk_result['adjusted_position']:.4f}"
    )
    logger.info("✅ Risk management integration test passed")

    return risk_result


def main():
    """Main test function"""
    logger.info("🚀 Starting SAC v435 Phase 5 Testing")

    results = {}

    try:
        # Test 1: Risk Management Integration
        results["risk_integration"] = test_risk_management_integration()

        # Test 2: Training Setup
        results["training"] = test_phase5_training()

        # Test 3: Evaluation Setup
        results["evaluation"] = test_phase5_evaluation()

        # Summary
        passed = sum(
            1
            for r in results.values()
            if r is not True and "status" in r and r["status"] == "success"
        ) + sum(1 for r in results.values() if r is True)

        logger.info(f"📊 Test Results: {passed}/{len(results)} tests passed")

        if passed == len(results):
            logger.info("🎉 All Phase 5 tests passed!")
            results["overall_status"] = "success"
        else:
            logger.warning("⚠️ Some Phase 5 tests failed")
            results["overall_status"] = "partial"

    except Exception as e:
        logger.error(f"❌ Phase 5 testing failed: {e}")
        results["overall_status"] = "failed"
        results["error"] = str(e)

    # Save test results
    results_dir = Path("results/v435")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "phase5_test_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"Phase 5 test results saved to {results_file}")

    return results


if __name__ == "__main__":
    import pandas as pd  # Import here to avoid import errors in test

    main()
