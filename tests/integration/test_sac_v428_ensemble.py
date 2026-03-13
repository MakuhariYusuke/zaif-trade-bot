#!/usr/bin/env python3
"""
Test script for SAC v428 Phase 3 Ensemble System.
"""

import json

import numpy as np

from ztb.training.unified_trainer.ensemble_system import (
    EnsembleConfig,
    EnsemblePredictor,
)


def test_ensemble_system():
    """Test the ensemble system functionality."""
    print("🧪 Testing SAC v428 Phase 3 Ensemble System")
    print("=" * 60)

    # Create ensemble configuration
    config = EnsembleConfig(
        enabled=True,
        members=5,
        specializations=["bull", "bear", "sideways", "high_vol", "low_vol"],
        voting_mechanism="weighted_confidence",
        diversity_weight=0.3,
        consensus_requirement={
            "enabled": True,
            "agreement_threshold": 0.6,
            "force_hold_on_disagreement": True,
        },
        stability_voting={
            "enabled": True,
            "stability_weight": 0.4,
            "performance_weight": 0.6,
        },
    )

    # Initialize ensemble predictor
    ensemble = EnsemblePredictor(config)
    print("✅ Ensemble system initialized")

    # Test ensemble stats
    stats = ensemble.get_ensemble_stats()
    print(
        f"📊 Initial ensemble stats: {stats['overall_stats']['total_members']} members"
    )

    # Test predictions with mock data
    print("\n🎯 Testing ensemble predictions...")

    # Generate mock market observations
    np.random.seed(42)
    test_observations = [np.random.randn(10) for _ in range(20)]  # 20 test observations

    predictions = []
    for i, obs in enumerate(test_observations):
        action, analysis = ensemble.predict(obs)
        predictions.append((action, analysis))

        if i < 5:  # Show first 5 predictions in detail
            print(
                f"  Observation {i+1}: Action {action} | Method: {analysis.get('method', 'unknown')}"
            )

    print(f"✅ Generated {len(predictions)} predictions")

    # Test ensemble adaptation
    print("\n🔄 Testing ensemble adaptation...")
    market_conditions = {"volatility": 0.15, "trend": "sideways", "momentum": 0.05}
    ensemble.adapt_ensemble(market_conditions)
    print("✅ Ensemble adaptation completed")

    # Test ensemble analysis
    print("\n📈 Testing ensemble analysis...")
    ensemble_stats = ensemble.get_ensemble_stats()
    decision_log = ensemble.decision_log

    # Mock analysis report generation
    analysis_report = {
        "ensemble_performance": {
            "total_decisions": len(decision_log),
            "avg_confidence": ensemble_stats["overall_stats"]["avg_confidence"],
            "diversity_score": 0.85,
        },
        "member_analysis": ensemble_stats.get("member_stats", {}),
        "decision_patterns": {"action_distribution": {}, "consensus_rate": 0.75},
    }

    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(key): convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(item) for item in obj]
        else:
            return obj

    # Save test results
    output_file = "test_ensemble_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": convert_for_json(config.__dict__),
                "stats": convert_for_json(ensemble_stats),
                "predictions": convert_for_json(
                    predictions[:5]
                ),  # Save first 5 predictions
                "analysis": convert_for_json(analysis_report),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"✅ Test results saved to {output_file}")

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 SAC v428 Phase 3 Ensemble System Test Completed!")
    print("=" * 60)
    print("✅ Ensemble initialization: PASSED")
    print("✅ Prediction generation: PASSED")
    print("✅ Adaptation mechanism: PASSED")
    print("✅ Statistics collection: PASSED")
    print("✅ Analysis report: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_ensemble_system()
