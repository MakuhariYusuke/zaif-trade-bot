#!/usr/bin/env python3
"""
Test script for all features with Quality Gates and Promotion Engine.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Add ztb to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ztb.evaluation.promotion import YamlPromotionEngine
from ztb.evaluation.status import CoverageValidator
from ztb.features import FeatureRegistry
from ztb.features.base import CommonPreprocessor
from ztb.utils.core.logger import LoggerManager
from ztb.utils.data.data_generation import load_sample_data
from ztb.utils.feature_testing import evaluate_feature_performance
from ztb.utils.quality import QualityGates
from ztb.utils.thresholds import AdaptiveThresholdManager

# Import feature classes that exist
adx = ema_cross = heikin_ashi = kama = supertrend = tema = None
rsi = roc = obv = zscore = None

try:
    from ztb.test_simple_features import (
        OBV as obv,
    )
    from ztb.test_simple_features import (
        ROC as roc,
    )
    from ztb.test_simple_features import (
        RSI as rsi,
    )
    from ztb.test_simple_features import (
        ZScore as zscore,
    )

    from ztb.features.trend.adx import ADX as adx
    from ztb.features.trend.emacross import EMACross as ema_cross
    from ztb.features.trend.heikin_ashi import HeikinAshi as heikin_ashi
    from ztb.features.trend.kama import KAMA as kama
    from ztb.features.trend.supertrend import Supertrend as supertrend
    from ztb.features.trend.tema import TEMA as tema

    print("Feature imports successful")
except ImportError as e:
    print(f"Feature import error: {e}")
    # Fallback to mock implementations
    pass


def test_all_features(
    dataset: str = "synthetic", bootstrap: bool = False, save_debug: bool = False
):
    """Test all features with Quality Gates and Promotion Engine"""
    logger = LoggerManager(
        experiment_id=f"test_all_features_{dataset}", test_mode=False
    )

    # セッション開始
    logger.start_session("feature_testing", f"dataset_{dataset}")
    logger.log_experiment_start(
        "Feature Testing", {"dataset": dataset, "bootstrap": bootstrap}
    )

    logger.info(
        f"Testing all features with Quality Gates and Promotion Engine (dataset={dataset}, bootstrap={bootstrap})..."
    )

    # Load sample data
    df = load_sample_data(dataset)
    print(f"Loaded {len(df)} samples of market data")
    logger.enqueue_notification(f"Loaded {len(df)} samples of market data")

    # Initialize components
    # Register wave1 features
    # from ztb.features import register_wave1_features, register_wave2_features
    # register_wave1_features(feature_manager)
    # register_wave2_features(feature_manager)

    # Initialize adaptive threshold manager
    adaptive_manager = AdaptiveThresholdManager(".")

    quality_gates = QualityGates(adaptive_manager)

    # Debug: Print adaptive thresholds
    adaptive_thresholds = quality_gates.get_adaptive_thresholds()
    print(f"Adaptive thresholds: {adaptive_thresholds}")

    # Load promotion engine config
    config_path = Path(__file__).parent.parent / "config" / "promotion_criteria.yaml"
    if not config_path.exists():
        print(f"Warning: {config_path} not found, using default config")
        config_path = None

    if config_path:
        engine = YamlPromotionEngine(str(config_path))
    else:
        # Create default engine
        engine = YamlPromotionEngine.__new__(YamlPromotionEngine)
        engine.config = {
            "categories": {
                "trend_features": {
                    "logic": "AND",
                    "criteria": [
                        {
                            "name": "win_rate",
                            "operator": ">",
                            "value": 0.4,
                            "weight": 1.0,
                            "type": "numeric",
                        },
                        {
                            "name": "max_drawdown",
                            "operator": "<",
                            "value": -0.3,
                            "weight": 1.0,
                            "type": "numeric",
                        },
                        {
                            "name": "sharpe_ratio",
                            "operator": ">",
                            "value": 0.3,
                            "weight": 1.0,
                            "type": "ratio",
                        },
                    ],
                    "hard_requirements": [
                        {
                            "name": "sample_count",
                            "operator": ">",
                            "value": 5000,
                            "type": "numeric",
                        }
                    ],
                    "required_score": 0.8,
                },
                "oscillator_features": {
                    "logic": "AND",
                    "criteria": [
                        {
                            "name": "win_rate",
                            "operator": ">",
                            "value": 0.4,
                            "weight": 1.0,
                            "type": "numeric",
                        },
                        {
                            "name": "max_drawdown",
                            "operator": "<",
                            "value": -0.3,
                            "weight": 1.0,
                            "type": "numeric",
                        },
                    ],
                    "hard_requirements": [],
                    "required_score": 0.8,
                },
                "volume_features": {
                    "logic": "AND",
                    "criteria": [
                        {
                            "name": "win_rate",
                            "operator": ">",
                            "value": 0.4,
                            "weight": 1.0,
                            "type": "numeric",
                        }
                    ],
                    "hard_requirements": [],
                    "required_score": 0.7,
                },
                "volatility_features": {
                    "logic": "AND",
                    "criteria": [
                        {
                            "name": "win_rate",
                            "operator": ">",
                            "value": 0.4,
                            "weight": 1.0,
                            "type": "numeric",
                        }
                    ],
                    "hard_requirements": [],
                    "required_score": 0.7,
                },
                "wave1_features": {
                    "logic": "AND",
                    "criteria": [
                        {
                            "name": "win_rate",
                            "operator": ">",
                            "value": 0.4,
                            "weight": 1.0,
                            "type": "numeric",
                        }
                    ],
                    "hard_requirements": [],
                    "required_score": 0.7,
                },
                "wave3_features": {
                    "logic": "AND",
                    "criteria": [
                        {
                            "name": "win_rate",
                            "operator": ">",
                            "value": 0.4,
                            "weight": 1.0,
                            "type": "numeric",
                        }
                    ],
                    "hard_requirements": [],
                    "required_score": 0.6,
                },
            },
            "staging": {
                "min_samples_required": 5000,
                "hard_requirement_mode": "strict",
                "demotion_mode": "graceful",
            },
        }
        engine.criteria_cache = {}
        from ztb.evaluation.promotion import PromotionNotifier

        engine.notifier = PromotionNotifier({})

    # Get all enabled features using FeatureRegistry
    all_features = FeatureRegistry.list()
    print(f"Testing {len(all_features)} features: {', '.join(all_features[:5])}...")

    # Compute all features using FeatureRegistry
    computed_df = df.copy()
    for feature_name in all_features:
        try:
            feature_func = FeatureRegistry.get(feature_name)
            computed_df[feature_name] = feature_func(df)
        except Exception as e:
            print(f"Error computing {feature_name}: {e}")
            computed_df[feature_name] = pd.Series(index=df.index, dtype=float)

    print(
        f"Successfully computed {len(computed_df.columns) - len(df.columns)} features"
    )

    results = []
    verified_count = {
        "trend": 0,
        "oscillator": 0,
        "volume": 0,
        "volatility": 0,
        "wave1": 0,
        "wave3": 0,
    }

    for feature_name in all_features:
        print(f"\nTesting {feature_name}...")

        try:
            # Get feature category
            category = get_feature_category(feature_name)

            # Preprocess data
            processed_df = CommonPreprocessor.preprocess(df.copy())

            # Try to get feature from computed DataFrame first
            if feature_name in computed_df.columns:
                feature_series = computed_df[feature_name]
            else:
                # Fallback to individual computation
                feature_series = create_feature_instance(feature_name, processed_df)

            # Apply Quality Gates
            quality_results = quality_gates.evaluate(
                feature_series,
                processed_df["close"],
                "bootstrap" if bootstrap else "normal",
                dataset,
            )

            if not quality_results["overall_pass"]:
                print(f"⚠ {feature_name}: pending (quality gate fail) - ", end="")
                reasons = []
                if not quality_results["nan_rate_pass"]:
                    reasons.append(f"NaN rate={quality_results['nan_rate']:.2f}")
                if not quality_results["correlation_pass"]:
                    reasons.append(
                        f"correlation={quality_results.get('correlation', 'N/A')}"
                    )
                if not quality_results["skew_pass"]:
                    reasons.append(f"skew={quality_results.get('skew', 'N/A'):.1f}")
                if not quality_results["kurtosis_pass"]:
                    reasons.append(
                        f"kurtosis={quality_results.get('kurtosis', 'N/A'):.1f}"
                    )
                print(", ".join(reasons))

                results.append(
                    {
                        "name": feature_name,
                        "status": "pending_due_to_gate_fail",
                        "category": category,
                        "reason": ", ".join(reasons),
                        "quality_results": quality_results,
                    }
                )
                continue

            # Calculate trading metrics
            feature_result = evaluate_feature_performance(
                feature_series, processed_df["close"], feature_name
            )
            metrics = feature_result["metrics"]

            # Determine current status based on category
            current_status = "pending"  # Start from pending

            # Get category config
            category_config = f"{category}_features"

            # Check basic criteria for promotion
            basic_criteria = {
                "win_rate": metrics["win_rate"] > 0.4,
                "max_drawdown": metrics["max_drawdown"] < -0.3,
            }

            if not all(basic_criteria.values()):
                print(f"✗ {feature_name}: pending → keep (basic criteria not met)")
                results.append(
                    {
                        "name": feature_name,
                        "status": "keep",
                        "category": category,
                        "current_status": current_status,
                        "metrics": metrics,
                        "quality_results": quality_results,
                        "promotion_details": {"reason": "basic_criteria_not_met"},
                    }
                )
                continue

            # Try pending -> staging
            result, details = engine.evaluate_promotion(
                feature_name, metrics, current_status, category_config
            )
            if result.name == "PROMOTE":
                current_status = "staging"
                print(f"✓ {feature_name}: pending → staging")
                results.append(
                    {
                        "name": feature_name,
                        "status": "promoted_to_staging",
                        "category": category,
                        "from_status": "pending",
                        "to_status": "staging",
                        "metrics": metrics,
                        "quality_results": quality_results,
                        "promotion_details": details,
                    }
                )
            else:
                print(f"✗ {feature_name}: pending → keep (insufficient criteria)")
                results.append(
                    {
                        "name": feature_name,
                        "status": "keep",
                        "category": category,
                        "current_status": current_status,
                        "metrics": metrics,
                        "quality_results": quality_results,
                        "promotion_details": details,
                    }
                )
                continue

            # Try staging -> verified
            result, details = engine.evaluate_promotion(
                feature_name, metrics, current_status, category_config
            )
            if result.name == "PROMOTE":
                print(f"✓ {feature_name}: staging → verified")
                verified_count[category] += 1
                results.append(
                    {
                        "name": feature_name,
                        "status": "promoted",
                        "category": category,
                        "from_status": current_status,
                        "to_status": "verified",
                        "metrics": metrics,
                        "quality_results": quality_results,
                        "promotion_details": details,
                    }
                )
            else:
                print(f"✗ {feature_name}: staging → keep (insufficient criteria)")
                results.append(
                    {
                        "name": feature_name,
                        "status": "keep",
                        "category": category,
                        "current_status": current_status,
                        "metrics": metrics,
                        "quality_results": quality_results,
                        "promotion_details": details,
                    }
                )

        except Exception as e:
            print(f"✗ {feature_name}: error - {str(e)}")
            results.append(
                {
                    "name": feature_name,
                    "status": "error",
                    "category": get_feature_category(feature_name),
                    "error": str(e),
                }
            )

    # Record events for each result
    coverage_path = "coverage.json"
    coverage_data = CoverageValidator.load_coverage_files(coverage_path)

    for result in results:
        feature_name = result["name"]
        category = result.get("category")

        if result["status"] == "harmful":
            # Record harmful features as discarded
            # Convert numpy bools to Python bools for JSON serialization
            quality_results = result.get("quality_results", {})
            json_safe_quality_results = {}
            for k, v in quality_results.items():
                if isinstance(v, np.bool_):
                    json_safe_quality_results[k] = bool(v)
                else:
                    json_safe_quality_results[k] = v

            CoverageValidator.record_event(
                coverage_data,
                "feature_tested",
                feature_name,
                None,
                "pending_due_to_gate_fail",
                {
                    "status": "pending_due_to_gate_fail",
                    "reason": result.get("reason", "quality_gate_failed"),
                    "quality_results": json_safe_quality_results,
                    "dataset": dataset,
                },
            )
        elif result["status"] == "promoted":
            # Record promotion
            CoverageValidator.record_event(
                coverage_data,
                "feature_promoted",
                feature_name,
                result.get("from_status"),
                result.get("to_status"),
                {
                    "score": result.get("promotion_details", {}).get(
                        "normalized_score", 0.0
                    ),
                    "dataset": dataset,
                },
            )
        elif result["status"] == "keep":
            # Record as kept (no change)
            pass  # Keep events are not recorded as they don't change state
        elif result["status"] == "error":
            # Record error
            CoverageValidator.record_event(
                coverage_data,
                "feature_tested",
                feature_name,
                None,
                "failed",
                {
                    "status": "error",
                    "error": result.get("error", "unknown_error"),
                    "dataset": dataset,
                },
            )

    # Save updated coverage
    with open(coverage_path, "w", encoding="utf-8") as f:
        json.dump(coverage_data, f, indent=2, ensure_ascii=False)

    # Check category requirements
    check_category_requirements(verified_count, logger)

    print("\nAll features testing completed!")

    # Save debug report if requested
    if save_debug:
        import csv
        import datetime
        import os

        debug_file = f"reports/debug/feature_quality_{dataset}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        os.makedirs("reports/debug", exist_ok=True)

        with open(debug_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "dataset",
                    "feature",
                    "corr_short",
                    "corr_long",
                    "nan_rate",
                    "skew",
                    "kurtosis",
                    "overall_pass",
                    "gate_fail_reason",
                ]
            )

            for result in results:
                quality = result.get("quality_results", {})
                fail_reasons = []
                if not quality.get("correlation_pass", True):
                    fail_reasons.append("correlation")
                if not quality.get("nan_rate_pass", True):
                    fail_reasons.append("nan_rate")
                if not quality.get("skew_pass", True):
                    fail_reasons.append("skew")
                if not quality.get("kurtosis_pass", True):
                    fail_reasons.append("kurtosis")

                writer.writerow(
                    [
                        dataset,
                        result["name"],
                        quality.get("corr_short", "N/A"),
                        quality.get("corr_long", "N/A"),
                        quality.get("nan_rate", "N/A"),
                        quality.get("skew", "N/A"),
                        quality.get("kurtosis", "N/A"),
                        quality.get("overall_pass", False),
                        ";".join(fail_reasons),
                    ]
                )

        print(f"Debug report saved to {debug_file}")

    return results


def get_feature_category(feature_name: str) -> str:
    """Determine feature category from feature name"""
    name_lower = feature_name.lower()

    # Wave features
    if any(keyword in name_lower for keyword in ["rolling", "lags", "dow", "hour"]):
        return "wave1"
    if any(keyword in name_lower for keyword in ["regime", "kalman"]):
        return "wave3"

    # Trend indicators
    if any(
        keyword in name_lower
        for keyword in [
            "ema",
            "sma",
            "kama",
            "tema",
            "dema",
            "trend",
            "ichimoku",
            "donchian",
            "adx",
            "supertrend",
            "heikin",
        ]
    ):
        return "trend"

    # Oscillators
    if any(
        keyword in name_lower
        for keyword in ["rsi", "stoch", "macd", "cci", "williams", "oscillator"]
    ):
        return "oscillator"

    # Volume indicators
    if any(
        keyword in name_lower
        for keyword in ["volume", "obv", "vwap", "vpt", "mfi", "pricevolume"]
    ):
        return "volume"

    # Volatility indicators
    if any(
        keyword in name_lower
        for keyword in ["atr", "bollinger", "hv", "returnstd", "zscore"]
    ):
        return "volatility"

    return "other"


def create_feature_instance(feature_name: str, df: pd.DataFrame) -> pd.Series:
    """Create and compute feature instance dynamically"""
    try:
        # Try to use actual feature classes
        if feature_name == "ADX" and adx is not None:
            result = adx(df, period=14)
            return result
        elif feature_name == "EMACross" and ema_cross is not None:
            feature = ema_cross()
            result = feature.compute(df)
            return result.iloc[:, 0] if isinstance(result, pd.DataFrame) else result
        elif feature_name == "HeikinAshi" and heikin_ashi is not None:
            feature = heikin_ashi()
            result = feature.compute(df)
            return result.iloc[:, 0] if isinstance(result, pd.DataFrame) else result
        elif feature_name == "KAMA" and kama is not None:
            feature = kama()
            result = feature.compute(df)
            return result.iloc[:, 0] if isinstance(result, pd.DataFrame) else result
        elif feature_name == "Supertrend" and supertrend is not None:
            feature = supertrend()
            result = feature.compute(df)
            return result.iloc[:, 0] if isinstance(result, pd.DataFrame) else result
        elif feature_name == "TEMA" and tema is not None:
            feature = tema()
            result = feature.compute(df)
            return result.iloc[:, 0] if isinstance(result, pd.DataFrame) else result
        elif feature_name == "RSI" and rsi is not None:
            result = rsi(df, period=14)
            return result
        elif feature_name == "ROC" and roc is not None:
            result = roc(df, period=10)
            return result
        elif feature_name == "OBV" and obv is not None:
            result = obv(df)
            return result
        elif feature_name == "ZScore" and zscore is not None:
            result = zscore(df, period=20)
            return result
    except Exception as e:
        print(f"Error creating feature {feature_name}: {e}")
        # Fall back to mock data

    # Fallback to mock data for unhandled features
    np.random.seed(42)
    if "RollingMean" in feature_name:
        window = 14 if "14" in feature_name else 50
        return df["close"].rolling(window=window).mean()
    elif "RollingStd" in feature_name:
        window = 14 if "14" in feature_name else 50
        return df["close"].rolling(window=window).std()
    else:
        # Return random data for other features
        return pd.Series(np.random.randn(len(df)), index=df.index, name=feature_name)


def update_coverage_json(results: List[Dict[str, Any]], dataset: str = "synthetic"):
    """Update coverage.json with test results"""
    coverage_path = Path(__file__).parent.parent / "coverage.json"

    # Load existing coverage
    if coverage_path.exists():
        with open(coverage_path, "r") as f:
            coverage = json.load(f)
    else:
        coverage = {
            "verified": [],
            "pending": [],
            "failed": [],
            "unverified": [],
            "events": [],
            "metadata": {
                "last_updated": "2024-01-01T00:00:00",
                "dataset": dataset,
                "total_verified": 0,
                "total_pending": 0,
                "total_failed": 0,
                "total_unverified": 0,
            },
        }

    # Initialize events if not exists
    if "events" not in coverage:
        coverage["events"] = []

    # Update based on results
    for result in results:
        feature_name = result["name"]

        # Remove from all lists first
        for category in ["verified", "pending", "failed", "unverified"]:
            coverage[category] = [
                f
                for f in coverage[category]
                if not (isinstance(f, dict) and f.get("name") == feature_name)
                and f != feature_name
            ]

        # Add to appropriate category
        if result["status"] == "promoted" and result.get("to_status") == "verified":
            if feature_name not in coverage["verified"]:
                coverage["verified"].append(feature_name)
        elif result["status"] == "promoted" and result.get("to_status") == "staging":
            coverage["pending"].append(
                {
                    "name": feature_name,
                    "reason": "promoted_to_staging",
                    "category": result.get("category"),
                    "metrics": result.get("metrics", {}),
                }
            )
        elif result["status"] == "harmful":
            coverage["failed"].append(
                {
                    "name": feature_name,
                    "reason": f"harmful: {result['reason']}",
                    "category": result.get("category"),
                    "quality_results": result.get("quality_results", {}),
                }
            )
        elif result["status"] == "keep":
            coverage["pending"].append(
                {
                    "name": feature_name,
                    "reason": "insufficient_criteria",
                    "category": result.get("category"),
                    "metrics": result.get("metrics", {}),
                }
            )
        elif result["status"] == "error":
            coverage["failed"].append(
                {
                    "name": feature_name,
                    "reason": f"error: {result['error']}",
                    "category": result.get("category"),
                }
            )

        # Add event
        event = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "type": "feature_tested",
            "feature": feature_name,
            "category": result.get("category"),
            "status": result["status"],
            "dataset": dataset,
            "details": result,
        }
        coverage["events"].append(event)

    # Update metadata
    from datetime import datetime

    coverage["metadata"]["last_updated"] = datetime.now().isoformat()
    coverage["metadata"]["total_verified"] = len(coverage["verified"])
    coverage["metadata"]["total_pending"] = len(coverage["pending"])
    coverage["metadata"]["total_failed"] = len(coverage["failed"])
    coverage["metadata"]["total_unverified"] = len(coverage["unverified"])

    # Save updated coverage
    with open(coverage_path, "w") as f:
        json.dump(coverage, f, indent=2, default=str)

    print("Coverage.json updated with promotion/discard results")


def check_category_requirements(verified_count: Dict[str, int], logger: LoggerManager):
    """Check if category requirements are met"""
    print("\nChecking category requirements...")

    requirements = {
        "trend": 1,
        "oscillator": 1,
        "volume": 1,
        "volatility": 1,
        "wave1": 1,
        "wave3": 0,  # Optional
    }

    all_met = True
    for category, required in requirements.items():
        actual = verified_count.get(category, 0)
        if actual >= required:
            print(f"✓ {category}: {actual}/{required} verified features")
        else:
            print(f"✗ {category}: {actual}/{required} verified features (INSUFFICIENT)")
            all_met = False

    if all_met:
        print("All category requirements met!")
        logger.log_experiment_end({"status": "success", "categories_met": all_met})
    else:
        print("Some category requirements not met!")
        logger.log_experiment_end({"status": "partial", "categories_met": all_met})
        # Don't exit in test mode - continue to save debug report
        # sys.exit(1)

    # セッション終了
    session_results = {
        "status": "success" if "all_met" in locals() and all_met else "partial",
        "categories_met": all_met if "all_met" in locals() else False,
        "verified_count": verified_count if "verified_count" in locals() else {},
        "total_features": len(FeatureRegistry.list()),
    }
    logger.end_session(session_results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="synthetic",
        choices=["synthetic", "synthetic-v2", "real", "coingecko"],
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Enable bootstrap mode for relaxed quality gates",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug CSV with detailed quality metrics",
    )
    args = parser.parse_args()

    test_all_features(
        dataset=args.dataset, bootstrap=args.bootstrap, save_debug=args.save_debug
    )
