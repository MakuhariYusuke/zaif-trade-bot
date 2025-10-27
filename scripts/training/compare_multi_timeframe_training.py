#!/usr/bin/env python3
"""
Multi-Timeframe Feature Comparison Training Script

Compares SAC v435 training performance with and without multi-timeframe features.
Each scenario runs for 10,000 steps to evaluate feature impact.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any
import json
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent.parent  # scripts/training -> scripts -> project_root
sys.path.insert(0, str(project_root))

from ztb.config.schema import ZaifTradeBotConfig
from ztb.training.unified_trainer.trainer import UnifiedTrainer
from ztb.training.unified_trainer.algorithms import create_algorithm_trainer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_training_scenario(config_name: str, timesteps: int = 10000) -> Dict[str, Any]:
    """
    Run a single training scenario.

    Args:
        config_name: Name of the configuration file (without .json)
        timesteps: Number of training timesteps

    Returns:
        Training results summary
    """
    config_path = f"config/{config_name}.json"

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Starting training with config: {config_name}, timesteps: {timesteps}")

    # Load and convert config to UnifiedTrainer format
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Convert to UnifiedTrainer expected format
        config_dict_for_trainer = {
            "training": {
                "algorithm": config_dict["training"]["algorithm"],
                "total_timesteps": timesteps,  # Use the parameter value
                "model_name": config_dict["training"]["model_name"],
            },
            "data_path": config_dict["training"]["data_config"]["csv_path"],
            "data_config": config_dict["training"]["data_config"],
            "environment": config_dict["training"]["environment"],
            "features": config_dict["training"]["features"],
            "sac_hyperparameters": config_dict["training"]["sac_hyperparameters"],
        }
        
        # Create algorithm trainer directly
        algorithm = config_dict_for_trainer["training"]["algorithm"]
        trainer = create_algorithm_trainer(algorithm, config_dict_for_trainer, logger)
        success = trainer.train(total_timesteps=timesteps)
        
        return {
            "config": config_name,
            "timesteps": timesteps,
            "success": success,
            "output": "Training completed" if success else "Training failed",
            "error": "" if success else "Training execution failed",
            "return_code": 0 if success else 1
        }

    except Exception as e:
        logger.error(f"Training failed for {config_name}: {e}")
        return {
            "config": config_name,
            "timesteps": timesteps,
            "success": False,
            "output": "",
            "error": str(e),
            "return_code": -1
        }

    except subprocess.TimeoutExpired:
        logger.error(f"Training timed out for {config_name}")
        return {
            "config": config_name,
            "timesteps": timesteps,
            "success": False,
            "output": "",
            "error": "Training timed out",
            "return_code": -1
        }
    except Exception as e:
        logger.error(f"Training failed for {config_name}: {e}")
        return {
            "config": config_name,
            "timesteps": timesteps,
            "success": False,
            "output": "",
            "error": str(e),
            "return_code": -1
        }


def save_comparison_results(results: Dict[str, Dict[str, Any]], output_file: str) -> None:
    """
    Save comparison results to JSON file.

    Args:
        results: Dictionary of training results
        output_file: Path to output file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Comparison results saved to: {output_file}")


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(
        description="Compare SAC v435 training with/without multi-timeframe features"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10000,
        help="Number of training timesteps per scenario (default: 10000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="multi_timeframe_comparison_10k_results.json",
        help="Output file for comparison results"
    )
    parser.add_argument(
        "--with-multi-timeframe",
        action="store_true",
        help="Run only multi-timeframe enabled scenario"
    )
    parser.add_argument(
        "--without-multi-timeframe",
        action="store_true",
        help="Run only multi-timeframe disabled scenario"
    )

    args = parser.parse_args()

    # Define scenarios
    scenarios = []

    if not args.with_multi_timeframe and not args.without_multi_timeframe:
        # Run both scenarios
        scenarios = ["sac_v435_unified_config", "sac_v435_unified_config_no_multi_timeframe"]
    elif args.with_multi_timeframe:
        scenarios = ["sac_v435_unified_config"]
    elif args.without_multi_timeframe:
        scenarios = ["sac_v435_unified_config_no_multi_timeframe"]

    logger.info(f"Running {len(scenarios)} training scenarios with {args.timesteps} timesteps each")

    results = {}

    for scenario in scenarios:
        logger.info(f"Running scenario: {scenario}")
        result = run_training_scenario(scenario, args.timesteps)
        results[scenario] = result

        # Log summary
        if result["success"]:
            logger.info(f"✅ {scenario}: Training completed successfully")
        else:
            logger.error(f"❌ {scenario}: Training failed - {result['error']}")

    # Save results
    save_comparison_results(results, args.output)

    # Print summary
    print("\n" + "="*60)
    print("MULTI-TIMEFRAME FEATURE COMPARISON SUMMARY")
    print("="*60)

    for scenario, result in results.items():
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(f"{scenario}: {status}")

        if "multi_timeframe" in scenario:
            feature_status = "DISABLED" if "no_multi_timeframe" in scenario else "ENABLED"
            print(f"  Multi-timeframe features: {feature_status}")

    print("="*60)


if __name__ == "__main__":
    main()