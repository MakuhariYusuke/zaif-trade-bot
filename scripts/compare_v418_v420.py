#!/usr/bin/env python3
"""
Compare SAC v418 (original) vs v420 (hold-relaxed) configurations.
"""

import json
import os
import sys
from pathlib import Path

# Ensure we're using the correct Python environment
if sys.version_info < (3, 11):
    print("Error: Python 3.11+ required")
    sys.exit(1)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.unified_trainer.main import main as train_main


def run_comparison():
    """Run comparison between v418 and v420 configurations."""

    configs = [
        ("config/sac_v418_balanced_adjusted_config.json", "SAC v418 (Original)"),
        ("config/sac_v420_hold_relaxed_config.json", "SAC v420 (Hold Relaxed)")
    ]

    results = {}

    for config_path, description in configs:
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            continue

        print(f"\n{'='*80}")
        print(f"🧪 TESTING: {description}")
        print(f"Config: {config_path}")
        print(f"{'='*80}")

        # Load config to get model name
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        model_name = config.get('model_name', 'unknown')

        # Run training
        print(f"🚀 Starting training for {description}...")
        sys.argv = ['compare_training.py', '--config', config_path]

        try:
            # Call main function directly
            from ztb.training.unified_trainer.main import main as trainer_main
            # Set sys.argv for the main function
            original_argv = sys.argv.copy()
            sys.argv = ['compare_training.py', '--config', config_path]
            success = trainer_main()
            sys.argv = original_argv  # Restore original argv

            if success is None:  # main() returns None on success
                success = True

            if success:
                print(f"✅ {description} training completed successfully")
                results[description] = {
                    'success': True,
                    'model_name': model_name,
                    'config_path': config_path
                }
            else:
                print(f"❌ {description} training failed")
                results[description] = {
                    'success': False,
                    'model_name': model_name,
                    'config_path': config_path
                }

        except Exception as e:
            print(f"❌ {description} training failed with error: {e}")
            results[description] = {
                'success': False,
                'model_name': model_name,
                'config_path': config_path,
                'error': str(e)
            }

    # Print comparison summary
    print(f"\n{'='*80}")
    print("📊 COMPARISON RESULTS")
    print(f"{'='*80}")

    successful_configs = [name for name, result in results.items() if result['success']]
    failed_configs = [name for name, result in results.items() if not result['success']]

    print(f"✅ Successful: {len(successful_configs)}")
    for config in successful_configs:
        print(f"   - {config}")

    if failed_configs:
        print(f"❌ Failed: {len(failed_configs)}")
        for config in failed_configs:
            error = results[config].get('error', 'Unknown error')
            print(f"   - {config}: {error}")

    print(f"\n🎯 Recommendation:")
    if len(successful_configs) == 2:
        print("Both configurations completed successfully. Compare the training reports and metrics to determine which performs better.")
    elif len(successful_configs) == 1:
        print(f"Only {successful_configs[0]} completed successfully. This appears to be the better configuration.")
    else:
        print("Both configurations failed. Check the error messages above and fix the issues.")

    return results


if __name__ == "__main__":
    run_comparison()