#!/usr/bin/env python3
"""
SAC v420 Parameter Tuning Script

Executes systematic parameter tuning for SAC v420 using short test configurations.
Tests learning rate, buffer size, batch size, entropy coefficient, reward scale, and gamma.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure we're using the correct Python environment
if sys.version_info < (3, 11):
    print("Error: Python 3.11+ required")
    sys.exit(1)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class SACParameterTuner:
    """SAC parameter tuning orchestrator."""

    def __init__(self):
        self.configs_dir = Path("configs")
        self.results_dir = Path("results/sac_v420_tuning")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Define parameter sweep configurations
        self.parameter_configs = {
            "baseline": ["sac_v420_baseline_1k.json", "sac_v420_baseline_5k.json"],
            "learning_rate": [
                "sac_v420_lr_sweep_0.0001_1k.json",
                "sac_v420_lr_sweep_0.001_1k.json",
            ],
            "buffer_size": [
                "sac_v420_buffer_sweep_100k_1k.json",
                "sac_v420_buffer_sweep_200k_1k.json",
            ],
            "batch_size": [
                "sac_v420_batch_sweep_64_1k.json",
                "sac_v420_batch_sweep_256_1k.json",
            ],
            "entropy_coef": [
                "sac_v420_ent_coef_sweep_0.01_1k.json",
                "sac_v420_ent_coef_sweep_0.1_1k.json",
            ],
            "reward_scale": [
                "sac_v420_reward_scale_sweep_0.1_1k.json",
                "sac_v420_reward_scale_sweep_10.0_1k.json",
            ],
            "gamma": [
                "sac_v420_gamma_sweep_0.95_1k.json",
                "sac_v420_gamma_sweep_0.999_1k.json",
            ],
        }

    def run_single_config(self, config_file: str) -> Dict[str, Any]:
        """Run training with a single configuration file."""
        config_path = self.configs_dir / config_file

        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return {"status": "failed", "error": "config_not_found"}

        print(f"🚀 Starting training with config: {config_file}")

        start_time = time.time()

        try:
            # Load config to get model name
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            model_name = config.get("model_name", "unknown")

            # Run training using subprocess
            cmd = [
                sys.executable,
                "-m",
                "ztb.training.unified_trainer.main",
                "--config",
                str(config_path),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )  # 5 minute timeout

            end_time = time.time()
            duration = end_time - start_time

            if result.returncode == 0:
                print(f"✅ Training completed in {duration:.2f} seconds")
                return {
                    "status": "success",
                    "config_file": config_file,
                    "model_name": model_name,
                    "duration": duration,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            else:
                print(
                    f"❌ Training failed after {duration:.2f} seconds: {result.stderr}"
                )
                return {
                    "status": "failed",
                    "config_file": config_file,
                    "duration": duration,
                    "error": result.stderr,
                    "stdout": result.stdout,
                }

        except subprocess.TimeoutExpired:
            end_time = time.time()
            duration = end_time - start_time
            print(f"❌ Training timed out after {duration:.2f} seconds")
            return {
                "status": "failed",
                "config_file": config_file,
                "duration": duration,
                "error": "timeout",
            }
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time

            print(f"❌ Training failed after {duration:.2f} seconds: {e}")
            return {
                "status": "failed",
                "config_file": config_file,
                "duration": duration,
                "error": str(e),
            }

    def run_parameter_sweep(self, parameter_group: str) -> List[Dict[str, Any]]:
        """Run all configurations for a parameter group."""
        if parameter_group not in self.parameter_configs:
            print(f"❌ Unknown parameter group: {parameter_group}")
            return []

        configs = self.parameter_configs[parameter_group]
        results = []

        print(f"🔬 Running parameter sweep for: {parameter_group}")
        print(f"📋 Configurations to test: {len(configs)}")
        print()

        for i, config_file in enumerate(configs, 1):
            print(f"[{i}/{len(configs)}] Testing: {config_file}")
            result = self.run_single_config(config_file)
            results.append(result)
            print()

        return results

    def run_full_tuning_suite(self) -> Dict[str, List[Dict[str, Any]]]:
        """Run complete parameter tuning suite."""
        print("🎯 Starting SAC v420 Parameter Tuning Suite")
        print("=" * 50)

        all_results = {}

        # Run baseline tests first
        print("📊 Phase 1: Baseline Validation")
        all_results["baseline"] = self.run_parameter_sweep("baseline")

        # Run parameter sweeps
        parameter_groups = [
            "learning_rate",
            "buffer_size",
            "batch_size",
            "entropy_coef",
            "reward_scale",
            "gamma",
        ]

        for group in parameter_groups:
            print(f"📊 Phase: {group.replace('_', ' ').title()} Sweep")
            all_results[group] = self.run_parameter_sweep(group)

        # Save results summary
        self.save_results_summary(all_results)

        print("✅ Parameter tuning suite completed!")
        return all_results

    def save_results_summary(self, results: Dict[str, List[Dict[str, Any]]]) -> None:
        """Save tuning results summary."""
        summary_file = self.results_dir / "tuning_summary.json"

        # Calculate summary statistics
        summary = {
            "timestamp": time.time(),
            "total_configs_tested": sum(
                len(group_results) for group_results in results.values()
            ),
            "successful_runs": sum(
                1
                for group_results in results.values()
                for result in group_results
                if result["status"] == "success"
            ),
            "failed_runs": sum(
                1
                for group_results in results.values()
                for result in group_results
                if result["status"] == "failed"
            ),
            "total_duration": sum(
                result.get("duration", 0)
                for group_results in results.values()
                for result in group_results
            ),
            "results_by_group": {},
        }

        for group_name, group_results in results.items():
            group_summary = {
                "configs_tested": len(group_results),
                "successful": sum(1 for r in group_results if r["status"] == "success"),
                "failed": sum(1 for r in group_results if r["status"] == "failed"),
                "avg_duration": sum(r.get("duration", 0) for r in group_results)
                / len(group_results),
                "results": group_results,
            }
            summary["results_by_group"][group_name] = group_summary

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"📄 Results summary saved to: {summary_file}")


def main():
    """Main entry point for parameter tuning."""
    tuner = SACParameterTuner()

    # Check if configs directory exists
    if not tuner.configs_dir.exists():
        print(f"❌ Configs directory not found: {tuner.configs_dir}")
        return

    # Run full tuning suite
    results = tuner.run_full_tuning_suite()

    # Print final summary
    print("\n" + "=" * 50)
    print("🎯 TUNING SUITE SUMMARY")
    print("=" * 50)

    total_configs = sum(len(group_results) for group_results in results.values())
    successful = sum(
        1
        for group_results in results.values()
        for result in group_results
        if result["status"] == "success"
    )
    total_duration = sum(
        result.get("duration", 0)
        for group_results in results.values()
        for result in group_results
    )

    print(f"Total configurations tested: {total_configs}")
    print(f"Successful runs: {successful}")
    print(f"Failed runs: {total_configs - successful}")
    print(f"Total duration: {total_duration:.1f} seconds")

    if successful > 0:
        print("\n✅ Parameter tuning completed successfully!")
        print("📊 Review results in: results/sac_v420_tuning/tuning_summary.json")
    else:
        print("\n❌ All tuning runs failed. Check logs for details.")


if __name__ == "__main__":
    main()
