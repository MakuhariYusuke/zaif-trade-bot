#!/usr/bin/env python3
"""
Smoke Test for Reinforcement Learning Experiments
高速な健全性チェックテスト - CI/CD統合用

Usage:
    python ztb/experiments/smoke_test.py --steps 10000 --dataset coingecko
"""

import sys
import time
import psutil
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from ztb.experiments.base import ExperimentBase, ExperimentResult
from ztb.utils import LoggerManager
from ztb.utils.feature_testing import evaluate_feature_performance


class SmokeTestExperiment(ExperimentBase):
    """Smoke test experiment for quick validation"""

    PASS_RATE_THRESHOLD = 0.5  # 成功判定の閾値

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "smoke_test")
        self.total_steps = config.get('total_steps', 10000)
        self.dataset = config.get('dataset', 'coingecko')

        # Performance tracking
        self.start_memory = None
        self.peak_memory = 0
        self.step_times = []
        self.feature_pass_rates = []

    def run(self) -> ExperimentResult:
        """Execute smoke test"""
        self.logger.info(f"Starting smoke test: {self.total_steps} steps, dataset: {self.dataset}")

        # Record initial memory
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Load dataset
        try:
            data = load_sample_data(self.dataset, cache_dir="data/cache")
            self.logger.info(f"Loaded dataset: {len(data)} samples")
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            return ExperimentResult(
                experiment_name=self.experiment_name,
                timestamp=datetime.now().isoformat(),
                status="failed",
                config=self.config,
                metrics={"error": str(e)},
                artifacts={}
            )

        # Prepare close column slice for feature evaluation
        close_slice = data.iloc[:100]['close']

        # Run smoke test steps
        for step in range(self.total_steps):
            step_start = time.time()

            try:
                # Simulate feature testing
                features = {
                    'rsi': [30, 50, 70][step % 3],
                    'macd': step * 0.01,
                    'volume': step * 100
                }

                # Evaluate features
                result = evaluate_feature_performance(close_slice, close_slice, 'rsi')
                pass_rate = result['metrics']['win_rate']
                self.feature_pass_rates.append(pass_rate)

                # Track memory usage
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                self.peak_memory = max(self.peak_memory, current_memory)

                # Log progress every 1000 steps
                if step % 1000 == 0:
                    recent_rates = self.feature_pass_rates[-1000:]
                    if len(recent_rates) > 0:
                        avg_pass_rate = sum(recent_rates) / len(recent_rates)
                    else:
                        avg_pass_rate = 0.0
                    self.logger.info(f"Step {step}/{self.total_steps}: Pass rate {avg_pass_rate:.2%}, Memory: {current_memory:.1f} MB")

            except Exception as e:
                self.logger.error(f"Error at step {step}: {e}")
                return ExperimentResult(
                    experiment_name=self.experiment_name,
                    timestamp=datetime.now().isoformat(),
                    status="failed",
                    config=self.config,
                    metrics={"error": str(e), "failed_at_step": step},
                    artifacts={}
                )

            step_time = time.time() - step_start
            self.step_times.append(step_time)

        # Calculate final metrics
        avg_pass_rate = sum(self.feature_pass_rates) / len(self.feature_pass_rates)
        avg_step_time = sum(self.step_times) / len(self.step_times)
        estimated_total_time = avg_step_time * 100000  # For 100k steps

        metrics = {
            'total_steps': self.total_steps,
            'avg_feature_pass_rate': avg_pass_rate,
            'avg_step_time': avg_step_time,
            'memory_peak_mb': self.peak_memory,
            'estimated_total_time_hours': estimated_total_time / 3600,
            'dataset_size': len(data)
        }
        return ExperimentResult(
            experiment_name=self.experiment_name,
            timestamp=datetime.now().isoformat(),
            status="success" if avg_pass_rate > self.PASS_RATE_THRESHOLD else "warning",
            config=self.config,
            metrics=metrics,
            artifacts={}
        )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run smoke test for RL experiments")
    parser.add_argument('--steps', type=int, default=10000,
                       help='Number of steps to run (default: 10000)')
    parser.add_argument('--dataset', type=str, default='coingecko',
                       choices=['synthetic', 'synthetic-v2', 'coingecko'],
                       help='Dataset to use (default: coingecko)')
    parser.add_argument('--notify-discord', action='store_true',
                       help='Send Discord notification on completion')

    args = parser.parse_args()

    # Setup logging
    logger_manager = LoggerManager(experiment_id="smoke_test")

    config = {
        'total_steps': args.steps,
        'dataset': args.dataset,
        'notify_discord': args.notify_discord
    }

    experiment = SmokeTestExperiment(config)
    result = experiment.run()

    # Print summary
    print(f"\nSmoke Test Results:")
    print(f"Status: {result.status}")
    print(f"Steps: {result.metrics.get('total_steps', 0)}")
    print(f"Pass Rate: {result.metrics.get('avg_feature_pass_rate', 0):.2%}")
    print(f"Memory Peak: {result.metrics.get('memory_peak_mb', 0):.1f} MB")
    if result.status == "success":
        print("✅ Smoke test passed!")
        sys.exit(0)
    else:
        print("❌ Smoke test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()