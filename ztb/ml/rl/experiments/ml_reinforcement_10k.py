#!/usr/bin/env python3
"""
10k step reinforcement learning experiment execution script
ExperimentBase class-based feature evaluation experiment with strategy support
"""

import os
import sys
import time
import json
import psutil
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Union

# Local module imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from ztb.experiments.base import ScalingExperiment
from ztb.utils.error_handler import catch_and_notify
from ztb.utils.checkpoint import HAS_LZ4
from ztb.utils.parallel_experiments import ResourceMonitor
from ztb.utils import LoggerManager


# Type definitions for better type safety
from dataclasses import dataclass

@dataclass
class StepResult:
    """ステップ結果のデータ構造"""
    step: int
    reward: float
    done: bool
    info: Dict[str, Any]

@dataclass
class CheckpointData:
    """チェックポイントデータのデータ構造"""
    episode: int
    total_steps: int
    best_reward: float
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    step_results: List[StepResult]
MetricsData = Dict[str, Union[int, float, str, List[float], List[int], Dict[str, Any]]]


class MLReinforcement10KExperiment(ScalingExperiment):
    """10k step reinforcement learning experiment class with strategy support"""

    def __init__(self, config: Dict[str, Any]):
        self.strategy = config.get('strategy', 'generalization')
        self.experiment_name = config.get('name', f"ml_reinforcement_10k_{self.strategy}")
        total_steps = config.get('total_steps', 10000)  # 10k steps for testing
        
        # Dynamic checkpoint interval setting
        checkpoint_interval = 1000  # 1k steps for 10k experiment
        
        config['checkpoint_interval'] = checkpoint_interval
        
        super().__init__(config, total_steps=total_steps)  # type: ignore
        self.dataset = config.get('dataset', 'coingecko')
        self.current_step = 0  # Track current step

        # LoggerManager for 10k experiment
        self.logger_manager = LoggerManager(
            experiment_id=self.experiment_name,
            experiment_type="10k"
        )

        # Set metrics callback for AsyncNotifier
        if self.logger_manager.async_notifier:
            self.logger_manager.async_notifier.set_metrics_callback(self._get_current_metrics)

        # CheckpointManager settings with differential checkpoints
        self.checkpoint_manager.max_queue_size = 5
        self.checkpoint_manager.compress = "lz4" if HAS_LZ4 else "zstd"
        self.checkpoint_manager.differential = True  # Enable differential checkpoints

        # ResourceMonitor for memory leak monitoring
        self.resource_monitor = ResourceMonitor(log_interval_seconds=60)  # 1 minute interval

        # Strategy-specific parameters
        self._setup_strategy_params()

        # Monitoring data
        self.monitoring_data: List[Dict[str, Any]] = []
        self.memory_usage: List[float] = []
        self.step_results: List[StepResult] = []

        # Statistics
        self.harmful_count: int = 0
        self.pending_count: int = 0
        self.verified_count: int = 0
        self.reward_history: List[float] = []
        self.pnl_history: List[float] = []

    def _setup_strategy_params(self):
        """Setup strategy-specific parameters"""
        if self.strategy == 'generalization':
            self.exploration_rate = 0.3
            self.learning_rate = 0.001
            self.risk_multiplier = 0.7
        elif self.strategy == 'aggressive':
            self.exploration_rate = 0.5
            self.learning_rate = 0.002
            self.risk_multiplier = 1.2
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def step(self, step_num: int) -> StepResult:
        """Execute a single reinforcement learning step"""
        try:
            step_result = self._run_single_step(step_num)
            self.step_results.append(step_result)

            # Update monitoring data
            if 'reward' in step_result.info:
                self.reward_history.append(step_result.info['reward'])
            if 'pnl' in step_result.info:
                self.pnl_history.append(step_result.info['pnl'])

            # Update statistics (simplified for now)
            # These would be calculated based on actual step results
            if random.random() < 0.1:  # 10% chance of harmful
                self.harmful_count += 1
            if random.random() < 0.2:  # 20% chance of pending
                self.pending_count += 1
            if random.random() < 0.7:  # 70% chance of verified
                self.verified_count += 1

            # ResourceMonitorでメモリリーク監視を実行
            self.resource_monitor.log_resources(self.experiment_name)

            return step_result

        except Exception as e:
            self.logger_manager.logger.error(f"Error in step {step_num}: {e}")
            return StepResult(step=step_num, reward=0.0, done=True, info={'error': str(e)})

    def get_checkpoint_data(self) -> CheckpointData:
        """Get data for checkpoint saving"""
        return CheckpointData(
            episode=self.current_step,
            total_steps=self.total_steps,
            best_reward=max(self.reward_history) if self.reward_history else 0.0,
            model_state={
                'strategy': self.strategy,
                'exploration_rate': self.exploration_rate,
                'learning_rate': self.learning_rate,
                'risk_multiplier': self.risk_multiplier
            },
            optimizer_state={},  # Would contain optimizer state in real implementation
            step_results=self.step_results
        )

    def load_from_checkpoint(self, data: CheckpointData) -> None:
        """Load state from checkpoint"""
        self.current_step = data.episode
        self.total_steps = data.total_steps
        # Note: best_reward is derived, not stored
        self.strategy = data.model_state.get('strategy', 'generalization')
        self.exploration_rate = data.model_state.get('exploration_rate', 0.3)
        self.learning_rate = data.model_state.get('learning_rate', 0.001)
        self.risk_multiplier = data.model_state.get('risk_multiplier', 1.0)
        self.step_results = data.step_results

        # Rebuild history from step results
        self.reward_history = [result.reward for result in self.step_results]
        self.pnl_history = [result.info.get('pnl', 0.0) for result in self.step_results]

    def collect_metrics(self) -> MetricsData:
        """Collect final metrics"""
        total_steps = len(self.step_results)
        if total_steps == 0:
            return {}

        # Calculate metrics
        avg_reward = sum(self.reward_history) / len(self.reward_history) if self.reward_history else 0
        total_pnl = sum(self.pnl_history) if self.pnl_history else 0
        harmful_ratio = self.harmful_count / total_steps if total_steps > 0 else 0
        pending_ratio = self.pending_count / total_steps if total_steps > 0 else 0
        verified_ratio = self.verified_count / total_steps if total_steps > 0 else 0

        return {
            'total_steps': total_steps,
            'avg_reward': avg_reward,
            'total_pnl': total_pnl,
            'harmful_ratio': harmful_ratio,
            'pending_ratio': pending_ratio,
            'verified_ratio': verified_ratio,
            'strategy': self.strategy,
            'execution_time_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            # メモリリーク監視サマリーを追加
            'memory_leak_analysis': self.resource_monitor.get_memory_leak_summary()
        }

    def _get_current_metrics(self) -> MetricsData:
        """ハートビート通知用の現在のメトリクスを取得"""
        try:
            # メモリ使用量
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)

            # オブジェクト数（GCから）
            import gc
            object_count = len(gc.get_objects())

            # 平均報酬（最近100ステップ）
            avg_reward = 0.0
            if self.reward_history:
                recent_rewards = self.reward_history[-100:] if len(self.reward_history) >= 100 else self.reward_history
                avg_reward = sum(recent_rewards) / len(recent_rewards)

            return {
                'memory_mb': round(memory_mb, 1),
                'object_count': object_count,
                'avg_reward': avg_reward,
                'current_step': self.current_step,
                'total_steps': self.total_steps
            }
        except Exception as e:
            return {'error': str(e)}

    def _run_single_step(self, step: int) -> StepResult:
        """Execute single step feature evaluation with strategy-specific logic"""
        self.logger.debug(f"Running step {step}/{self.total_steps} (strategy: {self.strategy})")

        # Record memory usage
        memory_before = self._get_memory_usage()

        # Simulate feature evaluation (instead of calling external script)
        import random
        start_time = time.time()

        # Strategy-specific simulation
        if self.strategy == 'generalization':
            # Conservative approach - more stable but slower
            execution_time = random.uniform(0.05, 0.2)  # 0.05-0.2 seconds
            success_rate = random.uniform(0.7, 0.9)   # 70-90% success rate
        elif self.strategy == 'aggressive':
            # Aggressive approach - faster but more variable
            execution_time = random.uniform(0.02, 0.15)  # 0.02-0.15 seconds
            success_rate = random.uniform(0.5, 0.95)  # 50-95% success
        else:
            execution_time = random.uniform(0.03, 0.18)
            success_rate = random.uniform(0.6, 0.9)

        # Simulate processing time
        time.sleep(execution_time)

        # Recalculate execution time after sleep
        execution_time = time.time() - start_time

        # Simulate return code based on success rate
        return_code = 0 if random.random() < success_rate else 1

        # Record memory usage
        memory_after = self._get_memory_usage()

        # Calculate reward and PnL
        reward = self._calculate_reward(return_code, execution_time)
        pnl = self._calculate_pnl(reward, self.risk_multiplier)

        self.reward_history.append(reward)
        self.pnl_history.append(pnl)

        # Simulate feature statistics
        if return_code == 0:
            self.verified_count += random.randint(1, 3)
        else:
            self.harmful_count += random.randint(0, 2)
            self.pending_count += random.randint(0, 1)

        # Create StepResult
        done = step >= self.total_steps
        return StepResult(
            step=step,
            reward=reward,
            done=done,
            info={
                'execution_time': execution_time,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'return_code': return_code,
                'stdout': f"Simulated feature evaluation: {success_rate:.2%} success rate",
                'stderr': "" if return_code == 0 else "Simulated error",
                'timestamp': datetime.now().isoformat(),
                'reward': reward,
                'pnl': pnl,
                'strategy': self.strategy
            }
        )

    def _calculate_reward(self, return_code: int, execution_time: float) -> float:
        """Calculate reward based on execution result and strategy"""
        base_reward = 1.0 if return_code == 0 else -1.0

        # Strategy-specific reward adjustments
        if self.strategy == 'generalization':
            # Penalize slow execution more
            time_penalty = max(0, execution_time - 1.0) * 0.1
            return base_reward - time_penalty
        elif self.strategy == 'aggressive':
            # Reward faster execution
            time_bonus = max(0, 2.0 - execution_time) * 0.2
            return base_reward + time_bonus

        return base_reward

    def _calculate_pnl(self, reward: float, risk_multiplier: float) -> float:
        """Calculate PnL based on reward and risk"""
        return reward * risk_multiplier * 100  # Simplified PnL calculation

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': process.memory_percent()
        }

# エラー発生時に通知を行うデコレータ（例外をキャッチして通知サービスへ送信）
@catch_and_notify
def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="10k step reinforcement learning experiment with strategy support")
    parser.add_argument("--strategy", choices=['generalization', 'aggressive'], default="generalization",
                       help="Trading strategy (default: generalization)")
    parser.add_argument("--steps", type=int, default=10000, help="Total steps (default: 10000)")
    parser.add_argument("--name", help="Experiment name override")
    parser.add_argument("--dataset", default="coingecko", help="Dataset (default: coingecko)")

    args = parser.parse_args()

    config = {
        'strategy': args.strategy,
        'total_steps': args.steps,
        'name': args.name or f"run10k_{args.strategy}",
        'dataset': args.dataset
    }

    experiment = MLReinforcement10KExperiment(config)
    result = experiment.execute()

    # 実験完了通知
    logger = experiment.logger_manager
    logger.enqueue_notification(f"10k Experiment completed. Results saved to {experiment.results_dir}")
    logger.enqueue_notification(f"Status: {result.status}")
    logger.enqueue_notification(f"Strategy: {args.strategy}")
    logger.enqueue_notification(f"Metrics: {result.metrics}")


if __name__ == "__main__":
    main()