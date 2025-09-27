#!/usr/bin/env python3
"""
100k step reinforcement learning experiment execution script
Parallel processing with joblib for 1M total steps (10 jobs × 100k steps)
ExperimentBase class-based feature evaluation experiment with strategy support
"""

import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Union, cast

import psutil
import random
import joblib  # type: ignore[import-untyped]

# Local module imports
current_dir = Path(__file__).parent.parent
project_root = current_dir.parent  # Go up one more level to project root
sys.path.insert(0, str(project_root))
from ztb.experiments.base import ScalingExperiment, ExperimentResult
from ztb.utils.error_handler import catch_and_notify  # type: ignore[import-not-found]
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


class MLReinforcement100KJob(ScalingExperiment):
    """100k step reinforcement learning experiment job class with strategy support"""

    def __init__(self, config: Dict[str, Any]):
        self.strategy = config.get('strategy', 'generalization')
        experiment_name = config.get('name', f"ml_reinforcement_100k_{self.strategy}")
        total_steps = config.get('total_steps', 100000)  # Use config value instead of hardcoded
        
        # 動的チェックポイント間隔設定
        if total_steps >= 100000:
            # 100k以上は間隔を広げる
            checkpoint_interval = 5000  # 5kステップごと
        else:
            checkpoint_interval = 1000  # 通常は1kステップごと
        
        config['checkpoint_interval'] = checkpoint_interval
        
        super().__init__(config, total_steps=total_steps)  # type: ignore
        self.dataset = config.get('dataset', 'coingecko')
        self.current_step = 0  # 現在のステップを追跡

        # 100k実験用のLoggerManager再初期化（通知粒度調整）
        self.logger_manager = LoggerManager(
            experiment_id=self.experiment_name,
            experiment_type="100k"
        )

        # AsyncNotifierにメトリクスコールバックを設定
        if self.logger_manager.async_notifier:
            self.logger_manager.async_notifier.set_metrics_callback(self._get_current_metrics)

        # CheckpointManager設定の最適化
        self.checkpoint_manager.max_queue_size = 5  # キューサイズ削減
        self.checkpoint_manager.compress = "lz4" if HAS_LZ4 else "zstd"  # 高速圧縮
        
        # ライトモード有効化（100k実験のみ）
        if total_steps >= 100000:
            self.checkpoint_light = True

        # ResourceMonitor初期化（メモリリーク監視用）
        self.resource_monitor = ResourceMonitor(log_interval_seconds=300)  # 5分間隔

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

    def _setup_strategy_params(self) -> None:
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
            self.logger.error(f"Error in step {step_num}: {e}")
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
            success_rate = random.uniform(0.7, 0.9)   # 70-90% success
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

    def _parse_evaluation_results(self, stdout: str, step: int) -> None:
        """Extract statistics from evaluation results"""
        lines = stdout.split('\n')

        for line in lines:
            if '✗' in line and 'harmful' in line.lower():
                self.harmful_count += 1
            elif 'pending' in line.lower():
                self.pending_count += 1
            elif '✓' in line and 'verified' in line.lower():
                self.verified_count += 1

    def _generate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate final metrics"""
        total_steps = len(results)
        avg_execution_time = sum(r['execution_time'] for r in results) / total_steps if total_steps > 0 else 0
        avg_memory_usage = sum(r['memory_after']['rss'] for r in results) / total_steps if total_steps > 0 else 0

        # Reward statistics
        rewards = [r['reward'] for r in results if 'reward' in r]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        reward_std = (sum((r - avg_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5 if rewards else 0

        # PnL statistics
        pnls = [r['pnl'] for r in results if 'pnl' in r]
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0
        pnl_std = (sum((p - avg_pnl) ** 2 for p in pnls) / len(pnls)) ** 0.5 if pnls else 0
        max_drawdown = min(pnls) if pnls else 0

        # Sharpe ratio (simplified)
        sharpe_ratio = avg_pnl / pnl_std if pnl_std > 0 else 0

        # Add final_portfolio_value for aggregation (example: sum of all pnls as a placeholder)
        final_portfolio_value = sum(pnls) if pnls else 0

        return {
            'total_steps': total_steps,
            'avg_execution_time_per_step': avg_execution_time,
            'avg_memory_usage_mb': avg_memory_usage,
            'harmful_features': self.harmful_count,
            'pending_features': self.pending_count,
            'verified_features': self.verified_count,
            'total_features': self.harmful_count + self.pending_count + self.verified_count,
            'strategy': self.strategy,
            'avg_reward': avg_reward,
            'reward_std': reward_std,
            'avg_pnl': avg_pnl,
            'pnl_std': pnl_std,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate,
            'risk_multiplier': self.risk_multiplier,
            'final_portfolio_value': final_portfolio_value
        }

    def _save_artifacts(self, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """Save artifacts"""
        artifacts = {}

        # Save detailed results as JSON
        results_file = self.results_dir / f"{self.experiment_name}_detailed_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        artifacts['detailed_results'] = str(results_file)

        # Save statistics summary
        summary_file = self.results_dir / f"{self.experiment_name}_summary.json"
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'metrics': self._generate_metrics(results)
        }
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        artifacts['summary'] = str(summary_file)

        return artifacts

    def _start_heartbeat(self) -> None:
        """定期的なハートビート通知を開始"""
        import threading

        def heartbeat_worker() -> None:
            step = 0
            start_time = time.time()
            while step < self.total_steps:
                time.sleep(300)  # 5分ごと
                if hasattr(self, 'current_step'):
                    step = self.current_step
                    # メモリ使用量を直接取得
                    mem_bytes = psutil.Process().memory_info().rss
                    mem_gb = mem_bytes / 1024 / 1024 / 1024
                    elapsed_time = max(1, time.time() - start_time)
                    steps_per_sec = step / elapsed_time
                    self.logger_manager.log_heartbeat(step, mem_gb, steps_per_sec)
                    self.logger_manager.enqueue_notification(f"Step {step}/{self.total_steps} completed")

        # バックグラウンドでハートビートを開始
        threading.Thread(target=heartbeat_worker, daemon=True).start()

    def _prepare_session_results(self, result: ExperimentResult) -> Dict[str, Any]:
        """セッション終了通知用の詳細な結果データを準備"""
        # 実験結果から統計データを抽出
        metrics = result.metrics

        # 取引統計の計算
        total_trades_raw = metrics.get('total_trades', 0)
        winning_trades_raw = metrics.get('winning_trades', 0)

        total_trades: int = total_trades_raw if isinstance(total_trades_raw, int) else 0
        winning_trades: int = winning_trades_raw if isinstance(winning_trades_raw, int) else 0
        win_rate_percent = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # 報酬統計
        reward_stats = {
            'mean_total_reward': metrics.get('mean_reward', 0),
            'std_total_reward': metrics.get('std_reward', 0),
            'max_reward': metrics.get('max_reward', 0),
            'min_reward': metrics.get('min_reward', 0)
        }

        # PnL統計
        pnl_stats = {
            'mean_total_pnl': metrics.get('mean_pnl', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0)
        }

        # 取引統計
        trading_stats = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate_percent': win_rate_percent,
            'profit_factor': metrics.get('profit_factor', 0),
            'mean_trades_per_episode': metrics.get('mean_trades_per_episode', 0),
            'buy_ratio': metrics.get('buy_ratio', 0.5),
            'sell_ratio': metrics.get('sell_ratio', 0.5)
        }

        return {
            'reward_stats': reward_stats,
            'pnl_stats': pnl_stats,
            'trading_stats': trading_stats,
            'execution_time_seconds': result.execution_time_seconds,
            'status': result.status,
            'total_steps': self.total_steps
        }


@catch_and_notify  # type: ignore[misc]
def main() -> None:
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="100k step reinforcement learning experiment with parallel processing")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--output-dir", default="runs/100k_experiment", help="Output directory")
    parser.add_argument("--jobs", type=int, default=10, help="Number of parallel jobs")
    parser.add_argument("--parallel", action="store_true", help="Run in parallel mode")
    parser.add_argument("--strategy", choices=['generalization', 'aggressive'], default="generalization",
                       help="Trading strategy (default: generalization)")
    parser.add_argument("--steps", type=int, default=100000, help="Total steps per job (default: 100000)")
    parser.add_argument("--name", help="Experiment name override")
    parser.add_argument("--dataset", default="coingecko", help="Dataset (default: coingecko)")

    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        try:
            from ztb.utils.config_loader import load_config
            config = load_config(args.config)
        except ImportError:
            print("Warning: Could not load config loader, using defaults")

    # Override config with command line args
    config.update({
        'strategy': args.strategy,
        'total_steps': args.steps,
        'name': args.name or f"run100k_{args.strategy}",
        'dataset': args.dataset
    })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.parallel:
        # Run parallel experiments
        print(f"Running {args.jobs} parallel jobs...")
        job_configs = []
        for job_id in range(args.jobs):
            job_config = config.copy()
            job_config['job_id'] = job_id
            job_config['output_dir'] = str(output_dir / f"job_{job_id}")
            job_configs.append(job_config)

        results = joblib.Parallel(n_jobs=args.jobs, backend='loky')(
            joblib.delayed(run_single_job)(job_config) for job_config in job_configs
        )
    else:
        # Run single experiment
        print("Running single experiment...")
        experiment = MLReinforcement100KJob(config)
        result = experiment.execute()
        results = [result]

    # Aggregate results
    aggregate_results(results, output_dir)

    print(f"Experiment completed. Results saved to {output_dir}")


    def run_single_job(config: Dict[str, Any]) -> ExperimentResult:
        """Run a single experiment job"""
        experiment = MLReinforcement100KJob(config)
        return experiment.execute()
    
def aggregate_results(results: List[ExperimentResult], output_dir: Path) -> Dict[str, Any]:
    """Aggregate results from all jobs"""
    # Filter only successful results (status == "success")
    successful = [r for r in results if hasattr(r, "status") and r.status == "success"]

    if successful:
        aggregated = {
            'total_jobs': len(results),
            'successful_jobs': len(successful),
            'total_steps': sum(cast(float, r.metrics.get('total_steps', 0)) for r in successful),
            'avg_reward': sum(cast(float, r.metrics.get('avg_reward', 0)) for r in successful) / len(successful),
            'avg_total_pnl': sum(cast(float, r.metrics.get('total_pnl', 0)) for r in successful) / len(successful),
            'best_sharpe_ratio': max(cast(float, r.metrics.get('sharpe_ratio', 0)) for r in successful),
            'best_portfolio': max(cast(float, r.metrics.get('final_portfolio_value', 0)) for r in successful)
        }
    else:
        aggregated = {'error': 'No successful experiments'}  # type: ignore[dict-item]

    # Save aggregated results
    with open(output_dir / "aggregated_results.json", 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)
    
    return aggregated


if __name__ == "__main__":
    main()