#!/usr/bin/env python3
"""
250k step reinforcement learning experiment execution script
ExperimentBase class-based feature evaluation experiment with strategy support
"""

import sys
import time
import json
import psutil
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Union, cast

# Local module imports
current_dir = Path(__file__).parent.parent
project_root = current_dir.parent  # Go up one more level to project root
sys.path.insert(0, str(project_root))
from ztb.experiments.base import ExperimentBase, ExperimentResult, ExperimentConfig, ExperimentMetrics
from ztb.utils.error_handler import catch_and_notify
from ztb.utils.checkpoint import HAS_LZ4
from ztb.utils.parallel_experiments import ResourceMonitor


# Type definitions for better type safety
from dataclasses import dataclass
from typing import Dict, Any, List

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


class MLReinforcement250k(ExperimentBase):
    """
    25万ステップ強化学習実験クラス

    250kステップの学習を実行し、定期的なチェックポイント保存とメトリクス収集を行う。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(cast(ExperimentConfig, config), experiment_name=config.get('experiment_id', 'ml_rl_250k'))

        # 250kステップ設定
        self.total_steps = 250000
        self.checkpoint_freq = {
            'light': [1000, 5000],  # 1k, 5k steps
            'full': 10000,          # 10k steps
            'archive': 50000        # 50k steps
        }
        self.report_freq = {
            'progress': 2500,       # 2.5k steps
            'metrics': 10000        # 10k steps
        }

        # ログ設定
        self.logger = self.logger
        self.logger.info(f"Initialized 250k RL experiment: {self.experiment_name}")

    def should_checkpoint(self, step: int, checkpoint_type: str) -> bool:
        """指定ステップでチェックポイント保存が必要か判定"""
        if checkpoint_type == 'light':
            freqs = self.checkpoint_freq['light']
            if step in freqs:
                return True
            if step > max(freqs):
                return (step - max(freqs)) % freqs[1] == 0
        elif checkpoint_type == 'full':
            return step % self.checkpoint_freq['full'] == 0
        elif checkpoint_type == 'archive':
            return step % self.checkpoint_freq['archive'] == 0
        return False

    def should_report(self, step: int, report_type: str) -> bool:
        """指定ステップでレポート出力が必要か判定"""
        if report_type == 'progress':
            return step % self.report_freq['progress'] == 0
        elif report_type == 'metrics':
            return step % self.report_freq['metrics'] == 0
        return False

    @catch_and_notify
    def run(self) -> ExperimentResult:
        """
        250kステップ強化学習実験実行

        Returns:
            ExperimentResult: 実験結果
        """
        start_time = time.time()
        self.logger.info(f"Starting 250k RL experiment: {self.experiment_name}")

        try:
            # 実験初期化
            self._initialize_experiment()

            # 学習ループ
            step_results = []
            total_reward = 0.0
            best_reward = float('-inf')

            for step in range(self.total_steps):
                # ステップ実行（モック）
                reward = random.uniform(-1.0, 1.0)
                done = (step + 1) % 1000 == 0  # 1000ステップごとにエピソード終了
                info = {
                    'step': step,
                    'epsilon': max(0.01, 1.0 - step / self.total_steps),
                    'loss': random.uniform(0.1, 2.0)
                }

                step_result = StepResult(
                    step=step,
                    reward=reward,
                    done=done,
                    info=info
                )
                step_results.append(step_result)
                total_reward += reward

                # ベストリワード更新
                if total_reward > best_reward:
                    best_reward = total_reward

                # チェックポイント保存
                for cp_type in ['light', 'full', 'archive']:
                    if self.should_checkpoint(step + 1, cp_type):
                        self._save_checkpoint(step + 1, cp_type, {
                            'total_reward': total_reward,
                            'best_reward': best_reward,
                            'step_results': [r.__dict__ for r in step_results[-100:]]  # 直近100ステップ
                        })

                # 進捗レポート
                if self.should_report(step + 1, 'progress'):
                    progress_pct = (step + 1) / self.total_steps * 100
                    self.logger.info(".1f"
                                   f"Best: {best_reward:.2f}")

                # メトリクスレポート
                if self.should_report(step + 1, 'metrics'):
                    self._report_metrics(step + 1, {
                        'total_reward': total_reward,
                        'best_reward': best_reward,
                        'avg_reward_per_step': total_reward / (step + 1),
                        'checkpoint_count': self._count_checkpoints(),
                        'memory_usage': self._get_memory_usage()
                    })

            # 最終結果
            execution_time = time.time() - start_time
            final_metrics = {
                'total_steps': self.total_steps,
                'total_reward': total_reward,
                'best_reward': best_reward,
                'avg_reward_per_step': total_reward / self.total_steps,
                'execution_time': execution_time,
                'steps_per_second': self.total_steps / execution_time,
                'checkpoint_summary': self._get_checkpoint_summary(),
                'log_file_size': self._get_log_file_size(),
                'memory_peak': self._get_memory_usage()
            }

            self.logger.info(f"Completed 250k RL experiment: {self.experiment_name}")
            self.logger.info(f"Final metrics: {json.dumps(final_metrics, indent=2)}")

            return ExperimentResult(
                experiment_name=self.experiment_name,
                timestamp=datetime.now().isoformat(),
                status="completed",
                config=self.config,
                metrics=cast(ExperimentMetrics, final_metrics),
                artifacts={},
                execution_time_seconds=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Experiment failed: {e}")
            return ExperimentResult(
                experiment_name=self.experiment_name,
                timestamp=datetime.now().isoformat(),
                status="failed",
                config=self.config,
                metrics=cast(ExperimentMetrics, {}),
                artifacts={},
                execution_time_seconds=execution_time,
                error_message=str(e)
            )

    def _initialize_experiment(self):
        """実験初期化"""
        # シード設定
        from ztb.features.registry import FeatureRegistry
        FeatureRegistry.initialize(seed=42)

        # リソース監視開始
        self.resource_monitor = ResourceMonitor()
        self.resource_monitor.log_resources(self.experiment_name)

    def _save_checkpoint(self, step: int, cp_type: str, data: Dict[str, Any]):
        """チェックポイント保存"""
        from ztb.utils.checkpoint import HierarchicalCheckpointManager

        manager = HierarchicalCheckpointManager()
        manager.save_checkpoint(
            step=step,
            model_state={'mock_model': 'data'},
            optimizer_state={'mock_optimizer': 'data'},
            metrics=data,
            checkpoint_type=cp_type
        )

        self.logger.info(f"Saved {cp_type} checkpoint at step {step}")

    def _report_metrics(self, step: int, metrics: Dict[str, Any]):
        """メトリクスレポート"""
        from ztb.monitoring import get_exporter

        exporter = get_exporter()
        exporter.record_job_completion("success", 0.1)  # Mock completion

        self.logger.info(f"Metrics at step {step}: {json.dumps(metrics, indent=2)}")

    def _count_checkpoints(self) -> Dict[str, int]:
        """チェックポイント数カウント"""
        from pathlib import Path

        cp_dir = Path("models/checkpoints")
        if not cp_dir.exists():
            return {'light': 0, 'full': 0, 'archive': 0}

        return {
            'light': len(list(cp_dir.glob("checkpoint_light_*.pkl*"))),
            'full': len(list(cp_dir.glob("checkpoint_full_*.pkl*"))),
            'archive': len(list(cp_dir.glob("checkpoint_archive_*.pkl*")))
        }

    def _get_checkpoint_summary(self) -> Dict[str, Any]:
        """チェックポイント要約"""
        from ztb.utils.checkpoint import HierarchicalCheckpointManager

        manager = HierarchicalCheckpointManager()
        stats = manager.get_stats()

        total_size = sum(
            cp_path.stat().st_size
            for pattern in ["checkpoint_light_*.pkl*", "checkpoint_full_*.pkl*", "checkpoint_archive_*.pkl*"]
            for cp_path in Path("models/checkpoints").glob(pattern)
        ) / (1024 * 1024)  # MB

        return {
            'counts': {
                'light': stats.get('light_count', 0),
                'full': stats.get('full_count', 0),
                'archive': stats.get('archive_count', 0)
            },
            'total_size_mb': total_size
        }

    def _get_log_file_size(self) -> float:
        """ログファイルサイズ取得 (MB)"""
        log_dir = Path("logs")
        if not log_dir.exists():
            return 0.0

        total_size = sum(
            f.stat().st_size
            for f in log_dir.glob("*.log")
            if f.is_file()
        )
        return total_size / (1024 * 1024)  # MB

    def _get_memory_usage(self) -> Dict[str, float]:
        """メモリ使用量取得"""
        process = psutil.Process()
        mem_info = process.memory_info()

        return {
            'rss_mb': mem_info.rss / (1024 * 1024),
            'vms_mb': mem_info.vms / (1024 * 1024),
            'percent': process.memory_percent()
        }


def main():
    """メイン実行関数"""
    # 設定
    config = {
        'experiment_id': f"rl_250k_{int(time.time())}",
        'description': '250k step reinforcement learning dry run',
        'tags': ['reinforcement-learning', '250k-steps', 'dry-run']
    }

    # 実験実行
    experiment = MLReinforcement250k(config)
    result = experiment.run()

    # 結果出力
    print(f"Experiment completed: {result.status}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Metrics: {json.dumps(result.metrics, indent=2)}")

    # チェックポイント統計出力
    if result.status == "completed":
        cp_summary = result.metrics.get('checkpoint_summary', {})
        print("\nCheckpoint Summary:")
        print(f"  Light: {cp_summary.get('counts', {}).get('light', 0)} files")
        print(f"  Full: {cp_summary.get('counts', {}).get('full', 0)} files")
        print(f"  Archive: {cp_summary.get('counts', {}).get('archive', 0)} files")
        print(f"  Total size: {cp_summary.get('total_size_mb', 0):.1f} MB")
        print(f"Log file size: {result.metrics.get('log_file_size', 0):.1f} MB")
        print(f"Memory peak: {result.metrics.get('memory_peak', {}).get('rss_mb', 0):.1f} MB")


if __name__ == "__main__":
    main()