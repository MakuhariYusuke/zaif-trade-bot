#!/usr/bin/env python3
"""
Base class for ML reinforcement learning experiments
ML強化学習実験の基底クラス
"""

import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union, cast

import psutil

from dataclasses import asdict, dataclass

from ztb.utils.core.error_handler import catch_and_notify

from ztb.experiments.base import ExperimentConfig, ExperimentMetrics, ExperimentResult, ScalingExperiment
from ztb.utils import LoggerManager
from ztb.utils.checkpoint import HAS_LZ4
from ztb.utils.parallel_experiments import ResourceMonitor


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


class MLReinforcementExperiment(ScalingExperiment):
    """Base class for ML reinforcement learning experiments"""

    def __init__(self, config: Dict[str, Any], total_steps: int):
        self.strategy = config.get("strategy", "generalization")
        experiment_name = config.get("name", f"ml_reinforcement_{total_steps}_{self.strategy}")

        # 動的チェックポイント間隔設定
        if total_steps >= 100000:
            checkpoint_interval = 5000  # 5kステップごと
        else:
            checkpoint_interval = 1000  # 通常は1kステップごと

        config["checkpoint_interval"] = checkpoint_interval

        super().__init__(cast(ExperimentConfig, config), total_steps=total_steps)
        self.dataset = config.get("dataset", "coingecko")
        self.current_step = 0  # 現在のステップを追跡

        # LoggerManager初期化
        self.logger_manager = LoggerManager(
            experiment_id=self.experiment_name, experiment_type=str(total_steps)
        )

        # AsyncNotifierにメトリクスコールバックを設定
        if self.logger_manager.async_notifier:
            self.logger_manager.async_notifier.set_metrics_callback(
                self._get_current_metrics
            )

        # CheckpointManager設定の最適化
        self.checkpoint_manager.max_queue_size = 5  # キューサイズ削減
        self.checkpoint_manager.compress = "lz4" if HAS_LZ4 else "zstd"  # 高速圧縮

        # ライトモード有効化（100k実験のみ）
        if total_steps >= 100000:
            self.checkpoint_light = True

        # ResourceMonitor初期化（メモリリーク監視用）
        self.resource_monitor = ResourceMonitor()
        self.resource_monitor.log_resources(self.experiment_name)

        # 実験状態
        self.best_reward = float('-inf')
        self.strategy = config.get("strategy", "generalization")
        experiment_name = config.get("name", f"ml_reinforcement_{total_steps}_{self.strategy}")

        # 動的チェックポイント間隔設定
        if total_steps >= 100000:
            checkpoint_interval = 5000  # 5kステップごと
        else:
            checkpoint_interval = 1000  # 通常は1kステップごと

        config["checkpoint_interval"] = checkpoint_interval

        super().__init__(cast(ExperimentConfig, config), total_steps=total_steps)
        self.dataset = config.get("dataset", "coingecko")
        self.current_step = 0  # 現在のステップを追跡

        # LoggerManager初期化
        self.logger_manager = LoggerManager(
            experiment_id=self.experiment_name, experiment_type=str(total_steps)
        )

        # AsyncNotifierにメトリクスコールバックを設定
        if self.logger_manager.async_notifier:
            self.logger_manager.async_notifier.set_metrics_callback(
                self._get_current_metrics
            )

        # CheckpointManager設定の最適化
        self.checkpoint_manager.max_queue_size = 5  # キューサイズ削減
        self.checkpoint_manager.compress = "lz4" if HAS_LZ4 else "zstd"  # 高速圧縮

        # ライトモード有効化（100k実験のみ）
        if total_steps >= 100000:
            self.checkpoint_light = True

        # ResourceMonitor初期化（メモリリーク監視用）
        self.resource_monitor = ResourceMonitor()
        self.resource_monitor.log_resources(self.experiment_name)

        # 実験状態
        self.best_reward = float('-inf')
        self.step_results: List[StepResult] = []

    def run(self) -> ExperimentResult:
        """実験実行"""
        try:
            start_time = time.time()

            # チェックポイントから再開
            checkpoint_data, start_step, metadata = self.checkpoint_load()
            self.current_step = start_step

            if checkpoint_data:
                self.load_from_checkpoint(checkpoint_data)
                self.logger.info(f"Resumed from step {start_step}")

            # メイン実行ループ
            while self.current_step < self.total_steps:
                # ステップ実行
                step_result = self.step(self.current_step)

                # 定期的なチェックポイント保存
                if self.should_checkpoint(self.current_step):
                    self._save_checkpoint(self.current_step, 0)  # episodeは0で固定

                # 定期的なレポート
                if self.should_report(self.current_step):
                    self._report_metrics(self.current_step)

                # 進捗ログ
                if self.current_step % 100 == 0:
                    progress = (self.current_step / self.total_steps) * 100
                    self.logger.info(
                        f"Progress: {self.current_step}/{self.total_steps} ({progress:.1f}%)"
                    )

                self.current_step += 1

            end_time = time.time()
            total_time = end_time - start_time

            result = ExperimentResult(
                experiment_name=self.experiment_name,
                timestamp=datetime.now().isoformat(),
                status="completed",
                config=self.config,
                metrics=self._get_current_metrics(),
                artifacts={}
            )
            return result

        except Exception as e:
            self.logger_manager.log_error(f"Experiment failed: {e}")
            raise

    def step(self, step_num: int) -> StepResult:
        """ステップ実行"""
        # 学習ステップのシミュレーション
        reward = random.uniform(-1.0, 1.0)
        done = step_num >= self.total_steps - 1
        info = {"step": step_num, "reward": reward}

        step_result = StepResult(step=step_num, reward=reward, done=done, info=info)
        self.step_results.append(step_result)

        if reward > self.best_reward:
            self.best_reward = reward

        return step_result

    def should_checkpoint(self, step: int) -> bool:
        """チェックポイント保存が必要か"""
        return (step + 1) % self.checkpoint_interval == 0 or step == self.total_steps - 1

    def should_report(self, step: int) -> bool:
        """レポート出力が必要か"""
        return (step + 1) % 1000 == 0 or step == self.total_steps - 1  # 1kステップごと

    def _save_checkpoint(self, step: int, episode: int) -> None:
        """チェックポイント保存"""
        checkpoint_data = CheckpointData(
            episode=episode,
            total_steps=self.total_steps,
            best_reward=self.best_reward,
            model_state={"dummy": "model"},
            optimizer_state={"dummy": "optimizer"},
            step_results=self.step_results[-100:]  # last 100 steps
        )
        self.checkpoint_manager.save_async(
            asdict(checkpoint_data), step, {"episode": episode, "best_reward": self.best_reward}
        )

    def _report_metrics(self, step: int) -> None:
        """メトリクスレポート"""
        metrics = self._get_current_metrics()
        self.logger_manager.enqueue_notification(f"Metrics at step {step}: {json.dumps(metrics)}")

    def _get_current_metrics(self) -> MetricsData:
        """現在のメトリクス取得"""
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "best_reward": self.best_reward,
            "memory_usage": psutil.virtual_memory().percent,
            "cpu_usage": psutil.cpu_percent(interval=1),
            "experiment_name": self.experiment_name,
            "strategy": self.strategy,
            "dataset": self.dataset
        }

    def get_checkpoint_data(self) -> Any:
        """チェックポイント保存用のデータを返す"""
        return {
            "step_results": self.step_results,
            "best_reward": self.best_reward
        }

    def load_from_checkpoint(self, data: Any) -> None:
        """チェックポイントデータから状態を復元"""
        self.step_results = data.get("step_results", [])
        self.best_reward = data.get("best_reward", float('-inf'))

    def collect_metrics(self) -> ExperimentMetrics:
        """メトリクス収集"""
        return self._get_current_metrics()


def main() -> None:
    """メイン関数"""
    config: Dict[str, Any] = {
        "strategy": "generalization",
        "dataset": "coingecko",
        "total_steps": 1000  # default, subclasses override
    }
    experiment = MLReinforcementExperiment(config, total_steps=config["total_steps"])
    result = experiment.run()
    print(f"Experiment completed: {result}")


if __name__ == "__main__":
    main()