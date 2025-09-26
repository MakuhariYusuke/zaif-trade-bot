"""
Base classes for experiments in the trading RL system.
実験基底クラス - 取引RLシステムの実験実行を統一
"""

import json
import logging
import glob
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict

import numpy as np

from ztb.utils import LoggerManager
from ztb.utils.checkpoint import CheckpointManager

# Type definitions for better type safety
from typing import TypedDict, Protocol
from dataclasses import dataclass

class ExperimentConfig(TypedDict, total=False):
    """実験設定の型定義"""
    steps: int
    strategy: str
    dataset: str
    report_interval: int
    learning_rate: float
    batch_size: int
    max_episodes: int
    test_split: float
    validation_split: float
    random_seed: int

@dataclass
class AggregationResult:
    """集約結果のデータ構造"""
    aggregation_timestamp: str
    experiment_pattern: str
    total_experiments: int
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]

MetricsValue = Union[int, float, str, List[float], List[int], Dict[str, Any]]
ExperimentMetrics = Dict[str, MetricsValue]
class Notifiable(Protocol):
    """通知可能インターフェース"""
    def notify(self, message: str, level: str = "info") -> None: ...

class Checkpointable(Protocol):
    """チェックポイント可能インターフェース"""
    def save_checkpoint(self, path: str) -> None: ...
    def load_checkpoint(self, path: str) -> None: ...


@dataclass
class ExperimentResult:
    """実験結果の標準データ構造"""
    experiment_name: str
    timestamp: str
    status: str  # "success", "failed", "partial"
    config: ExperimentConfig
    metrics: ExperimentMetrics
    artifacts: Dict[str, str]  # ファイルパス
    error_message: Optional[str] = None
    execution_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentBase(ABC):
    """
    実験実行の基底クラス
    すべての実験スクリプトはこのクラスを継承する
    """

    def __init__(self, config: ExperimentConfig, experiment_name: Optional[str] = None):
        self.config = config
        # 統一された命名規則: サブクラスは experiment_name を明示的に渡すか、クラス名から 'Experiment' を除去して小文字化
        self.experiment_name = experiment_name or self._default_experiment_name()
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger_manager = LoggerManager(experiment_id=self.experiment_name)

        # 結果保存ディレクトリ
        self.results_dir = Path("results") / "experiments" / self.experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # チェックポイントディレクトリ
        self.checkpoint_dir = Path("checkpoints") / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # チェックポイントマネージャー
        self.checkpoint_manager = CheckpointManager(
            save_dir=str(self.checkpoint_dir),
            keep_last=5,
            compress="zstd"
        )

    @classmethod
    def _default_experiment_name(cls) -> str:
        """サブクラス名から一貫した experiment_name を生成"""
        name = cls.__name__
        if name.endswith('Experiment'):
            name = name[:-len('Experiment')]
        return name.lower()

    @abstractmethod
    def run(self) -> ExperimentResult:
        """
        実験のメイン実行ロジック
        サブクラスで実装必須
        """
        pass

    def execute(self) -> ExperimentResult:
        """
        実験実行の標準フロー
        エラーハンドリングと結果保存を含む
        """
        self.start_time = datetime.now()
        self.logger_manager.start_session("experiment", self.experiment_name)
        self.logger_manager.log_experiment_start(self.experiment_name, dict(self.config))
        self.logger.info(f"Starting experiment: {self.experiment_name}")

        result = None  # result を初期化してバインドされていない問題を回避
        try:
            result = self.run()
            if not isinstance(result, ExperimentResult):
                raise TypeError(f"self.run() must return an ExperimentResult, got {type(result).__name__}")
            result.status = "success"
            self.logger.info(f"Experiment completed successfully: {self.experiment_name}")
            self.logger_manager.log_experiment_end(result.to_dict())
        except Exception as e:
            self.logger.error(f"Experiment failed: {self.experiment_name}, error: {str(e)}")
            self.logger_manager.log_error(str(e), f"Experiment: {self.experiment_name}")

            # 致命的なエラーの場合は強制終了
            if isinstance(e, (KeyboardInterrupt, SystemExit, MemoryError, OSError)):
                self.logger.critical(f"Fatal error occurred: {type(e).__name__}. Terminating process.")
                import sys
                sys.exit(1)

            result = ExperimentResult(
                experiment_name=self.experiment_name,
                timestamp=datetime.now().isoformat(),
                status="failed",
                config=self.config,
                metrics={},
                artifacts={},
                error_message=str(e)
            )
        finally:
            self.end_time = datetime.now()
            if result is not None:  # チェックを簡略化
                result.execution_time_seconds = (self.end_time - self.start_time).total_seconds()
                self.logger_manager.end_session(result.to_dict())

        # resultがNoneでないことを確認
        if result is None:
            raise RuntimeError("Experiment execution failed to produce a result")

        # 結果保存
        self.save_results(result)

        # 通知送信
        self.notify_results(result)

        return result

    @classmethod
    def aggregate_results(cls, experiment_pattern: str = "*", output_file: Optional[str] = None) -> AggregationResult:
        """
        複数の実験結果を集約してレポート生成
        
        Args:
            experiment_pattern: 集約対象の実験パターン（例: "ml_reinforcement*"）
            output_file: 出力ファイルパス（指定しない場合は自動生成）
            
        Returns:
            集約された結果データ
        """
        from pathlib import Path
        
        results_dir = Path("results") / "experiments"
        if not results_dir.exists():
            return AggregationResult(
                aggregation_timestamp=datetime.now().isoformat(),
                experiment_pattern=experiment_pattern,
                total_experiments=0,
                results=[],
                summary={"error": "No results directory found"}
            )
        
        # パターンにマッチする結果ファイルを検索
        pattern = str(results_dir / f"{experiment_pattern}_*.json")
        result_files = glob.glob(pattern)
        
        if not result_files:
            return AggregationResult(
                aggregation_timestamp=datetime.now().isoformat(),
                experiment_pattern=experiment_pattern,
                total_experiments=0,
                results=[],
                summary={"error": f"No result files found matching pattern: {experiment_pattern}"}
            )
        
        # 結果を集約
        results = []
        summary: Dict[str, Any] = {
            "success_count": 0,
            "failed_count": 0,
            "total_execution_time": 0.0,
            "avg_execution_time": 0.0,
            "best_reward": float('-inf'),
            "worst_reward": float('inf'),
            "total_pnl": 0.0
        }
        
        for file_path in result_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                results.append(data)
                
                # サマリー統計更新
                if data.get("status") == "success":
                    summary["success_count"] += 1
                    
                    metrics = data.get("metrics", {})
                    execution_time = data.get("execution_time_seconds", 0)
                    reward = metrics.get("avg_reward", 0)
                    pnl = metrics.get("total_pnl", 0)
                    
                    summary["total_execution_time"] += execution_time
                    summary["best_reward"] = max(summary["best_reward"], reward)
                    summary["worst_reward"] = min(summary["worst_reward"], reward)
                    summary["total_pnl"] += pnl
                    
                else:
                    summary["failed_count"] += 1
                    
            except Exception as e:
                results.append({"error": f"Failed to load {file_path}: {str(e)}"})
        
        # 統計計算の拡張
        success_count = summary["success_count"]
        if success_count > 0:
            summary["avg_execution_time"] = summary["total_execution_time"] / success_count
            
            # 詳細な統計情報
            all_rewards = []
            all_execution_times = []
            
            for result in results:
                if isinstance(result, dict) and result.get("status") == "success":
                    metrics = result.get("metrics", {})
                    if "avg_reward" in metrics:
                        all_rewards.append(metrics["avg_reward"])
                    execution_time = result.get("execution_time_seconds", 0)
                    if execution_time > 0:
                        all_execution_times.append(execution_time)
            
            if all_rewards:
                import numpy as np
                summary["reward_stats"] = {
                    "mean": float(np.mean(all_rewards)),
                    "median": float(np.median(all_rewards)),
                    "std": float(np.std(all_rewards)),
                    "min": float(np.min(all_rewards)),
                    "max": float(np.max(all_rewards)),
                    "percentile_25": float(np.percentile(all_rewards, 25)),
                    "percentile_75": float(np.percentile(all_rewards, 75)),
                    "percentile_95": float(np.percentile(all_rewards, 95))
                }
            
            if all_execution_times:
                summary["execution_time_stats"] = {
                    "mean": float(np.mean(all_execution_times)),
                    "median": float(np.median(all_execution_times)),
                    "std": float(np.std(all_execution_times)),
                    "min": float(np.min(all_execution_times)),
                    "max": float(np.max(all_execution_times))
                }
        
        # 出力ファイル保存
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"reports/aggregated_results_{experiment_pattern}_{timestamp}.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        aggregated = AggregationResult(
            aggregation_timestamp=datetime.now().isoformat(),
            experiment_pattern=experiment_pattern,
            total_experiments=len(result_files),
            results=results,
            summary=summary
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated.__dict__, f, indent=2, ensure_ascii=False)
        
        print(f"Aggregated results saved to: {output_path}")
        
        # レポートローテーション（古いファイルを削除）
        cls._rotate_old_reports(results_dir, experiment_pattern, max_reports=10)
        
        return aggregated

    @classmethod
    def _rotate_old_reports(cls, results_dir: Path, pattern: str, max_reports: int = 10):
        """古いレポートファイルをローテーション"""
        try:
            # パターンにマッチするファイルを検索
            report_files = list(results_dir.glob(f"{pattern}_*.json"))
            
            if len(report_files) <= max_reports:
                return
            
            # 作成日時でソート（古い順）
            report_files.sort(key=lambda x: x.stat().st_mtime)
            
            # 古いファイルを削除
            files_to_delete = report_files[:-max_reports]  # 最新max_reports個を残す
            for old_file in files_to_delete:
                try:
                    old_file.unlink()
                    print(f"Rotated old report: {old_file.name}")
                except Exception as e:
                    print(f"Failed to delete {old_file.name}: {e}")
                    
        except Exception as e:
            print(f"Report rotation failed: {e}")

    def save_results(self, result: ExperimentResult) -> str:
        """実験結果をJSONファイルに保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.experiment_name}_{timestamp}.json"
        filepath = self.results_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results saved to: {filepath}")
        return str(filepath)

    def notify_results(self, result: ExperimentResult) -> None:
        """実験結果をDiscordに通知"""
        try:
            self.logger_manager.log_experiment_end(result.to_dict())
        except Exception as e:
            self.logger.warning(f"Failed to send notification: {e}")


class ScalingExperiment(ExperimentBase):
    """
    大規模実験用の基底クラス
    ステップベースの実行と定期的なチェックポイント保存をサポート
    """

    def __init__(self, config: ExperimentConfig, total_steps: int, checkpoint_interval: int = 1000):
        super().__init__(config)
        self.total_steps = total_steps
        self.checkpoint_interval = checkpoint_interval
        self.current_step = 0

    def run(self) -> ExperimentResult:
        """
        ステップベースの実験実行
        サブクラスで step() メソッドを実装する
        """
        self.logger.info(f"Starting scaling experiment with {self.total_steps} steps")

        try:
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
                if self.current_step % self.checkpoint_interval == 0:
                    self.save_checkpoint()

                # 進捗ログ
                if self.current_step % 100 == 0:
                    progress = (self.current_step / self.total_steps) * 100
                    self.logger.info(f"Progress: {self.current_step}/{self.total_steps} ({progress:.1f}%)")

                self.current_step += 1

                # 致命的なエラーチェック
                if isinstance(step_result, Exception):
                    raise step_result

            # 最終チェックポイント保存
            self.save_checkpoint()

            # メトリクス収集
            metrics = self.collect_metrics()

            return ExperimentResult(
                experiment_name=self.experiment_name,
                timestamp=datetime.now().isoformat(),
                status="success",
                config=self.config,
                metrics=metrics,
                artifacts=self.collect_artifacts()
            )

        except Exception as e:
            self.logger.error(f"Scaling experiment failed at step {self.current_step}: {e}")
            raise

    @abstractmethod
    def step(self, step_num: int) -> Any:
        """
        単一ステップの実行
        サブクラスで実装必須
        """
        pass

    def save_checkpoint(self) -> None:
        """現在の状態をチェックポイント保存"""
        checkpoint_data = self.get_checkpoint_data()
        metadata = {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'timestamp': datetime.now().isoformat()
        }
        self.checkpoint_save(checkpoint_data, self.current_step, metadata)

    @abstractmethod
    def get_checkpoint_data(self) -> Any:
        """チェックポイント保存用のデータを返す"""
        pass

    @abstractmethod
    def load_from_checkpoint(self, data: Any) -> None:
        """チェックポイントデータから状態を復元"""
        pass

    @abstractmethod
    def collect_metrics(self) -> ExperimentMetrics:
        """実験メトリクスを収集"""
        pass

    def collect_artifacts(self) -> Dict[str, str]:
        """実験アーティファクトを収集"""
        artifacts = {}
        # チェックポイントディレクトリ
        if self.checkpoint_dir.exists():
            artifacts['checkpoints'] = str(self.checkpoint_dir)
        # 結果ディレクトリ
        if self.results_dir.exists():
            artifacts['results'] = str(self.results_dir)
        return artifacts

    def checkpoint_load(self) -> tuple:
        """最新のチェックポイントを読み込み"""
        try:
            return self.checkpoint_manager.load_latest()
        except FileNotFoundError:
            self.logger.info("No checkpoint found, starting fresh")
            return None, 0, {}

    def checkpoint_save(self, obj: Any, step: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        """チェックポイントを非同期保存"""
        self.checkpoint_manager.save_async(obj, step, metadata)