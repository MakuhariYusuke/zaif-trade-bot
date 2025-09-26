"""
Parallel Experiment Execution Utilities
複数実験の並行実行ユーティリティ

Supports running generalization and aggressive strategies in parallel
with shared resources and separate checkpoint directories.
"""

import os
import sys
import subprocess
import time
import psutil
import gc
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Type definitions for better type safety
ConfigValue = Any  # Keep flexible for experiment configs
ExperimentConfig = Dict[str, ConfigValue]
ProcessInfo = Dict[str, Any]  # Process information dictionary

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from ztb.experiments.base import ExperimentResult
from ztb.utils import LoggerManager
from ztb.utils.resource.process_priority import ProcessPriorityManager


@dataclass
class ParallelExperimentConfig:
    """Configuration for parallel experiment execution"""
    experiment_class: type
    configs: List[ExperimentConfig]
    max_workers: int = 2
    shared_data_cache: Optional[str] = None
    enable_resource_monitoring: bool = True
    priority_configs: Optional[Dict[str, str]] = None  # model_type -> priority_level
    # 効率化のための追加パラメータ
    enable_priority_scheduling: bool = True  # 優先順位スケジューリング有効化
    batch_size: int = 5  # バッチ実行サイズ
    resource_limits: Optional[Dict[str, float]] = None  # リソース制限（CPU%, Memory%）


class ParallelExperimentRunner:
    """Runner for executing multiple experiments in parallel"""

    def __init__(self, config: ParallelExperimentConfig):
        self.config = config
        self.shared_logger = LoggerManager(experiment_id="parallel_experiments")
        self.resource_monitor = ResourceMonitor() if config.enable_resource_monitoring else None

        # 効率化のための属性
        self.enable_priority_scheduling = config.enable_priority_scheduling
        self.batch_size = config.batch_size
        self.resource_limits = config.resource_limits or {'cpu_percent': 80.0, 'memory_percent': 85.0}

        # GPUリソース監視 - 遅延評価で効率化
        self._gpu_available: Optional[bool] = None  # キャッシュ用
        self._gpu_stats_cache: Optional[Dict[str, Dict[str, float]]] = None  # GPU統計キャッシュ
        self._gpu_cache_time: float = 0.0  # キャッシュ時間
        self._gpu_cache_ttl: float = 5.0  # キャッシュ有効期間（秒）

        # Shared data cache
        self.shared_data: Dict[str, Any] = {}
        if config.shared_data_cache:
            self._load_shared_data()

    @property
    def gpu_available(self) -> bool:
        """GPU利用可能性を遅延評価で取得"""
        if self._gpu_available is None:
            self._gpu_available = self._detect_gpu()
            # GPUが利用可能な場合、リソース制限にGPUメモリ制限を追加
            if self._gpu_available:
                self.resource_limits['gpu_memory_percent'] = 90.0
        return self._gpu_available

    def _load_shared_data(self):
        """Load shared data cache"""
        if self.config.shared_data_cache:
            cache_path = Path(self.config.shared_data_cache)
            if cache_path.exists():
                import pickle
                with open(cache_path, 'rb') as f:
                    self.shared_data = pickle.load(f)
                self.shared_logger.logger.info(f"Loaded shared data cache: {len(self.shared_data)} items")

    def _save_shared_data(self):
        """Save shared data cache"""
        if self.config.shared_data_cache:
            import pickle
            cache_path = Path(self.config.shared_data_cache)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.shared_data, f)
            self.shared_logger.logger.info(f"Saved shared data cache: {len(self.shared_data)} items")

    def run_parallel(self) -> List[ExperimentResult]:
        """Execute experiments in parallel with efficiency optimizations"""
        self.shared_logger.start_session("parallel_execution", "multiple_experiments")

        start_time = time.time()
        results = []

        try:
            # 優先順位に基づいて設定をソート
            if self.enable_priority_scheduling and self.config.priority_configs:
                sorted_configs = self._sort_configs_by_priority(self.config.configs)
            else:
                sorted_configs = self.config.configs

            # バッチ実行
            for batch_start in range(0, len(sorted_configs), self.batch_size):
                batch_configs = sorted_configs[batch_start:batch_start + self.batch_size]
                batch_results = self._run_batch(batch_configs, batch_start)
                results.extend(batch_results)

                # リソースチェック
                if not self._check_resource_limits():
                    self.shared_logger.logger.warning("Resource limits exceeded, pausing execution")
                    time.sleep(30)  # 30秒待機

        finally:
            total_time = time.time() - start_time
            self._save_shared_data()

            # Final summary
            success_count = sum(1 for r in results if r.status == "success")
            summary: Dict[str, Any] = {
                "total_experiments": len(results),
                "successful": success_count,
                "failed": len(results) - success_count,
                "total_time_seconds": total_time,
                "avg_time_per_experiment": total_time / len(results) if results else 0,
                "batch_size": self.batch_size,
                "priority_scheduling": self.enable_priority_scheduling
            }
            
            # メモリリーク監視サマリーを追加
            if self.resource_monitor:
                memory_summary = self.resource_monitor.get_memory_leak_summary()
                summary["memory_leak_analysis"] = memory_summary
                if memory_summary.get("potential_leak", False):
                    self.shared_logger.logger.warning(
                        f"Memory leak detected during parallel execution: "
                        f"{memory_summary['memory_increase_percent']:.1f}% increase, "
                        f"{memory_summary['objects_increase']} more objects"
                    )

            self.shared_logger.end_session(summary)

        return results

    def _sort_configs_by_priority(self, configs: List[ExperimentConfig]) -> List[ExperimentConfig]:
        """設定を優先順位に基づいてソート"""
        priority_order = {'high': 0, 'normal': 1, 'low': 2}
        
        def get_priority(config: Dict[str, Any]) -> int:
            model_type = config.get('model_type', 'generalization')
            if self.config.priority_configs:
                priority_level = self.config.priority_configs.get(model_type, 'normal')
            else:
                priority_level = 'normal'
            return priority_order.get(priority_level, 1)
        
        return sorted(configs, key=get_priority)

    def _run_batch(self, batch_configs: List[Dict[str, Any]], batch_start: int) -> List[ExperimentResult]:
        """バッチ単位で実験を実行（リソース監視付き）"""
        # リソース制限チェック
        if not self._check_resource_limits():
            self.shared_logger.logger.warning("Resource limits exceeded, skipping batch execution")
            return []
        
        results = []
        active_processes: Dict[int, Dict[str, Any]] = {}  # pid -> experiment_info
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit batch experiments
            future_to_config = {}
            for i, config in enumerate(batch_configs):
                global_index = batch_start + i
                # Set process priority if configured
                model_type = config.get('model_type', 'generalization')
                if self.config.priority_configs and model_type in self.config.priority_configs:
                    priority_level = self.config.priority_configs[model_type]
                    config['_priority_level'] = priority_level

                future = executor.submit(self._run_single_experiment, config, global_index)
                future_to_config[future] = (config, global_index)

            # Collect results as they complete with resource monitoring
            for future in as_completed(future_to_config):
                config, index = future_to_config[future]
                
                # リソースチェック - 制限を超えている場合
                if not self._check_resource_limits():
                    self.shared_logger.logger.warning(
                        f"Resource limits exceeded during batch execution, "
                        f"forcing checkpoint saves for running experiments"
                    )
                    # 実行中のプロセスにチェックポイント保存シグナルを送る
                    self._force_checkpoint_save(active_processes)
                    # リソース回復を待つ
                    time.sleep(60)
                
                try:
                    result = future.result()
                    results.append(result)
                    self.shared_logger.enqueue_notification(
                        f"Experiment {index} completed: {result.status} ({result.experiment_name})"
                    )
                except Exception as e:
                    error_result = ExperimentResult(
                        experiment_name=f"experiment_{index}",
                        timestamp=datetime.now().isoformat(),
                        status="failed",
                        config=config,  # type: ignore
                        metrics={},
                        artifacts={},
                        error_message=str(e)
                    )
                    results.append(error_result)
                    self.shared_logger.enqueue_notification(
                        f"Experiment {index} failed: {str(e)}"
                    )
        
        return results

    def _force_checkpoint_save(self, active_processes: Dict[int, Dict[str, Any]]):
        """実行中の実験にチェックポイント保存を強制"""
        try:
            for pid, experiment_info in active_processes.items():
                try:
                    process = psutil.Process(pid)
                    if process.is_running():
                        # チェックポイント保存シグナルを送る（クロスプラットフォーム対応）
                        try:
                            # WindowsではSIGTERMを使用
                            signal_to_send = getattr(signal, 'SIGUSR1', None) or signal.SIGTERM
                            process.send_signal(signal_to_send)
                            self.shared_logger.logger.info(
                                f"Sent checkpoint signal to process {pid} ({experiment_info.get('name', 'unknown')})"
                            )
                        except (ProcessLookupError, PermissionError):
                            pass  # プロセスが既に終了している場合
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass  # プロセスが既に終了している場合
        except Exception as e:
            self.shared_logger.logger.error(f"Failed to force checkpoint saves: {e}")

    def _check_resource_limits(self) -> bool:
        """リソース制限をチェック（CPU、メモリ、GPU）- 効率化"""
        try:
            # CPUとメモリのチェック（効率化）
            cpu_percent = psutil.cpu_percent(interval=0.1)  # 短いインターバルで効率化
            memory = psutil.virtual_memory()

            cpu_ok = cpu_percent < self.resource_limits['cpu_percent']
            memory_ok = memory.percent < self.resource_limits['memory_percent']

            # GPUチェック（GPUが利用可能な場合のみ）
            gpu_ok = True
            if self.gpu_available and 'gpu_memory_percent' in self.resource_limits:
                gpu_stats = self._get_gpu_usage()
                if gpu_stats:
                    # 最も使用率の高いGPUをチェック（効率化）
                    max_gpu_usage = max(
                        stats.get('memory_percent', 0)
                        for stats in gpu_stats.values()
                    )
                    gpu_ok = max_gpu_usage < self.resource_limits['gpu_memory_percent']

                    if not gpu_ok:
                        self.shared_logger.logger.warning(
                            f"GPU memory usage too high: {max_gpu_usage:.1f}% "
                            f"(limit: {self.resource_limits['gpu_memory_percent']}%)"
                        )
                else:
                    # GPU統計が取得できない場合はスキップ
                    gpu_ok = True

            # ログ出力の最適化（制限を超えた場合のみ）
            if not cpu_ok:
                self.shared_logger.logger.warning(
                    f"CPU usage too high: {cpu_percent:.1f}% "
                    f"(limit: {self.resource_limits['cpu_percent']}%)"
                )
            if not memory_ok:
                self.shared_logger.logger.warning(
                    f"Memory usage too high: {memory.percent:.1f}% "
                    f"(limit: {self.resource_limits['memory_percent']}%)"
                )

            return cpu_ok and memory_ok and gpu_ok
        except Exception as e:
            self.shared_logger.logger.error(f"Resource check failed: {e}")
            return True  # エラーの場合は続行

    def _detect_gpu(self) -> bool:
        """GPU利用可能性を検出"""
        try:
            # NVIDIA GPU検出
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0 and len(result.stdout.strip()) > 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        try:
            # AMD GPU検出（ROCm）
            result = subprocess.run(['rocm-smi', '--showid'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
            
        return False

    def _get_gpu_usage(self) -> Dict[str, Dict[str, float]]:
        """GPU使用状況を取得（キャッシュ対応で効率化）"""
        current_time = time.time()

        # キャッシュが有効な場合は再利用
        if (self._gpu_stats_cache is not None and
            current_time - self._gpu_cache_time < self._gpu_cache_ttl):
            return self._gpu_stats_cache

        gpu_stats = {}

        try:
            # NVIDIA GPU
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=2)  # タイムアウト短縮

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    parts = [x.strip() for x in line.split(',')]
                    if len(parts) >= 4:
                        try:
                            used_mb = float(parts[0])
                            total_mb = float(parts[1])
                            temp = float(parts[2])
                            util = float(parts[3])

                            gpu_stats[f'gpu_{i}'] = {
                                'memory_used_mb': used_mb,
                                'memory_total_mb': total_mb,
                                'memory_percent': (used_mb / total_mb) * 100.0 if total_mb > 0 else 0.0,
                                'temperature_c': temp,
                                'utilization_percent': util
                            }
                        except (ValueError, ZeroDivisionError):
                            continue  # 解析エラーの場合はスキップ
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass

        # キャッシュ更新
        self._gpu_stats_cache = gpu_stats
        self._gpu_cache_time = current_time

        return gpu_stats

    def _run_single_experiment(self, config: Dict[str, Any], index: int) -> ExperimentResult:
        """Run single experiment in subprocess（効率化）"""
        try:
            # Set process priority（効率化: 必要な場合のみ）
            priority_level = config.pop('_priority_level', 'normal')
            if priority_level and priority_level != 'normal':
                pm = ProcessPriorityManager()
                pm.set_process_priority(config.get('model_type', 'generalization'))

            # Create experiment instance
            experiment = self.config.experiment_class(config)
            experiment.experiment_name = f"{experiment.experiment_name}_{index}"

            # Run experiment
            result = experiment.execute()

            return result

        except Exception as e:
            # Return error result（効率化: エラーハンドリング改善）
            error_msg = f"Experiment failed: {str(e)}"
            self.shared_logger.logger.error(error_msg)

            return ExperimentResult(
                experiment_name=f"experiment_{index}",
                timestamp=datetime.now().isoformat(),
                status="failed",
                config=config,  # type: ignore
                metrics={},
                artifacts={},
                error_message=error_msg
            )


class ResourceMonitor:
    """Resource monitoring for parallel experiments"""

    def __init__(self, log_interval_seconds: int = 60):
        # 親クラスの初期化（明示的に呼び出し）
        super().__init__()

        # 基本設定
        self.log_interval = log_interval_seconds
        self.logger = LoggerManager(experiment_id="resource_monitor")

        # 時間追跡
        self.start_time = time.time()
        self.last_log_time = 0

        # リソース統計 - 効率的な初期化
        self.peak_cpu_percent = 0.0
        self.peak_memory_mb = 0.0
        self.avg_cpu_percent = 0.0
        self.measurements: List[float] = []

        # メモリリーク監視 - 効率的なデータ構造
        self.memory_history: List[float] = []
        self.gc_stats: List[Tuple[int, int, int]] = []
        self.object_counts: List[int] = []

        # メモリリーク検知の閾値 - 設定可能に
        self.memory_leak_threshold_percent = 50.0
        self.object_leak_threshold = 10000

    def log_resources(self, experiment_name: str = ""):
        """Log current resource usage"""
        current_time = time.time()
        if current_time - self.last_log_time < self.log_interval:
            return

        try:
            # System-wide resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Process-specific (if available)
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            process_cpu = process.cpu_percent()

            # GC統計とメモリリーク検知
            gc_counts = gc.get_count()
            gc_stats = gc.get_stats()
            current_objects = len(gc.get_objects())
            
            # メモリリーク検知: メモリ使用量の履歴を追跡（効率化）
            self.memory_history.append(process_memory)
            self.gc_stats.append(gc_counts)
            self.object_counts.append(current_objects)

            # メモリリーク警告: 設定された閾値を超えた場合（計算効率化）
            if len(self.memory_history) >= 10:
                recent_memory = self.memory_history[-10:]
                memory_ratio = recent_memory[-1] / recent_memory[0] if recent_memory[0] > 0 else 1.0
                if memory_ratio > (1.0 + self.memory_leak_threshold_percent / 100.0):
                    self.logger.logger.warning(
                        f"Potential memory leak detected: memory increased from "
                        f"{recent_memory[0]:.1f}MB to {recent_memory[-1]:.1f}MB "
                        f"(+{(memory_ratio - 1.0) * 100:.1f}%)"
                    )

                # オブジェクト数の増加もチェック
                recent_objects = self.object_counts[-10:]
                object_increase = recent_objects[-1] - recent_objects[0]
                if object_increase > self.object_leak_threshold:
                    self.logger.logger.warning(
                        f"Potential object leak detected: objects increased by {object_increase} "
                        f"(from {recent_objects[0]} to {recent_objects[-1]})"
                    )

            resources = {
                "timestamp": datetime.now().isoformat(),
                "experiment": experiment_name,
                "system_cpu_percent": cpu_percent,
                "system_memory_percent": memory.percent,
                "system_memory_used_gb": memory.used / (1024**3),
                "system_disk_percent": disk.percent,
                "process_memory_mb": process_memory,
                "process_cpu_percent": process_cpu,
                "uptime_seconds": current_time - self.start_time,
                # GC統計情報
                "gc_counts": gc_counts,
                "gc_stats": gc_stats,
                "object_count": current_objects,
                "memory_trend_mb": process_memory - (self.memory_history[0] if self.memory_history else 0)
            }

            # Log to JSONL file directly
            import json
            jsonl_path = Path("logs") / "resource_monitor.jsonl"
            jsonl_path.parent.mkdir(exist_ok=True)
            with open(jsonl_path, 'a') as f:
                f.write(json.dumps(resources) + '\n')

            # Update peak and average values
            self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)
            self.peak_memory_mb = max(self.peak_memory_mb, memory.used / (1024 * 1024))
            self.measurements.append(cpu_percent)
            if self.measurements:
                self.avg_cpu_percent = sum(self.measurements) / len(self.measurements)

        except Exception as e:
            self.logger.logger.error(f"Resource monitoring error: {e}")

    def get_memory_leak_summary(self) -> Dict[str, Any]:
        """Get summary of memory usage and potential leaks（効率化）"""
        if not self.memory_history:
            return {"status": "no_data"}

        # 効率的な計算
        initial_memory = self.memory_history[0]
        final_memory = self.memory_history[-1]
        memory_increase_percent = ((final_memory / initial_memory) - 1.0) * 100.0 if initial_memory > 0 else 0.0

        # GC統計の変化（効率化）
        initial_gc = self.gc_stats[0] if self.gc_stats else (0, 0, 0)
        final_gc = self.gc_stats[-1] if self.gc_stats else (0, 0, 0)
        gc_increase = tuple(final_gc[i] - initial_gc[i] for i in range(3))

        # オブジェクト数の変化（効率化）
        initial_objects = self.object_counts[0] if self.object_counts else 0
        final_objects = self.object_counts[-1] if self.object_counts else 0
        object_increase = final_objects - initial_objects

        # リーク判定（設定された閾値を使用）
        potential_leak = (
            memory_increase_percent > self.memory_leak_threshold_percent or
            object_increase > self.object_leak_threshold
        )

        return {
            "memory_mb_start": initial_memory,
            "memory_mb_end": final_memory,
            "memory_increase_percent": memory_increase_percent,
            "gc_collections_start": initial_gc,
            "gc_collections_end": final_gc,
            "gc_increase": gc_increase,
            "objects_start": initial_objects,
            "objects_end": final_objects,
            "objects_increase": object_increase,
            "potential_leak": potential_leak,
            "measurements_count": len(self.memory_history),
            "leak_threshold_percent": self.memory_leak_threshold_percent,
            "object_leak_threshold": self.object_leak_threshold
        }


def run_parallel_experiments(
    experiment_class: type,
    configs: List[Dict[str, Any]],
    max_workers: int = 2,
    shared_data_cache: Optional[str] = None,
    enable_monitoring: bool = True
) -> List[ExperimentResult]:
    """
    Convenience function to run experiments in parallel

    Args:
        experiment_class: Experiment class to instantiate
        configs: List of configuration dictionaries
        max_workers: Maximum number of parallel workers
        shared_data_cache: Path to shared data cache file
        enable_monitoring: Enable resource monitoring

    Returns:
        List of experiment results
    """
    config = ParallelExperimentConfig(
        experiment_class=experiment_class,
        configs=configs,
        max_workers=max_workers,
        shared_data_cache=shared_data_cache,
        enable_resource_monitoring=enable_monitoring,
        priority_configs={
            'generalization': 'normal',
            'aggressive': 'high'
        }
    )

    runner = ParallelExperimentRunner(config)
    return runner.run_parallel()