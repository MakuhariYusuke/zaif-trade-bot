"""
Online Learning Pipeline Implementation
インクリメンタル学習とストリーミングデータ処理
"""

import gc
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.optim import SGD, Adagrad, Adam

from .config import OnlineLearningConfig
from .types import (
    DataBatch,
    DriftAdaptation,
    LearningState,
    MemoryStrategy,
    ModelCheckpoint,
    ResourceMetrics,
    UpdateResult,
    UpdateStrategy,
)

logger = logging.getLogger(__name__)


class OnlineLearningPipeline:
    """オンライン学習パイプライン"""

    def __init__(self, config: OnlineLearningConfig, model: nn.Module):
        self.config = config
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # オプティマイザー設定
        self.optimizer = self._create_optimizer()

        # 学習状態管理
        self.learning_state = LearningState(
            model_version="1.0.0",
            total_samples_processed=0,
            current_learning_rate=config.learning_rate,
            gradient_norm=0.0,
            loss_history=[],
            last_update_time=datetime.now(),
            memory_usage_mb=0.0,
            gpu_memory_usage_mb=None,
        )

        # メモリ管理
        self.memory_buffer: List[DataBatch] = []
        self.sample_weights: Dict[str, float] = {}
        self.importance_scores: Dict[str, float] = {}

        # ストリーミング処理
        self.streaming_buffer: List[DataBatch] = []
        self.streaming_thread: Optional[threading.Thread] = None
        self.is_streaming = False

        # チェックポイント管理
        self.checkpoints: List[ModelCheckpoint] = []
        self.last_checkpoint_time = datetime.now()

        # パフォーマンス監視
        self.performance_metrics: List[Dict[str, Any]] = []
        self.resource_monitor = ResourceMonitor()

        # 適応制御
        self.drift_detector = DriftDetector(config.adaptation_trigger_threshold)
        self.last_adaptation_time = datetime.now()

        logger.info(
            f"Online Learning Pipeline initialized with mode: {config.learning_mode}"
        )

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """オプティマイザー作成"""
        params = self.model.parameters()

        if self.config.update_strategy == UpdateStrategy.ADAM:
            return Adam(params, lr=self.config.learning_rate)
        elif self.config.update_strategy == UpdateStrategy.SGD:
            return SGD(params, lr=self.config.learning_rate)
        elif self.config.update_strategy == UpdateStrategy.ADAGRAD:
            return Adagrad(params, lr=self.config.learning_rate)
        else:
            return Adam(params, lr=self.config.learning_rate)  # デフォルト

    def start_streaming(self, data_iterator: Iterator[DataBatch]) -> None:
        """ストリーミング学習開始"""
        if self.is_streaming:
            logger.warning("Streaming already running")
            return

        self.is_streaming = True
        self.streaming_thread = threading.Thread(
            target=self._streaming_worker, args=(data_iterator,), daemon=True
        )
        self.streaming_thread.start()
        logger.info("Streaming learning started")

    def stop_streaming(self) -> None:
        """ストリーミング学習停止"""
        self.is_streaming = False
        if self.streaming_thread:
            self.streaming_thread.join(timeout=5.0)
        logger.info("Streaming learning stopped")

    def _streaming_worker(self, data_iterator: Iterator[DataBatch]) -> None:
        """ストリーミングワーカー"""
        try:
            for batch in data_iterator:
                if not self.is_streaming:
                    break

                # バッファに追加
                self.streaming_buffer.append(batch)

                # バッファサイズチェック
                if (
                    len(self.streaming_buffer)
                    >= self.config.streaming_config.batch_size
                ):
                    # バッチ処理
                    combined_batch = self._combine_batches(self.streaming_buffer)
                    self.update_model(combined_batch)
                    self.streaming_buffer.clear()

                # 定期チェックポイント
                if (
                    datetime.now() - self.last_checkpoint_time
                ).seconds >= self.config.streaming_config.checkpoint_interval:
                    self._create_checkpoint()

        except Exception as e:
            logger.error(f"Streaming worker error: {e}")
            self.is_streaming = False

    def _combine_batches(self, batches: List[DataBatch]) -> DataBatch:
        """バッチ結合"""
        if not batches:
            raise ValueError("No batches to combine")

        # 特徴量とターゲットの結合
        features = np.concatenate([b.features for b in batches], axis=0)
        targets = np.concatenate([b.targets for b in batches], axis=0)

        # 重みの結合（ある場合）
        weights = None
        if all(b.weights is not None for b in batches):
            weights = np.concatenate([b.weights for b in batches], axis=0)

        # タイムスタンプの結合
        timestamps = []
        for b in batches:
            timestamps.extend(b.timestamps)

        return DataBatch(
            features=features,
            targets=targets,
            weights=weights,
            timestamps=timestamps,
            batch_id=f"combined_{datetime.now().isoformat()}",
            priority=1.0,
        )

    def update_model(self, batch: DataBatch) -> UpdateResult:
        """モデル更新"""
        start_time = time.time()

        try:
            # データ変換
            features = torch.FloatTensor(batch.features).to(self.device)
            targets = torch.FloatTensor(batch.targets).to(self.device)

            # メモリ管理
            self._manage_memory(batch)

            # 順伝播
            self.model.train()
            outputs = self.model(features)
            loss = self._compute_loss(outputs, targets, batch.weights)

            # 逆伝播
            self.optimizer.zero_grad()
            loss.backward()

            # 勾配クリッピング
            if self.config.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clipping
                )

            # パラメータ更新
            self.optimizer.step()

            # 状態更新
            self._update_learning_state(loss.item(), start_time)

            # ドリフト検知と適応
            drift_adaptation = self._check_and_adapt_drift(batch)

            # リソース監視
            resource_metrics = self.resource_monitor.get_metrics()

            result = UpdateResult(
                success=True,
                loss_change=loss.item(),
                gradient_norm=self.learning_state.gradient_norm,
                parameter_updates=sum(
                    p.numel() for p in self.model.parameters() if p.grad is not None
                ),
                processing_time_ms=(time.time() - start_time) * 1000,
                memory_delta_mb=resource_metrics.memory_usage_mb
                - self.learning_state.memory_usage_mb,
                error_message=None,
            )

            # パフォーマンス記録
            self._record_performance(result, resource_metrics)

            return result

        except Exception as e:
            logger.error(f"Model update failed: {e}")
            return UpdateResult(
                success=False,
                loss_change=0.0,
                gradient_norm=0.0,
                parameter_updates=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                memory_delta_mb=0.0,
                error_message=str(e),
            )

    def _manage_memory(self, batch: DataBatch) -> None:
        """メモリ管理"""
        # メモリバッファ管理
        self.memory_buffer.append(batch)

        # 戦略に応じたメモリ管理
        if self.config.memory_strategy == MemoryStrategy.SLIDING_WINDOW:
            if len(self.memory_buffer) > self.config.max_memory_samples:
                self.memory_buffer.pop(0)

        elif self.config.memory_strategy == MemoryStrategy.IMPORTANCE_SAMPLING:
            self._update_importance_scores()
            self._prune_low_importance_samples()

        elif self.config.memory_strategy == MemoryStrategy.TIME_DECAY:
            self._apply_time_decay()

        # GPUメモリ管理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # CPUメモリ管理
        gc.collect()

    def _update_importance_scores(self) -> None:
        """重要度スコア更新"""
        for batch in self.memory_buffer:
            # 簡易的な重要度計算（損失ベース）
            if batch.batch_id not in self.importance_scores:
                # 推論時の損失を重要度として使用
                with torch.no_grad():
                    features = torch.FloatTensor(batch.features).to(self.device)
                    targets = torch.FloatTensor(batch.targets).to(self.device)
                    outputs = self.model(features)
                    loss = self._compute_loss(outputs, targets, batch.weights)
                    self.importance_scores[batch.batch_id] = loss.item()

    def _prune_low_importance_samples(self) -> None:
        """低重要度サンプル削除"""
        if len(self.memory_buffer) <= self.config.max_memory_samples:
            return

        # 重要度でソート
        sorted_batches = sorted(
            self.memory_buffer,
            key=lambda b: self.importance_scores.get(b.batch_id, 0),
            reverse=True,
        )

        # 上位のみ保持
        self.memory_buffer = sorted_batches[: self.config.max_memory_samples]

    def _apply_time_decay(self) -> None:
        """時間減衰適用"""
        current_time = datetime.now()
        decayed_scores = {}

        for batch in self.memory_buffer:
            if batch.timestamps:
                # 最新タイムスタンプからの経過時間
                age_hours = (
                    current_time - max(batch.timestamps)
                ).total_seconds() / 3600
                decay_factor = np.exp(
                    -age_hours * (1 - self.config.memory_decay_factor)
                )
                decayed_scores[batch.batch_id] = decay_factor

        # 減衰適用
        for batch in self.memory_buffer:
            batch_id = batch.batch_id
            if batch_id in decayed_scores:
                if batch.weights is not None:
                    batch.weights *= decayed_scores[batch_id]

    def _compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """損失計算"""
        if isinstance(self.model, nn.Module) and hasattr(self.model, "compute_loss"):
            # カスタム損失関数がある場合
            return self.model.compute_loss(outputs, targets, weights)
        else:
            # デフォルトMSE損失
            loss = nn.MSELoss(reduction="none")(outputs.squeeze(), targets.squeeze())

            if weights is not None:
                weights_tensor = torch.FloatTensor(weights).to(self.device)
                loss = (loss * weights_tensor).mean()
            else:
                loss = loss.mean()

            return loss

    def _update_learning_state(self, loss: float, start_time: float) -> None:
        """学習状態更新"""
        self.learning_state.total_samples_processed += 1
        self.learning_state.loss_history.append(loss)
        self.learning_state.last_update_time = datetime.now()

        # 勾配ノルム計算
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        self.learning_state.gradient_norm = total_norm**0.5

        # メモリ使用量更新
        resource_metrics = self.resource_monitor.get_metrics()
        self.learning_state.memory_usage_mb = resource_metrics.memory_usage_mb
        self.learning_state.gpu_memory_usage_mb = resource_metrics.gpu_memory_mb

        # 学習履歴制限
        if len(self.learning_state.loss_history) > 1000:
            self.learning_state.loss_history = self.learning_state.loss_history[-1000:]

    def _check_and_adapt_drift(self, batch: DataBatch) -> DriftAdaptation:
        """ドリフト検知と適応"""
        adaptation = DriftAdaptation(
            drift_detected=False,
            drift_type="none",
            adaptation_applied=False,
            adaptation_params={},
            performance_impact=0.0,
        )

        if not self.config.enable_drift_adaptation:
            return adaptation

        # ドリフト検知
        drift_detected, drift_type = self.drift_detector.detect_drift(batch)

        if drift_detected:
            adaptation.drift_detected = True
            adaptation.drift_type = drift_type

            # 適応が必要かチェック
            time_since_last_adaptation = datetime.now() - self.last_adaptation_time
            if time_since_last_adaptation > timedelta(
                hours=self.config.adaptation_cooldown_hours
            ):
                # 適応実行
                adaptation_params = self._adapt_to_drift(drift_type, batch)
                adaptation.adaptation_applied = True
                adaptation.adaptation_params = adaptation_params
                self.last_adaptation_time = datetime.now()

                logger.info(f"Drift adaptation applied: {drift_type}")

        return adaptation

    def _adapt_to_drift(self, drift_type: str, batch: DataBatch) -> Dict[str, Any]:
        """ドリフト適応"""
        adaptation_params = {}

        if drift_type == "sudden_drift":
            # 学習率一時増加
            original_lr = self.config.learning_rate
            self.config.learning_rate *= 2.0
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.config.learning_rate

            adaptation_params["learning_rate_boost"] = 2.0
            adaptation_params["original_lr"] = original_lr

        elif drift_type == "gradual_drift":
            # メモリバッファリセット
            self.memory_buffer.clear()
            self.importance_scores.clear()
            adaptation_params["buffer_reset"] = True

        elif drift_type == "variance_drift":
            # バッチサイズ調整
            original_batch_size = self.config.batch_size
            self.config.batch_size = max(16, original_batch_size // 2)
            adaptation_params["batch_size_adjusted"] = self.config.batch_size
            adaptation_params["original_batch_size"] = original_batch_size

        return adaptation_params

    def _create_checkpoint(self) -> None:
        """チェックポイント作成"""
        try:
            checkpoint = ModelCheckpoint(
                version=f"{self.learning_state.model_version}.{len(self.checkpoints)}",
                timestamp=datetime.now(),
                model_state=self.model.state_dict(),
                optimizer_state=self.optimizer.state_dict(),
                metrics={
                    "total_samples": self.learning_state.total_samples_processed,
                    "current_loss": self.learning_state.loss_history[-1]
                    if self.learning_state.loss_history
                    else 0.0,
                    "gradient_norm": self.learning_state.gradient_norm,
                },
                data_signature=self._compute_data_signature(),
            )

            self.checkpoints.append(checkpoint)
            self.last_checkpoint_time = datetime.now()

            # 古いチェックポイント削除（最新10個のみ保持）
            if len(self.checkpoints) > 10:
                self.checkpoints = self.checkpoints[-10:]

            logger.info(f"Checkpoint created: {checkpoint.version}")

        except Exception as e:
            logger.error(f"Checkpoint creation failed: {e}")

    def _compute_data_signature(self) -> str:
        """データ署名計算"""
        if not self.memory_buffer:
            return "empty"

        # 最新バッチの統計情報で署名生成
        latest_batch = self.memory_buffer[-1]
        features = latest_batch.features

        signature_data = {
            "mean": float(np.mean(features)),
            "std": float(np.std(features)),
            "min": float(np.min(features)),
            "max": float(np.max(features)),
            "sample_count": len(features),
        }

        import hashlib
        import json

        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.md5(signature_str.encode()).hexdigest()

    def _record_performance(
        self, result: UpdateResult, resource_metrics: ResourceMetrics
    ) -> None:
        """パフォーマンス記録"""
        performance_record = {
            "timestamp": datetime.now(),
            "update_result": {
                "success": result.success,
                "loss_change": result.loss_change,
                "processing_time_ms": result.processing_time_ms,
                "memory_delta_mb": result.memory_delta_mb,
            },
            "resource_metrics": {
                "cpu_usage_percent": resource_metrics.cpu_usage_percent,
                "memory_usage_mb": resource_metrics.memory_usage_mb,
                "gpu_memory_mb": resource_metrics.gpu_memory_mb,
                "disk_io_mb_per_sec": resource_metrics.disk_io_mb_per_sec,
                "network_io_mb_per_sec": resource_metrics.network_io_mb_per_sec,
            },
            "learning_state": {
                "total_samples_processed": self.learning_state.total_samples_processed,
                "current_learning_rate": self.learning_state.current_learning_rate,
                "gradient_norm": self.learning_state.gradient_norm,
            },
        }

        self.performance_metrics.append(performance_record)

        # パフォーマンス履歴制限
        if len(self.performance_metrics) > 10000:
            self.performance_metrics = self.performance_metrics[-10000:]

    def get_learning_state(self) -> LearningState:
        """学習状態取得"""
        return self.learning_state

    def get_performance_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """パフォーマンスメトリクス取得"""
        return self.performance_metrics[-limit:] if self.performance_metrics else []

    def get_checkpoints(self) -> List[ModelCheckpoint]:
        """チェックポイント取得"""
        return self.checkpoints.copy()

    def load_checkpoint(self, checkpoint: ModelCheckpoint) -> bool:
        """チェックポイント読み込み"""
        try:
            self.model.load_state_dict(checkpoint.model_state)
            self.optimizer.load_state_dict(checkpoint.optimizer_state)
            self.learning_state.model_version = checkpoint.version
            logger.info(f"Checkpoint loaded: {checkpoint.version}")
            return True
        except Exception as e:
            logger.error(f"Checkpoint loading failed: {e}")
            return False


class DriftDetector:
    """ドリフト検知器"""

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.reference_stats: Optional[Dict[str, float]] = None
        self.drift_history: List[Dict[str, Any]] = []

    def detect_drift(self, batch: DataBatch) -> tuple[bool, str]:
        """ドリフト検知"""
        current_stats = self._compute_batch_stats(batch)

        if self.reference_stats is None:
            # 初期参照統計設定
            self.reference_stats = current_stats
            return False, "none"

        # 統計的検定
        drift_scores = {}
        for key in current_stats:
            if key in self.reference_stats:
                ref_val = self.reference_stats[key]
                curr_val = current_stats[key]
                if ref_val != 0:
                    drift_scores[key] = abs(curr_val - ref_val) / abs(ref_val)
                else:
                    drift_scores[key] = abs(curr_val)

        # 最大ドリフトスコア
        max_drift = max(drift_scores.values()) if drift_scores else 0.0

        if max_drift > self.threshold:
            # ドリフトタイプ判定
            drift_type = self._classify_drift_type(drift_scores)

            # ドリフト履歴記録
            self.drift_history.append(
                {
                    "timestamp": datetime.now(),
                    "drift_score": max_drift,
                    "drift_type": drift_type,
                    "stats_diff": drift_scores,
                }
            )

            # 参照統計更新（適応）
            self.reference_stats = current_stats

            return True, drift_type

        return False, "none"

    def _compute_batch_stats(self, batch: DataBatch) -> Dict[str, float]:
        """バッチ統計計算"""
        features = batch.features

        return {
            "mean": float(np.mean(features)),
            "std": float(np.std(features)),
            "skewness": float(self._compute_skewness(features)),
            "kurtosis": float(self._compute_kurtosis(features)),
            "range": float(np.ptp(features)),
            "iqr": float(np.subtract(*np.percentile(features, [75, 25]))),
        }

    def _compute_skewness(self, data: np.ndarray) -> float:
        """歪度計算"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """尖度計算"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _classify_drift_type(self, drift_scores: Dict[str, float]) -> str:
        """ドリフトタイプ分類"""
        max_key = max(drift_scores, key=drift_scores.get)

        if max_key in ["mean", "std"]:
            return "sudden_drift"  # 平均・分散の変化
        elif max_key in ["skewness", "kurtosis"]:
            return "gradual_drift"  # 分布形状の変化
        elif max_key == "range":
            return "variance_drift"  # 分散の変化
        else:
            return "unknown_drift"


class ResourceMonitor:
    """リソース監視器"""

    def __init__(self):
        self.last_disk_io = psutil.disk_io_counters()
        self.last_net_io = psutil.net_io_counters()
        self.last_time = time.time()

    def get_metrics(self) -> ResourceMetrics:
        """メトリクス取得"""
        current_time = time.time()
        time_delta = current_time - self.last_time

        # CPU使用率
        cpu_usage = psutil.cpu_percent(interval=0.1)

        # メモリ使用量
        memory = psutil.virtual_memory()
        memory_usage_mb = memory.used / 1024 / 1024

        # GPUメモリ（利用可能な場合）
        gpu_memory_mb = None
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024

        # ディスクI/O
        current_disk_io = psutil.disk_io_counters()
        disk_io_mb_per_sec = 0.0
        if self.last_disk_io and time_delta > 0:
            disk_read_mb = (
                (current_disk_io.read_bytes - self.last_disk_io.read_bytes)
                / 1024
                / 1024
            )
            disk_write_mb = (
                (current_disk_io.write_bytes - self.last_disk_io.write_bytes)
                / 1024
                / 1024
            )
            disk_io_mb_per_sec = (disk_read_mb + disk_write_mb) / time_delta

        # ネットワークI/O
        current_net_io = psutil.net_io_counters()
        net_io_mb_per_sec = 0.0
        if self.last_net_io and time_delta > 0:
            net_sent_mb = (
                (current_net_io.bytes_sent - self.last_net_io.bytes_sent) / 1024 / 1024
            )
            net_recv_mb = (
                (current_net_io.bytes_recv - self.last_net_io.bytes_recv) / 1024 / 1024
            )
            net_io_mb_per_sec = (net_sent_mb + net_recv_mb) / time_delta

        # 状態更新
        self.last_disk_io = current_disk_io
        self.last_net_io = current_net_io
        self.last_time = current_time

        return ResourceMetrics(
            cpu_usage_percent=cpu_usage,
            memory_usage_mb=memory_usage_mb,
            gpu_memory_mb=gpu_memory_mb,
            disk_io_mb_per_sec=disk_io_mb_per_sec,
            network_io_mb_per_sec=net_io_mb_per_sec,
            timestamp=datetime.now(),
        )
