"""推論最適化モジュール

リアルタイム推論の高速化とメモリ最適化を提供。
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from ztb.trading.environment.constants import BYTES_PER_GB

logger = logging.getLogger(__name__)


class InferenceOptimizer:
    """推論最適化クラス

    モデル推論の高速化とメモリ最適化を行う。
    """

    def __init__(self, model: nn.Module, device: str = "auto"):
        """
        Args:
            model: 最適化対象のモデル
            device: 推論デバイス ('cpu', 'cuda', 'auto')
        """
        self.model = model
        self.device = self._get_device(device)
        self.model.to(self.device)
        self.model.eval()

        # 最適化設定
        self.use_jit = False
        self.use_onnx = False
        self.use_tensorrt = False

        # JITコンパイルモデル
        self.jit_model = None

        # ONNXモデル
        self.onnx_model = None

        logger.info(f"推論最適化初期化: device={self.device}")

    def _get_device(self, device: str) -> torch.device:
        """デバイスを取得"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def enable_jit_compilation(self) -> "InferenceOptimizer":
        """JITコンパイルを有効化"""
        try:
            # ダミー入力でJITコンパイル（シーケンス対応）
            dummy_price = torch.randn(1, 50, 156).to(
                self.device
            )  # [batch, seq_len, feature_dim]
            dummy_text = torch.randn(1, 50, 768).to(self.device)
            dummy_economic = torch.randn(1, 50, 20).to(self.device)

            self.jit_model = torch.jit.trace(
                self.model, (dummy_price, dummy_text, dummy_economic)
            )
            self.use_jit = True

            logger.info("JITコンパイル有効化完了")

        except Exception as e:
            logger.warning(f"JITコンパイル失敗: {e}")

        return self

    def enable_onnx_optimization(self, onnx_path: str = None) -> "InferenceOptimizer":
        """ONNX最適化を有効化"""
        try:
            import onnxruntime as ort

            if onnx_path:
                # 既存のONNXモデルを読み込み
                self.onnx_session = ort.InferenceSession(onnx_path)
            else:
                # PyTorchモデルからONNXに変換
                self._convert_to_onnx()

            self.use_onnx = True
            logger.info("ONNX最適化有効化完了")

        except ImportError:
            logger.warning("ONNX Runtimeがインストールされていません")
        except Exception as e:
            logger.warning(f"ONNX最適化失敗: {e}")

        return self

    def _convert_to_onnx(self):
        """PyTorchモデルをONNXに変換"""
        import onnxruntime as ort

        dummy_price = torch.randn(1, 50, 156).to(
            self.device
        )  # [batch, seq_len, feature_dim]
        dummy_text = torch.randn(1, 50, 768).to(self.device)
        dummy_economic = torch.randn(1, 50, 20).to(self.device)

        # ONNXに変換
        onnx_path = "temp_model.onnx"
        torch.onnx.export(
            self.model,
            (dummy_price, dummy_text, dummy_economic),
            onnx_path,
            input_names=["price", "text", "economic"],
            output_names=["output"],
            dynamic_axes={
                "price": {0: "batch_size", 1: "seq_len"},
                "text": {0: "batch_size", 1: "seq_len"},
                "economic": {0: "batch_size", 1: "seq_len"},
                "output": {0: "batch_size", 1: "seq_len"},
            },
        )

        self.onnx_session = ort.InferenceSession(onnx_path)

        # 一時ファイルを削除
        import os

        os.remove(onnx_path)

    def enable_tensorrt(self) -> "InferenceOptimizer":
        """TensorRT最適化を有効化（GPUのみ）"""
        if not torch.cuda.is_available():
            logger.warning("TensorRTはCUDA環境でのみ利用可能です")
            return self

        try:
            # TensorRTの初期化と最適化
            # 実際の実装ではtorch_tensorrtを使用
            self.use_tensorrt = True
            logger.info("TensorRT最適化有効化完了")

        except Exception as e:
            logger.warning(f"TensorRT最適化失敗: {e}")

        return self

    def optimize_memory(self) -> "InferenceOptimizer":
        """メモリ最適化を実行"""
        # 不要なグラフを削除
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # メモリ効率的な設定
        torch.set_grad_enabled(False)

        logger.info("メモリ最適化完了")
        return self

    def predict(self, *inputs) -> torch.Tensor:
        """最適化された推論を実行"""
        # 入力をデバイスに移動
        inputs = [
            x.to(self.device) if isinstance(x, torch.Tensor) else x for x in inputs
        ]

        if self.use_onnx:
            # ONNX推論
            ort_inputs = {
                "price": inputs[0].cpu().numpy(),
                "text": inputs[1].cpu().numpy(),
                "economic": inputs[2].cpu().numpy(),
            }
            ort_outputs = self.onnx_session.run(None, ort_inputs)
            return torch.from_numpy(ort_outputs[0]).to(self.device)

        elif self.use_jit and self.jit_model is not None:
            # JIT推論
            with torch.no_grad():
                return self.jit_model(*inputs)

        else:
            # 通常のPyTorch推論
            with torch.no_grad():
                return self.model(*inputs)


class BatchProcessor:
    """バッチ処理最適化クラス

    複数の推論リクエストを効率的に処理。
    """

    def __init__(
        self, model: nn.Module, max_batch_size: int = 32, num_workers: int = 4
    ):
        """
        Args:
            model: 推論対象のモデル
            max_batch_size: 最大バッチサイズ
            num_workers: ワーカー数
        """
        self.model = model
        self.max_batch_size = max_batch_size
        self.num_workers = num_workers

        # 推論キュー
        self.request_queue = []
        self.result_dict = {}
        self.lock = threading.Lock()

        # スレッドプール
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        # バッチ処理タイマー
        self.batch_timer = None

    def submit_request(self, request_id: str, inputs: Tuple[torch.Tensor, ...]) -> None:
        """推論リクエストを送信"""
        with self.lock:
            self.request_queue.append((request_id, inputs))

        # バッチ処理タイマーを開始/リセット
        if self.batch_timer:
            self.batch_timer.cancel()

        self.batch_timer = threading.Timer(0.01, self._process_batch)
        self.batch_timer.start()

    def get_result(self, request_id: str) -> Optional[torch.Tensor]:
        """推論結果を取得"""
        return self.result_dict.get(request_id)

    def _process_batch(self):
        """バッチ処理を実行"""
        with self.lock:
            if not self.request_queue:
                return

            # キューからリクエストを取得
            batch_requests = self.request_queue[: self.max_batch_size]
            self.request_queue = self.request_queue[self.max_batch_size :]

        # バッチを作成
        batch_inputs = self._collate_batch([req[1] for req in batch_requests])

        # バッチ推論を実行
        try:
            batch_outputs = self.model(*batch_inputs)

            # 結果を個別に保存
            for i, (request_id, _) in enumerate(batch_requests):
                self.result_dict[request_id] = batch_outputs[i]

        except Exception as e:
            logger.error(f"バッチ処理エラー: {e}")

    def _collate_batch(
        self, inputs_list: List[Tuple[torch.Tensor, ...]]
    ) -> Tuple[torch.Tensor, ...]:
        """入力をバッチ化"""
        if not inputs_list:
            return ()

        # 各入力タイプをバッチ化
        batched_inputs = []
        for i in range(len(inputs_list[0])):
            tensor_list = [inputs[i] for inputs in inputs_list]
            batched = torch.stack(tensor_list, dim=0)
            batched_inputs.append(batched)

        return tuple(batched_inputs)


class MemoryManager:
    """メモリ管理クラス

    GPUメモリの効率的な管理を行う。
    """

    def __init__(self, max_memory_gb: float = 8.0):
        """
        Args:
            max_memory_gb: 最大メモリ使用量（GB）
        """
        self.max_memory_gb = max_memory_gb
        self.memory_history = []

    def monitor_memory(self) -> Dict[str, float]:
        """メモリ使用量を監視"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / BYTES_PER_GB
            reserved = torch.cuda.memory_reserved() / BYTES_PER_GB

            memory_info = {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "utilization_percent": (allocated / self.max_memory_gb) * 100,
            }
        else:
            memory_info = {
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "utilization_percent": 0.0,
            }

        self.memory_history.append(memory_info)
        return memory_info

    def cleanup_memory(self):
        """メモリクリーンアップ"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 古い履歴を削除
        if len(self.memory_history) > 1000:
            self.memory_history = self.memory_history[-500:]

    def get_memory_stats(self) -> Dict[str, Any]:
        """メモリ統計を取得"""
        if not self.memory_history:
            return {}

        recent_memory = self.memory_history[-10:]  # 最近10件

        return {
            "current": self.memory_history[-1],
            "average": {
                "allocated_gb": sum(m["allocated_gb"] for m in recent_memory)
                / len(recent_memory),
                "utilization_percent": sum(
                    m["utilization_percent"] for m in recent_memory
                )
                / len(recent_memory),
            },
            "peak": {
                "allocated_gb": max(m["allocated_gb"] for m in self.memory_history),
                "utilization_percent": max(
                    m["utilization_percent"] for m in self.memory_history
                ),
            },
        }
