"""量子化モジュール

動的量子化、静的量子化、量子化対応トレーニングを提供。
"""

import logging
from typing import Any

import torch  # type: ignore
import torch.nn as nn
from ztb.trading.environment.constants import BYTES_PER_MB  # type: ignore

logger = logging.getLogger(__name__)

class DynamicQuantization:
    """動的量子化クラス

    推論時に動的に量子化を実行。
    """

    def __init__(self, dtype: torch.dtype = torch.qint8):
        """
        Args:
            dtype: 量子化データ型 (torch.qint8, torch.quint8, etc.)
        """
        self.dtype = dtype

    def quantize_model(self, model: nn.Module) -> nn.Module:
        """モデルを動的量子化

        Args:
            model: 量子化対象のモデル

        Returns:
            量子化されたモデル
        """
        # 量子化設定
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

        # モデルを量子化対応に変換
        model_prepared = torch.quantization.prepare(model, inplace=False)

        # キャリブレーション（ダミーデータで実行）
        self._calibrate_model(model_prepared)

        # 量子化実行
        model_quantized = torch.quantization.convert(model_prepared, inplace=False)

        logger.info(f"動的量子化完了: dtype={self.dtype}")
        return model_quantized

    def _calibrate_model(self, model: nn.Module, num_batches: int = 10):
        """キャリブレーション実行"""
        model.eval()

        with torch.no_grad():
            for _ in range(num_batches):
                # ダミー入力でキャリブレーション
                batch_size = 4
                seq_len = 8

                # マルチモーダル入力のダミーデータ
                price_data = torch.randn(batch_size, 156)
                text_data = torch.randint(0, 1000, (batch_size, seq_len))
                economic_data = torch.randn(batch_size, 20)

                try:
                    model(price_data, text_data, economic_data)
                except Exception:
                    # 単純なテンソル入力で試行
                    dummy_input = torch.randn(batch_size, 256)
                    model(dummy_input)

class StaticQuantization:
    """静的量子化クラス

    事前キャリブレーションによる静的量子化。
    """

    def __init__(self, dtype: torch.dtype = torch.qint8):
        """
        Args:
            dtype: 量子化データ型
        """
        self.dtype = dtype

    def quantize_model(
        self, model: nn.Module, calibration_data: list[torch.Tensor]
    ) -> nn.Module:
        """モデルを静的量子化

        Args:
            model: 量子化対象のモデル
            calibration_data: キャリブレーションデータ

        Returns:
            量子化されたモデル
        """
        # 量子化設定
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

        # モデルを量子化対応に変換
        model_prepared = torch.quantization.prepare(model, inplace=False)

        # キャリブレーション実行
        self._calibrate_with_data(model_prepared, calibration_data)

        # 量子化実行
        model_quantized = torch.quantization.convert(model_prepared, inplace=False)

        logger.info(f"静的量子化完了: dtype={self.dtype}")
        return model_quantized

    def _calibrate_with_data(
        self, model: nn.Module, calibration_data: list[torch.Tensor]
    ):
        """キャリブレーションデータでキャリブレーション"""
        model.eval()

        with torch.no_grad():
            for data in calibration_data:
                try:
                    model(data)
                except Exception as e:
                    logger.warning(f"キャリブレーション中にエラー: {e}")

class QuantizationAwareTraining:
    """量子化対応トレーニングクラス

    トレーニング中に量子化を考慮した学習を実行。
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: 量子化対応トレーニング対象のモデル
        """
        self.model = model

    def prepare_model(self) -> nn.Module:
        """量子化対応トレーニング用にモデルを準備"""
        # 量子化設定
        self.model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")

        # モデルをQAT対応に変換
        model_prepared = torch.quantization.prepare_qat(self.model, inplace=False)

        logger.info("量子化対応トレーニング用モデル準備完了")
        return model_prepared

    def convert_to_quantized(self, model_prepared: nn.Module) -> nn.Module:
        """トレーニング済みモデルを量子化モデルに変換"""
        model_quantized = torch.quantization.convert(model_prepared, inplace=False)

        logger.info("量子化モデル変換完了")
        return model_quantized

class QuantizationUtils:
    """量子化ユーティリティクラス"""

    @staticmethod
    def get_model_size(model: nn.Module) -> dict[str, float]:
        """モデルのサイズ情報を取得"""
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        total_size = param_size + buffer_size

        return {
            "parameters_mb": param_size / BYTES_PER_MB,
            "buffers_mb": buffer_size / BYTES_PER_MB,
            "total_mb": total_size / BYTES_PER_MB,
        }

    @staticmethod
    def measure_inference_time(
        model: nn.Module, input_data: torch.Tensor, num_runs: int = 100
    ) -> dict[str, float]:
        """推論時間を測定"""
        import time

        model.eval()

        # ウォームアップ
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_data)

        # 測定
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(input_data)
                end_time = time.time()
                times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        std_time = torch.tensor(times).std().item()

        return {
            "avg_inference_time": avg_time,
            "std_inference_time": std_time,
            "fps": 1.0 / avg_time,
        }

    @staticmethod
    def compare_models(
        original_model: nn.Module, quantized_model: nn.Module, test_data: torch.Tensor
    ) -> dict[str, Any]:
        """オリジナルモデルと量子化モデルの比較"""

        # サイズ比較
        orig_size = QuantizationUtils.get_model_size(original_model)
        quant_size = QuantizationUtils.get_model_size(quantized_model)

        # 推論時間比較
        orig_time = QuantizationUtils.measure_inference_time(original_model, test_data)
        quant_time = QuantizationUtils.measure_inference_time(
            quantized_model, test_data
        )

        # 出力比較（数値精度）
        original_model.eval()
        quantized_model.eval()

        with torch.no_grad():
            orig_output = original_model(test_data)
            quant_output = quantized_model(test_data)

        # MSEで精度比較
        mse = nn.MSELoss()(orig_output, quant_output).item()

        return {
            "size_comparison": {
                "original_mb": orig_size["total_mb"],
                "quantized_mb": quant_size["total_mb"],
                "compression_ratio": quant_size["total_mb"] / orig_size["total_mb"],
            },
            "performance_comparison": {
                "original_fps": orig_time["fps"],
                "quantized_fps": quant_time["fps"],
                "speedup_ratio": quant_time["fps"] / orig_time["fps"],
            },
            "accuracy": {"mse": mse, "output_similarity": 1.0 - mse},  # 簡易的な類似度
        }
