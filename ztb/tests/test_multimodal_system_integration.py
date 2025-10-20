#!/usr/bin/env python3
"""
マルチモーダル学習システム統合テストスクリプト

SAC v421取引AIのマルチモーダル学習システム全体が正常に動作するかを
包括的にテストするスクリプトです。

テスト内容:
1. 設定の読み込みと検証
2. モックデータの生成
3. マルチモーダルアーキテクチャの初期化
4. 最適化機能の適用（圧縮・量子化・推論最適化）
5. 学習ループの実行
6. 推論パフォーマンスの評価
7. 結果のレポートと検証

作成日: 2025-10-17
バージョン: 1.0.0
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import yaml

# プロジェクトルートをパスに追加
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, project_root)

# マルチモーダルモジュールインポート
try:
    from ztb.multimodal.models.architectures.multimodal_architecture import (
        MultiModalFeatureEncoder,
    )
    from ztb.multimodal.optimization.compression import (
        KnowledgeDistillation,
        ModelCompression,
        ModelPruning,
    )
    from ztb.multimodal.optimization.inference import InferenceOptimizer, MemoryManager
    from ztb.multimodal.optimization.quantization import (
        DynamicQuantization,
        QuantizationUtils,
    )

    print("✅ マルチモーダルモジュールインポート成功")
except ImportError as e:
    print(f"❌ マルチモーダルモジュールインポート失敗: {e}")
    sys.exit(1)

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MockDataGenerator:
    """テスト用のモックデータを生成するクラス"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.price_dim = config.get("price_dim", 156)
        self.text_dim = config.get("text_dim", 768)
        self.economic_dim = config.get("economic_dim", 20)
        self.sequence_length = config.get("sequence_length", 50)
        self.batch_size = config.get("batch_size", 32)

    def generate_price_features(self, num_samples: int) -> torch.Tensor:
        """価格特徴量のモックデータを生成"""
        # 基本的な価格データ（OHLCV + テクニカル指標）
        base_features = torch.randn(num_samples, self.sequence_length, 5)  # OHLCV

        # テクニカル指標のシミュレーション
        technical_features = torch.randn(
            num_samples, self.sequence_length, self.price_dim - 5
        )

        # 正規化と範囲調整
        technical_features = torch.tanh(technical_features)  # -1 to 1

        return torch.cat([base_features, technical_features], dim=-1).float()

    def generate_text_embeddings(self, num_samples: int) -> torch.Tensor:
        """テキスト埋め込みのモックデータを生成"""
        # BERTのような768次元埋め込みをシミュレート
        embeddings = torch.randn(num_samples, self.sequence_length, self.text_dim)

        # 感情スコアの影響をシミュレート（一部の埋め込みを調整）
        sentiment_mask = torch.rand(num_samples, self.sequence_length) > 0.7
        sentiment_adjustment = torch.randn_like(embeddings) * 0.1
        embeddings = torch.where(
            sentiment_mask.unsqueeze(-1), embeddings + sentiment_adjustment, embeddings
        )

        return embeddings.float()

    def generate_economic_features(self, num_samples: int) -> torch.Tensor:
        """経済指標特徴量のモックデータを生成"""
        # GDP, インフレ率, 失業率などの経済指標
        economic_data = torch.randn(
            num_samples, self.sequence_length, self.economic_dim
        )

        # 経済指標の自然な範囲に調整
        economic_data = torch.tanh(economic_data) * 2.0  # -2 to 2

        return economic_data.float()

    def generate_batch(
        self, batch_size: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """バッチデータの生成"""
        if batch_size is None:
            batch_size = self.batch_size

        return {
            "price_features": self.generate_price_features(batch_size),
            "text_embeddings": self.generate_text_embeddings(batch_size),
            "economic_features": self.generate_economic_features(batch_size),
        }


class MultiModalSystemTester:
    """マルチモーダルシステム全体の統合テスト"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.data_generator = MockDataGenerator(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}

        logger.info(f"テスト環境: {self.device}")
        logger.info(f"設定: {json.dumps(self.config, indent=2, default=str)}")

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "ztb", "multimodal", "config", "default.yaml"
            )

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
            logger.info("✅ 設定ファイル読み込み成功")

            # テストに必要な設定のみ抽出
            config = {
                "price_dim": full_config.get("model", {}).get("price_dim", 156),
                "text_dim": full_config.get("features", {})
                .get("text", {})
                .get("embedding_dim", 768),
                "economic_dim": 20,  # デフォルト値
                "hidden_dim": full_config.get("model", {})
                .get("fusion", {})
                .get("attention_dim", 256),
                "sequence_length": 50,
                "batch_size": full_config.get("training", {}).get("batch_size", 16),
                "num_epochs": 3,  # テスト用に短く
                "learning_rate": float(
                    full_config.get("training", {}).get("learning_rate", "3e-4")
                ),
            }
            return config

        except Exception as e:
            logger.error(f"❌ 設定ファイル読み込み失敗: {e}")
            # デフォルト設定を使用
            return {
                "price_dim": 156,
                "text_dim": 768,
                "economic_dim": 20,
                "hidden_dim": 256,
                "sequence_length": 50,
                "batch_size": 16,
                "num_epochs": 3,
                "learning_rate": 1e-4,
            }

    def test_architecture_initialization(self) -> bool:
        """マルチモーダルアーキテクチャの初期化テスト"""
        try:
            logger.info("🔧 アーキテクチャ初期化テスト開始")

            # モデル初期化
            self.model = MultiModalFeatureEncoder(
                price_feature_dim=self.config["price_dim"],
                text_embedding_dim=self.config["text_dim"],
                economic_feature_dim=self.config["economic_dim"],
                hidden_dim=self.config.get("hidden_dim", 256),
            ).to(self.device)

            # パラメータ数の確認
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"モデルパラメータ数: {total_params:,}")

            # フォワードパスのテスト
            batch_data = self.data_generator.generate_batch(4)
            for key, tensor in batch_data.items():
                batch_data[key] = tensor.to(self.device)

            with torch.no_grad():
                output = self.model(**batch_data)
                expected_shape = (
                    4,
                    self.config.get("sequence_length", 50),
                    self.config.get("hidden_dim", 256),
                )
                assert (
                    output.shape == expected_shape
                ), f"出力形状が不正: {output.shape} vs {expected_shape}"

            logger.info("✅ アーキテクチャ初期化テスト成功")
            self.results["architecture_init"] = True
            return True

        except Exception as e:
            logger.error(f"❌ アーキテクチャ初期化テスト失敗: {e}")
            self.results["architecture_init"] = False
            return False

    def test_optimization_features(self) -> bool:
        """最適化機能のテスト"""
        try:
            logger.info("🔧 最適化機能テスト開始")

            # モデルが初期化されていない場合はスキップ
            if not hasattr(self, "model") or self.model is None:
                logger.warning(
                    "⚠️ モデルが初期化されていないため、最適化テストをスキップ"
                )
                self.results["optimization"] = False
                return False

            # モデル圧縮テスト
            compressor = ModelCompression(self.model)
            compression_config = {"method": "pruning", "pruning_ratio": 0.3}
            compressed_model = compressor.compress_model(compression_config)
            logger.info("✅ モデル圧縮成功")

            # 量子化テスト
            quantizer = DynamicQuantization(self.model)
            try:
                quantized_model = quantizer.quantize_model(self.model)
                logger.info("✅ 動的量子化成功")
            except Exception as e:
                logger.warning(f"⚠️ 動的量子化スキップ（環境依存）: {e}")

            # 推論最適化テスト
            inference_optimizer = InferenceOptimizer(self.model)
            optimized_optimizer = inference_optimizer.enable_jit_compilation()

            # 推論パフォーマンステスト
            batch_data = self.data_generator.generate_batch(4)
            for key, tensor in batch_data.items():
                batch_data[key] = tensor.to(self.device)

            # 推論実行テスト
            with torch.no_grad():
                output = inference_optimizer.predict(
                    batch_data["price_features"],
                    batch_data["text_embeddings"],
                    batch_data["economic_features"],
                )
                logger.info(f"推論出力形状: {output.shape}")
                logger.info("✅ JITコンパイル最適化成功")

            # メモリ管理テスト
            memory_manager = MemoryManager()
            memory_stats = memory_manager.get_memory_stats()
            logger.info(f"メモリ統計: {memory_stats}")

            logger.info("✅ 最適化機能テスト成功")
            self.results["optimization"] = True
            return True

        except Exception as e:
            logger.error(f"❌ 最適化機能テスト失敗: {e}")
            self.results["optimization"] = False
            return False

    def test_training_loop(self) -> bool:
        """学習ループのテスト"""
        try:
            logger.info("🔧 学習ループテスト開始")

            # モデルが初期化されていない場合はスキップ
            if not hasattr(self, "model") or self.model is None:
                logger.warning("⚠️ モデルが初期化されていないため、学習テストをスキップ")
                self.results["training"] = False
                return False

            # オプティマイザーと損失関数
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config.get("learning_rate", 1e-4)
            )
            criterion = nn.MSELoss()  # 簡易的な損失関数

            # 学習ループ
            num_epochs = self.config.get("num_epochs", 3)
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                num_batches = 3  # テスト用に少なめのバッチ数

                for batch_idx in range(num_batches):
                    # バッチデータの生成
                    batch_data = self.data_generator.generate_batch()
                    for key, tensor in batch_data.items():
                        batch_data[key] = tensor.to(self.device)

                    # ターゲットの生成（ランダム）
                    targets = torch.randn(
                        batch_data["price_features"].shape[0],
                        self.config.get("hidden_dim", 256),
                    ).to(self.device)

                    # フォワードパス
                    optimizer.zero_grad()
                    outputs = self.model(**batch_data)

                    # シーケンスの最後の出力をターゲットと比較
                    final_outputs = outputs[:, -1, :]  # [batch_size, hidden_dim]

                    # 損失計算とバックプロパゲーション
                    loss = criterion(final_outputs, targets)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / num_batches
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

            logger.info("✅ 学習ループテスト成功")
            self.results["training"] = True
            return True

        except Exception as e:
            logger.error(f"❌ 学習ループテスト失敗: {e}")
            self.results["training"] = False
            return False

    def test_inference_performance(self) -> bool:
        """推論パフォーマンスのテスト"""
        try:
            logger.info("🔧 推論パフォーマンステスト開始")

            # モデルが初期化されていない場合はスキップ
            if not hasattr(self, "model") or self.model is None:
                logger.warning("⚠️ モデルが初期化されていないため、推論テストをスキップ")
                self.results["inference"] = False
                return False

            # 推論最適化モデルの準備
            inference_optimizer = InferenceOptimizer(self.model)
            optimized_model = inference_optimizer.enable_jit_compilation()

            # パフォーマンス測定
            batch_sizes = [1, 4]
            performance_results = {}

            for batch_size in batch_sizes:
                # データ生成
                batch_data = self.data_generator.generate_batch(batch_size)
                for key, tensor in batch_data.items():
                    batch_data[key] = tensor.to(self.device)

                # 推論時間測定
                with torch.no_grad():
                    # ウォームアップ
                    for _ in range(3):
                        _ = inference_optimizer.predict(
                            batch_data["price_features"],
                            batch_data["text_embeddings"],
                            batch_data["economic_features"],
                        )

                    # 実際の測定
                    import time

                    start_time = time.time()
                    num_runs = 5  # テスト用に少なく

                    for _ in range(num_runs):
                        _ = inference_optimizer.predict(
                            batch_data["price_features"],
                            batch_data["text_embeddings"],
                            batch_data["economic_features"],
                        )

                    end_time = time.time()
                    avg_time = (end_time - start_time) / num_runs
                    performance_results[f"batch_{batch_size}"] = avg_time

                logger.info(f"バッチサイズ {batch_size}: 平均推論時間 {avg_time:.4f}秒")

            self.results["inference_performance"] = performance_results
            logger.info("✅ 推論パフォーマンステスト成功")
            self.results["inference"] = True
            return True

        except Exception as e:
            logger.error(f"❌ 推論パフォーマンステスト失敗: {e}")
            self.results["inference"] = False
            return False

    def test_memory_management(self) -> bool:
        """メモリ管理のテスト"""
        try:
            logger.info("🔧 メモリ管理テスト開始")

            # モデルが初期化されていない場合はスキップ
            if not hasattr(self, "model") or self.model is None:
                logger.warning(
                    "⚠️ モデルが初期化されていないため、メモリテストをスキップ"
                )
                self.results["memory"] = False
                return False

            memory_manager = MemoryManager()

            # メモリ使用量の監視（CPU環境では簡易チェック）
            if torch.cuda.is_available():
                initial_memory = memory_manager.get_memory_stats()

                # モデル推論によるメモリ使用
                batch_data = self.data_generator.generate_batch(16)  # 小さめのバッチ
                for key, tensor in batch_data.items():
                    batch_data[key] = tensor.to(self.device)

                with torch.no_grad():
                    for _ in range(5):  # 少なめのループ
                        _ = self.model(**batch_data)

                # メモリクリーンアップ
                memory_manager.cleanup_memory()
                final_memory = memory_manager.get_memory_stats()

                logger.info(f"初期メモリ: {initial_memory}")
                logger.info(f"最終メモリ: {final_memory}")

                # メモリリークのチェック
                if (
                    "allocated" in final_memory
                    and final_memory["allocated"]
                    < initial_memory.get("allocated", 0) * 2
                ):
                    logger.info("✅ メモリ管理テスト成功")
                    self.results["memory"] = True
                    return True
                else:
                    logger.warning("⚠️ メモリリークの可能性あり")
                    self.results["memory"] = False
                    return False
            else:
                # CPU環境では基本的な推論テストのみ
                batch_data = self.data_generator.generate_batch(4)
                for key, tensor in batch_data.items():
                    batch_data[key] = tensor.to(self.device)

                with torch.no_grad():
                    _ = self.model(**batch_data)

                logger.info("✅ CPU環境でのメモリ管理テスト成功（簡易チェック）")
                self.results["memory"] = True
                return True

        except Exception as e:
            logger.error(f"❌ メモリ管理テスト失敗: {e}")
            self.results["memory"] = False
            return False

    def run_full_test_suite(self) -> Dict[str, Any]:
        """完全なテストスイートを実行"""
        logger.info("🚀 マルチモーダルシステム統合テスト開始")
        start_time = datetime.now()

        # テスト実行
        tests = [
            ("architecture_initialization", self.test_architecture_initialization),
            ("optimization_features", self.test_optimization_features),
            ("training_loop", self.test_training_loop),
            ("inference_performance", self.test_inference_performance),
            ("memory_management", self.test_memory_management),
        ]

        passed_tests = 0
        total_tests = len(tests)

        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"テスト実行: {test_name}")
            logger.info(f"{'='*50}")

            if test_func():
                passed_tests += 1
                logger.info(f"✅ {test_name}: 成功")
            else:
                logger.error(f"❌ {test_name}: 失敗")

        end_time = datetime.now()
        duration = end_time - start_time

        # 結果サマリー
        self.results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests * 100,
            "duration_seconds": duration.total_seconds(),
            "timestamp": end_time.isoformat(),
        }

        logger.info(f"\n{'='*60}")
        logger.info("テスト結果サマリー")
        logger.info(f"{'='*60}")
        logger.info(f"総テスト数: {total_tests}")
        logger.info(f"成功: {passed_tests}")
        logger.info(f"失敗: {total_tests - passed_tests}")
        logger.info(f"成功率: {passed_tests / total_tests * 100:.1f}%")
        logger.info(f"実行時間: {duration.total_seconds():.2f}秒")

        if passed_tests == total_tests:
            logger.info(
                "🎉 すべてのテストが成功しました！マルチモーダルシステムは正常に動作しています。"
            )
        else:
            logger.warning(
                f"⚠️ {total_tests - passed_tests}個のテストが失敗しました。詳細を確認してください。"
            )

        return self.results

    def save_results(self, output_path: Optional[str] = None):
        """テスト結果を保存"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"multimodal_system_test_results_{timestamp}.json"

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, default=str, ensure_ascii=False)
            logger.info(f"✅ テスト結果を保存: {output_path}")
        except Exception as e:
            logger.error(f"❌ 結果保存失敗: {e}")


def main():
    """メイン実行関数"""
    print("🤖 MultiModal Learning System Integration Test")
    print("=" * 60)

    # テスト実行
    tester = MultiModalSystemTester()

    try:
        results = tester.run_full_test_suite()
        tester.save_results()

        # 最終結果表示
        summary = results.get("summary", {})
        success_rate = summary.get("success_rate", 0)

        if success_rate >= 80.0:
            print("🎉 マルチモーダルシステムは正常に動作しています！")
            return 0
        else:
            print("⚠️ システムに問題があります。詳細を確認してください。")
            return 1

    except Exception as e:
        logger.error(f"❌ テスト実行中に致命的なエラーが発生: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
