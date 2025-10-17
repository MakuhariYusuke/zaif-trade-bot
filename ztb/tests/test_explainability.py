#!/usr/bin/env python3
"""
Explainability Module Test
説明可能性モジュールのテスト
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import logging

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ztb.adaptation.explainability.analyzer import ExplainabilityAnalyzer
from ztb.adaptation.explainability.config import ExplainabilityConfig

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleTradingModel(nn.Module):
    """シンプルな取引モデル（テスト用）"""

    def __init__(self, input_size=10, hidden_size=32, output_size=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self._version = "1.0.0"

    def forward(self, x):
        return self.network(x)


def test_explainability_analyzer():
    """説明可能性アナライザーのテスト"""
    logger.info("Starting Explainability Analyzer Test")

    try:
        # 設定の初期化
        config = ExplainabilityConfig(
            enabled=True,
            generate_natural_language=True,
            max_features_to_explain=5
        )

        # アナライザーの初期化
        analyzer = ExplainabilityAnalyzer(config)

        # テストモデルの作成
        model = SimpleTradingModel(input_size=10)
        model.eval()

        # テストデータの生成
        test_data = torch.randn(1, 10)  # 1サンプル、10特徴量
        background_data = torch.randn(50, 10)  # 背景データ

        # 予測の実行
        with torch.no_grad():
            prediction = model(test_data)
            predicted_class = torch.argmax(prediction, dim=1).item()

        logger.info("Testing prediction explanation...")

        # 説明の生成
        explanation = analyzer.explain_prediction(
            model=model,
            input_data=test_data,
            prediction=predicted_class,
            background_data=background_data
        )

        # 結果の検証
        assert explanation.explanation_id is not None
        assert explanation.timestamp is not None
        assert len(explanation.feature_importance) > 0
        assert explanation.processing_time_seconds > 0

        logger.info(f"Explanation generated successfully in {explanation.processing_time_seconds:.3f}s")
        logger.info(f"Top features: {[fi.feature_name for fi in explanation.feature_importance[:3]]}")

        if explanation.decision_explanation:
            logger.info(f"Decision explanation: {explanation.decision_explanation.natural_language_explanation}")

        # キャッシュ機能のテスト
        logger.info("Testing caching functionality...")
        cached_result = analyzer.get_cached_explanation(explanation.explanation_id)
        assert cached_result is not None
        assert cached_result.explanation_id == explanation.explanation_id

        # 特徴量重要度サマリーのテスト
        logger.info("Testing feature importance summary...")
        explanations = [explanation]  # 複数の説明でテスト
        summary = analyzer.get_feature_importance_summary(explanations)
        assert len(summary) > 0

        logger.info("Explainability Analyzer Test completed successfully")
        return True

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """設定テスト"""
    logger.info("Testing Explainability Configuration")

    try:
        # 有効な設定
        config = ExplainabilityConfig()
        assert config.enabled == True
        assert config.max_features_to_explain == 10

        # カスタム設定
        custom_config = ExplainabilityConfig(
            enabled=False,
            max_features_to_explain=5,
            generate_natural_language=False
        )
        assert custom_config.enabled == False
        assert custom_config.max_features_to_explain == 5

        logger.info("Configuration test passed")
        return True

    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        return False


def test_feature_mapping():
    """特徴量マッピングテスト"""
    logger.info("Testing Feature Mapping")

    try:
        config = ExplainabilityConfig()

        # 特徴量名のテスト
        analyzer = ExplainabilityAnalyzer(config)

        # インデックスから特徴量名を取得
        feature_name = analyzer._get_feature_name(0)
        assert feature_name in config.feature_names

        # 特徴量カテゴリのテスト
        category = config.feature_categories.get(feature_name)
        assert category is not None

        logger.info("Feature mapping test passed")
        return True

    except Exception as e:
        logger.error(f"Feature mapping test failed: {e}")
        return False


def main():
    """メイン実行関数"""
    logger.info("=== Explainability Module Test ===")

    # 設定テスト
    if not test_configuration():
        logger.error("Configuration test failed")
        return 1

    # 特徴量マッピングテスト
    if not test_feature_mapping():
        logger.error("Feature mapping test failed")
        return 1

    # 説明可能性アナライザーテスト
    if not test_explainability_analyzer():
        logger.error("Explainability analyzer test failed")
        return 1

    logger.info("All tests passed successfully!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)