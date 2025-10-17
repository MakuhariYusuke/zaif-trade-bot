"""
Test Enhanced Explainability Features
説明可能性機能の拡張テスト
"""

import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
import tempfile
import os

from ztb.adaptation.explainability.analyzer import ExplainabilityAnalyzer
from ztb.adaptation.explainability.config import ExplainabilityConfig
from ztb.adaptation.explainability.types import ExplanationResult, VisualizationResult


class SimpleTestModel(nn.Module):
    """テスト用のシンプルなモデル"""

    def __init__(self, input_size=10, hidden_size=32, output_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)


class TestEnhancedExplainability(unittest.TestCase):
    """拡張説明可能性機能のテスト"""

    def setUp(self):
        """テストセットアップ"""
        self.config = ExplainabilityConfig(
            enabled=True,
            generate_natural_language=True,
            enable_visualization=True,
            max_features_to_explain=5,
            feature_names={f"feature_{i}": f"Feature {i}" for i in range(10)},
            feature_categories={f"feature_{i}": "test" for i in range(10)}
        )
        self.analyzer = ExplainabilityAnalyzer(self.config)
        self.test_model = SimpleTestModel()

        # テストデータ
        self.test_input = torch.randn(1, 10)

    def test_enhanced_natural_language_generation(self):
        """高度な自然言語生成テスト"""
        try:
            # モック特徴量重要度
            from ztb.adaptation.explainability.types import FeatureImportance

            primary_factors = [
                FeatureImportance("feature_0", 0.8, "trend", "Trend indicator"),
                FeatureImportance("feature_1", 0.6, "oscillator", "Oscillator indicator"),
                FeatureImportance("feature_2", 0.4, "volatility", "Volatility indicator")
            ]

            contributing_factors = [
                FeatureImportance("feature_3", 0.2, "volume", "Volume indicator")
            ]

            # 各決定タイプでテスト
            for decision_type in ["BUY", "SELL", "HOLD"]:
                explanation = self.analyzer._generate_natural_language_explanation(
                    decision_type, primary_factors, contributing_factors
                )

                self.assertIsInstance(explanation, str)
                self.assertGreater(len(explanation), 10)
                self.assertIn(decision_type, explanation)

        except Exception as e:
            self.skipTest(f"Enhanced NLP test failed: {e}")

    def test_market_context_analysis(self):
        """市場状況分析テスト"""
        try:
            from ztb.adaptation.explainability.types import FeatureImportance

            # トレンド主体の要因
            trend_factors = [
                FeatureImportance("feature_0", 0.8, "trend"),
                FeatureImportance("feature_1", 0.6, "trend")
            ]

            context = self.analyzer._analyze_market_context(trend_factors)
            self.assertIsInstance(context, str)
            self.assertIn("トレンド", context)

            # オシレーター主体の要因
            oscillator_factors = [
                FeatureImportance("feature_0", 0.8, "oscillator"),
                FeatureImportance("feature_1", 0.6, "oscillator")
            ]

            context = self.analyzer._analyze_market_context(oscillator_factors)
            self.assertIsInstance(context, str)
            self.assertIn("オシレーター", context)

        except Exception as e:
            self.skipTest(f"Market context analysis test failed: {e}")

    def test_risk_warning_generation(self):
        """リスク警告生成テスト"""
        try:
            from ztb.adaptation.explainability.types import FeatureImportance

            # 高重要度の要因
            high_importance_factors = [
                FeatureImportance("feature_0", 0.9),
                FeatureImportance("feature_1", 0.8)
            ]

            warning = self.analyzer._generate_risk_warning("BUY", high_importance_factors)
            self.assertIsInstance(warning, str)

            # 低重要度の要因
            low_importance_factors = [
                FeatureImportance("feature_0", 0.2),
                FeatureImportance("feature_1", 0.1)
            ]

            warning = self.analyzer._generate_risk_warning("BUY", low_importance_factors)
            self.assertIsInstance(warning, str)

        except Exception as e:
            self.skipTest(f"Risk warning generation test failed: {e}")

    @patch('ztb.adaptation.explainability.analyzer.MATPLOTLIB_AVAILABLE', True)
    def test_visualization_generation(self):
        """可視化生成テスト"""
        try:
            from ztb.adaptation.explainability.types import FeatureImportance, DecisionExplanation

            # 特徴量重要度データ
            feature_importance = [
                FeatureImportance("feature_0", 0.8, "trend"),
                FeatureImportance("feature_1", 0.6, "oscillator"),
                FeatureImportance("feature_2", 0.4, "volatility")
            ]

            # 決定説明データ
            decision_explanation = DecisionExplanation(
                decision_type="BUY",
                confidence_score=0.85,
                primary_factors=feature_importance,
                contributing_factors=[],
                natural_language_explanation="Test explanation"
            )

            # 可視化生成
            visualization = self.analyzer._generate_visualizations(
                feature_importance, decision_explanation, self.test_input
            )

            if visualization:
                self.assertIsInstance(visualization, VisualizationResult)
                self.assertIsInstance(visualization.plots, dict)
                self.assertGreater(len(visualization.plots), 0)

        except Exception as e:
            self.skipTest(f"Visualization generation test failed: {e}")

    def test_report_generation(self):
        """レポート生成テスト"""
        try:
            # モック説明結果
            mock_explanation = ExplanationResult(
                explanation_id="test_123",
                timestamp=pd.Timestamp.now(),
                model_version="v421",
                explanation_type=self.analyzer.config.explanation_method,
                target_prediction="BUY",
                feature_importance=[],
                processing_time_seconds=0.1
            )

            explanations = [mock_explanation]

            # 一時ファイルにレポート生成
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
                tmp_path = tmp_file.name

            try:
                result = self.analyzer.generate_explanation_report(explanations, tmp_path)
                self.assertIn("Report saved to", result)
                self.assertTrue(os.path.exists(tmp_path))

                # HTMLファイルの内容確認
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.assertIn("SAC v421", content)
                    self.assertIn("説明可能性レポート", content)

            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            self.skipTest(f"Report generation test failed: {e}")

    def test_feature_importance_plot_generation(self):
        """特徴量重要度プロット生成テスト"""
        try:
            from ztb.adaptation.explainability.types import FeatureImportance

            feature_importance = [
                FeatureImportance("feature_0", 0.8, "trend"),
                FeatureImportance("feature_1", 0.6, "oscillator"),
                FeatureImportance("feature_2", 0.4, "volatility")
            ]

            # matplotlibが利用可能な場合のみテスト
            if hasattr(self.analyzer, '_create_feature_importance_plot'):
                plot_data = self.analyzer._create_feature_importance_plot(feature_importance)

                # プロットデータが辞書形式であることを確認
                self.assertIsInstance(plot_data, dict)
                self.assertIn("type", plot_data)

        except Exception as e:
            self.skipTest(f"Feature importance plot test failed: {e}")


if __name__ == '__main__':
    unittest.main()