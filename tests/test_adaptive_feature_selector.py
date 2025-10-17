"""
Test Adaptive Feature Selector
適応型特徴量選択のテスト
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from ztb.adaptation.adaptive_feature_selector import (
    AdaptiveFeatureSelector,
    AdaptiveFeatureConfig,
    FeatureSelectionMethod,
    MarketCondition,
    FeatureImportance,
    FeatureSelectionResult
)


class TestAdaptiveFeatureSelector(unittest.TestCase):
    """適応型特徴量選択のテスト"""

    def setUp(self):
        """テストセットアップ"""
        # モックオブジェクトの作成
        self.mock_online_learning = Mock()
        self.mock_evaluation_manager = Mock()

        # テスト設定
        self.config = AdaptiveFeatureConfig(
            min_features=5,
            max_features=20,
            target_features=10,
            adaptation_interval_minutes=5
        )

        # テストデータ生成
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'trend_sma_20': np.random.randn(100),
            'trend_ema_50': np.random.randn(100),
            'oscillator_rsi': np.random.randn(100),
            'oscillator_stoch': np.random.randn(100),
            'volatility_bb_upper': np.random.randn(100),
            'volatility_bb_lower': np.random.randn(100),
            'volume_sma': np.random.randn(100),
            'momentum_macd': np.random.randn(100),
            'support_resistance': np.random.randn(100),
            'ichimoku_tenkan': np.random.randn(100),
            'close': np.random.randn(100),
            'high': np.random.randn(100),
            'low': np.random.randn(100),
            'volume': np.random.randn(100),
            'returns': np.random.randn(100)
        })
        self.test_target = pd.Series(np.random.randn(100))

    def test_initialization(self):
        """初期化テスト"""
        with patch.object(AdaptiveFeatureSelector, '_get_available_features', return_value=list(self.test_data.columns)):
            selector = AdaptiveFeatureSelector(
                self.mock_online_learning,
                self.mock_evaluation_manager,
                self.config
            )

            self.assertIsNotNone(selector)
            self.assertEqual(len(selector.all_features), len(self.test_data.columns))
            self.assertIsInstance(selector.current_market_condition, MarketCondition)

    def test_market_condition_evaluation(self):
        """市場条件評価テスト"""
        with patch.object(AdaptiveFeatureSelector, '_get_available_features', return_value=list(self.test_data.columns)):
            selector = AdaptiveFeatureSelector(
                self.mock_online_learning,
                self.mock_evaluation_manager,
                self.config
            )

            # テストデータで市場条件を評価
            condition = selector._evaluate_market_condition(self.test_data)
            self.assertIsInstance(condition, MarketCondition)

    def test_importance_based_selection(self):
        """重要度ベース選択テスト"""
        with patch.object(AdaptiveFeatureSelector, '_get_available_features', return_value=list(self.test_data.columns)):
            selector = AdaptiveFeatureSelector(
                self.mock_online_learning,
                self.mock_evaluation_manager,
                self.config
            )

            result = selector._importance_based_selection(
                self.test_data,
                self.test_target,
                MarketCondition.TRENDING
            )

            self.assertIsInstance(result, FeatureSelectionResult)
            self.assertEqual(result.selection_method, FeatureSelectionMethod.IMPORTANCE_BASED)
            self.assertLessEqual(len(result.selected_features), self.config.target_features)
            self.assertGreater(len(result.selected_features), 0)

    def test_correlation_based_selection(self):
        """相関ベース選択テスト"""
        with patch.object(AdaptiveFeatureSelector, '_get_available_features', return_value=list(self.test_data.columns)):
            selector = AdaptiveFeatureSelector(
                self.mock_online_learning,
                self.mock_evaluation_manager,
                self.config
            )

            result = selector._correlation_based_selection(
                self.test_data,
                self.test_target,
                MarketCondition.RANGING
            )

            self.assertIsInstance(result, FeatureSelectionResult)
            self.assertEqual(result.selection_method, FeatureSelectionMethod.CORRELATION_BASED)
            self.assertLessEqual(len(result.selected_features), self.config.target_features)

    def test_market_condition_based_selection(self):
        """市場条件ベース選択テスト"""
        with patch.object(AdaptiveFeatureSelector, '_get_available_features', return_value=list(self.test_data.columns)):
            selector = AdaptiveFeatureSelector(
                self.mock_online_learning,
                self.mock_evaluation_manager,
                self.config
            )

            result = selector._market_condition_based_selection(
                self.test_data,
                self.test_target,
                MarketCondition.TRENDING
            )

            self.assertIsInstance(result, FeatureSelectionResult)
            self.assertEqual(result.selection_method, FeatureSelectionMethod.MARKET_CONDITION_BASED)
            self.assertEqual(result.market_condition, MarketCondition.TRENDING)

    def test_combine_selections(self):
        """選択結果統合テスト"""
        with patch.object(AdaptiveFeatureSelector, '_get_available_features', return_value=list(self.test_data.columns)):
            selector = AdaptiveFeatureSelector(
                self.mock_online_learning,
                self.mock_evaluation_manager,
                self.config
            )

            # 複数の選択結果を作成
            selection_results = {
                FeatureSelectionMethod.IMPORTANCE_BASED: FeatureSelectionResult(
                    selected_features=['trend_sma_20', 'oscillator_rsi'],
                    feature_weights={'trend_sma_20': 0.6, 'oscillator_rsi': 0.4},
                    selection_method=FeatureSelectionMethod.IMPORTANCE_BASED,
                    market_condition=MarketCondition.TRENDING,
                    timestamp=datetime.now()
                ),
                FeatureSelectionMethod.CORRELATION_BASED: FeatureSelectionResult(
                    selected_features=['trend_ema_50', 'volatility_bb_upper'],
                    feature_weights={'trend_ema_50': 0.7, 'volatility_bb_upper': 0.3},
                    selection_method=FeatureSelectionMethod.CORRELATION_BASED,
                    market_condition=MarketCondition.TRENDING,
                    timestamp=datetime.now()
                )
            }

            combined = selector._combine_selections(
                selection_results,
                MarketCondition.TRENDING,
                datetime.now()
            )

            self.assertIsInstance(combined, FeatureSelectionResult)
            self.assertGreater(len(combined.selected_features), 0)
            self.assertAlmostEqual(sum(combined.feature_weights.values()), 1.0, places=5)

    def test_adapt_features(self):
        """特徴量適応テスト"""
        with patch.object(AdaptiveFeatureSelector, '_get_available_features', return_value=list(self.test_data.columns)):
            selector = AdaptiveFeatureSelector(
                self.mock_online_learning,
                self.mock_evaluation_manager,
                self.config
            )

            result = selector.adapt_features(self.test_data, self.test_target)

            self.assertIsInstance(result, FeatureSelectionResult)
            self.assertGreater(len(result.selected_features), 0)
            self.assertGreater(len(result.feature_weights), 0)

    def test_get_market_condition_weights(self):
        """市場条件重み取得テスト"""
        with patch.object(AdaptiveFeatureSelector, '_get_available_features', return_value=list(self.test_data.columns)):
            selector = AdaptiveFeatureSelector(
                self.mock_online_learning,
                self.mock_evaluation_manager,
                self.config
            )

            # 各市場条件で重みを取得
            for condition in MarketCondition:
                weights = selector._get_market_condition_weights(condition)
                self.assertIsInstance(weights, dict)
                self.assertIn('trend', weights)
                self.assertIn('oscillator', weights)
                self.assertIn('volatility', weights)
                self.assertIn('volume', weights)

    def test_selection_history(self):
        """選択履歴テスト"""
        with patch.object(AdaptiveFeatureSelector, '_get_available_features', return_value=list(self.test_data.columns)):
            selector = AdaptiveFeatureSelector(
                self.mock_online_learning,
                self.mock_evaluation_manager,
                self.config
            )

            # 適応を実行
            selector.adapt_features(self.test_data, self.test_target)

            # 履歴を取得
            history = selector.get_selection_history(hours=1)
            self.assertIsInstance(history, list)
            self.assertGreaterEqual(len(history), 1)

    def test_feature_importance_stats(self):
        """特徴量重要度統計テスト"""
        with patch.object(AdaptiveFeatureSelector, '_get_available_features', return_value=list(self.test_data.columns)):
            selector = AdaptiveFeatureSelector(
                self.mock_online_learning,
                self.mock_evaluation_manager,
                self.config
            )

            # 適応を実行して履歴を作成
            selector.adapt_features(self.test_data, self.test_target)

            # 統計を取得
            stats = selector.get_feature_importance_stats('trend_sma_20', hours=1)
            self.assertIsInstance(stats, dict)

    def test_callback_system(self):
        """コールバックシステムテスト"""
        with patch.object(AdaptiveFeatureSelector, '_get_available_features', return_value=list(self.test_data.columns)):
            selector = AdaptiveFeatureSelector(
                self.mock_online_learning,
                self.mock_evaluation_manager,
                self.config
            )

            # コールバックを追加
            callback_called = False
            def test_callback(result):
                nonlocal callback_called
                callback_called = True
                self.assertIsInstance(result, FeatureSelectionResult)

            selector.add_selection_callback(test_callback)

            # 適応を実行
            selector.adapt_features(self.test_data, self.test_target)

            # コールバックが呼ばれたことを確認
            self.assertTrue(callback_called)

    def test_error_handling(self):
        """エラーハンドリングテスト"""
        with patch.object(AdaptiveFeatureSelector, '_get_available_features', return_value=list(self.test_data.columns)):
            selector = AdaptiveFeatureSelector(
                self.mock_online_learning,
                self.mock_evaluation_manager,
                self.config
            )

            # 空のデータで適応を実行
            empty_data = pd.DataFrame()
            empty_target = pd.Series(dtype=float)

            result = selector.adapt_features(empty_data, empty_target)

            # エラーハンドリングにより結果が返されることを確認
            self.assertIsInstance(result, FeatureSelectionResult)


if __name__ == '__main__':
    unittest.main()