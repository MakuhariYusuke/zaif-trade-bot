"""
Test Dynamic Hyperparameter Adaptation System
動的ハイパーパラメータ適応システムテスト
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from ztb.adaptation.dynamic_hyperparameter_adapter import (
    DynamicHyperparameterAdapter,
    HyperparameterConfig,
    HyperparameterType,
    AdaptationStrategy,
    AdaptationResult,
    HyperparameterAdaptation
)
from ztb.adaptation.market_aware_hyperparameter_manager import (
    MarketAwareHyperparameterManager,
    MarketAwareConfig,
    MarketCondition
)
from ztb.adaptation.hyperparameter_adaptation_system import HyperparameterAdaptationSystem


class TestDynamicHyperparameterAdapter(unittest.TestCase):
    """動的ハイパーパラメータアダプターテスト"""

    def setUp(self):
        """テストセットアップ"""
        self.mock_online_learning = Mock()
        self.mock_evaluation_manager = Mock()

        # モックの設定
        self.mock_evaluation_manager.get_current_performance.return_value = 0.7

        self.config = HyperparameterConfig()
        self.adapter = DynamicHyperparameterAdapter(
            self.mock_online_learning,
            self.mock_evaluation_manager,
            self.config
        )

    def test_initialization(self):
        """初期化テスト"""
        self.assertIsNotNone(self.adapter.current_parameters)
        self.assertEqual(len(self.adapter.performance_history), 1)
        self.assertFalse(self.adapter.is_active)

    def test_evaluate_market_conditions(self):
        """市場条件評価テスト"""
        # テストデータ作成
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1min')
        prices = np.random.normal(100, 1, 100)
        market_data = pd.DataFrame({
            'close': prices,
            'volume': np.random.normal(1000, 100, 100)
        }, index=dates)

        conditions = self.adapter._evaluate_market_conditions(market_data)

        self.assertIn('volatility', conditions)
        self.assertIn('trend_strength', conditions)
        self.assertIn('market_state', conditions)
        self.assertIsInstance(conditions['volatility'], float)

    def test_performance_based_adaptation(self):
        """パフォーマンスベース適応テスト"""
        market_conditions = {
            'volatility': 0.02,
            'trend_strength': 0.05,
            'market_state': 1.0
        }

        # パフォーマンス履歴を追加
        base_time = datetime.now()
        for i in range(10):
            self.adapter.performance_history.append(
                (base_time + timedelta(minutes=i), 0.5 + i * 0.01)
            )

        adaptations = self.adapter._performance_based_adaptation(market_conditions)

        # 適応結果を検証
        self.assertIsInstance(adaptations, list)
        for adaptation in adaptations:
            self.assertIsInstance(adaptation, HyperparameterAdaptation)
            self.assertIn(adaptation.parameter_type, self.config.enabled_parameters)

    def test_volatility_based_adaptation(self):
        """ボラティリティベース適応テスト"""
        # 高ボラティリティ条件
        high_vol_conditions = {
            'volatility': 0.05,
            'trend_strength': 0.05,
            'market_state': 2.0
        }

        adaptations = self.adapter._volatility_based_adaptation(high_vol_conditions)

        self.assertIsInstance(adaptations, list)
        # 高ボラティリティ時は適応が発生するはず
        if adaptations:
            for adaptation in adaptations:
                self.assertIsInstance(adaptation, HyperparameterAdaptation)

        # 低ボラティリティ条件
        low_vol_conditions = {
            'volatility': 0.005,
            'trend_strength': 0.05,
            'market_state': 0.0
        }

        adaptations_low = self.adapter._volatility_based_adaptation(low_vol_conditions)

        self.assertIsInstance(adaptations_low, list)

    def test_combine_adaptations(self):
        """適応統合テスト"""
        adaptations = [
            HyperparameterAdaptation(
                parameter_type=HyperparameterType.LEARNING_RATE,
                old_value=1e-3,
                new_value=1.1e-3,
                adaptation_strategy=AdaptationStrategy.PERFORMANCE_BASED,
                performance_score=0.02,
                volatility_score=0.02,
                timestamp=datetime.now()
            ),
            HyperparameterAdaptation(
                parameter_type=HyperparameterType.LEARNING_RATE,
                old_value=1e-3,
                new_value=0.9e-3,
                adaptation_strategy=AdaptationStrategy.VOLATILITY_BASED,
                performance_score=0.0,
                volatility_score=0.05,
                timestamp=datetime.now()
            )
        ]

        combined = self.adapter._combine_adaptations(adaptations)

        self.assertIsInstance(combined, list)
        self.assertEqual(len(combined), 1)  # 同じパラメータタイプは統合される
        self.assertEqual(combined[0].parameter_type, HyperparameterType.LEARNING_RATE)

    def test_apply_adaptations(self):
        """適応適用テスト"""
        adaptations = [
            HyperparameterAdaptation(
                parameter_type=HyperparameterType.LEARNING_RATE,
                old_value=self.adapter.current_parameters[HyperparameterType.LEARNING_RATE],
                new_value=1.5e-3,
                adaptation_strategy=AdaptationStrategy.PERFORMANCE_BASED,
                performance_score=0.02,
                volatility_score=0.02,
                timestamp=datetime.now()
            )
        ]

        applied = self.adapter._apply_adaptations(adaptations)

        self.assertEqual(len(applied), 1)
        self.assertEqual(
            self.adapter.current_parameters[HyperparameterType.LEARNING_RATE],
            1.5e-3
        )

    def test_adaptation_safety(self):
        """適応安全性テスト"""
        # 安全な適応
        safe_adaptation = HyperparameterAdaptation(
            parameter_type=HyperparameterType.LEARNING_RATE,
            old_value=1e-3,
            new_value=1.1e-3,  # 10%変更
            adaptation_strategy=AdaptationStrategy.PERFORMANCE_BASED,
            performance_score=0.02,
            volatility_score=0.02,
            timestamp=datetime.now()
        )

        self.assertTrue(self.adapter._is_adaptation_safe(safe_adaptation))

        # 危険な適応（範囲外）
        unsafe_adaptation = HyperparameterAdaptation(
            parameter_type=HyperparameterType.LEARNING_RATE,
            old_value=1e-3,
            new_value=1e-1,  # 範囲外
            adaptation_strategy=AdaptationStrategy.PERFORMANCE_BASED,
            performance_score=0.02,
            volatility_score=0.02,
            timestamp=datetime.now()
        )

        self.assertFalse(self.adapter._is_adaptation_safe(unsafe_adaptation))

    def test_adapt_hyperparameters(self):
        """ハイパーパラメータ適応テスト"""
        result = self.adapter.adapt_hyperparameters()

        self.assertIsInstance(result, AdaptationResult)
        self.assertIsInstance(result.adaptations, list)
        self.assertIsInstance(result.overall_performance_improvement, float)
        self.assertIsInstance(result.adaptation_confidence, float)


class TestMarketAwareHyperparameterManager(unittest.TestCase):
    """市場対応ハイパーパラメータマネージャーテスト"""

    def setUp(self):
        """テストセットアップ"""
        self.mock_online_learning = Mock()
        self.mock_evaluation_manager = Mock()
        self.mock_market_detector = Mock()

        # モックの設定
        self.mock_evaluation_manager.get_current_performance.return_value = 0.7
        self.mock_market_detector.detect_regime.return_value = {'regime': 'neutral'}

        self.config = MarketAwareConfig()
        self.manager = MarketAwareHyperparameterManager(
            self.mock_online_learning,
            self.mock_evaluation_manager,
            self.mock_market_detector,
            self.config
        )

    def test_initialization(self):
        """初期化テスト"""
        self.assertIsNotNone(self.manager.hyperparameter_adapter)
        self.assertIsNotNone(self.manager.performance_predictors)
        self.assertFalse(self.manager.is_active)

    def test_get_market_condition(self):
        """市場条件取得テスト"""
        # テストデータ作成
        dates = pd.date_range(start='2023-01-01', periods=50, freq='1min')
        prices = np.random.normal(100, 2, 50)
        market_data = pd.DataFrame({
            'close': prices,
            'volume': np.random.normal(1000, 100, 50)
        }, index=dates)

        condition = self.manager._get_current_market_condition(market_data)

        self.assertIsInstance(condition, MarketCondition)
        self.assertIsInstance(condition.volatility, float)
        self.assertIsInstance(condition.trend_strength, float)

    def test_predict_optimal_parameters(self):
        """最適パラメータ予測テスト"""
        market_condition = MarketCondition(
            timestamp=datetime.now(),
            volatility=0.02,
            trend_strength=0.05,
            market_regime='neutral',
            volume_profile=1000.0,
            liquidity_score=0.8
        )

        # 学習データを追加（実際の予測には必要）
        features = market_condition.to_features()
        for param_type in HyperparameterType:
            self.manager.training_data[param_type].append((features, 0.7))

        # モデルを学習
        self.manager._retrain_prediction_models()

        predictions = self.manager._predict_optimal_parameters(market_condition)

        self.assertIsInstance(predictions, dict)
        # 予測が成功するかは学習データによる

    def test_select_adaptation_strategies(self):
        """適応戦略選択テスト"""
        market_condition = MarketCondition(
            timestamp=datetime.now(),
            volatility=0.05,  # 高ボラティリティ
            trend_strength=0.15,  # 強いトレンド
            market_regime='volatile',
            volume_profile=1000.0,
            liquidity_score=0.8
        )

        predictions = {}  # 空の予測

        strategies = self.manager._select_adaptation_strategies(market_condition, predictions)

        self.assertIsInstance(strategies, list)
        self.assertIn(AdaptationStrategy.VOLATILITY_BASED, strategies)

    def test_adapt_hyperparameters_market_aware(self):
        """市場対応適応テスト"""
        result = self.manager.adapt_hyperparameters_market_aware()

        self.assertIsInstance(result, AdaptationResult)
        self.assertIsInstance(result.adaptations, list)

    def test_get_adaptation_recommendations(self):
        """適応推奨取得テスト"""
        recommendations = self.manager.get_adaptation_recommendations()

        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

    def test_get_performance_predictions(self):
        """パフォーマンス予測取得テスト"""
        predictions = self.manager.get_performance_predictions()

        self.assertIsInstance(predictions, dict)
        # 市場条件が取得できない場合は空の辞書

    def test_get_adaptation_statistics(self):
        """適応統計取得テスト"""
        stats = self.manager.get_adaptation_statistics()

        self.assertIsInstance(stats, dict)
        self.assertIn('total_adaptations', stats)
        self.assertIn('market_conditions_count', stats)


class TestIntegration(unittest.TestCase):
    """統合テスト"""

    def test_full_adaptation_cycle(self):
        """完全適応サイクルテスト"""
        # モック設定
        mock_online_learning = Mock()
        mock_evaluation_manager = Mock()
        mock_evaluation_manager.get_current_performance.return_value = 0.6

        # アダプター作成
        adapter = DynamicHyperparameterAdapter(
            mock_online_learning,
            mock_evaluation_manager
        )

        # マネージャー作成
        manager = MarketAwareHyperparameterManager(
            mock_online_learning,
            mock_evaluation_manager
        )

        # 適応実行
        result = manager.adapt_hyperparameters_market_aware()

        self.assertIsInstance(result, AdaptationResult)

        # 統計取得
        stats = manager.get_adaptation_statistics()
        self.assertIsInstance(stats, dict)

        # 推奨取得
        recommendations = manager.get_adaptation_recommendations()
        self.assertIsInstance(recommendations, list)


class TestBacktestIntegration(unittest.TestCase):
    """バックテスト統合テスト"""

    def setUp(self):
        """テストセットアップ"""
        self.mock_online_learning = Mock()
        self.mock_evaluation_manager = Mock()
        self.mock_evaluation_manager.get_current_performance.return_value = 0.7

        # 適応システムの初期化
        self.adaptation_system = HyperparameterAdaptationSystem(
            self.mock_online_learning,
            self.mock_evaluation_manager
        )

    def test_backtest_engine_with_adaptation(self):
        """適応機能付きバックテストエンジンテスト"""
        try:
            from ztb.trading.backtest.runner import BacktestEngine
            from ztb.trading.backtest.adapters import BuyAndHoldAdapter

            # 適応機能付きエンジンの作成
            engine = BacktestEngine(
                initial_capital=10000.0,
                enable_adaptation=True,
                adaptation_config={
                    'hyperparameter_config': {
                        'adaptation_interval_minutes': 5,
                        'safety_margin': 0.1
                    }
                }
            )

            # 適応システムが初期化されていることを確認
            self.assertTrue(engine.enable_adaptation)
            self.assertIsNotNone(engine.adaptation_system)

            # テストデータ生成
            data = engine.load_data("test")

            # 戦略アダプター作成
            strategy = BuyAndHoldAdapter()

            # バックテスト実行
            equity_curve, orders, adaptation_summary = engine.run_backtest(strategy, data)

            # 結果検証
            self.assertIsInstance(equity_curve, pd.Series)
            self.assertIsInstance(orders, pd.DataFrame)
            self.assertIsInstance(adaptation_summary, (dict, type(None)))

            if adaptation_summary:
                self.assertIn('total_adaptations', adaptation_summary)
                self.assertIn('final_hyperparameters', adaptation_summary)

        except ImportError:
            self.skipTest("Backtest components not available")

    def test_strategy_adapter_hyperparameter_update(self):
        """戦略アダプターのハイパーパラメータ更新テスト"""
        try:
            from ztb.trading.backtest.adapters import SMACrossoverAdapter

            # SMAアダプター作成
            adapter = SMACrossoverAdapter(fast_period=10, slow_period=20)

            # 初期パラメータ確認
            self.assertEqual(adapter.fast_period, 10)
            self.assertEqual(adapter.slow_period, 20)

            # ハイパーパラメータ更新
            new_params = {
                'fast_period': 15,
                'slow_period': 30
            }
            adapter.update_hyperparameters(new_params)

            # 更新後のパラメータ確認
            self.assertEqual(adapter.fast_period, 15)
            self.assertEqual(adapter.slow_period, 30)

        except ImportError:
            self.skipTest("Strategy adapters not available")

    def test_adaptation_during_backtest_simulation(self):
        """バックテスト中の適応シミュレーションテスト"""
        try:
            from ztb.trading.backtest.runner import BacktestEngine, MockOnlineLearningPipeline, MockEvaluationManager
            from ztb.trading.backtest.adapters import BuyAndHoldAdapter

            # モックコンポーネント
            mock_online = MockOnlineLearningPipeline()
            mock_eval = MockEvaluationManager()

            # 適応システム作成
            adaptation_system = HyperparameterAdaptationSystem(mock_online, mock_eval)
            self.assertTrue(adaptation_system.initialize())

            # テスト市場データ作成
            dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
            prices = np.random.normal(30000, 1000, 100)
            market_data = pd.DataFrame({
                'open': prices * 0.99,
                'high': prices * 1.01,
                'low': prices * 0.98,
                'close': prices,
                'volume': np.random.uniform(1000, 10000, 100)
            }, index=dates)

            # 適応実行
            result = adaptation_system.adapt_hyperparameters(market_data)

            # 結果検証
            self.assertIn('success', result)
            self.assertIn('adaptations', result)
            self.assertIn('performance_improvement', result)

        except ImportError:
            self.skipTest("Adaptation system components not available")


if __name__ == '__main__':
    unittest.main()