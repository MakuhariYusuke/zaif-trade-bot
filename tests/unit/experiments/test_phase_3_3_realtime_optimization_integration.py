# test_phase_3_3_realtime_optimization_integration.py

"""
Phase 3-3 リアルタイム最適化統合テスト

リアルタイム最適化モジュールの機能を検証します。
"""

import pytest
import unittest.mock as mock
from datetime import datetime
from ztb.realtime_optimization.realtime_optimizer import RealtimeOptimizer, MarketCondition
from ztb.realtime_optimization.adaptive_learning_system import AdaptiveLearningSystem


class TestRealtimeOptimizationIntegration:
    """リアルタイム最適化統合テスト"""

    def setup_method(self):
        """テスト前準備"""
        self.base_optimizer = mock.Mock()
        self.market_analyzer = mock.Mock()

    def test_realtime_optimizer_initialization(self):
        """RealtimeOptimizer初期化テスト"""
        optimizer = RealtimeOptimizer(
            base_optimizer=self.base_optimizer,
            market_analyzer=self.market_analyzer
        )

        assert optimizer.base_optimizer == self.base_optimizer
        assert optimizer.market_analyzer == self.market_analyzer
        assert not optimizer.is_running
        assert optimizer.optimization_interval == 3600
        assert optimizer.performance_window == 24

    def test_market_condition_creation(self):
        """市場条件作成テスト"""
        condition = MarketCondition(
            volatility=0.5,
            trend_strength=0.3,
            volume=100.0,
            regime="normal",
            timestamp=datetime.now()
        )

        assert condition.volatility == 0.5
        assert condition.trend_strength == 0.3
        assert condition.volume == 100.0
        assert condition.regime == "normal"

    def test_performance_evaluation(self):
        """パフォーマンス評価テスト"""
        optimizer = RealtimeOptimizer(
            base_optimizer=self.base_optimizer,
            market_analyzer=self.market_analyzer
        )

        # パフォーマンススコア追加
        optimizer.performance_scores = [0.6, 0.7, 0.8, 0.5, 0.9]

        score = optimizer._evaluate_performance()
        expected_score = sum([0.6, 0.7, 0.8, 0.5, 0.9]) / 5

        assert score == expected_score

    def test_optimization_decision_making(self):
        """最適化判断テスト"""
        optimizer = RealtimeOptimizer(
            base_optimizer=self.base_optimizer,
            market_analyzer=self.market_analyzer
        )

        # 低パフォーマンスの場合
        market_condition = MarketCondition(
            volatility=0.5, trend_strength=0.3, volume=100.0,
            regime="normal", timestamp=datetime.now()
        )

        should_optimize = optimizer._should_optimize(market_condition, 0.3)  # 低スコア
        assert should_optimize

        # 高パフォーマンスの場合
        should_optimize = optimizer._should_optimize(market_condition, 0.8)  # 高スコア
        assert not should_optimize

    def test_kelly_tolerance_selection(self):
        """Kelly許容度選択テスト"""
        optimizer = RealtimeOptimizer(
            base_optimizer=self.base_optimizer,
            market_analyzer=self.market_analyzer
        )

        # 高ボラティリティ
        high_vol_condition = MarketCondition(
            volatility=0.9, trend_strength=0.3, volume=100.0,
            regime="high_volatility", timestamp=datetime.now()
        )
        tolerance = optimizer._get_kelly_tolerance(high_vol_condition)
        assert tolerance == "quarter"

        # 通常ボラティリティ
        normal_condition = MarketCondition(
            volatility=0.4, trend_strength=0.3, volume=100.0,
            regime="normal", timestamp=datetime.now()
        )
        tolerance = optimizer._get_kelly_tolerance(normal_condition)
        assert tolerance == "full"

    def test_risk_mode_selection(self):
        """リスクモード選択テスト"""
        optimizer = RealtimeOptimizer(
            base_optimizer=self.base_optimizer,
            market_analyzer=self.market_analyzer
        )

        # 高ボラティリティレジーム
        high_vol_condition = MarketCondition(
            volatility=0.8, trend_strength=0.3, volume=100.0,
            regime="high_volatility", timestamp=datetime.now()
        )
        mode = optimizer._get_risk_mode(high_vol_condition)
        assert mode == "conservative"

        # トレンドレジーム
        trending_condition = MarketCondition(
            volatility=0.3, trend_strength=0.7, volume=100.0,
            regime="trending", timestamp=datetime.now()
        )
        mode = optimizer._get_risk_mode(trending_condition)
        assert mode == "moderate"

    @mock.patch('ztb.realtime_optimization.realtime_optimizer.time.sleep')
    def test_optimization_cycle_execution(self, mock_sleep):
        """最適化サイクル実行テスト"""
        # モック設定
        self.market_analyzer.return_value = MarketCondition(
            volatility=0.5, trend_strength=0.3, volume=100.0,
            regime="normal", timestamp=datetime.now()
        )

        optimizer = RealtimeOptimizer(
            base_optimizer=self.base_optimizer,
            market_analyzer=self.market_analyzer,
            optimization_interval=1  # 短い間隔でテスト
        )

        # 最適化が必要な状態にする
        optimizer.performance_scores = [0.3]  # 低パフォーマンス

        # 1サイクル実行
        with mock.patch.object(optimizer, '_run_optimization', return_value={'test': 'params'}):
            optimizer._execute_optimization_cycle()

            # 市場分析器が呼ばれたことを確認
            self.market_analyzer.assert_called_once()

    def test_adaptive_learning_system_initialization(self):
        """AdaptiveLearningSystem初期化テスト"""
        learning_system = AdaptiveLearningSystem()

        assert not learning_system.is_learning
        assert learning_system.learning_interval == 1800
        assert learning_system.experience_window == 1000
        assert learning_system.strategy_evaluation_period == 24

    def test_experience_addition(self):
        """学習経験追加テスト"""
        learning_system = AdaptiveLearningSystem()

        market_condition = {'volatility': 0.5, 'trend': 0.2}
        next_state = {'volatility': 0.6, 'trend': 0.3}

        learning_system.add_experience(
            market_condition=market_condition,
            action_taken='buy',
            reward=100.0,
            next_state=next_state,
            strategy_used='momentum'
        )

        assert len(learning_system.learning_experiences) == 1
        exp = learning_system.learning_experiences[0]
        assert exp.action_taken == 'buy'
        assert exp.reward == 100.0
        assert exp.strategy_used == 'momentum'

    def test_strategy_performance_calculation(self):
        """戦略パフォーマンス計算テスト"""
        learning_system = AdaptiveLearningSystem()

        # テストデータ追加
        experiences = []
        for i in range(10):
            exp = mock.Mock()
            exp.reward = 10.0 if i % 2 == 0 else -5.0  # 勝ち:負け = 5:5
            experiences.append(exp)

        performance = learning_system._calculate_strategy_performance('test_strategy', experiences)

        assert performance.strategy_name == 'test_strategy'
        assert performance.total_trades == 10
        assert performance.win_rate == 0.5  # 50%勝率

    def test_best_strategy_selection(self):
        """最適戦略選択テスト"""
        learning_system = AdaptiveLearningSystem()

        # 戦略パフォーマンス設定
        perf1 = mock.Mock()
        perf1.win_rate = 0.6
        perf1.profit_factor = 2.0
        perf1.max_drawdown = 0.1
        perf1.sharpe_ratio = 1.5

        perf2 = mock.Mock()
        perf2.win_rate = 0.8
        perf2.profit_factor = 1.5
        perf2.max_drawdown = 0.15
        perf2.sharpe_ratio = 1.2

        learning_system.strategy_performances = {
            'strategy1': perf1,
            'strategy2': perf2
        }

        learning_system._select_best_strategy()

        # より良いスコアの戦略が選択されるはず
        assert learning_system.current_best_strategy is not None

    def test_action_recommendation(self):
        """アクション推奨テスト"""
        learning_system = AdaptiveLearningSystem()

        market_condition = {'volatility': 0.3, 'trend': 0.1}
        available_actions = ['buy', 'sell', 'hold']

        # 最適戦略を設定
        learning_system.current_best_strategy = 'conservative'

        action = learning_system.recommend_action(market_condition, available_actions)

        assert action in available_actions

    def test_risk_based_action_selection(self):
        """リスクベースアクション選択テスト"""
        learning_system = AdaptiveLearningSystem()

        # 高ボラティリティ条件
        high_vol_condition = {'volatility': 0.9, 'trend': 0.0}
        action = learning_system._get_risk_based_action(high_vol_condition, ['buy', 'sell', 'hold'])
        assert action == 'hold'

        # 上昇トレンド条件
        uptrend_condition = {'volatility': 0.3, 'trend': 0.5}
        action = learning_system._get_risk_based_action(uptrend_condition, ['buy', 'sell', 'hold'])
        assert action == 'buy'

    @mock.patch('ztb.realtime_optimization.adaptive_learning_system.time.sleep')
    def test_learning_cycle_execution(self, mock_sleep):
        """学習サイクル実行テスト"""
        learning_system = AdaptiveLearningSystem(learning_interval=1)  # 短い間隔

        # 学習経験追加
        learning_system.add_experience(
            market_condition={'volatility': 0.5},
            action_taken='buy',
            reward=10.0,
            next_state={'volatility': 0.6},
            strategy_used='test_strategy'
        )

        # 1サイクル実行
        learning_system._execute_learning_cycle()

        # 戦略パフォーマンスが計算されているはず
        assert len(learning_system.strategy_performances) > 0

    def test_data_cleanup(self):
        """データクリーンアップテスト"""
        from datetime import timedelta

        learning_system = AdaptiveLearningSystem()

        # 古い経験データ追加（8日前）
        old_exp = mock.Mock()
        old_exp.timestamp = datetime.now() - timedelta(days=8)
        learning_system.learning_experiences = [old_exp]

        # 新しい経験データ追加
        learning_system.add_experience(
            market_condition={'volatility': 0.5},
            action_taken='buy',
            reward=10.0,
            next_state={'volatility': 0.6},
            strategy_used='test_strategy'
        )

        # クリーンアップ実行
        learning_system._cleanup_experiences()

        # 古いデータが削除され、新しいデータのみ残る
        assert len(learning_system.learning_experiences) == 1
        assert learning_system.learning_experiences[0].strategy_used == 'test_strategy'


if __name__ == '__main__':
    pytest.main([__file__])