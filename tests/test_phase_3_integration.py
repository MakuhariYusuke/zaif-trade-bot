#!/usr/bin/env python3
"""
Phase 3 Integration Tests
Risk Management & Statistical Validation

Phase 3コンポーネントの統合テスト。
EnhancedRiskManager, StatisticalValidator, IntegratedBacktestRunnerの連携を確認。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.metrics.statistical_validator import StatisticalValidator
from ztb.risk.enhanced_risk_manager import EnhancedRiskManager
from ztb.trading.backtest.integrated_backtest_runner import IntegratedBacktestRunner
from ztb.trading.signal.multi_timeframe_analyzer import Timeframe


class TestPhase3Integration:
    """Phase 3統合テスト"""

    def setup_method(self):
        """テスト前準備"""
        # 設定
        self.risk_config = {
            "enabled": True,
            "multi_timeframe_enabled": True,
            "convergence_risk_weight": 0.3,
            "timeframe_risk_weights": {
                Timeframe.M1: 0.2,
                Timeframe.M5: 0.3,
                Timeframe.M15: 0.5,
            },
        }

        self.validation_config = {
            "alpha_level": 0.05,
            "confidence_level": 0.95,
            "bootstrap_samples": 1000,  # テスト用に少なく
            "min_sample_size": 30,
        }

        self.backtest_config = {
            "enable_risk_management": True,
            "enable_statistical_validation": True,
            "multi_timeframe_enabled": True,
            "n_iterations": 5,  # テスト用に少なく
            "confidence_level": 0.95,
        }

        # サンプルデータ生成
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=1000, freq="1min")
        self.market_data = pd.DataFrame(
            {
                "open": np.random.randn(1000) + 100,
                "high": np.random.randn(1000) + 102,
                "low": np.random.randn(1000) + 98,
                "close": np.random.randn(1000) + 100,
                "volume": np.random.randint(1000, 10000, 1000),
            },
            index=dates,
        )

        # リターンデータ生成
        self.sample_returns = np.random.normal(0.001, 0.02, 500).tolist()

    def test_enhanced_risk_manager_initialization(self):
        """EnhancedRiskManager初期化テスト"""
        risk_manager = EnhancedRiskManager(self.risk_config)

        assert risk_manager.enabled == True
        assert risk_manager.multi_timeframe_enabled == True
        assert risk_manager.multi_timeframe_analyzer is not None
        assert risk_manager.convergence_calculator is not None

        # ダッシュボード取得テスト
        dashboard = risk_manager.get_risk_dashboard()
        assert "multi_timeframe_status" in dashboard
        assert dashboard["multi_timeframe_status"]["enabled"] == True

    def test_enhanced_risk_adjustment(self):
        """拡張リスク調整テスト"""
        risk_manager = EnhancedRiskManager(self.risk_config)

        # テストデータ投入
        for i in range(0, len(self.market_data), 5):
            price = self.market_data.iloc[i]["close"]
            volume = self.market_data.iloc[i]["volume"]
            risk_manager.multi_timeframe_analyzer.update_timeframe_data(
                Timeframe.M5, price, volume
            )

        for i in range(0, len(self.market_data), 15):
            price = self.market_data.iloc[i]["close"]
            volume = self.market_data.iloc[i]["volume"]
            risk_manager.multi_timeframe_analyzer.update_timeframe_data(
                Timeframe.M15, price, volume
            )

        # リスク調整実行
        result = risk_manager.calculate_enhanced_risk_adjusted_position(
            base_position=1.0,
            current_price=100.0,
            portfolio_value=10000.0,
            atr=1.0,
            df=self.market_data,
        )

        assert "adjusted_position" in result
        assert "multi_timeframe_adjusted" in result
        assert "convergence_score" in result
        assert "integrated_risk_multiplier" in result

    def test_statistical_validator_initialization(self):
        """StatisticalValidator初期化テスト"""
        validator = StatisticalValidator(self.validation_config)

        assert validator.alpha_level == 0.05
        assert validator.confidence_level == 0.95
        assert validator.min_sample_size == 30

    def test_performance_metrics_validation(self):
        """性能指標検証テスト"""
        validator = StatisticalValidator(self.validation_config)

        result = validator.validate_performance_metrics(self.sample_returns)

        assert result["valid"] == True
        assert "basic_stats" in result
        assert "sharpe_ratio" in result
        assert "stability_analysis" in result

        # Sharpe ratio確認
        sharpe = result["sharpe_ratio"]
        assert "value" in sharpe
        assert "confidence_interval" in sharpe
        assert len(sharpe["confidence_interval"]) == 2

    def test_multiple_strategies_validation(self):
        """複数戦略検証テスト"""
        validator = StatisticalValidator(self.validation_config)

        strategies = {
            "strategy_1": self.sample_returns,
            "strategy_2": [r * 1.1 for r in self.sample_returns],  # 少し良い戦略
            "strategy_3": [r * 0.9 for r in self.sample_returns],  # 少し悪い戦略
        }

        result = validator.validate_multiple_strategies(strategies)

        assert result["valid"] == True
        assert "individual_results" in result
        assert "strategy_comparison" in result
        assert len(result["individual_results"]) == 3

    def test_signal_quality_validation(self):
        """シグナル品質検証テスト"""
        validator = StatisticalValidator(self.validation_config)

        # 予測値と実際のリターンを生成
        predictions = np.random.normal(0, 1, 200).tolist()
        actual_returns = [p * 0.5 + np.random.normal(0, 0.01) for p in predictions]

        result = validator.validate_signal_quality(predictions, actual_returns)

        assert result["valid"] == True
        assert "correlation" in result
        assert "prediction_accuracy" in result

        correlation = result["correlation"]
        assert "value" in correlation
        assert "p_value" in correlation
        assert "confidence_interval" in correlation

    def test_integrated_backtest_runner_initialization(self):
        """IntegratedBacktestRunner初期化テスト"""
        runner = IntegratedBacktestRunner(self.backtest_config)

        assert runner.enable_risk_management == True
        assert runner.enable_statistical_validation == True
        assert runner.n_iterations == 5

    def test_integrated_backtest_execution(self):
        """統合バックテスト実行テスト"""
        runner = IntegratedBacktestRunner(self.backtest_config)

        # シンプルな戦略関数
        def simple_strategy(data_point, portfolio_value):
            # ランダムな取引シグナル
            signal = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
            position_size = 0.1 if signal != 0 else 0.0
            return {
                "signal": signal,
                "position_size": position_size,
                "price": data_point["close"],
            }

        # テスト用にデータを小さく
        test_data = self.market_data.head(100)

        result = runner.run_integrated_backtest(
            simple_strategy, test_data, initial_capital=1000.0
        )

        assert result["success"] == True
        assert "summary" in result
        assert "iterations" in result
        assert len(result["iterations"]) == 5

        summary = result["summary"]
        assert "total_iterations" in summary
        assert "successful_iterations" in summary
        assert "avg_final_portfolio" in summary

    def test_full_integration_workflow(self):
        """完全統合ワークフローテスト"""
        # 1. EnhancedRiskManager初期化
        risk_manager = EnhancedRiskManager(self.risk_config)

        # 2. StatisticalValidator初期化
        validator = StatisticalValidator(self.validation_config)

        # 3. IntegratedBacktestRunner初期化
        runner = IntegratedBacktestRunner(self.backtest_config)

        # 4. 統合バックテスト実行
        def test_strategy(data_point, portfolio_value):
            signal = np.random.choice([-1, 0, 1])
            return {
                "signal": signal,
                "position_size": 0.05,
                "price": data_point["close"],
            }

        test_data = self.market_data.head(50)  # 小さくしてテスト
        backtest_result = runner.run_integrated_backtest(
            test_strategy, test_data, initial_capital=1000.0
        )

        # 5. 結果検証
        assert backtest_result["success"] == True

        # 統計的検証実行
        if backtest_result.get("iterations"):
            successful_iterations = [
                it
                for it in backtest_result["iterations"]
                if it.get("success") and "portfolio_values" in it
            ]

            if successful_iterations:
                # 最初のイテレーションの統計検証
                first_iteration = successful_iterations[0]
                portfolio_values = first_iteration.get("portfolio_values", [])

                if len(portfolio_values) > 1:
                    returns = []
                    for i in range(1, len(portfolio_values)):
                        ret = (
                            portfolio_values[i] - portfolio_values[i - 1]
                        ) / portfolio_values[i - 1]
                        returns.append(ret)

                    if returns:
                        validation_result = validator.validate_performance_metrics(
                            returns
                        )
                        assert validation_result["valid"] == True

        # レポート生成テスト
        report = runner.generate_integrated_report(backtest_result)
        assert "Integrated Backtest Report" in report
        assert "Summary" in report
        assert "Performance Metrics" in report


if __name__ == "__main__":
    # 直接実行時のテスト
    test_instance = TestPhase3Integration()
    test_instance.setup_method()

    print("Running Phase 3 Integration Tests...")

    try:
        test_instance.test_enhanced_risk_manager_initialization()
        print("✓ EnhancedRiskManager initialization test passed")

        test_instance.test_statistical_validator_initialization()
        print("✓ StatisticalValidator initialization test passed")

        test_instance.test_performance_metrics_validation()
        print("✓ Performance metrics validation test passed")

        test_instance.test_integrated_backtest_runner_initialization()
        print("✓ IntegratedBacktestRunner initialization test passed")

        test_instance.test_full_integration_workflow()
        print("✓ Full integration workflow test passed")

        print("\n🎉 All Phase 3 integration tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
