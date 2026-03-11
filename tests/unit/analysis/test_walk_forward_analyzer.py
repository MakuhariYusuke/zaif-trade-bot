"""
ウォークフォワード分析器の単体テスト

WalkForwardAnalyzerクラスの機能をテストします。
"""

import numpy as np
import pandas as pd
import pytest

from ztb.analysis.walk_forward_analyzer import (
    ParameterSet,
    WalkForwardAnalyzer,
    WalkForwardWindow,
)


class TestWalkForwardAnalyzer:
    """WalkForwardAnalyzerのテスト"""

    @pytest.fixture
    def sample_market_data(self):
        """ウォークフォワード向けの日次 OHLC データ."""
        dates = pd.date_range("2023-01-01", periods=160, freq="D")
        close = pd.Series(np.linspace(100.0, 140.0, len(dates)), index=dates)
        return pd.DataFrame(
            {
                "open": close * 0.995,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(len(dates), 1000.0),
            },
            index=dates,
        )

    @pytest.fixture
    def mock_strategy_func(self):
        """モック戦略評価関数"""

        def strategy_evaluator(data: pd.DataFrame, params: ParameterSet) -> dict:
            returns = data["close"].pct_change().dropna()
            if len(returns) == 0:
                total_return = 0.0
                volatility = 1.0
                sharpe_ratio = 0.0
            else:
                prod_result = returns.prod()
                total_return = (
                    float(prod_result - 1)
                    if isinstance(prod_result, (int, float))
                    else 0.0
                )
                std_result = returns.std()
                volatility = (
                    float(std_result) if isinstance(std_result, (int, float)) else 1.0
                )
                sharpe_ratio = (
                    float(returns.mean() / volatility * np.sqrt(252))
                    if volatility > 0
                    else 0.0
                )

            return {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "win_rate": 0.6,
                "max_drawdown": 0.15,
                "total_trades": len(returns),
                "trades": [],
            }

        return strategy_evaluator

    def test_initialization(self):
        """初期化テスト"""
        analyzer = WalkForwardAnalyzer()
        assert analyzer.parameter_space is not None
        assert hasattr(analyzer, "create_sliding_windows")
        assert hasattr(analyzer, "optimize_parameters")
        assert hasattr(analyzer, "walk_forward_optimization")

    def test_create_sliding_windows(self, sample_market_data):
        """スライディングウィンドウ作成テスト"""
        analyzer = WalkForwardAnalyzer()

        windows = analyzer.create_sliding_windows(
            sample_market_data,
            train_days=30,
            test_days=10,
            step_days=15,
            min_samples=10,
        )

        assert len(windows) > 0
        assert all(isinstance(w, WalkForwardWindow) for w in windows)

        # 最初のウィンドウの検証
        first_window = windows[0]
        assert first_window.train_days == 30
        assert first_window.test_days == 10
        assert first_window.window_id == 0

        # データ存在チェック
        train_data = sample_market_data.loc[
            first_window.train_start : first_window.train_end
        ]
        test_data = sample_market_data.loc[
            first_window.test_start : first_window.test_end
        ]
        assert len(train_data) >= 10
        assert len(test_data) >= 10

    def test_create_sliding_windows_insufficient_data(self):
        """データ不足時のウィンドウ作成テスト"""
        analyzer = WalkForwardAnalyzer()

        # 短いデータ
        short_data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104]},
            index=pd.date_range("2023-01-01", periods=5, freq="D"),
        )

        # データ不足時はValueErrorが投げられる
        with pytest.raises(ValueError, match="データ期間が不足しています"):
            analyzer.create_sliding_windows(
                short_data, train_days=30, test_days=10, step_days=15, min_samples=10
            )

    def test_optimize_parameters(self, sample_market_data, mock_strategy_func):
        """パラメータ最適化テスト"""
        analyzer = WalkForwardAnalyzer()

        # 最適化実行
        best_params, best_performance = analyzer.optimize_parameters(
            sample_market_data,
            mock_strategy_func,
            parameter_sets=[analyzer.parameter_space.get_conservative_defaults()],
        )

        assert isinstance(best_params, ParameterSet)
        assert isinstance(best_performance, dict)
        assert "sharpe_ratio" in best_performance
        assert "total_return" in best_performance
        assert "win_rate" in best_performance

    def test_optimize_parameters_no_valid_results(self, sample_market_data):
        """有効な結果がない場合のパラメータ最適化テスト"""
        analyzer = WalkForwardAnalyzer()

        # 常に例外を投げる戦略関数
        def failing_strategy(data, params):
            raise ValueError("Test error")

        best_params, best_performance = analyzer.optimize_parameters(
            sample_market_data,
            failing_strategy,
            parameter_sets=[analyzer.parameter_space.get_conservative_defaults()],
        )

        # デフォルト値が返されることを確認
        assert isinstance(best_params, ParameterSet)
        assert isinstance(best_performance, dict)
        assert best_performance["sharpe_ratio"] == 0.0

    def test_walk_forward_optimization(self, sample_market_data, mock_strategy_func):
        """ウォークフォワード最適化テスト"""
        analyzer = WalkForwardAnalyzer()

        results = analyzer.walk_forward_optimization(
            data=sample_market_data,
            strategy_func=mock_strategy_func,
            train_days=30,
            test_days=10,
            step_days=15,
            parameter_sets=[analyzer.parameter_space.get_conservative_defaults()],
            min_samples=10,
        )

        assert len(results) > 0
        assert all(hasattr(r, "best_parameters") for r in results)
        assert all(hasattr(r, "in_sample_performance") for r in results)
        assert all(hasattr(r, "out_of_sample_performance") for r in results)
        assert all(hasattr(r, "window") for r in results)

    def test_evaluate_out_of_sample(self, sample_market_data, mock_strategy_func):
        """アウトオブサンプル評価テスト"""
        analyzer = WalkForwardAnalyzer()

        params = analyzer.parameter_space.get_conservative_defaults()
        performance = analyzer.evaluate_out_of_sample(
            sample_market_data, params, mock_strategy_func
        )

        assert isinstance(performance, dict)
        assert "sharpe_ratio" in performance
        assert "total_return" in performance

    def test_evaluate_out_of_sample_error_handling(self, sample_market_data):
        """アウトオブサンプル評価のエラーハンドリングテスト"""
        analyzer = WalkForwardAnalyzer()

        def failing_strategy(data, params):
            raise RuntimeError("Evaluation failed")

        params = analyzer.parameter_space.get_conservative_defaults()
        performance = analyzer.evaluate_out_of_sample(
            sample_market_data, params, failing_strategy
        )

        # エラー時はデフォルト値が返される
        assert performance["sharpe_ratio"] == 0.0
        assert performance["total_return"] == 0.0
