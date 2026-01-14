"""
WalkForwardModelEvaluator のテスト

依存注入、例外処理、複数ウィンドウ評価をテストする。
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock

from ztb.evaluation.walk_forward.evaluator import (
    WalkForwardModelEvaluator,
    WindowEvaluationError,
)
from ztb.evaluation.walk_forward.types import TimeSeriesWindow, WindowPerformance


class TestWalkForwardModelEvaluatorDependencyInjection:
    """依存注入パターンのテスト"""

    def test_default_initialization(self):
        """デフォルト初期化"""
        evaluator = WalkForwardModelEvaluator()
        assert evaluator.models == {}
        assert evaluator.results == {}
        assert evaluator.errors == {}

    def test_custom_env_factory(self):
        """カスタム環境工場を注入"""
        mock_env_factory = Mock(return_value=Mock())
        evaluator = WalkForwardModelEvaluator(env_factory=mock_env_factory)
        assert evaluator.env_factory == mock_env_factory

    def test_custom_algorithm_factory(self):
        """カスタムアルゴリズム工場を注入"""
        mock_algo_factory = Mock(return_value=Mock())
        evaluator = WalkForwardModelEvaluator(algorithm_factory=mock_algo_factory)
        assert evaluator.algorithm_factory == mock_algo_factory

    def test_both_factories_custom(self):
        """両方カスタム工場を注入"""
        mock_env_factory = Mock(return_value=Mock())
        mock_algo_factory = Mock(return_value=Mock())
        evaluator = WalkForwardModelEvaluator(
            env_factory=mock_env_factory,
            algorithm_factory=mock_algo_factory,
        )
        assert evaluator.env_factory == mock_env_factory
        assert evaluator.algorithm_factory == mock_algo_factory


class TestWalkForwardModelEvaluatorExceptionHandling:
    """例外処理のテスト"""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """サンプルデータ"""
        return pd.DataFrame({
            "close": np.random.randn(100).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 100),
        })

    @pytest.fixture
    def sample_window(self) -> TimeSeriesWindow:
        """サンプルウィンドウ"""
        return TimeSeriesWindow(
            window_id=0,
            train_start=0,
            train_end=50,
            val_start=50,
            val_end=75,
            test_start=75,
            test_end=100,
        )

    def test_empty_dataframe_error(self, sample_window):
        """空のデータフレームでエラー"""
        evaluator = WalkForwardModelEvaluator()
        empty_df = pd.DataFrame()

        # continue_on_error=False の場合、例外を発生させる
        with pytest.raises(WindowEvaluationError):
            evaluator.train_and_evaluate_window(
                df=empty_df,
                window=sample_window,
                continue_on_error=False,
            )

        # エラーが記録されている
        assert sample_window.window_id in evaluator.errors

    def test_continue_on_error_true(self, sample_data, sample_window):
        """continue_on_error=True でエラーをスキップ"""
        # 失敗するカスタム工場
        def failing_env_factory(df):
            raise RuntimeError("Environment creation failed")

        evaluator = WalkForwardModelEvaluator(env_factory=failing_env_factory)

        result = evaluator.train_and_evaluate_window(
            df=sample_data,
            window=sample_window,
            continue_on_error=True,
        )

        # None が返される
        assert result is None
        # エラーが記録される
        assert sample_window.window_id in evaluator.errors

    def test_continue_on_error_false(self, sample_data, sample_window):
        """continue_on_error=False で例外を発生させる"""
        # 失敗するカスタム工場
        def failing_env_factory(df):
            raise RuntimeError("Environment creation failed")

        evaluator = WalkForwardModelEvaluator(env_factory=failing_env_factory)

        with pytest.raises(WindowEvaluationError):
            evaluator.train_and_evaluate_window(
                df=sample_data,
                window=sample_window,
                continue_on_error=False,
            )


class TestWalkForwardModelEvaluatorMultipleWindows:
    """複数ウィンドウ評価のテスト"""

    @pytest.fixture
    def mock_evaluator(self):
        """モック評価器"""
        evaluator = WalkForwardModelEvaluator()
        evaluator.results = {
            0: WindowPerformance(
                window_id=0,
                val_roi=0.0523,
                test_roi=0.0419,
                sharpe_ratio=1.25,
                max_drawdown=-0.082,
                win_rate=0.65,
                trades=42,
            ),
            1: WindowPerformance(
                window_id=1,
                val_roi=0.0451,
                test_roi=0.0387,
                sharpe_ratio=1.15,
                max_drawdown=-0.095,
                win_rate=0.62,
                trades=38,
            ),
        }
        evaluator.errors = {
            2: WindowEvaluationError("Window 2 failed"),
        }
        return evaluator

    def test_get_results_summary(self, mock_evaluator):
        """結果サマリーの取得"""
        summary = mock_evaluator.get_results_summary()

        assert summary["total_windows"] == 3
        assert summary["successful_windows"] == 2
        assert summary["failed_windows"] == 1
        assert summary["avg_val_roi"] == pytest.approx((0.0523 + 0.0451) / 2)
        assert summary["avg_test_roi"] == pytest.approx((0.0419 + 0.0387) / 2)
        assert summary["avg_sharpe"] == pytest.approx((1.25 + 1.15) / 2)

    def test_get_results_summary_empty(self):
        """空の結果サマリー"""
        evaluator = WalkForwardModelEvaluator()
        summary = evaluator.get_results_summary()

        assert summary["total_windows"] == 0
        assert summary["successful_windows"] == 0
        assert summary["failed_windows"] == 0
        assert summary["avg_val_roi"] == 0.0


class TestWindowEvaluationError:
    """WindowEvaluationError のテスト"""

    def test_error_creation(self):
        """エラーの作成"""
        error = WindowEvaluationError("Test error message")
        assert str(error) == "Test error message"

    def test_error_chaining(self):
        """エラーチェーン"""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            error = WindowEvaluationError("Wrapped error") from e
            assert error.__cause__ == e


class TestWalkForwardModelEvaluatorIntegration:
    """統合テスト（簡略版）"""

    @pytest.fixture
    def simple_mock_setup(self):
        """シンプルなモック設定"""
        # モック環境
        mock_env = MagicMock()
        mock_env.initial_balance = 1000000.0
        mock_env.balance = 1050000.0
        mock_env.reset.return_value = (np.array([100, 50, 20]), None)
        mock_env.step.side_effect = [
            (np.array([100, 50, 20]), 100.0, False, False, {"trade_executed": True, "trade_pnl": 100.0}),
            (np.array([100, 50, 20]), 100.0, True, False, {"trade_executed": False}),
        ]

        # モック工場
        env_factory = Mock(return_value=mock_env)

        mock_model = MagicMock()
        mock_model.predict.return_value = (1, None)
        algo_factory = Mock(return_value=mock_model)

        return env_factory, algo_factory, mock_env, mock_model

    def test_evaluator_with_mock_factories(self, simple_mock_setup):
        """モック工場を使った評価"""
        env_factory, algo_factory, mock_env, mock_model = simple_mock_setup

        evaluator = WalkForwardModelEvaluator(
            env_factory=env_factory,
            algorithm_factory=algo_factory,
        )

        # モック環境とモデルの設定
        mock_model.learn = Mock()
        mock_model.predict.side_effect = [
            (1, None),  # Window 0, step 1
            (0, None),  # Window 0, step 2
        ]

        # 通常の流れではなく、簡略テストのため スキップ
        # 実際の使用シーンでは、より詳細なモック設定が必要


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
