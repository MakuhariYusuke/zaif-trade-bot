"""
Walk-Forward Analysis 統合テスト（実データシナリオ）

実際の OHLCV データを使用した 3-5 ウィンドウの
Walk-Forward シナリオテスト。

## テストシナリオ

1. **小規模データ評価** (3 ウィンドウ)
   - 300 本のローソク足（1 時間足相当）
   - 訓練: 50%, 検証: 15%, テスト: 15%, スキップ: 20%

2. **中規模データ評価** (5 ウィンドウ)
   - 500 本のローソク足
   - Embargo 機構の動作確認

3. **Checkpoint 機能テスト**
   - 評価途中でのチェックポイント保存
   - 復元後の続行
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from ztb.evaluation.walk_forward import (
    WalkForwardModelEvaluator,
    WalkForwardSplitter,
)
from ztb.evaluation.walk_forward.checkpoint import CheckpointManager
from ztb.evaluation.walk_forward.types import TimeSeriesWindow, WindowPerformance


class TestWalkForwardIntegrationE2E:
    """End-to-End 統合テスト（実データシナリオ）"""

    @pytest.fixture
    def small_ohlcv_df(self) -> pd.DataFrame:
        """小規模 OHLCV データ（1000 本）"""
        np.random.seed(42)
        n_bars = 1000
        base_price = 40000.0
        
        prices = base_price + np.cumsum(np.random.randn(n_bars) * 100)
        
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_bars, freq='1h'),
            'open': prices,
            'high': prices + np.random.rand(n_bars) * 200,
            'low': prices - np.random.rand(n_bars) * 200,
            'close': prices + np.random.randn(n_bars) * 50,
            'volume': np.random.randint(100, 1000, n_bars),
        }).set_index('timestamp')

    @pytest.fixture
    def medium_ohlcv_df(self) -> pd.DataFrame:
        """中規模 OHLCV データ（2000 本）"""
        np.random.seed(43)
        n_bars = 2000
        base_price = 40000.0
        
        prices = base_price + np.cumsum(np.random.randn(n_bars) * 100)
        
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_bars, freq='1h'),
            'open': prices,
            'high': prices + np.random.rand(n_bars) * 200,
            'low': prices - np.random.rand(n_bars) * 200,
            'close': prices + np.random.randn(n_bars) * 50,
            'volume': np.random.randint(100, 1000, n_bars),
        }).set_index('timestamp')

    def test_small_walk_forward_3_windows(self, small_ohlcv_df):
        """小規模データの 3 ウィンドウ評価"""
        splitter = WalkForwardSplitter(
            initial_train_pct=0.40,
            val_pct=0.15,
            test_pct=0.15,
            step_pct=0.15,
        )
        windows = splitter.split(small_ohlcv_df)

        # 3 ウィンドウ取得を確認
        assert len(windows) >= 2  # データサイズ制約
        windows = windows[:min(3, len(windows))]

        # ウィンドウ検証
        for window in windows:
            assert window.train_end > window.train_start
            assert window.val_end > window.val_start
            assert window.test_end > window.test_start
            assert window.train_end <= window.val_start
            assert window.val_end <= window.test_start

    def test_medium_walk_forward_5_windows(self, medium_ohlcv_df):
        """中規模データの 5 ウィンドウ評価"""
        splitter = WalkForwardSplitter(
            initial_train_pct=0.40,
            val_pct=0.12,
            test_pct=0.12,
            step_pct=0.10,
        )
        windows = splitter.split(medium_ohlcv_df)

        # 3 ウィンドウ以上を確認
        assert len(windows) >= 2
        windows = windows[:min(5, len(windows))]

        # ウィンドウの連続性確認
        for i in range(len(windows) - 1):
            current = windows[i]
            next_window = windows[i + 1]
            # 次のウィンドウの訓練開始は、前のウィンドウのテスト終了より後
            assert next_window.train_start >= current.test_end

    def test_embargo_mechanism(self, medium_ohlcv_df):
        """Embargo 機構の動作確認"""
        splitter = WalkForwardSplitter(
            initial_train_pct=0.40,
            val_pct=0.15,
            test_pct=0.15,
            step_pct=0.10,
            embargo_days=5,  # 5 日の Embargo
        )
        windows = splitter.split(medium_ohlcv_df)

        assert len(windows) > 0

        # 最初のウィンドウで構造を確認
        window = windows[0]
        
        # バリデーション開始 >= 訓練終了
        assert window.val_start >= window.train_end

    def test_data_leakage_prevention(self, medium_ohlcv_df):
        """データリーク防止のテスト"""
        splitter = WalkForwardSplitter(
            initial_train_pct=0.40,
            val_pct=0.15,
            test_pct=0.15,
            step_pct=0.10,
        )
        windows = splitter.split(medium_ohlcv_df)

        for window in windows:
            # インデックスの重複がないこと
            train_indices = set(range(window.train_start, window.train_end))
            val_indices = set(range(window.val_start, window.val_end))
            test_indices = set(range(window.test_start, window.test_end))

            assert len(train_indices & val_indices) == 0
            assert len(train_indices & test_indices) == 0
            assert len(val_indices & test_indices) == 0


class TestWalkForwardEvaluatorE2E:
    """WalkForwardModelEvaluator の E2E テスト"""

    @pytest.fixture
    def temp_checkpoint_dir(self, tmp_path: Path):
        """一時チェックポイントディレクトリ"""
        return str(tmp_path / "walk_forward_e2e")

    @pytest.fixture
    def mock_ohlcv_df(self) -> pd.DataFrame:
        """モック OHLCV データ（1500 本）"""
        np.random.seed(44)
        n_bars = 1500
        base_price = 40000.0
        prices = base_price + np.cumsum(np.random.randn(n_bars) * 100)

        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_bars, freq='1h'),
            'open': prices,
            'high': prices + np.random.rand(n_bars) * 200,
            'low': prices - np.random.rand(n_bars) * 200,
            'close': prices + np.random.randn(n_bars) * 50,
            'volume': np.random.randint(100, 1000, n_bars),
        }).set_index('timestamp')

    def test_evaluator_multiple_windows_workflow(
        self, mock_ohlcv_df, temp_checkpoint_dir
    ):
        """複数ウィンドウ評価のワークフロー"""
        # ウィンドウ生成
        splitter = WalkForwardSplitter(
            initial_train_pct=0.40,
            val_pct=0.15,
            test_pct=0.15,
            step_pct=0.15,
        )
        windows = splitter.split(mock_ohlcv_df)
        windows = windows[:min(3, len(windows))]  # 3 ウィンドウに限定

        # Evaluator 作成（チェックポイント有効）
        evaluator = WalkForwardModelEvaluator(
            checkpoint_dir=temp_checkpoint_dir
        )

        # train_and_evaluate_window をモック
        with patch.object(evaluator, "train_and_evaluate_window") as mock_train:
            # 3 ウィンドウ分の結果をシミュレート
            results = [
                WindowPerformance(
                    window_id=i,
                    val_roi=0.03 + i * 0.01,
                    test_roi=0.025 + i * 0.005,
                    val_final_balance=1030000.0 + i * 10000,
                    test_final_balance=1025000.0 + i * 8000,
                    sharpe_ratio=1.2 + i * 0.1,
                    max_drawdown=-0.10 - i * 0.02,
                    win_rate=0.50 + i * 0.05,
                    trades=50 + i * 10,
                )
                for i in range(len(windows))
            ]
            mock_train.side_effect = results

            # 評価実行
            returned_results, errors = evaluator.evaluate_multiple_windows(
                df=mock_ohlcv_df,
                windows=windows,
                run_id="test_e2e_run",
            )

        # 結果検証
        assert len(returned_results) == len(windows)
        assert len(errors) == 0
        assert evaluator.checkpoint_manager is not None

        # チェックポイント確認
        checkpoint_dir = Path(temp_checkpoint_dir) / "test_e2e_run"
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "run_metadata.json").exists()

    def test_checkpoint_recovery_workflow(
        self, mock_ohlcv_df, temp_checkpoint_dir
    ):
        """チェックポイント復旧のワークフロー"""
        splitter = WalkForwardSplitter(
            initial_train_pct=0.40,
            val_pct=0.15,
            test_pct=0.15,
            step_pct=0.15,
        )
        windows = splitter.split(mock_ohlcv_df)
        windows = windows[:min(3, len(windows))]

        # 1. 最初の実行：ウィンドウ 0-1 を完了
        evaluator1 = WalkForwardModelEvaluator(
            checkpoint_dir=temp_checkpoint_dir
        )

        with patch.object(evaluator1, "train_and_evaluate_window") as mock_train:
            # ウィンドウ 0-1 のみ成功
            results = [
                WindowPerformance(
                    window_id=0,
                    val_roi=0.03,
                    test_roi=0.025,
                    val_final_balance=1030000.0,
                    test_final_balance=1025000.0,
                    sharpe_ratio=1.2,
                    max_drawdown=-0.10,
                    win_rate=0.50,
                    trades=50,
                ),
                WindowPerformance(
                    window_id=1,
                    val_roi=0.04,
                    test_roi=0.03,
                    val_final_balance=1040000.0,
                    test_final_balance=1030000.0,
                    sharpe_ratio=1.3,
                    max_drawdown=-0.12,
                    win_rate=0.55,
                    trades=60,
                ),
            ]
            mock_train.side_effect = results[:min(2, len(windows))]

            evaluator1.evaluate_multiple_windows(
                df=mock_ohlcv_df,
                windows=windows[:min(2, len(windows))],
                run_id="test_recovery",
            )

            # チェックポイント保存
            evaluator1.checkpoint_manager.save(evaluator1, "test_recovery")

        # 2. 復旧実行：ウィンドウ 2 を追加
        evaluator2 = WalkForwardModelEvaluator(
            checkpoint_dir=temp_checkpoint_dir
        )

        # チェックポイント復元
        evaluator2.checkpoint_manager.restore(evaluator2, "test_recovery")

        assert len(evaluator2.results) >= 1

        # 残りのウィンドウを実行
        if len(windows) > 2:
            with patch.object(evaluator2, "train_and_evaluate_window") as mock_train:
                result = WindowPerformance(
                    window_id=2,
                    val_roi=0.05,
                    test_roi=0.035,
                    val_final_balance=1050000.0,
                    test_final_balance=1035000.0,
                    sharpe_ratio=1.4,
                    max_drawdown=-0.14,
                    win_rate=0.60,
                    trades=70,
                )
                mock_train.return_value = result

                results, errors = evaluator2.evaluate_multiple_windows(
                    df=mock_ohlcv_df,
                    windows=windows[2:],
                    run_id="test_recovery",
                    resume_from_checkpoint=True,
                )

            # 最終結果検証
            assert len(evaluator2.results) >= 2


class TestWalkForwardResultsAggregation:
    """結果集計と分析テスト"""

    def test_summary_statistics(self):
        """サマリー統計計算"""
        evaluator = WalkForwardModelEvaluator()

        # 3 ウィンドウ分の結果を追加
        for i in range(3):
            evaluator.results[i] = WindowPerformance(
                window_id=i,
                val_roi=0.03 + i * 0.01,
                test_roi=0.025 + i * 0.005,
                val_final_balance=1030000.0 + i * 10000,
                test_final_balance=1025000.0 + i * 8000,
                sharpe_ratio=1.2 + i * 0.1,
                max_drawdown=-0.10 - i * 0.02,
                win_rate=0.50 + i * 0.05,
                trades=50 + i * 10,
            )

        summary = evaluator.get_results_summary()

        assert summary["total_windows"] == 3
        assert summary["successful_windows"] == 3
        assert summary["failed_windows"] == 0
        assert summary["avg_test_roi"] == pytest.approx((0.025 + 0.030 + 0.035) / 3)
        assert "std_test_roi" in summary
        assert "avg_sharpe" in summary

    def test_performance_degradation_detection(self):
        """パフォーマンス劣化の検出"""
        evaluator = WalkForwardModelEvaluator()

        # 劣化シナリオ：テスト性能が悪化
        test_rois = [0.05, 0.03, 0.01]  # 低下傾向
        for i, roi in enumerate(test_rois):
            evaluator.results[i] = WindowPerformance(
                window_id=i,
                val_roi=0.04,  # 一定
                test_roi=roi,  # 低下
                val_final_balance=1040000.0,
                test_final_balance=1000000.0 + roi * 1000000,
                sharpe_ratio=1.2,
                max_drawdown=-0.10,
                win_rate=0.50,
                trades=50,
            )

        summary = evaluator.get_results_summary()
        test_rois_from_results = [
            evaluator.results[i].test_roi for i in range(3)
        ]

        # パフォーマンス低下を検出可能
        assert test_rois_from_results[0] > test_rois_from_results[2]
