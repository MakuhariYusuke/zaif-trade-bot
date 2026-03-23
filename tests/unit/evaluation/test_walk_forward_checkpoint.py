"""
Walk-Forward Checkpoint/Resume テストスイート

CheckpointManager と WalkForwardModelEvaluator の
チェックポイント統合機能をテストします。
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from ztb.evaluation.walk_forward.checkpoint import CheckpointManager
from ztb.evaluation.walk_forward.evaluator import WalkForwardModelEvaluator
from ztb.evaluation.walk_forward.types import TimeSeriesWindow, WindowPerformance


class TestCheckpointManagerBasics:
    """CheckpointManager の基本機能"""

    @pytest.fixture
    def temp_checkpoint_dir(self, tmp_path: Path):
        """一時的なチェックポイントディレクトリ"""
        return str(tmp_path / "checkpoints_basic")

    @pytest.fixture
    def checkpoint_manager(self, temp_checkpoint_dir):
        """チェックポイントマネージャー"""
        return CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

    def test_initialization(self, checkpoint_manager, temp_checkpoint_dir):
        """初期化テスト"""
        assert checkpoint_manager.checkpoint_dir == Path(temp_checkpoint_dir)
        assert Path(temp_checkpoint_dir).exists()

    def test_list_runs_empty(self, checkpoint_manager):
        """実行リストが空の場合"""
        runs = checkpoint_manager.list_runs()
        assert runs == []

    def test_list_runs_with_existing(self, checkpoint_manager):
        """既存の実行IDを列挙"""
        # ダミーディレクトリを作成
        (checkpoint_manager.checkpoint_dir / "run_001").mkdir()
        (checkpoint_manager.checkpoint_dir / "run_002").mkdir()
        (checkpoint_manager.checkpoint_dir / "not_a_run.txt").touch()

        runs = checkpoint_manager.list_runs()
        assert set(runs) == {"run_001", "run_002"}


class TestCheckpointManagerSaveRestore:
    """CheckpointManager の save/restore 機能"""

    @pytest.fixture
    def temp_checkpoint_dir(self, tmp_path: Path):
        """一時的なチェックポイントディレクトリ"""
        return str(tmp_path / "checkpoints_save_restore")

    @pytest.fixture
    def checkpoint_manager(self, temp_checkpoint_dir):
        """チェックポイントマネージャー"""
        return CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

    @pytest.fixture
    def mock_evaluator(self):
        """モック evaluator"""
        evaluator = Mock(spec=WalkForwardModelEvaluator)
        evaluator.results = {
            0: WindowPerformance(
                window_id=0,
                val_roi=0.05,
                test_roi=0.04,
                val_final_balance=1050000.0,
                test_final_balance=1040000.0,
                sharpe_ratio=1.5,
                max_drawdown=-0.10,
                win_rate=0.55,
                trades=50,
            ),
            1: WindowPerformance(
                window_id=1,
                val_roi=0.06,
                test_roi=0.05,
                val_final_balance=1060000.0,
                test_final_balance=1050000.0,
                sharpe_ratio=1.7,
                max_drawdown=-0.08,
                win_rate=0.60,
                trades=60,
            ),
        }
        evaluator.errors = {}
        evaluator.models = {}
        return evaluator

    def test_save_basic(self, checkpoint_manager, mock_evaluator):
        """基本的な保存テスト"""
        result = checkpoint_manager.save(mock_evaluator, run_id="test_run")

        assert result["run_id"] == "test_run"
        assert result["total_windows_completed"] == 2
        assert result["total_windows_failed"] == 0

        # ディレクトリ構造確認
        run_dir = checkpoint_manager.checkpoint_dir / "test_run"
        assert (run_dir / "run_metadata.json").exists()
        assert (run_dir / "runtime_data.pkl").exists()
        assert (run_dir / "window_0").exists()
        assert (run_dir / "window_1").exists()

    def test_save_with_errors(self, checkpoint_manager, mock_evaluator):
        """エラー付き保存テスト"""
        mock_evaluator.errors = {2: Exception("Test error")}

        result = checkpoint_manager.save(mock_evaluator, run_id="test_run")

        assert result["total_windows_failed"] == 1

    def test_save_window_metadata(self, checkpoint_manager, mock_evaluator):
        """ウィンドウメタデータ保存確認"""
        checkpoint_manager.save(mock_evaluator, run_id="test_run")

        metadata_path = (
            checkpoint_manager.checkpoint_dir / "test_run" / "window_0"
            / "checkpoint_metadata.json"
        )
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["window_id"] == 0
        assert metadata["status"] == "completed"
        assert "timestamp" in metadata

    def test_save_window_results(self, checkpoint_manager, mock_evaluator):
        """ウィンドウ結果保存確認"""
        checkpoint_manager.save(mock_evaluator, run_id="test_run")

        results_path = (
            checkpoint_manager.checkpoint_dir / "test_run" / "window_0"
            / "window_results.json"
        )
        with open(results_path) as f:
            perf_data = json.load(f)

        assert perf_data["window_id"] == 0
        assert perf_data["val_roi"] == 0.05
        assert perf_data["test_roi"] == 0.04
        assert perf_data["sharpe_ratio"] == 1.5

    def test_restore_basic(self, checkpoint_manager, mock_evaluator):
        """基本的な復元テスト"""
        # 先に保存
        checkpoint_manager.save(mock_evaluator, run_id="test_run")

        # 新しい evaluator で復元
        new_evaluator = Mock(spec=WalkForwardModelEvaluator)
        new_evaluator.results = {}
        new_evaluator.errors = {}
        new_evaluator.models = {}

        result = checkpoint_manager.restore(new_evaluator, run_id="test_run")

        assert result["run_id"] == "test_run"
        assert result["restored_windows"] == 2
        assert len(new_evaluator.results) == 2
        assert 0 in new_evaluator.results
        assert 1 in new_evaluator.results

    def test_restore_performance_values(self, checkpoint_manager, mock_evaluator):
        """復元されたパフォーマンス値確認"""
        checkpoint_manager.save(mock_evaluator, run_id="test_run")

        new_evaluator = Mock(spec=WalkForwardModelEvaluator)
        new_evaluator.results = {}
        new_evaluator.errors = {}
        new_evaluator.models = {}

        checkpoint_manager.restore(new_evaluator, run_id="test_run")

        perf0 = new_evaluator.results[0]
        assert perf0.val_roi == 0.05
        assert perf0.test_roi == 0.04
        assert perf0.sharpe_ratio == 1.5


class TestCheckpointManagerStatus:
    """CheckpointManager のステータス確認機能"""

    @pytest.fixture
    def temp_checkpoint_dir(self, tmp_path: Path):
        """一時的なチェックポイントディレクトリ"""
        return str(tmp_path / "checkpoints_status")

    @pytest.fixture
    def checkpoint_manager(self, temp_checkpoint_dir):
        """チェックポイントマネージャー"""
        return CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

    @pytest.fixture
    def mock_evaluator(self):
        """モック evaluator"""
        evaluator = Mock(spec=WalkForwardModelEvaluator)
        evaluator.results = {
            0: WindowPerformance(
                window_id=0,
                val_roi=0.05,
                test_roi=0.04,
                val_final_balance=1050000.0,
                test_final_balance=1040000.0,
                sharpe_ratio=1.5,
                max_drawdown=-0.10,
                win_rate=0.55,
                trades=50,
            ),
            1: WindowPerformance(
                window_id=1,
                val_roi=0.06,
                test_roi=0.05,
                val_final_balance=1060000.0,
                test_final_balance=1050000.0,
                sharpe_ratio=1.7,
                max_drawdown=-0.08,
                win_rate=0.60,
                trades=60,
            ),
        }
        evaluator.errors = {2: Exception("Test error")}
        evaluator.models = {}
        return evaluator

    def test_get_run_status(self, checkpoint_manager, mock_evaluator):
        """実行ステータス取得"""
        checkpoint_manager.save(mock_evaluator, run_id="test_run")

        status = checkpoint_manager.get_run_status("test_run")

        assert status["run_id"] == "test_run"
        assert status["completed_windows"] == 2
        assert status["failed_windows"] == 1
        assert status["total_windows"] == 3
        assert status["progress_pct"] == pytest.approx(66.666, abs=0.1)

    def test_get_results_summary(self, checkpoint_manager, mock_evaluator):
        """結果サマリー取得"""
        checkpoint_manager.save(mock_evaluator, run_id="test_run")

        summary = checkpoint_manager.get_results_summary("test_run")

        assert summary["total_windows"] == 2
        assert summary["avg_val_roi"] == pytest.approx((0.05 + 0.06) / 2)
        assert summary["avg_test_roi"] == pytest.approx((0.04 + 0.05) / 2)
        assert summary["avg_sharpe"] == pytest.approx((1.5 + 1.7) / 2)

    def test_get_completed_windows(self, checkpoint_manager, mock_evaluator):
        """完了ウィンドウID取得"""
        checkpoint_manager.save(mock_evaluator, run_id="test_run")

        completed = checkpoint_manager.get_completed_windows("test_run")

        assert set(completed) == {0, 1}

    def test_delete_run(self, checkpoint_manager, mock_evaluator):
        """実行削除"""
        checkpoint_manager.save(mock_evaluator, run_id="test_run")
        assert (checkpoint_manager.checkpoint_dir / "test_run").exists()

        result = checkpoint_manager.delete_run("test_run")

        assert result is True
        assert not (checkpoint_manager.checkpoint_dir / "test_run").exists()


class TestWalkForwardModelEvaluatorCheckpoint:
    """WalkForwardModelEvaluator のチェックポイント統合"""

    @pytest.fixture
    def temp_checkpoint_dir(self, tmp_path: Path):
        """一時的なチェックポイントディレクトリ"""
        return str(tmp_path / "checkpoints_evaluator")

    def test_evaluator_with_checkpoint_dir(self, temp_checkpoint_dir):
        """チェックポイント有効な evaluator"""
        evaluator = WalkForwardModelEvaluator(checkpoint_dir=temp_checkpoint_dir)

        assert evaluator.checkpoint_dir == temp_checkpoint_dir
        assert evaluator.checkpoint_manager is not None

    def test_evaluator_without_checkpoint_dir(self):
        """チェックポイント無効な evaluator"""
        evaluator = WalkForwardModelEvaluator()

        assert evaluator.checkpoint_dir is None
        assert evaluator.checkpoint_manager is None

    @pytest.fixture
    def mock_windows(self):
        """モックウィンドウ"""
        return [
            TimeSeriesWindow(
                window_id=0,
                train_start=0,
                train_end=100,
                val_start=100,
                val_end=120,
                test_start=120,
                test_end=150,
            ),
            TimeSeriesWindow(
                window_id=1,
                train_start=150,
                train_end=250,
                val_start=250,
                val_end=270,
                test_start=270,
                test_end=300,
            ),
        ]

    def test_evaluate_multiple_windows_with_checkpoint(
        self, temp_checkpoint_dir, mock_windows
    ):
        """チェックポイント付きの複数ウィンドウ評価"""
        evaluator = WalkForwardModelEvaluator(checkpoint_dir=temp_checkpoint_dir)

        # モックデータフレーム
        df = Mock()

        # train_and_evaluate_window をモック
        with patch.object(evaluator, "train_and_evaluate_window") as mock_train:
            mock_perf = WindowPerformance(
                window_id=0,
                val_roi=0.05,
                test_roi=0.04,
                val_final_balance=1050000.0,
                test_final_balance=1040000.0,
                sharpe_ratio=1.5,
                max_drawdown=-0.10,
                win_rate=0.55,
                trades=50,
            )
            mock_train.return_value = mock_perf

            # results に直接追加（モック）
            evaluator.results[0] = mock_perf
            evaluator.results[1] = WindowPerformance(
                window_id=1,
                val_roi=0.06,
                test_roi=0.05,
                val_final_balance=1060000.0,
                test_final_balance=1050000.0,
                sharpe_ratio=1.7,
                max_drawdown=-0.08,
                win_rate=0.60,
                trades=60,
            )

            results, errors = evaluator.evaluate_multiple_windows(
                df=df,
                windows=mock_windows,
                run_id="test_run",
            )

        assert len(evaluator.results) == 2
        # チェックポイント保存確認
        assert (
            Path(temp_checkpoint_dir) / "test_run" / "run_metadata.json"
        ).exists()

    def test_resume_from_checkpoint(self, temp_checkpoint_dir, mock_windows):
        """チェックポイントから再開"""
        # 最初の実行で部分的に完了
        evaluator1 = WalkForwardModelEvaluator(checkpoint_dir=temp_checkpoint_dir)
        evaluator1.results[0] = WindowPerformance(
            window_id=0,
            val_roi=0.05,
            test_roi=0.04,
            val_final_balance=1050000.0,
            test_final_balance=1040000.0,
            sharpe_ratio=1.5,
            max_drawdown=-0.10,
            win_rate=0.55,
            trades=50,
        )
        evaluator1.errors = {}
        evaluator1.models = {}
        evaluator1.checkpoint_manager.save(evaluator1, run_id="test_run")

        # 再開
        evaluator2 = WalkForwardModelEvaluator(checkpoint_dir=temp_checkpoint_dir)
        df = Mock()

        with patch.object(evaluator2, "train_and_evaluate_window") as mock_train:
            mock_train.return_value = None  # エラーシミュレート

            results, errors = evaluator2.evaluate_multiple_windows(
                df=df,
                windows=mock_windows,
                run_id="test_run",
                resume_from_checkpoint=True,
            )

        # 復元確認
        assert 0 in evaluator2.results
        # ウィンドウ 0 は既に完了しているので、ウィンドウ 1 だけが評価されるはず
        # （mock_train は 1 回だけ呼ばれる）
        assert mock_train.call_count == 1


class TestCheckpointIntegration:
    """統合テスト"""

    @pytest.fixture
    def temp_checkpoint_dir(self, tmp_path: Path):
        """一時的なチェックポイントディレクトリ"""
        return str(tmp_path / "checkpoints_integration")

    def test_full_checkpoint_cycle(self, temp_checkpoint_dir):
        """チェックポイント全サイクル"""
        # 1. 作成・保存
        evaluator1 = WalkForwardModelEvaluator(checkpoint_dir=temp_checkpoint_dir)
        evaluator1.results[0] = WindowPerformance(
            window_id=0,
            val_roi=0.05,
            test_roi=0.04,
            val_final_balance=1050000.0,
            test_final_balance=1040000.0,
            sharpe_ratio=1.5,
            max_drawdown=-0.10,
            win_rate=0.55,
            trades=50,
        )
        evaluator1.errors = {}
        evaluator1.models = {}

        evaluator1.checkpoint_manager.save(evaluator1, run_id="full_test")

        # 2. 復元
        evaluator2 = WalkForwardModelEvaluator(checkpoint_dir=temp_checkpoint_dir)
        evaluator2.checkpoint_manager.restore(evaluator2, run_id="full_test")

        assert len(evaluator2.results) == 1
        assert 0 in evaluator2.results
        assert evaluator2.results[0].val_roi == 0.05

        # 3. ステータス確認
        status = evaluator2.checkpoint_manager.get_run_status("full_test")
        assert status["completed_windows"] == 1

        # 4. 削除
        evaluator2.checkpoint_manager.delete_run("full_test")
        assert not (Path(temp_checkpoint_dir) / "full_test").exists()
