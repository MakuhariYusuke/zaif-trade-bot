"""
Unit tests for curriculum_learning.py module.
"""

from pathlib import Path
from unittest.mock import Mock, mock_open, patch

from ztb.training.curriculum_learning import (
    evaluate_stage_performance,
    run_curriculum_stage,
)


class TestRunCurriculumStage:
    """Test cases for run_curriculum_stage function."""

    @patch("ztb.training.curriculum_learning.safe_json_load")
    @patch("ztb.training.curriculum_learning.UnifiedTrainer")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("os.remove")
    def test_run_curriculum_stage_success(
        self, mock_remove, mock_exists, mock_file, mock_trainer_class, mock_load
    ):
        """Test run_curriculum_stage with successful training."""
        # Mock base config
        base_config = {"session_id": "base", "total_timesteps": 1000}
        mock_load.return_value = base_config

        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.train.return_value = True
        mock_trainer_class.return_value = mock_trainer

        # Mock file operations
        mock_exists.return_value = True

        # Test
        config_updates = {"total_timesteps": 50000, "ent_coef": 0.8}
        result = run_curriculum_stage("test_stage", config_updates, "test_config.json")

        assert result is True
        mock_load.assert_called_once_with(Path("test_config.json"))

        # Verify config was updated
        expected_config = {
            "session_id": "curriculum_test_stage",
            "total_timesteps": 50000,
            "ent_coef": 0.8,
        }
        # Check that json.dump was called with updated config
        mock_file.assert_called()
        # Verify trainer was created and trained
        mock_trainer_class.assert_called_once_with(expected_config)
        mock_trainer.train.assert_called_once()

        # Verify cleanup
        mock_remove.assert_called_once()

    @patch("ztb.training.curriculum_learning.safe_json_load")
    @patch("ztb.training.curriculum_learning.UnifiedTrainer")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("os.remove")
    def test_run_curriculum_stage_failure(
        self, mock_remove, mock_exists, mock_file, mock_trainer_class, mock_load
    ):
        """Test run_curriculum_stage with failed training."""
        # Mock base config
        base_config = {"session_id": "base"}
        mock_load.return_value = base_config

        # Mock trainer that fails
        mock_trainer = Mock()
        mock_trainer.train.return_value = False
        mock_trainer_class.return_value = mock_trainer

        # Mock file operations
        mock_exists.return_value = True

        # Test
        result = run_curriculum_stage("test_stage", {}, "test_config.json")

        assert result is False
        mock_trainer.train.assert_called_once()
        mock_remove.assert_called_once()

    @patch("ztb.training.curriculum_learning.safe_json_load")
    @patch("ztb.training.curriculum_learning.UnifiedTrainer")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("os.remove")
    def test_run_curriculum_stage_cleanup_on_exception(
        self, mock_remove, mock_exists, mock_file, mock_trainer_class, mock_load
    ):
        """Test run_curriculum_stage cleans up temp file even on exception."""
        # Mock base config
        base_config = {"session_id": "base"}
        mock_load.return_value = base_config

        # Mock trainer that raises exception
        mock_trainer = Mock()
        mock_trainer.train.side_effect = Exception("Training failed")
        mock_trainer_class.return_value = mock_trainer

        # Mock file operations
        mock_exists.return_value = True

        # Test - should not raise exception, should return False and cleanup
        result = run_curriculum_stage("test_stage", {}, "test_config.json")

        assert result is False
        mock_remove.assert_called_once()  # Should still clean up

    @patch("ztb.training.curriculum_learning.safe_json_load")
    @patch("ztb.training.curriculum_learning.UnifiedTrainer")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("os.remove")
    def test_run_curriculum_stage_custom_base_config(
        self, mock_remove, mock_exists, mock_file, mock_trainer_class, mock_load
    ):
        """Test run_curriculum_stage with custom base config path."""
        # Mock base config
        base_config = {"session_id": "base"}
        mock_load.return_value = base_config

        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.train.return_value = True
        mock_trainer_class.return_value = mock_trainer

        # Mock file operations
        mock_exists.return_value = True

        # Test with custom base config path
        result = run_curriculum_stage("test_stage", {}, "custom_config.json")

        assert result is True
        mock_load.assert_called_once_with(Path("custom_config.json"))


class TestEvaluateStagePerformance:
    """Test cases for evaluate_stage_performance function."""

    @patch("os.path.exists")
    def test_evaluate_stage_performance_no_model_file(self, mock_exists):
        """Test evaluate_stage_performance when model file doesn't exist."""
        mock_exists.return_value = False

        # Should not raise exception, just return
        evaluate_stage_performance("test_stage")

        mock_exists.assert_called_once_with("models/curriculum_test_stage.zip")

    @patch("os.path.exists")
    @patch("subprocess.run")
    def test_evaluate_stage_performance_subprocess_success(
        self, mock_subprocess, mock_exists
    ):
        """Test evaluate_stage_performance with successful subprocess call."""
        mock_exists.return_value = True

        # Mock subprocess success
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        evaluate_stage_performance("test_stage")

        # Verify subprocess was called correctly
        mock_subprocess.assert_called_once_with(
            [
                "python",
                "regime_evaluation.py",
                "--models",
                "test_stage:models/curriculum_test_stage.zip",
                "--price-data",
                "ml-dataset-enhanced.csv",
            ],
            capture_output=True,
            text=True,
        )

    @patch("os.path.exists")
    @patch("subprocess.run")
    def test_evaluate_stage_performance_subprocess_failure(
        self, mock_subprocess, mock_exists
    ):
        """Test evaluate_stage_performance with failed subprocess call."""
        mock_exists.return_value = True

        # Mock subprocess failure
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Evaluation failed"
        mock_subprocess.return_value = mock_result

        evaluate_stage_performance("test_stage")

        # Should handle failure gracefully
        mock_subprocess.assert_called_once()

    @patch("os.path.exists")
    @patch("subprocess.run")
    @patch("ztb.training.curriculum_learning.safe_json_load")
    @patch("pathlib.Path.exists")
    def test_evaluate_stage_performance_with_results_file(
        self, mock_path_exists, mock_load, mock_subprocess, mock_exists
    ):
        """Test evaluate_stage_performance with results file parsing."""
        mock_exists.return_value = True

        # Mock subprocess success
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        # Mock results file exists and has data
        mock_path_exists.return_value = True
        mock_load.return_value = {
            "regime_metrics": {
                "test_stage": {
                    "bull_market": {
                        "action_distribution": {"BUY": 40.5, "SELL": 30.2, "HOLD": 29.3}
                    },
                    "bear_market": {
                        "action_distribution": {"BUY": 20.1, "SELL": 50.8, "HOLD": 29.1}
                    },
                }
            }
        }

        evaluate_stage_performance("test_stage")

        mock_load.assert_called_once()
        mock_path_exists.assert_called_once()
