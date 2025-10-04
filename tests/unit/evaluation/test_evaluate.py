import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from ztb.evaluation.evaluate import TradingEvaluator


class TestTradingEvaluator:
    """Test TradingEvaluator functionality."""

    @patch("ztb.evaluation.evaluate.PPO.load")
    def test_init_with_valid_paths(self, mock_ppo_load):
        """Test initialization with valid model and data paths."""
        mock_model = MagicMock()
        mock_ppo_load.return_value = mock_model

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.zip"
            data_path = Path(tmp) / "data.csv"

            # Create dummy files
            model_path.write_text("dummy model")
            data_path.write_text("timestamp,price\n2023-01-01,50000")

            evaluator = TradingEvaluator(str(model_path), str(data_path))

            assert evaluator.model_path == model_path
            assert evaluator.data_path == data_path
            assert evaluator.config is not None
            mock_ppo_load.assert_called_once_with(str(model_path))

    @patch("ztb.evaluation.evaluate.PPO.load")
    def test_init_with_config(self, mock_ppo_load):
        """Test initialization with custom config."""
        mock_model = MagicMock()
        mock_ppo_load.return_value = mock_model

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.zip"
            data_path = Path(tmp) / "data.csv"

            model_path.write_text("dummy model")
            data_path.write_text("timestamp,price\n2023-01-01,50000")

            config = {"results_dir": "./results/", "custom": "config"}
            evaluator = TradingEvaluator(str(model_path), str(data_path), config)

            assert evaluator.config == config

    def test_init_missing_model_file(self):
        """Test initialization with missing model file."""
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "missing_model.zip"
            data_path = Path(tmp) / "data.csv"

            data_path.write_text("timestamp,price\n2023-01-01,50000")

            # Should raise because model file doesn't exist
            try:
                evaluator = TradingEvaluator(str(model_path), str(data_path))
                assert False, "Should have raised FileNotFoundError"
            except FileNotFoundError as e:
                assert "Model not found" in str(e)

    def test_init_missing_data_file(self):
        """Test initialization with missing data file."""
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.zip"
            data_path = Path(tmp) / "missing_data.csv"

            model_path.write_text("dummy model")

            # Should raise because data file doesn't exist
            try:
                evaluator = TradingEvaluator(str(model_path), str(data_path))
                assert False, "Should have raised FileNotFoundError"
            except FileNotFoundError:
                pass  # Expected
