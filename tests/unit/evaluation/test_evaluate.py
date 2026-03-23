from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    from ztb.analysis.evaluator import TradingEvaluator
except ImportError:
    import pytest

    pytest.skip(
        "ztb.evaluation.evaluate module not available (stable_baselines3 dependency)",
        allow_module_level=True,
    )


class TestTradingEvaluator:
    """Test TradingEvaluator functionality."""

    @patch("ztb.evaluation.evaluate.PPO.load")
    def test_init_with_valid_paths(self, mock_ppo_load, tmp_path: Path):
        """Test initialization with valid model and data paths."""
        mock_model = MagicMock()
        mock_ppo_load.return_value = mock_model

        model_path = tmp_path / "model.zip"
        data_path = tmp_path / "data.csv"

        model_path.write_text("dummy model")
        data_path.write_text("timestamp,price\n2023-01-01,50000")

        evaluator = TradingEvaluator(str(model_path), str(data_path))

        assert evaluator.model_path == model_path
        assert evaluator.data_path == data_path
        assert evaluator.config is not None
        mock_ppo_load.assert_called_once_with(str(model_path))

    @patch("ztb.evaluation.evaluate.PPO.load")
    def test_init_with_config(self, mock_ppo_load, tmp_path: Path):
        """Test initialization with custom config."""
        mock_model = MagicMock()
        mock_ppo_load.return_value = mock_model

        model_path = tmp_path / "model.zip"
        data_path = tmp_path / "data.csv"

        model_path.write_text("dummy model")
        data_path.write_text("timestamp,price\n2023-01-01,50000")

        config = {"results_dir": "./results/", "custom": "config"}
        evaluator = TradingEvaluator(str(model_path), str(data_path), config)

        assert evaluator.config == config

    def test_init_missing_model_file(self, tmp_path: Path):
        """Test initialization with missing model file."""
        model_path = tmp_path / "missing_model.zip"
        data_path = tmp_path / "data.csv"

        data_path.write_text("timestamp,price\n2023-01-01,50000")

        try:
            TradingEvaluator(str(model_path), str(data_path))
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            assert "Model not found" in str(e)

    def test_init_missing_data_file(self, tmp_path: Path):
        """Test initialization with missing data file."""
        model_path = tmp_path / "model.zip"
        data_path = tmp_path / "missing_data.csv"

        model_path.write_text("dummy model")

        try:
            TradingEvaluator(str(model_path), str(data_path))
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass
