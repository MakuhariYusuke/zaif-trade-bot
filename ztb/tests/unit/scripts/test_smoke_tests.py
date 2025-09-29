"""
Unit tests for regression smoke tests.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from ztb.scripts.smoke_tests import (
    create_synthetic_data,
    run_paper_trader_smoke_test,
    run_ppo_trainer_smoke_test,
    run_venue_health_check_smoke_test,
)


class TestSyntheticDataCreation:
    """Test synthetic data creation."""

    def test_create_synthetic_data(self):
        """Test that synthetic data is created correctly."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            data_path = Path(f.name)

        try:
            create_synthetic_data(data_path, n_samples=50)

            assert data_path.exists()

            # Load and verify data
            import pandas as pd

            df = pd.read_parquet(data_path)

            assert len(df) == 50
            assert all(
                col in df.columns
                for col in ["timestamp", "open", "high", "low", "close", "volume"]
            )
            assert df["close"].notna().all()
            assert (df["volume"] > 0).all()

        finally:
            if data_path.exists():
                data_path.unlink()

    def test_create_synthetic_data_reproducibility(self):
        """Test that synthetic data is reproducible with same seed."""
        with (
            tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f1,
            tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f2,
        ):
            path1 = Path(f1.name)
            path2 = Path(f2.name)

        try:
            create_synthetic_data(path1, n_samples=25)
            create_synthetic_data(path2, n_samples=25)

            import pandas as pd

            df1 = pd.read_parquet(path1)
            df2 = pd.read_parquet(path2)

            # Should be identical due to same seed
            pd.testing.assert_frame_equal(df1, df2)

        finally:
            for path in [path1, path2]:
                if path.exists():
                    path.unlink()


class TestSmokeTests:
    """Test smoke test functions."""

    @patch("subprocess.run")
    def test_run_venue_health_check_smoke_test_success(self, mock_run):
        """Test venue health check smoke test success."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = run_venue_health_check_smoke_test()
        assert result is True

    @patch("subprocess.run")
    def test_run_venue_health_check_smoke_test_failure(self, mock_run):
        """Test venue health check smoke test graceful failure."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Network error"
        )

        result = run_venue_health_check_smoke_test()
        assert result is True  # Should be True for graceful failure

    @patch("subprocess.run")
    def test_run_venue_health_check_smoke_test_crash(self, mock_run):
        """Test venue health check smoke test crash."""
        mock_run.return_value = MagicMock(returncode=2, stdout="", stderr="Crash")

        result = run_venue_health_check_smoke_test()
        assert result is False

    @patch("subprocess.run")
    def test_run_paper_trader_smoke_test_success(self, mock_run):
        """Test paper trader smoke test success."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            data_path = Path(f.name)

        result = run_paper_trader_smoke_test(data_path)
        assert result is True

    @patch("subprocess.run")
    def test_run_paper_trader_smoke_test_failure(self, mock_run):
        """Test paper trader smoke test failure."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error")

        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            data_path = Path(f.name)

        result = run_paper_trader_smoke_test(data_path)
        assert result is False

    @patch("subprocess.run")
    def test_run_ppo_trainer_smoke_test_success(self, mock_run):
        """Test PPO trainer smoke test success."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Smoke test passed", stderr=""
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            data_path = Path(f.name)

        result = run_ppo_trainer_smoke_test(data_path)
        assert result is True

    @patch("subprocess.run")
    def test_run_ppo_trainer_smoke_test_failure(self, mock_run):
        """Test PPO trainer smoke test failure."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Import error"
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            data_path = Path(f.name)

        result = run_ppo_trainer_smoke_test(data_path)
        assert result is False
