"""Unit tests for run seal reproducibility."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from ztb.training.run_seal import EnvironmentSnapshot, RunSeal, RunSealManager


class TestRunSealManager:
    """Test run seal management functionality."""

    def test_create_seal(self):
        """Test creating a run seal."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RunSealManager(temp_dir)

            config = {"learning_rate": 0.001, "batch_size": 32}
            metadata = {"experiment": "test_run"}

            seal = manager.create_seal(seed=42, config=config, metadata=metadata)

            assert seal.run_id.startswith("run_")
            assert seal.seed == 42
            assert seal.config == config
            assert seal.metadata == metadata
            assert isinstance(seal.environment, EnvironmentSnapshot)

            # Check seal was saved
            seal_files = list(Path(temp_dir).glob("*.json"))
            assert len(seal_files) == 1

    def test_load_seal(self):
        """Test loading a run seal."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RunSealManager(temp_dir)

            # Create and save seal
            original_seal = manager.create_seal(seed=123)

            # Load seal
            loaded_seal = manager.load_seal(original_seal.run_id)

            assert loaded_seal is not None
            assert loaded_seal.run_id == original_seal.run_id
            assert loaded_seal.seed == original_seal.seed
            assert loaded_seal.config == original_seal.config

    def test_load_nonexistent_seal(self):
        """Test loading a nonexistent seal."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RunSealManager(temp_dir)

            assert manager.load_seal("nonexistent") is None

    def test_list_seals(self):
        """Test listing run seals."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RunSealManager(temp_dir)

            # Create multiple seals
            seal1 = manager.create_seal()
            seal2 = manager.create_seal()

            seals = manager.list_seals()
            assert len(seals) == 2
            assert seal1.run_id in seals
            assert seal2.run_id in seals

    @patch("subprocess.check_output")
    @patch("platform.python_version")
    @patch("platform.platform")
    @patch("platform.node")
    def test_validate_environment(
        self, mock_node, mock_platform, mock_python, mock_subprocess
    ):
        """Test environment validation."""
        mock_python.return_value = "3.11.0"
        mock_platform.return_value = "Linux-5.4.0"
        mock_node.return_value = "testhost"
        mock_subprocess.return_value = "abc123"

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RunSealManager(temp_dir)

            # Create seal
            seal = manager.create_seal()

            # Validate with same environment
            validation = manager.validate_environment(seal)
            assert validation["python_version"] is True
            assert validation["platform"] is True

    def test_run_seal_to_dict(self):
        """Test RunSeal to_dict conversion."""
        environment = EnvironmentSnapshot(
            python_version="3.11.0",
            platform="Linux",
            hostname="test",
            user="testuser",
            working_directory="/tmp",
        )

        seal = RunSeal(
            run_id="test_run",
            seed=42,
            timestamp="2023-01-01T00:00:00",
            environment=environment,
            config={"test": "config"},
            metadata={"test": "meta"},
        )

        data = seal.to_dict()
        assert data["run_id"] == "test_run"
        assert data["seed"] == 42
        assert data["config"] == {"test": "config"}
        assert data["metadata"] == {"test": "meta"}
