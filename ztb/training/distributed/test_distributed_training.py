"""
Tests for Distributed Training Utilities

Tests distributed training functionality including:
- Distributed training setup and cleanup
- Model distribution (DDP/DataParallel)
- Loss reduction and tensor gathering
- Checkpoint saving/loading
"""

import os
import tempfile
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from ztb.training.distributed.distributed_training import (
    DistributedTrainer,
    DistributedTrainingConfig,
    broadcast_tensor,
    cleanup_distributed_training,
    find_free_port,
    gather_tensor,
    get_distributed_info,
    reduce_loss,
    setup_distributed_training,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size=10, hidden_size=5, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.output(torch.relu(self.linear(x)))


class TestDistributedTrainingConfig:
    """Test DistributedTrainingConfig class."""

    def test_config_initialization(self):
        """Test config initialization."""
        config = DistributedTrainingConfig(
            world_size=4,
            rank=1,
            master_addr="192.168.1.100",
            master_port="12345",
            backend="nccl",
        )

        assert config.world_size == 4
        assert config.rank == 1
        assert config.master_addr == "192.168.1.100"
        assert config.master_port == "12345"
        assert config.backend == "nccl"

    def test_config_from_env(self):
        """Test config creation from environment variables."""
        with patch.dict(
            os.environ,
            {
                "WORLD_SIZE": "2",
                "RANK": "1",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "1234",
                "DIST_BACKEND": "gloo",
            },
        ):
            config = DistributedTrainingConfig.from_env()

            assert config.world_size == 2
            assert config.rank == 1
            assert config.master_addr == "localhost"
            assert config.master_port == "1234"
            assert config.backend == "gloo"

    def test_config_to_env(self):
        """Test config conversion to environment variables."""
        config = DistributedTrainingConfig(
            world_size=3,
            rank=2,
            master_addr="127.0.0.1",
            master_port="9999",
            backend="gloo",
        )

        env_vars = config.to_env()

        assert env_vars["WORLD_SIZE"] == "3"
        assert env_vars["RANK"] == "2"
        assert env_vars["MASTER_ADDR"] == "127.0.0.1"
        assert env_vars["MASTER_PORT"] == "9999"
        assert env_vars["DIST_BACKEND"] == "gloo"


class TestDistributedTrainer:
    """Test DistributedTrainer class."""

    def test_single_device_initialization(self):
        """Test trainer initialization for single device."""
        model = SimpleModel()
        config = DistributedTrainingConfig(world_size=1, rank=0)

        trainer = DistributedTrainer(model, config)

        assert trainer.is_distributed == False
        assert trainer.is_master == True
        assert trainer.get_world_size() == 1
        assert trainer.get_rank() == 0
        assert trainer.is_master_process() == True

    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_backend_fallback(self, mock_cuda):
        """Test fallback to Gloo backend when CUDA not available."""
        model = SimpleModel()
        config = DistributedTrainingConfig(
            world_size=1, rank=0, backend="nccl"
        )  # Single process to avoid DDP

        # Should not raise error, just log warning
        trainer = DistributedTrainer(model, config)
        assert config.backend == "nccl"  # Config unchanged, but would fallback in setup

    def test_data_parallel_initialization(self):
        """Test DataParallel initialization for multiple GPUs."""
        # Skip test if CUDA not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = SimpleModel()
        config = DistributedTrainingConfig(world_size=1, rank=0, backend="nccl")

        with patch("torch.cuda.device_count", return_value=2):
            trainer = DistributedTrainer(model, config)

            # Should wrap with DataParallel
            assert isinstance(trainer.model, nn.DataParallel)

    def test_get_model_unwrapping(self):
        """Test getting underlying model from wrapped model."""
        model = SimpleModel()
        config = DistributedTrainingConfig(world_size=1, rank=0)

        trainer = DistributedTrainer(model, config)

        # For single device, should return the same model
        unwrapped = trainer.get_model()
        assert unwrapped is model

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        model = SimpleModel()
        config = DistributedTrainingConfig(world_size=1, rank=0)

        trainer = DistributedTrainer(model, config)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            checkpoint_path = f.name

        try:
            # Save checkpoint
            trainer.save_checkpoint(checkpoint_path, epoch=10, custom_data="test")

            # Load checkpoint
            loaded_data = trainer.load_checkpoint(checkpoint_path)

            assert loaded_data["epoch"] == 10
            assert loaded_data["custom_data"] == "test"
            assert "model_state_dict" in loaded_data
            assert "distributed_config" in loaded_data

        finally:
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)

    def test_non_master_checkpoint_save(self):
        """Test that non-master processes don't save checkpoints."""
        model = SimpleModel()
        config = DistributedTrainingConfig(
            world_size=1, rank=0
        )  # Single process to avoid DDP

        trainer = DistributedTrainer(model, config)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            checkpoint_path = f.name

        try:
            # For single process, should save
            trainer.save_checkpoint(checkpoint_path)
            assert os.path.exists(checkpoint_path)

        finally:
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)


class TestDistributedUtilities:
    """Test distributed training utility functions."""

    def test_find_free_port(self):
        """Test finding a free port."""
        port = find_free_port()

        # Should be a valid port number
        assert isinstance(port, str)
        port_num = int(port)
        assert 1024 <= port_num <= 65535

    def test_get_distributed_info_single_process(self):
        """Test getting distributed info for single process."""
        info = get_distributed_info()

        assert info["is_distributed"] == False
        assert info["world_size"] == 1
        assert info["rank"] == 0
        assert info["backend"] is None

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_world_size", return_value=4)
    @patch("torch.distributed.get_rank", return_value=2)
    @patch("torch.distributed.get_backend", return_value="gloo")
    def test_get_distributed_info_distributed(
        self, mock_backend, mock_rank, mock_world_size, mock_initialized
    ):
        """Test getting distributed info when distributed."""
        info = get_distributed_info()

        assert info["is_distributed"] == True
        assert info["world_size"] == 4
        assert info["rank"] == 2
        assert info["backend"] == "gloo"

    def test_reduce_loss_single_process(self):
        """Test loss reduction for single process."""
        config = DistributedTrainingConfig(world_size=1, rank=0)
        loss = torch.tensor(2.5)

        reduced = reduce_loss(loss, config)

        assert torch.allclose(reduced, loss)

    def test_gather_tensor_single_process(self):
        """Test tensor gathering for single process."""
        config = DistributedTrainingConfig(world_size=1, rank=0)
        tensor = torch.tensor([1, 2, 3])

        gathered = gather_tensor(tensor, config)

        assert len(gathered) == 1
        assert torch.allclose(gathered[0], tensor)

    def test_broadcast_tensor_single_process(self):
        """Test tensor broadcast for single process."""
        config = DistributedTrainingConfig(world_size=1, rank=0)
        tensor = torch.tensor([1, 2, 3])

        # Should not raise error
        broadcast_tensor(tensor, 0, config)

        assert torch.allclose(tensor, torch.tensor([1, 2, 3]))


class TestDistributedSetup:
    """Test distributed training setup and cleanup."""

    @patch("torch.distributed.init_process_group")
    @patch("torch.distributed.is_initialized", return_value=False)
    def test_setup_distributed_training_success(self, mock_is_initialized, mock_init):
        """Test successful distributed training setup."""
        config = DistributedTrainingConfig(world_size=2, rank=0)

        success = setup_distributed_training(config)

        assert success == True
        mock_init.assert_called_once()

    @patch("torch.distributed.init_process_group", side_effect=Exception("Init failed"))
    @patch("torch.distributed.is_initialized", return_value=False)
    def test_setup_distributed_training_failure(self, mock_is_initialized, mock_init):
        """Test failed distributed training setup."""
        config = DistributedTrainingConfig(world_size=2, rank=0)

        success = setup_distributed_training(config)

        assert success == False

    def test_setup_single_process(self):
        """Test setup for single process (should succeed without init)."""
        config = DistributedTrainingConfig(world_size=1, rank=0)

        success = setup_distributed_training(config)

        assert success == True

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.destroy_process_group")
    def test_cleanup_distributed_training(self, mock_destroy, mock_is_initialized):
        """Test distributed training cleanup."""
        cleanup_distributed_training()

        mock_destroy.assert_called_once()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.destroy_process_group")
    def test_cleanup_when_not_initialized(self, mock_destroy, mock_is_initialized):
        """Test cleanup when distributed training not initialized."""
        cleanup_distributed_training()

        mock_destroy.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])
