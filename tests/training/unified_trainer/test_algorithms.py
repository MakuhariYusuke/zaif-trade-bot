"""
Unit Tests for Unified Trainer Algorithms
Unified Trainerアルゴリズムの単体テスト

This module contains unit tests for the unified trainer algorithms,
including the newly added SelfSupervisedTrainer.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

from ztb.training.unified_trainer.algorithms import (
    PPOTrainer,
    SACTrainer,
    SelfSupervisedTrainer,
    create_algorithm_trainer,
)


_HAS_TRANSFORMER_ENCODER = hasattr(torch.nn, "TransformerEncoderLayer")


class _FakeIntegrationSSPTrainer:
    def __init__(
        self,
        input_dim: int,
        device: str,
        checkpoint_dir: str,
        memory_manager: object | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.memory_manager = memory_manager
        self.training_history = {}

    def train_all_stages(self, train_data, val_data, config) -> None:
        self.training_history = {
            "mpm": {"epochs": [1]},
            "train_shape": tuple(getattr(train_data, "shape", ())),
            "val_shape": tuple(getattr(val_data, "shape", ())),
            "config_keys": sorted(config.keys()),
        }

    def save_checkpoint(self, path: str | None = None) -> None:
        target = path or os.path.join(self.checkpoint_dir, "fake_checkpoint.pth")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as handle:
            handle.write(b"fake-ssp-checkpoint")

    def save_training_history(self) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        with open(
            os.path.join(self.checkpoint_dir, "training_history.txt"),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write("ok")

    def get_pretrained_encoders(self) -> dict[str, str]:
        return {"mpm_encoder": "encoder"}


def _make_tiny_ssp_config(temp_dir: str, **overrides):
    """Create a minimal but real SSP config for integration coverage."""
    config = {
        "input_dim": 156,
        "device": "cpu",
        "checkpoint_dir": os.path.join(temp_dir, "checkpoints"),
        "config_type": "lightweight",
        "synthetic_batch_size": 4,
        "synthetic_val_batch_size": 2,
        "seq_len": 8,
        "training": {
            "ssp_hyperparameters": {
                "learning_rate": 1e-3,
                "batch_size": 2,
                "num_epochs": 1,
                "patience": 1,
                "save_best": False,
                "seq_len": 8,
            }
        },
        "custom_config": {
            "mpm": {
                "hidden_dim": 32,
                "num_layers": 1,
                "num_heads": 2,
                "max_seq_len": 8,
                "dropout": 0.1,
            },
            "mpm_training": {
                "epochs": 1,
                "batch_size": 2,
                "patience": 1,
                "save_best": False,
            },
            "contrastive": {
                "hidden_dim": 32,
                "projection_dim": 16,
                "learning_rate": 1e-3,
            },
            "contrastive_training": {
                "epochs": 1,
                "batch_size": 2,
                "patience": 1,
                "save_best": False,
            },
            "anomaly": {
                "hidden_dims": [16, 8],
                "latent_dim": 4,
                "lstm_hidden_dim": 8,
                "lstm_num_layers": 1,
                "seq_len": 8,
                "learning_rate": 1e-3,
            },
            "anomaly_training": {
                "epochs": 1,
                "batch_size": 2,
                "patience": 1,
                "save_best": False,
            },
        },
    }
    config.update(overrides)
    return config


class TestCreateAlgorithmTrainer:
    """Test the create_algorithm_trainer factory function"""

    def test_create_sac_trainer(self):
        """Test creating SAC trainer"""
        config = {"environment": "test_env"}
        trainer = create_algorithm_trainer("sac", config)
        assert isinstance(trainer, SACTrainer)

    def test_create_ppo_trainer(self):
        """Test creating PPO trainer"""
        config = {"environment": "test_env"}
        trainer = create_algorithm_trainer("ppo", config)
        assert isinstance(trainer, PPOTrainer)

    def test_create_self_supervised_trainer(self):
        """Test creating self-supervised trainer"""
        config = {"input_dim": 156, "device": "cpu"}
        trainer = create_algorithm_trainer("self_supervised", config)
        assert isinstance(trainer, SelfSupervisedTrainer)

    def test_create_unsupported_algorithm(self):
        """Test creating unsupported algorithm raises error"""
        config = {}
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            create_algorithm_trainer("unsupported", config)

    def test_case_insensitive_algorithm(self):
        """Test algorithm names are case insensitive"""
        config = {"input_dim": 156, "device": "cpu"}
        trainer = create_algorithm_trainer("SELF_SUPERVISED", config)
        assert isinstance(trainer, SelfSupervisedTrainer)


class TestSelfSupervisedTrainer:
    """Unit tests for SelfSupervisedTrainer"""

    @pytest.fixture
    def basic_config(self):
        """Basic configuration for self-supervised trainer"""
        return {
            "input_dim": 156,
            "device": "cpu",
            "checkpoint_dir": "test_checkpoints",
            "config_type": "lightweight",
        }

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data"""
        batch_size, seq_len, input_dim = 10, 50, 156
        return torch.randn(batch_size, seq_len, input_dim)

    def test_initialization(self, basic_config):
        """Test SelfSupervisedTrainer initialization"""
        trainer = SelfSupervisedTrainer(basic_config)
        assert trainer.config == basic_config
        assert trainer.ssp_trainer is None
        assert trainer.train_data is None
        assert trainer.val_data is None

    def test_validate_config_valid(self, basic_config):
        """Test configuration validation with valid config"""
        trainer = SelfSupervisedTrainer(basic_config)
        assert trainer.validate_config() is True

    def test_validate_config_missing_input_dim(self):
        """Test configuration validation with missing input_dim"""
        config = {"device": "cpu"}
        trainer = SelfSupervisedTrainer(config)
        assert trainer.validate_config() is False

    def test_validate_config_missing_device(self):
        """Test configuration validation with missing device"""
        config = {"input_dim": 156}
        trainer = SelfSupervisedTrainer(config)
        assert trainer.validate_config() is False

    @patch("os.path.exists")
    def test_validate_config_invalid_train_data_path(self, mock_exists, basic_config):
        """Test configuration validation with invalid training data path"""
        mock_exists.return_value = False
        config = basic_config.copy()
        config["train_data_path"] = "nonexistent.csv"

        trainer = SelfSupervisedTrainer(config)
        assert trainer.validate_config() is False

    @patch("os.path.exists")
    def test_validate_config_invalid_val_data_path(self, mock_exists, basic_config):
        """Test configuration validation with invalid validation data path"""
        mock_exists.return_value = False
        config = basic_config.copy()
        config["val_data_path"] = "nonexistent.csv"

        trainer = SelfSupervisedTrainer(config)
        assert trainer.validate_config() is False

    @patch(
        "ztb.training.unified_trainer.algorithms.self_supervised_trainer.DataLoader.load_csv_strict"
    )
    def test_load_data_synthetic(self, mock_load_csv, basic_config):
        """Test loading synthetic data when no data paths provided"""
        trainer = SelfSupervisedTrainer(basic_config)
        assert trainer._load_data() is True
        assert trainer.train_data is not None
        assert trainer.val_data is not None
        assert trainer.train_data.shape[1:] == (100, 156)  # seq_len, input_dim
        assert trainer.val_data.shape[1:] == (100, 156)
        mock_load_csv.assert_not_called()

    @patch(
        "ztb.training.unified_trainer.algorithms.self_supervised_trainer.DataLoader.load_csv_strict"
    )
    def test_load_data_custom_sizes(self, mock_load_csv):
        """Test loading synthetic data with custom sizes"""
        config = {
            "input_dim": 156,
            "device": "cpu",
            "synthetic_batch_size": 20,
            "synthetic_val_batch_size": 5,
            "seq_len": 75,
        }
        trainer = SelfSupervisedTrainer(config)
        assert trainer._load_data() is True
        assert trainer.train_data.shape == (20, 75, 156)
        assert trainer.val_data.shape == (5, 75, 156)
        mock_load_csv.assert_not_called()

    @patch("pandas.read_csv")
    @patch(
        "ztb.training.unified_trainer.algorithms.self_supervised_trainer.DataLoader.load_csv_strict"
    )
    def test_load_data_from_csv(
        self, mock_load_csv, mock_read_csv, basic_config, sample_data
    ):
        """Test loading data from CSV files"""
        # Mock CSV data
        mock_df = pd.DataFrame(sample_data[0].numpy())  # First sample
        mock_read_csv.return_value = mock_df
        mock_load_csv.return_value = mock_df

        config = basic_config.copy()
        config["train_data_path"] = "train.csv"
        config["val_data_path"] = "val.csv"

        trainer = SelfSupervisedTrainer(config)
        assert trainer._load_data() is True

        # Check that data was loaded and reshaped
        assert trainer.train_data is not None
        assert trainer.val_data is not None

    @patch("ztb.multimodal.pretraining.SelfSupervisedTrainer")
    @patch.object(SelfSupervisedTrainer, "_load_data")
    def test_train_success(
        self, mock_load_data, mock_ssp_trainer, basic_config, sample_data
    ):
        """Test successful training execution"""
        mock_load_data.return_value = True

        # Mock the SSP trainer
        mock_instance = MagicMock()
        mock_ssp_trainer.return_value = mock_instance

        trainer = SelfSupervisedTrainer(basic_config)
        result = trainer.train()

        assert result is True
        mock_load_data.assert_called_once()
        mock_ssp_trainer.assert_called_once()
        mock_instance.train_all_stages.assert_called_once()
        mock_instance.save_checkpoint.assert_called_once()
        mock_instance.save_training_history.assert_called_once()

    @patch.object(SelfSupervisedTrainer, "_load_data")
    def test_train_data_load_failure(self, mock_load_data, basic_config):
        """Test training failure when data loading fails"""
        mock_load_data.return_value = False

        trainer = SelfSupervisedTrainer(basic_config)
        result = trainer.train()

        assert result is False

    @patch("ztb.multimodal.pretraining.SelfSupervisedTrainer")
    @patch.object(SelfSupervisedTrainer, "_load_data")
    def test_train_exception_handling(
        self, mock_load_data, mock_ssp_trainer, basic_config
    ):
        """Test exception handling during training"""
        mock_load_data.return_value = True
        mock_instance = MagicMock()
        mock_instance.train_all_stages.side_effect = Exception("Training failed")
        mock_ssp_trainer.return_value = mock_instance

        trainer = SelfSupervisedTrainer(basic_config)
        result = trainer.train()

        assert result is False

    def test_get_training_stats_no_trainer(self, basic_config):
        """Test getting training stats when no trainer initialized"""
        trainer = SelfSupervisedTrainer(basic_config)
        stats = trainer.get_training_stats()
        assert stats["status"] == "partial"
        assert "data_shapes" in stats
        assert stats["data_shapes"] == {"train": None, "val": None}

    @patch.object(SelfSupervisedTrainer, "_load_data")
    @patch("ztb.multimodal.pretraining.SelfSupervisedTrainer")
    def test_get_training_stats_with_trainer(
        self, mock_ssp_trainer, mock_load_data, basic_config, sample_data
    ):
        """Test getting training stats when trainer is initialized"""
        mock_load_data.return_value = True

        # Mock SSP trainer
        mock_instance = MagicMock()
        mock_instance.training_history = {"mpm": {"epochs": [1, 2]}}
        mock_instance.get_pretrained_encoders.return_value = {"mpm_encoder": "encoder"}
        mock_ssp_trainer.return_value = mock_instance

        trainer = SelfSupervisedTrainer(basic_config)
        trainer.train()  # This will initialize the trainer

        stats = trainer.get_training_stats()

        assert "training_history" in stats
        assert "encoders_available" in stats
        assert "data_shapes" in stats

    @patch.object(SelfSupervisedTrainer, "save_model")
    @patch.object(SelfSupervisedTrainer, "_load_data")
    @patch("ztb.multimodal.pretraining.SelfSupervisedTrainer")
    def test_get_training_stats_survive_save_failure(
        self,
        mock_ssp_trainer,
        mock_load_data,
        mock_save_model,
        basic_config,
    ):
        """Late persistence failures should not discard collected training stats."""
        mock_load_data.return_value = True
        mock_save_model.side_effect = RuntimeError("disk full")

        mock_instance = MagicMock()
        mock_instance.training_history = {"mpm": {"epochs": [1]}}
        mock_instance.get_pretrained_encoders.return_value = {"mpm_encoder": "encoder"}
        mock_ssp_trainer.return_value = mock_instance

        trainer = SelfSupervisedTrainer(basic_config)
        assert trainer.train() is False

        stats = trainer.get_training_stats()
        assert stats["status"] == "trained"
        assert "training_history" in stats
        assert "data_shapes" in stats

    @patch("ztb.multimodal.pretraining.SelfSupervisedTrainer")
    def test_load_model_uses_effective_config(self, mock_ssp_trainer):
        """Test load_model uses the merged SSP config rather than nested defaults only."""
        mock_instance = MagicMock()
        mock_ssp_trainer.return_value = mock_instance

        trainer = SelfSupervisedTrainer(
            {
                "input_dim": 156,
                "device": "cpu",
                "checkpoint_dir": "custom_checkpoints",
                "config_type": "lightweight",
                "custom_config": {"mpm": {"hidden_dim": 32}},
            }
        )

        assert trainer.load_model("checkpoint.pth") is True
        mock_ssp_trainer.assert_called_once_with(
            input_dim=156,
            device="cpu",
            checkpoint_dir="custom_checkpoints",
            memory_manager=None,
        )
        mock_instance.load_checkpoint.assert_called_once_with("checkpoint.pth")

    @patch("os.path.exists")
    def test_validate_config_with_valid_data_paths(self, mock_exists, basic_config):
        """Test configuration validation with valid data paths"""
        mock_exists.return_value = True
        config = basic_config.copy()
        config["train_data_path"] = "train.csv"
        config["val_data_path"] = "val.csv"

        trainer = SelfSupervisedTrainer(config)
        assert trainer.validate_config() is True

    def test_custom_config_override(self, basic_config):
        """Test custom configuration override"""
        config = basic_config.copy()
        config["custom_config"] = {
            "mpm": {"hidden_dim": 512},
            "training": {"epochs": 50},
        }

        trainer = SelfSupervisedTrainer(config)

        # This would normally be tested in train() method
        # but we're testing the config structure here
        assert "custom_config" in trainer.config

    def test_build_ssp_model_config_applies_config_type_and_custom_overrides(self):
        """Test that SSP config composition honors lightweight and nested overrides."""
        trainer = SelfSupervisedTrainer(
            {
                "input_dim": 156,
                "device": "cpu",
                "config_type": "lightweight",
                "seq_len": 12,
                "training": {
                    "ssp_hyperparameters": {
                        "learning_rate": 1e-3,
                        "batch_size": 2,
                        "num_epochs": 1,
                        "patience": 1,
                        "save_best": False,
                    }
                },
                "custom_config": {
                    "mpm": {"hidden_dim": 32},
                    "contrastive": {"projection_dim": 16},
                },
            }
        )

        effective_config = trainer._build_ssp_model_config()

        assert effective_config["mpm"]["hidden_dim"] == 32
        assert effective_config["mpm"]["max_seq_len"] == 12
        assert effective_config["contrastive"]["projection_dim"] == 16
        assert effective_config["mpm_training"]["epochs"] == 1
        assert effective_config["mpm_training"]["batch_size"] == 2
        assert effective_config["anomaly_training"]["save_best"] is False

    @pytest.mark.parametrize("config_type", ["default", "lightweight", "production"])
    def test_different_config_types(self, config_type):
        """Test different configuration types"""
        config = {"input_dim": 156, "device": "cpu", "config_type": config_type}

        trainer = SelfSupervisedTrainer(config)
        assert trainer.validate_config() is True


class TestSelfSupervisedTrainerIntegration:
    """Integration tests for SelfSupervisedTrainer"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.mark.skipif(
        not _HAS_TRANSFORMER_ENCODER,
        reason="Self-supervised integration requires full torch.nn transformer support.",
    )
    @patch(
        "ztb.multimodal.pretraining.SelfSupervisedTrainer",
        _FakeIntegrationSSPTrainer,
    )
    def test_full_training_pipeline(self, temp_dir):
        """Test the complete training pipeline"""
        config = _make_tiny_ssp_config(temp_dir)

        trainer = SelfSupervisedTrainer(config)

        # Validate config
        assert trainer.validate_config() is True

        # Training should succeed (though it will use synthetic data)
        result = trainer.train()
        assert result is True

        # Check that checkpoints were created
        checkpoint_files = os.listdir(config["checkpoint_dir"])
        assert len(checkpoint_files) > 0

    @pytest.mark.skipif(
        not _HAS_TRANSFORMER_ENCODER,
        reason="Self-supervised integration requires full torch.nn transformer support.",
    )
    @patch(
        "ztb.multimodal.pretraining.SelfSupervisedTrainer",
        _FakeIntegrationSSPTrainer,
    )
    def test_training_stats_after_training(self, temp_dir):
        """Test training statistics after successful training"""
        config = _make_tiny_ssp_config(temp_dir)

        trainer = SelfSupervisedTrainer(config)
        trainer.train()

        stats = trainer.get_training_stats()

        # Verify stats structure
        assert isinstance(stats, dict)
        assert "training_history" in stats
        assert "encoders_available" in stats
        assert "data_shapes" in stats

        # Check data shapes
        assert stats["data_shapes"]["train"] is not None
        assert stats["data_shapes"]["val"] is not None
