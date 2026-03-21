#!/usr/bin/env python3
"""
Integration tests for SAC Algorithm with Model Compression.

Tests the integration of model compression techniques with the SAC algorithm,
including configuration validation, model creation, and compression application.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ztb.training.algorithms.sac.sac_algorithm import DEFAULT_SAC_CONFIG, SACAlgorithm


class MockEnv:
    """Mock environment for testing."""

    def __init__(self, observation_space_shape=(10,)):
        self.observation_space = Mock()
        self.observation_space.shape = observation_space_shape


class TestSACCompressionIntegration:
    """Integration tests for SAC algorithm with compression."""

    def test_compression_config_validation(self):
        """Test that compression configuration is properly validated."""
        sac = SACAlgorithm()

        # Valid compression config
        valid_config = {
            **DEFAULT_SAC_CONFIG,
            "compression_enabled": True,
            "compression_techniques": ["quantization"],
            "quantization_type": "dynamic",
        }

        assert sac.validate_config(valid_config)

        # Invalid compression techniques
        invalid_config = {
            **DEFAULT_SAC_CONFIG,
            "compression_enabled": True,
            "compression_techniques": ["invalid_technique"],
        }

        with pytest.raises(ValueError, match="Unsupported compression technique"):
            sac.validate_config(invalid_config)

    def test_compression_config_validation_quantization(self):
        """Test quantization configuration validation."""
        sac = SACAlgorithm()

        # Valid quantization config
        valid_config = {
            **DEFAULT_SAC_CONFIG,
            "compression_enabled": True,
            "compression_techniques": ["quantization"],
            "quantization_type": "dynamic",
        }

        assert sac.validate_config(valid_config)

        # Invalid quantization type
        invalid_config = {
            **DEFAULT_SAC_CONFIG,
            "compression_enabled": True,
            "compression_techniques": ["quantization"],
            "quantization_type": "invalid_type",
        }

        with pytest.raises(ValueError, match="Unsupported quantization_type"):
            sac.validate_config(invalid_config)

    def test_compression_config_validation_pruning(self):
        """Test pruning configuration validation."""
        sac = SACAlgorithm()

        # Valid pruning config
        valid_config = {
            **DEFAULT_SAC_CONFIG,
            "compression_enabled": True,
            "compression_techniques": ["pruning"],
            "pruning_type": "l1_unstructured",
            "pruning_amount": 0.3,
        }

        assert sac.validate_config(valid_config)

        # Invalid pruning type
        invalid_config = {
            **DEFAULT_SAC_CONFIG,
            "compression_enabled": True,
            "compression_techniques": ["pruning"],
            "pruning_type": "invalid_type",
        }

        with pytest.raises(ValueError, match="Unsupported pruning_type"):
            sac.validate_config(invalid_config)

        # Invalid pruning amount
        invalid_config_amount = {
            **DEFAULT_SAC_CONFIG,
            "compression_enabled": True,
            "compression_techniques": ["pruning"],
            "pruning_amount": 1.5,  # > 1.0
        }

        with pytest.raises(
            ValueError, match="pruning_amount must be between 0.0 and 1.0"
        ):
            sac.validate_config(invalid_config_amount)

    def test_compression_config_validation_distillation(self):
        """Test distillation configuration validation."""
        sac = SACAlgorithm()

        # Valid distillation config
        valid_config = {
            **DEFAULT_SAC_CONFIG,
            "compression_enabled": True,
            "compression_techniques": ["distillation"],
            "teacher_model_path": "/path/to/teacher/model.zip",
            "distillation_temperature": 2.0,
            "distillation_alpha": 0.5,
        }

        assert sac.validate_config(valid_config)

        # Missing teacher model path
        invalid_config = {
            **DEFAULT_SAC_CONFIG,
            "compression_enabled": True,
            "compression_techniques": ["distillation"],
            # teacher_model_path missing
        }

        with pytest.raises(ValueError, match="teacher_model_path not specified"):
            sac.validate_config(invalid_config)

        # Invalid distillation temperature
        invalid_temp_config = {
            **DEFAULT_SAC_CONFIG,
            "compression_enabled": True,
            "compression_techniques": ["distillation"],
            "teacher_model_path": "/path/to/teacher/model.zip",
            "distillation_temperature": -1.0,  # Invalid negative temperature
        }

        with pytest.raises(
            ValueError, match="distillation_temperature must be positive"
        ):
            sac.validate_config(invalid_temp_config)

        # Invalid distillation alpha
        invalid_alpha_config = {
            **DEFAULT_SAC_CONFIG,
            "compression_enabled": True,
            "compression_techniques": ["distillation"],
            "teacher_model_path": "/path/to/teacher/model.zip",
            "distillation_alpha": 1.5,  # > 1.0
        }

        with pytest.raises(
            ValueError, match="distillation_alpha must be between 0.0 and 1.0"
        ):
            sac.validate_config(invalid_alpha_config)

    @patch("ztb.training.algorithms.sac.sac_algorithm.SAC")
    def test_create_model_with_compression(self, mock_sac_class):
        """Test creating SAC model with compression enabled."""
        # Mock SAC model
        mock_sac_instance = Mock()
        mock_sac_instance.policy = Mock()
        mock_sac_instance.device = "cpu"
        mock_sac_class.return_value = mock_sac_instance

        sac = SACAlgorithm()
        env = MockEnv()

        config = {
            **DEFAULT_SAC_CONFIG,
            "compression_enabled": True,
            "compression_techniques": ["quantization"],
            "quantization_type": "dynamic",
            "compressed_model_path": None,
        }

        # Mock the compression manager
        with patch(
            "ztb.training.algorithms.sac.sac_algorithm.create_compression_pipeline"
        ) as mock_create_pipeline:
            mock_manager = Mock()
            mock_manager.compress_model.return_value = Mock()  # Compressed policy
            mock_create_pipeline.return_value = mock_manager

            model = sac.create_model(env, config)

            # Check that compression pipeline was created
            mock_create_pipeline.assert_called_once()
            pipeline_config = mock_create_pipeline.call_args[0][0]
            assert "quantization" in pipeline_config

            # Check that compression was applied
            mock_manager.compress_model.assert_called_once()

            # Check that compression manager was stored
            assert sac.compression_manager is mock_manager

    @patch("ztb.training.algorithms.sac.sac_algorithm.SAC")
    def test_create_model_compression_disabled(self, mock_sac_class):
        """Test creating SAC model with compression disabled."""
        # Mock SAC model
        mock_sac_instance = Mock()
        mock_sac_instance.policy = Mock()
        mock_sac_instance.device = "cpu"
        mock_sac_class.return_value = mock_sac_instance

        sac = SACAlgorithm()
        env = MockEnv()

        config = {**DEFAULT_SAC_CONFIG, "compression_enabled": False}

        model = sac.create_model(env, config)

        # Check that compression manager was not created
        assert sac.compression_manager is None

    @patch("ztb.training.algorithms.sac.sac_algorithm.SAC")
    def test_create_model_compression_save_path(self, mock_sac_class, tmp_path):
        """Test saving compressed model when path is specified."""
        # Mock SAC model
        mock_sac_instance = Mock()
        mock_sac_instance.policy = Mock()
        mock_sac_instance.device = "cpu"
        mock_sac_class.return_value = mock_sac_instance

        sac = SACAlgorithm()
        env = MockEnv()

        save_path = tmp_path / "compressed_model.zip"

        config = {
            **DEFAULT_SAC_CONFIG,
            "compression_enabled": True,
            "compression_techniques": ["quantization"],
            "quantization_type": "dynamic",
            "compressed_model_path": str(save_path),
        }

        with patch(
            "ztb.training.algorithms.sac.sac_algorithm.create_compression_pipeline"
        ) as mock_create_pipeline:
            mock_manager = Mock()
            mock_manager.compress_model.return_value = Mock()
            mock_create_pipeline.return_value = mock_manager

            model = sac.create_model(env, config)

            # Check that save_compressed_model was called
            mock_manager.save_compressed_model.assert_called_once_with(
                mock_sac_instance, str(save_path)
            )

    @patch("ztb.training.algorithms.sac.sac_algorithm.SAC")
    def test_apply_model_compression_quantization_only(self, mock_sac_class):
        """Test applying quantization compression."""
        # Mock SAC model
        mock_sac_instance = Mock()
        mock_sac_instance.policy = Mock()
        mock_sac_instance.device = "cpu"
        mock_sac_class.return_value = mock_sac_instance

        sac = SACAlgorithm()
        env = MockEnv()

        config = {
            **DEFAULT_SAC_CONFIG,
            "compression_enabled": True,
            "compression_techniques": ["quantization"],
            "quantization_type": "dynamic",
        }

        with patch(
            "ztb.training.algorithms.sac.sac_algorithm.create_compression_pipeline"
        ) as mock_create_pipeline:
            mock_manager = Mock()
            compressed_policy = Mock()
            mock_manager.compress_model.return_value = compressed_policy
            mock_manager.get_compression_report.return_value = {"test": "report"}
            mock_create_pipeline.return_value = mock_manager

            model = sac.create_model(env, config)

            # Check that quantization config was passed correctly
            call_args = mock_create_pipeline.call_args[0][0]
            assert "quantization" in call_args
            assert call_args["quantization"]["type"] == "quantization"
            assert call_args["quantization"]["quantization_type"] == "dynamic"

    @patch("ztb.training.algorithms.sac.sac_algorithm.SAC")
    def test_apply_model_compression_pruning_only(self, mock_sac_class):
        """Test applying pruning compression."""
        # Mock SAC model
        mock_sac_instance = Mock()
        mock_sac_instance.policy = Mock()
        mock_sac_instance.device = "cpu"
        mock_sac_class.return_value = mock_sac_instance

        sac = SACAlgorithm()
        env = MockEnv()

        config = {
            **DEFAULT_SAC_CONFIG,
            "compression_enabled": True,
            "compression_techniques": ["pruning"],
            "pruning_type": "l1_unstructured",
            "pruning_amount": 0.25,
        }

        with patch(
            "ztb.training.algorithms.sac.sac_algorithm.create_compression_pipeline"
        ) as mock_create_pipeline:
            mock_manager = Mock()
            compressed_policy = Mock()
            mock_manager.compress_model.return_value = compressed_policy
            mock_manager.get_compression_report.return_value = {"test": "report"}
            mock_create_pipeline.return_value = mock_manager

            model = sac.create_model(env, config)

            # Check that pruning config was passed correctly
            call_args = mock_create_pipeline.call_args[0][0]
            assert "pruning" in call_args
            assert call_args["pruning"]["type"] == "pruning"
            assert call_args["pruning"]["pruning_type"] == "l1_unstructured"
            assert call_args["pruning"]["amount"] == 0.25

    @patch("ztb.training.algorithms.sac.sac_algorithm.SAC.load")
    @patch("ztb.training.algorithms.sac.sac_algorithm.SAC")
    def test_apply_model_compression_distillation_only(
        self, mock_sac_class, mock_sac_load, tmp_path
    ):
        """Test applying distillation compression."""
        # Mock teacher model
        mock_teacher = Mock()
        mock_teacher.device = "cpu"
        mock_sac_load.return_value = mock_teacher

        # Mock SAC model
        mock_sac_instance = Mock()
        mock_sac_instance.policy = Mock()
        mock_sac_instance.device = "cpu"
        mock_sac_class.return_value = mock_sac_instance

        sac = SACAlgorithm()
        env = MockEnv()

        teacher_path = tmp_path / "teacher_model.zip"

        config = {
            **DEFAULT_SAC_CONFIG,
            "compression_enabled": True,
            "compression_techniques": ["distillation"],
            "teacher_model_path": str(teacher_path),
            "distillation_temperature": 2.5,
            "distillation_alpha": 0.7,
        }

        with patch(
            "ztb.training.algorithms.sac.sac_algorithm.create_compression_pipeline"
        ) as mock_create_pipeline:
            mock_manager = Mock()
            compressed_policy = Mock()
            mock_manager.compress_model.return_value = compressed_policy
            mock_manager.get_compression_report.return_value = {"test": "report"}
            mock_create_pipeline.return_value = mock_manager

            model = sac.create_model(env, config)

            # Check that teacher model was loaded
            mock_sac_load.assert_called_once_with(str(teacher_path), device="cpu")

            # Check that distillation config was passed correctly
            call_args = mock_create_pipeline.call_args[0][0]
            assert "distillation" in call_args
            assert call_args["distillation"]["type"] == "distillation"
            assert call_args["distillation"]["temperature"] == 2.5
            assert call_args["distillation"]["alpha"] == 0.7

    @patch("ztb.training.algorithms.sac.sac_algorithm.SAC")
    def test_apply_model_compression_multiple_techniques(self, mock_sac_class):
        """Test applying multiple compression techniques."""
        # Mock SAC model
        mock_sac_instance = Mock()
        mock_sac_instance.policy = Mock()
        mock_sac_instance.device = "cpu"
        mock_sac_class.return_value = mock_sac_instance

        sac = SACAlgorithm()
        env = MockEnv()

        config = {
            **DEFAULT_SAC_CONFIG,
            "compression_enabled": True,
            "compression_techniques": ["quantization", "pruning"],
            "quantization_type": "dynamic",
            "pruning_type": "l1_unstructured",
            "pruning_amount": 0.2,
        }

        with patch(
            "ztb.training.algorithms.sac.sac_algorithm.create_compression_pipeline"
        ) as mock_create_pipeline:
            mock_manager = Mock()
            compressed_policy = Mock()
            mock_manager.compress_model.return_value = compressed_policy
            mock_manager.get_compression_report.return_value = {"test": "report"}
            mock_create_pipeline.return_value = mock_manager

            model = sac.create_model(env, config)

            # Check that both techniques were configured
            call_args = mock_create_pipeline.call_args[0][0]
            assert "quantization" in call_args
            assert "pruning" in call_args

            # Check that compress_model was called with both techniques
            compress_call_args = mock_manager.compress_model.call_args
            techniques_arg = compress_call_args[0][1]  # Second argument
            assert "quantization" in techniques_arg
            assert "pruning" in techniques_arg

    @patch("ztb.training.algorithms.sac.sac_algorithm.SAC")
    def test_apply_model_compression_no_techniques(self, mock_sac_class):
        """Test compression with no techniques specified."""
        # Mock SAC model
        mock_sac_instance = Mock()
        mock_sac_instance.policy = Mock()
        mock_sac_instance.device = "cpu"
        mock_sac_class.return_value = mock_sac_instance

        sac = SACAlgorithm()
        env = MockEnv()

        config = {
            **DEFAULT_SAC_CONFIG,
            "compression_enabled": True,
            "compression_techniques": [],
        }  # Empty techniques

        # Should not apply compression but also not fail
        model = sac.create_model(env, config)

        # Compression manager should still be None since no techniques were applied
        assert sac.compression_manager is None

    def test_sac_algorithm_compression_manager_initialization(self):
        """Test that SAC algorithm initializes compression manager as None."""
        sac = SACAlgorithm()

        assert sac.compression_manager is None

        # After initialization, it should still be None
        assert sac.compression_manager is None


if __name__ == "__main__":
    pytest.main([__file__])
