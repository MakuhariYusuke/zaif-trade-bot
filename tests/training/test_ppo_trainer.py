#!/usr/bin/env python3
"""
Unit tests for PPO Trainer implementations.

Tests cover:
- PPOTrainerAutoHalt initialization and validation
- Environment and model creation
- Training execution paths
- Error handling and edge cases
- Policy bias neutralization
"""

# mypy: disable-error-code="no-untyped-def,arg-type,attr-defined,var-annotated,union-attr,import-untyped,no-any-return,misc"
# ci: trigger python-validation (noop)

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
import pandas as pd

from ztb.training.trainers.ppo_trainer import PPOAlgorithmTrainer

if TYPE_CHECKING:
    from ztb.training.config.trainer_params import TrainerParams
    from ztb.training.core.ppo_trainer import (
        PPOTrainer,
        PPOTrainerAutoHalt,
        TrainingConfig,
        build_ppo_model_kwargs,
        load_ppo_model_for_env,
    )
    from ztb.training.core.runtime_limits import (
        load_training_dataframe_with_limit,
        resolve_max_features,
    )
else:
    # At runtime we need the real classes; import them rather than using
    # the `object` fallback used for type checking. This avoids tests
    # attempting to construct `object()` which raises TypeError.
    from ztb.training.config.trainer_params import TrainerParams
    from ztb.training.core.ppo_trainer import (
        PPOTrainer,
        PPOTrainerAutoHalt,
        build_ppo_model_kwargs,
        load_ppo_model_for_env,
    )
    from ztb.training.core.ppo_trainer import PPOTrainingConfig as TrainingConfig
    from ztb.training.core.runtime_limits import (
        load_training_dataframe_with_limit,
        resolve_max_features,
    )


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for tests that do not define a class-local fixture."""
    return str(tmp_path)


@pytest.fixture
def sample_config():
    """Default PPO config for tests that do not define a class-local fixture."""
    return {
        "ppo": {
            "learning_rate": 3e-4,
            "n_steps": 1024,
            "batch_size": 32,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "total_timesteps": 10000,
            "reward_scaling": 1.0,
            "use_custom_ppo": False,
        },
        "eval_gates_enabled": False,
    }


def _make_trainer_with_mocked_runtime(
    trainer_params: TrainerParams,
) -> tuple[PPOTrainerAutoHalt, Mock]:
    """Create a trainer with lightweight mocked runtime dependencies."""
    trainer = PPOTrainerAutoHalt(trainer_params)
    model = Mock()
    trainer.env = Mock()
    trainer.model = model
    return trainer, model


class TestPPOTrainerAutoHalt:
    """Test PPOTrainerAutoHalt functionality."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for testing."""
        return str(tmp_path)

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "ppo": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "total_timesteps": 10000,
                "reward_scaling": 1.0,
                "use_custom_ppo": False,
            },
            "eval_gates_enabled": False,
        }

    @pytest.fixture
    def trainer_params(self, temp_dir, sample_config):
        """Create trainer parameters for testing."""
        return TrainerParams(
            data_path="dummy_path.csv", config=sample_config, checkpoint_dir=temp_dir
        )

    def test_initialization_validation_success(self, trainer_params):
        """Test successful initialization with valid parameters."""
        trainer = PPOTrainerAutoHalt(trainer_params)
        assert trainer.params == trainer_params
        assert trainer.data_path == "dummy_path.csv"
        assert str(trainer.checkpoint_dir) == str(trainer_params.checkpoint_dir)
        assert hasattr(trainer, "training_config")
        assert isinstance(trainer.training_config, TrainingConfig)
        assert trainer.eval_gates.enabled is False  # Should be disabled from config

    def test_initialization_validation_missing_data_path(self, temp_dir, sample_config):
        """Test initialization fails with missing data_path."""
        params = TrainerParams(
            data_path="", config=sample_config, checkpoint_dir=temp_dir
        )
        with pytest.raises(ValueError, match="data_path is required"):
            PPOTrainerAutoHalt(params)

    def test_initialization_validation_missing_checkpoint_dir(self, sample_config):
        """Test initialization fails with missing checkpoint_dir."""
        params = TrainerParams(
            data_path="dummy_path.csv", config=sample_config, checkpoint_dir=""
        )
        with pytest.raises(ValueError, match="checkpoint_dir is required"):
            PPOTrainerAutoHalt(params)

    def test_initialization_validation_invalid_config_type(self, temp_dir):
        """Test initialization fails with invalid config type."""
        params = TrainerParams(
            data_path="dummy_path.csv",
            config="invalid_config",
            checkpoint_dir=temp_dir,  # Should be dict
        )
        with pytest.raises(ValueError, match="config must be a dictionary"):
            PPOTrainerAutoHalt(params)

    def test_build_ppo_model_kwargs_reuses_core_training_config(self) -> None:
        config = TrainingConfig(
            learning_rate=1e-4,
            n_steps=128,
            batch_size=16,
            n_epochs=2,
            gamma=0.95,
            gae_lambda=0.91,
            clip_range=0.15,
            clip_range_vf=0.12,
            normalize_advantage=False,
            ent_coef=0.01,
            vf_coef=0.7,
            max_grad_norm=0.4,
            target_kl=0.03,
            verbose=0,
        )

        kwargs = build_ppo_model_kwargs(
            config,
            tensorboard_log="tb_logs",
            device="cpu",
        )

        assert kwargs["learning_rate"] == pytest.approx(1e-4)
        assert kwargs["n_steps"] == 128
        assert kwargs["batch_size"] == 16
        assert kwargs["clip_range_vf"] == pytest.approx(0.12)
        assert kwargs["normalize_advantage"] is False
        assert kwargs["tensorboard_log"] == "tb_logs"
        assert kwargs["device"] == "cpu"

    def test_resolve_max_features_prefers_top_level_value(self) -> None:
        config = {
            "max_features": 48,
            "memory_optimization": {"max_features": 32},
            "ppo": {"max_features": 24},
        }
        assert resolve_max_features(config) == 48

    def test_load_training_dataframe_with_limit_uses_memory_optimization_section(self) -> None:
        df = pd.DataFrame({"close": range(100)})
        loader = Mock(return_value=df)
        logger = Mock()

        result = load_training_dataframe_with_limit(
            "dummy.csv",
            config={"memory_optimization": {"data_rows_limit": 25}},
            loader=loader,
            logger=logger,
        )

        loader.assert_called_once_with("dummy.csv")
        assert len(result) == 25
        assert result.iloc[0]["close"] == 0
        assert result.iloc[-1]["close"] == 24

    @patch("ztb.training.ppo_trainer.CustomPPO")
    def test_load_ppo_model_for_env_uses_current_class_resolution(
        self,
        mock_custom_ppo,
        trainer_params,
    ) -> None:
        env = Mock()
        loaded_model = Mock()
        mock_custom_ppo.load.return_value = loaded_model

        model = load_ppo_model_for_env(
            Path("dummy_model.zip"),
            use_custom_ppo=True,
            env=env,
        )

        mock_custom_ppo.load.assert_called_once_with("dummy_model.zip")
        loaded_model.set_env.assert_called_once_with(env)
        assert model is loaded_model

    @patch("ztb.training.core.ppo_trainer.DataLoader.load_csv_optimized")
    @patch("ztb.training.core.ppo_trainer.HeavyTradingEnv")
    @patch("ztb.training.core.ppo_trainer.ActionMasker")
    def test_create_environment(
        self, mock_action_masker, mock_env, mock_load_data, trainer_params
    ):
        """Test environment creation."""
        # Setup mocks
        mock_df = Mock()
        mock_df.shape = (1000, 10)
        mock_df.index.min.return_value = "2020-01-01"
        mock_df.index.max.return_value = "2020-12-31"
        mock_df.columns = [
            "col1",
            "col2",
            "col3",
            "col4",
            "col5",
            "col6",
            "col7",
            "col8",
            "col9",
            "col10",
        ]
        mock_load_data.return_value = mock_df

        mock_env_instance = Mock()
        mock_env.return_value = mock_env_instance

        mock_action_masker_instance = Mock()
        mock_action_masker.return_value = mock_action_masker_instance

        # Create trainer and test
        trainer = PPOTrainerAutoHalt(trainer_params)
        env = trainer._create_environment()

        # Verify calls
        mock_load_data.assert_called_once_with("dummy_path.csv")
        mock_env.assert_called_once()
        mock_action_masker.assert_called_once()
        _, masker_kwargs = mock_action_masker.call_args
        assert "mask_fn" in masker_kwargs
        assert "action_mask_fn" not in masker_kwargs

        assert env == mock_action_masker_instance

    @patch("ztb.training.ppo_trainer.CustomPPO")
    @patch("ztb.training.ppo_trainer.MaskablePPO")
    def test_create_model_custom_ppo(
        self, mock_maskable_ppo, mock_custom_ppo, trainer_params
    ):
        """Test model creation with custom PPO enabled."""
        # Setup trainer with custom PPO enabled
        trainer_params.config["ppo"]["use_custom_ppo"] = True
        trainer = PPOTrainerAutoHalt(trainer_params)

        # Mock environment
        trainer.env = Mock()

        # Configure mock to have __name__ attribute
        mock_custom_ppo.configure_mock(**{"__name__": "CustomPPO"})

        # Create model
        model = trainer._create_model()

        # Verify CustomPPO was used
        mock_custom_ppo.assert_called_once()
        mock_maskable_ppo.assert_not_called()
        assert model == mock_custom_ppo.return_value

    @patch("ztb.training.ppo_trainer.CustomPPO")
    @patch("ztb.training.ppo_trainer.MaskablePPO")
    def test_create_model_standard_ppo(
        self, mock_maskable_ppo, mock_custom_ppo, trainer_params
    ):
        """Test model creation with standard PPO."""
        # Setup trainer with custom PPO disabled
        trainer_params.config["ppo"]["use_custom_ppo"] = False
        trainer = PPOTrainerAutoHalt(trainer_params)

        # Mock environment
        trainer.env = Mock()

        # Configure mocks to have __name__ attribute
        mock_maskable_ppo.configure_mock(**{"__name__": "MaskablePPO"})
        mock_custom_ppo.configure_mock(**{"__name__": "CustomPPO"})

        # Create model
        model = trainer._create_model()

        # Verify MaskablePPO was used
        mock_maskable_ppo.assert_called_once()
        mock_custom_ppo.assert_not_called()
        assert model == mock_maskable_ppo.return_value

    @patch("ztb.training.core.ppo_trainer.CompositeTrainingCallback")
    def test_create_callback(self, mock_callback, trainer_params):
        """Test callback creation."""
        trainer = PPOTrainerAutoHalt(trainer_params)
        callback = trainer._create_callback()

        mock_callback.assert_called_once()
        assert callback == mock_callback.return_value

    @patch("ztb.training.core.ppo_trainer.neutralize_policy_bias")
    def test_neutralize_policy_bias_with_model(self, mock_neutralize, trainer_params):
        """Test policy bias neutralization when model exists."""
        trainer = PPOTrainerAutoHalt(trainer_params)
        trainer.model = Mock()

        trainer.neutralize_policy_bias()

        mock_neutralize.assert_called_once_with(trainer.model)

    @patch("ztb.training.core.ppo_trainer.neutralize_policy_bias")
    def test_neutralize_policy_bias_without_model(
        self, mock_neutralize, trainer_params
    ):
        """Test policy bias neutralization when no model exists."""
        trainer = PPOTrainerAutoHalt(trainer_params)
        trainer.model = None

        trainer.neutralize_policy_bias()

        mock_neutralize.assert_not_called()

    @patch.object(PPOTrainerAutoHalt, "_cleanup_training_resources")
    @patch.object(PPOTrainerAutoHalt, "_learn_model")
    @patch.object(PPOTrainerAutoHalt, "_create_model")
    @patch.object(PPOTrainerAutoHalt, "_create_environment")
    def test_train_success_path(
        self,
        mock_create_environment,
        mock_create_model,
        mock_learn_model,
        mock_cleanup,
        trainer_params,
    ):
        """Test training initialization (simplified test)."""
        trainer, mock_model = _make_trainer_with_mocked_runtime(trainer_params)
        mock_create_environment.return_value = trainer.env
        mock_create_model.return_value = mock_model
        mock_learn_model.side_effect = KeyboardInterrupt()

        # Current trainer propagates KeyboardInterrupt so outer runners can stop cleanly.
        with pytest.raises(KeyboardInterrupt):
            trainer.train("test_session")

        mock_create_environment.assert_called_once_with()
        mock_create_model.assert_called_once_with()
        mock_learn_model.assert_called_once_with(
            mock_model,
            total_timesteps=trainer.training_config.total_timesteps,
            session_id="test_session",
        )
        mock_cleanup.assert_called_once_with()

    @patch.object(PPOTrainerAutoHalt, "_cleanup_training_resources")
    @patch.object(PPOTrainerAutoHalt, "_learn_model")
    @patch.object(PPOTrainerAutoHalt, "_create_model")
    @patch.object(PPOTrainerAutoHalt, "_create_environment")
    def test_train_with_exception(
        self,
        mock_create_environment,
        mock_create_model,
        mock_learn_model,
        mock_cleanup,
        trainer_params,
    ):
        """Test training with exception handling (simplified test)."""
        trainer, mock_model = _make_trainer_with_mocked_runtime(trainer_params)
        mock_create_environment.return_value = trainer.env
        mock_create_model.return_value = mock_model
        mock_learn_model.side_effect = ValueError("Mock environment error")

        with pytest.raises(ValueError, match="Mock environment error"):
            trainer.train("test_session")

        mock_cleanup.assert_called_once_with()

    @patch.object(PPOTrainerAutoHalt, "_cleanup_training_resources")
    @patch.object(PPOTrainerAutoHalt, "_learn_model")
    @patch.object(PPOTrainerAutoHalt, "_create_model")
    @patch.object(PPOTrainerAutoHalt, "_create_environment")
    def test_train_complete_success_path(
        self,
        mock_create_environment,
        mock_create_model,
        mock_learn_model,
        mock_cleanup,
        trainer_params,
    ):
        """Test complete training success path with logging."""
        trainer, mock_model = _make_trainer_with_mocked_runtime(trainer_params)
        mock_create_environment.return_value = trainer.env
        mock_create_model.return_value = mock_model
        mock_learn_model.return_value = mock_model

        # Should complete successfully and return model
        result = trainer.train("test_session")
        assert result == mock_model  # Training completed successfully
        mock_cleanup.assert_called_once_with()

    @patch.object(PPOTrainerAutoHalt, "_cleanup_training_resources")
    @patch.object(PPOTrainerAutoHalt, "_learn_model")
    @patch.object(PPOTrainerAutoHalt, "_create_model")
    @patch.object(PPOTrainerAutoHalt, "_create_environment")
    def test_train_with_exception_logging(
        self,
        mock_create_environment,
        mock_create_model,
        mock_learn_model,
        mock_cleanup,
        trainer_params,
    ):
        """Test training exception handling with logging."""
        trainer, mock_model = _make_trainer_with_mocked_runtime(trainer_params)
        mock_create_environment.return_value = trainer.env
        mock_create_model.return_value = mock_model
        mock_learn_model.side_effect = ValueError("Test error")

        # Should handle exception and re-raise it
        with pytest.raises(ValueError, match="Test error"):
            trainer.train("test_session")
        mock_cleanup.assert_called_once_with()

    @patch("ztb.training.core.ppo_trainer.DataLoader.load_csv_optimized")
    def test_create_environment_data_loading(self, mock_load_data, trainer_params):
        """Test environment creation data loading path."""
        # Setup mock data
        mock_df = Mock()
        mock_df.shape = (1000, 10)
        mock_df.index.min.return_value = "2020-01-01"
        mock_df.index.max.return_value = "2020-12-31"
        mock_df.columns = [
            "col1",
            "col2",
            "col3",
            "col4",
            "col5",
            "col6",
            "col7",
            "col8",
            "col9",
            "col10",
        ]
        mock_load_data.return_value = mock_df

        trainer = PPOTrainerAutoHalt(trainer_params)

        with patch("ztb.training.core.ppo_trainer.HeavyTradingEnv") as mock_env, patch(
            "ztb.training.core.ppo_trainer.ActionMasker"
        ) as mock_action_masker:
            mock_env_instance = Mock()
            mock_env.return_value = mock_env_instance

            mock_action_masker_instance = Mock()
            mock_action_masker.return_value = mock_action_masker_instance

            # Call _create_environment to cover data loading
            env = trainer._create_environment()

            # Verify data loading was called
            mock_load_data.assert_called_once_with(trainer.data_path)
            assert env == mock_action_masker_instance


class TestPPOTrainerBasicExtended:
    """Test PPOTrainer functionality."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "total_timesteps": 10000,
            "reward_scaling": 1.0,
            "use_custom_ppo": False,
        }

    def test_ppo_trainer_initialization(self, temp_dir, sample_config):
        """Test PPOTrainer initialization."""
        trainer = PPOTrainer(
            data_path="dummy_path.csv", config=sample_config, checkpoint_dir=temp_dir
        )

        assert trainer.data_path == "dummy_path.csv"
        assert str(trainer.checkpoint_dir) == str(Path(temp_dir))
        assert hasattr(trainer, "training_config")
        assert isinstance(trainer.training_config, TrainingConfig)


class TestTrainingConfig:
    """Test TrainingConfig functionality."""

    def test_from_dict_basic(self):
        """Test TrainingConfig.from_dict with basic config."""
        config_dict = {
            "ppo": {
                "learning_rate": 1e-3,
                "n_steps": 1024,
                "batch_size": 32,
                "n_epochs": 5,
                "gamma": 0.95,
                "gae_lambda": 0.9,
                "clip_range": 0.1,
                "ent_coef": 0.01,
                "vf_coef": 0.6,
                "max_grad_norm": 1.0,
                "total_timesteps": 5000,
                "reward_scaling": 2.0,
                "use_custom_ppo": True,
            }
        }

        config = TrainingConfig.from_dict(config_dict)

        assert config.learning_rate == 1e-3
        assert config.n_steps == 1024
        assert config.batch_size == 32
        assert config.n_epochs == 5
        assert config.gamma == 0.95
        assert config.gae_lambda == 0.9
        assert config.clip_range == 0.1
        assert config.ent_coef == 0.01
        assert config.vf_coef == 0.6
        assert config.max_grad_norm == 1.0
        assert config.total_timesteps == 5000
        assert config.reward_scaling == 2.0
        assert config.use_custom_ppo is True

    def test_from_dict_defaults(self):
        """Test TrainingConfig.from_dict with minimal config (uses defaults)."""
        config_dict = {"ppo": {}}

        config = TrainingConfig.from_dict(config_dict)

        # Check that defaults are applied
        assert config.learning_rate == 3e-4
        assert config.n_steps == 1024
        assert config.batch_size == 32
        assert config.n_epochs == 10
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_range == 0.2
        assert config.ent_coef == 0.0
        assert config.vf_coef == 0.5
        assert config.max_grad_norm == 0.5
        # Note: use_custom_ppo defaults to True in TrainingConfig
        assert config.use_custom_ppo is True


class TestPPOTrainerBasic:
    """Test PPOTrainer functionality."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for testing."""
        return str(tmp_path)

    @pytest.fixture
    def trainer_params(self, temp_dir, sample_config):
        """Create trainer parameters for testing."""
        return TrainerParams(
            data_path="dummy_path.csv", config=sample_config, checkpoint_dir=temp_dir
        )

    def test_ppo_trainer_initialization(self, temp_dir, sample_config):
        """Test PPOTrainer initialization."""
        config = {
            "ppo": {
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "total_timesteps": 10000,
                "reward_scaling": 1.0,
                "use_custom_ppo": False,
            }
        }

        trainer = PPOTrainer(
            data_path="dummy_path.csv", config=config, checkpoint_dir=temp_dir
        )

        assert trainer.data_path == "dummy_path.csv"
        assert str(trainer.checkpoint_dir) == temp_dir
        assert trainer.training_config.learning_rate == 3e-4
        assert trainer.training_config.n_steps == 2048
        assert trainer.training_config.batch_size == 64
        assert trainer.training_config.n_epochs == 10
        assert trainer.training_config.gamma == 0.99
        assert trainer.training_config.gae_lambda == 0.95
        assert trainer.training_config.clip_range == 0.2
        assert trainer.training_config.ent_coef == 0.0
        assert trainer.training_config.vf_coef == 0.5
        assert trainer.training_config.max_grad_norm == 0.5
        assert trainer.training_config.use_custom_ppo is False

    def test_ppo_trainer_initialization_missing_data_path(
        self, temp_dir, sample_config
    ):
        """Test PPOTrainer initialization with missing data path."""
        config = {
            "ppo": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "total_timesteps": 10000,
                "reward_scaling": 1.0,
                "use_custom_ppo": False,
            }
        }

        with pytest.raises(
            ValueError, match="data_path is required and cannot be empty"
        ):
            PPOTrainer(
                data_path="", config=config, checkpoint_dir=temp_dir
            )  # Empty data path

    def test_ppo_trainer_initialization_missing_checkpoint_dir(self, sample_config):
        """Test PPOTrainer initialization with missing checkpoint directory."""
        config = {
            "ppo": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "total_timesteps": 10000,
                "reward_scaling": 1.0,
                "use_custom_ppo": False,
            }
        }

        with pytest.raises(
            ValueError, match="checkpoint_dir is required and cannot be empty"
        ):
            PPOTrainer(
                data_path="dummy_path.csv", config=config, checkpoint_dir=""
            )  # Empty checkpoint dir

    def test_ppo_trainer_initialization_invalid_config_type(self, temp_dir):
        """Test PPOTrainer initialization with invalid config type."""
        with pytest.raises(AttributeError, match="items"):
            PPOTrainer(
                data_path="dummy_path.csv",
                config="invalid_config",
                checkpoint_dir=temp_dir,
            )  # Not a dict

    def test_ppo_trainer_initialization_eval_gates_disabled(
        self, temp_dir, sample_config
    ):
        """Test PPOTrainer initialization with eval gates disabled."""
        config = {
            "ppo": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "total_timesteps": 10000,
                "reward_scaling": 1.0,
                "use_custom_ppo": False,
            },
            "eval_gates_enabled": False,
        }

        trainer = PPOTrainer(
            data_path="dummy_path.csv", config=config, checkpoint_dir=temp_dir
        )

        assert trainer.eval_gates is not None
        # The initialization logging should have taken the else branch

    def test_ppo_trainer_initialization_missing_data_path_in_ppo_trainer(
        self, temp_dir
    ):
        """Test PPOTrainer initialization with missing data path in PPOTrainer."""
        config = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "total_timesteps": 10000,
            "reward_scaling": 1.0,
            "use_custom_ppo": False,
        }

        with pytest.raises(
            ValueError, match="data_path is required and cannot be empty"
        ):
            PPOTrainer(
                data_path="", config=config, checkpoint_dir=temp_dir
            )  # Empty data path

    def test_ppo_trainer_initialization_missing_checkpoint_dir_in_ppo_trainer(
        self, temp_dir
    ):
        """Test PPOTrainer initialization with missing checkpoint directory in PPOTrainer."""
        config = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "total_timesteps": 10000,
            "reward_scaling": 1.0,
            "use_custom_ppo": False,
        }

        with pytest.raises(
            ValueError, match="checkpoint_dir is required and cannot be empty"
        ):
            PPOTrainer(
                data_path="dummy_path.csv", config=config, checkpoint_dir=""
            )  # Empty checkpoint dir


class TestPPOAlgorithmTrainer:
    """Test PPOAlgorithmTrainer functionality."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for testing."""
        return str(tmp_path)

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "total_timesteps": 10000,
            "reward_scaling": 1.0,
            "use_custom_ppo": False,
        }

    def test_ppo_algorithm_trainer_initialization(self, temp_dir, sample_config):
        """Test PPOAlgorithmTrainer initialization."""
        config_manager = Mock()
        trainer = PPOAlgorithmTrainer(config_manager, progress_bar_enabled=True)

        assert trainer.config_manager == config_manager
        assert trainer.progress_bar_enabled is True
        assert hasattr(trainer, "ui_manager")
        assert hasattr(trainer, "reporter")
        assert trainer.logger is not None

    @patch("ztb.training.core.ppo_trainer.PPOTrainerAutoHalt")
    def test_create_standard_ppo_trainer_uses_trainer_params(
        self, mock_auto_halt, temp_dir
    ):
        """Standard PPO path should instantiate PPOTrainerAutoHalt with TrainerParams."""
        from ztb.training.config.trainer_params import TrainerParams
        config_manager = Mock()
        trainer = PPOAlgorithmTrainer(config_manager, progress_bar_enabled=True)
        unified_config = {
            "data_path": "dummy_path.csv",
            "checkpoint_dir": temp_dir,
            "checkpoint_interval": 1234,
        }

        trainer._create_ppo_trainer(unified_config, enable_sell_mitigation=False)

        mock_auto_halt.assert_called_once()
        (params,), _ = mock_auto_halt.call_args
        assert isinstance(params, TrainerParams)
        assert params.data_path == "dummy_path.csv"
        assert str(params.checkpoint_dir) == temp_dir
        assert params.checkpoint_interval == 1234

    @patch("ztb.training.sell_mitigation_ppo_trainer.SELLBiasMitigationPPOTrainer")
    def test_create_sell_mitigation_ppo_trainer_uses_shim(
        self, mock_sell_mitigation_trainer, temp_dir
    ):
        """SELL mitigation path should go through the compatibility shim."""
        from ztb.training.config.trainer_params import SELLMitigationParams
        config_manager = Mock()
        trainer = PPOAlgorithmTrainer(config_manager, progress_bar_enabled=True)
        unified_config = {
            "data_path": "dummy_path.csv",
            "checkpoint_dir": temp_dir,
            "checkpoint_interval": 4321,
            "enable_lagrange": True,
            "enable_pan": True,
        }

        trainer._create_ppo_trainer(unified_config, enable_sell_mitigation=True)

        mock_sell_mitigation_trainer.assert_called_once()
        (params,), _ = mock_sell_mitigation_trainer.call_args
        assert isinstance(params, SELLMitigationParams)
        assert params.data_path == "dummy_path.csv"
        assert str(params.checkpoint_dir) == temp_dir
        assert params.checkpoint_interval == 4321
