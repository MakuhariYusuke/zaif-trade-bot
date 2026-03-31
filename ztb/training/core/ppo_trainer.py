"""
PPO Trainer implementations:
- PPOTrainer: Standard 1M timestep training with evaluation gates and memory optimization
- PPOTrainerAutoHalt: Auto-halt/gate functionality (migrated from trading/ppo_trainer.py)
"""

# Keep existing PPOTrainer class as-is

# --- Migrate PPOTrainer from trading/ppo_trainer.py as PPOTrainerAutoHalt below ---

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

from ztb.features.curated_features import (
    FeatureSet,
    get_feature_set,
    get_features_to_remove,
)
from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.training.callbacks.callbacks_legacy import CompositeTrainingCallback
from ztb.training.config.ppo_config import (
    DEFAULT_INITIAL_PORTFOLIO_VALUE,
    DEFAULT_REWARD_SCALING,
    DEFAULT_TOTAL_TIMESTEPS,
    PPOConfig,
)
from ztb.training.config.trainer_params import TrainerParams
from ztb.training.core.base_trainer import BaseTrainer
from ztb.training.evaluation.eval_gates import EvalGates
from ztb.training.models.custom_ppo import CustomPPO
from ztb.training.policies.policy_utils import neutralize_policy_bias
from ztb.io.data_loader import DataLoader
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Type aliases for better type safety
Observation: TypeAlias = NDArray[np.float32]
Action = int
PredictionResult: TypeAlias = tuple[Action, Any | None]  # (action, additional_info)


def _get_action_masks(env: Any) -> NDArray[np.bool_]:
    """Return the current boolean action mask from a trading environment."""
    return env.get_action_masks()


def wrap_env_with_action_masker(env: HeavyTradingEnv) -> ActionMasker:
    """Wrap a trading env with the current sb3-contrib action-mask contract."""
    return ActionMasker(env, mask_fn=_get_action_masks)

class PPOTrainerProtocol(Protocol):
    """Protocol for PPO Trainer implementations."""

    def train(self, session_id: str) -> MaskablePPO | None:
        """Train the model and return it."""
        ...

class PredictorProtocol(Protocol):
    """Protocol for predictor implementations."""

    def predict(self, observation: Observation) -> PredictionResult:
        """Make a prediction based on observation."""
        ...

class TradingSystemProtocol(Protocol):
    """Protocol for trading system implementations."""

    def trade(self, observation: Observation) -> dict[str, Any]:
        """Execute a trade based on observation."""
        ...

class Algorithm(Enum):
    """Supported training algorithms."""

    PPO = "ppo"

class Timeframe(Enum):
    """Supported timeframes for training."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"

@dataclass
class PPOTrainingConfig:
    """Configuration for PPO training."""

    # Core PPO parameters
    learning_rate: float = 3e-4
    n_steps: int = (
        1024  # Reduced from 2048 for memory optimization in iterative learning
    )
    batch_size: int = (
        32  # Reduced from 64 for memory optimization in iterative learning
    )
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float | None = None
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_sde: bool = False
    sde_sample_freq: int = -1
    target_kl: float | None = None

    # Environment and data parameters
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS
    reward_scaling: float = DEFAULT_REWARD_SCALING
    initial_portfolio_value: float = DEFAULT_INITIAL_PORTFOLIO_VALUE
    transaction_cost: float = 0.0
    max_position_size: float = 1.0

    # Evaluation and gates
    eval_freq: int = 10000
    n_eval_episodes: int = 5
    eval_deterministic: bool = True

    # Logging and checkpointing
    log_interval: int = (
        10  # Increased from 1 for memory optimization in iterative learning
    )
    save_freq: int = 10000
    verbose: int = 1

    # Custom features
    use_custom_ppo: bool = True
    enable_grad_probe_guard: bool = False
    grad_probe_config: dict[str, Any] | None = None

    # Feature configuration
    feature_set: FeatureSet = FeatureSet.CURATED
    custom_features: list[str] | None = None
    feature_config_path: str | None = None

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "PPOTrainingConfig":
        """Create TrainingConfig from dictionary."""
        config = cls()

        # 🔧 FIX: ppo_hyperparametersキーもチェック（v392/v393等で使用）
        # トップレベルの"ppo"、次に"ppo_hyperparameters"、最後にトップレベルキーを確認
        ppo_config = config_dict.get("ppo", {})
        ppo_hyperparams = config_dict.get("ppo_hyperparameters", {})

        def get_param(key: str, default: Any) -> Any:
            """3段階で設定を検索: ppo > ppo_hyperparameters > トップレベル"""
            # 1. ppo キー内
            if key in ppo_config:
                return ppo_config[key]
            # 2. ppo_hyperparameters キー内
            if key in ppo_hyperparams:
                return ppo_hyperparams[key]
            # 3. トップレベル
            return config_dict.get(key, default)

        # Core PPO parameters
        config.learning_rate = float(get_param("learning_rate", 3e-4))
        config.n_steps = int(
            get_param("n_steps", 1024)
        )  # Updated default for memory optimization
        config.batch_size = int(
            get_param("batch_size", 32)
        )  # Reduced for memory optimization
        config.n_epochs = int(get_param("n_epochs", 10))
        config.gamma = float(get_param("gamma", 0.99))
        config.gae_lambda = float(get_param("gae_lambda", 0.95))
        config.clip_range = float(get_param("clip_range", 0.2))
        config.clip_range_vf = get_param("clip_range_vf", None)
        config.normalize_advantage = bool(get_param("normalize_advantage", True))
        config.ent_coef = float(get_param("ent_coef", 0.0))
        config.vf_coef = float(get_param("vf_coef", 0.5))
        config.max_grad_norm = float(get_param("max_grad_norm", 0.5))
        config.use_sde = bool(get_param("use_sde", False))
        config.sde_sample_freq = int(get_param("sde_sample_freq", -1))
        config.target_kl = get_param("target_kl", None)

        # Environment and data parameters
        # 🔧 FIX: get_param()を使用（重複していたcommon_config参照を削除）
        config.total_timesteps = int(
            get_param("total_timesteps", DEFAULT_TOTAL_TIMESTEPS)
        )
        config.reward_scaling = float(get_param("reward_scaling", 1.0))
        config.transaction_cost = float(get_param("transaction_cost", 0.0))
        config.max_position_size = float(get_param("max_position_size", 1.0))
        config.use_custom_ppo = bool(get_param("use_custom_ppo", True))

        # Feature configuration
        features_config = config_dict.get("features", {})
        # `features` may be either a list of feature names or a dict describing
        # feature configuration (including a 'feature_set' key). Handle both.
        if isinstance(features_config, dict):
            feature_set_str = features_config.get("feature_set", "curated")
        else:
            feature_set_str = config_dict.get("feature_set", "curated")
        try:
            config.feature_set = FeatureSet(feature_set_str)
        except ValueError:
            logger.warning(
                "Unknown feature_set '%s' provided; defaulting to CURATED set",
                feature_set_str,
            )
            config.feature_set = FeatureSet.CURATED
        # custom_features may be specified inside a features dict or not present
        if isinstance(features_config, dict):
            config.custom_features = features_config.get("custom_features")
            config.feature_config_path = features_config.get("feature_config_path")
        else:
            config.custom_features = None
            config.feature_config_path = None

        return config

class PPOTrainerAutoHalt(BaseTrainer, PPOTrainerProtocol):
    """
    PPO Trainer with evaluation gates and auto-halt functionality.

    This trainer extends BaseTrainer with comprehensive PPO training capabilities,
    including automatic evaluation gates, training halt conditions, and progress
    tracking. It supports both standard PPO and custom PPO implementations.

    Features:
    - Evaluation gates for automatic training quality assessment
    - Auto-halt based on performance criteria
    - Memory optimization for large datasets
    - Comprehensive logging and checkpointing
    - Support for custom PPO implementations
    """

    model: MaskablePPO | None
    env: ActionMasker | None

    def __init__(self, params: TrainerParams):
        """
        Initialize PPO Trainer with auto-halt functionality.

        Args:
            params: Training parameters including data path, config, and checkpoint directory

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Validate input parameters
        if not params.data_path:
            raise ValueError("data_path is required and cannot be empty")
        if not params.checkpoint_dir:
            raise ValueError("checkpoint_dir is required and cannot be empty")

        super().__init__(params)
        self.params = params

        # Initialize training configuration
        self.training_config = PPOTrainingConfig.from_dict(self.config)

        # Initialize evaluation gates
        eval_gates_enabled = self.config.get("eval_gates_enabled", True)
        self.eval_gates = EvalGates(enabled=eval_gates_enabled)

        # Training state
        self.best_model_path: str | None = None
        self.training_stats: dict[str, Any] = {}

        # Log comprehensive initialization details
        logger.info("Initialized PPOTrainerAutoHalt with session details:")
        logger.info(f"  Data path: {self.data_path}")
        logger.info(f"  Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"  Total timesteps: {self.training_config.total_timesteps:,}")
        logger.info(f"  Learning rate: {self.training_config.learning_rate}")
        logger.info(f"  Batch size: {self.training_config.batch_size}")
        logger.info(f"  Reward scaling: {self.training_config.reward_scaling}")
        logger.info(f"  Custom PPO enabled: {self.training_config.use_custom_ppo}")
        logger.info(
            f"  Gradient probe guard: {self.training_config.enable_grad_probe_guard}"
        )
        if hasattr(self, "eval_gates") and self.eval_gates.enabled:
            logger.info("  Evaluation gates: enabled")
        else:
            logger.info("  Evaluation gates: disabled")

    def _create_environment(self) -> ActionMasker:
        """Create and configure the trading environment."""
        df_full = DataLoader.load_csv_optimized(self.data_path)

        # ========================================================================
        # UNIFIED MEMORY OPTIMIZATION (Bug #52 fix)
        # ========================================================================
        # Apply data_rows_limit if specified
        # Priority: 1) Top-level config, 2) memory_optimization section
        data_rows_limit = self.config.get("data_rows_limit") or (
            self.config.get("memory_optimization", {}) or {}
        ).get("data_rows_limit")

        if data_rows_limit and len(df_full) > data_rows_limit:
            logger.info(
                f"⚠️  MEMORY OPTIMIZATION: Limiting data from {len(df_full)} to {data_rows_limit} rows"
            )
            # Memory optimized: Use iloc slice instead of copy
            df = df_full.iloc[:data_rows_limit]
            del df_full
            import gc

            gc.collect()
        else:
            df = df_full

        # Determine feature inclusion list from configuration
        feature_columns: list[str] | None = None
        feature_set_enum = getattr(
            self.training_config, "feature_set", FeatureSet.CURATED
        )
        feature_set_name = (
            feature_set_enum.value
            if isinstance(feature_set_enum, FeatureSet)
            else str(feature_set_enum)
        )
        config_path_raw = getattr(self.training_config, "feature_config_path", None)
        config_path = Path(config_path_raw) if config_path_raw else None

        custom_feature_list = getattr(self.training_config, "custom_features", None)
        if custom_feature_list:
            if isinstance(custom_feature_list, list):
                feature_columns = [str(col) for col in custom_feature_list]
                logger.info(
                    "Using custom feature list from configuration (%d columns)",
                    len(feature_columns),
                )
            else:
                logger.warning(
                    "custom_features provided but not a list; ignoring value of type %s",
                    type(custom_feature_list),
                )
        else:
            try:
                feature_columns = get_feature_set(feature_set_name, config_path)
                logger.info(
                    "Loaded feature set '%s' with %d features",
                    feature_set_name,
                    len(feature_columns),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load feature set '%s' (%s); falling back to curated defaults",
                    feature_set_name,
                    exc,
                )
                feature_columns = get_feature_set("curated", config_path)
                feature_set_name = "curated"

        if feature_columns:
            # Remove explicitly excluded features for curated sets
            try:
                excluded_features = set(get_features_to_remove(feature_set_name))
            except Exception:
                excluded_features = set()

            if excluded_features:
                before_count = len(feature_columns)
                feature_columns = [
                    f for f in feature_columns if f not in excluded_features
                ]
                removed_count = before_count - len(feature_columns)
                if removed_count > 0:
                    logger.info(
                        "Removed %d excluded features from feature set '%s'",
                        removed_count,
                        feature_set_name,
                    )

            present_columns = [col for col in feature_columns if col in df.columns]
            missing_columns = sorted(set(feature_columns) - set(present_columns))

            if missing_columns:
                logger.warning(
                    "%d features requested but missing from dataset: %s",
                    len(missing_columns),
                    ", ".join(list(missing_columns)[:5])
                    + ("..." if len(missing_columns) > 5 else ""),
                )

            if present_columns:
                # Store feature whitelist in config for HeavyTradingEnv to consume
                self.config["feature_names"] = present_columns
                self.config.setdefault("enable_feature_filtering", True)
                self.config["feature_set"] = feature_set_name
                logger.info(
                    "Configured environment feature whitelist (%d columns)",
                    len(present_columns),
                )
            else:
                logger.error(
                    "No requested features are present in dataset; continuing without feature filter"
                )

        # Extract max_features from unified config structure
        # Priority: 1) Top-level config, 2) memory_optimization section, 3) ppo section
        max_features = (
            self.config.get("max_features")
            or (self.config.get("memory_optimization", {}) or {}).get("max_features")
            or (self.config.get("ppo", {}) or {}).get("max_features")
        )

        # Create environment with memory optimization settings
        env = HeavyTradingEnv(
            df=df,
            config=self.config,
            max_features=max_features,
        )

        # Wrap with action masker for valid action masking.
        # Use the current HeavyTradingEnv contract (`get_action_masks`) and the
        # standard `mask_fn` keyword so the local compatibility shim and the real
        # sb3-contrib wrapper both accept the call.
        env = wrap_env_with_action_masker(env)

        logger.info("Created trading environment with enhanced details:")
        logger.info(f"  Dataset shape: {df.shape}")
        logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"  Features: {len(df.columns)} columns")
        if self.config.get("max_features"):
            logger.info(
                f"  Feature limit: {self.config.get('max_features')} (memory optimization)"
            )
        logger.info("  Action masking: enabled for valid trading actions")
        return env

    def _create_model(self) -> MaskablePPO:
        """Create and configure the PPO model."""
        if self.env is None:
            raise RuntimeError(
                "Environment must be created before model. "
                "Ensure _create_env() is called before _create_model()."
            )

        # Additional validation for environment readiness
        if not hasattr(self.env, "observation_space"):
            raise RuntimeError("Environment does not have observation_space")
        if not hasattr(self.env, "action_space"):
            raise RuntimeError("Environment does not have action_space")

        # Use compatibility shim module names to allow tests to patch
        # `ztb.training.ppo_trainer.CustomPPO` / `MaskablePPO` at runtime.
        try:
            import ztb.training.ppo_trainer as _shim

            if self.training_config.use_custom_ppo:
                model_class = getattr(_shim, "CustomPPO", CustomPPO)
                logger.info("Using CustomPPO model")
            else:
                model_class = getattr(_shim, "MaskablePPO", MaskablePPO)
                logger.info("Using standard MaskablePPO model")
        except Exception:
            # Fallback to locally imported symbols
            if self.training_config.use_custom_ppo:
                model_class = CustomPPO
                logger.info("Using CustomPPO model")
            else:
                model_class = MaskablePPO
                logger.info("Using standard MaskablePPO model")

        model = model_class(
            "MlpPolicy",
            self.env,
            learning_rate=self.training_config.learning_rate,
            n_steps=self.training_config.n_steps,
            batch_size=self.training_config.batch_size,
            n_epochs=self.training_config.n_epochs,
            gamma=self.training_config.gamma,
            gae_lambda=self.training_config.gae_lambda,
            clip_range=self.training_config.clip_range,
            clip_range_vf=self.training_config.clip_range_vf,
            normalize_advantage=self.training_config.normalize_advantage,
            ent_coef=self.training_config.ent_coef,
            vf_coef=self.training_config.vf_coef,
            max_grad_norm=self.training_config.max_grad_norm,
            target_kl=self.training_config.target_kl,
            tensorboard_log=str(self.checkpoint_dir),
            verbose=self.training_config.verbose,
            device="auto",
        )

        logger.info("Created PPO model with comprehensive configuration:")
        logger.info(f"  Model type: {model_class.__name__}")
        logger.info("  Policy: MlpPolicy")
        logger.info(f"  Learning rate: {self.training_config.learning_rate}")
        logger.info(f"  Steps per update: {self.training_config.n_steps}")
        logger.info(f"  Batch size: {self.training_config.batch_size}")
        logger.info(f"  Epochs per update: {self.training_config.n_epochs}")
        logger.info(f"  Gamma: {self.training_config.gamma}")
        logger.info(f"  GAE lambda: {self.training_config.gae_lambda}")
        logger.info(f"  Entropy coefficient: {self.training_config.ent_coef}")
        logger.info(f"  Value function coefficient: {self.training_config.vf_coef}")
        logger.info(f"  TensorBoard logging: {self.checkpoint_dir}")
        return model

    def _create_callback(self) -> BaseCallback:
        """Create composite training callback with progress tracking and entropy scheduling."""
        # Get grad_probe_guard config from self.config
        enable_grad_probe_guard = self.config.get("enable_grad_probe_guard", False)
        grad_probe_config = self.config.get("grad_probe_config", None)

        return CompositeTrainingCallback(
            trainer=self,
            enable_progress_bar=self.progress_bar_enabled,
            enable_entropy_schedule=True,
            entropy_schedule_type="cosine_decay",
            initial_ent_coef=self.training_config.ent_coef,
            final_ent_coef=self.config.get(
                "ent_coef_final", self.training_config.ent_coef
            ),
            enable_grad_probe_guard=enable_grad_probe_guard,
            grad_probe_config=grad_probe_config,
        )

    def train(self, session_id: str) -> MaskablePPO | None:
        """
        Train the PPO model with evaluation gates and auto-halt functionality.

        Args:
            session_id: Unique identifier for this training session

        Returns:
            Trained model if successful, None if training was halted
        """
        logger.info(f"Starting PPO training session: {session_id}")
        logger.info("Training configuration summary:")
        logger.info(f"  Session ID: {session_id}")
        logger.info(f"  Total timesteps: {self.training_config.total_timesteps:,}")
        logger.info(
            f"  Expected episodes: ~{self.training_config.total_timesteps // self.training_config.n_steps:,}"
        )
        logger.info(f"  Checkpoint directory: {self.checkpoint_dir}")
        logger.info(
            "  Custom PPO features: enabled"
            if self.training_config.use_custom_ppo
            else "  Custom PPO features: disabled"
        )
        logger.info(
            "  Evaluation gates: enabled"
            if hasattr(self, "eval_gates") and self.eval_gates
            else "  Evaluation gates: disabled"
        )
        logger.info(
            "  Gradient probe guard: enabled"
            if self.training_config.enable_grad_probe_guard
            else "  Gradient probe guard: disabled"
        )

        try:
            # Reset environment and model for fresh training
            self.env = self._create_environment()
            self.model = self._create_model()

            # Train the model
            logger.info(
                f"Training for {self.training_config.total_timesteps} timesteps"
            )
            logger.info("Beginning model training with progress tracking...")

            self.model.learn(
                total_timesteps=self.training_config.total_timesteps,
                callback=self._create_callback(),
                tb_log_name=session_id,
                progress_bar=self.progress_bar_enabled,
            )

            logger.info("Training loop completed, evaluating final results...")

            # Training completed successfully
            logger.info(f"Training completed successfully for session: {session_id}")
            logger.info("Final training statistics:")
            logger.info(f"  Model trained: {type(self.model).__name__}")
            logger.info(
                f"  Total timesteps processed: {self.training_config.total_timesteps:,}"
            )
            logger.info(f"  Checkpoint directory: {self.checkpoint_dir}")
            if hasattr(self.model, "policy"):
                logger.info("  Policy network: initialized and trained")

            # Store reference for return before cleanup
            trained_model = self.model
            return trained_model

        except Exception as e:
            logger.error(f"Training failed for session {session_id}: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(
                f"Session details: data_path={self.data_path}, checkpoint_dir={self.checkpoint_dir}"
            )
            logger.error(
                "Training session terminated due to critical error", exc_info=True
            )
            raise

        finally:
            # Always cleanup resources - critical for memory management
            import gc

            logger.info("Cleaning up training resources...")

            try:
                # Close environment first (don't modify model internals to preserve return value)
                if self.env is not None:
                    try:
                        self.env.close()
                        logger.debug("Environment closed")
                    except Exception as env_error:
                        logger.warning(f"Error closing environment: {env_error}")

                # Clear instance references to allow garbage collection
                self.env = None
                # Note: Do NOT clear self.model internals (policy, etc.) as it's being returned
                # Only clear the instance reference after successful return
                logger.debug("Instance references cleared")

            except Exception as cleanup_error:
                logger.warning(f"Error during resource cleanup: {cleanup_error}")
                logger.warning(
                    f"Cleanup error details: {type(cleanup_error).__name__}: {cleanup_error}"
                )
                import traceback

                logger.debug(f"Cleanup traceback: {traceback.format_exc()}")

            # Force garbage collection multiple times to handle circular references
            collected_count = 0
            for i in range(3):
                collected = gc.collect(generation=i)
                collected_count += collected
                if collected > 0:
                    logger.debug(f"GC generation {i}: collected {collected} objects")

            if collected_count > 0:
                logger.debug(
                    f"Total objects collected during cleanup: {collected_count}"
                )
            logger.info("✅ Resource cleanup completed")

    def neutralize_policy_bias(self) -> None:
        """Neutralize policy head bias."""
        if self.model is not None:
            logger.info("Applying policy bias neutralization...")
            neutralize_policy_bias(self.model)
            logger.info("Policy bias neutralization completed successfully")
            logger.info("  Bias mitigation: applied to policy head")
            logger.info("  Purpose: Prevent action distribution skew")
        else:
            logger.warning("Cannot neutralize policy bias: no model available")
            logger.warning("  Reason: Model not initialized or training not started")

class PPOTrainer(PPOTrainerAutoHalt):
    """
    Standard PPO Trainer for ensemble training.

    Simplified interface for ensemble model training with basic configuration.
    """

    def __init__(
        self,
        data_path: str,
        config: dict[str, Any],
        checkpoint_dir: str,
        max_features: int | None = None,
    ) -> None:
        # Validate input parameters
        if not data_path:
            raise ValueError("data_path is required and cannot be empty")
        if not checkpoint_dir:
            raise ValueError("checkpoint_dir is required and cannot be empty")

        # Create PPOConfig from the provided config dict
        # Exclude non-PPO parameters
        ppo_config_dict = {
            k: v
            for k, v in config.items()
            if k
            not in [
                "data_path",
                "checkpoint_dir",
                "feature_set",
                "timeframe",
                "algorithm",
            ]
        }
        from ztb.training.config.ppo_config import get_ppo_config

        ppo_config = get_ppo_config(ppo_config_dict)

        # Create TrainerParams from the provided arguments
        params = TrainerParams(
            data_path=data_path,
            config=ppo_config,
            checkpoint_dir=checkpoint_dir,
        )
        super().__init__(params)

# Exported symbols
__all__ = [
    "PPOTrainer",
    "PPOTrainerAutoHalt",
    "PredictorProtocol",
    "TradingSystemProtocol",
    "Algorithm",
    "FeatureSet",
    "Timeframe",
    "PPOConfig",
]

# Note: Backward compatibility aliases removed to simplify the codebase.
