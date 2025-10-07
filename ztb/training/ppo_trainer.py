"""
PPO Trainer implementations:
- PPOTrainer: 標準的な1Mタイムステップ学習・評価ゲート・メモリ最適化付き
- PPOTrainerAutoHalt: auto-halt/gate機能付き（旧trading/ppo_trainer.py由来）
"""

# 既存のPPOTrainerクラスはそのまま残す

# --- 以下、trading/ppo_trainer.pyのPPOTrainerをPPOTrainerAutoHaltとして移植 ---

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Protocol

import numpy as np
from numpy.typing import NDArray
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

from ztb.training.custom_ppo import CustomPPO
from ztb.training.trainer_params import TrainerParams
from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.training.base_trainer import BaseTrainer
from ztb.training.callbacks import CompositeTrainingCallback
from ztb.training.eval_gates import EvalGates
from ztb.training.policy_utils import neutralize_policy_bias
from ztb.training.ppo_config import DEFAULT_PPO_CONFIG, DEFAULT_REWARD_SCALING, DEFAULT_TOTAL_TIMESTEPS, DEFAULT_INITIAL_PORTFOLIO_VALUE
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Type aliases for better type safety
Observation = NDArray[np.float32]
Action = int
PredictionResult = tuple[Action, Optional[Any]]  # (action, additional_info)


class PPOTrainerProtocol(Protocol):
    """Protocol for PPO Trainer implementations."""

    def train(self, session_id: str) -> Optional[MaskablePPO]:
        """Train the model and return it."""
        ...

    def get_reward_stats(self) -> Dict[str, float]:
        """Get training reward statistics."""
        ...

    def neutralize_policy_bias(self) -> None:
        """Neutralize policy head bias."""
        ...


class PredictorProtocol(Protocol):
    """Protocol for predictor implementations."""

    def predict(self, observation: Observation) -> PredictionResult:
        """Make a prediction based on observation."""
        ...


class TradingSystemProtocol(Protocol):
    """Protocol for trading system implementations."""

    def trade(self, observation: Observation) -> Dict[str, Any]:
        """Execute a trade based on observation."""
        ...


class Algorithm(Enum):
    """Supported training algorithms."""

    PPO = "ppo"


class FeatureSet(Enum):
    """Supported feature sets for training."""

    FULL = "full"
    BASIC = "basic"
    MINIMAL = "minimal"


class Timeframe(Enum):
    """Supported timeframes for training."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


@dataclass
class PPOConfig:
    """Configuration for PPO training."""

    algorithm: Algorithm = Algorithm.PPO
    data_path: str = ""
    total_timesteps: int = DEFAULT_PPO_CONFIG.get("n_steps", 2048) * 100  # Rough estimate
    n_steps: int = DEFAULT_PPO_CONFIG.get("n_steps", 2048)
    gamma: float = DEFAULT_PPO_CONFIG.get("gamma", 0.99)
    vf_coef: float = DEFAULT_PPO_CONFIG.get("vf_coef", 0.5)
    max_grad_norm: float = DEFAULT_PPO_CONFIG.get("max_grad_norm", 0.5)
    ent_coef: float = DEFAULT_PPO_CONFIG.get("ent_coef", 0.0)
    tensorboard_log: str = ""
    model_dir: str = ""
    checkpoint_dir: str = ""
    log_dir: str = ""
    offline_mode: bool = False
    feature_set: FeatureSet = FeatureSet.FULL
    timeframe: Timeframe = Timeframe.M1
    reward_scaling: float = DEFAULT_PPO_CONFIG.get("reward_scaling", 1.0)
    transaction_cost: float = DEFAULT_PPO_CONFIG.get("transaction_cost", 0.0)
    max_position_size: float = DEFAULT_PPO_CONFIG.get("max_position_size", 1.0)
    seed: int = 42
    learning_rate: float = DEFAULT_PPO_CONFIG.get("learning_rate", 3e-4)
    batch_size: int = DEFAULT_PPO_CONFIG.get("batch_size", 64)
    clip_range: float = DEFAULT_PPO_CONFIG.get("clip_range", 0.2)
    gae_lambda: float = DEFAULT_PPO_CONFIG.get("gae_lambda", 0.95)

    @classmethod
    def from_common_config(cls, overrides: Optional[Dict[str, Any]] = None) -> "PPOConfig":
        """Create PPOConfig using DEFAULT_PPO_CONFIG values."""
        config = cls()
        # Copy config as Dict and update with overrides
        common_config: Dict[str, Any] = DEFAULT_PPO_CONFIG.copy()  # type: ignore[assignment]
        if overrides:
            common_config.update(overrides)

        # Apply common PPO config values
        config.n_steps = int(common_config.get("n_steps", 2048))
        config.gamma = float(common_config.get("gamma", 0.99))
        config.vf_coef = float(common_config.get("vf_coef", 0.5))
        config.max_grad_norm = float(common_config.get("max_grad_norm", 0.5))
        config.ent_coef = float(common_config.get("ent_coef", 0.0))
        config.reward_scaling = float(common_config.get("reward_scaling", 1.0))
        config.transaction_cost = float(common_config.get("transaction_cost", 0.0))
        config.max_position_size = float(common_config.get("max_position_size", 1.0))
        config.learning_rate = float(common_config.get("learning_rate", 3e-4))
        config.batch_size = int(common_config.get("batch_size", 64))
        config.clip_range = float(common_config.get("clip_range", 0.2))
        config.gae_lambda = float(common_config.get("gae_lambda", 0.95))

        return config


class PPOTrainerAutoHalt(BaseTrainer, PPOTrainerProtocol):
    """
    PPO Trainer with evaluation gates and auto-halt functionality.

    This trainer extends BaseTrainer with comprehensive PPO training capabilities,
    including automatic evaluation gates, training halt conditions, and progress
    tracking. It supports both standard PPO and custom PPO implementations.

    Features:
    - Evaluation gates for automatic training quality assessment
    - Auto-halt functionality based on gate results
    - Checkpoint management with configurable intervals
    - Progress tracking and statistics collection
    - Custom callback integration for advanced training control

    Args:
        params: TrainerParams containing all training configuration
    """

    def __init__(
        self,
        params: TrainerParams,
    ):
        # Convert PPOConfig to dict for BaseTrainer
        config_dict = dict(params.config)  # PPOConfig is TypedDict, so convert to dict
        
        super().__init__(params)
        self.ppo_config = params.config
        self.model: Optional[CustomPPO] = None

    # Note: The following methods are inherited from BaseTrainer:
    # - start_training, stop_training, update_progress, get_reward_stats
    # - _check_gates_and_halt_if_needed, _should_auto_halt, get_training_status
    # - save_checkpoint, load_checkpoint (from CheckpointMixin)

    def _create_callback(self) -> BaseCallback:
        """
        Create and configure composite training callback.

        Sets up a CompositeTrainingCallback with appropriate configuration
        for PPO training, including progress tracking, gradient probe guards,
        and entropy scheduling based on trainer configuration.

        The callback configuration is determined by:
        - enable_grad_probe_guard: Whether to enable gradient monitoring
        - grad_probe_config: Configuration for gradient probes
        - enable_progress_bar: Always enabled for user feedback
        - enable_entropy_schedule: Disabled (handled elsewhere)

        Returns:
            BaseCallback: Configured CompositeTrainingCallback instance
                         ready for use in PPO training loop.
        """
        # Get grad_probe_guard config from self.config
        enable_grad_probe_guard = self.config.get("enable_grad_probe_guard", False)
        grad_probe_config = self.config.get("grad_probe_config", None)
        
        return CompositeTrainingCallback(
            trainer=self,
            enable_progress_bar=True,
            enable_entropy_schedule=False,
            enable_grad_probe_guard=enable_grad_probe_guard,
            grad_probe_config=grad_probe_config,
            verbose=0,
        )

    def neutralize_policy_bias(self) -> None:
        """Neutralize policy head bias using centralized policy_utils."""
        if self.model is None:
            logger.warning("Model not initialized, cannot neutralize bias")
            return
        
        neutralize_policy_bias(self.model)
        logger.info("Policy head bias neutralized via policy_utils")

    def train(self, session_id: str) -> CustomPPO:
        """
        Execute the complete PPO training pipeline.

        This method orchestrates the entire training process including:
        1. Model initialization (if not already created)
        2. Environment setup with action masking
        3. Policy bias neutralization
        4. Training execution with callbacks
        5. Model return for evaluation/inference

        The training process includes:
        - Data loading from CSV file specified in data_path
        - Environment creation with trading-specific configuration
        - Action masking for valid trading actions only
        - Custom PPO model with bias mitigation features
        - Policy bias neutralization before training
        - Training with progress tracking and evaluation gates
        - TensorBoard logging with session identifier

        Args:
            session_id: Unique identifier for this training session.
                       Used for logging, checkpointing, and TensorBoard naming.

        Returns:
            CustomPPO: Trained PPO model ready for evaluation or inference.
                       The model includes all trained parameters and can be saved/loaded.

        Raises:
            FileNotFoundError: If the data file specified in data_path doesn't exist.
            ValueError: If configuration parameters are invalid.
            RuntimeError: If training fails due to environment or model issues.

        Note:
            This method modifies the trainer's internal state (self.model).
            Subsequent calls will reuse the trained model unless manually reset.
        """
        if self.model is None:
            import pandas as pd

            df = pd.read_csv(self.data_path)
            # Use self.config (Dict) directly, as HeavyTradingEnv expects Dict
            env = HeavyTradingEnv(df=df, config=self.config)

            def mask_fn(env: Any) -> None:
                # Return action mask from environment
                return env.action_mask()

            env = ActionMasker(env, mask_fn)  # type: ignore[arg-type,assignment]
            self.model = CustomPPO(
                policy="MlpPolicy",
                env=env,
                verbose=1,
                # Custom bias mitigation parameters
                enable_pan=True,
                enable_target_entropy=True,
                enable_stratified_sampling=False,
            )
        self.neutralize_policy_bias()
        self.start_training()
        total_timesteps = self.ppo_config.total_timesteps
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self._create_callback(),
            tb_log_name=session_id,
        )
        return self.model


"""
PPO Trainer with auto-halt functionality for training gates.
"""

from typing import Any, Dict, Optional
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

from ztb.training.custom_ppo import CustomPPO
from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.training.eval_gates import EvalGates
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PPOTrainer(BaseTrainer, PPOTrainerProtocol):
    """
    Proximal Policy Optimization (PPO) Trainer with Advanced Features.

    This trainer implements PPO algorithm with evaluation gates, automatic halting,
    checkpointing, and comprehensive monitoring for trading strategy training.

    Key Features:
    - Evaluation Gates: Automatic training validation with configurable thresholds
    - Auto-Halt: Stops training when performance criteria are met or degraded
    - Checkpointing: Regular model saving with configurable intervals
    - Reward Tracking: Maintains rolling history for performance analysis
    - Memory Efficient: Uses deques with maxlen for bounded memory usage

    Training Flow:
    1. Initialize with data, config, and evaluation criteria
    2. Train in episodes with periodic evaluation
    3. Check evaluation gates after each validation
    4. Halt training when gates trigger or manual stop requested
    5. Save final model and training statistics

    Evaluation Gates:
    - Performance thresholds for reward, win rate, max drawdown
    - Early stopping on convergence or degradation
    - Custom gate functions for domain-specific criteria

    Args:
        params: TrainerParams containing all training configuration
    """

    def __init__(  # type: ignore[misc]  # mypy incorrectly reports missing super() for protocol implementations
        self,
        params: TrainerParams,
    ):
        """
        Initialize PPO trainer.

        Args:
            params: TrainerParams containing data path, config, checkpoint directory,
                   evaluation gates, halt callback, and checkpoint interval
        """
        """
        Initialize PPO trainer.

        Args:
            data_path: Path to training data
            config: Training configuration
            checkpoint_dir: Directory for checkpoints
            eval_gates: Evaluation gates for training validation
            halt_callback: Callback function called when training should halt
            checkpoint_interval: Steps between checkpoints
        """
        super().__init__(params)
        self.model: Optional[CustomPPO] = None

    def _create_callback(self) -> BaseCallback:
        """Create composite training callback with progress tracking and entropy scheduling."""
        # Get grad_probe_guard config from self.config
        enable_grad_probe_guard = self.config.get("enable_grad_probe_guard", False)
        grad_probe_config = self.config.get("grad_probe_config", None)
        
        return CompositeTrainingCallback(
            trainer=self,
            enable_progress_bar=True,
            enable_entropy_schedule=True,
            entropy_schedule_type="cosine_decay",
            initial_ent_coef=self.config.get("ent_coef", 0.0),
            final_ent_coef=self.config.get("ent_coef_final", self.config.get("ent_coef", 0.0)),
            enable_grad_probe_guard=enable_grad_probe_guard,
            grad_probe_config=grad_probe_config,
            verbose=0,
        )

    def neutralize_policy_bias(self) -> None:
        """Neutralize policy head bias using centralized policy_utils."""
        if self.model is None:
            logger.warning("Model not initialized, cannot neutralize bias")
            return
        
        neutralize_policy_bias(self.model)
        logger.info("Policy head bias neutralized via policy_utils")

    def _setup_sell_bonus_weighting(self) -> None:
        """Setup action frequency weighting to correct SELL bias."""
        # This will be implemented in the callback to monitor action distribution
        # and apply bonus weighting for underrepresented actions

    def train(self, session_id: str) -> CustomPPO:
        """Train the PPO model."""
        if self.model is None:
            # Load data
            import pandas as pd

            df = pd.read_csv(self.data_path)

            # Create environment
            env = HeavyTradingEnv(df=df, config=self.config)

            # Wrap with ActionMasker for MaskablePPO
            def mask_fn(env: Any) -> Any:
                return env.get_legal_actions().astype(bool)

            env = ActionMasker(env, mask_fn)  # type: ignore[assignment]

            self.model = CustomPPO(
                policy=self.config.get("policy", "MlpPolicy"),
                env=env,
                learning_rate=self.config.get("learning_rate", 3e-4),
                n_steps=self.config.get("n_steps", 2048),
                batch_size=self.config.get("batch_size", 64),
                n_epochs=self.config.get("n_epochs", 10),
                gamma=self.config.get("gamma", 0.99),
                gae_lambda=self.config.get("gae_lambda", 0.95),
                clip_range=self.config.get("clip_range", 0.2),
                clip_range_vf=self.config.get("clip_range_vf"),
                normalize_advantage=self.config.get("normalize_advantage", True),
                ent_coef=self.config.get("ent_coef", 0.0),
                vf_coef=self.config.get("vf_coef", 0.5),
                max_grad_norm=self.config.get("max_grad_norm", 0.5),
                target_kl=self.config.get("target_kl"),
                tensorboard_log=self.config.get("tensorboard_log"),
                policy_kwargs=self.config.get("policy_kwargs"),
                verbose=self.config.get("verbose", 1),
                seed=self.config.get("seed"),
                device=self.config.get("device", "auto"),
                _init_setup_model=self.config.get("_init_setup_model", True),
                # Custom bias mitigation parameters
                enable_pan=True,
                enable_target_entropy=True,
                enable_stratified_sampling=False,
            )

        # Neutralize policy bias
        self.neutralize_policy_bias()

        # Add action frequency weighting for SELL bias correction
        self._setup_sell_bonus_weighting()

        # Start training session
        self.start_training()

        # Train the model
        total_timesteps = self.config.get("total_timesteps", 100000)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self._create_callback(),
            tb_log_name=session_id,
        )

        return self.model
