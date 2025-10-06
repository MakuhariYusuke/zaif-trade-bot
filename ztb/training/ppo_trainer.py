"""
PPO Trainer implementations:
- PPOTrainer: 標準的な1Mタイムステップ学習・評価ゲート・メモリ最適化付き
- PPOTrainerAutoHalt: auto-halt/gate機能付き（旧trading/ppo_trainer.py由来）
"""

# 既存のPPOTrainerクラスはそのまま残す

# --- 以下、trading/ppo_trainer.pyのPPOTrainerをPPOTrainerAutoHaltとして移植 ---

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Protocol

import numpy as np
from numpy.typing import NDArray
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.training.eval_gates import EvalGates, GateResult, GateStatus
from ztb.utils.file_utils import safe_json_dump, safe_json_load
from ztb.utils.logging_utils import get_logger
from ztb.utils.path_utils import ensure_dir

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
    total_timesteps: int = 1000000
    n_steps: int = 2048
    gamma: float = 0.99
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    ent_coef: float = 0.0
    tensorboard_log: str = ""
    model_dir: str = ""
    checkpoint_dir: str = ""
    log_dir: str = ""
    offline_mode: bool = False
    feature_set: FeatureSet = FeatureSet.FULL
    timeframe: Timeframe = Timeframe.M1
    reward_scaling: float = 1.0
    transaction_cost: float = 0.0
    max_position_size: float = 1.0
    seed: int = 42
    learning_rate: float = 3e-4
    batch_size: int = 64
    clip_range: float = 0.2
    gae_lambda: float = 0.95

    def as_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for compatibility."""
        return asdict(self)


class PPOTrainerAutoHalt(PPOTrainerProtocol):
    """PPO Trainer with evaluation gates and auto-halt functionality (旧trading/ppo_trainer.py)."""
    def __init__(
        self,
        data_path: str,
        config: PPOConfig,
        checkpoint_dir: str,
        eval_gates: Optional[EvalGates] = None,
        halt_callback: Optional[Callable[[str], None]] = None,
        checkpoint_interval: int = 10000,
    ):
        self.eval_gates = eval_gates or EvalGates()
        self.halt_callback = halt_callback
        self.checkpoint_interval = checkpoint_interval
        self.data_path = data_path
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.model: Optional[MaskablePPO] = None
        # Training state
        self.current_step = 0
        self.rewards_history: deque[float] = deque(maxlen=50000)
        self.steps_history: deque[int] = deque(maxlen=50000)
        self.is_training = False
        self.halt_reason: Optional[str] = None
        # Statistics for efficiency
        self.reward_sum = 0.0
        self.reward_count = 0
        self.reward_mean = 0.0
        self.reward_m2 = 0.0
        # Auto-halt state
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        self.last_gate_check_step = 0

    def start_training(self) -> None:
        self.is_training = True
        self.halt_reason = None
        self.consecutive_failures = 0
        self.last_gate_check_step = 0
        logger.info("Training started")

    def stop_training(self, reason: str = "Manual stop") -> None:
        self.is_training = False
        self.halt_reason = reason
        logger.info(f"Training stopped: {reason}")
        if self.halt_callback:
            self.halt_callback(reason)

    def update_progress(self, step: int, reward: float) -> None:
        if not self.is_training:
            return
        self.current_step = step
        self.rewards_history.append(reward)
        self.steps_history.append(step)
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_m2 += delta * delta2
        if step - self.last_gate_check_step >= self.checkpoint_interval:
            self._check_gates_and_halt_if_needed()

    def get_reward_stats(self) -> Dict[str, float]:
        if self.reward_count == 0:
            return {"mean": 0.0, "variance": 0.0, "std": 0.0, "count": 0}
        variance = self.reward_m2 / self.reward_count if self.reward_count > 1 else 0.0
        std = variance**0.5
        return {
            "mean": self.reward_mean,
            "variance": variance,
            "std": std,
            "count": self.reward_count,
        }

    def _check_gates_and_halt_if_needed(self) -> None:
        if not self.eval_gates.enabled:
            return
        gate_results = self.eval_gates.evaluate_all(
            rewards=self.rewards_history,
            steps=self.steps_history,
            final_eval_reward=self.rewards_history[-1] if self.rewards_history else 0.0,
        )
        failed_gates = [r for r in gate_results.values() if r.status == GateStatus.FAIL]
        if failed_gates:
            self.consecutive_failures += 1
            logger.warning(f"Gate check failed: {len(failed_gates)} gates failed")
            if self._should_auto_halt(gate_results):
                self.stop_training("Auto-halt: gate failure")
        else:
            self.consecutive_failures = 0
        self.last_gate_check_step = self.current_step

    def _should_auto_halt(self, gate_results: Dict[str, GateResult]) -> bool:
        if not gate_results:
            return False
        critical_gates = ["memory_rss", "no_dup_steps"]
        for gate_name in critical_gates:
            if (
                gate_name in gate_results
                and gate_results[gate_name].status == GateStatus.FAIL
            ):
                return True
        if self.consecutive_failures >= self.max_consecutive_failures:
            logger.error(f"Too many consecutive failures: {self.consecutive_failures}")
            return True
        if "reward_trend_300k" in gate_results:
            trend_result = gate_results["reward_trend_300k"]
            if (
                trend_result.status == GateStatus.FAIL
                and self.consecutive_failures >= 2
            ):
                return True
        return False

    def get_training_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "is_training": self.is_training,
            "current_step": self.current_step,
            "halt_reason": self.halt_reason,
            "consecutive_failures": self.consecutive_failures,
            "reward_stats": self.get_reward_stats(),
        }
        if self.eval_gates.enabled and self.rewards_history:
            stats = self.get_reward_stats()
            gate_results = self.eval_gates.evaluate_all(
                rewards=self.rewards_history,
                steps=self.steps_history,
                final_eval_reward=(
                    self.rewards_history[-1] if self.rewards_history else 0.0
                ),
                reward_stats=stats,
            )
            status["gate_results"] = {
                name: {
                    "status": result.status.value,
                    "reason": result.reason,
                    "value": result.value,
                }
                for name, result in gate_results.items()
            }
        return status

    def save_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint_data = {
            "current_step": self.current_step,
            "rewards_history": list(self.rewards_history),
            "steps_history": list(self.steps_history),
            "consecutive_failures": self.consecutive_failures,
            "last_gate_check_step": self.last_gate_check_step,
            "halt_reason": self.halt_reason,
            "is_training": self.is_training,
        }
        ensure_dir(Path(checkpoint_path).parent)
        safe_json_dump(checkpoint_data, Path(checkpoint_path), indent=2)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        if not Path(checkpoint_path).exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        checkpoint_data = safe_json_load(Path(checkpoint_path))
        self.current_step = checkpoint_data.get("current_step", 0)
        self.rewards_history = checkpoint_data.get("rewards_history", [])
        self.steps_history = checkpoint_data.get("steps_history", [])
        self.consecutive_failures = checkpoint_data.get("consecutive_failures", 0)
        self.last_gate_check_step = checkpoint_data.get("last_gate_check_step", 0)
        self.halt_reason = checkpoint_data.get("halt_reason")

    def _create_callback(self) -> BaseCallback:
        class TrainingCallback(BaseCallback):
            def __init__(self, trainer: "PPOTrainerAutoHalt"):
                super().__init__()
                self.trainer = trainer
            def _on_step(self) -> bool:
                # ここで進捗を更新
                self.trainer.update_progress(self.locals["n_calls"], self.locals.get("rewards", [0])[-1])
                return True
        return TrainingCallback(self)

    def neutralize_policy_bias(self) -> None:
        if self.model is None:
            logger.warning("Model not initialized, cannot neutralize bias")
            return

        # For MaskablePPO, try different policy structures
        policy = self.model.policy

        # Try policy_net
        if hasattr(policy, 'policy_net'):
            policy_head = policy.policy_net[-1] if isinstance(policy.policy_net, list) else policy.policy_net
            if hasattr(policy_head, 'bias') and getattr(policy_head, 'bias', None) is not None:
                bias = getattr(policy_head, 'bias')
                if hasattr(bias, 'data'):
                    bias.data.zero_()
                    logger.info("Policy head bias neutralized (policy_net)")
                    return

        # Try mlp_extractor
        if hasattr(policy, 'mlp_extractor') and hasattr(policy.mlp_extractor, 'policy_net'):
            policy_head = policy.mlp_extractor.policy_net[-1]
            if hasattr(policy_head, 'bias') and getattr(policy_head, 'bias', None) is not None:
                bias = getattr(policy_head, 'bias')
                if hasattr(bias, 'data'):
                    bias.data.zero_()
                    logger.info("Policy head bias neutralized (mlp_extractor)")
                    return

        # Try action_net
        if hasattr(policy, 'action_net'):
            policy_head = policy.action_net[-1] if isinstance(policy.action_net, list) else policy.action_net
            if hasattr(policy_head, 'bias') and policy_head.bias is not None:
                policy_head.bias.data.zero_()  # type: ignore[operator]
                logger.info("Policy head bias neutralized (action_net)")
                return

        logger.warning("Policy head bias not found - tried policy_net, mlp_extractor, action_net")

    def train(self, session_id: str) -> MaskablePPO:
        if self.model is None:
            import pandas as pd
            df = pd.read_csv(self.data_path)
            env = HeavyTradingEnv(df=df, config=self.config.as_dict())
            def mask_fn(env: Any) -> None:
                # 必要に応じてaction maskを返す関数を実装
                return None
            env = ActionMasker(env, mask_fn)  # type: ignore[arg-type,assignment]
            self.model = MaskablePPO("MlpPolicy", env, verbose=1)
        self.neutralize_policy_bias()
        self.start_training()
        total_timesteps = self.config.total_timesteps
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self._create_callback(),
            tb_log_name=session_id,
        )
        return self.model
"""
PPO Trainer with auto-halt functionality for training gates.
"""

import math
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from rich.progress import Progress, TaskID
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.training.eval_gates import EvalGates, GateResult, GateStatus
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PPOTrainer(PPOTrainerProtocol):
    """PPO Trainer with evaluation gates and auto-halt functionality."""

    def __init__(  # type: ignore[misc]  # mypy incorrectly reports missing super() for protocol implementations
        self,
        data_path: str,
        config: Dict[str, Any],
        checkpoint_dir: str,
        eval_gates: Optional[EvalGates] = None,
        halt_callback: Optional[Callable[[str], None]] = None,
        checkpoint_interval: int = 10000,
    ):
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
        self.eval_gates = eval_gates or EvalGates()
        self.halt_callback = halt_callback
        self.checkpoint_interval = checkpoint_interval
        self.data_path = data_path
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.model: Optional[MaskablePPO] = None

        # Training state
        self.current_step = 0
        self.rewards_history: deque[float] = deque(
            maxlen=50000
        )  # Keep last 50k rewards for efficiency
        self.steps_history: deque[int] = deque(maxlen=50000)  # Keep last 50k steps
        self.is_training = False
        self.halt_reason: Optional[str] = None

        # Statistics for efficiency
        self.reward_sum = 0.0
        self.reward_count = 0
        self.reward_mean = 0.0
        self.reward_m2 = 0.0  # For variance calculation

        # Auto-halt state
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        self.last_gate_check_step = 0

    def start_training(self) -> None:
        """Start training session."""
        self.is_training = True
        self.halt_reason = None
        self.consecutive_failures = 0
        self.last_gate_check_step = 0
        logger.info("Training started")

    def stop_training(self, reason: str = "Manual stop") -> None:
        """Stop training session."""
        self.is_training = False
        self.halt_reason = reason
        logger.info(f"Training stopped: {reason}")

        if self.halt_callback:
            self.halt_callback(reason)

    def update_progress(self, step: int, reward: float) -> None:
        """
        Update training progress and check gates.

        Args:
            step: Current training step
            reward: Current episode reward
        """
        if not self.is_training:
            return

        self.current_step = step

        # Update history (deque automatically manages size)
        self.rewards_history.append(reward)
        self.steps_history.append(step)

        # Update online statistics (Welford's algorithm)
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_m2 += delta * delta2

        # Check gates periodically
        if step - self.last_gate_check_step >= self.checkpoint_interval:
            self._check_gates_and_halt_if_needed()

    def get_reward_stats(self) -> Dict[str, float]:
        """
        Get reward statistics efficiently.

        Returns:
            Dictionary with mean, variance, std, count
        """
        if self.reward_count == 0:
            return {"mean": 0.0, "variance": 0.0, "std": 0.0, "count": 0}

        variance = self.reward_m2 / self.reward_count if self.reward_count > 1 else 0.0
        std = variance**0.5

        return {
            "mean": self.reward_mean,
            "variance": variance,
            "std": std,
            "count": self.reward_count,
        }

    def _check_gates_and_halt_if_needed(self) -> None:
        """Check evaluation gates and halt training if necessary."""
        if not self.eval_gates.enabled:
            return

        # Run gate checks
        gate_results = self.eval_gates.evaluate_all(
            rewards=self.rewards_history,
            steps=self.steps_history,
            final_eval_reward=self.rewards_history[-1] if self.rewards_history else 0.0,
        )

        # Count failures
        failed_gates = [r for r in gate_results.values() if r.status == GateStatus.FAIL]

        if failed_gates:
            self.consecutive_failures += 1
            logger.warning(f"Gate check failed: {len(failed_gates)} gates failed")

            # Check if we should auto-halt
            if self._should_auto_halt(gate_results):
                reasons = [f"{r.name}: {r.reason}" for r in failed_gates]
                halt_reason = f"Auto-halt: {len(failed_gates)} gates failed - {', '.join(reasons)}"
                self.stop_training(halt_reason)
                return
        else:
            # Reset consecutive failures on success
            self.consecutive_failures = 0

        self.last_gate_check_step = self.current_step

    def _should_auto_halt(self, gate_results: Dict[str, GateResult]) -> bool:
        """
        Determine if training should auto-halt based on gate results.

        Auto-halt conditions:
        1. Critical gates fail (memory, duplicate steps)
        2. Consecutive failures exceed threshold
        3. Reward trend is consistently negative

        Args:
            gate_results: Results from gate evaluation

        Returns:
            True if training should halt
        """
        if not gate_results:
            return False

        # Critical failures that should always halt
        critical_gates = ["memory_rss", "no_dup_steps"]
        for gate_name in critical_gates:
            if (
                gate_name in gate_results
                and gate_results[gate_name].status == GateStatus.FAIL
            ):
                logger.error(f"Critical gate failed: {gate_name}")
                return True

        # Consecutive failures
        if self.consecutive_failures >= self.max_consecutive_failures:
            logger.error(f"Too many consecutive failures: {self.consecutive_failures}")
            return True

        # Persistent negative reward trend
        if "reward_trend_300k" in gate_results:
            trend_result = gate_results["reward_trend_300k"]
            if (
                trend_result.status == GateStatus.FAIL
                and self.consecutive_failures >= 2
            ):
                logger.error("Persistent negative reward trend")
                return True

        return False

    def get_training_status(self) -> Dict[str, Any]:
        """
        Get current training status.

        Returns:
            Status dictionary with training state and gate results
        """
        status: Dict[str, Any] = {
            "is_training": self.is_training,
            "current_step": self.current_step,
            "halt_reason": self.halt_reason,
            "consecutive_failures": self.consecutive_failures,
            "reward_stats": self.get_reward_stats(),
        }

        if self.eval_gates.enabled and self.rewards_history:
            stats = self.get_reward_stats()
            gate_results = self.eval_gates.evaluate_all(
                rewards=list(self.rewards_history),
                steps=list(self.steps_history),
                final_eval_reward=(
                    self.rewards_history[-1] if self.rewards_history else 0.0
                ),
                reward_stats=stats,  # Pass statistics for efficiency
            )
            status["gate_results"] = {
                name: {
                    "status": result.status.value,
                    "reason": result.reason,
                    "value": result.value,
                }
                for name, result in gate_results.items()
            }

        return status

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save training checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint
        """
        checkpoint_data = {
            "current_step": self.current_step,
            "rewards_history": list(
                self.rewards_history
            ),  # Convert deque to list for JSON serialization
            "steps_history": list(
                self.steps_history
            ),  # Convert deque to list for JSON serialization
            "consecutive_failures": self.consecutive_failures,
            "last_gate_check_step": self.last_gate_check_step,
            "halt_reason": self.halt_reason,
            "is_training": self.is_training,
        }

        # Save to file (simplified - in real implementation would use proper serialization)
        ensure_dir(Path(checkpoint_path).parent)
        safe_json_dump(checkpoint_data, Path(checkpoint_path), indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to load checkpoint from
        """
        if not Path(checkpoint_path).exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return

        checkpoint_data = safe_json_load(Path(checkpoint_path))

        self.current_step = checkpoint_data.get("current_step", 0)
        self.rewards_history = checkpoint_data.get("rewards_history", [])
        self.steps_history = checkpoint_data.get("steps_history", [])
        self.consecutive_failures = checkpoint_data.get("consecutive_failures", 0)
        self.last_gate_check_step = checkpoint_data.get("last_gate_check_step", 0)
        self.halt_reason = checkpoint_data.get("halt_reason")

    def _create_callback(self) -> BaseCallback:
        """Create training callback."""
        class TrainingCallback(BaseCallback):
            def __init__(self, trainer: "PPOTrainer"):
                super().__init__()
                self.trainer = trainer
                self.progress: Optional[Progress] = None
                self.task_id: Optional[TaskID] = None

            def _on_training_start(self) -> None:
                """Initialize progress bar when training starts."""
                try:
                    from rich.console import Console
                    console = Console()
                    self.progress = Progress(console=console)
                    total_timesteps = self.trainer.config.get("total_timesteps", 100000)
                    self.task_id = self.progress.add_task(
                        "[green]Training PPO...",
                        total=total_timesteps,
                        completed=0
                    )
                    self.progress.start()
                except ImportError:
                    logger.warning("Rich not available, progress bar disabled")

            def _on_step(self) -> bool:
                if self.locals.get("done"):
                    reward = self.locals.get("rewards", 0)
                    self.trainer.update_progress(self.num_timesteps, reward)

                # Update progress bar
                if self.progress and self.task_id is not None:
                    self.progress.update(self.task_id, completed=self.num_timesteps)

                # Update entropy coefficient with cosine decay if configured
                ent_coef_schedule = self.trainer.config.get("ent_coef_schedule")
                if ent_coef_schedule == "cosine_decay":
                    total_timesteps = self.trainer.config.get("total_timesteps", 100000)
                    ent_coef_initial = self.trainer.config.get("ent_coef", 0.0)
                    ent_coef_final = self.trainer.config.get("ent_coef_final", ent_coef_initial)

                    # Cosine decay: ent_coef = ent_coef_final + (ent_coef_initial - ent_coef_final) * (1 + cos(pi * t / T)) / 2
                    progress = min(self.num_timesteps / total_timesteps, 1.0)
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                    new_ent_coef = ent_coef_final + (ent_coef_initial - ent_coef_final) * cosine_decay

                    # Update model's entropy coefficient
                    if hasattr(self.model, 'ent_coef'):
                        setattr(self.model, 'ent_coef', new_ent_coef)

                return not self.trainer.halt_reason

            def _on_training_end(self) -> None:
                """Clean up progress bar when training ends."""
                if self.progress:
                    self.progress.stop()

        return TrainingCallback(self)

    def neutralize_policy_bias(self) -> None:
        """policy headのbiasを中立化"""
        if self.model is None:
            logger.warning("Model not initialized, cannot neutralize bias")
            return

        # For MaskablePPO, try different policy structures
        policy = self.model.policy

        # Try policy_net
        if hasattr(policy, 'policy_net'):
            policy_head = policy.policy_net[-1] if isinstance(policy.policy_net, list) else policy.policy_net
            if hasattr(policy_head, 'bias') and getattr(policy_head, 'bias', None) is not None:
                bias = getattr(policy_head, 'bias')
                if hasattr(bias, 'data'):
                    bias.data.zero_()
                    logger.info("Policy head bias neutralized (policy_net)")
                    return

        # Try mlp_extractor
        if hasattr(policy, 'mlp_extractor') and hasattr(policy.mlp_extractor, 'policy_net'):
            policy_head = policy.mlp_extractor.policy_net[-1]
            if hasattr(policy_head, 'bias') and getattr(policy_head, 'bias', None) is not None:
                bias = getattr(policy_head, 'bias')
                if hasattr(bias, 'data'):
                    bias.data.zero_()
                    logger.info("Policy head bias neutralized (mlp_extractor)")
                    return

        # Try action_net
        if hasattr(policy, 'action_net'):
            policy_head = policy.action_net[-1] if isinstance(policy.action_net, list) else policy.action_net
            if hasattr(policy_head, 'bias') and policy_head.bias is not None:
                policy_head.bias.data.zero_()  # type: ignore[operator]
                logger.info("Policy head bias neutralized (action_net)")
                return

        logger.warning("Policy head bias not found - tried policy_net, mlp_extractor, action_net")

    def _setup_sell_bonus_weighting(self) -> None:
        """Setup action frequency weighting to correct SELL bias."""
        # This will be implemented in the callback to monitor action distribution
        # and apply bonus weighting for underrepresented actions
        pass

    def train(self, session_id: str) -> MaskablePPO:
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
            
            self.model = MaskablePPO(
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
