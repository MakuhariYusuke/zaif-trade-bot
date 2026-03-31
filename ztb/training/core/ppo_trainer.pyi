"""Type stubs for PPO trainer implementations."""

from typing import Any, Dict, Optional, Protocol

from sb3_contrib import MaskablePPO
from ztb.training.config.trainer_params import TrainerParams


class PPOTrainingConfig:
    """Runtime training config dataclass."""

    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    clip_range_vf: float | None
    normalize_advantage: bool
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    use_sde: bool
    sde_sample_freq: int
    target_kl: float | None
    verbose: int
    use_custom_ppo: bool

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PPOTrainingConfig": ...

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

class PPOTrainerAutoHalt:
    """Auto-halt capable PPO trainer."""

    data_path: str
    training_config: PPOTrainingConfig

    def __init__(self, params: TrainerParams) -> None: ...
    def train(self, session_id: str) -> Optional[MaskablePPO]:
        """Train the model with auto-halt capability."""
        ...
    def get_reward_stats(self) -> Dict[str, float]:
        """Get training reward statistics."""
        ...
    def neutralize_policy_bias(self) -> None:
        """Neutralize policy head bias."""
        ...


class PPOTrainer(PPOTrainerAutoHalt):
    """Standard PPO trainer compatibility surface."""

    def __init__(
        self,
        data_path: str,
        config: Dict[str, Any],
        checkpoint_dir: str,
        max_features: int | None = None,
    ) -> None: ...
