"""Trainer parameter definitions for interface unification."""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union

from ztb.training.evaluation.eval_gates import EvalGates
from ztb.training.config.ppo_config import PPOConfig


@dataclass
class TrainerParams:
    """
    Common trainer parameters for all PPO trainers.

    This dataclass encapsulates all the parameters needed to initialize
    any PPO-based trainer in the system, providing a unified interface
    for trainer configuration.

    Args:
        data_path: Path to the training dataset file
        config: PPOConfig containing algorithm hyperparameters
        checkpoint_dir: Directory path for saving model checkpoints
        eval_gates: Optional evaluation gates for training validation
        halt_callback: Optional callback function called when training halts
        checkpoint_interval: Number of steps between checkpoint saves
    """

    data_path: str
    config: PPOConfig
    checkpoint_dir: str
    eval_gates: Optional[EvalGates] = None
    halt_callback: Optional[Callable[[str], None]] = None
    checkpoint_interval: int = 10000
    progress_bar: bool = True


@dataclass
class SELLMitigationParams(TrainerParams):
    """
    Extended parameters for SELL bias mitigation trainer.

    This dataclass extends TrainerParams with additional configuration
    options specific to SELL bias mitigation techniques, including
    Lagrange constraints, gradient probes, action weighting, and
    advanced normalization methods.

    Args:
        enable_lagrange: Enable Lagrange constraint for minimum action rate
        enable_probes: Enable gradient probes for monitoring and failsafe
        enable_weights: Enable enhanced action weighting
        enable_pan: Enable Per-Action Advantage Normalization
        enable_target_entropy: Enable automatic entropy control
        enable_stratified_sampling: Enable stratified mini-batch sampling
        allow_reverse: Enable Reverse-as-Close flag for environment config
        probe_csv_path: Optional path for probe CSV output
        lagrange_params: Optional dict with Lagrange constraint parameters
            (r_target, tolerance, eta, lambda_max, warmup_steps)
    """

    enable_lagrange: bool = True
    enable_probes: bool = True
    enable_weights: bool = True
    enable_pan: bool = True  # Per-Action Advantage Normalization
    enable_target_entropy: bool = True  # Automatic entropy control
    enable_stratified_sampling: bool = True  # Stratified mini-batch
    allow_reverse: bool = False  # Reverse-as-Close flag (env config)
    probe_csv_path: Optional[str] = None
    lagrange_params: Optional[Dict[str, Union[int, float]]] = None