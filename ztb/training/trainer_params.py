"""Trainer parameter definitions for interface unification."""

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ztb.training.eval_gates import EvalGates
from ztb.training.ppo_config import PPOConfig


@dataclass
class TrainerParams:
    """Common trainer parameters for all PPO trainers."""

    data_path: str
    config: PPOConfig
    checkpoint_dir: str
    eval_gates: Optional[EvalGates] = None
    halt_callback: Optional[Callable[[str], None]] = None
    checkpoint_interval: int = 10000


@dataclass
class SELLMitigationParams(TrainerParams):
    """Extended parameters for SELL bias mitigation trainer."""

    enable_lagrange: bool = True
    enable_probes: bool = True
    enable_weights: bool = True
    enable_pan: bool = True  # Per-Action Advantage Normalization
    enable_target_entropy: bool = True  # Automatic entropy control
    enable_stratified_sampling: bool = True  # Stratified mini-batch
    allow_reverse: bool = False  # Reverse-as-Close flag (env config)
    probe_csv_path: Optional[str] = None