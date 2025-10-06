#!/usr/bin/env python3
"""
Configuration Fingerprinting for Training/Evaluation Consistency.

Generates a deterministic hash of all environment/training/inference settings
to ensure training and evaluation use identical configurations.

Usage:
    # During training
    fingerprint = ConfigFingerprint.from_training(env_config, trainer_config)
    fingerprint.save("models/model_name/config_fingerprint.json")

    # During evaluation
    eval_fingerprint = ConfigFingerprint.from_evaluation(env_config, inference_config)
    saved_fingerprint = ConfigFingerprint.load("models/model_name/config_fingerprint.json")

    if not saved_fingerprint.matches(eval_fingerprint):
        print("Config mismatch!")
        print(saved_fingerprint.diff(eval_fingerprint))
        raise ValueError("Training/evaluation config mismatch")
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ConfigFingerprint:
    """Configuration fingerprint for reproducibility."""

    # Environment settings
    initial_portfolio_value: float
    transaction_cost: float
    max_position_size: float
    risk_free_rate: float
    timeframe: str
    feature_set: str
    curriculum_stage: str

    # Reward settings
    reward_scaling: float
    reward_settings: Dict[str, Any] = field(default_factory=dict)

    # Training settings (optional, only for training phase)
    learning_rate: Optional[float] = None
    n_steps: Optional[int] = None
    batch_size: Optional[int] = None
    n_epochs: Optional[int] = None
    gamma: Optional[float] = None
    gae_lambda: Optional[float] = None
    clip_range: Optional[float] = None
    ent_coef: Optional[float] = None
    vf_coef: Optional[float] = None
    max_grad_norm: Optional[float] = None

    # Inference settings
    deterministic: bool = True
    temperature: float = 1.0

    # Normalization settings
    normalize_observations: bool = False
    normalize_rewards: bool = False

    # Version info
    fingerprint_version: str = "1.0"

    @classmethod
    def from_training(
        cls,
        env_config: Any,
        trainer_config: Dict[str, Any],
    ) -> "ConfigFingerprint":
        """Create fingerprint from training configuration.

        Args:
            env_config: Environment configuration object
            trainer_config: Trainer configuration dict

        Returns:
            ConfigFingerprint instance
        """
        return cls(
            # Environment
            initial_portfolio_value=float(env_config.initial_portfolio_value),
            transaction_cost=float(env_config.transaction_cost),
            max_position_size=float(env_config.max_position_size),
            risk_free_rate=float(env_config.risk_free_rate),
            timeframe=str(env_config.timeframe),
            feature_set=str(env_config.feature_set),
            curriculum_stage=str(env_config.curriculum_stage),
            # Reward
            reward_scaling=float(env_config.reward_scaling),
            reward_settings=(
                dict(env_config.reward_settings)
                if hasattr(env_config, "reward_settings")
                else {}
            ),
            # Training
            learning_rate=trainer_config.get("learning_rate"),
            n_steps=trainer_config.get("n_steps"),
            batch_size=trainer_config.get("batch_size"),
            n_epochs=trainer_config.get("n_epochs"),
            gamma=trainer_config.get("gamma"),
            gae_lambda=trainer_config.get("gae_lambda"),
            clip_range=trainer_config.get("clip_range"),
            ent_coef=trainer_config.get("ent_coef"),
            vf_coef=trainer_config.get("vf_coef"),
            max_grad_norm=trainer_config.get("max_grad_norm"),
            # Inference (defaults for training)
            deterministic=True,
            temperature=1.0,
            # Normalization
            normalize_observations=trainer_config.get("normalize_observations", False),
            normalize_rewards=trainer_config.get("normalize_rewards", False),
        )

    @classmethod
    def from_evaluation(
        cls,
        env_config: Any,
        deterministic: bool = True,
        temperature: float = 1.0,
    ) -> "ConfigFingerprint":
        """Create fingerprint from evaluation configuration.

        Args:
            env_config: Environment configuration object
            deterministic: Whether to use deterministic action selection
            temperature: Temperature for action sampling

        Returns:
            ConfigFingerprint instance (training fields will be None)
        """
        return cls(
            # Environment (must match training)
            initial_portfolio_value=float(env_config.initial_portfolio_value),
            transaction_cost=float(env_config.transaction_cost),
            max_position_size=float(env_config.max_position_size),
            risk_free_rate=float(env_config.risk_free_rate),
            timeframe=str(env_config.timeframe),
            feature_set=str(env_config.feature_set),
            curriculum_stage=str(env_config.curriculum_stage),
            # Reward (must match training)
            reward_scaling=float(env_config.reward_scaling),
            reward_settings=(
                dict(env_config.reward_settings)
                if hasattr(env_config, "reward_settings")
                else {}
            ),
            # Inference
            deterministic=deterministic,
            temperature=temperature,
            # Normalization
            normalize_observations=getattr(env_config, "normalize_observations", False),
            normalize_rewards=getattr(env_config, "normalize_rewards", False),
        )

    def compute_hash(self, include_training_params: bool = False) -> str:
        """Compute deterministic hash of configuration.

        Args:
            include_training_params: Whether to include training-specific params
                                    (should be False for training/eval comparison)

        Returns:
            SHA256 hash of configuration
        """
        config_dict = asdict(self)

        # Exclude training params if requested (for train/eval comparison)
        if not include_training_params:
            training_keys = [
                "learning_rate",
                "n_steps",
                "batch_size",
                "n_epochs",
                "gamma",
                "gae_lambda",
                "clip_range",
                "ent_coef",
                "vf_coef",
                "max_grad_norm",
            ]
            for key in training_keys:
                config_dict.pop(key, None)

        # Remove version (not part of actual config)
        config_dict.pop("fingerprint_version", None)

        # Sort keys for deterministic hashing
        config_json = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_json.encode()).hexdigest()

    def matches(
        self,
        other: "ConfigFingerprint",
        ignore_inference_params: bool = False,
    ) -> bool:
        """Check if two fingerprints match.

        Args:
            other: Other fingerprint to compare
            ignore_inference_params: Whether to ignore deterministic/temperature
                                    (useful when comparing with different inference modes)

        Returns:
            True if fingerprints match (environment/reward settings identical)
        """
        # Compare hashes (excluding training params)
        self_hash = self.compute_hash(include_training_params=False)
        other_hash = other.compute_hash(include_training_params=False)

        if ignore_inference_params:
            # Create copies without inference params
            self_dict = asdict(self)
            other_dict = asdict(other)

            for key in ["deterministic", "temperature"]:
                self_dict.pop(key, None)
                other_dict.pop(key, None)

            self_json = json.dumps(self_dict, sort_keys=True)
            other_json = json.dumps(other_dict, sort_keys=True)

            self_hash = hashlib.sha256(self_json.encode()).hexdigest()
            other_hash = hashlib.sha256(other_json.encode()).hexdigest()

        return self_hash == other_hash

    def diff(self, other: "ConfigFingerprint") -> Dict[str, tuple]:
        """Compute differences between two fingerprints.

        Args:
            other: Other fingerprint to compare

        Returns:
            Dict of {field: (self_value, other_value)} for differing fields
        """
        self_dict = asdict(self)
        other_dict = asdict(other)

        diffs = {}
        all_keys = set(self_dict.keys()) | set(other_dict.keys())

        for key in all_keys:
            self_val = self_dict.get(key)
            other_val = other_dict.get(key)

            if self_val != other_val:
                diffs[key] = (self_val, other_val)

        return diffs

    def save(self, path: Path | str) -> None:
        """Save fingerprint to JSON file.

        Args:
            path: Path to save fingerprint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Path | str) -> "ConfigFingerprint":
        """Load fingerprint from JSON file.

        Args:
            path: Path to load fingerprint from

        Returns:
            ConfigFingerprint instance
        """
        with open(path, "r") as f:
            data = json.load(f)

        return cls(**data)

    def summary(self) -> str:
        """Generate human-readable summary.

        Returns:
            Multi-line string summary
        """
        lines = [
            "Configuration Fingerprint",
            "=" * 50,
            f"Hash: {self.compute_hash()[:16]}...",
            "",
            "Environment:",
            f"  Initial portfolio: ${self.initial_portfolio_value:,.2f}",
            f"  Transaction cost: {self.transaction_cost:.4f}",
            f"  Max position size: {self.max_position_size}",
            f"  Risk-free rate: {self.risk_free_rate:.4f}",
            f"  Timeframe: {self.timeframe}",
            f"  Feature set: {self.feature_set}",
            f"  Curriculum stage: {self.curriculum_stage}",
            "",
            "Reward:",
            f"  Scaling: {self.reward_scaling}",
            f"  Custom settings: {len(self.reward_settings)} params",
            "",
            "Inference:",
            f"  Deterministic: {self.deterministic}",
            f"  Temperature: {self.temperature}",
            "",
            "Normalization:",
            f"  Observations: {self.normalize_observations}",
            f"  Rewards: {self.normalize_rewards}",
        ]

        if self.learning_rate is not None:
            lines.extend(
                [
                    "",
                    "Training:",
                    f"  Learning rate: {self.learning_rate}",
                    f"  Steps: {self.n_steps}",
                    f"  Batch size: {self.batch_size}",
                    f"  Epochs: {self.n_epochs}",
                    f"  Gamma: {self.gamma}",
                    f"  GAE lambda: {self.gae_lambda}",
                    f"  Clip range: {self.clip_range}",
                    f"  Ent coef: {self.ent_coef}",
                    f"  VF coef: {self.vf_coef}",
                    f"  Max grad norm: {self.max_grad_norm}",
                ]
            )

        return "\n".join(lines)
