"""PPO sidecar foundation config helpers.

675# の PPO sidecar 設計を、そのまま scheduler 実装に落とせる
最小 config 契約として整理する。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ztb.utils.safety import ensure_dict, safe_to_bool, safe_to_float, safe_to_int


@dataclass(frozen=True, slots=True)
class PPOSidecarConfig:
    """PPO sidecar retrain / signal 更新の最小設定."""

    data_path: str
    model_path: Path = field(default_factory=lambda: Path("models/v461/ppo_sidecar.zip"))
    signal_path: Path = field(default_factory=lambda: Path("cache/ppo_sidecar_signal.json"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("models/v461"))
    total_timesteps: int = 200_000
    incremental_timesteps: int = 50_000
    check_interval_sec: int = 300
    retrain_interval_sec: int = 7200
    min_override_confidence: float = 0.55
    min_action_probability_gap: float = 0.10
    use_continuous_actions: bool = False
    action_space_type: str = "discrete"
    enable_pan: bool = True
    enable_target_entropy: bool = True
    enable_stratified_sampling: bool = False
    allow_reverse: bool = False
    ppo_hyperparameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.data_path:
            raise ValueError("data_path must not be empty")
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps must be >= 1")
        if self.incremental_timesteps < 1:
            raise ValueError("incremental_timesteps must be >= 1")
        if self.check_interval_sec < 1:
            raise ValueError("check_interval_sec must be >= 1")
        if self.retrain_interval_sec < self.check_interval_sec:
            raise ValueError(
                "retrain_interval_sec must be >= check_interval_sec"
            )
        if not (0.0 <= self.min_override_confidence <= 1.0):
            raise ValueError("min_override_confidence must be in [0.0, 1.0]")
        if not (0.0 <= self.min_action_probability_gap <= 1.0):
            raise ValueError(
                "min_action_probability_gap must be in [0.0, 1.0]"
            )
        if self.use_continuous_actions:
            raise ValueError("PPO sidecar must run with discrete actions")
        if self.action_space_type != "discrete":
            raise ValueError("action_space_type must be 'discrete' for PPO sidecar")

    @classmethod
    def from_yaml_dict(cls, config: Mapping[str, object]) -> "PPOSidecarConfig":
        """YAML dict から PPO sidecar foundation config を構築する."""
        root = ensure_dict(config)
        data_cfg = ensure_dict(root.get("data"))
        training_cfg = ensure_dict(root.get("training"))
        output_cfg = ensure_dict(root.get("output"))
        ppo_sidecar_cfg = ensure_dict(root.get("ppo_sidecar"))
        ppo_hyperparameters = ensure_dict(root.get("ppo_hyperparameters"))

        model_dir = Path(str(output_cfg.get("model_dir", "models/v461")))
        checkpoint_dir = Path(
            str(ppo_sidecar_cfg.get("checkpoint_dir", model_dir))
        )
        return cls(
            data_path=str(
                ppo_sidecar_cfg.get(
                    "data_path",
                    data_cfg.get("data_path", training_cfg.get("data_path", "")),
                )
            ),
            model_path=model_dir
            / str(ppo_sidecar_cfg.get("model_name", "ppo_sidecar.zip")),
            signal_path=Path(
                str(
                    ppo_sidecar_cfg.get(
                        "signal_path",
                        "cache/ppo_sidecar_signal.json",
                    )
                )
            ),
            checkpoint_dir=checkpoint_dir,
            total_timesteps=safe_to_int(
                training_cfg.get("total_timesteps", 200_000),
                200_000,
            ),
            incremental_timesteps=safe_to_int(
                ppo_sidecar_cfg.get("incremental_timesteps", 50_000),
                50_000,
            ),
            check_interval_sec=safe_to_int(
                ppo_sidecar_cfg.get("check_interval_sec", 300),
                300,
            ),
            retrain_interval_sec=safe_to_int(
                ppo_sidecar_cfg.get("retrain_interval_sec", 7200),
                7200,
            ),
            min_override_confidence=safe_to_float(
                ppo_sidecar_cfg.get("min_override_confidence", 0.55),
                0.55,
            ),
            min_action_probability_gap=safe_to_float(
                ppo_sidecar_cfg.get("min_action_probability_gap", 0.10),
                0.10,
            ),
            use_continuous_actions=safe_to_bool(
                ppo_sidecar_cfg.get("use_continuous_actions", False),
                False,
            ),
            action_space_type=str(
                ppo_sidecar_cfg.get("action_space_type", "discrete")
            ),
            enable_pan=safe_to_bool(
                ppo_sidecar_cfg.get("enable_pan", True),
                True,
            ),
            enable_target_entropy=safe_to_bool(
                ppo_sidecar_cfg.get("enable_target_entropy", True),
                True,
            ),
            enable_stratified_sampling=safe_to_bool(
                ppo_sidecar_cfg.get("enable_stratified_sampling", False),
                False,
            ),
            allow_reverse=safe_to_bool(
                ppo_sidecar_cfg.get("allow_reverse", False),
                False,
            ),
            ppo_hyperparameters=ppo_hyperparameters,
        )

    def build_trainer_config(self) -> dict[str, object]:
        """current PPO trainer が受け取れる最小 config を構築する."""
        trainer_config: dict[str, object] = {
            "algorithm": "ppo",
            "data_path": self.data_path,
            "checkpoint_dir": str(self.checkpoint_dir),
            "total_timesteps": self.total_timesteps,
            "use_continuous_actions": False,
            "action_space_type": "discrete",
            "enable_pan": self.enable_pan,
            "enable_target_entropy": self.enable_target_entropy,
            "enable_stratified_sampling": self.enable_stratified_sampling,
            "allow_reverse": self.allow_reverse,
            "ppo": {
                **self.ppo_hyperparameters,
                "use_custom_ppo": bool(
                    self.ppo_hyperparameters.get("use_custom_ppo", True)
                ),
            },
        }
        return trainer_config
