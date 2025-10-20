"""
SAC v426 Configuration

Configuration settings for SAC v426 improvements including
bias correction, market adaptation, and validation parameters.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SACv426Config:
    """Configuration for SAC v426 improvements."""

    # Model hyperparameters
    learning_rate: float = 0.0003
    buffer_size: int = 20000
    learning_starts: int = 500
    batch_size: int = 128
    tau: float = 0.005
    gamma: float = 0.99
    ent_coef: float = 0.01

    # Environment settings
    initial_balance: float = 200000.0
    transaction_cost: float = 1e-05
    max_position_size: float = 1.0
    use_continuous_actions: bool = True
    use_standardized_observations: bool = True

    # v426 specific improvements
    bias_correction: Optional[Dict[str, Any]] = None
    market_adaptation: Optional[Dict[str, Any]] = None
    validation_settings: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.bias_correction is None:
            self.bias_correction = {
                "sell_bias_threshold": 0.6,  # SELLバイアス67%を60%以下に修正
                "action_balance_tolerance": 0.05,
                "force_diversity": True,
                "balance_penalty_multiplier": 3.0,
            }

        if self.market_adaptation is None:
            self.market_adaptation = {
                "regime_detection_window": 20,
                "trend_adaptation_rate": 0.001,
                "volatility_threshold": 0.02,
                "correlation_target": 0.1,  # 価格相関0.019 → 0.1以上
            }

        if self.validation_settings is None:
            self.validation_settings = {
                "stochastic_episodes": 10,
                "regime_analysis_enabled": True,
                "stress_test_enabled": True,
                "walk_forward_windows": [20, 50, 100],
            }

    @classmethod
    def from_json(cls, config_path: str) -> "SACv426Config":
        """Load configuration from JSON file."""
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract relevant sections
        env_config = data.get("environment", {})
        reward_config = data.get("reward_settings", {})

        return cls(
            learning_rate=data.get("sac_hyperparameters", {}).get(
                "learning_rate", 0.0003
            ),
            buffer_size=data.get("sac_hyperparameters", {}).get("buffer_size", 20000),
            learning_starts=data.get("sac_hyperparameters", {}).get(
                "learning_starts", 500
            ),
            batch_size=data.get("sac_hyperparameters", {}).get("batch_size", 128),
            tau=data.get("sac_hyperparameters", {}).get("tau", 0.005),
            gamma=data.get("sac_hyperparameters", {}).get("gamma", 0.99),
            ent_coef=data.get("sac_hyperparameters", {}).get("ent_coef", 0.01),
            initial_balance=env_config.get("initial_balance", 200000.0),
            transaction_cost=env_config.get("transaction_cost", 1e-05),
            max_position_size=env_config.get("max_position_size", 1.0),
            use_continuous_actions=env_config.get("use_continuous_actions", True),
            use_standardized_observations=env_config.get(
                "use_standardized_observations", True
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "gamma": self.gamma,
            "ent_coef": self.ent_coef,
            "initial_balance": self.initial_balance,
            "transaction_cost": self.transaction_cost,
            "max_position_size": self.max_position_size,
            "use_continuous_actions": self.use_continuous_actions,
            "use_standardized_observations": self.use_standardized_observations,
            "bias_correction": self.bias_correction,
            "market_adaptation": self.market_adaptation,
            "validation_settings": self.validation_settings,
        }

    def save(self, config_path: str):
        """Save configuration to JSON file."""
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
