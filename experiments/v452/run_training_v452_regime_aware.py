import sys
from pathlib import Path

# Import torch first to avoid DLL issues

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from ztb.training.unified_trainer import UnifiedTrainer


def run_training_v452():
    """
    Phase 8 (v452) Training: Regime Aware & Threshold Optimized

    Builds upon v451 but with corrected threshold configuration to allow
    the dynamic threshold manager to function correctly without clamping.

    Key Changes:
    - max_action_threshold: 1.0 (was 0.05)
    - adaptive_threshold_mode: True
    - Regime-based threshold multipliers (handled by ThresholdManager)
    """

    config = {
        "model_name": "sac_v452_regime_aware",
        "training": {
            "algorithm": "sac",
            "total_timesteps": 50000,  # Increased steps for better convergence
            "log_interval": 100,
            "checkpoint_interval": 5000,
            "checkpoint_dir": "checkpoints/v452/phase8",
            "sac_hyperparameters": {
                "learning_rate": 5e-5,
                "batch_size": 2048,
                "buffer_size": 200000,
                "learning_starts": 1000,  # Warmup
                "tau": 0.005,
                "gamma": 0.80,  # HFT focus
                "train_freq": 1,
                "gradient_steps": 1,
                "ent_coef": 0.05,  # High exploration
            },
            "environment": {
                "config": {
                    "initial_portfolio_value": 1000000.0,
                    "max_position_size": 1.0,
                    "transaction_cost": 0.0005,  # 0.05% Fee
                    "reward_scaling": 1.0,
                    "timeframe": "1m",
                    "feature_set": "v451",  # Use registered v451 set
                    "use_continuous_actions": True,
                    "action_space_type": "continuous",
                    "curriculum_stage": "trading_focused",
                    # Threshold Configuration (CRITICAL FIXES)
                    "max_action_threshold": 1.0,  # Allow full range for dynamic thresholds
                    "min_action_threshold": 0.001,
                    "continuous_to_discrete_threshold": 0.01,
                    "adaptive_threshold_mode": True,
                    "threshold_volatility_multiplier": 1.0,
                    # Disable internal feature generation since we provide them
                    "enable_feature_filtering": False,
                    "include_multi_timeframe_features": False,
                    "reward_settings": {
                        "behavior": {
                            "trading_focused": {
                                "hold_penalty_rate": 0.0,
                                "trading_bonus_multiplier": 1.0,
                                "balance_penalty": 0.0,
                                "fee_penalty": 0.0,
                            }
                        },
                        # Asymmetric Reward (Loss Penalty 1.2x)
                        "profit_bonus_multipliers": {"profit": 1.0, "loss": 1.2},
                    },
                    "execution_model": {
                        "base_slippage": 0.0001,
                        "atr_slippage_factor": 0.1,
                        "base_latency_ms": 10.0,
                        "latency_jitter_ms": 5.0,
                    },
                    "bankruptcy_threshold": 2000.0,
                    "bankruptcy_penalty": 1000.0,
                    "drawdown_penalty_threshold": 0.05,
                    "drawdown_penalty_factor": 0.5,
                },
                "feature_set": "v451",
            },
            "data_config": {
                "data_path": str(project_root / "data" / "btc_jpy_1m_v451.csv"),
            },
            "evaluation": {
                "enabled": True,
                "eval_freq": 5000,
                "n_eval_episodes": 3,
                "deterministic": True,
                "config": {
                    "transaction_cost": 0.0000,
                    "initial_portfolio_value": 1000000.0,
                    # Ensure eval env also has the fix
                    "max_action_threshold": 1.0,
                },
            },
        },
    }

    print(f"Starting training for {config['model_name']}...")
    print(
        f"Configured max_action_threshold: {config['training']['environment']['config']['max_action_threshold']}"
    )

    trainer = UnifiedTrainer(config)
    trainer.train()


if __name__ == "__main__":
    run_training_v452()
