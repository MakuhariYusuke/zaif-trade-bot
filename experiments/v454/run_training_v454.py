import sys
from pathlib import Path

# Import torch first to avoid DLL issues

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from ztb.training.unified_trainer import UnifiedTrainer


def run_training_v454():
    """
    v454 Training: Signal Quality & Confidence Penalty

    Builds upon v452 but with:
    1. Confidence Penalty in Reward Function (Inverse Confidence Paradox fix)
    2. v454 Feature Set (Noise Filtering, Smoothed Volatility)
    3. Adjusted Entropy Coefficient (auto_0.1)
    """

    config = {
        "model_name": "sac_v454_signal_quality",
        "training": {
            "algorithm": "sac",
            "total_timesteps": 100000,  # Increased steps
            "log_interval": 100,
            "checkpoint_interval": 5000,
            "checkpoint_dir": "checkpoints/v454/phase1",
            "sac_hyperparameters": {
                "learning_rate": 5e-5,
                "batch_size": 2048,
                "buffer_size": 200000,
                "learning_starts": 1000,
                "tau": 0.005,
                "gamma": 0.80,
                "train_freq": 1,
                "gradient_steps": 1,
                "ent_coef": "auto_0.1",  # Automatic entropy tuning starting at 0.1
                "net_arch": [512, 512],  # Increased network capacity for v454 features
            },
            "environment": {
                "config": {
                    "initial_portfolio_value": 1000000.0,
                    "max_position_size": 1.0,
                    "transaction_cost": 0.0005,
                    "reward_scaling": 1.0,
                    "timeframe": "1m",
                    "feature_set": "v454",  # Use v454 set
                    "use_continuous_actions": True,
                    "action_space_type": "continuous",
                    "curriculum_stage": "trading_focused",
                    
                    # Threshold Configuration
                    "max_action_threshold": 1.0,
                    "min_action_threshold": 0.001,
                    "continuous_to_discrete_threshold": 0.01,
                    "adaptive_threshold_mode": True,
                    "threshold_volatility_multiplier": 1.0, # Ensure 1.0x
                    
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
                        "profit_bonus_multipliers": {"profit": 1.0, "loss": 1.2},
                        
                        # v454 Confidence Penalty Settings
                        "confidence_penalty_threshold": 0.1,
                        "confidence_penalty_factor": 2.0, # 2x penalty for high confidence loss
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
                "feature_set": "v454",
            },
            "data_config": {
                "data_path": str(project_root / "data" / "btc_jpy_1m_v454.csv"),
            },
            "evaluation": {
                "enabled": True,
                "eval_freq": 5000,
                "n_eval_episodes": 3,
                "deterministic": True,
                "config": {
                    "transaction_cost": 0.0000,
                    "initial_portfolio_value": 1000000.0,
                }
            }
        }
    }

    trainer = UnifiedTrainer(config)
    trainer.train()


if __name__ == "__main__":
    run_training_v454()
