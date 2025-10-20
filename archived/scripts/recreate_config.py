import json

config = {
    "model_name": "sac_v418_balanced_adjusted",
    "algorithm": "sac",
    "total_timesteps": 1000,
    "data_source": "csv",
    "data_path": "btc_jpy_real_dataset.csv",
    "data_config": {"csv_path": "btc_jpy_real_dataset.csv", "use_real_data": True},
    "sac_hyperparameters": {
        "learning_rate": 0.0003,
        "buffer_size": 20000,
        "learning_starts": 500,
        "batch_size": 128,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": 0.01,
        "target_update_interval": 1,
        "target_entropy": -1.0,
    },
    "environment": {
        "initial_balance": 200000,
        "transaction_cost": 0.00001,
        "max_position_size": 1.0,
        "enable_action_masking": False,
        "use_continuous_actions": True,
        "use_standardized_observations": True,
        "random_start": True,
        "curriculum_stage": "profit_optimized",
        "continuous_to_discrete_threshold": 0.1,
    },
    "reward_settings": {
        "reward_scale": 500.0,
        "reward_clip_min": -200.0,
        "reward_clip_max": 200.0,
        "profit_bonuses": {
            "base_profit_atr_coefficient": 1.5,
            "base_profit_portfolio_coefficient": 1.2,
            "profit_multipliers": [
                2.0,  # BUY action profit multiplier
                0.6,  # SELL action profit multiplier
                0.4,  # HOLD action profit multiplier
            ],
            "trading_bonus": 0.01,
            "trading_bonus_multiplier": 4.0,
        },
        "action_bonuses": {
            "buy_action_bonus": -0.01,
            "sell_action_bonus": 0.02,
            "hold_action_bonus": 0.0,
            "win_rate_bonus": 0.1,
            "momentum_bonus": 0.05,
            "diversity_bonus": 0.02,
        },
        "behavior_penalties": {
            "loss_penalty_multiplier": 3.0,
            "balance_penalty": 3.0,
            "balance_penalty_tolerance": 0.05,
            "action_frequency_penalty": 0.005,
            "inactivity_penalty_rate": 0.005,
            "opportunity_cost_rate": 0.005,
        },
        "risk_penalties": {
            "volatility_penalty": 0.02,
            "position_penalty_soft_cap": 0.8,
            "constraint_penalty": 1.0,
        },
        "flags": {
            "enable_forced_diversity": False,
            "enable_inactivity_penalty": True,
            "enable_opportunity_cost": True,
            "enable_trade_execution_bonus": True,
        },
    },
    "checkpoint_interval": 1000,
    "notes": "SAC v418: Refactored reward parameters with clear bonus/penalty separation",
    "fixes_implemented": {
        "fix_1": "Reorganized reward_settings into logical groups: profit_bonuses, action_bonuses, behavior_penalties, risk_penalties, flags",
        "fix_2": "Unified action_bonuses naming (sell_action_penalty→sell_action_bonus, hold_action_penalty→hold_action_bonus)",
        "fix_3": "Added comments to profit_multipliers array indicating BUY/SELL/HOLD order",
        "fix_4": "Improved parameter naming for better UI/UX clarity",
        "fix_5": "Maintained all advanced reward parameters with better organization",
    },
}

with open("config/sac_v418_balanced_adjusted_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print("Config file updated successfully")
