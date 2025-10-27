#!/usr/bin/env python3
"""
SAC v435 Trading Behavior Deep Analysis
SAC v435 取引行動の詳細分析

Analyzes why SAC v435 models execute only 1 trade by examining:
1. Reward function behavior
2. Action selection patterns
3. Environment constraints
4. Model decision-making process
"""

import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from stable_baselines3 import SAC

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ztb.trading.environment.environment import HeavyTradingEnv


def analyze_reward_function_detailed(
    model_path: str, data_path: str, max_steps: int = 50
) -> Dict[str, Any]:
    """
    Detailed analysis of reward function behavior and trading decisions

    Args:
        model_path: Path to the trained model
        data_path: Path to the data file
        max_steps: Maximum steps to analyze

    Returns:
        Detailed analysis results
    """
    print(f"Loading model from {model_path}")
    model = SAC.load(model_path)

    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    if len(df) > max_steps:
        df = df.head(max_steps)

    print(f"Analyzing {len(df)} steps of data")

    # Create environment with the same config as training
    env_config = {
        "transaction_cost": 0.0005,
        "enable_correlation_reduction": True,
        "correlation_threshold": 0.95,
        "max_position_size": 0.5,
        "curriculum_stage": "forced_balance",
        "reward_trade_frequency_penalty": 0.01,
        "reward_trade_frequency_halflife": 1.0,
        "reward_trade_cooldown_steps": 0,
        "reward_trade_cooldown_penalty": 0.01,
        "reward_max_consecutive_trades": 20,
        "reward_consecutive_trade_penalty": 0.01,
        "reward_position_penalty_scale": 0.1,
        "reward_position_penalty_exponent": 2.0,
        "reward_inventory_penalty_scale": 0.01,
        "reward_volatility_penalty_scale": 0.01,
    }

    env = HeavyTradingEnv(df=df, config=env_config, random_start=False)

    # Detailed tracking
    step_data = []
    obs, _ = env.reset()

    print("Running detailed reward analysis...")

    for step in range(max_steps):
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        action_value = float(action)

        # Convert to discrete action
        if action > 0.1:  # SAC_CONTINUOUS_THRESHOLD
            discrete_action = 1  # BUY
        elif action < -0.1:  # SAC_CONTINUOUS_THRESHOLD_NEG
            discrete_action = 2  # SELL
        else:
            discrete_action = 0  # HOLD

        # Step environment
        prev_position = env.position
        obs, reward, terminated, truncated, info = env.step(discrete_action)
        done = terminated or truncated

        current_position = env.position
        position_changed = abs(current_position - prev_position) > 0.001

        # Record detailed step data
        step_info = {
            "step": step,
            "action_value": action_value,
            "discrete_action": discrete_action,
            "prev_position": prev_position,
            "current_position": current_position,
            "position_changed": position_changed,
            "reward": float(reward),
            "price": env.df.iloc[min(env.current_step, len(env.df) - 1)]["close"],
            "done": done,
        }

        # Add reward components if available in info
        if info:
            step_info.update({f"info_{k}": v for k, v in info.items()})

        step_data.append(step_info)

        if done:
            break

    # Analyze the collected data
    df_steps = pd.DataFrame(step_data)

    analysis = {
        "total_steps": len(df_steps),
        "action_distribution": df_steps["discrete_action"].value_counts().to_dict(),
        "position_changes": int(df_steps["position_changed"].sum()),
        "total_reward": float(df_steps["reward"].sum()),
        "avg_reward_per_step": float(df_steps["reward"].mean()),
        "reward_std": float(df_steps["reward"].std()),
        "action_value_stats": {
            "mean": float(df_steps["action_value"].mean()),
            "std": float(df_steps["action_value"].std()),
            "min": float(df_steps["action_value"].min()),
            "max": float(df_steps["action_value"].max()),
        },
        "position_stats": {
            "mean": float(df_steps["current_position"].mean()),
            "std": float(df_steps["current_position"].std()),
            "max_abs": float(df_steps["current_position"].abs().max()),
        },
        "trading_decisions": [],
        "reward_analysis": {
            "positive_rewards": int((df_steps["reward"] > 0).sum()),
            "negative_rewards": int((df_steps["reward"] < 0).sum()),
            "zero_rewards": int((df_steps["reward"] == 0).sum()),
            "reward_range": [
                float(df_steps["reward"].min()),
                float(df_steps["reward"].max()),
            ],
        },
    }

    # Analyze trading decisions
    for i, row in df_steps.iterrows():
        if row["position_changed"]:
            decision = {
                "step": int(row["step"]),
                "action": int(row["discrete_action"]),
                "prev_position": float(row["prev_position"]),
                "new_position": float(row["current_position"]),
                "price": float(row["price"]),
                "reward": float(row["reward"]),
            }
            analysis["trading_decisions"].append(decision)

    return analysis


def analyze_environment_constraints() -> Dict[str, Any]:
    """
    Analyze environment constraints that might limit trading

    Returns:
        Analysis of environment constraints
    """
    constraints = {
        "transaction_cost": 0.0005,  # 0.05%
        "max_position_size": 0.5,  # 50% of portfolio
        "reward_trade_frequency_penalty": 0.01,
        "reward_trade_cooldown_steps": 0,
        "reward_max_consecutive_trades": 20,
        "reward_position_penalty_scale": 0.1,
        "reward_position_penalty_exponent": 2.0,
        "analysis": {
            "high_transaction_cost": "0.05% transaction cost may discourage frequent trading",
            "position_penalty": "Position penalty (scale=0.1, exponent=2.0) heavily penalizes large positions",
            "frequency_penalty": "Trade frequency penalty of 0.01 may reduce trading activity",
            "cooldown_effect": "No cooldown steps, so penalty is immediate",
        },
    }

    return constraints


def analyze_reward_function_logic() -> Dict[str, Any]:
    """
    Analyze the reward function logic to understand trading incentives

    Returns:
        Analysis of reward function components
    """
    reward_components = {
        "profit_reward": "Price movement * position * (1 - transaction_cost)",
        "position_penalty": "-position_penalty_scale * |position| ^ position_penalty_exponent",
        "trade_frequency_penalty": "-reward_trade_frequency_penalty * trade_count / halflife",
        "inventory_penalty": "-reward_inventory_penalty_scale * |position|",
        "volatility_penalty": "-reward_volatility_penalty_scale * price_volatility",
        "analysis": {
            "profit_incentive": "Profit reward is the main positive incentive",
            "position_penalty_impact": "Position penalty heavily penalizes any position holding (^2 exponent)",
            "frequency_penalty_impact": "Trade frequency penalty discourages multiple trades",
            "net_effect": "Position penalty likely dominates, making holding positions unattractive",
        },
    }

    return reward_components


def comprehensive_trading_analysis() -> Dict[str, Any]:
    """Comprehensive analysis of why trading frequency is low"""
    model_path = "checkpoints/sac_v435_test_1000_steps.zip"
    data_path = "ml-dataset-enhanced.csv"

    print("=== SAC v435 Comprehensive Trading Analysis ===")

    results = {}

    # 1. Environment constraints analysis
    print("\n1. Analyzing environment constraints...")
    results["environment_constraints"] = analyze_environment_constraints()

    # 2. Reward function analysis
    print("2. Analyzing reward function logic...")
    results["reward_function"] = analyze_reward_function_logic()

    # 3. Model behavior analysis
    print("3. Analyzing model behavior...")
    try:
        behavior_analysis = analyze_reward_function_detailed(
            model_path, data_path, max_steps=100
        )
        results["model_behavior"] = behavior_analysis
        print("✓ Model behavior analysis completed")
    except Exception as e:
        print(f"✗ Model behavior analysis failed: {e}")
        results["model_behavior"] = {"error": str(e)}

    # 4. Root cause analysis
    print("4. Performing root cause analysis...")
    root_causes = analyze_root_causes(results)
    results["root_cause_analysis"] = root_causes

    # 5. Recommendations
    print("5. Generating recommendations...")
    recommendations = generate_recommendations(results)
    results["recommendations"] = recommendations

    return results


def analyze_root_causes(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze root causes of low trading frequency"""

    root_causes = {
        "primary_causes": [],
        "contributing_factors": [],
        "severity_assessment": {},
    }

    # Check environment constraints
    env = analysis_results.get("environment_constraints", {})
    if env.get("reward_position_penalty_scale", 0) > 0:
        root_causes["primary_causes"].append(
            {
                "cause": "High position penalty",
                "description": f"Position penalty scale={env['reward_position_penalty_scale']} with exponent={env['reward_position_penalty_exponent']} heavily penalizes position holding",
                "impact": "Makes any position unattractive, encourages flat position",
            }
        )

    # Check reward function
    reward = analysis_results.get("reward_function", {})
    if "position_penalty_impact" in reward.get("analysis", {}):
        root_causes["contributing_factors"].append(
            {
                "factor": "Dominant position penalty",
                "description": "Position penalty likely outweighs profit incentives",
                "impact": "Reduces willingness to hold positions",
            }
        )

    # Check model behavior
    behavior = analysis_results.get("model_behavior", {})
    if not isinstance(behavior, dict) or "error" in behavior:
        root_causes["contributing_factors"].append(
            {
                "factor": "Model analysis unavailable",
                "description": "Could not analyze model behavior due to technical issues",
                "impact": "Unable to confirm model-specific issues",
            }
        )
    else:
        action_dist = behavior.get("action_distribution", {})
        hold_percentage = (
            action_dist.get(0, 0) / sum(action_dist.values()) * 100
            if action_dist
            else 0
        )

        if hold_percentage > 80:
            root_causes["primary_causes"].append(
                {
                    "cause": "Model learned to hold",
                    "description": f"Model chooses HOLD action {hold_percentage:.1f}% of the time",
                    "impact": "Very conservative trading behavior",
                }
            )

        position_changes = behavior.get("position_changes", 0)
        if position_changes <= 1:
            root_causes["primary_causes"].append(
                {
                    "cause": "Minimal position changes",
                    "description": f"Only {position_changes} position changes in {behavior.get('total_steps', 0)} steps",
                    "impact": "Extremely low trading activity",
                }
            )

    # Severity assessment
    root_causes["severity_assessment"] = {
        "overall_severity": "High",
        "confidence_level": "Medium",
        "key_indicators": [
            "Only 1 trade executed",
            "High position penalty parameters",
            "Conservative action distribution",
        ],
    }

    return root_causes


def generate_recommendations(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate recommendations to improve trading frequency"""

    recommendations = {
        "immediate_actions": [],
        "parameter_adjustments": [],
        "reward_function_changes": [],
        "training_improvements": [],
    }

    # Immediate actions
    recommendations["immediate_actions"] = [
        "Reduce position penalty scale from 0.1 to 0.01 or lower",
        "Reduce position penalty exponent from 2.0 to 1.0 for linear penalty",
        "Increase reward_trade_frequency_penalty halflife to reduce frequency penalty impact",
        "Test with zero position penalty to establish baseline",
    ]

    # Parameter adjustments
    recommendations["parameter_adjustments"] = [
        {
            "parameter": "reward_position_penalty_scale",
            "current": 0.1,
            "recommended": 0.01,
            "reason": "Current value heavily penalizes any position holding",
        },
        {
            "parameter": "reward_position_penalty_exponent",
            "current": 2.0,
            "recommended": 1.0,
            "reason": "Quadratic penalty creates strong disincentive for positions",
        },
        {
            "parameter": "reward_trade_frequency_penalty",
            "current": 0.01,
            "recommended": 0.001,
            "reason": "May be discouraging necessary trading activity",
        },
    ]

    # Reward function changes
    recommendations["reward_function_changes"] = [
        "Implement position size rewards for optimal position management",
        "Add time-based incentives for maintaining profitable positions",
        "Consider asymmetric penalties (less penalty for profitable positions)",
        "Add exploration bonuses for trying different position sizes",
    ]

    # Training improvements
    recommendations["training_improvements"] = [
        "Increase training steps from 1000 to at least 10000",
        "Implement curriculum learning with progressive penalty reduction",
        "Add position diversity requirements during training",
        "Use reward shaping to encourage balanced trading behavior",
    ]

    return recommendations


def main():
    """Main analysis function"""
    try:
        results = comprehensive_trading_analysis()

        # Save results
        import json

        output_file = "sac_v435_trading_analysis.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nAnalysis results saved to: {output_file}")

        # Print summary
        print("\n=== Analysis Summary ===")

        # Root causes
        root_causes = results.get("root_cause_analysis", {})
        print("\n🔍 Primary Root Causes:")
        for cause in root_causes.get("primary_causes", []):
            print(f"• {cause['cause']}: {cause['description']}")

        print("\n📊 Contributing Factors:")
        for factor in root_causes.get("contributing_factors", []):
            print(f"• {factor['factor']}: {factor['description']}")

        # Recommendations
        recs = results.get("recommendations", {})
        print("\n💡 Key Recommendations:")
        for action in recs.get("immediate_actions", [])[:3]:  # Top 3
            print(f"• {action}")

        print(
            f"\nSeverity: {root_causes.get('severity_assessment', {}).get('overall_severity', 'Unknown')}"
        )

    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
