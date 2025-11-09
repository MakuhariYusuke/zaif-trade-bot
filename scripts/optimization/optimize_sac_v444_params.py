#!/usr/bin/env python3
"""
SAC v444 Parameter Optimization Script
段階的なBalance Penalty最適化とReward Function改善
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_config(config_path: str) -> Dict:
    """Load JSON config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: Dict, config_path: str) -> None:
    """Save JSON config file."""
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def create_balance_penalty_variants(
    base_config: Dict, penalty_scales: List[float]
) -> List[Tuple[Dict, str]]:
    """
    Create config variants with different balance penalty scales.
    
    Args:
        base_config: Base configuration dictionary
        penalty_scales: List of balance penalty scale values to test
        
    Returns:
        List of (config, variant_name) tuples
    """
    variants = []
    
    for penalty_scale in penalty_scales:
        config = json.loads(json.dumps(base_config))  # Deep copy
        
        # Update balance_penalty in curriculum settings
        if "curriculum_learning" in config:
            if "stages" in config["curriculum_learning"]:
                for stage_name, stage_config in config["curriculum_learning"]["stages"].items():
                    if "behavior_optimization" in stage_config:
                        stage_config["behavior_optimization"]["balance_penalty"] = penalty_scale
        
        # Also update in action_learning if present
        if "action_learning" in config:
            if "action_bonuses" in config["action_learning"]:
                if "balance_penalty" in config["action_learning"]:
                    config["action_learning"]["balance_penalty"] = penalty_scale
        
        variant_name = f"sac_v444_2_balance_penalty_{penalty_scale:.0f}"
        variants.append((config, variant_name))
    
    return variants


def create_action_bonus_variants(
    base_config: Dict, buy_bonuses: List[float], sell_bonuses: List[float]
) -> List[Tuple[Dict, str]]:
    """
    Create config variants with different action bonuses.
    
    Args:
        base_config: Base configuration dictionary
        buy_bonuses: List of buy action bonus values
        sell_bonuses: List of sell action bonus values
        
    Returns:
        List of (config, variant_name) tuples
    """
    variants = []
    
    for buy_bonus in buy_bonuses:
        for sell_bonus in sell_bonuses:
            config = json.loads(json.dumps(base_config))  # Deep copy
            
            # Update action bonuses
            if "curriculum_learning" in config:
                if "stages" in config["curriculum_learning"]:
                    for stage_name, stage_config in config["curriculum_learning"]["stages"].items():
                        if "action_bonuses" in stage_config:
                            stage_config["action_bonuses"]["buy_action_bonus"] = buy_bonus
                            stage_config["action_bonuses"]["sell_action_bonus"] = sell_bonus
            
            variant_name = f"sac_v444_2_bonus_buy{buy_bonus:.3f}_sell{sell_bonus:.3f}"
            variants.append((config, variant_name))
    
    return variants


def analyze_current_metrics() -> Dict:
    """Analyze current training metrics and statistics."""
    metrics = {
        "current_state": {
            "hold_ratio": 0.1515,
            "buy_ratio": 0.18,
            "sell_ratio": 0.6685,
            "continuous_mean": -0.4968,
            "continuous_std": 0.6516,
            "reward_mean": -9845.1924,
            "reward_std": 1108.9875,
            "positive_reward_ratio": 0.002,
        },
        "targets": {
            "hold_ratio": 0.25,
            "buy_ratio": 0.35,
            "sell_ratio": 0.40,
            "continuous_mean": 0.1,  # Near zero, slightly positive
            "continuous_std": 0.5,  # Reduced variance
            "reward_mean": -1000.0,  # Significant improvement
            "positive_reward_ratio": 0.15,
        },
        "analysis": {
            "buy_sell_imbalance": abs(0.18 - 0.6685),  # 0.4885
            "penalty_from_imbalance": abs(0.18 - 0.6685) * 1000.0,  # 488.5
            "estimated_balance_penalty_contribution": 48.85,  # % of mean reward
            "recommended_penalty_scale": 200.0,  # Conservative estimate
        }
    }
    return metrics


def print_optimization_plan() -> None:
    """Print the optimization plan."""
    print("\n" + "="*80)
    print("SAC v444 Parameter Optimization Plan")
    print("="*80)
    
    metrics = analyze_current_metrics()
    
    print("\n[Current State Analysis]")
    for key, value in metrics["current_state"].items():
        print(f"  {key}: {value}")
    
    print("\n[Target Metrics]")
    for key, value in metrics["targets"].items():
        print(f"  {key}: {value}")
    
    print("\n[Problem Diagnosis]")
    print(f"  BUY/SELL Imbalance: {metrics['analysis']['buy_sell_imbalance']:.4f} ({metrics['analysis']['buy_sell_imbalance']*100:.2f}%)")
    print(f"  Penalty from imbalance (current 1000.0): {metrics['analysis']['penalty_from_imbalance']:.2f}")
    print(f"  Estimated penalty contribution to reward: ~{metrics['analysis']['estimated_balance_penalty_contribution']:.1f}%")
    
    print("\n[Optimization Strategy]")
    print("  Phase 1: Balance Penalty Scale Optimization")
    print("    - Test range: 50, 100, 150, 200, 300, 500")
    print("    - Goal: Find sweet spot where penalties guide but don't dominate")
    print("    - Expected best: 150-250 range")
    
    print("\n  Phase 2: Action Bonus Tuning")
    print("    - Increase buy_action_bonus to incentivize buying")
    print("    - Reduce or maintain sell_action_bonus")
    print("    - Typical: buy=0.1-0.3, sell=0.0-0.1")
    
    print("\n  Phase 3: Entropy & Learning Rate Adjustment")
    print("    - Increase entropy coefficient for better exploration")
    print("    - Adjust learning rates per regime")
    
    print("\n  Phase 4: Validation & Fine-tuning")
    print("    - Monitor action distribution convergence")
    print("    - Track reward statistics improvement")
    print("    - Backtest on new data")
    
    print("\n" + "="*80)


def create_penalty_scale_test_configs() -> None:
    """Create test configs with different balance penalty scales."""
    base_config_path = r"c:\Users\Admin\dev\zaif-trade-bot\config\sac_v444_2_integrated_regime_adaptation_config.json"
    output_dir = Path(r"c:\Users\Admin\dev\zaif-trade-bot\config\sac_v444_variants")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base config
    base_config = load_config(base_config_path)
    
    # Create variants with different penalty scales
    penalty_scales = [50.0, 100.0, 150.0, 200.0, 300.0, 500.0]
    variants = create_balance_penalty_variants(base_config, penalty_scales)
    
    print(f"\nCreating {len(variants)} variant configs for balance penalty optimization...")
    
    for config, variant_name in variants:
        output_path = output_dir / f"{variant_name}.json"
        save_config(config, str(output_path))
        print(f"  ✓ Created: {output_path}")
    
    print(f"\nVariant configs saved to: {output_dir}")


def create_action_bonus_test_configs() -> None:
    """Create test configs with different action bonuses."""
    base_config_path = r"c:\Users\Admin\dev\zaif-trade-bot\config\sac_v444_2_integrated_regime_adaptation_config.json"
    output_dir = Path(r"c:\Users\Admin\dev\zaif-trade-bot\config\sac_v444_variants_bonus")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base config
    base_config = load_config(base_config_path)
    
    # Create variants with different action bonuses
    buy_bonuses = [0.05, 0.1, 0.2, 0.3]
    sell_bonuses = [0.0, 0.05, 0.1]
    
    variants = create_action_bonus_variants(base_config, buy_bonuses, sell_bonuses)
    
    print(f"\nCreating {len(variants)} variant configs for action bonus optimization...")
    
    for config, variant_name in variants:
        output_path = output_dir / f"{variant_name}.json"
        save_config(config, str(output_path))
        print(f"  ✓ Created: {output_path}")
    
    print(f"\nVariant configs saved to: {output_dir}")


def generate_training_commands() -> None:
    """Generate training commands for all variants."""
    output_file = Path(r"c:\Users\Admin\dev\zaif-trade-bot\scripts\run_sac_variants.ps1")
    
    penalty_scales = [50.0, 100.0, 150.0, 200.0, 300.0, 500.0]
    
    commands = []
    commands.append("# SAC v444 Variant Training Commands")
    commands.append("# Generated for systematic parameter optimization")
    commands.append("")
    commands.append("$configs = @(")
    
    for penalty_scale in penalty_scales:
        config_name = f"sac_v444_2_balance_penalty_{penalty_scale:.0f}"
        commands.append(f'    @{{ config = "{config_name}"; steps = 2000; }}')
    
    commands.append(")")
    commands.append("")
    commands.append("foreach ($item in $configs) {")
    commands.append('    $config = $item.config')
    commands.append('    $steps = $item.steps')
    commands.append('    Write-Host "Training with config: $config for $steps steps" -ForegroundColor Green')
    commands.append('    python quick_train_v444.py --config "config/sac_v444_variants/$config.json" --steps $steps')
    commands.append("    Start-Sleep -Seconds 5")
    commands.append("}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(commands))
    
    print(f"\nGenerated training commands: {output_file}")


if __name__ == "__main__":
    # Print optimization plan
    print_optimization_plan()
    
    # Create variant configs
    create_penalty_scale_test_configs()
    create_action_bonus_test_configs()
    
    # Generate training commands
    generate_training_commands()
    
    print("\n" + "="*80)
    print("✓ Optimization setup complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the variant configs in:")
    print("   - config/sac_v444_variants/")
    print("   - config/sac_v444_variants_bonus/")
    print("2. Run training with variants (start with balance penalty optimization)")
    print("3. Compare results and identify best parameters")
    print("4. Fine-tune with combined parameters")
    print("="*80 + "\n")
