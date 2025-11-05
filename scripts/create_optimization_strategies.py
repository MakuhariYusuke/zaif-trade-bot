#!/usr/bin/env python3
"""
Detailed Configuration Strategy for SAC v444.2 Action Balance Improvement

複数の実装戦略を組み合わせたハイブリッドアプローチ:
1. Balance Penalty Scale Reduction: 1000 → 200
2. Action Bonus Enhancement: Buy bias counter
3. Entropy Increase: Better exploration
4. Reward Scaling: Positive reward amplification
"""

import json
from pathlib import Path
from typing import Dict


def create_strategy_configs() -> Dict[str, Dict]:
    """Create multiple strategy configurations with different approaches."""
    
    strategies = {
        "strategy_1_aggressive_balance_reduction": {
            "description": "Aggressive reduction of balance penalty to 100, minimal action bonuses",
            "balance_penalty": 100.0,
            "entropy_regularization": 0.025,
            "buy_action_bonus": 5.0,
            "sell_action_bonus": 0.0,
            "hold_action_bonus": 0.0,
            "redundant_trade_penalty": 2.0,
            "base_action_penalty": 0.3,
        },
        "strategy_2_balanced_moderate": {
            "description": "Balanced approach with moderate penalty and bonuses",
            "balance_penalty": 200.0,
            "entropy_regularization": 0.02,
            "buy_action_bonus": 10.0,
            "sell_action_bonus": 2.0,
            "hold_action_bonus": 1.0,
            "redundant_trade_penalty": 3.0,
            "base_action_penalty": 0.5,
        },
        "strategy_3_reward_emphasis": {
            "description": "Emphasis on reward bonuses with moderate penalty",
            "balance_penalty": 150.0,
            "entropy_regularization": 0.015,
            "buy_action_bonus": 15.0,
            "sell_action_bonus": 3.0,
            "hold_action_bonus": 2.0,
            "redundant_trade_penalty": 4.0,
            "base_action_penalty": 0.4,
        },
        "strategy_4_high_entropy_exploration": {
            "description": "High entropy for better exploration with lower penalties",
            "balance_penalty": 120.0,
            "entropy_regularization": 0.04,
            "buy_action_bonus": 8.0,
            "sell_action_bonus": 1.0,
            "hold_action_bonus": 0.5,
            "redundant_trade_penalty": 2.5,
            "base_action_penalty": 0.2,
        },
        "strategy_5_conservative_tuning": {
            "description": "Conservative adjustments from current baseline",
            "balance_penalty": 500.0,
            "entropy_regularization": 0.015,
            "buy_action_bonus": 7.5,
            "sell_action_bonus": 0.5,
            "hold_action_bonus": 0.25,
            "redundant_trade_penalty": 5.0,
            "base_action_penalty": 0.7,
        },
    }
    
    return strategies


def apply_strategy_to_config(base_config_path: str, strategy: Dict, output_path: str) -> None:
    """Apply a strategy to base config and save."""
    
    # Load base config
    with open(base_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Apply strategy parameters
    config["environment"]["behavior_optimization"]["balance_penalty"] = strategy["balance_penalty"]
    config["environment"]["behavior_optimization"]["entropy_regularization"] = strategy["entropy_regularization"]
    config["environment"]["behavior_optimization"]["redundant_trade_penalty"] = strategy["redundant_trade_penalty"]
    config["environment"]["base_action_penalty"] = strategy["base_action_penalty"]
    
    config["environment"]["action_bonuses"]["buy_action_bonus"] = strategy["buy_action_bonus"]
    config["environment"]["action_bonuses"]["sell_action_bonus"] = strategy["sell_action_bonus"]
    config["environment"]["action_bonuses"]["hold_action_bonus"] = strategy["hold_action_bonus"]
    
    # Save config
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def create_all_strategy_configs() -> None:
    """Create all strategy configuration files."""
    
    base_config_path = r"c:\Users\Admin\dev\zaif-trade-bot\config\sac_v444_2_integrated_regime_adaptation_config.json"
    output_dir = Path(r"c:\Users\Admin\dev\zaif-trade-bot\config\sac_v444_strategies")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    strategies = create_strategy_configs()
    
    print("\n" + "="*100)
    print("Creating SAC v444.2 Strategy Configurations")
    print("="*100)
    
    for strategy_name, strategy_params in strategies.items():
        output_path = output_dir / f"{strategy_name}.json"
        
        try:
            apply_strategy_to_config(base_config_path, strategy_params, str(output_path))
            print(f"\n✓ {strategy_name}")
            print(f"  Description: {strategy_params['description']}")
            print(f"  Balance Penalty: {strategy_params['balance_penalty']}")
            print(f"  Buy Bonus: {strategy_params['buy_action_bonus']}")
            print(f"  Entropy: {strategy_params['entropy_regularization']}")
            print(f"  → {output_path}")
        except Exception as e:
            print(f"\n✗ Error creating {strategy_name}: {e}")
    
    print("\n" + "="*100)
    print(f"Strategy configs saved to: {output_dir}")
    print("="*100 + "\n")


def generate_strategy_comparison_script() -> None:
    """Generate a script to compare all strategies."""
    
    script_path = Path(r"c:\Users\Admin\dev\zaif-trade-bot\scripts\run_strategy_comparison.ps1")
    
    script_content = """
# SAC v444.2 Strategy Comparison Script
# 5つの異なる改善戦略を実行し、結果を比較

$strategies = @(
    @{ name = "strategy_1_aggressive_balance_reduction"; steps = 5000 },
    @{ name = "strategy_2_balanced_moderate"; steps = 5000 },
    @{ name = "strategy_3_reward_emphasis"; steps = 5000 },
    @{ name = "strategy_4_high_entropy_exploration"; steps = 5000 },
    @{ name = "strategy_5_conservative_tuning"; steps = 5000 }
)

Write-Host "Starting SAC v444.2 Strategy Comparison..." -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green

foreach ($strategy in $strategies) {
    $configPath = "config/sac_v444_strategies/$($strategy.name).json"
    $steps = $strategy.steps
    
    Write-Host "`nTraining with: $($strategy.name)" -ForegroundColor Cyan
    Write-Host "Config: $configPath" -ForegroundColor Cyan
    Write-Host "Steps: $steps" -ForegroundColor Cyan
    
    python quick_train_v444_optimized.py --config $configPath --steps $steps --analyze
    
    Write-Host "✓ Completed: $($strategy.name)" -ForegroundColor Green
    Start-Sleep -Seconds 3
}

Write-Host "`nAll strategies trained! Analyzing results..." -ForegroundColor Green
python scripts/analyze_optimization_results.py results

Write-Host "`n✓ Strategy comparison complete!" -ForegroundColor Green
"""
    
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print(f"✓ Comparison script created: {script_path}")


def print_strategy_summary() -> None:
    """Print summary of all strategies."""
    
    strategies = create_strategy_configs()
    
    print("\n" + "="*100)
    print("SAC v444.2 Optimization Strategies Summary")
    print("="*100)
    
    print("""
5つの改善戦略を実装しました:

┌─ STRATEGY 1: Aggressive Balance Reduction ─────────────────────────┐
│ Balance Penalty: 100.0 (↓ from 1000)                                │
│ Buy Bonus: 5.0, Entropy: 0.025                                      │
│ 目標: 激進的なペナルティ削減で自由度を最大化                            │
│ 予想: BUY アクションが大幅に増加する可能性                            │
└────────────────────────────────────────────────────────────────────┘

┌─ STRATEGY 2: Balanced Moderate (推奨) ────────────────────────────┐
│ Balance Penalty: 200.0 (↓ from 1000)                                │
│ Buy Bonus: 10.0, Sell: 2.0, Hold: 1.0, Entropy: 0.02              │
│ 目標: バランスとインセンティブの調和                                  │
│ 予想: BUY/SELL/HOLDが比較的均衡した分布                            │
└────────────────────────────────────────────────────────────────────┘

┌─ STRATEGY 3: Reward Emphasis ──────────────────────────────────────┐
│ Balance Penalty: 150.0 (↓ from 1000)                                │
│ Buy Bonus: 15.0, Sell: 3.0, Hold: 2.0, Entropy: 0.015             │
│ 目標: アクションボーナスの強調                                        │
│ 予想: Buy重視でアクション多様性向上                                   │
└────────────────────────────────────────────────────────────────────┘

┌─ STRATEGY 4: High Entropy Exploration ──────────────────────────────┐
│ Balance Penalty: 120.0 (↓ from 1000)                                │
│ Buy Bonus: 8.0, Entropy: 0.04 (↑ 高い)                             │
│ 目標: エントロピー増加で探索性を向上                                  │
│ 予想: より多くのアクション多様性、収束は遅い可能性                    │
└────────────────────────────────────────────────────────────────────┘

┌─ STRATEGY 5: Conservative Tuning ──────────────────────────────────┐
│ Balance Penalty: 500.0 (↓ from 1000、緩和的)                        │
│ Buy Bonus: 7.5, Entropy: 0.015                                      │
│ 目標: 現在の設定から最小限の変更                                      │
│ 予想: 安全だが改善幅は限定的                                        │
└────────────────────────────────────────────────────────────────────┘

推奨実行順序:
1. STRATEGY 2 (Balanced Moderate) - 最初に実行し検証
2. STRATEGY 4 (High Entropy) - エクスプロアレーション向上を検証
3. STRATEGY 3 (Reward Emphasis) - ボーナス効果を検証
4. STRATEGY 1 (Aggressive) - 激進的改善の効果を検証
5. STRATEGY 5 (Conservative) - 安全な改善路線の検証

期待される成果 (各戦略):
- BUY/SELL Difference: < 0.2 (現在 0.49)
- Mean Reward: > -5000 (現在 -9845)
- Action Diversity: HOLD > 15%, BUY > 25%, SELL < 50%
""")
    
    print("="*100 + "\n")


if __name__ == "__main__":
    # Print summary
    print_strategy_summary()
    
    # Create all strategy configs
    create_all_strategy_configs()
    
    # Generate comparison script
    generate_strategy_comparison_script()
    
    print("\n✓ Strategy configuration setup complete!")
    print("\nNext steps:")
    print("1. Run: python quick_train_v444_optimized.py --config config/sac_v444_strategies/strategy_2_balanced_moderate.json --steps 5000 --analyze")
    print("2. Compare results across strategies")
    print("3. Or use PowerShell script to run all: scripts/run_strategy_comparison.ps1")
