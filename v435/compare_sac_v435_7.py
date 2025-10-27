#!/usr/bin/env python3
"""
SAC v435.7 Performance Comparison
各バリアントのパフォーマンス比較
"""
import json
from pathlib import Path


def main():
    print("🚀 SAC v435.7 Performance Comparison")
    print("=" * 50)

    config_dir = Path("backtest_experiments/v435.7")
    models = [
        ("sac_v435.7a", config_dir / "sac_v435_7a_config.json"),
        ("sac_v435.7b", config_dir / "sac_v435_7b_config.json"),
        ("sac_v435.7c", config_dir / "sac_v435_7c_config.json"),
    ]

    print("\n📋 Model Configurations:")
    for model_name, config_path in models:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            freq_penalty = config["training"]["reward_function"][
                "action_frequency_penalty"
            ]
            profit_atr = config["training"]["reward_function"][
                "base_profit_bonus_atr_coeff"
            ]
            profit_portfolio = config["training"]["reward_function"][
                "base_profit_bonus_portfolio_coeff"
            ]
            symmetric = config["training"]["environment"]["symmetric_thresholds"]

            print(f"\n{model_name}:")
            print(f"  Frequency Penalty: {freq_penalty}")
            print(f"  Profit Bonus ATR: {profit_atr}")
            print(f"  Profit Bonus Portfolio: {profit_portfolio}")
            print(f"  Symmetric Thresholds: {symmetric}")

            # モデルファイルの存在確認
            model_path = Path(f"models/{model_name}.zip")
            if model_path.exists():
                print(f"  ✅ Model file exists: {model_path}")
            else:
                print(f"  ❌ Model file missing: {model_path}")
        else:
            print(f"❌ Config not found: {config_path}")

    print("\n🎯 Analysis Summary:")
    print(
        "• v435.7a: Ultra-micro frequency penalty (0.0001) - balances activity with profit"
    )
    print("• v435.7b: Zero frequency penalty (0.0) - maximizes trading activity")
    print(
        "• v435.7c: Enhanced victory bonuses (3x profit) - prioritizes profit over frequency"
    )
    print("• All variants use symmetric thresholds (±0.3333) to prevent value sticking")

    print("\n💡 Recommendation:")
    print("Run backtests with each model to determine which approach works best")
    print("for your scalping strategy. Consider market conditions:")
    print("- Use v435.7a for stable markets (balanced approach)")
    print("- Use v435.7b for volatile markets (high frequency)")
    print("- Use v435.7c for trending markets (profit-focused)")


if __name__ == "__main__":
    main()
