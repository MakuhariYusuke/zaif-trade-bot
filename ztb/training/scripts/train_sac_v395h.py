"""
SAC v395h - Normalized Observations + Improved Rewards

Environment diagnostics revealed 2 root causes:
1. Observations not normalized (max value: 18 million)
2. Zero rewards: 64.3% (insufficient learning signal)

This version fixes these issues:
- Enable use_standardized_observations
- Add inactivity_penalty for HOLD with no position
- Add opportunity_cost for HOLD with position

Expected improvements:
- Critic Loss < 1000 (was 1e7-1e10)
- Actor Loss 0.1-100 (was 1e6)
- ent_coef 0.5-1.5 (was 3.58)
- Zero rewards < 30% (was 64.3%)
"""

import json

from ztb.training.unified_trainer import UnifiedTrainer


def main():
    print("🔬 SAC v395h - Normalized + Improved Rewards")
    print("=" * 80)

    config_path = "configs/sac_v395h_normalized.json"

    # 設定ファイル読み込み
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print("🔍 Root Causes Found in Diagnostics:")
    print("-" * 80)
    fixes = config["fixes_from_diagnostics"]
    print(f"  Problem 1: {fixes['problem_1']}")
    print(f"  Solution 1: {fixes['solution_1']}")
    print()
    print(f"  Problem 2: {fixes['problem_2']}")
    print(f"  Solution 2: {fixes['solution_2']}")
    print()

    print("📊 Diagnostic Findings:")
    print("-" * 80)
    findings = fixes["diagnostic_findings"]
    for key, value in findings.items():
        print(f"  • {key:25s}: {value}")
    print()

    print("🎯 Expected Improvements:")
    print("-" * 80)
    for metric, target in fixes["expected_improvements"].items():
        print(f"  • {metric:25s}: {target}")
    print()

    print("🚀 Starting 5k timesteps training with fixes...")
    print("=" * 80)

    trainer = UnifiedTrainer(config)
    result = trainer.train()

    print("\n✅ Training completed!")
    return result


if __name__ == "__main__":
    main()
