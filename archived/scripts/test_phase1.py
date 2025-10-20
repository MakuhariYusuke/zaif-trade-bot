#!/usr/bin/env python3
"""
SAC v428 Phase 1 Configuration Test
フェーズ1の設定と機能をテストするスクリプト
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from ztb.evaluation.position_duration_validator import PositionDurationValidator


def main():
    # SAC v428設定を読み込み
    with open("configs/sac_v428_position_optimized.json", "r") as f:
        config = json.load(f)

    # 検証クラスを初期化
    validator = PositionDurationValidator(config)

    print("=== SAC v428 Phase 1 Configuration Check ===")
    print(
        f'Min Position Hold Time: {config["environment"].get("min_position_hold_time", "Not set")}'
    )
    print(
        f'Action Confidence Threshold: {config["environment"].get("action_confidence_threshold", "Not set")}'
    )
    print(
        f'Transaction Penalty: {config["reward_settings"]["action_bonuses"].get("transaction_penalty", "Not set")}'
    )
    print(
        f'Position Age Weighting Enabled: {config["environment"].get("position_age_weighting", {}).get("enabled", False)}'
    )

    # テストデータで検証
    test_actions = [0, 0, 1, 1, 1, 2, 2, 0, 0, 0]  # サンプルアクションシーケンス
    validation = validator.validate_position_durations(test_actions)

    print("\nTest Validation Results:")
    print(f'SELL→BUY Duration OK: {validation["validation"]["sell_buy_duration_ok"]}')
    print(f'BUY→SELL Duration OK: {validation["validation"]["buy_sell_duration_ok"]}')
    print(f'HOLD Ratio OK: {validation["validation"]["hold_ratio_ok"]}')
    print(f'Overall Score: {validation["validation"]["overall_score"]:.3f}')

    # 推奨事項を表示
    recommendations = validation.get("validation", {}).get("recommendations", [])
    if recommendations:
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"- {rec}")
    else:
        print("\nNo specific recommendations (validation passed)")

    print("\n=== Phase 1 Implementation Status ===")
    phase1_features = {
        "min_position_hold_time": config["environment"].get("min_position_hold_time")
        == 3,
        "action_confidence_threshold": config["environment"].get(
            "action_confidence_threshold"
        )
        == 0.7,
        "transaction_penalty": config["reward_settings"]["action_bonuses"].get(
            "transaction_penalty"
        )
        == -0.5,
        "position_age_weighting": config["environment"]
        .get("position_age_weighting", {})
        .get("enabled")
        == True,
    }

    for feature, implemented in phase1_features.items():
        status = "✅" if implemented else "❌"
        print(f"{status} {feature}: {implemented}")

    all_implemented = all(phase1_features.values())
    print(f'\nPhase 1 Complete: {"✅" if all_implemented else "❌"}')


if __name__ == "__main__":
    main()
