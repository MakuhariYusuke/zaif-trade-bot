#!/usr/bin/env python3
"""
カリキュラム学習実行スクリプト
P0→P2までのレジーム層で段階的に学習
"""

import json
import os
from pathlib import Path
from typing import Any, Dict

from ztb.training.core.unified_trainer import UnifiedTrainer
from ztb.utils.file_utils import safe_json_load


def run_curriculum_stage(
    stage_name: str,
    config_updates: Dict[str, Any],
    base_config_path: str = "unified_training_config.json",
) -> bool:
    """指定されたカリキュラムステージを実行"""
    print(f"\n=== カリキュラム学習: {stage_name} ===")

    # ベース設定を読み込み
    config = safe_json_load(Path(base_config_path))

    # ステージ固有の設定を更新
    config.update(config_updates)

    # セッションIDを更新
    config["session_id"] = f"curriculum_{stage_name}"

    # 設定ファイルを一時保存
    stage_config_path = f"curriculum_{stage_name}_config.json"
    with open(stage_config_path, "w") as f:
        json.dump(config, f, indent=2)

    try:
        # トレーニング実行
        trainer = UnifiedTrainer(config)
        result = trainer.train()

        if result:
            print(f"✓ {stage_name} 完了")
            return True
        else:
            print(f"✗ {stage_name} 失敗")
            return False

    except Exception as e:
        print(f"✗ {stage_name} 失敗: {e}")
        return False

    finally:
        # 一時ファイルを削除
        if os.path.exists(stage_config_path):
            os.remove(stage_config_path)


def main() -> None:
    """カリキュラム学習メイン実行"""
    print("カリキュラム学習を開始します...")

    # データ読み込み確認
    data_path = "ml-dataset-enhanced.csv"
    if not os.path.exists(data_path):
        print(f"データファイルが見つかりません: {data_path}")
        return

    # カリキュラムステージ定義
    curriculum_stages: list[dict[str, Any]] = [
        {
            "name": "P0_forced_balance",
            "config": {
                "curriculum_stage": "forced_balance",
                "total_timesteps": 50000,
                "ent_coef": 0.8,  # 高エントロピー探索
                "target_kl": 0.05,  # 緩いKL制約
            },
        },
        {
            "name": "P1_balanced_transition",
            "config": {
                "curriculum_stage": "balanced_transition",
                "total_timesteps": 100000,
                "ent_coef": 0.6,
                "target_kl": 0.03,
            },
        },
        {
            "name": "P2_full_curriculum",
            "config": {
                "curriculum_stage": "full",
                "total_timesteps": 200000,
                "ent_coef": 0.4,
                "target_kl": 0.02,
            },
        },
    ]

    # 各ステージを実行
    for stage in curriculum_stages:
        stage_name = stage["name"]
        stage_config = stage["config"]
        success = run_curriculum_stage(stage_name, stage_config)

        if not success:
            print(f"カリキュラム学習が {stage_name} で失敗しました")
            return

        # 各ステージ後に評価
        evaluate_stage_performance(stage_name)

    print("\n🎉 カリキュラム学習完了!")


def evaluate_stage_performance(stage_name: str) -> None:
    """ステージ後の性能評価"""
    print(f"\n--- {stage_name} 評価 ---")

    model_path = f"models/curriculum_{stage_name}.zip"
    if not os.path.exists(model_path):
        print(f"モデルファイルが見つかりません: {model_path}")
        return

    # run_regime_eval.py を使用して評価
    import subprocess

    result = subprocess.run(
        [
            "python",
            "-m",
            "ztb.analysis.regime.run_regime_eval",
            "--price-data",
            "ml-dataset-enhanced.csv",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("評価完了")
        # 結果ファイルを解析
        results_file = Path("results/regime_analysis/regime_analysis_results.json")
        if results_file.exists():
            results = safe_json_load(results_file)

            model_results = results.get("regime_metrics", {}).get(stage_name, {})
            for regime, metrics in model_results.items():
                action_dist = metrics.get("action_distribution", {})
                print(
                    f"{regime}: BUY={action_dist.get('BUY', 0):.1f}%, SELL={action_dist.get('SELL', 0):.1f}%, HOLD={action_dist.get('HOLD', 0):.1f}%"
                )
    else:
        print(f"評価失敗: {result.stderr}")


if __name__ == "__main__":
    main()
