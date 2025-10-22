#!/usr/bin/env python3
"""
SAC v434.2 報酬関数改良スクリプト
v434.1の問題（0%収益、過度な取引）を解決するための報酬関数改良
"""

import json
from pathlib import Path
from typing import Any, Dict


def create_v434_2_reward_config() -> Dict[str, Any]:
    """
    v434.2の改良された報酬関数設定を作成
    """
    reward_config = {
        "curriculum_stage": "pnl_focused",  # v434.1と同じステージを使用
        # 利益ボーナスの大幅強化
        "base_profit_bonus_atr_coeff": 5.0,  # 1.5 → 5.0 (利益ボーナス3.3倍)
        "base_profit_bonus_portfolio_coeff": 10.0,  # 1.2 → 10.0 (ポートフォリオ係数8.3倍)
        # アクションペナルティの強化（過度な取引抑制）
        "base_action_penalty": 0.15,  # 0.015 → 0.15 (取引コスト10倍)
        # HOLDペナルティの調整（ポジション維持時の機会費用）
        "hold_penalty_base": 0.02,  # 0.01 → 0.02
        "hold_penalty_position_factor": 0.08,  # 0.04 → 0.08 (ポジション維持コスト2倍)
        # 損失ペナルティの強化
        "loss_penalty_coeff": -1.0,  # -0.2 → -1.0 (損失ペナルティ5倍)
        # アクション頻度ペナルティ（連続取引抑制）
        "action_frequency_penalty": 0.05,  # 新規：連続取引に対する追加ペナルティ
        "max_consecutive_trades": 3,  # 最大連続取引回数
        # 取引間隔ボーナス（取引間隔を空けることを奨励）
        "trade_interval_bonus": 0.03,  # 取引間隔ごとのボーナス
        "min_trade_interval": 5,  # 最小取引間隔（ステップ数）
        # 利益実現ボーナス（大きな利益に対する追加報酬）
        "profit_realization_bonus": 2.0,  # 利益実現時の追加乗数
        "min_profit_threshold": 100.0,  # 利益ボーナス適用閾値（円）
        # ポジションサイズペナルティの調整
        "position_penalty_base": 0.001,  # ポジションサイズに対する基本ペナルティ
        "position_penalty_scaling": 0.01,  # ポジションサイズ超過に対するペナルティ
        # 報酬スケーリングの調整
        "reward_scaling": 1.0,  # 基本スケーリング
        "profit_scaling": 2.0,  # 利益に対する追加スケーリング
        # クリッピング範囲の拡大（より大きな報酬/ペナルティを許可）
        "reward_clip_min": -200.0,  # -80.0 → -200.0
        "reward_clip_max": 200.0,  # 80.0 → 200.0
        # トレンド分析の強化
        "trend_analysis_weight": 1.5,  # トレンド分析の重み
        # ボラティリティ調整
        "volatility_penalty": 0.02,  # 高ボラティリティ時の追加ペナルティ
        # 説明
        "_description": "v434.2報酬関数改良設定",
        "_improvements": [
            "取引コストを10倍に強化（0.015→0.15）で過度な取引を抑制",
            "利益ボーナスを大幅増加（ATR係数3.3倍、ポートフォリオ係数8.3倍）",
            "損失ペナルティを5倍に強化（-0.2→-1.0）",
            "アクション頻度ペナルティを追加（連続取引抑制）",
            "取引間隔ボーナスを追加（取引間隔を奨励）",
            "利益実現ボーナスを追加（大きな利益を奨励）",
            "報酬クリッピング範囲を拡大（より大きな報酬/ペナルティを許可）",
        ],
    }

    return reward_config


def create_v434_2_environment_config() -> Dict[str, Any]:
    """
    v434.2の環境設定を作成
    """
    env_config = {
        # 学習戦略の改善
        "curriculum_stage": "pnl_focused",
        # 取引コストの現実化
        "transaction_cost": 0.0015,  # 0.15% (Zaifの実取引コストに近づける)
        # 最大ポジションサイズの調整
        "max_position_size": 0.1,  # ポートフォリオの10%に制限
        # 報酬スケーリング
        "reward_scaling": 1.0,
        # 特徴量数の削減（次元削減）
        "enable_correlation_reduction": True,
        "correlation_threshold": 0.85,
        # ランダムスタートの有効化（決定論的行動対策）
        "random_start": True,
        # 説明
        "_description": "v434.2環境設定",
        "_improvements": [
            "取引コストを実取引に近づける（0.15%）",
            "最大ポジションサイズを10%に制限",
            "相関削減を有効化して特徴量数を削減",
            "ランダムスタートを有効化して決定論的行動を回避",
        ],
    }

    return env_config


def save_v434_2_config():
    """
    v434.2の設定をJSONファイルに保存
    """
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    # 報酬設定
    reward_config = create_v434_2_reward_config()
    reward_path = config_dir / "sac_v434_2_reward_config.json"
    with open(reward_path, "w", encoding="utf-8") as f:
        json.dump(reward_config, f, indent=2, ensure_ascii=False)

    # 環境設定
    env_config = create_v434_2_environment_config()
    env_path = config_dir / "sac_v434_2_environment_config.json"
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(env_config, f, indent=2, ensure_ascii=False)

    print("v434.2設定ファイルを保存しました:")
    print(f"  報酬設定: {reward_path}")
    print(f"  環境設定: {env_path}")

    # 設定内容の表示
    print("\n=== v434.2 報酬関数改良内容 ===")
    for improvement in reward_config["_improvements"]:
        print(f"• {improvement}")

    print("\n=== v434.2 環境設定改良内容 ===")
    for improvement in env_config["_improvements"]:
        print(f"• {improvement}")


if __name__ == "__main__":
    save_v434_2_config()
