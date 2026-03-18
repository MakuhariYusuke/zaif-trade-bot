#!/usr/bin/env python3
"""
P1: 基準モデル作成 - PnLのみ報酬で基準を確立（簡略版）

89#に基づき、ペナルティなしのPnLのみ報酬で基準モデルを作成。
Day11のrun_day11_verification.pyを参考に実装。

P1-1のみ実行（1シード、5000ステップで検証）
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer
from ztb.training.utils.env_metrics import (
    extract_trainer_env_metrics,
    compute_balance_roi,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# P1 実験設定
# ============================================================================

DATA_PATH = str(project_root / "data" / "btc_jpy_1m_v451_optimized_features.parquet")
OUTPUT_DIR = project_root / "experiments" / "p1_baseline" / datetime.now().strftime("%Y%m%d_%H%M%S")

# SAC基本設定（Day11ベース）
SAC_DEFAULT = {
    "learning_rate": 0.0003,
    "buffer_size": 100000,
    "learning_starts": 1000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "target_entropy": "auto",
}


def create_p1_config(
    experiment_id: str,
    reward_params: dict[str, float],
    seed: int = 42,
    total_timesteps: int = 5000,
) -> dict:
    """P1実験設定を生成"""
    return {
        "experiment_name": experiment_id,
        "training": {
            "algorithm": "SAC",
            "total_timesteps": total_timesteps,
            "eval_freq": 1000,
            "n_eval_episodes": 3,
            "log_interval": 100,
            "seed": seed,
            "sac_hyperparameters": SAC_DEFAULT.copy(),
            "data_config": {
                "data_path": DATA_PATH,
                "window_size": 60,
            },
            "environment": {
                "use_continuous_actions": True,
                "action_space_type": "continuous",
                "initial_portfolio_value": 100000.0,
                "transaction_cost": 0.001,
                # MTF特徴量生成を無効化（高速化）
                "feature_flags": {
                    "include_multi_timeframe_features": False,
                },
            },
            "walk_forward": {
                "enabled": False,
            },
        },
        "reward": reward_params,
    }


def run_p1_experiment(config: dict) -> dict[str, Any]:
    """P1実験を実行"""
    start_time = time.time()
    experiment_id = config["experiment_name"]

    logger.info(f"\n{'='*60}")
    logger.info(f"🔬 実験開始: {experiment_id}")
    logger.info(f"{'='*60}")
    logger.info(f"  Reward params: {config['reward']}")
    logger.info(f"  Total timesteps: {config['training']['total_timesteps']}")
    logger.info(f"  Seed: {config['training']['seed']}")

    try:
        # SACTrainer実行
        trainer = SACTrainer(config=config, logger=logger)
        result = trainer.train()
        
        elapsed_time = time.time() - start_time

        # 結果を整形
        training_result = {
            "experiment_id": experiment_id,
            "seed": config["training"]["seed"],
            "total_time_seconds": elapsed_time,
            "total_timesteps": config["training"]["total_timesteps"],
            "reward_params": config["reward"],
            "success": True,
            "timestamp": datetime.now().isoformat(),
        }

        # 環境メトリクスを抽出
        try:
            metrics = extract_trainer_env_metrics(trainer, include_optional=True)
            if metrics:
                logger.info("環境メトリクス取得成功:")
                
                # gross_pnl/net_pnl等を取得
                for key in ['final_balance', 'initial_balance', 'gross_pnl', 'net_pnl', 
                           'total_fees', 'total_slippage', 'total_trades']:
                    if key in metrics:
                        training_result[key] = metrics[key]
                        logger.info(f"  {key}: {metrics[key]}")
                
                # ROI計算
                roi = compute_balance_roi(metrics)
                if roi is not None:
                    training_result["balance_roi"] = roi
                    logger.info(f"  Balance ROI: {roi:+.2f}%")
                
                # Gross/Net ROI計算
                if 'gross_pnl' in metrics and 'initial_balance' in metrics:
                    gross_roi = (metrics['gross_pnl'] / metrics['initial_balance']) * 100
                    training_result["gross_roi"] = gross_roi
                    logger.info(f"  Gross ROI: {gross_roi:+.2f}%")
                
                if 'net_pnl' in metrics and 'initial_balance' in metrics:
                    net_roi = (metrics['net_pnl'] / metrics['initial_balance']) * 100
                    training_result["net_roi"] = net_roi
                    logger.info(f"  Net ROI: {net_roi:+.2f}%")
            else:
                logger.warning("環境メトリクス取得失敗")
        except Exception as env_error:
            logger.error(f"環境メトリクス取得エラー: {env_error}", exc_info=True)

        logger.info(f"\n✅ 実験完了: {experiment_id} ({elapsed_time:.1f}秒)")
        return training_result

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ 実験失敗: {experiment_id} - {e}", exc_info=True)
        return {
            "experiment_id": experiment_id,
            "seed": config["training"]["seed"],
            "total_time_seconds": elapsed_time,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def main():
    """P1-1実験のみ実行（検証用）"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("P1: 基準モデル作成実験（簡略版）")
    logger.info(f"出力ディレクトリ: {OUTPUT_DIR}")
    logger.info("=" * 70)
    
    # P1-1: PnLのみ（ペナルティ全無効）
    logger.info("\nP1-1: PnLのみ（ペナルティ全無効）実験を実行...")
    
    reward_params_p1_1 = {
        "alpha": 0.0,              # position change penalty OFF
        "beta": 0.0,               # holding time penalty OFF
        "gamma": 0.0,              # inventory risk OFF
        "fee_penalty_weight": 0.0, # extra fee penalty OFF
        "edge_penalty_rate": 0.0,  # edge penalty OFF
        "vol_floor_penalty": 0.0,  # vol floor penalty OFF
        "hold_ramp": 0.0,          # time decay OFF
    }
    
    config = create_p1_config(
        experiment_id="P1-1_pnl_only",
        reward_params=reward_params_p1_1,
        seed=42,
        total_timesteps=5000,  # 検証用に短縮
    )
    
    result = run_p1_experiment(config)
    
    # 結果保存
    result_file = OUTPUT_DIR / "p1_1_result.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n結果保存完了: {result_file}")
    
    # 結果サマリー
    logger.info("\n" + "=" * 70)
    logger.info("P1-1 結果サマリー")
    logger.info("=" * 70)
    logger.info(f"Experiment ID: {result.get('experiment_id')}")
    logger.info(f"Success: {result.get('success')}")
    logger.info(f"Gross ROI: {result.get('gross_roi', 'N/A')}")
    logger.info(f"Net ROI: {result.get('net_roi', 'N/A')}")
    logger.info(f"Balance ROI: {result.get('balance_roi', 'N/A')}")
    logger.info(f"Total Trades: {result.get('total_trades', 'N/A')}")
    logger.info(f"Total Fees: {result.get('total_fees', 'N/A')}")
    logger.info("=" * 70)
    
    # 判断基準
    balance_roi = result.get('balance_roi')
    if balance_roi is not None:
        if balance_roi > 0:
            logger.info("✅ P1-1 (PnLのみ) > 0%: 取引自体は利益。コスト/ペナルティ調整で改善可能")
        else:
            logger.info("⚠️ P1-1 (PnLのみ) < 0%: 取引戦略自体が損失。学習設計見直し必要")


if __name__ == "__main__":
    main()
