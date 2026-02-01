#!/usr/bin/env python3
"""
Phase 4 Day 6-7: 報酬調整A/Bテスト実行スクリプト
52番計画: 5 configs × 2 seeds = 10実験

実験構成:
  A (Baseline): 現状設定（45番結果: ROI -5.074%）
  B (Stage1): stage1_basic.yaml（0番Stage 1準拠）
  C (Hold削除): stage1_hold_removed.yaml（49番優先1）
  D (取引抑制): stage1_trade_reduced.yaml（49番優先2）
  E (探索調整): stage1_exploration_tuned.yaml（49番優先3 + SAC調整）

Usage:
    python scripts/v459/run_day6_reward_tuning.py [--limit N]
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import logging

# Project root setup
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Environment setup
os.environ.setdefault("ZTB_SIGINT_POLICY", "ignore" if os.name == "nt" else "default")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Improve Windows/long-run stability and skip heavy imports for smoke runs
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("ZTB_SAFE_DATETIME", "1")
os.environ.setdefault("ZTB_SKIP_SCIPY", "1")
os.environ.setdefault("ZTB_SKIP_SKLEARN", "1")
os.environ.setdefault("SKIP_HEAVY_IMPORTS", "1")

from ztb.training.unified_trainer.trainer import UnifiedTrainer
from ztb.training.reward_config_schema import load_reward_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 実験設定
SEEDS = [42, 123]
DATA_PATH = str(project_root / "data" / "btc_jpy_1m_v451_optimized_features.parquet")

# 報酬設定（実験A-E）
REWARD_CONFIGS = [
    None,  # A: Baseline（報酬設定なし、デフォルト）
    "configs/rewards/stage1_basic.yaml",  # B: Stage 1
    "configs/rewards/stage1_hold_removed.yaml",  # C: Hold削除
    "configs/rewards/stage1_trade_reduced.yaml",  # D: 取引抑制
    "configs/rewards/stage1_exploration_tuned.yaml"  # E: 探索調整
]

EXPERIMENT_NAMES = ["A_Baseline", "B_Stage1", "C_HoldRemoved", "D_TradeReduced", "E_ExplorationTuned"]

# SAC設定（実験Eのみ特別）
SAC_DEFAULT = {
    "learning_rate": 0.0003,
    "buffer_size": 25000,  # 最適化: 50000 → 25000（時間-10%、メモリ-100MB）
    "learning_starts": 500,  # 最適化: 1000 → 500（早期学習開始）
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "target_entropy": "auto"
}

SAC_EXPLORATION_TUNED = {
    "learning_rate": 0.0005,  # 0.0003 → 0.0005
    "buffer_size": 25000,  # 最適化適用
    "learning_starts": 500,  # 最適化適用
    "batch_size": 128,  # 256 → 128
    "tau": 0.005,
    "gamma": 0.95,  # 0.99 → 0.95
    "train_freq": 1,
    "gradient_steps": 2,  # 1 → 2
    "ent_coef": 0.01,  # "auto" → 0.01（固定）
    "target_update_interval": 1,
    "target_entropy": "auto"
}


def create_experiment_config(
    experiment_name: str,
    seed: int,
    reward_config_path: str | None,
    sac_params: Dict[str, Any]
) -> Dict[str, Any]:
    """実験設定を作成"""
    config = {
        "training": {
            "algorithm": "SAC",
            "total_timesteps": 50000,
            "eval_freq": 10000,
            "n_eval_episodes": 5,
            "log_interval": 500,
            "seed": seed,
            "sac_hyperparameters": sac_params,
            "data_config": {
                "data_path": DATA_PATH,
                "window_size": 60
            },
            "environment": {
                "use_continuous_actions": True,
                "action_space_type": "continuous",
                "initial_portfolio_value": 100000.0,
                "transaction_cost": 0.001,  # 0.1%
                "use_precomputed_features": True,
                "feature_set": "minimal"
            },
            "walk_forward": {
                "enabled": False  # 単一実行（52番仕様）
            }
        },
        "experiment_name": f"{experiment_name}_seed{seed}",
        "output_dir": str(project_root / "results" / "phase4_day6_reward_tuning")
    }
    
    # 報酬設定を環境設定に注入（dict形式で）
    if reward_config_path:
        from ztb.training.reward_config_schema import RewardConfigSchema
        # load_and_validate returns a dict suitable for EnvironmentConfig.reward_settings
        reward_dict = RewardConfigSchema.load_and_validate(str(project_root / reward_config_path))
        # Extract behavior_optimization so it propagates to EnvironmentConfig correctly
        behavior_opt = reward_dict.pop("behavior_optimization", None)
        # Inject reward settings into environment section so EnvironmentConfig.from_dict can construct RewardSettings
        config["training"]["environment"]["reward_settings"] = reward_dict
        # If behavior optimization present, inject it directly into environment so the mapping logic applies
        if behavior_opt:
            config["training"]["environment"]["behavior_optimization"] = behavior_opt

    return config


def run_single_experiment(
    experiment_name: str,
    seed: int,
    reward_config_path: str | None,
    sac_params: Dict[str, Any]
) -> Dict[str, Any]:
    """単一実験を実行"""
    exp_id = f"{experiment_name}_seed{seed}"
    logger.info(f"=" * 80)
    logger.info(f"実験開始: {exp_id}")
    logger.info(f"=" * 80)
    
    start_time = time.time()
    
    try:
        # 設定作成
        config = create_experiment_config(experiment_name, seed, reward_config_path, sac_params)
        
        # Trainer作成
        trainer = UnifiedTrainer(config)
        
        # 学習実行
        success = trainer.run()
        
        elapsed = time.time() - start_time
        
        if not success:
            raise ValueError("Training failed")
        
        # レポート取得
        report = trainer.get_training_report()
        
        # レポート全体をJSON互換に変換
        import numpy as np
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        report = convert_numpy(report)
        
        # メトリクス抽出（エラー時はデフォルト値）
        try:
            metrics = extract_metrics(report)
        except Exception as e:
            logger.warning(f"メトリクス抽出失敗、デフォルト値使用: {e}")
            metrics = {
                "final_reward": 0.0,
                "steps_per_second": 0.0,
                "sell_ratio": 0.0,
                "action_distribution": {}
            }
        
        metrics["elapsed_seconds"] = elapsed
        
        logger.info(f"実験完了: {exp_id} ({elapsed:.1f}秒)")
        logger.info(f"  Final Reward: {metrics.get('final_reward', 0):.6f}")
        logger.info(f"  SELL比率: {metrics.get('sell_ratio', 0):.2%}")
        logger.info(f"  Steps/sec: {metrics.get('steps_per_second', 0):.1f}")
        
        return {
            "experiment_name": exp_id,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "metrics": metrics,
            "report": report
        }
        
    except Exception as e:
        import traceback
        logger.error(f"実験失敗: {exp_id} - {e}")
        logger.error(traceback.format_exc())
        
        return {
            "experiment_name": exp_id,
            "status": "failed",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


def extract_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    """レポートからメトリクス抽出"""
    import numpy as np
    
    training_stats = report.get("training_stats", {})
    
    # numpy型をPython標準型に変換
    def to_python_type(value):
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif hasattr(value, 'item'):  # numpy scalar
            return value.item()
        return value
    
    # action_distributionから取引比率を計算
    action_dist_raw = training_stats.get("action_distribution", {})
    action_dist = {k: to_python_type(v) for k, v in action_dist_raw.items()}
    total_actions = sum(action_dist.values()) if action_dist else 1
    sell_count = action_dist.get("SELL", 0)
    
    return {
        "final_reward": to_python_type(training_stats.get("final_reward", 0)),
        "steps_per_second": to_python_type(training_stats.get("steps_per_second", 0)),
        "total_timesteps": to_python_type(training_stats.get("total_timesteps", 0)),
        "training_time": to_python_type(training_stats.get("training_time", 0)),
        "sell_ratio": to_python_type(sell_count / total_actions if total_actions > 0 else 0),
        "action_distribution": action_dist
    }


def run_all_experiments(limit: int | None = None) -> List[Dict[str, Any]]:
    """全実験を実行"""
    results = []
    
    total_experiments = len(EXPERIMENT_NAMES) * len(SEEDS)
    if limit:
        total_experiments = min(total_experiments, limit)
    
    logger.info(f"総実験数: {total_experiments}")
    logger.info(f"Seeds: {SEEDS}")
    logger.info(f"Configs: {EXPERIMENT_NAMES}")
    
    experiment_count = 0
    
    for i, (exp_name, reward_path) in enumerate(zip(EXPERIMENT_NAMES, REWARD_CONFIGS)):
        # 実験Eの場合、SAC設定を変更
        sac_params = SAC_EXPLORATION_TUNED if exp_name == "E_ExplorationTuned" else SAC_DEFAULT
        
        for seed in SEEDS:
            if limit and experiment_count >= limit:
                logger.info(f"実験数制限に到達: {limit}")
                break
            
            result = run_single_experiment(exp_name, seed, reward_path, sac_params)
            results.append(result)
            
            experiment_count += 1
        
        if limit and experiment_count >= limit:
            break
    
    return results


def save_results(results: List[Dict[str, Any]], output_dir: Path) -> None:
    """結果を保存"""
    import numpy as np
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    import dataclasses
    import pathlib

    # numpy型やdataclass等をJSON互換にする変換関数
    def convert_to_json_serializable(obj):
        # NumPy
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # dataclass (e.g., RewardSettings)
        elif dataclasses.is_dataclass(obj):
            return convert_to_json_serializable(dataclasses.asdict(obj))
        # pathlib.Path
        elif isinstance(obj, pathlib.Path):
            return str(obj)
        # datetime
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        return obj

    # 結果を変換
    results_serializable = convert_to_json_serializable(results)
    
    # 詳細結果
    results_file = output_dir / f"day6_reward_tuning_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    logger.info(f"結果保存: {results_file}")
    
    # サマリー
    summary = {
        "timestamp": timestamp,
        "total_experiments": len(results),
        "completed": sum(1 for r in results if r["status"] == "completed"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "experiments": [
            {
                "name": r["experiment_name"],
                "status": r["status"],
                "final_reward": r.get("metrics", {}).get("final_reward", 0),
                "sell_ratio": r.get("metrics", {}).get("sell_ratio", 0),
                "training_time": r.get("metrics", {}).get("training_time", 0)
            }
            for r in results
        ]
    }
    
    summary_file = output_dir / f"day6_summary_{timestamp}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"サマリー保存: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Day 6 報酬調整A/Bテスト")
    parser.add_argument("--limit", type=int, help="実験数制限（デバッグ用）")
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Phase 4 Day 6-7: 報酬調整A/Bテスト")
    logger.info("52番計画: 5 configs × 2 seeds = 10実験")
    logger.info("=" * 80)
    
    # データファイル確認
    if not Path(DATA_PATH).exists():
        logger.error(f"データファイル未発見: {DATA_PATH}")
        logger.error("実行: python scripts/v459/precompute_optimized_features.py")
        return 1
    
    # 実験実行
    results = run_all_experiments(limit=args.limit)
    
    # 結果保存
    output_dir = project_root / "results" / "phase4_day6_reward_tuning"
    save_results(results, output_dir)
    
    # 最終サマリー
    logger.info("=" * 80)
    logger.info("全実験完了")
    logger.info(f"総実験数: {len(results)}")
    logger.info(f"成功: {sum(1 for r in results if r['status'] == 'completed')}")
    logger.info(f"失敗: {sum(1 for r in results if r['status'] == 'failed')}")
    logger.info("=" * 80)
    
    # ROI改善確認
    completed = [r for r in results if r["status"] == "completed"]
    if completed:
        logger.info("\nFinal Reward結果（2seed平均）:")
        for exp_name in EXPERIMENT_NAMES:
            exp_results = [r for r in completed if exp_name in r["experiment_name"]]
            if exp_results:
                avg_reward = sum(r["metrics"]["final_reward"] for r in exp_results) / len(exp_results)
                avg_sell_ratio = sum(r["metrics"]["sell_ratio"] for r in exp_results) / len(exp_results)
                logger.info(f"  {exp_name}: reward={avg_reward:.6f}, SELL={avg_sell_ratio:.2%}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
