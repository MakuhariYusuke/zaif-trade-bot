#!/usr/bin/env python3
"""
Day 8 Phase B: ent_coef Ablation Study

70# / 71# Review対応:
- ent_coef仮説の直接検証
- SAC_TUNEDでent_coef=[0.01, 0.05, 0.1, "auto"]を比較
- 実効設定の明示的ログ出力
- 4 seeds for statistical power
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json
import numpy as np
import logging

# Project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Skip scipy/sklearn imports
os.environ.setdefault("ZTB_SKIP_SCIPY", "1")
os.environ.setdefault("ZTB_SKIP_SKLEARN", "1")

from ztb.training.unified_trainer import UnifiedTrainer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)
logging.getLogger(__name__).setLevel(logging.WARNING)


def to_python(obj):
    """NumPy型をPython標準型に変換"""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(item) for item in obj]
    return obj


# Constants
DATA_PATH = str(project_root / "data" / "btc_jpy_1m_v451_optimized_features.parquet")
OUTPUT_DIR = str(project_root / "results" / "phase4_day8_ent_coef_ablation")
MINUTES_PER_YEAR = 525600

# SAC Base (from SAC_TUNED minus ent_coef)
SAC_TUNED_BASE = {
    "learning_rate": 0.0005,
    "buffer_size": 25000,
    "learning_starts": 500,
    "batch_size": 128,
    "tau": 0.005,
    "gamma": 0.95,
    "train_freq": 1,
    "gradient_steps": 2,
    "target_update_interval": 1,
    "target_entropy": "auto"
}

# ent_coef variations
ENT_COEFS = {
    "ent_001": 0.01,
    "ent_005": 0.05,
    "ent_010": 0.1,
    "ent_auto": "auto"
}

# Reward config (scale=100 unified)
REWARD_CONFIG = {
    "name": "pure_pnl_scaled",
    "description": "Pure PnL with scale=100",
    "curriculum_stage": "simple",
    "use_simple_reward": True,
    "reward_scale": 100.0,
    "reward_scaling": 100.0,
    "reward_clip_min": -100.0,
    "reward_clip_max": 100.0,
    "profit_weight": 1.0,
    "risk_weight": 0.0,
    "consistency_weight": 0.0,
    "trading_bonus": 0.0,
    "hold_penalty": 0.0,
    "trade_frequency_penalty": 0.0,
    "action_smoothing": 0.0,
}

# Seeds (4 for statistical power)
SEEDS = [42, 123, 456, 789]


def run_experiment(
    ent_coef_name: str,
    ent_coef_value: Any,
    seed: int,
    total_timesteps: int = 25000
) -> Dict[str, Any]:
    """Run single ent_coef ablation experiment."""
    
    exp_id = f"ent_{ent_coef_name}_seed{seed}"
    
    # SAC config with ent_coef
    sac_config = {**SAC_TUNED_BASE, "ent_coef": ent_coef_value}
    
    # 71# Review対応: 実効設定の明示的ログ
    logger.warning(f"\n{'='*80}")
    logger.warning(f"🔬 EXPERIMENT: {exp_id}")
    logger.warning(f"{'='*80}")
    logger.warning(f"📊 Effective Configuration [71# Review]:")
    logger.warning(f"  ent_coef: {ent_coef_value}")
    logger.warning(f"  reward_scale: {REWARD_CONFIG['reward_scale']}")
    logger.warning(f"  reward_scaling: {REWARD_CONFIG['reward_scaling']}")
    logger.warning(f"  reward_clip: [{REWARD_CONFIG['reward_clip_min']}, {REWARD_CONFIG['reward_clip_max']}]")
    logger.warning(f"  gamma: {sac_config['gamma']}")
    logger.warning(f"  batch_size: {sac_config['batch_size']}")
    logger.warning(f"  gradient_steps: {sac_config['gradient_steps']}")
    logger.warning(f"  learning_rate: {sac_config['learning_rate']}")
    logger.warning(f"  seed: {seed}")
    logger.warning(f"  total_timesteps: {total_timesteps}")
    logger.warning(f"{'='*80}\n")
    
    # UnifiedTrainer config (構造は過去スクリプトと一致させる)
    config = {
        "training": {
            "algorithm": "SAC",
            "total_timesteps": total_timesteps,
            "eval_freq": 10000,
            "n_eval_episodes": 5,
            "log_interval": 500,
            "seed": seed,
            "sac_hyperparameters": sac_config,  # "sac"ではなく"sac_hyperparameters"
            "data_config": {
                "data_path": DATA_PATH,
                "window_size": 60
            },
            "environment": {
                "use_continuous_actions": True,
                "action_space_type": "continuous",
                "initial_portfolio_value": 100000.0,
                "transaction_cost": 0.001,
                "use_precomputed_features": True,
                "feature_set": "minimal",
                "reward_settings": REWARD_CONFIG,
            },
            "walk_forward": {
                "enabled": False
            }
        },
        "experiment_name": exp_id,
        "output_dir": OUTPUT_DIR
    }
    
    # Train
    start_time = datetime.now()
    trainer = UnifiedTrainer(config)
    success = trainer.run()
    elapsed = (datetime.now() - start_time).total_seconds()
    
    if not success:
        logger.error(f"❌ Experiment failed: {exp_id}")
        return {
            "status": "failed",
            "experiment_id": exp_id,
            "timestamp": datetime.now().isoformat()
        }
    
    # Get stats
    training_stats = trainer.get_training_stats()
    
    # Basic metrics (numpy型をPython標準型に変換)
    final_reward = float(training_stats.get("final_reward", 0.0))
    action_dist_raw = training_stats.get("action_distribution", {})
    action_dist = {k: float(v) for k, v in action_dist_raw.items()}
    
    # ROI estimation (from final reward)
    estimated_roi = final_reward * 100
    
    logger.warning(f"\n✅ {exp_id} COMPLETED")
    logger.warning(f"  Final Reward: {final_reward:.6e}")
    logger.warning(f"  Estimated ROI: {estimated_roi:.2f}%")
    logger.warning(f"  HOLD ratio: {action_dist.get('HOLD', 0.0)*100:.1f}%")
    logger.warning(f"  Training time: {elapsed:.1f}s\n")
    
    result = {
        "status": "completed",
        "experiment_id": exp_id,
        "ent_coef_name": ent_coef_name,
        "ent_coef_value": str(ent_coef_value),  # "auto" -> string
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "final_reward": final_reward,
            "estimated_roi_pct": estimated_roi,
            "action_distribution": action_dist,
            "elapsed_seconds": float(elapsed),
        },
        "config": {
            "sac_ent_coef": str(ent_coef_value),
            "sac_gamma": sac_config["gamma"],
            "sac_batch_size": sac_config["batch_size"],
            "sac_gradient_steps": sac_config["gradient_steps"],
            "reward_scale": REWARD_CONFIG["reward_scale"],
            "reward_scaling": REWARD_CONFIG["reward_scaling"],
        }
    }
    
    # NumPy型を確実に変換
    return to_python(result)


def main():
    """Run ent_coef ablation study."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.warning("\n" + "="*80)
    logger.warning("🧪 DAY 8 PHASE B: ent_coef ABLATION STUDY")
    logger.warning("="*80)
    logger.warning(f"Experiments: {len(ENT_COEFS)} ent_coef values × {len(SEEDS)} seeds = {len(ENT_COEFS) * len(SEEDS)} runs")
    logger.warning(f"Output: {OUTPUT_DIR}")
    logger.warning("="*80 + "\n")
    
    all_results = []
    interim_path = Path(OUTPUT_DIR) / "day8b_ablation_interim.json"  # 固定ファイル名
    
    for ent_name, ent_val in ENT_COEFS.items():
        for seed in SEEDS:
            try:
                result = run_experiment(ent_name, ent_val, seed)
                all_results.append(result)
                
                # Save interim results (overwrite)
                with open(interim_path, "w") as f:
                    json.dump(all_results, f, indent=2)
                    
            except Exception as e:
                logger.error(f"❌ Exception in {ent_name}_seed{seed}: {e}")
                all_results.append({
                    "status": "exception",
                    "experiment_id": f"ent_{ent_name}_seed{seed}",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                # Continue with next experiment
    
    # Aggregate results
    summary = {}
    for ent_name in ENT_COEFS.keys():
        ent_results = [r for r in all_results if r.get("ent_coef_name") == ent_name and r.get("status") == "completed"]
        
        if not ent_results:
            continue
        
        rois = [r["metrics"]["estimated_roi_pct"] for r in ent_results]
        hold_ratios = [r["metrics"]["action_distribution"].get("HOLD", 0.0) for r in ent_results]
        
        summary[ent_name] = {
            "n_seeds": len(ent_results),
            "roi_mean": float(np.mean(rois)),
            "roi_std": float(np.std(rois, ddof=1)) if len(rois) > 1 else 0.0,
            "roi_min": float(np.min(rois)),
            "roi_max": float(np.max(rois)),
            "hold_ratio_mean": float(np.mean(hold_ratios)),
        }
    
    # Save final analysis
    analysis = {
        "summary": to_python(summary),  # numpy型変換
        "interpretation": [
            "ent_coef='auto'が最良ならent_coef=0.01仮説を支持",
            "ent_coef=0.1が'auto'に近いなら、scale=100環境では高ent_coefが適切",
            "HOLD率の減少とROI改善が相関すれば、探索不足仮説を支持"
        ],
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_ent_coefs": len(ENT_COEFS),
            "n_seeds": len(SEEDS),
            "total_experiments": len(all_results),
            "successful": sum(1 for r in all_results if r.get("status") == "completed"),
        }
    }
    
    analysis_path = Path(OUTPUT_DIR) / f"day8b_ablation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    # Print summary
    logger.warning("\n" + "="*80)
    logger.warning("📊 ABLATION STUDY RESULTS SUMMARY")
    logger.warning("="*80)
    for ent_name, stats in summary.items():
        logger.warning(f"\n{ent_name}:")
        logger.warning(f"  ROI: {stats['roi_mean']:.2f}% ± {stats['roi_std']:.2f}%")
        logger.warning(f"  Range: [{stats['roi_min']:.2f}%, {stats['roi_max']:.2f}%]")
        logger.warning(f"  HOLD ratio: {stats['hold_ratio_mean']*100:.1f}%")
    
    logger.warning(f"\n✅ Analysis saved: {analysis_path}")
    logger.warning("="*80 + "\n")


if __name__ == "__main__":
    main()
