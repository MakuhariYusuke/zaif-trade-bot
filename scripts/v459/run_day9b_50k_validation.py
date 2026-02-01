#!/usr/bin/env python3
"""
Day 9 Phase B: 50k Steps Validation

75#/77# 対応:
- gamma=0.99 + ent_coef=0.01 の最適設定で50k stepsを実行
- 25k (ROI=-6.29%) → 50k で改善確認
- 45# Day5 (50k, ROI=-5.07%) との比較

期待結果:
- ROI > -6% (25kからの改善)
- ROI > -5% (45# Day5水準達成)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
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
OUTPUT_DIR = str(project_root / "results" / "phase4_day9_50k_validation")

# SAC Optimized config (75#で確定)
SAC_OPTIMIZED = {
    "learning_rate": 0.0005,
    "buffer_size": 25000,
    "learning_starts": 500,
    "batch_size": 128,
    "tau": 0.005,
    "train_freq": 1,
    "gradient_steps": 2,
    "target_update_interval": 1,
    "target_entropy": "auto",
    "ent_coef": 0.01,   # 72# best
    "gamma": 0.99,      # 75# best
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

# Total timesteps
TOTAL_TIMESTEPS = 50000  # 25k → 50k


def run_experiment(
    seed: int,
    total_timesteps: int = TOTAL_TIMESTEPS
) -> Dict[str, Any]:
    """Run single 50k steps experiment."""
    
    exp_id = f"50k_seed{seed}"
    
    # 明示的ログ
    logger.warning(f"\n{'='*80}")
    logger.warning(f"🔬 EXPERIMENT: {exp_id}")
    logger.warning(f"{'='*80}")
    logger.warning(f"📊 Effective Configuration [75#/77# Optimized]:")
    logger.warning(f"  gamma: {SAC_OPTIMIZED['gamma']} (75# best)")
    logger.warning(f"  ent_coef: {SAC_OPTIMIZED['ent_coef']} (72# best)")
    logger.warning(f"  reward_scale: {REWARD_CONFIG['reward_scale']}")
    logger.warning(f"  reward_clip: [{REWARD_CONFIG['reward_clip_min']}, {REWARD_CONFIG['reward_clip_max']}]")
    logger.warning(f"  batch_size: {SAC_OPTIMIZED['batch_size']}")
    logger.warning(f"  gradient_steps: {SAC_OPTIMIZED['gradient_steps']}")
    logger.warning(f"  learning_rate: {SAC_OPTIMIZED['learning_rate']}")
    logger.warning(f"  seed: {seed}")
    logger.warning(f"  total_timesteps: {total_timesteps}")
    logger.warning(f"{'='*80}\n")
    
    # UnifiedTrainer config
    config = {
        "training": {
            "algorithm": "SAC",
            "total_timesteps": total_timesteps,
            "eval_freq": 10000,
            "n_eval_episodes": 5,
            "log_interval": 500,
            "seed": seed,
            "sac_hyperparameters": SAC_OPTIMIZED,
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
    
    # Basic metrics
    final_reward = float(training_stats.get("final_reward", 0.0))
    action_dist_raw = training_stats.get("action_distribution", {})
    action_dist = {k: float(v) for k, v in action_dist_raw.items()}
    
    # ROI estimation
    estimated_roi = final_reward * 100
    
    logger.warning(f"\n✅ {exp_id} COMPLETED")
    logger.warning(f"  Final Reward: {final_reward:.6e}")
    logger.warning(f"  Estimated ROI: {estimated_roi:.2f}%")
    logger.warning(f"  HOLD ratio: {action_dist.get('HOLD', 0.0)*100:.1f}%")
    logger.warning(f"  Training time: {elapsed:.1f}s\n")
    
    result = {
        "status": "completed",
        "experiment_id": exp_id,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "final_reward": final_reward,
            "estimated_roi_pct": estimated_roi,
            "action_distribution": action_dist,
            "elapsed_seconds": float(elapsed),
        },
        "config": {
            "sac_gamma": SAC_OPTIMIZED["gamma"],
            "sac_ent_coef": SAC_OPTIMIZED["ent_coef"],
            "sac_batch_size": SAC_OPTIMIZED["batch_size"],
            "sac_gradient_steps": SAC_OPTIMIZED["gradient_steps"],
            "reward_scale": REWARD_CONFIG["reward_scale"],
        }
    }
    
    return to_python(result)


def main():
    """Run 50k steps validation."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.warning("\n" + "="*80)
    logger.warning("🧪 DAY 9 PHASE B: 50k STEPS VALIDATION")
    logger.warning("="*80)
    logger.warning(f"Experiments: {len(SEEDS)} seeds × 50k steps = {len(SEEDS)} runs")
    logger.warning(f"Config: gamma=0.99 (75# best), ent_coef=0.01 (72# best)")
    logger.warning(f"Reference: 75# Day9 25k ROI=-6.29%, 45# Day5 50k ROI=-5.07%")
    logger.warning(f"Output: {OUTPUT_DIR}")
    logger.warning("="*80 + "\n")
    
    all_results = []
    interim_path = Path(OUTPUT_DIR) / "day9b_50k_validation_interim.json"
    
    for seed in SEEDS:
        try:
            result = run_experiment(seed)
            all_results.append(result)
            
            # Save interim results
            with open(interim_path, "w") as f:
                json.dump(all_results, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Exception in seed{seed}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            all_results.append({
                "status": "exception",
                "experiment_id": f"50k_seed{seed}",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    # Aggregate results
    completed = [r for r in all_results if r.get("status") == "completed"]
    
    if completed:
        rois = [r["metrics"]["estimated_roi_pct"] for r in completed]
        hold_ratios = [r["metrics"]["action_distribution"].get("HOLD", 0.0) for r in completed]
        
        summary = {
            "n_seeds": len(completed),
            "roi_mean": float(np.mean(rois)),
            "roi_std": float(np.std(rois, ddof=1)) if len(rois) > 1 else 0.0,
            "roi_min": float(np.min(rois)),
            "roi_max": float(np.max(rois)),
            "hold_ratio_mean": float(np.mean(hold_ratios)),
        }
    else:
        summary = {}
    
    # Interpretation
    interpretation = []
    if summary:
        roi_mean = summary["roi_mean"]
        # 25k比較
        diff_25k = roi_mean - (-6.29)  # 75# Day9 25k result
        if diff_25k > 0:
            interpretation.append(f"✅ 25k比: +{diff_25k:.2f}% 改善")
        else:
            interpretation.append(f"⚠️ 25k比: {diff_25k:.2f}% (改善なし)")
        
        # 45# Day5比較
        diff_45 = roi_mean - (-5.07)  # 45# Day5 50k result
        if diff_45 > 0:
            interpretation.append(f"✅ 45# Day5超え: +{diff_45:.2f}%")
        else:
            interpretation.append(f"⚠️ 45# Day5未達: {diff_45:.2f}%")
        
        # 判定
        if roi_mean > -5.0:
            interpretation.append("🎯 目標達成: ROI > -5%")
        elif roi_mean > -6.0:
            interpretation.append("⚠️ 部分改善: -6% < ROI < -5%")
        else:
            interpretation.append("❌ 改善不足: ROI < -6%")
    
    # Save final analysis
    analysis = {
        "summary": to_python(summary),
        "interpretation": interpretation,
        "comparison": {
            "day9_25k_roi": -6.29,
            "day5_50k_roi": -5.07,
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_timesteps": TOTAL_TIMESTEPS,
            "n_seeds": len(SEEDS),
            "total_experiments": len(all_results),
            "successful": len(completed),
            "sac_gamma": SAC_OPTIMIZED["gamma"],
            "sac_ent_coef": SAC_OPTIMIZED["ent_coef"],
        }
    }
    
    analysis_path = Path(OUTPUT_DIR) / f"day9b_50k_validation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    # Print summary
    logger.warning("\n" + "="*80)
    logger.warning("📊 50k STEPS VALIDATION RESULTS SUMMARY")
    logger.warning("="*80)
    if summary:
        logger.warning(f"\nResults (n={summary['n_seeds']}):")
        logger.warning(f"  ROI: {summary['roi_mean']:.2f}% ± {summary['roi_std']:.2f}%")
        logger.warning(f"  Range: [{summary['roi_min']:.2f}%, {summary['roi_max']:.2f}%]")
        logger.warning(f"  HOLD ratio: {summary['hold_ratio_mean']*100:.1f}%")
    
    logger.warning("\n" + "-"*40)
    logger.warning("COMPARISON:")
    logger.warning(f"  75# Day9 (25k): ROI = -6.29%")
    logger.warning(f"  45# Day5 (50k): ROI = -5.07%")
    if summary:
        logger.warning(f"  This (50k):     ROI = {summary['roi_mean']:.2f}%")
    
    logger.warning("\n" + "-"*40)
    logger.warning("INTERPRETATION:")
    for line in interpretation:
        logger.warning(f"  {line}")
    
    logger.warning(f"\n✅ Analysis saved: {analysis_path}")
    logger.warning("="*80 + "\n")


if __name__ == "__main__":
    main()
