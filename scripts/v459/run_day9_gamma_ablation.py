#!/usr/bin/env python3
"""
Day 9: gamma Ablation Study

73# Review対応:
- gamma=0.95 (SAC_TUNED) vs gamma=0.99 (SAC_DEFAULT) の直接比較
- 45# (Day5: -5%@50k) vs 72# (Day8: -26%@25k) の差分原因追求
- ent_coef=0.01 固定（72#で最良と判明）
- 4 seeds for statistical power

仮説:
- SAC_DEFAULTがDay5で-5%、SAC_TUNEDがDay8で-26%
- gamma=0.95→0.99で改善すれば、gammaが主因
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
OUTPUT_DIR = str(project_root / "results" / "phase4_day9_gamma_ablation")

# SAC Base config (ent_coef=0.01 固定: 72#で最良)
SAC_BASE = {
    "learning_rate": 0.0005,
    "buffer_size": 25000,
    "learning_starts": 500,
    "batch_size": 128,
    "tau": 0.005,
    "train_freq": 1,
    "gradient_steps": 2,
    "target_update_interval": 1,
    "target_entropy": "auto",
    "ent_coef": 0.01,  # 72# best
}

# gamma variations (primary ablation target)
GAMMA_VALUES = {
    "gamma_095": 0.95,  # SAC_TUNED (Day8: -26%)
    "gamma_099": 0.99,  # SAC_DEFAULT (Day5: -5%)
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
    gamma_name: str,
    gamma_value: float,
    seed: int,
    total_timesteps: int = 25000
) -> Dict[str, Any]:
    """Run single gamma ablation experiment."""
    
    exp_id = f"{gamma_name}_seed{seed}"
    
    # SAC config with gamma
    sac_config = {**SAC_BASE, "gamma": gamma_value}
    
    # 明示的ログ（73# Review対応）
    logger.warning(f"\n{'='*80}")
    logger.warning(f"🔬 EXPERIMENT: {exp_id}")
    logger.warning(f"{'='*80}")
    logger.warning(f"📊 Effective Configuration [73# Review]:")
    logger.warning(f"  gamma: {gamma_value} ← PRIMARY ABLATION TARGET")
    logger.warning(f"  ent_coef: {sac_config['ent_coef']} (fixed: 72# best)")
    logger.warning(f"  reward_scale: {REWARD_CONFIG['reward_scale']}")
    logger.warning(f"  reward_clip: [{REWARD_CONFIG['reward_clip_min']}, {REWARD_CONFIG['reward_clip_max']}]")
    logger.warning(f"  batch_size: {sac_config['batch_size']}")
    logger.warning(f"  gradient_steps: {sac_config['gradient_steps']}")
    logger.warning(f"  learning_rate: {sac_config['learning_rate']}")
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
            "sac_hyperparameters": sac_config,
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
        "gamma_name": gamma_name,
        "gamma_value": gamma_value,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "final_reward": final_reward,
            "estimated_roi_pct": estimated_roi,
            "action_distribution": action_dist,
            "elapsed_seconds": float(elapsed),
        },
        "config": {
            "sac_gamma": gamma_value,
            "sac_ent_coef": sac_config["ent_coef"],
            "sac_batch_size": sac_config["batch_size"],
            "sac_gradient_steps": sac_config["gradient_steps"],
            "reward_scale": REWARD_CONFIG["reward_scale"],
        }
    }
    
    return to_python(result)


def main():
    """Run gamma ablation study."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.warning("\n" + "="*80)
    logger.warning("🧪 DAY 9: gamma ABLATION STUDY")
    logger.warning("="*80)
    logger.warning(f"Experiments: {len(GAMMA_VALUES)} gamma values × {len(SEEDS)} seeds = {len(GAMMA_VALUES) * len(SEEDS)} runs")
    logger.warning(f"Hypothesis: gamma=0.99 (DEFAULT) may outperform gamma=0.95 (TUNED)")
    logger.warning(f"Reference: 45# Day5 -5% (DEFAULT) vs 72# Day8 -26% (TUNED)")
    logger.warning(f"Output: {OUTPUT_DIR}")
    logger.warning("="*80 + "\n")
    
    all_results = []
    interim_path = Path(OUTPUT_DIR) / "day9_gamma_ablation_interim.json"
    
    for gamma_name, gamma_val in GAMMA_VALUES.items():
        for seed in SEEDS:
            try:
                result = run_experiment(gamma_name, gamma_val, seed)
                all_results.append(result)
                
                # Save interim results
                with open(interim_path, "w") as f:
                    json.dump(all_results, f, indent=2)
                    
            except Exception as e:
                logger.error(f"❌ Exception in {gamma_name}_seed{seed}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                all_results.append({
                    "status": "exception",
                    "experiment_id": f"{gamma_name}_seed{seed}",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
    
    # Aggregate results
    summary = {}
    for gamma_name in GAMMA_VALUES.keys():
        gamma_results = [r for r in all_results if r.get("gamma_name") == gamma_name and r.get("status") == "completed"]
        
        if not gamma_results:
            continue
        
        rois = [r["metrics"]["estimated_roi_pct"] for r in gamma_results]
        hold_ratios = [r["metrics"]["action_distribution"].get("HOLD", 0.0) for r in gamma_results]
        
        summary[gamma_name] = {
            "n_seeds": len(gamma_results),
            "roi_mean": float(np.mean(rois)),
            "roi_std": float(np.std(rois, ddof=1)) if len(rois) > 1 else 0.0,
            "roi_min": float(np.min(rois)),
            "roi_max": float(np.max(rois)),
            "hold_ratio_mean": float(np.mean(hold_ratios)),
        }
    
    # Analysis interpretation
    interpretation = []
    if "gamma_095" in summary and "gamma_099" in summary:
        diff = summary["gamma_099"]["roi_mean"] - summary["gamma_095"]["roi_mean"]
        if diff > 5:
            interpretation.append(f"✅ gamma=0.99が+{diff:.1f}%優位 → gamma=0.99へ変更推奨")
            interpretation.append("→ SAC_TUNEDの劣化はgamma=0.95が主因")
        elif diff < -5:
            interpretation.append(f"❌ gamma=0.95が{-diff:.1f}%優位 → gamma=0.95維持")
            interpretation.append("→ 劣化原因は他のパラメータ")
        else:
            interpretation.append(f"⚠️ gamma差分 {diff:.1f}% は微小 → 他要因も検討必要")
    
    # Save final analysis
    analysis = {
        "summary": to_python(summary),
        "interpretation": interpretation,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_gamma_values": len(GAMMA_VALUES),
            "n_seeds": len(SEEDS),
            "total_experiments": len(all_results),
            "successful": sum(1 for r in all_results if r.get("status") == "completed"),
            "fixed_ent_coef": SAC_BASE["ent_coef"],
        }
    }
    
    analysis_path = Path(OUTPUT_DIR) / f"day9_gamma_ablation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    # Print summary
    logger.warning("\n" + "="*80)
    logger.warning("📊 GAMMA ABLATION STUDY RESULTS SUMMARY")
    logger.warning("="*80)
    for gamma_name, stats in summary.items():
        logger.warning(f"\n{gamma_name}:")
        logger.warning(f"  ROI: {stats['roi_mean']:.2f}% ± {stats['roi_std']:.2f}%")
        logger.warning(f"  Range: [{stats['roi_min']:.2f}%, {stats['roi_max']:.2f}%]")
        logger.warning(f"  HOLD ratio: {stats['hold_ratio_mean']*100:.1f}%")
    
    logger.warning("\n" + "-"*40)
    logger.warning("INTERPRETATION:")
    for line in interpretation:
        logger.warning(f"  {line}")
    
    logger.warning(f"\n✅ Analysis saved: {analysis_path}")
    logger.warning("="*80 + "\n")


if __name__ == "__main__":
    main()
