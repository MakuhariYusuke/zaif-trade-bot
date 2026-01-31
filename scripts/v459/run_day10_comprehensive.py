#!/usr/bin/env python3
"""
Day 10: Comprehensive Experiment Suite

79# Codex Review対応 + 宿題事項を全て実施:
1. 45# Day5設定再現（SAC_DEFAULT, 50k）- ベースライン
2. gamma×ent_coef 2×2 実験（50k）
3. batch×grad_steps 2×2 ablation（25k）
4. stage2報酬構造実験（25k）

推定実行時間: ~16時間（無人実行向け）
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import numpy as np
import logging
import copy
import time

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

# ============================================================================
# Constants
# ============================================================================

DATA_PATH = str(project_root / "data" / "btc_jpy_1m_v451_optimized_features.parquet")
OUTPUT_DIR = str(project_root / "results" / "phase4_day10_comprehensive")

# Seeds for experiments (2 for faster iteration, can increase)
SEEDS = [42, 123]


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


# ============================================================================
# SAC Configurations
# ============================================================================

# 45# Day5: SAC_DEFAULT（唯一50kで-5%達成）
SAC_DEFAULT = {
    "learning_rate": 0.0003,
    "buffer_size": 100000,
    "learning_starts": 1000,
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
    "train_freq": 1,
    "gradient_steps": 1,
    "target_update_interval": 1,
    "target_entropy": "auto",
    "ent_coef": "auto",
}

# 78# Day9b: SAC_TUNED（50kで-36%に崩壊）
SAC_TUNED = {
    "learning_rate": 0.0005,
    "buffer_size": 25000,
    "learning_starts": 500,
    "batch_size": 128,
    "gamma": 0.99,
    "tau": 0.005,
    "train_freq": 1,
    "gradient_steps": 2,
    "target_update_interval": 1,
    "target_entropy": "auto",
    "ent_coef": 0.01,
}


# ============================================================================
# Reward Configurations
# ============================================================================

# Simple reward (baseline)
REWARD_SIMPLE = {
    "name": "simple_pnl",
    "description": "Simple PnL with scale=100",
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

# Stage2 Extended reward (from configs/rewards/stage2_extended.yaml)
REWARD_STAGE2 = {
    "name": "stage2_extended",
    "description": "Stage2 with risk-adjusted rewards",
    "curriculum_stage": "stage2",
    "use_simple_reward": False,
    "reward_scale": 100.0,
    "reward_scaling": 100.0,
    "reward_clip_min": -100.0,
    "reward_clip_max": 100.0,
    "profit_weight": 1.0,
    "risk_weight": 0.3,
    "consistency_weight": 0.15,
    "sharpe_bonus_scale": 0.03,
    "sortino_bonus_scale": 0.025,
    "trade_frequency_penalty": 0.002,
    "trade_cooldown_steps": 8,
    "consecutive_loss_penalty_scale": 0.015,
    "max_loss_streak_tracked": 5,
    "hold_penalty": 0.0,
    "dynamic_reward_shaping": True,
    "exploration_bonus_scale": 0.05,
    "exploration_bonus_decay": 0.995,
    "asymmetric_reward_scaling": True,
    "positive_reward_mult": 1.0,
    "negative_reward_mult": 1.5,
}

# No reward scaling (control group)
REWARD_NO_SCALE = {
    "name": "no_scale",
    "description": "No reward scaling (default)",
    "curriculum_stage": "simple",
    "use_simple_reward": True,
    "reward_scale": 1.0,
    "reward_scaling": 1.0,
    "reward_clip_min": -10.0,
    "reward_clip_max": 10.0,
    "profit_weight": 1.0,
    "risk_weight": 0.0,
    "consistency_weight": 0.0,
    "trading_bonus": 0.0,
    "hold_penalty": 0.0,
    "trade_frequency_penalty": 0.0,
    "action_smoothing": 0.0,
}


# ============================================================================
# Experiment Definitions
# ============================================================================

def get_all_experiments() -> List[Dict[str, Any]]:
    """全実験定義を返す"""
    experiments = []
    
    # =========================================================================
    # Category A: Baseline Reproduction (45# Day5 SAC_DEFAULT, 50k)
    # =========================================================================
    for seed in SEEDS:
        experiments.append({
            "category": "A_baseline",
            "name": f"A1_day5_default_50k_seed{seed}",
            "description": "45# Day5 SAC_DEFAULT reproduction at 50k",
            "sac_config": copy.deepcopy(SAC_DEFAULT),
            "reward_config": copy.deepcopy(REWARD_SIMPLE),
            "total_timesteps": 50000,
            "seed": seed,
            "priority": 1,
        })
    
    # =========================================================================
    # Category B: gamma × ent_coef 2×2 (50k)
    # =========================================================================
    gamma_values = [0.95, 0.99]
    ent_coef_values = [0.01, "auto"]
    
    exp_idx = 1
    for gamma in gamma_values:
        for ent_coef in ent_coef_values:
            ent_label = "ent001" if ent_coef == 0.01 else "entauto"
            gamma_label = f"g{int(gamma*100)}"
            
            for seed in SEEDS:
                sac = copy.deepcopy(SAC_DEFAULT)  # Use DEFAULT as base
                sac["gamma"] = gamma
                sac["ent_coef"] = ent_coef
                
                experiments.append({
                    "category": "B_gamma_ent",
                    "name": f"B{exp_idx}_{gamma_label}_{ent_label}_seed{seed}",
                    "description": f"gamma={gamma}, ent_coef={ent_coef}",
                    "sac_config": sac,
                    "reward_config": copy.deepcopy(REWARD_SIMPLE),
                    "total_timesteps": 50000,
                    "seed": seed,
                    "priority": 2,
                })
            exp_idx += 1
    
    # =========================================================================
    # Category C: batch × grad_steps 2×2 (25k, faster ablation)
    # =========================================================================
    batch_values = [128, 256]
    grad_steps_values = [1, 2]
    
    exp_idx = 1
    for batch in batch_values:
        for grad_steps in grad_steps_values:
            for seed in SEEDS:
                sac = copy.deepcopy(SAC_DEFAULT)
                sac["batch_size"] = batch
                sac["gradient_steps"] = grad_steps
                # ent_coef=0.01を使用（72#で最良）
                sac["ent_coef"] = 0.01
                
                experiments.append({
                    "category": "C_batch_grad",
                    "name": f"C{exp_idx}_b{batch}_g{grad_steps}_seed{seed}",
                    "description": f"batch={batch}, grad_steps={grad_steps}",
                    "sac_config": sac,
                    "reward_config": copy.deepcopy(REWARD_SIMPLE),
                    "total_timesteps": 25000,
                    "seed": seed,
                    "priority": 3,
                })
            exp_idx += 1
    
    # =========================================================================
    # Category D: Reward Structure (25k)
    # =========================================================================
    reward_configs = [
        ("D1_simple", REWARD_SIMPLE, "Simple PnL scaled"),
        ("D2_stage2", REWARD_STAGE2, "Stage2 risk-adjusted"),
        ("D3_no_scale", REWARD_NO_SCALE, "No scaling control"),
    ]
    
    for reward_name, reward_config, desc in reward_configs:
        for seed in SEEDS:
            sac = copy.deepcopy(SAC_DEFAULT)
            sac["ent_coef"] = 0.01  # 72#設定
            
            experiments.append({
                "category": "D_reward",
                "name": f"{reward_name}_seed{seed}",
                "description": desc,
                "sac_config": sac,
                "reward_config": copy.deepcopy(reward_config),
                "total_timesteps": 25000,
                "seed": seed,
                "priority": 4,
            })
    
    return experiments


# ============================================================================
# Experiment Runner
# ============================================================================

def run_single_experiment(exp: Dict[str, Any]) -> Dict[str, Any]:
    """単一実験を実行"""
    
    exp_name = exp["name"]
    sac_config = exp["sac_config"]
    reward_config = exp["reward_config"]
    total_timesteps = exp["total_timesteps"]
    seed = exp["seed"]
    
    # 明示的ログ
    logger.warning(f"\n{'='*80}")
    logger.warning(f"🔬 EXPERIMENT: {exp_name}")
    logger.warning(f"   Category: {exp['category']}")
    logger.warning(f"   Description: {exp['description']}")
    logger.warning(f"{'='*80}")
    logger.warning(f"📊 SAC Configuration:")
    logger.warning(f"  gamma: {sac_config['gamma']}")
    logger.warning(f"  ent_coef: {sac_config['ent_coef']}")
    logger.warning(f"  batch_size: {sac_config['batch_size']}")
    logger.warning(f"  gradient_steps: {sac_config['gradient_steps']}")
    logger.warning(f"  buffer_size: {sac_config['buffer_size']}")
    logger.warning(f"  learning_rate: {sac_config['learning_rate']}")
    logger.warning(f"📊 Reward Configuration:")
    logger.warning(f"  name: {reward_config['name']}")
    logger.warning(f"  use_simple_reward: {reward_config['use_simple_reward']}")
    logger.warning(f"  reward_scale: {reward_config['reward_scale']}")
    logger.warning(f"📊 Training:")
    logger.warning(f"  total_timesteps: {total_timesteps}")
    logger.warning(f"  seed: {seed}")
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
                "reward_settings": reward_config,
            },
            "walk_forward": {
                "enabled": False
            }
        },
        "experiment_name": exp_name,
        "output_dir": OUTPUT_DIR
    }
    
    # Train
    start_time = datetime.now()
    trainer = UnifiedTrainer(config)
    success = trainer.run()
    elapsed = (datetime.now() - start_time).total_seconds()
    
    if not success:
        logger.error(f"❌ Experiment failed: {exp_name}")
        return {
            "status": "failed",
            "experiment_name": exp_name,
            "category": exp["category"],
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
        }
    
    # Get training stats
    training_stats = trainer.get_training_stats()
    
    # Basic metrics
    final_reward = float(training_stats.get("final_reward", 0.0))
    action_dist_raw = training_stats.get("action_distribution", {})
    action_dist = {k: float(v) for k, v in action_dist_raw.items()}
    
    # ROI estimation (old method - for comparison)
    estimated_roi_old = final_reward * 100
    
    # Try to get accurate ROI from environment
    final_balance = None
    initial_balance = 100000.0
    accurate_roi = None
    
    try:
        env = None
        if hasattr(trainer, 'model') and hasattr(trainer.model, 'env') and trainer.model.env is not None:
            env = trainer.model.env
        elif hasattr(trainer, 'model') and hasattr(trainer.model, 'get_env'):
            env = trainer.model.get_env()
        elif hasattr(trainer, 'env') and trainer.env is not None:
            env = trainer.env
        
        if env is not None:
            actual_env = env
            if hasattr(env, 'envs') and len(env.envs) > 0:
                actual_env = env.envs[0]
            
            unwrapped_env = actual_env
            for _ in range(5):
                if hasattr(unwrapped_env, 'env'):
                    unwrapped_env = unwrapped_env.env
                else:
                    break
            
            if hasattr(unwrapped_env, 'portfolio_value'):
                final_balance = float(unwrapped_env.portfolio_value)
            if hasattr(unwrapped_env, 'initial_portfolio_value'):
                initial_balance = float(unwrapped_env.initial_portfolio_value)
            
            if final_balance is not None:
                accurate_roi = (final_balance - initial_balance) / initial_balance * 100
    except Exception as e:
        logger.warning(f"Could not get environment metrics: {e}")
    
    # Use accurate ROI if available, otherwise fallback
    roi_to_use = accurate_roi if accurate_roi is not None else estimated_roi_old
    roi_source = "balance" if accurate_roi is not None else "reward"
    
    logger.warning(f"\n✅ {exp_name} COMPLETED")
    logger.warning(f"  Final Reward: {final_reward:.6e}")
    if accurate_roi is not None:
        logger.warning(f"  Accurate ROI (balance): {accurate_roi:.2f}%")
        logger.warning(f"  Final Balance: {final_balance:.2f}")
    logger.warning(f"  Estimated ROI (reward): {estimated_roi_old:.2f}%")
    logger.warning(f"  HOLD ratio: {action_dist.get('HOLD', 0.0)*100:.1f}%")
    logger.warning(f"  Training time: {elapsed:.1f}s\n")
    
    result = {
        "status": "completed",
        "experiment_name": exp_name,
        "category": exp["category"],
        "description": exp["description"],
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "final_reward": final_reward,
            "estimated_roi_reward": estimated_roi_old,
            "accurate_roi_balance": accurate_roi,
            "final_balance": final_balance,
            "roi_used": roi_to_use,
            "roi_source": roi_source,
            "action_distribution": action_dist,
            "elapsed_seconds": elapsed,
        },
        "config": {
            "sac_gamma": sac_config["gamma"],
            "sac_ent_coef": sac_config["ent_coef"],
            "sac_batch_size": sac_config["batch_size"],
            "sac_gradient_steps": sac_config["gradient_steps"],
            "sac_buffer_size": sac_config["buffer_size"],
            "sac_lr": sac_config["learning_rate"],
            "reward_name": reward_config["name"],
            "reward_scale": reward_config["reward_scale"],
            "use_simple_reward": reward_config["use_simple_reward"],
            "total_timesteps": total_timesteps,
        }
    }
    
    return to_python(result)


def aggregate_category_results(results: List[Dict], category: str) -> Dict[str, Any]:
    """カテゴリ別結果を集計"""
    category_results = [r for r in results if r.get("category") == category and r.get("status") == "completed"]
    
    if not category_results:
        return {"n_experiments": 0}
    
    # Group by experiment base name (without seed)
    groups = {}
    for r in category_results:
        name = r["experiment_name"]
        # Remove seed suffix
        base_name = "_".join(name.split("_")[:-1]) if "seed" in name else name
        if base_name not in groups:
            groups[base_name] = []
        groups[base_name].append(r)
    
    summary = {}
    for base_name, group_results in groups.items():
        rois = [r["metrics"]["roi_used"] for r in group_results if r["metrics"]["roi_used"] is not None]
        
        if not rois:
            continue
        
        summary[base_name] = {
            "n_seeds": len(rois),
            "roi_mean": float(np.mean(rois)),
            "roi_std": float(np.std(rois, ddof=1)) if len(rois) > 1 else 0.0,
            "roi_min": float(np.min(rois)),
            "roi_max": float(np.max(rois)),
            "config": group_results[0]["config"],
        }
    
    return {
        "n_experiments": len(category_results),
        "n_groups": len(summary),
        "groups": summary,
    }


def main():
    """Run comprehensive experiment suite."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    experiments = get_all_experiments()
    
    logger.warning("\n" + "="*80)
    logger.warning("🧪 DAY 10: COMPREHENSIVE EXPERIMENT SUITE")
    logger.warning("="*80)
    logger.warning("79# Codex Review対応 + 宿題事項実施")
    logger.warning("")
    logger.warning("Experiments by Category:")
    for cat in ["A_baseline", "B_gamma_ent", "C_batch_grad", "D_reward"]:
        cat_exps = [e for e in experiments if e["category"] == cat]
        logger.warning(f"  {cat}: {len(cat_exps)} experiments")
    logger.warning(f"\nTotal: {len(experiments)} experiments")
    logger.warning(f"Estimated time: ~{len(experiments) * 40 // 60}h {len(experiments) * 40 % 60}min")
    logger.warning(f"Output: {OUTPUT_DIR}")
    logger.warning("="*80 + "\n")
    
    all_results = []
    interim_path = Path(OUTPUT_DIR) / "day10_comprehensive_interim.json"
    
    # Sort by priority
    experiments.sort(key=lambda x: (x["priority"], x["name"]))
    
    for i, exp in enumerate(experiments):
        logger.warning(f"\n[{i+1}/{len(experiments)}] Starting: {exp['name']}")
        
        try:
            result = run_single_experiment(exp)
            all_results.append(result)
            
            # Save interim results after each experiment
            with open(interim_path, "w") as f:
                json.dump(to_python(all_results), f, indent=2)
            
            logger.warning(f"Interim results saved: {interim_path}")
            
        except Exception as e:
            logger.error(f"❌ Exception in {exp['name']}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            all_results.append({
                "status": "exception",
                "experiment_name": exp["name"],
                "category": exp["category"],
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })
    
    # =========================================================================
    # Final Analysis
    # =========================================================================
    
    analysis = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(experiments),
            "completed": sum(1 for r in all_results if r.get("status") == "completed"),
            "failed": sum(1 for r in all_results if r.get("status") in ["failed", "exception"]),
        },
        "category_results": {},
        "interpretation": [],
    }
    
    for cat in ["A_baseline", "B_gamma_ent", "C_batch_grad", "D_reward"]:
        analysis["category_results"][cat] = aggregate_category_results(all_results, cat)
    
    # Interpretation
    interp = analysis["interpretation"]
    
    # A: Baseline
    a_results = analysis["category_results"].get("A_baseline", {}).get("groups", {})
    if a_results:
        for name, stats in a_results.items():
            interp.append(f"[A] {name}: ROI={stats['roi_mean']:.2f}% ± {stats['roi_std']:.2f}%")
    
    # B: gamma×ent_coef - find best
    b_results = analysis["category_results"].get("B_gamma_ent", {}).get("groups", {})
    if b_results:
        best_b = max(b_results.items(), key=lambda x: x[1]["roi_mean"])
        interp.append(f"[B] Best gamma×ent_coef: {best_b[0]} → ROI={best_b[1]['roi_mean']:.2f}%")
        for name, stats in b_results.items():
            interp.append(f"    {name}: {stats['roi_mean']:.2f}% ± {stats['roi_std']:.2f}%")
    
    # C: batch×grad_steps - find best
    c_results = analysis["category_results"].get("C_batch_grad", {}).get("groups", {})
    if c_results:
        best_c = max(c_results.items(), key=lambda x: x[1]["roi_mean"])
        interp.append(f"[C] Best batch×grad_steps: {best_c[0]} → ROI={best_c[1]['roi_mean']:.2f}%")
        for name, stats in c_results.items():
            interp.append(f"    {name}: {stats['roi_mean']:.2f}% ± {stats['roi_std']:.2f}%")
    
    # D: Reward structure - find best
    d_results = analysis["category_results"].get("D_reward", {}).get("groups", {})
    if d_results:
        best_d = max(d_results.items(), key=lambda x: x[1]["roi_mean"])
        interp.append(f"[D] Best reward: {best_d[0]} → ROI={best_d[1]['roi_mean']:.2f}%")
        for name, stats in d_results.items():
            interp.append(f"    {name}: {stats['roi_mean']:.2f}% ± {stats['roi_std']:.2f}%")
    
    # Save final analysis
    analysis_path = Path(OUTPUT_DIR) / f"day10_comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(analysis_path, "w") as f:
        json.dump(to_python(analysis), f, indent=2)
    
    # Save all results
    results_path = Path(OUTPUT_DIR) / f"day10_comprehensive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(to_python(all_results), f, indent=2)
    
    # Print summary
    logger.warning("\n" + "="*80)
    logger.warning("📊 DAY 10 COMPREHENSIVE RESULTS SUMMARY")
    logger.warning("="*80)
    logger.warning(f"Completed: {analysis['metadata']['completed']}/{analysis['metadata']['total_experiments']}")
    logger.warning(f"Failed: {analysis['metadata']['failed']}")
    logger.warning("")
    for line in interp:
        logger.warning(line)
    logger.warning("")
    logger.warning(f"✅ Analysis saved: {analysis_path}")
    logger.warning(f"✅ Results saved: {results_path}")
    logger.warning("="*80 + "\n")


if __name__ == "__main__":
    main()
