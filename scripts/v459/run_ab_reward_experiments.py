#!/usr/bin/env python3
"""
Phase 3 Day 4-5: Reward Design AB Experiments Runner
48実験実行: 4 seeds × 4 windows × 3 reward stages

Usage:
    python scripts/v459/run_ab_reward_experiments.py [--parallel 4] [--resume]
"""

import argparse
import copy
import os
import json
import sys
import faulthandler
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import time

# Enable faulthandler for debugging crashes
faulthandler.enable()

# Project root setup
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Avoid heavy optional imports during training runs on unstable environments
os.environ.setdefault("SKIP_HEAVY_IMPORTS", "1")
os.environ.setdefault("ZTB_SKIP_SCIPY", "1")
os.environ.setdefault("ZTB_SKIP_SKLEARN", "1")
os.environ.setdefault("ZTB_SAFE_DATETIME", "1")
os.environ.setdefault(
    "ZTB_SIGINT_POLICY", "ignore" if os.name == "nt" else "default"
)
# Limit native thread usage to reduce Windows contention between runs
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from ztb.training.unified_trainer.trainer import UnifiedTrainer
from ztb.training.reward_config_schema import load_reward_config
from ztb.utils.parallel_experiments import run_parallel_experiments, ExperimentResult
from ztb.utils.signal_utils import configure_signal_handling

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("="*70)
logger.info("🚀 AB REWARD EXPERIMENTS SCRIPT STARTING")
logger.info("="*70)
configure_signal_handling(os.environ.get("ZTB_SIGINT_POLICY", "default"), logger)


def _limit_torch_threads() -> None:
    """Limit PyTorch thread usage to avoid Windows multiprocessing issues"""
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        logger.info("✅ PyTorch threads limited to 1")
    except Exception as exc:
        logger.debug("Torch thread limiting skipped: %s", exc)


# Call thread limiting immediately
_limit_torch_threads()

# Experiment Configuration
SEEDS = [42, 123, 456, 789]
REWARD_STAGES = ["stage1_basic", "stage2_extended", "stage3_advanced"]
WINDOWS = 4  # Walk-Forward windows

# Base Configuration Template
BASE_CONFIG = {
    "training": {
        "algorithm": "SAC",
        "total_timesteps": 5000,  # テスト用に削減
        "eval_freq": 5000,
        "n_eval_episodes": 3,
        "log_interval": 100,
        "sac_hyperparameters": {
            "learning_rate": 0.0003,
            "buffer_size": 10000,  # さらに削減
            "learning_starts": 100,  # 早期開始
            "batch_size": 64,  # さらに削減
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": "auto",
            "target_update_interval": 1,
            "target_entropy": "auto"
        },
        "data_config": {
            "data_path": str(project_root / "data" / "btc_jpy_1m_v451.csv"),
            "window_size": 60
        },
        "environment": {
            "use_continuous_actions": True,  # SAC requires continuous action space
            "action_space_type": "continuous",
            "initial_portfolio_value": 1000000.0,
            "transaction_cost": 0.0
        },
        "walk_forward": {
            "enabled": True,
            "n_splits": 4,
            "train_size": 0.6,
            "validation_size": 0.2,
            "test_size": 0.2
        }
    }
}


class ABRewardExperiment:
    """Single AB Reward Experiment"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiment_name = config["experiment_name"]
        self.logger = logging.getLogger(f"ABRewardExperiment.{self.experiment_name}")
        
        # v459最適化: 特徴生成済みParquetを優先使用
        self._setup_optimized_data_path()
    
    def _setup_optimized_data_path(self):
        """特徴生成済みParquetファイルが存在すれば優先使用"""
        from pathlib import Path
        
        # data_pathはtraining.data_config.data_pathに格納
        current_data_path = self.config.get("training", {}).get("data_config", {}).get("data_path", "")
        
        # 元のCSVパスからParquetパスを生成
        csv_path = Path(current_data_path)
        
        if csv_path.suffix == '.csv':
            # btc_jpy_1m_v451.csv → btc_jpy_1m_v451_optimized_features.parquet
            parquet_path = csv_path.parent / f"{csv_path.stem}_optimized_features.parquet"
            
            if parquet_path.exists():
                # data_pathを上書き
                if "training" not in self.config:
                    self.config["training"] = {}
                if "data_config" not in self.config["training"]:
                    self.config["training"]["data_config"] = {}
                self.config["training"]["data_config"]["data_path"] = str(parquet_path)
                
                # 特徴生成をスキップ（既にParquetに含まれている）
                # Parquetファイルには OHLCV + 特徴量が含まれているため、
                # 追加の特徴生成は不要
                if "environment" not in self.config["training"]:
                    self.config["training"]["environment"] = {}
                # use_precomputed_features フラグを追加
                self.config["training"]["environment"]["use_precomputed_features"] = True
                # minimal feature set (OHLCV以外の列を特徴として使用)
                self.config["training"]["environment"]["feature_set"] = "minimal"
                
                logger.info(f"✅ Using precomputed features: {parquet_path.name}")
                logger.info(f"   Feature generation will be skipped")
            else:
                logger.info(f"⚠️ No precomputed features found, will use CSV: {csv_path.name}")
                logger.info(f"   Run: python scripts/v459/precompute_optimized_features.py")
    
    def execute(self) -> ExperimentResult:
        """Execute single experiment"""
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        
        # Memory cleanup before experiment
        import gc
        gc.collect()
        
        try:
            start_time = time.time()
            
            # Create trainer
            self.logger.info("Creating trainer...")
            trainer = UnifiedTrainer(self.config)
            
            # Run training
            self.logger.info("Running training...")
            success = trainer.run()
            
            elapsed = time.time() - start_time
            self.logger.info(f"Training completed, success={success}, elapsed={elapsed:.1f}s")
            
            if not success:
                raise ValueError("Training failed (success=False)")
            
            # Get training report
            self.logger.info("Getting training report...")
            report = trainer.get_training_report()
            self.logger.info(f"Training report received, type={type(report)}")
            
            # Extract metrics
            self.logger.info("Extracting metrics...")
            metrics = self._extract_metrics_from_report(report)
            metrics["elapsed_seconds"] = elapsed
            self.logger.info(f"Metrics extracted: {list(metrics.keys())}")
            
            self.logger.info(f"Completed: {self.experiment_name} ({elapsed:.1f}s)")
            self.logger.info(f"  Test ROI: {metrics.get('test_roi', 0):.2%}")
            self.logger.info(f"  Sharpe: {metrics.get('test_sharpe', 0):.2f}")
            
            # Memory cleanup after training
            self.logger.info("Cleaning up...")
            del trainer
            gc.collect()
            
            self.logger.info("Creating ExperimentResult...")
            result = ExperimentResult(
                experiment_name=self.experiment_name,
                timestamp=datetime.now().isoformat(),
                status="completed",
                config=self.config,
                metrics=metrics,
                artifacts={"report": report}
            )
            self.logger.info("ExperimentResult created successfully")
            
            return result
            
        except Exception as e:
            import traceback
            self.logger.error(f"Failed: {self.experiment_name} - {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return ExperimentResult(
                experiment_name=self.experiment_name,
                timestamp=datetime.now().isoformat(),
                status="failed",
                config=self.config,
                metrics={"error": str(e)},
                artifacts={}
            )
    
    def _extract_metrics_from_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from training report"""
        # Training stats from report
        training_stats = report.get("training_stats", {})
        
        # Walk-Forward結果を取得
        wf_results = training_stats.get("walk_forward", {})
        
        if wf_results and "splits" in wf_results:
            # 最終split（window 4）のtest結果
            splits = wf_results["splits"]
            if splits:
                final_split = splits[-1]
                test_metrics = final_split.get("test", {})
                val_metrics = final_split.get("validation", {})
                
                # Overfitting計算
                val_roi = val_metrics.get("roi", 0)
                test_roi = test_metrics.get("roi", 0)
                overfitting_ratio = (
                    abs(val_roi - test_roi) / abs(val_roi)
                    if val_roi != 0 else 0
                )
                
                return {
                    "test_roi": test_roi,
                    "test_sharpe": test_metrics.get("sharpe_ratio", 0),
                    "test_max_drawdown": test_metrics.get("max_drawdown", 0),
                    "test_win_rate": test_metrics.get("win_rate", 0),
                    "val_roi": val_roi,
                    "val_sharpe": val_metrics.get("sharpe_ratio", 0),
                    "overfitting_ratio": overfitting_ratio,
                    "avg_roi": wf_results.get("avg_roi", 0),
                    "avg_sharpe": wf_results.get("avg_sharpe", 0),
                    "consistency_score": self._calculate_consistency(wf_results)
                }
        
        # Fallback: 単一結果の場合
        return {
            "test_roi": training_stats.get("final_reward", 0),
            "test_sharpe": 0,
            "test_max_drawdown": 0,
            "test_win_rate": 0,
            "val_roi": 0,
            "val_sharpe": 0,
            "overfitting_ratio": 0,
            "avg_roi": training_stats.get("final_reward", 0),
            "avg_sharpe": 0,
            "consistency_score": 0
        }
    
    def _calculate_consistency(self, wf_results: Dict[str, Any]) -> float:
        """Calculate consistency score across windows"""
        if "splits" not in wf_results:
            return 0.0
        
        test_rois = [
            split["test"]["roi"]
            for split in wf_results["splits"]
            if "test" in split and "roi" in split["test"]
        ]
        
        if len(test_rois) < 2:
            return 0.0
        
        # 標準偏差の逆数（低いほど安定）
        import numpy as np
        std = np.std(test_rois)
        mean = np.mean(test_rois)
        
        if std == 0:
            return 1.0
        
        # Coefficient of Variation の逆数
        cv = std / abs(mean) if mean != 0 else float('inf')
        consistency = 1.0 / (1.0 + cv)
        
        return consistency


def generate_experiment_configs() -> List[Dict[str, Any]]:
    """Generate all 48 experiment configurations"""
    configs = []
    
    for reward_stage in REWARD_STAGES:
        for seed in SEEDS:
            # Load reward config
            reward_config_path = project_root / f"configs/rewards/{reward_stage}.yaml"
            from ztb.training.reward_config_schema import RewardConfigSchema
            reward_dict = RewardConfigSchema.load_and_validate(reward_config_path)
            behavior_opt = reward_dict.pop("behavior_optimization", None)
            
            # Create experiment config with deep copy to prevent state contamination
            config = copy.deepcopy(BASE_CONFIG)
            config["seed"] = seed
            # Inject reward into environment for EnvironmentConfig.from_dict compatibility
            if "environment" not in config["training"]:
                config["training"]["environment"] = {}
            config["training"]["environment"]["reward_settings"] = reward_dict
            if behavior_opt:
                config["training"]["environment"]["behavior_optimization"] = behavior_opt
            
            # Experiment naming
            exp_name = f"reward_{reward_stage}_seed{seed}"
            config["experiment_name"] = exp_name
            config["training"]["model_name"] = f"ab_reward_{reward_stage}_s{seed}"
            config["training"]["output_dir"] = str(
                project_root / f"results/ab_rewards/{reward_stage}/seed_{seed}"
            )
            
            configs.append(config)
            
            logger.info(f"Generated config: {exp_name}")
    
    logger.info(f"Total {len(configs)} experiments generated")
    return configs


def save_results(results: List[ExperimentResult], output_path: Path):
    """Save experiment results"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    results_data = []
    for result in results:
        # ConfigからRewardSettingsを除外（JSON非シリアル化不可）
        config_copy = result.config.copy()
        training_copy = config_copy.get("training")
        if isinstance(training_copy, dict):
            config_copy["training"] = training_copy.copy()
            if "reward_settings" in config_copy["training"]:
                del config_copy["training"]["reward_settings"]
        
        results_data.append({
            "experiment_name": result.experiment_name,
            "timestamp": result.timestamp,
            "status": result.status,
            "config": config_copy,
            "metrics": result.metrics,
            "artifacts": result.artifacts
        })
    
    # Convert numpy/torch types to native Python types for JSON serialization
    def convert_to_native(obj):
        """Convert numpy/torch/Path/datetime types to native Python types"""
        import numpy as np
        from pathlib import Path
        from datetime import datetime, date
        
        # Handle None
        if obj is None:
            return None
        # Handle Path objects
        elif isinstance(obj, Path):
            return str(obj)
        # Handle datetime objects
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        # Handle numpy types
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Handle collections
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, set):
            return [convert_to_native(item) for item in obj]
        # Handle custom objects with __dict__
        elif hasattr(obj, '__dict__'):
            return convert_to_native(obj.__dict__)
        # Handle primitive types (str, int, float, bool)
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            # For any other type, try to convert to string
            try:
                return str(obj)
            except Exception as e:
                logger.warning(f"Could not convert object of type {type(obj).__name__}: {e}")
                return f"<non-serializable: {type(obj).__name__}>"
    
    results_data = convert_to_native(results_data)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")


def analyze_results(results: List[ExperimentResult]) -> Dict[str, Any]:
    """Quick analysis of experiment results"""
    import numpy as np
    
    # Group by stage
    stage_results = {stage: [] for stage in REWARD_STAGES}
    
    for result in results:
        if result.status != "completed":
            continue
        
        exp_name = result.experiment_name
        stage = next(s for s in REWARD_STAGES if s in exp_name)
        stage_results[stage].append(result.metrics)
    
    # Calculate statistics
    summary = {}
    for stage, metrics_list in stage_results.items():
        if not metrics_list:
            continue
        
        test_rois = [m["test_roi"] for m in metrics_list]
        test_sharpes = [m["test_sharpe"] for m in metrics_list]
        consistency_scores = [m["consistency_score"] for m in metrics_list]
        
        summary[stage] = {
            "n_samples": len(metrics_list),
            "test_roi": {
                "mean": np.mean(test_rois),
                "std": np.std(test_rois),
                "min": np.min(test_rois),
                "max": np.max(test_rois)
            },
            "test_sharpe": {
                "mean": np.mean(test_sharpes),
                "std": np.std(test_sharpes),
                "min": np.min(test_sharpes),
                "max": np.max(test_sharpes)
            },
            "consistency": {
                "mean": np.mean(consistency_scores),
                "std": np.std(consistency_scores)
            }
        }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 AB Reward Experiments")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel workers (Windows: use 1)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--output", type=str, default="results/ab_rewards/experiment_results.json", help="Output path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of experiments (for testing)")
    args = parser.parse_args()
    _limit_torch_threads()
    
    logger.info("="*70)
    logger.info("Phase 3 Day 4-5: AB Reward Experiments")
    logger.info("="*70)
    logger.info(f"Total Experiments: {len(SEEDS) * len(REWARD_STAGES)} (12 = 4 seeds × 3 stages)")
    logger.info(f"Seeds: {SEEDS}")
    logger.info(f"Reward Stages: {REWARD_STAGES}")
    logger.info(f"Windows: {WINDOWS} (Walk-Forward)")
    logger.info(f"Execution Mode: Sequential (Windows compatible)")
    logger.info("="*70)
    
    # Generate experiment configs
    configs = generate_experiment_configs()
    
    if args.limit:
        configs = configs[:args.limit]
        logger.info(f"Limited to {args.limit} experiments for testing")
    
    # Check for existing results (resume support)
    output_path = Path(args.output)
    if args.resume and output_path.exists():
        logger.info(f"Resuming from checkpoint: {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
        completed_names = {r["experiment_name"] for r in existing_results if r["status"] == "completed"}
        configs = [c for c in configs if c["experiment_name"] not in completed_names]
        logger.info(f"Remaining experiments: {len(configs)}")
    
    if not configs:
        logger.info("No experiments to run (all completed)")
        return
    
    # Run experiments sequentially (Windows-compatible)
    start_time = time.time()
    results = []
    
    # Progress tracking
    stage_progress = {stage: {"completed": 0, "total": len(SEEDS)} for stage in REWARD_STAGES}
    
    for i, config in enumerate(configs, 1):
        # Stage info
        exp_name = config["experiment_name"]
        current_stage = next(s for s in REWARD_STAGES if s in exp_name)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 Experiment {i}/{len(configs)}: {exp_name}")
        logger.info(f"   Stage: {current_stage} [{stage_progress[current_stage]['completed']+1}/{stage_progress[current_stage]['total']}]")
        logger.info(f"   Seed: {config['seed']}")
        logger.info(f"{'='*70}")
        
        experiment = ABRewardExperiment(config)
        result = experiment.execute()
        results.append(result)
        
        # Update progress
        stage_progress[current_stage]["completed"] += 1
        
        # Show immediate result - with error handling
        try:
            if result.status == "completed":
                metrics = result.metrics
                logger.info(f"\n✅ Success!")
                logger.info(f"   Test ROI: {metrics.get('test_roi', 0):.2%}")
                logger.info(f"   Sharpe: {metrics.get('test_sharpe', 0):.2f}")
                logger.info(f"   Consistency: {metrics.get('consistency_score', 0):.3f}")
                logger.info(f"   Elapsed: {metrics.get('elapsed_seconds', 0):.1f}s")
            else:
                logger.error(f"\n❌ Failed: {result.metrics.get('error', 'Unknown')}")
        except Exception as e:
            logger.error(f"Error displaying result metrics: {e}")
            logger.error(f"Result status: {result.status}")
            logger.error(f"Result metrics type: {type(result.metrics)}")
            if hasattr(result.metrics, '__dict__'):
                logger.error(f"Result metrics content: {result.metrics.__dict__}")
            else:
                logger.error(f"Result metrics content: {result.metrics}")
        
        # Progress summary
        completed = len([r for r in results if r.status == "completed"])
        failed = len([r for r in results if r.status == "failed"])
        remaining = len(configs) - i
        avg_time = (time.time() - start_time) / i
        eta = remaining * avg_time / 60
        
        logger.info(f"\n📈 Progress: {i}/{len(configs)} ({i/len(configs)*100:.1f}%)")
        logger.info(f"   ✅ {completed} completed | ❌ {failed} failed | ⏳ {remaining} remaining")
        logger.info(f"   ETA: {eta:.1f} minutes")
        
        # Stage-wise summary
        logger.info(f"\n📊 Stage Progress:")
        for stage, prog in stage_progress.items():
            pct = prog["completed"] / prog["total"] * 100 if prog["total"] > 0 else 0
            bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
            logger.info(f"   {stage:20s} [{bar}] {prog['completed']}/{prog['total']} ({pct:.0f}%)")
        
        # Periodic checkpoint
        if i % 3 == 0:
            save_results(results, output_path)
            logger.info(f"\n💾 Checkpoint saved ({i}/{len(configs)} completed)")
    
    elapsed = time.time() - start_time
    logger.info(f"\nAll experiments completed in {elapsed/60:.1f} minutes")
    
    # Save final results
    try:
        logger.info("\n💾 Attempting to save results...")
        save_results(results, output_path)
        logger.info(f"✅ Results successfully saved to {output_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save results: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        # Try to identify which object is causing the issue
        logger.error("\n🔍 Debugging serialization issue:")
        for i, result in enumerate(results):
            logger.error(f"\nResult {i+1}/{len(results)}:")
            logger.error(f"  experiment_name: {type(result.experiment_name).__name__}")
            logger.error(f"  timestamp: {type(result.timestamp).__name__}")
            logger.error(f"  status: {type(result.status).__name__}")
            logger.error(f"  config: {type(result.config).__name__}")
            logger.error(f"  metrics type: {type(result.metrics).__name__}")
            if result.metrics:
                for k, v in list(result.metrics.items())[:5]:  # Show first 5
                    logger.error(f"    {k}: {type(v).__name__}")
            logger.error(f"  artifacts type: {type(result.artifacts).__name__}")
            if result.artifacts:
                for k, v in list(result.artifacts.items())[:5]:  # Show first 5
                    logger.error(f"    {k}: {type(v).__name__}")
    
    # Quick analysis
    summary = analyze_results(results)
    
    logger.info("\n" + "="*80)
    logger.info("📊 FINAL RESULTS SUMMARY")
    logger.info("="*80)
    
    # Overall stats
    completed = len([r for r in results if r.status == "completed"])
    failed = len([r for r in results if r.status == "failed"])
    total = len(results)
    
    logger.info(f"\n🎯 Execution Summary:")
    logger.info(f"   Total Experiments: {total}")
    logger.info(f"   ✅ Completed: {completed} ({completed/total*100:.1f}%)")
    logger.info(f"   ❌ Failed: {failed} ({failed/total*100:.1f}%)")
    logger.info(f"   ⏱️  Total Time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    logger.info(f"   ⚡ Avg Time/Experiment: {elapsed/total/60:.1f} minutes")
    
    # Stage-wise comparison
    logger.info(f"\n📈 Stage-wise Performance Comparison:")
    logger.info("="*80)
    
    # Table header
    logger.info(f"{'Stage':<25} {'N':<5} {'ROI Mean':<12} {'Sharpe Mean':<12} {'Consistency':<12}")
    logger.info("-"*80)
    
    for stage in REWARD_STAGES:
        if stage not in summary:
            continue
        stats = summary[stage]
        roi_mean = stats['test_roi']['mean']
        roi_std = stats['test_roi']['std']
        sharpe_mean = stats['test_sharpe']['mean']
        sharpe_std = stats['test_sharpe']['std']
        consistency = stats['consistency']['mean']
        
        logger.info(
            f"{stage:<25} "
            f"{stats['n_samples']:<5} "
            f"{roi_mean:>6.2%} ± {roi_std:4.2%}  "
            f"{sharpe_mean:>6.2f} ± {sharpe_std:4.2f}  "
            f"{consistency:>8.3f}"
        )
    
    # Detailed statistics
    logger.info(f"\n📋 Detailed Statistics:")
    logger.info("="*80)
    for stage, stats in summary.items():
        logger.info(f"\n🔹 {stage}:")
        logger.info(f"   Sample Size: {stats['n_samples']}")
        logger.info(f"   Test ROI:")
        logger.info(f"      Mean: {stats['test_roi']['mean']:>8.2%}")
        logger.info(f"      Std:  {stats['test_roi']['std']:>8.2%}")
        logger.info(f"      Min:  {stats['test_roi']['min']:>8.2%}")
        logger.info(f"      Max:  {stats['test_roi']['max']:>8.2%}")
        logger.info(f"   Sharpe Ratio:")
        logger.info(f"      Mean: {stats['test_sharpe']['mean']:>8.2f}")
        logger.info(f"      Std:  {stats['test_sharpe']['std']:>8.2f}")
        logger.info(f"      Min:  {stats['test_sharpe']['min']:>8.2f}")
        logger.info(f"      Max:  {stats['test_sharpe']['max']:>8.2f}")
        logger.info(f"   Consistency Score:")
        logger.info(f"      Mean: {stats['consistency']['mean']:>8.3f}")
        logger.info(f"      Std:  {stats['consistency']['std']:>8.3f}")
    
    logger.info("\n" + "="*80)
    logger.info("🎉 All experiments completed!")
    logger.info("="*80)
    logger.info(f"📁 Results saved to: {output_path}")
    logger.info("\n📊 Next Steps:")
    logger.info("   1. Statistical Analysis:")
    logger.info("      python scripts/v459/analyze_ab_reward_results.py")
    logger.info("   2. Visualization:")
    logger.info("      python scripts/v459/plot_ab_comparison.py")
    logger.info("="*80)


logger.info("🔍 Script loaded, __name__ = " + __name__)

if __name__ == "__main__":
    logger.info("🎯 Entering main()")
    main()
    logger.info("✅ main() completed")
