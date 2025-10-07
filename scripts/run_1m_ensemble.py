#!/usr/bin/env python3
"""
1M Ensemble Training Runner

3つのモデルを並列/シーケンシャルで学習:
- Ensemble A: Conservative (ent_coef=0.6)
- Ensemble B: Moderate (ent_coef=0.7)  
- Ensemble C: Aggressive (ent_coef=0.8, reverse=True)

Usage:
    # シーケンシャル実行
    python scripts/run_1m_ensemble.py

    # 並列実行（推奨: 各モデルを別ターミナルで）
    python scripts/run_1m_ensemble.py --model A
    python scripts/run_1m_ensemble.py --model B
    python scripts/run_1m_ensemble.py --model C
"""

import argparse
import subprocess
import sys
from pathlib import Path

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_preflight_check(model_dir: Path, data_path: Path) -> bool:
    """
    Run preflight checks before training.
    
    Returns:
        True if checks pass, False otherwise
    """
    logger.info(f"🔍 Running preflight checks for {model_dir.name}...")
    
    if not model_dir.exists():
        logger.warning(f"⚠️ Model dir doesn't exist yet: {model_dir}")
        return True  # OK for first-time training
    
    cmd = [
        sys.executable,
        "scripts/preflight_schema_scaler_check.py",
        "--model-dir",
        str(model_dir),
        "--strict",
    ]
    
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✅ Preflight checks passed")
            return True
        else:
            logger.error(f"❌ Preflight checks failed:\n{result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ Preflight check error: {e}")
        return False


def run_training(config_path: Path, model_name: str) -> bool:
    """
    Run training for a single ensemble model.
    
    Returns:
        True if training succeeds, False otherwise
    """
    logger.info("=" * 60)
    logger.info(f"🚀 Starting training: {model_name}")
    logger.info(f"   Config: {config_path}")
    logger.info("=" * 60)
    
    cmd = [
        sys.executable,
        "-m",
        "ztb.training.unified_trainer",
        "--config",
        str(config_path),
    ]
    
    try:
        # Run training (will take hours)
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            logger.info(f"✅ {model_name} training completed successfully")
            return True
        else:
            logger.error(f"❌ {model_name} training failed with code {result.returncode}")
            return False
    except KeyboardInterrupt:
        logger.warning(f"⚠️ {model_name} training interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ {model_name} training error: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run 1M ensemble training"
    )
    parser.add_argument(
        "--model",
        choices=["A", "B", "C", "all"],
        default="all",
        help="Which ensemble model to train (default: all sequentially)",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip preflight checks (not recommended)",
    )
    
    args = parser.parse_args()
    
    # Config paths
    configs = {
        "A": Path("configs/train/ensemble_A_1M.json"),
        "B": Path("configs/train/ensemble_B_1M.json"),
        "C": Path("configs/train/ensemble_C_1M.json"),
    }
    
    # Data path
    data_path = Path("ml-dataset-enhanced.csv")
    if not data_path.exists():
        logger.error(f"❌ Data file not found: {data_path}")
        sys.exit(1)
    
    # Determine which models to train
    if args.model == "all":
        models_to_train = ["A", "B", "C"]
    else:
        models_to_train = [args.model]
    
    # Run training for each model
    results = {}
    for model_name in models_to_train:
        config_path = configs[model_name]
        
        if not config_path.exists():
            logger.error(f"❌ Config not found: {config_path}")
            results[model_name] = False
            continue
        
        # Preflight check (optional)
        if not args.skip_preflight:
            model_dir = Path(f"models/ensemble_{model_name}_1M")
            if not run_preflight_check(model_dir, data_path):
                logger.error(f"❌ Preflight failed for {model_name}, skipping")
                results[model_name] = False
                continue
        
        # Run training
        success = run_training(config_path, f"Ensemble {model_name}")
        results[model_name] = success
    
    # Print summary
    logger.info("=" * 60)
    logger.info("📋 Training Summary")
    logger.info("=" * 60)
    
    for model_name, success in results.items():
        status = "✅ COMPLETED" if success else "❌ FAILED"
        logger.info(f"  {status} Ensemble {model_name}")
    
    all_success = all(results.values())
    logger.info("=" * 60)
    
    if all_success:
        logger.info("✅ All ensemble models trained successfully!")
        logger.info("\n📊 Next steps:")
        logger.info("  1. Check TensorBoard logs:")
        logger.info("     tensorboard --logdir logs/ensemble_*_1M")
        logger.info("  2. Run ensemble aggregation:")
        logger.info("     python scripts/ensemble_aggregation.py")
    else:
        logger.error("❌ Some models failed - check logs above")
    
    logger.info("=" * 60)
    
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
