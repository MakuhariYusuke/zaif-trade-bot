#!/usr/bin/env python3
"""Quick validation test for reward_components persistence."""
import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ztb.training.unified_trainer.main import main as unified_trainer_main


def create_minimal_config(output_path: str) -> str:
    """Create minimal config for quick validation."""
    config = {
        "version": "test",
        "algorithm": "sac",
        "model_name": "reward_components_test",
        "description": "Quick test for reward_components persistence",
        "training": {
            "total_timesteps": 500,
            "algorithm": "SAC",
            "sac_hyperparameters": {
                "learning_rate": 0.0003,
                "buffer_size": 10000,
                "learning_starts": 100,
                "batch_size": 64,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "ent_coef": 0.01
            },
            "data_config": {
                "data_path": "data/btc_jpy_5m_dataset.csv",
                "validation_split": 0.2,
                "test_split": 0.1,
                "use_real_data": True
            },
            "features": {
                "feature_set": "minimal",
                "skip_quality_filtering": True
            }
        },
        "environment": {
            "env_type": "heavy",
            "initial_balance": 100000,
            "max_position_size": 0.5,
            "trading_fee_percent": 0.001
        },
        "reward": {
            "base_reward_type": "pnl",
            "behavioral_penalties": {
                "balance_penalty": 0.03,
                "skewness_penalty": 0.01,
                "balance_shaping": 0.1,
                "entropy_shaping": 0.05
            }
        },
        "evaluation": {
            "eval_freq": 200,
            "n_eval_episodes": 1
        },
        "logging": {
            "log_freq": 100,
            "verbose": 1
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return output_path


def check_training_report(report_dir: Path) -> bool:
    """Check if training reports contain reward_components."""
    reports = list(report_dir.glob("training_report_*.json"))
    
    if not reports:
        print("❌ No training reports found")
        return False
    
    # Check most recent report
    latest_report = max(reports, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(latest_report, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if "reward_components" in data:
            print(f"✅ reward_components found in {latest_report.name}")
            print("\nComponents:")
            for key, value in data["reward_components"].items():
                print(f"  {key}: {value:.6f}")
            return True
        else:
            print(f"❌ reward_components NOT found in {latest_report.name}")
            print("Report keys:", list(data.keys()))
            return False
            
    except Exception as e:
        print(f"❌ Error reading report: {e}")
        return False


if __name__ == "__main__":
    print("Creating minimal test config...")
    fd, config_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    Path(config_path).unlink(missing_ok=True)
    
    try:
        config_path = create_minimal_config(config_path)
        print(f"Config created: {config_path}")
        
        print("\nRunning quick training (500 steps)...")
        # Run training with minimal steps
        sys.argv = [
            "unified_trainer_main",
            "--config", config_path,
            "--timesteps", "500"
        ]
        
        unified_trainer_main()
        
        print("\n" + "="*60)
        print("Checking training report...")
        print("="*60)
        
        # Check report
        reports_dir = Path("reports")
        success = check_training_report(reports_dir)
        
        if success:
            print("\n✅ reward_components persistence validated!")
            sys.exit(0)
        else:
            print("\n❌ reward_components persistence validation failed")
            sys.exit(1)
            
    finally:
        # Cleanup
        if Path(config_path).exists():
            Path(config_path).unlink()
