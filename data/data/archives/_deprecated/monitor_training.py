#!/usr/bin/env python3
"""
Monitor training progress from checkpoint files.
"""

import os
import json
from pathlib import Path

def monitor_training() -> None:
    """Monitor training progress."""
    checkpoint_dir = Path("./models/optimized_checkpoints")
    
    if not checkpoint_dir.exists():
        print("❌ Checkpoint directory not found")
        return
    
    # Find checkpoint files
    checkpoints = sorted(checkpoint_dir.glob("optimized_ppo_*_steps.zip"))
    
    print("=" * 60)
    print("📊 Training Progress Monitor")
    print("=" * 60)
    
    if not checkpoints:
        print("\n⏳ No checkpoints found yet...")
        print("Training may still be in progress")
    else:
        print(f"\n✅ Found {len(checkpoints)} checkpoint(s):\n")
        for cp in checkpoints:
            # Extract step number from filename
            step = cp.stem.split('_')[-2]
            size_mb = cp.stat().st_size / (1024 * 1024)
            print(f"  📁 {cp.name}")
            print(f"     Steps: {step:>10s} | Size: {size_mb:.2f} MB")
    
    # Check for final model
    final_model = Path("./models/optimized_final.zip")
    if final_model.exists():
        print("\n🎉 TRAINING COMPLETE!")
        print(f"   Final model: {final_model}")
        print(f"   Size: {final_model.stat().st_size / (1024 * 1024):.2f} MB")
    else:
        print("\n⏳ Training still in progress...")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    monitor_training()
