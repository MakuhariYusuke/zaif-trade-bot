#!/usr/bin/env python3
"""
Test script for Step 3: Hierarchical checkpoint system
"""

import tempfile
from pathlib import Path
from ztb.utils.checkpoint import HierarchicalCheckpointManager

def test_hierarchical_checkpoints():
    print('=== Step 3 Dry-run: Hierarchical Checkpoint System ===')

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = HierarchicalCheckpointManager(save_dir=tmpdir, compress="zlib")

        # Mock model state
        model_state = {"weights": [1, 2, 3], "bias": 0.5}
        optimizer_state = {"lr": 0.001, "momentum": 0.9}
        metrics = {"loss": 0.123, "accuracy": 0.95}

        # Test checkpoint saving at various steps
        test_steps = [1000, 5000, 10000, 15000, 50000, 100000]

        for step in test_steps:
            print(f'Saving checkpoint at step {step}...')
            manager.save_checkpoint(
                step=step,
                model_state=model_state,
                optimizer_state=optimizer_state,
                metrics=metrics,
                checkpoint_type="auto"
            )

        # Wait for async saves to complete
        manager.shutdown()

        # Check saved files
        checkpoint_dir = Path(tmpdir)
        all_files = list(checkpoint_dir.glob("*.pkl*"))
        print(f'Total checkpoint files: {len(all_files)}')

        for f in sorted(all_files):
            print(f'  {f.name}')

        # Test recovery
        recovery_cp = manager.find_recovery_checkpoint()
        if recovery_cp:
            print(f'Recovery checkpoint: {recovery_cp.name}')

            loaded_data = manager.load_checkpoint()
            if loaded_data:
                print(f'Loaded step: {loaded_data["step"]}')
                print(f'Loaded type: {loaded_data["type"]}')
                print(f'Loaded metrics: {loaded_data["metrics"]}')
            else:
                print('Failed to load checkpoint')
        else:
            print('No recovery checkpoint found')

    print('=== Step 3 Test Completed ===')

if __name__ == "__main__":
    test_hierarchical_checkpoints()