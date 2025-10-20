#!/usr/bin/env python3
"""
Test action diagnostics with existing trained model.

This script runs paper trading with verbose diagnostics to analyze:
- Pre/post mask logits and probabilities
- Deterministic action selection order
- Action distribution and bias
"""

import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.utils.config import TypedConfig


def main() -> int:
    """Run paper trade with verbose diagnostics."""

    # Use the trained model from ppo_100k config - use config-based path
    config = TypedConfig()
    model_path = config.get_model_path("ppo_100k_tuned.zip")

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Available models:")
        models_dir = Path("models")
        if models_dir.exists():
            for model in sorted(models_dir.glob("*.zip")):
                print(f"  - {model.name}")
        print("\nPlease specify an existing model or train one with:")
        print("  python -m ztb.training.unified_trainer --config ppo_100k_config.json")
        return 1

    print(f"Testing action diagnostics with model: {model_path}")
    print("=" * 80)

    # Run paper trading with verbose mode via command line
    cmd = [
        sys.executable,
        "-m",
        "ztb.training.paper_trade",
        "--model-path",
        model_path,
        "--test-data",
        "btc_jpy_real_dataset.csv",
        "--episodes",
        "3",
        "--verbose",
        "--config",
        "ppo_100k_config.json",
    ]

    print(f"Running command: {' '.join(cmd)}")
    print("=" * 80)

    result = subprocess.run(cmd, cwd=Path.cwd())

    print("\n" + "=" * 80)
    if result.returncode == 0:
        print("Paper trading completed successfully!")
        print("Check results/paper_trading/ for detailed logs")
    else:
        print(f"Paper trading failed with exit code {result.returncode}")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
