#!/usr/bin/env python3
"""
SAC v427 Final Evaluation Script
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.sac_v427_advanced_trainer import SACv427AdvancedTrainer


def main():
    """Run final evaluation."""
    try:
        # Load trainer
        trainer = SACv427AdvancedTrainer(
            "configs/sac_v427_market_adaptive_ensemble.json"
        )

        # Run final evaluation
        results = trainer._final_evaluation()

        print("SAC v427 Final Evaluation Results:")
        print("=" * 50)
        print(json.dumps(results, indent=2, default=str))

        # Performance summary
        if "performance_metrics" in results:
            metrics = results["performance_metrics"]
            print("\nPerformance Summary:")
            print(f"Annual Return: {metrics.get('annual_return', 'N/A')}")
            print(f"Max Drawdown: {metrics.get('max_drawdown', 'N/A')}")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
            print(f"Win Rate: {metrics.get('win_rate', 'N/A')}")

        return 0

    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
