#!/usr/bin/env python3
"""
Basic test script for ensemble trading system.

アンサンブル取引システムの基本テストスクリプト。
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.ensemble import EnsembleTradingSystem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ensemble_basic():
    """Test basic ensemble functionality."""
    # Example model configurations (replace with actual paths)
    model_configs = [
        {
            "path": "models/scalping_15s_balance_test12_balanced_data.zip",
            "weight": 1.0,
            "feature_set": "scalping",
        },
        # Add more models when available
    ]

    try:
        # Create ensemble system
        ensemble_system = EnsembleTradingSystem(model_configs)
        logger.info("Ensemble system created successfully")

        # Get ensemble info
        info = ensemble_system.ensemble.get_ensemble_info()
        logger.info(f"Ensemble info: {info}")

        # Test confidence calculation (would need actual observation)
        logger.info("Basic ensemble test completed")

    except Exception as e:
        logger.error(f"Ensemble test failed: {e}")
        return False

    return True


if __name__ == "__main__":
    success = test_ensemble_basic()
    sys.exit(0 if success else 1)
