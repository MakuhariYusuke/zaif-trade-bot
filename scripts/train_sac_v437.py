#!/usr/bin/env python3
"""
SAC v437 Training Script - Unified Trainer Version

Enhanced SAC training with v427 features for improved trading performance.
Now uses unified trainer system for consistency and maintainability.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)




if __name__ == "__main__":
    main()
