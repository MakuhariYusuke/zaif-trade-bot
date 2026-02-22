#!/usr/bin/env python3
"""
Simple backtest for v384 (68 curated features) on historical BTC data.

This script tests the v384 model which uses curated features.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import json
from datetime import datetime

import numpy as np

from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.training.policies.policy_utils import predict_with_masks
from ztb.utils.config import TypedConfig
from ztb.io.data_loader import DataLoader
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)




if __name__ == "__main__":
    sys.exit(main())
