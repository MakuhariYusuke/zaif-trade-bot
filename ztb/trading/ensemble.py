#!/usr/bin/env python3
"""
Ensemble Trading System for Zaif Trade Bot.

This module is deprecated. Use ztb.training.ensemble instead.
"""

import warnings
from typing import Any, Dict, List, Optional

# Import from training module
from ztb.training.ensemble import (
    EnsemblePredictorLegacy as EnsemblePredictor,
    EnsembleTradingSystemLegacy as EnsembleTradingSystem,
    create_default_ensemble_legacy as create_default_ensemble,
)

# Issue deprecation warning
warnings.warn(
    "ztb.trading.ensemble is deprecated. Use ztb.training.ensemble instead.",
    DeprecationWarning,
    stacklevel=2,
)
