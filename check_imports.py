#!/usr/bin/env python3
"""
Check if key ZTB modules can be imported
"""

import logging

logger = logging.getLogger(__name__)

try:
    from ztb.features import FeatureRegistry

    logger.info("features available")
except ImportError as e:
    logger.error("features not available: %s", e)

try:
    from ztb.risk.advanced_auto_stop import AdvancedAutoStop

    logger.info("auto_stop available")
except ImportError as e:
    logger.error("auto_stop not available: %s", e)
