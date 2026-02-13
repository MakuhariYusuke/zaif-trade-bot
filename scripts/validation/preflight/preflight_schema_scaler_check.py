"""Compatibility wrapper for the preflight validation script.

Some tests import preflight_schema_scaler_check directly from the scripts
module; the original was moved to archived/scripts. Re-export the key
functions here for tests.
"""

from __future__ import annotations

from archived.scripts.preflight_schema_scaler_check import (
    check_config_fingerprint,
    check_feature_schema,
    check_normalization_stats,
    compare_with_training,
    main,
)

__all__ = [
    "check_feature_schema",
    "check_normalization_stats",
    "check_config_fingerprint",
    "compare_with_training",
    "main",
]
