"""
Multi-Timeframe Feature System Interface

Interface for multi-timeframe feature engineering system.
"""

from ztb.features.generators.multi_timeframe import (
	MultiTimeframeFeatureSystem,
)

__all__ = ["MultiTimeframeFeatureSystem"]

# Expose submodules under the `ztb.features.multi_timeframe.*` dotted names so
# code that imports `ztb.features.multi_timeframe.data_pipeline` still works
# even though the implementation lives under `ztb.features.generators.multi_timeframe`.
import importlib
import sys

for sub in ("data_pipeline", "engine", "config"):
	src = f"ztb.features.generators.multi_timeframe.{sub}"
	dst = f"ztb.features.multi_timeframe.{sub}"
	try:
		mod = importlib.import_module(src)
		sys.modules[dst] = mod
	except Exception:
		# Ignore: tests will handle presence/absence, but we try to make imports work
		pass
