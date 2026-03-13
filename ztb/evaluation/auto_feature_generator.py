"""Re-export shim for AutoFeatureGenerator.

Canonical location: ztb.analysis.features.auto_feature_generator
"""

from ztb.analysis.features.auto_feature_generator import (  # noqa: F401
    AutoFeatureGenerator,
    ParameterCombinationGenerator,
)

__all__ = ["ParameterCombinationGenerator", "AutoFeatureGenerator"]
