# Compatibility shim for trend features package
# Re-export commonly used trend submodules from generators.technical
from ztb.features.generators.technical.trend.ichimoku.ichimoku_cloud_expansion import (
    compute_ichimoku_cloud_expansion,
)

__all__ = ["compute_ichimoku_cloud_expansion"]
