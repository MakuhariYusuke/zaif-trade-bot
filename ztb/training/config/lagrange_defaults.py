"""Central defaults for Lagrange constraint hyperparameters.

These values are derived from the 2025-10 long-run binary search campaign
covering 50k timestep evaluations for each parameter. Keeping them in a
single module avoids drift between training entrypoints, binary search
scripts, and documentation.
"""

from __future__ import annotations

from typing import Optional

Number = int | float

# NOTE:
# - warmup_steps intentionally kept as ``int`` but stored in the dict as ``Number``
#   to simplify type interactions with call-sites expecting numeric types.
LAGRANGE_DEFAULTS: dict[str, Number] = {
    "r_target": 0.175,  # Target SELL rate (17.5%)
    "tolerance": 0.042625,  # Deviation tolerance (~4.26%)
    "eta": 0.062875,  # Dual ascent step size
    "lambda_max": 3.875,  # Maximum dual variable value
    "warmup_steps": 3_874,  # Steps before enforcing constraint strongly
}

def apply_lagrange_overrides(
    overrides: dict[str, Number | None] | None = None,
) -> dict[str, Number]:
    """Return defaults merged with any non-``None`` overrides.

    Args:
        overrides: Optional mapping of keys (``r_target``, ``tolerance``,
            ``eta``, ``lambda_max``, ``warmup_steps``) to override values.

    Returns:
        A new dict containing the defaults with any provided overrides applied.
    """
    params = dict(LAGRANGE_DEFAULTS)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                params[key] = value
    return params
