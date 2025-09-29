# Backward compatibility wrapper for python module
# This module has been moved to ztb/ directory

import warnings

warnings.warn(
    "The 'python' module has been moved to 'ztb'. "
    "Please update your imports to use 'ztb' instead of 'python'.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from ztb for backward compatibility
from ztb import *
