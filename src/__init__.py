# Backward compatibility wrapper for src module
# This module has been moved to python/ directory

import warnings

warnings.warn(
    "The 'src' module has been moved to 'python'. "
    "Please update your imports to use 'python' instead of 'src'.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from python for backward compatibility
from python import *
