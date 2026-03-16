"""
Centralized seed management for deterministic training.

This module provides unified seed setting across all random number generators
used in the training pipeline to ensure reproducibility.
"""

import logging
import os
import random
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

# Optional imports - gracefully handle missing dependencies
if TYPE_CHECKING:
    # For type checking, import numpy and torch types if available
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover - type checking only
        np = None  # type: ignore[assignment]
    try:
        import torch
    except Exception as e:  # pragma: no cover - type checking only
        torch = None  # type: ignore[assignment]

    HAS_NUMPY = True
    HAS_TORCH = True
else:
    # Avoid importing torch at module import time; import lazily inside functions
    HAS_TORCH = False
    torch = None

    try:
        import numpy as np

        HAS_NUMPY = True
    except ImportError:
        HAS_NUMPY = False
        np = None

class SeedManager:
    """Centralized seed management for reproducibility."""

    def __init__(self) -> None:
        self.current_seed: int | None = None
        self.determinism_enabled = True

    def set_seed(self, seed: int | None) -> None:
        """set seed across all random number generators.

        Args:
            seed: Random seed. If None, uses system entropy.
        """
        if seed is None:
            # Use system entropy for true randomness
            seed = int.from_bytes(os.urandom(4), byteorder="big")

        self.current_seed = seed

        # set Python random seed
        random.seed(seed)

        # set NumPy seed
        if HAS_NUMPY:
            np.random.seed(seed)

        # set PyTorch seeds and enable deterministic behavior
        tmod = None
        if HAS_TORCH and torch is not None:
            tmod = torch
        else:
            try:
                import importlib

                tmod = importlib.import_module("torch")
                # Cache torch global to avoid re-import
                globals()["torch"] = tmod
                globals()["HAS_TORCH"] = True
            except Exception as e:
                tmod = None

        if tmod is not None:
            tmod.manual_seed(seed)
            try:
                tmod.cuda.manual_seed(seed)
                tmod.cuda.manual_seed_all(seed)
            except Exception as e:
                # cuda.* may fail on CPU-only builds - ignore
                logger.debug("seed setting cuda failed: %s", e)

            # Enable deterministic algorithms
            if self.determinism_enabled:
                self._enable_torch_determinism(tmod)

    def _enable_torch_determinism(self, tmod) -> None:
        """Enable deterministic behavior in PyTorch."""
        if tmod is None:
            return

        # set deterministic algorithms
        try:
            tmod.backends.cudnn.deterministic = True
            tmod.backends.cudnn.benchmark = False
        except Exception as e:
            logger.debug("seed setting cudnn.deterministic failed: %s", e)

        # For reproducibility, disable TF32 on Ampere GPUs
        try:
            if hasattr(tmod.backends.cuda, "matmul"):
                tmod.backends.cuda.matmul.allow_tf32 = False
        except Exception as e:
            logger.debug("seed setting cuda.matmul.allow_tf32 failed: %s", e)
        try:
            if hasattr(tmod.backends.cudnn, "allow_tf32"):
                tmod.backends.cudnn.allow_tf32 = False
        except Exception as e:
            logger.debug("seed setting cudnn.allow_tf32 failed: %s", e)

        # set environment variables for additional determinism
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def disable_determinism(self) -> None:
        """Disable deterministic behavior for performance."""
        self.determinism_enabled = False

        try:
            import importlib

            tmod = importlib.import_module("torch")
        except Exception as e:
            tmod = None

        if tmod is not None:
            try:
                tmod.backends.cudnn.deterministic = False
                tmod.backends.cudnn.benchmark = True
            except Exception as e:
                logger.debug("seed setting cudnn.benchmark failed: %s", e)
            # Re-enable TF32
            try:
                if hasattr(tmod.backends.cuda, "matmul"):
                    tmod.backends.cuda.matmul.allow_tf32 = True
            except Exception as e:
                logger.debug("seed setting cuda.matmul.allow_tf32 failed: %s", e)
            try:
                if hasattr(tmod.backends.cudnn, "allow_tf32"):
                    tmod.backends.cudnn.allow_tf32 = True
            except Exception as e:
                logger.debug("seed setting cudnn.allow_tf32 failed: %s", e)

    def get_current_seed(self) -> int | None:
        """Get the currently set seed."""
        return self.current_seed

    def generate_deterministic_seed(self, base_seed: int, context: str) -> int:
        """Generate a deterministic seed from base seed and context.

        Args:
            base_seed: Base random seed
            context: Context string to derive seed from

        Returns:
            Deterministic seed derived from base and context
        """
        import hashlib

        combined = f"{base_seed}:{context}"
        hash_obj = hashlib.sha256(combined.encode())
        return int(hash_obj.hexdigest()[:8], 16)

    def fork_seed(self, context: str) -> int:
        """Fork current seed with context for independent randomization.

        Args:
            context: Context string for seed derivation

        Returns:
            New seed derived from current seed and context
        """
        if self.current_seed is None:
            raise ValueError("No base seed set. Call set_seed() first.")

        return self.generate_deterministic_seed(self.current_seed, context)

# Global seed manager instance
_seed_manager: SeedManager | None = None

def get_seed_manager() -> SeedManager:
    """Get global seed manager instance."""
    global _seed_manager
    if _seed_manager is None:
        _seed_manager = SeedManager()
    return _seed_manager

def set_global_seed(seed: int | None) -> None:
    """set seed globally across all random number generators."""
    manager = get_seed_manager()
    manager.set_seed(seed)

def get_current_global_seed() -> int | None:
    """Get the currently set global seed."""
    manager = get_seed_manager()
    return manager.get_current_seed()
