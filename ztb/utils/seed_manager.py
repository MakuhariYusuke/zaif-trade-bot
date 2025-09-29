"""
Centralized seed management for deterministic training.

This module provides unified seed setting across all random number generators
used in the training pipeline to ensure reproducibility.
"""

import os
import random
from typing import Optional

# Optional imports - gracefully handle missing dependencies
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]


class SeedManager:
    """Centralized seed management for reproducibility."""

    def __init__(self) -> None:
        self.current_seed: Optional[int] = None
        self.determinism_enabled = True

    def set_seed(self, seed: Optional[int]) -> None:
        """Set seed across all random number generators.

        Args:
            seed: Random seed. If None, uses system entropy.
        """
        if seed is None:
            # Use system entropy for true randomness
            seed = int.from_bytes(os.urandom(4), byteorder="big")

        self.current_seed = seed

        # Set Python random seed
        random.seed(seed)

        # Set NumPy seed
        if HAS_NUMPY:
            np.random.seed(seed)  # type: ignore[attr-defined]

        # Set PyTorch seeds and enable deterministic behavior
        if HAS_TORCH:
            torch.manual_seed(seed)  # type: ignore[attr-defined]
            torch.cuda.manual_seed(seed)  # type: ignore[attr-defined]
            torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]

            # Enable deterministic algorithms
            if self.determinism_enabled:
                self._enable_torch_determinism()

    def _enable_torch_determinism(self) -> None:  # type: ignore[misc]
        """Enable deterministic behavior in PyTorch."""
        if not HAS_TORCH:
            return

        # Set deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # For reproducibility, disable TF32 on Ampere GPUs
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

        # Set environment variables for additional determinism
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def disable_determinism(self) -> None:  # type: ignore[misc]
        """Disable deterministic behavior for performance."""
        self.determinism_enabled = False

        if HAS_TORCH:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

            # Re-enable TF32
            if hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True

    def get_current_seed(self) -> Optional[int]:
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
_seed_manager: Optional[SeedManager] = None


def get_seed_manager() -> SeedManager:
    """Get global seed manager instance."""
    global _seed_manager
    if _seed_manager is None:
        _seed_manager = SeedManager()  # type: ignore[no-untyped-call]
    return _seed_manager


def set_global_seed(seed: Optional[int]) -> None:
    """Set seed globally across all random number generators."""
    manager = get_seed_manager()
    manager.set_seed(seed)


def get_current_global_seed() -> Optional[int]:
    """Get the currently set global seed."""
    manager = get_seed_manager()
    return manager.get_current_seed()
