"""Lightweight shims for distributed performance tests.

These are simple placeholders used during test collection to avoid
import-time failures when the full distributed implementation is not
required for unit-level runs.
"""
from .coordinator import DistributedConfig, DistributedCoordinator
from .integration import DistributedTrainingManager
from .worker import WorkerPool

__all__ = ["DistributedConfig", "DistributedCoordinator", "DistributedTrainingManager", "WorkerPool"]
