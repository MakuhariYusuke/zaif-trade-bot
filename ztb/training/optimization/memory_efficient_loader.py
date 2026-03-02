#!/usr/bin/env python3
"""
Memory efficient data loader for training optimization.
"""

import gc
import logging
from typing import Iterator

import torch
try:
    from torch.utils.data import DataLoader, Dataset
except Exception:
    # Provide lightweight fallbacks for environments without full torch
    class _SimpleDataLoader(list):
        def __init__(self, dataset, *args, **kwargs):
            super().__init__(list(dataset))

        def __iter__(self):
            return super().__iter__()

        def __len__(self):
            return super().__len__()

    DataLoader = _SimpleDataLoader
    Dataset = object

logger = logging.getLogger(__name__)

class MemoryEfficientLoader:
    """
    Memory-efficient data loader that manages GPU memory during training.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        memory_threshold: float = 0.8,
        cleanup_interval: int = 100,
    ):
        """
        Initialize memory efficient loader.

        Args:
            dataset: PyTorch dataset
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            prefetch_factor: Prefetch factor for data loading
            persistent_workers: Whether to keep workers alive
            memory_threshold: GPU memory threshold for cleanup (0-1)
            cleanup_interval: Steps between memory cleanup
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.memory_threshold = memory_threshold
        self.cleanup_interval = cleanup_interval

        self.step_count = 0
        self.data_loader = self._create_loader()

    def _create_loader(self) -> DataLoader:
        """Create the underlying DataLoader."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers
            if self.num_workers > 0
            else False,
        )

    def __iter__(self) -> Iterator:
        """Iterate over the data loader."""
        for batch in self.data_loader:
            yield batch

            self.step_count += 1
            if self.step_count % self.cleanup_interval == 0:
                self._cleanup_memory()

    def _cleanup_memory(self) -> None:
        """Clean up GPU memory if threshold exceeded."""
        if torch.cuda.is_available():
            memory_used = (
                torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            )
            if memory_used > self.memory_threshold:
                torch.cuda.empty_cache()
                gc.collect()
                logger.debug(f"Memory cleanup triggered. Usage: {memory_used:.2%}")

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.data_loader)

    def set_epoch(self, epoch: int) -> None:
        """set epoch for distributed training."""
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def cleanup(self) -> None:
        """Manual cleanup of resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.debug("Manual memory cleanup completed")
