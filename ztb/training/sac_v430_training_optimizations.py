#!/usr/bin/env python3
"""
SAC v430 Training Optimization Guide & Scripts

This module provides additional optimization techniques for efficient SAC v430 training:
- Gradient accumulation for larger effective batch sizes
- Dynamic learning rate scheduling
- Early stopping with validation
- Memory-efficient data loading
- Parallel environment evaluation
"""

import copy
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ztb.trading.environment.constants import PYTORCH_CUDA_ALLOC_MB


class GradientAccumulator:
    """Gradient accumulation for effective larger batch sizes with limited memory."""

    def __init__(self, accumulation_steps: int = 4):
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
        self.scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None

    def step(
        self,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> bool:
        """
        Perform gradient accumulation step.

        Returns:
            True if optimizer.step() was called, False otherwise
        """
        loss = loss / self.accumulation_steps

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self.step_count += 1

        if self.step_count >= self.accumulation_steps:
            if self.scaler:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            self.step_count = 0
            return True

        return False


class DynamicLRScheduler:
    """Dynamic learning rate scheduler with plateau detection and recovery."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        patience: int = 10,
        factor: float = 0.5,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_loss = float("inf")
        self.counter = 0
        self.last_lr = self._get_lr()

    def _get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def _set_lr(self, lr: float):
        """Set learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self, current_loss: float) -> Dict[str, Any]:
        """Update learning rate based on current loss."""
        info = {"lr_changed": False, "lr": self._get_lr(), "action": "none"}

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
            info["action"] = "improvement"
        else:
            self.counter += 1
            info["action"] = f"plateau_{self.counter}"

        if self.counter >= self.patience:
            new_lr = max(self._get_lr() * self.factor, self.min_lr)
            if new_lr < self._get_lr():
                self._set_lr(new_lr)
                self.counter = 0
                info["lr_changed"] = True
                info["lr"] = new_lr
                info["action"] = "lr_decay"
            else:
                info["action"] = "lr_floor_reached"

        return info


class EarlyStopping:
    """Early stopping with validation-based criteria."""

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.001,
        restore_best_weights: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights = None
        self.best_epoch = 0

    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """
        Check if training should stop.

        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch

            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best_weights(self, model: nn.Module):
        """Restore best weights if available."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            print(f"Restored weights from epoch {self.best_epoch}")


class MemoryEfficientLoader:
    """Memory-efficient data loader with streaming and chunking."""

    def __init__(
        self, batch_size: int = 64, chunk_size: int = 1000, max_memory_gb: float = 4.0
    ):
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb

    def create_efficient_dataloader(self, data: Dict[str, np.ndarray]) -> DataLoader:
        """Create memory-efficient DataLoader."""

        # Convert to tensors with appropriate dtypes
        tensors = {}
        for key, array in data.items():
            if array.dtype == np.float64:
                tensors[key] = torch.from_numpy(array.astype(np.float32))
            else:
                tensors[key] = torch.from_numpy(array)

        # Create dataset
        dataset = TensorDataset(*tensors.values())

        # Calculate optimal batch size based on memory
        optimal_batch_size = self._calculate_optimal_batch_size(data)

        return DataLoader(
            dataset,
            batch_size=min(optimal_batch_size, self.batch_size),
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=0,  # Avoid multiprocessing issues
            prefetch_factor=None if torch.__version__ < "1.13" else 2,
        )

    def _calculate_optimal_batch_size(self, data: Dict[str, np.ndarray]) -> int:
        """Calculate optimal batch size based on available memory."""
        total_elements = sum(arr.nbytes for arr in data.values())
        elements_per_sample = total_elements / len(next(iter(data.values())))

        # Target: use 50% of available memory for batch
        target_memory_bytes = self.max_memory_gb * 0.5 * (1024**3)
        optimal_batch_size = int(target_memory_bytes / elements_per_sample)

        return max(1, min(optimal_batch_size, 512))  # Reasonable bounds


class ParallelEvaluator:
    """Parallel environment evaluation for faster validation."""

    def __init__(self, n_envs: int = 4, max_episode_steps: int = 1000):
        self.n_envs = n_envs
        self.max_episode_steps = max_episode_steps

    def evaluate_parallel(
        self, model, env_factory, n_episodes: int = 10
    ) -> Dict[str, float]:
        """Evaluate model in parallel environments."""

        from stable_baselines3.common.vec_env import SubprocVecEnv

        # Create vectorized environment
        envs = SubprocVecEnv([env_factory for _ in range(self.n_envs)])

        episode_rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            obs = envs.reset()
            episode_reward = np.zeros(self.n_envs)
            episode_length = np.zeros(self.n_envs, dtype=int)

            done = np.zeros(self.n_envs, dtype=bool)

            while not np.all(done) and np.max(episode_length) < self.max_episode_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = envs.step(action)

                episode_reward += reward * (1 - done)  # Only add reward if not done
                episode_length += 1

            episode_rewards.extend(episode_reward)
            episode_lengths.extend(episode_length)

        envs.close()

        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
            "n_episodes": len(episode_rewards),
        }


class SACv430Optimizer:
    """Additional optimization techniques for SAC v430 training."""

    def __init__(self):
        self.gradient_accumulator = GradientAccumulator(accumulation_steps=4)
        self.lr_scheduler = None
        self.early_stopping = EarlyStopping(patience=15)
        self.memory_loader = MemoryEfficientLoader()
        self.parallel_evaluator = ParallelEvaluator(n_envs=4)

    def optimize_training_loop(self, trainer, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize the training loop with advanced techniques.

        This is a high-level example of how to integrate these optimizations.
        In practice, this would be integrated into the UnifiedTrainer.
        """
        results = {"training_time": 0, "final_metrics": {}, "optimization_events": []}

        start_time = time.time()

        # Setup dynamic LR scheduling
        if hasattr(trainer, "optimizer"):
            self.lr_scheduler = DynamicLRScheduler(trainer.optimizer, patience=10)

        # Training loop with optimizations
        for epoch in range(config.get("n_epochs", 100)):
            epoch_start = time.time()

            # Training step with gradient accumulation
            train_loss = self._training_step_with_accumulation(trainer)

            # Validation with parallel evaluation
            if epoch % config.get("eval_interval", 10) == 0:
                val_metrics = self._parallel_validation(trainer)

                # Early stopping check
                if self.early_stopping(
                    val_metrics.get("val_loss", 0), trainer.model, epoch
                ):
                    results["optimization_events"].append(
                        {
                            "epoch": epoch,
                            "event": "early_stopping",
                            "reason": f"no improvement for {self.early_stopping.patience} epochs",
                        }
                    )
                    break

                # Dynamic LR adjustment
                if self.lr_scheduler:
                    lr_info = self.lr_scheduler.step(val_metrics.get("val_loss", 0))
                    if lr_info["lr_changed"]:
                        results["optimization_events"].append(
                            {
                                "epoch": epoch,
                                "event": "lr_decay",
                                "old_lr": lr_info.get("old_lr", 0),
                                "new_lr": lr_info["lr"],
                            }
                        )

            epoch_time = time.time() - epoch_start
            results["optimization_events"].append(
                {"epoch": epoch, "train_loss": train_loss, "epoch_time": epoch_time}
            )

        results["training_time"] = time.time() - start_time
        results["final_metrics"] = self._get_final_metrics(trainer)

        return results

    def _training_step_with_accumulation(self, trainer) -> float:
        """Perform training step with gradient accumulation."""
        # This is a simplified example - actual implementation would depend on trainer structure
        return 0.1  # Placeholder

    def _parallel_validation(self, trainer) -> Dict[str, float]:
        """Perform parallel validation."""
        # This is a simplified example - actual implementation would depend on trainer structure
        return {"val_loss": 0.05, "val_reward": 1.2}  # Placeholder

    def _get_final_metrics(self, trainer) -> Dict[str, float]:
        """Get final training metrics."""
        return {"final_loss": 0.02, "convergence_score": 0.95}  # Placeholder


# Utility functions for efficient training
def setup_efficient_training():
    """Setup environment for efficient training."""
    # Memory optimizations
    torch.set_num_threads(1)  # Avoid oversubscription
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Set memory efficient options
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{PYTORCH_CUDA_ALLOC_MB}"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def create_optimized_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create optimized configuration for efficient training."""
    config = copy.deepcopy(base_config)

    # Add optimization settings
    config["optimization"] = {
        "gradient_accumulation_steps": 4,
        "dynamic_lr_patience": 10,
        "early_stopping_patience": 15,
        "memory_efficient_loading": True,
        "parallel_evaluation": True,
        "mixed_precision": torch.cuda.is_available(),
    }

    # Adjust training parameters for efficiency
    if "training" in config:
        training = config["training"]

        # Optimize batch size based on memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
                1024**3
            )  # GB
            if gpu_memory < 8:  # Low memory GPU
                training["batch_size"] = min(training.get("batch_size", 256), 128)
            elif gpu_memory > 16:  # High memory GPU
                training["batch_size"] = max(training.get("batch_size", 256), 512)

        # Add memory optimization
        training["memory_efficient"] = True
        training[
            "gradient_checkpointing"
        ] = False  # Can be enabled for very large models

    return config


# Example usage
if __name__ == "__main__":
    print("SAC v430 Training Optimization Utilities")
    print("=" * 50)
    print("Available optimizations:")
    print("• Gradient Accumulation")
    print("• Dynamic Learning Rate Scheduling")
    print("• Early Stopping")
    print("• Memory-Efficient Data Loading")
    print("• Parallel Environment Evaluation")
    print("\nUse these utilities in your training scripts for better efficiency!")
