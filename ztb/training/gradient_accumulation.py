"""
Gradient Accumulation Utilities for SAC Training

This module provides gradient accumulation functionality to enable training
with effectively larger batch sizes on limited GPU memory.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class GradientAccumulator:
    """
    Gradient accumulation utility for training with large effective batch sizes.
    """

    def __init__(
        self,
        accumulation_steps: int = 1,
        clip_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
        mixed_precision: bool = False,
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ):
        """
        Initialize gradient accumulator.

        Args:
            accumulation_steps: Number of steps to accumulate gradients
            clip_grad_norm: Maximum gradient norm for clipping
            clip_grad_value: Maximum gradient value for clipping
            mixed_precision: Whether to use mixed precision
            scaler: GradScaler for mixed precision training
        """
        self.accumulation_steps = accumulation_steps
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value
        self.mixed_precision = mixed_precision
        self.scaler = scaler

        self.step_count = 0
        self.accumulated_loss = 0.0
        self.gradient_accumulated = False

    def should_accumulate(self) -> bool:
        """Check if gradients should be accumulated (not the last step)."""
        return self.step_count % self.accumulation_steps != self.accumulation_steps - 1

    def should_update(self) -> bool:
        """Check if parameters should be updated (last step in accumulation)."""
        return self.step_count % self.accumulation_steps == self.accumulation_steps - 1

    def accumulate_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Perform one accumulation step.

        Args:
            loss: Loss tensor to accumulate
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)

        Returns:
            Step information dictionary
        """
        loss_value = loss.item() / self.accumulation_steps  # Normalize loss
        self.accumulated_loss += loss_value

        # Scale loss for accumulation
        scaled_loss = loss / self.accumulation_steps

        # Check if this step should update parameters
        should_update_now = self.should_update()

        # Backward pass
        if self.mixed_precision and self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        self.step_count += 1

        step_info = {
            'step_loss': loss_value,
            'accumulated_loss': self.accumulated_loss,
            'step_count': self.step_count,
            'should_update': should_update_now
        }

        # Update parameters if accumulation is complete
        if should_update_now:
            self._update_parameters(optimizer, scheduler)
            step_info.update(self._get_update_info())

        return step_info

    def _update_parameters(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None
    ):
        """Update model parameters after gradient accumulation."""
        # Clip gradients if specified
        if self.clip_grad_norm is not None:
            if self.mixed_precision and self.scaler is not None:
                self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group['params']],
                self.clip_grad_norm
            )

        if self.clip_grad_value is not None:
            if self.mixed_precision and self.scaler is not None:
                self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_value_(
                [p for group in optimizer.param_groups for p in group['params']],
                self.clip_grad_value
            )

        if self.clip_grad_value is not None:
            if self.mixed_precision and self.scaler is not None:
                self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_value_(
                [p for group in optimizer.param_groups for p in group['params']],
                self.clip_grad_value
            )

        # Update parameters
        if self.mixed_precision and self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

        # Update learning rate scheduler
        if scheduler is not None:
            if hasattr(scheduler, 'step'):
                scheduler.step()

        # Zero gradients
        optimizer.zero_grad()

        # Reset accumulation state
        self.step_count = 0
        self.accumulated_loss = 0.0

    def _get_update_info(self) -> Dict[str, Any]:
        """Get information about the parameter update."""
        return {
            'parameters_updated': True,
            'accumulation_steps_completed': self.accumulation_steps
        }

    def reset(self):
        """Reset accumulator state."""
        self.step_count = 0
        self.accumulated_loss = 0.0

    @contextmanager
    def accumulation_context(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None
    ):
        """
        Context manager for gradient accumulation.

        Usage:
            with accumulator.accumulation_context(optimizer, scheduler) as accumulate:
                for batch in batches:
                    loss = model(batch)
                    accumulate(loss)
        """
        accumulated_losses = []

        def accumulate(loss: torch.Tensor):
            step_info = self.accumulate_step(loss, optimizer, scheduler)
            accumulated_losses.append(step_info)
            return step_info

        try:
            yield accumulate
        finally:
            # Ensure final update if accumulation cycle is incomplete
            if self.step_count > 0:
                logger.warning(f"Gradient accumulation incomplete ({self.step_count}/{self.accumulation_steps}). "
                             "Completing current accumulation cycle.")
                # Add dummy losses to complete the cycle
                remaining_steps = self.accumulation_steps - self.step_count
                for _ in range(remaining_steps):
                    dummy_loss = torch.tensor(0.0, requires_grad=True)
                    self.accumulate_step(dummy_loss, optimizer, scheduler)


class GradientAccumulationTrainer:
    """
    Trainer wrapper that adds gradient accumulation support.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 1,
        clip_grad_norm: Optional[float] = None,
        mixed_precision: bool = False,
        device: str = 'cpu'
    ):
        """
        Initialize gradient accumulation trainer.

        Args:
            model: Neural network model
            optimizer: Optimizer
            accumulation_steps: Number of steps to accumulate
            clip_grad_norm: Gradient norm clipping
            mixed_precision: Enable mixed precision
            device: Training device
        """
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.device = device

        # Setup mixed precision
        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision and device.startswith('cuda') else None

        # Setup gradient accumulator
        self.accumulator = GradientAccumulator(
            accumulation_steps=accumulation_steps,
            clip_grad_norm=clip_grad_norm,
            mixed_precision=mixed_precision,
            scaler=self.scaler
        )

    def training_step(
        self,
        batch: Any,
        loss_fn: Callable[[Any, Any], torch.Tensor],
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Perform one training step with gradient accumulation.

        Args:
            batch: Training batch
            loss_fn: Loss function (model_output, batch) -> loss
            scheduler: Learning rate scheduler

        Returns:
            Training step information
        """
        # Move batch to device
        if hasattr(batch, 'to'):
            batch = batch.to(self.device)
        elif isinstance(batch, (list, tuple)):
            batch = [b.to(self.device) if hasattr(b, 'to') else b for b in batch]

        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            model_output = self.model(batch)
            loss = loss_fn(model_output, batch)

        # Accumulate gradients
        step_info = self.accumulator.accumulate_step(loss, self.optimizer, scheduler)

        return {
            'loss': step_info['step_loss'],
            'accumulated_loss': step_info['accumulated_loss'],
            'step_count': step_info['step_count'],
            'parameters_updated': step_info.get('parameters_updated', False),
            'effective_batch_size': len(batch) * self.accumulation_steps
        }

    def get_effective_batch_size(self, actual_batch_size: int) -> int:
        """Get effective batch size after accumulation."""
        return actual_batch_size * self.accumulation_steps

    def reset_accumulator(self):
        """Reset gradient accumulator."""
        self.accumulator.reset()


# Utility functions
def create_gradient_accumulator(
    accumulation_steps: int = 4,
    clip_grad_norm: float = 1.0,
    mixed_precision: bool = False,
    device: str = 'cpu'
) -> GradientAccumulator:
    """
    Create gradient accumulator with sensible defaults.

    Args:
        accumulation_steps: Steps to accumulate
        clip_grad_norm: Gradient clipping norm
        mixed_precision: Enable mixed precision
        device: Target device

    Returns:
        Configured GradientAccumulator
    """
    scaler = None
    if mixed_precision and device.startswith('cuda'):
        scaler = torch.cuda.amp.GradScaler()

    return GradientAccumulator(
        accumulation_steps=accumulation_steps,
        clip_grad_norm=clip_grad_norm,
        mixed_precision=mixed_precision,
        scaler=scaler
    )


def effective_batch_size_info(
    actual_batch_size: int,
    accumulation_steps: int,
    num_epochs: int = 1
) -> Dict[str, Any]:
    """
    Get information about effective batch size and training implications.

    Args:
        actual_batch_size: Actual batch size per step
        accumulation_steps: Gradient accumulation steps
        num_epochs: Number of training epochs

    Returns:
        Information dictionary
    """
    effective_batch = actual_batch_size * accumulation_steps

    return {
        'actual_batch_size': actual_batch_size,
        'accumulation_steps': accumulation_steps,
        'effective_batch_size': effective_batch,
        'gradient_updates_per_epoch': actual_batch_size,  # One update per accumulation cycle
        'memory_efficiency': f"{accumulation_steps}x larger effective batch with same memory",
        'training_notes': [
            f"Effective batch size: {effective_batch}",
            f"Gradient updates per epoch: {actual_batch_size}",
            "Loss values are normalized by accumulation_steps",
            "Learning rate may need adjustment for larger effective batch"
        ]
    }