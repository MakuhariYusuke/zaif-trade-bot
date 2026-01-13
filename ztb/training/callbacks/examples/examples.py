#!/usr/bin/env python3
"""
Modern Callback System Usage Examples and Documentation.

This module demonstrates how to use the modern callback system in training scripts.
It provides comprehensive examples for all callback types and integration patterns.
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from ztb.training.callbacks.core.callback_implementations import (
    create_checkpoint_callback,
    create_logging_callback,
    create_metrics_callback,
    create_progress_callback,
)
from ztb.training.callbacks.core.modern_callback_system import CallbackContext, CallbackEvent, CallbackManager
from ztb.utils.logging_utils import setup_logging

def setup_logging_for_examples():
    """Set up logging for examples."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def basic_callback_usage_example():
    """
    Basic callback usage example.

    This example shows how to create and use individual callbacks.
    """
    print("=== Basic Callback Usage Example ===")

    # Create callbacks
    progress_cb = create_progress_callback(log_interval=10)
    metrics_cb = create_metrics_callback(collection_interval=5, log_interval=10)

    # Create manager and register callbacks
    manager = CallbackManager()
    manager.register_callback(progress_cb)
    manager.register_callback(metrics_cb)

    # Simulate training
    total_steps = 50

    # Training start
    start_context = CallbackContext(
        step=0, total_steps=total_steps, total_epochs=1, timestamp=0.0
    )
    manager.trigger_event(CallbackEvent.TRAINING_START, start_context)

    # Training steps
    for step in range(1, total_steps + 1):
        step_context = CallbackContext(
            step=step,
            total_steps=total_steps,
            timestamp=float(step) * 0.1,  # Simulate time progression
            metrics={
                "loss": 1.0 / (step + 1),  # Decreasing loss
                "reward": step * 0.02,  # Increasing reward
                "episode_reward": step * 0.05,
            },
        )
        manager.trigger_event(CallbackEvent.STEP_END, step_context)

    # Training end
    end_context = CallbackContext(
        step=total_steps, total_steps=total_steps, timestamp=float(total_steps) * 0.1
    )
    manager.trigger_event(CallbackEvent.TRAINING_END, end_context)

    # Get final metrics
    final_metrics = metrics_cb.get_metrics_history()
    print(f"Collected {len(final_metrics)} metrics entries")

    print("Basic example completed!\n")


def comprehensive_training_example():
    """
    Comprehensive training example with all callback types.

    This example demonstrates a complete training setup with all available callbacks.
    """
    print("=== Comprehensive Training Example ===")

    # Create all types of callbacks
    callbacks = [
        create_progress_callback(name="progress", log_interval=20, show_eta=True),
        create_checkpoint_callback(
            name="checkpoint", save_interval=50, save_path="./example_checkpoints"
        ),
        create_metrics_callback(
            name="metrics", collection_interval=10, log_interval=25
        ),
        create_logging_callback(name="logging", log_level="INFO"),
    ]

    # Create manager and register all callbacks
    manager = CallbackManager()
    for callback in callbacks:
        manager.register_callback(callback)

    # Simulate a more complex training scenario
    total_steps = 100
    episodes_per_step = 5

    print(f"Starting training simulation with {total_steps} steps...")

    # Training start
    start_context = CallbackContext(
        step=0,
        total_steps=total_steps,
        total_epochs=1,
        timestamp=0.0,
        custom_data={"model_name": "example_agent", "environment": "CartPole-v1"},
    )
    manager.trigger_event(CallbackEvent.TRAINING_START, start_context)

    # Training loop
    for step in range(1, total_steps + 1):
        # Simulate training step with varying performance
        base_reward = step * 0.1
        noise = (step % 10 - 5) * 0.01  # Add some noise
        episode_reward = base_reward + noise

        loss = max(0.01, 1.0 / (step**0.5))  # Decreasing loss with sqrt

        step_context = CallbackContext(
            step=step,
            total_steps=total_steps,
            timestamp=float(step) * 0.15,
            metrics={
                "loss": loss,
                "reward": episode_reward,
                "episode_reward": episode_reward * episodes_per_step,
                "episodes_completed": episodes_per_step,
                "steps_per_second": 6.67,  # ~100 steps per 15 seconds
            },
            custom_data={
                "learning_rate": 0.001 / (1 + step * 0.01),  # Decaying LR
                "epsilon": max(0.01, 1.0 - step * 0.005),  # Decaying epsilon
            },
        )

        # Trigger step end event
        manager.trigger_event(CallbackEvent.STEP_END, step_context)

        # Occasionally trigger metrics update event
        if step % 30 == 0:
            manager.trigger_event(CallbackEvent.METRICS_UPDATE, step_context)

    # Training end
    end_context = CallbackContext(
        step=total_steps,
        total_steps=total_steps,
        timestamp=float(total_steps) * 0.15,
        custom_data={"final_model_path": "./example_checkpoints/checkpoint_final.zip"},
    )
    manager.trigger_event(CallbackEvent.TRAINING_END, end_context)

    # Display final statistics
    stats = manager.get_statistics()
    print("\nTraining Statistics:")
    print(f"  Total callback executions: {stats['total_executions']}")
    print(f"  Successful executions: {stats['successful_executions']}")
    print(f"  Failed executions: {stats['failed_executions']}")

    # Get final metrics summary
    metrics_history = callbacks[2].get_metrics_history()  # metrics callback
    if metrics_history:
        latest = metrics_history[-1]
        print("\nFinal Metrics:")
        for key, value in latest.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")

    print("Comprehensive example completed!\n")


async def async_callback_example():
    """
    Asynchronous callback usage example.

    This example demonstrates how to use async callbacks for non-blocking operations.
    """
    print("=== Asynchronous Callback Example ===")

    # Create an async callback for demonstration
    from .callback_implementations import MetricsCallback

    class AsyncMetricsCallback(MetricsCallback):
        """Async version of metrics callback."""

        async def on_step_end_async(self, context: CallbackContext):
            """Async version of step end handling."""
            # Simulate async I/O operation (e.g., logging to remote server)
            await asyncio.sleep(0.01)  # Simulate network delay
            return await super().on_step_end_async(context)

    # Create manager and register async callback
    manager = CallbackManager()
    async_metrics_cb = AsyncMetricsCallback(collection_interval=5, log_interval=10)
    manager.register_callback(async_metrics_cb)

    # Simulate async training
    total_steps = 20

    print("Running async training simulation...")

    # Training start
    start_context = CallbackContext(step=0, total_steps=total_steps, timestamp=0.0)
    await manager.trigger_event_async(CallbackEvent.TRAINING_START, start_context)

    # Async training steps
    for step in range(1, total_steps + 1):
        step_context = CallbackContext(
            step=step,
            total_steps=total_steps,
            timestamp=float(step) * 0.1,
            metrics={"loss": 1.0 / step, "reward": step * 0.05},
        )

        # Use async triggering
        results = await manager.trigger_event_async(
            CallbackEvent.STEP_END, step_context
        )

        # Check results
        successful = sum(1 for r in results if r.success)
        print(f"Step {step}: {successful}/{len(results)} callbacks succeeded")

    # Training end
    end_context = CallbackContext(
        step=total_steps, total_steps=total_steps, timestamp=2.0
    )
    await manager.trigger_event_async(CallbackEvent.TRAINING_END, end_context)

    print("Async example completed!\n")


def error_handling_example():
    """
    Error handling example.

    This example shows how the callback system handles errors gracefully.
    """
    print("=== Error Handling Example ===")

    from .modern_callback_system import BaseCallback

    class FailingCallback(BaseCallback):
        """A callback that sometimes fails."""

        def __init__(self, fail_at_step: int):
            super().__init__()
            self.fail_at_step = fail_at_step

        def on_step_end(self, context):
            if context.step == self.fail_at_step:
                raise RuntimeError(f"Simulated failure at step {context.step}")
            return super().on_step_end(context)

    # Create manager with mix of working and failing callbacks
    manager = CallbackManager()
    manager.register_callback(create_progress_callback(log_interval=5))
    manager.register_callback(FailingCallback(fail_at_step=7))  # Will fail at step 7
    manager.register_callback(
        create_metrics_callback(collection_interval=3, log_interval=10)
    )

    # Simulate training with error
    total_steps = 15

    print("Running training with intentional failure at step 7...")

    # Training start
    start_context = CallbackContext(step=0, total_steps=total_steps, timestamp=0.0)
    results = manager.trigger_event(CallbackEvent.TRAINING_START, start_context)
    print(
        f"Training start: {sum(1 for r in results if r.success)}/{len(results)} succeeded"
    )

    # Training steps
    for step in range(1, total_steps + 1):
        step_context = CallbackContext(
            step=step,
            total_steps=total_steps,
            timestamp=float(step) * 0.1,
            metrics={"loss": 0.5, "reward": step * 0.1},
        )

        results = manager.trigger_event(CallbackEvent.STEP_END, step_context)
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        status = "✅" if failed == 0 else f"⚠️ ({failed} failed)"
        print(f"Step {step}: {successful}/{len(results)} succeeded {status}")

        if failed > 0 and step == 7:
            print("  Expected failure occurred - system continued gracefully")

    # Training end
    end_context = CallbackContext(
        step=total_steps, total_steps=total_steps, timestamp=1.5
    )
    results = manager.trigger_event(CallbackEvent.TRAINING_END, end_context)
    successful = sum(1 for r in results if r.success)
    print(f"Training end: {successful}/{len(results)} succeeded")

    # Show final statistics
    stats = manager.get_statistics()
    print("\nFinal Statistics:")
    print(f"  Total executions: {stats['total_executions']}")
    print(f"  Successful: {stats['successful_executions']}")
    print(f"  Failed: {stats['failed_executions']}")

    print("Error handling example completed!\n")


def custom_callback_example():
    """
    Custom callback implementation example.

    This example shows how to create custom callbacks for specific needs.
    """
    print("=== Custom Callback Example ===")

    from .modern_callback_system import BaseCallback, CallbackConfig, CallbackResult

    class EarlyStoppingCallback(BaseCallback):
        """Custom callback for early stopping based on reward threshold."""

        def __init__(self, reward_threshold: float, patience: int = 5):
            config = CallbackConfig(
                name="early_stopping",
                events=[CallbackEvent.STEP_END],
                priority=50,  # High priority to stop training early
            )
            super().__init__(config)
            self.reward_threshold = reward_threshold
            self.patience = patience
            self.best_reward = float("-inf")
            self.steps_without_improvement = 0
            self.should_stop = False

        def on_step_end(self, context):
            current_reward = context.metrics.get("episode_reward", 0)

            if current_reward > self.best_reward:
                self.best_reward = current_reward
                self.steps_without_improvement = 0
                self.logger.info(f"New best reward: {current_reward:.4f}")
            else:
                self.steps_without_improvement += 1

            if self.steps_without_improvement >= self.patience:
                self.should_stop = True
                self.logger.warning(
                    f"Early stopping triggered after {self.patience} steps without improvement. "
                    f"Best reward: {self.best_reward:.4f}"
                )

            return CallbackResult(success=True, data={"should_stop": self.should_stop})

    class LearningRateSchedulerCallback(BaseCallback):
        """Custom callback for learning rate scheduling."""

        def __init__(self, initial_lr: float, decay_factor: float, decay_steps: int):
            config = CallbackConfig(
                name="lr_scheduler", events=[CallbackEvent.STEP_END], priority=10
            )
            super().__init__(config)
            self.initial_lr = initial_lr
            self.decay_factor = decay_factor
            self.decay_steps = decay_steps

        def on_step_end(self, context):
            # Calculate new learning rate
            decay_count = context.step // self.decay_steps
            current_lr = self.initial_lr * (self.decay_factor**decay_count)

            self.logger.debug(f"Learning rate updated to: {current_lr:.6f}")
            # In a real implementation, this would update the optimizer's learning rate
            return CallbackResult(success=True, data={"learning_rate": current_lr})

    # Create manager with custom callbacks
    manager = CallbackManager()
    manager.register_callback(create_progress_callback(log_interval=10))
    manager.register_callback(EarlyStoppingCallback(reward_threshold=5.0, patience=3))
    manager.register_callback(
        LearningRateSchedulerCallback(
            initial_lr=0.001, decay_factor=0.9, decay_steps=10
        )
    )

    # Simulate training that should trigger early stopping
    total_steps = 25
    should_stop = False

    print("Running training with early stopping (threshold: 5.0, patience: 3)...")

    # Training start
    start_context = CallbackContext(step=0, total_steps=total_steps, timestamp=0.0)
    manager.trigger_event(CallbackEvent.TRAINING_START, start_context)

    # Training steps with rewards that eventually plateau
    for step in range(1, total_steps + 1):
        # Rewards increase then plateau
        if step <= 10:
            reward = step * 0.5  # Increasing: 0.5, 1.0, 1.5, ..., 5.0
        else:
            reward = 5.0  # Plateau

        step_context = CallbackContext(
            step=step,
            total_steps=total_steps,
            timestamp=float(step) * 0.1,
            metrics={"episode_reward": reward, "loss": 1.0 / (reward + 0.1)},
        )

        results = manager.trigger_event(CallbackEvent.STEP_END, step_context)

        # Check if early stopping was triggered
        for result in results:
            if result.data and result.data.get("should_stop"):
                should_stop = True
                print(f"Early stopping triggered at step {step}")
                break

        if should_stop:
            break

    print(
        f"Training completed at step {step} ({'early stopped' if should_stop else 'normal completion'})"
    )

    print("Custom callback example completed!\n")


def run_all_examples():
    """Run all usage examples."""
    setup_logging()

    print("Modern Callback System Usage Examples")
    print("=" * 50)
    print()

    # Run all examples
    basic_callback_usage_example()
    comprehensive_training_example()

    # Run async example
    asyncio.run(async_callback_example())

    error_handling_example()
    custom_callback_example()

    print("All examples completed successfully! 🎉")


if __name__ == "__main__":
    run_all_examples()
