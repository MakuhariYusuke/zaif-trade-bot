#!/usr/bin/env python3
"""
Integration Tests for Modern Callback System.

This module provides integration tests that verify the callback system
works correctly in realistic training scenarios.
"""

import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ztb.training.callbacks.core.callback_implementations import (
    CheckpointCallback,
    LoggingCallback,
    MetricsCallback,
    ProgressCallback,
)
from ztb.training.callbacks.core.modern_callback_system import (
    CallbackContext,
    CallbackEvent,
    CallbackManager,
    CallbackPriority,
)


class TestCallbackSystemIntegration(unittest.TestCase):
    """Integration tests for the complete callback system."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.manager = CallbackManager()

        # Create callbacks
        self.progress_callback = ProgressCallback(log_interval=5)
        self.checkpoint_callback = CheckpointCallback(
            save_interval=10, save_path=str(self.temp_dir)
        )
        self.metrics_callback = MetricsCallback(collection_interval=2, log_interval=5)
        self.logging_callback = LoggingCallback()

        # Register all callbacks
        self.manager.register_callback(self.progress_callback)
        self.manager.register_callback(self.checkpoint_callback)
        self.manager.register_callback(self.metrics_callback)
        self.manager.register_callback(self.logging_callback)


    def test_full_training_simulation(self):
        """Test a complete training simulation with all callbacks."""
        total_steps = 20

        # Training start
        start_context = CallbackContext(
            step=0, total_steps=total_steps, total_epochs=1, timestamp=time.time()
        )

        with patch.object(
            self.progress_callback.logger, "info"
        ) as progress_log, patch.object(
            self.logging_callback.logger, "log"
        ) as logging_log:
            results = self.manager.trigger_event(
                CallbackEvent.TRAINING_START, start_context
            )

            # All callbacks should succeed
            self.assertTrue(all(result.success for result in results))

            # Progress callback should log training start
            progress_log.assert_any_call("🚀 Training started")
            progress_log.assert_any_call("Total steps: 20")

            # Logging callback should log training start
            logging_log.assert_any_call(20, "🚀 Training session started")

        # Simulate training steps
        for step in range(1, total_steps + 1):
            step_context = CallbackContext(
                step=step,
                total_steps=total_steps,
                timestamp=time.time(),
                metrics={
                    "loss": 1.0 / step,
                    "reward": step * 0.1,
                    "episode_reward": step * 0.5,
                },
            )

            with patch.object(
                self.progress_callback.logger, "info"
            ) as progress_log, patch.object(
                self.metrics_callback.logger, "info"
            ) as metrics_log:
                results = self.manager.trigger_event(
                    CallbackEvent.STEP_END, step_context
                )

                # All callbacks should succeed
                self.assertTrue(all(result.success for result in results))

                # Progress logging at intervals
                if step % 5 == 0:
                    progress_log.assert_called()

                # Metrics logging at intervals
                if step % 5 == 0:
                    metrics_log.assert_called()

                # Checkpoint saving at intervals
                if step % 10 == 0:
                    checkpoint_files = list(Path(self.temp_dir).glob("*.zip"))
                    self.assertTrue(len(checkpoint_files) > 0)

        # Training end
        end_context = CallbackContext(
            step=total_steps, total_steps=total_steps, timestamp=time.time()
        )

        with patch.object(
            self.progress_callback.logger, "info"
        ) as progress_log, patch.object(
            self.logging_callback.logger, "log"
        ) as logging_log, patch.object(
            self.metrics_callback.logger, "info"
        ) as metrics_log:
            results = self.manager.trigger_event(
                CallbackEvent.TRAINING_END, end_context
            )

            # All callbacks should succeed
            self.assertTrue(all(result.success for result in results))

            # Progress callback should log completion
            progress_log.assert_any_call("✅ Training completed")

            # Logging callback should log completion
            logging_log.assert_any_call(
                20, "✅ Training session completed (duration: 0.00s)"
            )

            # Metrics callback should log final summary
            metrics_log.assert_any_call("📊 Final Metrics Summary")

    def test_error_handling_integration(self):
        """Test error handling across the callback system."""
        # Create a failing callback
        failing_callback = MagicMock()
        failing_callback.config = MagicMock()
        failing_callback.config.name = "failing"
        failing_callback.config.events = [CallbackEvent.TRAINING_START]
        failing_callback.config.priority = CallbackPriority.NORMAL
        failing_callback.on_training_start.side_effect = Exception(
            "Integration test error"
        )

        self.manager.register_callback(failing_callback)

        context = CallbackContext(step=0, total_steps=100)

        results = self.manager.trigger_event(CallbackEvent.TRAINING_START, context)

        # Should have results for callbacks that listen to TRAINING_START
        # ProgressCallback, LoggingCallback, and failing_callback
        self.assertEqual(len(results), 3)  # 2 working + 1 failing

        # Some should succeed, one should fail
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        self.assertEqual(len(successful_results), 2)
        self.assertEqual(len(failed_results), 1)
        self.assertIn("Integration test error", str(failed_results[0].error))

    def test_callback_priority_execution_order(self):
        """Test that callbacks execute in correct priority order."""
        # Create callbacks with different priorities
        high_priority = MagicMock()
        high_priority.config = MagicMock()
        high_priority.config.name = "high"
        high_priority.config.events = [CallbackEvent.TRAINING_START]
        high_priority.config.priority = CallbackPriority.HIGH
        high_priority.on_training_start.return_value = MagicMock(success=True)

        low_priority = MagicMock()
        low_priority.config = MagicMock()
        low_priority.config.name = "low"
        low_priority.config.events = [CallbackEvent.TRAINING_START]
        low_priority.config.priority = CallbackPriority.LOW
        low_priority.on_training_start.return_value = MagicMock(success=True)

        # Register in reverse order to test priority sorting
        self.manager.register_callback(low_priority)
        self.manager.register_callback(high_priority)

        context = CallbackContext(step=0, total_steps=100)

        execution_order = []
        high_priority.on_training_start.side_effect = (
            lambda ctx: execution_order.append("high")
        )
        low_priority.on_training_start.side_effect = lambda ctx: execution_order.append(
            "low"
        )

        self.manager.trigger_event(CallbackEvent.TRAINING_START, context)

        # High priority should execute first
        self.assertEqual(execution_order, ["high", "low"])

    def test_metrics_collection_and_checkpointing(self):
        """Test integrated metrics collection and checkpointing."""
        # Simulate training with metrics that should trigger best model saving
        self.checkpoint_callback.checkpoint_config.save_best_only = True
        self.checkpoint_callback.checkpoint_config.best_metric = "reward"

        steps_and_rewards = [
            (5, 1.0),
            (10, 2.5),  # Should trigger checkpoint (best so far)
            (15, 2.0),  # Worse than best
            (20, 3.0),  # Should trigger checkpoint (new best)
        ]

        checkpoint_count = 0

        for step, reward in steps_and_rewards:
            context = CallbackContext(
                step=step, total_steps=20, metrics={"reward": reward}
            )

            results = self.manager.trigger_event(CallbackEvent.STEP_END, context)

            # Check if checkpoint was saved for best rewards
            if reward > 2.5:  # Only the 2.5 and 3.0 should trigger saves
                checkpoint_files = list(Path(self.temp_dir).glob("*.zip"))
                if len(checkpoint_files) > checkpoint_count:
                    checkpoint_count = len(checkpoint_files)

        # Should have saved checkpoints for the two best rewards
        final_checkpoint_files = list(Path(self.temp_dir).glob("*.zip"))
        self.assertGreaterEqual(len(final_checkpoint_files), 2)

    def test_async_callback_execution(self):
        """Test asynchronous callback execution."""
        import asyncio

        async def run_async_test():
            # Create an async callback
            async_callback = MagicMock()
            async_callback.config = MagicMock()
            async_callback.config.name = "async"
            async_callback.config.events = [CallbackEvent.TRAINING_START]
            async_callback.config.priority = CallbackPriority.NORMAL
            async_callback.config.async_enabled = True  # Enable async execution
            async_callback.config.enabled = True

            def sync_on_training_start(ctx):
                return MagicMock(success=True)

            async_callback.on_training_start = sync_on_training_start

            self.manager.register_callback(async_callback)

            context = CallbackContext(step=0, total_steps=100)

            # First trigger sync callbacks
            sync_results = self.manager.trigger_event(
                CallbackEvent.TRAINING_START, context
            )

            # Then trigger async callbacks
            async_results = await self.manager.trigger_event_async(
                CallbackEvent.TRAINING_START, context
            )

            # Should have results for callbacks that listen to TRAINING_START
            # ProgressCallback and LoggingCallback (sync), and async_callback (async)
            total_results = len(sync_results) + len(async_results)
            self.assertEqual(total_results, 3)
            # All should succeed
            all_results = sync_results + async_results
            self.assertTrue(all(result.success for result in all_results))

        asyncio.run(run_async_test())

    def test_callback_statistics_tracking(self):
        """Test that callback execution statistics are properly tracked."""
        # Run several events
        context = CallbackContext(step=1, total_steps=100)

        # Trigger multiple events
        for _ in range(3):
            self.manager.trigger_event(CallbackEvent.TRAINING_START, context)
            self.manager.trigger_event(CallbackEvent.STEP_END, context)

        stats = self.manager.get_statistics()

        # Should have tracked executions
        self.assertGreater(stats["total_callbacks"], 0)
        self.assertGreater(stats["successful_callbacks"], 0)
        self.assertEqual(stats["failed_callbacks"], 0)  # No failures in this test

    def test_callback_unregistration_during_execution(self):
        """Test unregistering callbacks during execution."""
        # Create a callback that unregisters itself
        self_unregistering = MagicMock()
        self_unregistering.config = MagicMock()
        self_unregistering.config.name = "self_unregistering"
        self_unregistering.config.events = [CallbackEvent.TRAINING_START]
        self_unregistering.config.priority = CallbackPriority.NORMAL
        self_unregistering.name = "self_unregistering"  # Set the name attribute

        def unregister_self(ctx):
            self.manager.unregister_callback("self_unregistering")
            return MagicMock(success=True)

        self_unregistering.on_training_start = unregister_self

        self.manager.register_callback(self_unregistering)

        context = CallbackContext(step=0, total_steps=100)

        # First execution should work (callback unregisters itself during execution)
        results1 = self.manager.trigger_event(CallbackEvent.TRAINING_START, context)
        # The unregistering callback may or may not be included in results depending on timing
        self.assertGreaterEqual(len(results1), 2)  # At least the other callbacks

        # Verify callback was unregistered during execution
        self.assertNotIn("self_unregistering", self.manager.list_callbacks())

        # Second execution should not include the unregistered callback
        results2 = self.manager.trigger_event(CallbackEvent.TRAINING_START, context)
        # Should have fewer results (callback was unregistered)
        self.assertEqual(len(results2), 2)  # Only the original 2 callbacks remain


class TestCallbackSystemEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = CallbackManager()

    def test_empty_callback_list(self):
        """Test triggering events with no callbacks registered."""
        context = CallbackContext(step=1, total_steps=100)

        results = self.manager.trigger_event(CallbackEvent.TRAINING_START, context)

        self.assertEqual(len(results), 0)

    def test_callback_with_no_matching_events(self):
        """Test callback that doesn't handle the triggered event."""
        callback = MagicMock()
        callback.config = MagicMock()
        callback.config.name = "no_match"
        callback.config.events = [
            CallbackEvent.TRAINING_END
        ]  # Only handles training end
        callback.config.priority = 0

        self.manager.register_callback(callback)

        context = CallbackContext(step=1, total_steps=100)

        # Trigger training start - callback shouldn't be called
        results = self.manager.trigger_event(CallbackEvent.TRAINING_START, context)

        # Should have no results (callback doesn't handle this event)
        self.assertEqual(len(results), 0)

    def test_callback_returning_none(self):
        """Test callback that returns None instead of CallbackResult."""
        callback = MagicMock()
        callback.config = MagicMock()
        callback.config.name = "returns_none"
        callback.config.events = [CallbackEvent.TRAINING_START]
        callback.config.priority = 0
        callback.on_training_start.return_value = None  # Returns None

        self.manager.register_callback(callback)

        context = CallbackContext(step=1, total_steps=100)

        results = self.manager.trigger_event(CallbackEvent.TRAINING_START, context)

        self.assertEqual(len(results), 1)
        # Should be treated as successful
        self.assertTrue(results[0].success)

    def test_callback_with_exception_in_result_creation(self):
        """Test callback that raises exception when creating result."""
        callback = MagicMock()
        callback.config = MagicMock()
        callback.config.name = "exception_in_result"
        callback.config.events = [CallbackEvent.TRAINING_START]
        callback.config.priority = 0

        def raise_exception(ctx):
            raise ValueError("Result creation failed")

        callback.on_training_start = raise_exception

        self.manager.register_callback(callback)

        context = CallbackContext(step=1, total_steps=100)

        results = self.manager.trigger_event(CallbackEvent.TRAINING_START, context)

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertIn("Result creation failed", str(results[0].error))

    def test_concurrent_callback_execution(self):
        """Test concurrent execution of callbacks."""
        import threading

        execution_order = []
        lock = threading.Lock()

        def create_threaded_callback(name):
            callback = MagicMock()
            callback.config = MagicMock()
            callback.config.name = name
            callback.config.events = [CallbackEvent.TRAINING_START]
            callback.config.priority = 0

            def threaded_execution(ctx):
                with lock:
                    execution_order.append(name)
                return MagicMock(success=True)

            callback.on_training_start = threaded_execution
            return callback

        # Register multiple callbacks
        for i in range(5):
            self.manager.register_callback(create_threaded_callback(f"callback_{i}"))

        context = CallbackContext(step=1, total_steps=100)

        results = self.manager.trigger_event(CallbackEvent.TRAINING_START, context)

        self.assertEqual(len(results), 5)
        self.assertEqual(len(execution_order), 5)
        # All callbacks should have executed
        self.assertTrue(all(result.success for result in results))
