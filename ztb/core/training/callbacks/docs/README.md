# Modern Callback System

A comprehensive, event-driven callback system for training reinforcement learning agents. This system provides flexible, extensible callback functionality with async support, error handling, and comprehensive monitoring capabilities.

## Features

- **Event-Driven Architecture**: Trigger callbacks based on specific training events
- **Plugin-Based Design**: Easy registration and unregistration of callbacks
- **Async Support**: Asynchronous callback execution for non-blocking operations
- **Priority-Based Execution**: Control callback execution order
- **Comprehensive Error Handling**: Graceful failure handling with detailed error reporting
- **Metrics Collection**: Built-in metrics tracking and analysis
- **Statistics Tracking**: Monitor callback performance and execution statistics

## Architecture

### Core Components

1. **CallbackManager**: Central coordinator for callback registration and event triggering
2. **BaseCallback**: Abstract base class for all callback implementations
3. **CallbackEvent**: Enum defining training events (TRAINING_START, STEP_END, etc.)
4. **CallbackContext**: Data structure containing training state and metrics
5. **CallbackResult**: Standardized result format with success/failure status

### Event Types

- `TRAINING_START`: Fired when training begins
- `TRAINING_END`: Fired when training completes
- `STEP_END`: Fired after each training step
- `EPOCH_END`: Fired after each epoch (if applicable)
- `METRICS_UPDATE`: Fired when metrics are updated
- `ERROR_OCCURRED`: Fired when training errors occur

## Quick Start

```python
from ztb.training.callbacks import (
    CallbackManager, create_progress_callback,
    create_metrics_callback, CallbackEvent, CallbackContext
)

# Create callback manager
manager = CallbackManager()

# Create and register callbacks
progress_cb = create_progress_callback(log_interval=100)
metrics_cb = create_metrics_callback(collection_interval=50)

manager.register_callback(progress_cb)
manager.register_callback(metrics_cb)

# Use in training loop
for step in range(1, total_steps + 1):
    # ... training logic ...

    context = CallbackContext(
        step=step,
        total_steps=total_steps,
        metrics={"loss": current_loss, "reward": current_reward}
    )

    # Trigger callbacks
    results = manager.trigger_event(CallbackEvent.STEP_END, context)
```

## Available Callbacks

### ProgressCallback
Monitors training progress with ETA calculation and periodic logging.

```python
progress_cb = create_progress_callback(
    log_interval=100,      # Log every 100 steps
    show_eta=True         # Show estimated time of arrival
)
```

### CheckpointCallback
Automatically saves model checkpoints at specified intervals or when metrics improve.

```python
checkpoint_cb = create_checkpoint_callback(
    save_interval=1000,           # Save every 1000 steps
    save_path="./checkpoints",    # Checkpoint directory
    save_best_only=True,          # Only save when metrics improve
    best_metric="episode_reward"  # Metric to monitor
)
```

### MetricsCallback
Collects and logs training metrics with optional TensorBoard integration.

```python
metrics_cb = create_metrics_callback(
    collection_interval=50,   # Collect metrics every 50 steps
    log_interval=100,         # Log summary every 100 steps
    enable_tensorboard=True,  # Enable TensorBoard logging
    tensorboard_log_dir="./tensorboard"
)
```

### LoggingCallback
Provides enhanced logging for training events with configurable log levels.

```python
logging_cb = create_logging_callback(
    log_level="INFO",        # Logging level
    include_context=True     # Include training context in logs
)
```

## Advanced Usage

### Custom Callbacks

Create custom callbacks by inheriting from `BaseCallback`:

```python
from ztb.training.callbacks import BaseCallback, CallbackConfig, CallbackResult

class CustomCallback(BaseCallback):
    def __init__(self):
        config = CallbackConfig(
            name="custom",
            events=[CallbackEvent.STEP_END, CallbackEvent.TRAINING_END],
            priority=10
        )
        super().__init__(config)

    def on_step_end(self, context: str) -> CallbackResult:
        # Custom logic here
        if context.step % 500 == 0:
            print(f"Custom callback: Reached step {context.step}")
        return CallbackResult(success=True)

    def on_training_end(self, context) -> CallbackResult:
        print("Custom callback: Training completed!")
        return CallbackResult(success=True)
```

### Asynchronous Callbacks

For non-blocking operations, use async callbacks:

```python
import asyncio

class AsyncLoggingCallback(BaseCallback):
    async def on_step_end_async(self, context: str) -> CallbackResult:
        # Async logging operation
        await asyncio.sleep(0.01)  # Simulate I/O
        self.logger.info(f"Async log: Step {context.step}")
        return CallbackResult(success=True)

# Use with async triggering
results = await manager.trigger_event_async(CallbackEvent.STEP_END, context)
```

### Error Handling

The system gracefully handles callback failures:

```python
# Callbacks that fail don't stop training
results = manager.trigger_event(CallbackEvent.STEP_END, context)

# Check results
for result in results:
    if not result.success:
        print(f"Callback {result.callback_name} failed: {result.error}")
```

### Priority-Based Execution

Control callback execution order with priorities:

```python
# Higher priority numbers execute first
high_priority_cb = create_checkpoint_callback()
high_priority_cb.config.priority = 100  # High priority

low_priority_cb = create_logging_callback()
low_priority_cb.config.priority = 10   # Low priority
```

## Integration Examples

### With Stable Baselines3

```python
from stable_baselines3 import PPO
from ztb.training.callbacks import create_progress_callback, create_checkpoint_callback

# Create callbacks
progress_cb = create_progress_callback(log_interval=1000)
checkpoint_cb = create_checkpoint_callback(save_interval=5000)

# Create manager and register
manager = CallbackManager()
manager.register_callback(progress_cb)
manager.register_callback(checkpoint_cb)

# Custom SB3 callback that integrates with our system
class SB3IntegrationCallback(BaseCallback):
    def on_step_end(self, context):
        # Trigger our callback system
        manager.trigger_event(CallbackEvent.STEP_END, context)
        return CallbackResult(success=True)

# Use with SB3
model = PPO("MlpPolicy", "CartPole-v1")
model.learn(total_timesteps=10000, callback=SB3IntegrationCallback())
```

### With Custom Training Loops

```python
def train_agent(env, agent, total_steps):
    manager = CallbackManager()
    # ... register callbacks ...

    # Training start
    start_context = CallbackContext(step=0, total_steps=total_steps)
    manager.trigger_event(CallbackEvent.TRAINING_START, start_context)

    for step in range(1, total_steps + 1):
        # Training step
        obs = env.reset()
        action = agent.act(obs)
        next_obs, reward, done, info = env.step(action)

        agent.update(obs, action, reward, next_obs, done)

        # Create context with metrics
        context = CallbackContext(
            step=step,
            total_steps=total_steps,
            metrics={
                "loss": agent.get_loss(),
                "reward": reward,
                "episode_reward": info.get("episode_reward", 0)
            }
        )

        # Trigger callbacks
        results = manager.trigger_event(CallbackEvent.STEP_END, context)

        # Check for early stopping
        should_stop = any(
            result.data and result.data.get("should_stop", False)
            for result in results
        )
        if should_stop:
            break

    # Training end
    end_context = CallbackContext(step=step, total_steps=total_steps)
    manager.trigger_event(CallbackEvent.TRAINING_END, end_context)
```

## Testing

Run the comprehensive test suite:

```bash
# Unit tests
python -m pytest ztb/training/callbacks/test_callbacks.py -v

# Integration tests
python -m pytest ztb/training/callbacks/test_integration.py -v

# Run examples
python ztb/training/callbacks/examples.py
```

## Configuration

### Environment Variables

- `CALLBACK_LOG_LEVEL`: Set default logging level (default: INFO)
- `CALLBACK_MAX_WORKERS`: Maximum async workers (default: 4)

### Configuration Classes

Each callback type has its own configuration class for fine-tuned control:

```python
from ztb.training.callbacks import CheckpointCallbackConfig

config = CheckpointCallbackConfig(
    save_interval=2000,
    save_path="/custom/path",
    save_best_only=True,
    best_metric="custom_metric",
    max_checkpoints=10
)

callback = CheckpointCallback(config=config)
```

## Performance Considerations

- **Async Callbacks**: Use for I/O operations to avoid blocking training
- **Priority Ordering**: Critical callbacks (checkpointing) should have higher priority
- **Collection Intervals**: Balance monitoring frequency with performance impact
- **Error Handling**: Implement proper error handling to prevent training interruption

## Best Practices

1. **Start Simple**: Begin with basic progress and metrics callbacks
2. **Gradual Enhancement**: Add checkpointing and custom callbacks as needed
3. **Error Resilience**: Always handle callback failures gracefully
4. **Resource Management**: Clean up resources in `on_training_end`
5. **Testing**: Test callbacks independently before integration
6. **Monitoring**: Use the statistics API to monitor callback performance

## Troubleshooting

### Common Issues

**Callbacks not triggering:**
- Check event types match between registration and triggering
- Verify callback is properly registered with the manager

**Async callback errors:**
- Ensure proper async/await usage
- Check for blocking operations in async callbacks

**Performance impact:**
- Reduce collection/log intervals for high-frequency training
- Use async callbacks for I/O operations

**Memory usage:**
- Limit metrics history size in custom callbacks
- Clean up old checkpoints regularly

## Contributing

When adding new callback types:

1. Inherit from `BaseCallback`
2. Implement appropriate event handlers
3. Add configuration class if needed
4. Include comprehensive unit tests
5. Update this documentation

## License

This callback system is part of the ZAIF Trade Bot project.
