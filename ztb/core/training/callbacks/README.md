# Learning Callbacks System

A comprehensive callback system for machine learning training across multiple learning paradigms, providing performance optimization, monitoring, and distributed training capabilities.

## Overview

This system provides specialized callbacks for different learning types:

- **Reinforcement Learning**: SAC, PPO, DQN with adaptive algorithms
- **Supervised Learning**: Classification and regression with early stopping, LR scheduling
- **Unsupervised Learning**: Clustering and dimensionality reduction monitoring
- **Transfer Learning**: Domain adaptation and fine-tuning with interference detection
- **Multi-Task Learning**: Task balancing and shared representation monitoring
- **Meta Learning**: MAML and few-shot learning with adaptation tracking

## Features

### Core Capabilities
- **Performance Optimization**: Memory-efficient callbacks with LRU caching
- **Distributed Training**: Gradient synchronization and worker coordination
- **Adaptive Algorithms**: Dynamic parameter adjustment based on training progress
- **Comprehensive Monitoring**: Multi-metric tracking with statistical analysis
- **Error Isolation**: Robust error handling with graceful degradation

### Shared Base Classes
- `LearningCallback`: Abstract base class for all callbacks
- `MemoryOptimizedCallback`: Memory-efficient callback with caching
- `MetricsCallback`: Standardized metrics collection
- `AdaptiveCallback`: Self-tuning callback parameters
- `CallbackManager`: Unified callback orchestration

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from ztb.training.callbacks import CallbackManager
from ztb.training.callbacks.reinforcement.sac import SACTemperatureScheduler
from ztb.training.callbacks.supervised import EarlyStoppingCallback

# Create callback manager
manager = CallbackManager()

# Add callbacks
manager.add_callback(SACTemperatureScheduler())
manager.add_callback(EarlyStoppingCallback(patience=10))

# Use in training loop
for epoch in range(num_epochs):
    # Training code...

    # Execute callbacks
    context = LearningContext(epoch=epoch, total_epochs=num_epochs)
    manager.on_epoch_end(context, logs)
```

### Reinforcement Learning (SAC)

```python
from ztb.training.callbacks.reinforcement.sac import (
    SACTemperatureScheduler,
    SACValueFunctionMonitor,
    SACTargetNetworkUpdater,
    SACExplorationMonitor
)

# SAC-specific callbacks
callbacks = [
    SACTemperatureScheduler(initial_temp=1.0, final_temp=0.1),
    SACValueFunctionMonitor(monitor_frequency=5),
    SACTargetNetworkUpdater(update_frequency=2),
    SACExplorationMonitor()
]

manager = CallbackManager()
for callback in callbacks:
    manager.add_callback(callback)
```

### Supervised Learning

```python
from ztb.training.callbacks.supervised import (
    EarlyStoppingCallback,
    LearningRateSchedulerCallback,
    ClassificationMetricsCallback,
    RegressionMetricsCallback
)

# Supervised learning callbacks
callbacks = [
    EarlyStoppingCallback(patience=15, min_delta=0.001),
    LearningRateSchedulerCallback(schedule_type='cosine'),
    ClassificationMetricsCallback()  # or RegressionMetricsCallback()
]

manager = CallbackManager()
for callback in callbacks:
    manager.add_callback(callback)
```

### Unsupervised Learning

```python
from ztb.training.callbacks.unsupervised import (
    ClusteringMetricsCallback,
    DimensionalityReductionMetricsCallback,
    EmbeddingQualityCallback,
    ConvergenceMonitorCallback
)

# Unsupervised learning callbacks
callbacks = [
    ClusteringMetricsCallback(compute_frequency=1),
    DimensionalityReductionMetricsCallback(),
    EmbeddingQualityCallback(assessment_tasks=['clustering', 'neighborhood']),
    ConvergenceMonitorCallback(patience=10)
]

manager = CallbackManager()
for callback in callbacks:
    manager.add_callback(callback)
```

### Transfer Learning

```python
from ztb.training.callbacks.transfer import (
    DomainAdaptationCallback,
    FineTuningCallback,
    TransferPerformanceCallback
)

# Transfer learning callbacks
callbacks = [
    DomainAdaptationCallback(adaptation_method='dann'),
    FineTuningCallback(freeze_layers=['conv1', 'conv2']),
    TransferPerformanceCallback(evaluation_metrics=['accuracy', 'f1'])
]

manager = CallbackManager()
for callback in callbacks:
    manager.add_callback(callback)
```

### Multi-Task Learning

```python
from ztb.training.callbacks.multi_task import (
    TaskBalancingCallback,
    SharedRepresentationCallback,
    TaskInterferenceCallback
)

task_names = ['classification', 'regression', 'segmentation']

# Multi-task learning callbacks
callbacks = [
    TaskBalancingCallback(task_names, balance_threshold=0.2),
    SharedRepresentationCallback(representation_layers=['shared_encoder']),
    TaskInterferenceCallback(task_names, interference_threshold=-0.1)
]

manager = CallbackManager()
for callback in callbacks:
    manager.add_callback(callback)
```

### Meta Learning

```python
from ztb.training.callbacks.meta import (
    MAMLCallback,
    FewShotCallback,
    MetaAdaptationCallback
)

# Meta learning callbacks
callbacks = [
    MAMLCallback(num_inner_steps=5, adaptation_lr=0.01),
    FewShotCallback(n_way=5, k_shot=1),
    MetaAdaptationCallback(adaptation_steps=10)
]

manager = CallbackManager()
for callback in callbacks:
    manager.add_callback(callback)
```

## Advanced Usage

### Custom Callbacks

```python
from ztb.training.callbacks.shared.base import LearningCallback, LearningContext

class CustomCallback(LearningCallback):
    def __init__(self, custom_param: float = 1.0) -> None:
        super().__init__()
        self.custom_param = custom_param
        self.history = []

    def on_epoch_end(self, context: LearningContext, logs=None) -> None:
        if logs and 'custom_metric' in logs:
            self.history.append(logs['custom_metric'] * self.custom_param)

    def get_custom_stats(self) -> Dict[str, float]:
        return {
            'mean': np.mean(self.history) if self.history else 0,
            'latest': self.history[-1] if self.history else 0
        }
```

### Memory Optimization

```python
from ztb.training.callbacks.shared.base import MemoryOptimizedCallback

class OptimizedCallback(MemoryOptimizedCallback):
    def __init__(self, cache_size: int = 1000) -> None:
        super().__init__(cache_size=cache_size)

    def on_epoch_end(self, context: LearningContext, logs=None) -> None:
        # Cache expensive computations
        key = f"epoch_{context.epoch}"
        result = self.cache_metrics(key, self._expensive_computation(logs))

    def _expensive_computation(self, logs: Dict[str, Any]) -> Dict[str, float]:
        # Expensive computation logic
        return {'result': logs.get('value', 0) * 2}
```

### Callback Configuration

```python
# Configure callbacks from config file
import json

with open('callback_config.json', 'r') as f:
    config = json.load(f)

# Create callbacks based on configuration
callbacks = []
for cb_config in config['callbacks']:
    callback_class = globals()[cb_config['class']]
    callback = callback_class(**cb_config.get('params', {}))
    callbacks.append(callback)
```

## Configuration Files

### callback_config.json
```json
{
  "callbacks": [
    {
      "class": "EarlyStoppingCallback",
      "params": {
        "patience": 20,
        "min_delta": 0.001
      }
    },
    {
      "class": "LearningRateSchedulerCallback",
      "params": {
        "schedule_type": "cosine",
        "initial_lr": 0.001
      }
    },
    {
      "class": "ClassificationMetricsCallback",
      "params": {
        "compute_frequency": 1
      }
    }
  ]
}
```

## Testing

Run the comprehensive test suite:

```bash
python -m pytest ztb/training/callbacks/tests/test_learning_callbacks.py -v
```

Or run specific test categories:

```bash
# Test only reinforcement learning callbacks
python -m unittest test_learning_callbacks.TestReinforcementLearningCallbacks

# Test only supervised learning callbacks
python -m unittest test_learning_callbacks.TestSupervisedLearningCallbacks
```

## Performance Benchmarks

### Memory Usage
- Base callbacks: < 10MB per 1000 epochs
- Memory-optimized callbacks: < 50MB for large datasets
- Distributed callbacks: Scales linearly with worker count

### Execution Time
- Simple callbacks: < 1ms per epoch
- Complex metrics: < 100ms per epoch
- Clustering analysis: < 500ms for 1000 samples

## Integration Examples

### PyTorch Lightning Integration

```python
import pytorch_lightning as pl
from ztb.training.callbacks import CallbackManager

class LightningModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.callback_manager = CallbackManager()
        # Add callbacks...

    def training_epoch_end(self, outputs: str) -> None:
        context = LearningContext(
            epoch=self.current_epoch,
            total_epochs=self.trainer.max_epochs
        )
        logs = {'loss': torch.stack([x['loss'] for x in outputs]).mean()}
        self.callback_manager.on_epoch_end(context, logs)
```

### TensorFlow/Keras Integration

```python
import tensorflow as tf
from ztb.training.callbacks import CallbackManager

class KerasCallback(tf.keras.callbacks.Callback):
    def __init__(self) -> None :
        super().__init__()
        self.callback_manager = CallbackManager()
        # Add callbacks...

    def on_epoch_end(self, epoch, logs=None) -> None:
        context = LearningContext(epoch=epoch, total_epochs=self.params['epochs'])
        self.callback_manager.on_epoch_end(context, logs)
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce cache size or increase compute frequency
   ```python
   callback = ClusteringMetricsCallback(max_samples=1000)
   ```

2. **Slow Performance**: Increase compute frequency or use sampling
   ```python
   callback = EmbeddingQualityCallback(compute_frequency=10)
   ```

3. **Missing Metrics**: Ensure correct log keys are provided
   ```python
   logs = {
       'embeddings': embeddings,
       'cluster_labels': labels  # Required for clustering
   }
   ```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Callbacks will now log detailed information
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new callbacks
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this callback system in your research, please cite:

```bibtex
@misc{learning-callbacks,
  title={Learning Callbacks: A Comprehensive Callback System for Machine Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/learning-callbacks}
}
```
