"""
Component Integration Guide

This document describes how to integrate the extracted components from the UnifiedTrainer
refactoring into other training classes for improved SOLID principles compliance.

## Available Components

### TrainingConfigManager
Location: ztb/training/unified_trainer/components/config_manager.py

**Purpose**: Centralized configuration processing and validation.

**Key Features**:
- Configuration loading and validation
- Default value application
- Type checking and conversion
- Environment variable overrides

**Integration Example**:
```python
from ztb.training.unified_trainer.components.config_manager import TrainingConfigManager

class MyTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config_manager = TrainingConfigManager()
        self.config = self.config_manager.process_config(config)
```

### TrainingUIManager
Location: ztb/training/unified_trainer/components/ui_manager.py

**Purpose**: User interface management for training progress display.

**Key Features**:
- Progress bar management
- Status display formatting
- UI state tracking
- Cross-platform compatibility

**Integration Example**:
```python
from ztb.training.unified_trainer.components.ui_manager import TrainingUIManager

class MyTrainer:
    def __init__(self):
        self.ui_manager = TrainingUIManager(self.logger)

    def train(self):
        with self.ui_manager.progress_bar(total=1000) as pbar:
            for step in range(1000):
                # training logic
                pbar.update(1)
```

### TrainingReporter
Location: ztb/training/unified_trainer/reporting.py

**Deprecated**: `ztb/training/unified_trainer/components/reporter.py` is a compatibility shim.
Use `ztb.training.unified_trainer.reporting.TrainingReporter` for all new code.

**Purpose**: Structured logging and reporting for training events.

**Key Features**:
- Structured logging with context
- Performance metrics reporting
- Error reporting with stack traces
- Training statistics aggregation

**Integration Example**:
```python
from ztb.training.unified_trainer.reporting import TrainingReporter

class MyTrainer:
    def __init__(self):
        self.reporter = TrainingReporter(self.logger)

    def on_training_complete(self, stats: Dict[str, Any]):
        self.reporter.log_training_complete(True, stats)
```

## Integration Patterns

### Pattern 1: Full Component Integration (Recommended)
Integrate all three components for maximum SOLID compliance:

```python
from ztb.training.unified_trainer.components.config_manager import TrainingConfigManager
from ztb.training.unified_trainer.components.ui_manager import TrainingUIManager
from ztb.training.unified_trainer.reporting import TrainingReporter

class MyTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger(__name__)

        # Initialize components
        self.config_manager = TrainingConfigManager()
        self.ui_manager = TrainingUIManager(self.logger)
        self.reporter = TrainingReporter(self.logger)

        # Process configuration
        self.config = self.config_manager.process_config(config)

    def train(self):
        self.reporter.log_training_start(
            self.config.get("training", {}).get("algorithm", "unknown"),
            self.config,
        )

        with self.ui_manager.progress_bar(total=self.config['total_timesteps']) as pbar:
            # Training loop
            for step in range(self.config['total_timesteps']):
                # training logic
                pbar.update(1)

        self.reporter.log_training_complete(True, {"final_step": step})
```

### Pattern 2: Selective Integration
Use only needed components:

```python
# Only use configuration management
self.config_manager = TrainingConfigManager()
self.config = self.config_manager.process_config(config)

# Only use reporting
self.reporter = TrainingReporter(self.logger)
```

## Benefits

1. **Single Responsibility**: Each component has one clear purpose
2. **Reusability**: Components can be used across different trainers
3. **Testability**: Components can be unit tested independently
4. **Maintainability**: Changes to UI/logging don't affect training logic
5. **Consistency**: Standardized interfaces across all trainers

## Migration Guide

### From God Object Classes
1. Identify configuration processing code → move to TrainingConfigManager
2. Identify UI/progress display code → move to TrainingUIManager
3. Identify logging/reporting code → move to TrainingReporter
4. Update class initialization to use components
5. Update method calls to use component interfaces

### Testing
Each component should have comprehensive unit tests:
- TrainingConfigManager: configuration validation, defaults, overrides
- TrainingUIManager: progress bar behavior, status display
- TrainingReporter: logging output, error handling, statistics

## Constants Integration

Constants have been consolidated in ztb/training/constants.py:
- Learning rates: DEFAULT_LEARNING_RATE_*
- Batch sizes: DEFAULT_BATCH_SIZE_*, BATCH_SIZE_*
- Import from training.constants instead of scattering across files

Example:
```python
from ztb.training.constants import DEFAULT_LEARNING_RATE_PPO, BATCH_SIZE_MEDIUM
```
"""
