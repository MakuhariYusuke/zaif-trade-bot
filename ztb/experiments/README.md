# Experiment Framework Documentation

## Overview
The unified experiment framework provides a standardized way to run, monitor, and report on machine learning experiments in the trading RL system.

## Key Components

### ExperimentBase Class
Located in `ztb/experiments/base.py`, this abstract base class provides:
- Standardized experiment execution flow
- Automatic result saving and notification
- Error handling and logging
- Resource monitoring capabilities

### ExperimentResult Dataclass
Standardized data structure for experiment results containing:
- `experiment_name`: Name of the experiment
- `timestamp`: ISO format timestamp
- `status`: "success", "failed", or "partial"
- `config`: Experiment configuration
- `metrics`: Performance metrics
- `artifacts`: File paths to saved artifacts
- `error_message`: Error details if failed
- `execution_time_seconds`: Total execution time

### DiscordNotifier Class
Enhanced notification system in `ztb/utils/notify/discord.py` with:
- `send_experiment_results()`: Specialized experiment result notifications
- Retry logic with exponential backoff
- Structured embed formatting for experiment data

## Creating a New Experiment

1. **Inherit from ExperimentBase**:
```python
from ztb.experiments.base import ExperimentBase, ExperimentResult

class MyExperiment(ExperimentBase):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "my_experiment_name")

    def run(self) -> ExperimentResult:
        # Your experiment logic here
        # Return ExperimentResult with metrics and artifacts
        pass
```

2. **Implement the `run()` method**:
   - Perform your experiment logic
   - Return an `ExperimentResult` object
   - Handle any experiment-specific errors

3. **Use the framework**:
```python
config = {"param1": "value1", "param2": 42}
experiment = MyExperiment(config)
result = experiment.execute()  # This handles saving, notifications, etc.
```

## Directory Structure
```
ztb/experiments/
├── base.py                 # ExperimentBase class and ExperimentResult
├── ml_reinforcement_1k.py  # Example implementation
└── [future experiments]

results/experiments/[experiment_name]/
├── [experiment_name]_YYYYMMDD_HHMMSS.json  # Detailed results
└── [experiment_name]_detailed_results.json # Step-by-step data

checkpoints/[experiment_name]/
└── [checkpoint files]
```

## Notification System
Experiments automatically send Discord notifications with:
- Experiment status (success/failed/partial)
- Key metrics summary
- Execution time
- Links to saved artifacts
- Error details (if applicable)

## Scaling Experiments
For large-scale experiments (100k/1M steps), use the `ScalingExperiment` class:
```python
from ztb.experiments.base import ScalingExperiment

class MyScalingExperiment(ScalingExperiment):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, total_steps=100000)
        # checkpoint_interval defaults to 10000
```

## Best Practices

1. **Configuration**: Pass all experiment parameters through the config dict
2. **Logging**: Use `self.logger` for consistent logging
3. **Error Handling**: Let the framework handle top-level errors
4. **Artifacts**: Save important outputs to `self.results_dir`
5. **Checkpoints**: Use `self.save_checkpoint()` for resumable experiments
6. **Metrics**: Include comprehensive metrics in the result

## Migration from Legacy Scripts

To migrate existing experiment scripts:

1. Create a new class inheriting from `ExperimentBase`
2. Move main logic to the `run()` method
3. Return `ExperimentResult` instead of custom return values
4. Remove manual saving/notification code
5. Update command-line interface to use the new class

## Example: MLReinforcement1KExperiment

See `ztb/experiments/ml_reinforcement_1k.py` for a complete example of:
- Step-by-step execution
- Memory monitoring
- Result aggregation
- Artifact saving
- Proper error handling