# Hyperparameter Optimization Framework

This directory contains a refactored hyperparameter optimization framework using object-oriented design with inheritance and abstraction.

## Architecture

### Base Classes

- **`HyperparameterOptimizer`**: Abstract base class providing common functionality for hyperparameter optimization
- **`TrainingCallback`**: Callback class for monitoring training progress and collecting statistics
- **`BinarySearchArgumentParser`**: Utility class for creating consistent command-line interfaces

### Key Features

- **Abstract Base Class**: `HyperparameterOptimizer` defines the interface for parameter optimization
- **Template Method Pattern**: Common training workflow with customizable parameter updates
- **Unified Interface**: All optimizers support both single tests and binary search optimization
- **Consistent Reporting**: Standardized output format for training results and action distributions

## Optimizer Classes

Each parameter has its own optimizer class inheriting from `HyperparameterOptimizer`:

### Available Optimizers

1. **`BatchSizeOptimizer`** (`batch_size_optimized.py`)
   - Parameter: `batch_size` (16-256)
   - Purpose: Optimize mini-batch size for PPO training

2. **`LearningRateOptimizer`** (`learning_rate_optimized.py`)
   - Parameter: `learning_rate` (1e-5 to 1e-2)
   - Purpose: Optimize learning rate for stable convergence

3. **`GammaOptimizer`** (`gamma_optimized.py`)
   - Parameter: `gamma` (0.8-0.99)
   - Purpose: Optimize discount factor for reward calculation

4. **`RewardParamsOptimizer`** (`reward_params_optimized.py`)
   - Parameter: `reward_multipliers` (0.1-5.0)
   - Purpose: Optimize reward function parameters for balanced action distribution

5. **`EntCoefOptimizer`** (`ent_coef_optimized.py`)
   - Parameter: `ent_coef` (0.001-0.1)
   - Purpose: Optimize entropy coefficient for exploration vs exploitation balance

6. **`VfCoefOptimizer`** (`vf_coef_optimized.py`)
   - Parameter: `vf_coef` (0.1-1.0)
   - Purpose: Optimize value function coefficient for value loss weighting

7. **`MaxGradNormOptimizer`** (`max_grad_norm_optimized.py`)
   - Parameter: `max_grad_norm` (0.1-10.0)
   - Purpose: Optimize gradient clipping threshold for stable training

8. **`GaeLambdaOptimizer`** (`gae_lambda_optimized.py`)
   - Parameter: `gae_lambda` (0.8-1.0)
   - Purpose: Optimize Generalized Advantage Estimation lambda parameter

9. **`ClipRangeOptimizer`** (`clip_range_optimized.py`)
   - Parameter: `clip_range` (0.1-0.5)
   - Purpose: Optimize PPO clipping range for policy updates

10. **`TargetKLOptimizer`** (`target_kl_optimized.py`)
    - Parameter: `target_kl` (0.001-0.1)
    - Purpose: Optimize target KL divergence for early stopping

11. **`NEpochsOptimizer`** (`n_epochs_optimized.py`)
    - Parameter: `n_epochs` (4-20)
    - Purpose: Optimize number of epochs per update

12. **`NStepsOptimizer`** (`n_steps_optimized.py`)
    - Parameter: `n_steps` (1024-4096)
    - Purpose: Optimize number of steps per environment interaction

13. **`NormalizeAdvantageOptimizer`** (`normalize_advantage_optimized.py`)
    - Parameter: `normalize_advantage` (True/False)
    - Purpose: Test whether to normalize advantages during training

## Usage

### Single Parameter Test

```bash
# Test specific batch_size value
python batch_size_optimized.py --mode single --batch_size 128 --timesteps 50000

# Test specific learning_rate value
python learning_rate_optimized.py --mode single --learning_rate 0.001 --timesteps 50000
```

### Binary Search Optimization

```bash
# Optimize batch_size with binary search
python batch_size_optimized.py --mode binary --max_iterations 8 --timesteps 100000

# Optimize learning_rate with binary search
python learning_rate_optimized.py --mode binary --max_iterations 8 --timesteps 100000
```

### Customizing for New Parameters

To add optimization for a new parameter:

1. Create a new class inheriting from `HyperparameterOptimizer`
2. Implement required abstract methods:
   - `parameter_name`: Return parameter name as string
   - `get_parameter_range()`: Return (min, max) tuple for search range
   - `update_ppo_params(value)`: Update PPO parameters with test value
3. Optionally override `evaluate_result()` for custom scoring logic

Example:

```python
class CustomParameterOptimizer(HyperparameterOptimizer):
    @property
    def parameter_name(self) -> str:
        return "custom_param"

    def get_parameter_range(self) -> tuple[float, float]:
        return (0.1, 10.0)

    def update_ppo_params(self, value: Union[int, float]) -> None:
        self.ppo_params["custom_param"] = float(value)
```

## Output

All optimizers provide consistent output including:

- Training statistics (average reward, standard deviation, best/worst episodes)
- Action distribution (HOLD/BUY/SELL percentages)
- Model save location
- Optimization progress (for binary search mode)

## Configuration

Default configurations can be modified in the base class:

- **Environment config**: `self.env_config`
- **PPO parameters**: `self.ppo_params`
- **Training timesteps**: Command line argument `--timesteps`

## Migration from Old Scripts

The old individual scripts (`batch_size.py`, `learning_rate.py`, etc.) are kept for reference but should be replaced with the new optimized versions. The new classes provide:

- Better code reuse
- Consistent interfaces
- Easier maintenance
- Extensibility for new parameters