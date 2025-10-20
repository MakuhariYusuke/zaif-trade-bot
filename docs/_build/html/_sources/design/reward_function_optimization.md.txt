# Reward Function Structure Optimization

This document describes the reward function structure optimization framework for the Zaif Trade Bot. This framework enables systematic optimization of reward function parameters beyond simple scaling, allowing for data-driven reward design.

## Overview

The reward function optimization framework consists of three main components:

1. **RewardFunctionOptimizer**: Handles the optimization process using Optuna
2. **RewardFunctionEvaluator**: Evaluates parameter performance across market conditions
3. **Optimization Scripts**: Command-line tools for running optimizations

## Key Features

- **Multi-objective optimization**: Optimize for profit, risk, consistency, and other metrics simultaneously
- **Cross-market validation**: Test parameters across bull, bear, sideways, and volatile market conditions
- **Automated parameter tuning**: Systematic exploration of reward function parameter spaces
- **Comprehensive reporting**: Detailed reports with optimization history and recommendations

## Supported Reward Stages

The framework supports optimization for the following reward function stages:

- `balanced_transition`: Early training stage balancing exploration and exploitation
- `trading_focused`: Mid-training stage emphasizing trading activity
- `profit_optimized`: Late training stage focused on profitability
- `ultra_profit`: Aggressive profit maximization stage
- `pnl_focused`: Pure P&L optimization stage

## Quick Start

### Basic Optimization

```bash
# Optimize balanced_transition stage with default settings
python optimize_reward_function.py --stage balanced_transition

# Optimize with custom number of trials
python optimize_reward_function.py --stage profit_optimized --n-trials 200

# Optimize multiple objectives
python optimize_reward_function.py --stage trading_focused \
    --objectives profit sharpe win_rate consistency max_drawdown
```

### Advanced Usage

```bash
# Custom evaluation settings
python optimize_reward_function.py --stage ultra_profit \
    --n-trials 150 \
    --evaluation-episodes 20 \
    --max-steps 2000 \
    --output-dir custom_results \
    --verbose
```

## Configuration

The framework uses a JSON configuration file (`configs/reward_optimization.json`) to define:

- Parameter spaces for each reward stage
- Evaluation settings (episodes, market conditions, etc.)
- Optimization constraints and objectives
- Reporting preferences

### Parameter Spaces

Each reward stage has its own parameter space definition. For example, the `balanced_transition` stage includes:

```json
{
  "balance_penalty_tolerance": {
    "type": "float",
    "low": 0.01,
    "high": 0.2,
    "log_scale": false,
    "description": "Tolerance for balance deviation before penalty"
  },
  "balance_penalty": {
    "type": "float",
    "low": 1.0,
    "high": 20.0,
    "log_scale": false,
    "description": "Penalty multiplier for balance deviation"
  }
}
```

## Evaluation Metrics

The framework evaluates parameters using comprehensive metrics:

- **Profit Metrics**: Total return, profit factor, recovery factor
- **Risk Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio, maximum drawdown, volatility
- **Performance Metrics**: Win rate, consistency score
- **Market Adaptation**: Performance across different market conditions

## Output Files

Optimization generates several output files:

- `reward_optimization_{stage}_result.json`: Complete optimization results
- `reward_optimization_{stage}_report.md`: Human-readable optimization report
- `reward_optimization_summary.json`: Summary for multi-stage optimizations

## Testing

Run the test suite to verify the framework:

```bash
python test_reward_optimization.py
```

## Integration with Training

After optimization, integrate the best parameters back into your reward function:

1. Load the optimization results
2. Extract the best parameters for your target stage
3. Update the RewardSettings with optimized parameters
4. Retrain your model with the optimized reward function

## Best Practices

### Parameter Selection
- Start with `balanced_transition` for stable optimization
- Use 100-200 trials for initial optimization
- Include multiple objectives to avoid overfitting to single metrics

### Evaluation Settings
- Use 10+ evaluation episodes for robust results
- Test across all market conditions for generalization
- Increase max_steps for more comprehensive evaluation

### Validation
- Always validate optimized parameters on held-out data
- Monitor for overfitting to optimization objectives
- Consider ensemble approaches combining multiple optimized configurations

## Troubleshooting

### Common Issues

1. **Optuna not installed**: Install with `pip install optuna`
2. **Memory issues**: Reduce evaluation episodes or max_steps
3. **Poor convergence**: Increase number of trials or adjust parameter bounds
4. **Unrealistic results**: Check evaluation function implementation

### Performance Optimization

- Use parallel evaluation when possible
- Cache evaluation results for repeated parameter sets
- Use early stopping for faster convergence

## API Reference

### RewardFunctionOptimizer

```python
from ztb.optimization.reward_function_optimizer import RewardFunctionOptimizer

optimizer = RewardFunctionOptimizer()
result = optimizer.optimize_reward_function(
    stage="balanced_transition",
    evaluation_function=my_eval_func,
    n_trials=100,
    objectives=["profit", "sharpe", "win_rate"]
)
```

### RewardFunctionEvaluator

```python
from ztb.optimization.reward_function_evaluator import RewardFunctionEvaluator

evaluator = RewardFunctionEvaluator()
eval_func = evaluator.create_evaluation_function("profit_optimized")
scores = eval_func(parameters)
```

## Future Enhancements

- Pareto front optimization for multi-objective problems
- Neural reward function architectures
- Meta-learning for reward function adaptation
- Bayesian optimization integration
- Automated curriculum learning optimization
