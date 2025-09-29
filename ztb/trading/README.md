# Trading Engine Developer Guide

This guide covers the trading components in the `ztb/trading/` module, including reinforcement learning environments, PPO training, and checkpoint management.

## Checkpoint Manager

**Role**: Model and experiment state persistence with async operations, compression, and generation management.

**Key Classes/Functions**:

- `CheckpointManager`: Asynchronous checkpoint management
- `save_async()`, `load()`, `cleanup_old_generations()`

**Usage Example**:

```python
from ztb.utils import CheckpointManager

manager = CheckpointManager(base_path="checkpoints", max_generations=5)
await manager.save_async(model_state, "exp_001")
```

## PPO Trainer

**Role**: Proximal Policy Optimization implementation for trading strategy learning.

**Key Components**:

- `ppo_trainer.py`: Main PPO training implementation
- Environment integration with trading features
- Policy optimization with actor-critic architecture

**Training Features**:

- Asynchronous checkpointing during training
- Memory-efficient batch processing
- Configurable hyperparameters for different market conditions

## Trading Environment

**Role**: Reinforcement learning environment for trading strategy development.

**Key Classes**:

- `environment.py`: Trading environment implementation
- `bridge.py`: Interface between trading logic and RL framework

**Environment Features**:

- Realistic market simulation with historical data
- Reward functions based on Sharpe ratio and risk metrics
- Action space for position sizing and timing decisions

## Integration with Data Pipeline

The trading components integrate with the data pipeline for:

- Real-time feature computation
- Historical data replay for training
- Live trading execution with learned policies

## Configuration

Training and checkpoint parameters are configured through environment variables and config files:

- `ZTB_CHECKPOINT_INTERVAL`: Checkpoint frequency during training
- `ZTB_MAX_MEMORY_GB`: Memory limits for training
- Training hyperparameters in `trade-config.json`
