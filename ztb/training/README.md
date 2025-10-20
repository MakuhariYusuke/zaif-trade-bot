# Training Engine Developer Guide

This guide covers the training components for reinforcement learning model development.

## Evaluation Gates

**Role**: Quality control gates that validate training success and trigger auto-halt when training fails.

**Key Components**:

- `ztb.training.eval_gates.EvalGates`: Gate evaluation logic
- `ztb.training.eval_gates.GateResult`: Individual gate check results
- Auto-halt functionality for failed training runs

**Gate Types**:

- **Resume Time**: Training resume speed validation
- **Memory Usage**: RSS memory limit enforcement
- **Duplicate Steps**: Global step uniqueness checks
- **Reward Trend**: Positive reward progression after 300k steps
- **Baseline Comparison**: Final evaluation above baseline threshold

**Auto-Halt Conditions**:

- **Critical Failures**: Memory limits exceeded, duplicate training steps
- **Consecutive Failures**: Too many failed gate checks in sequence
- **Persistent Issues**: Negative reward trends that don't improve

**Usage Example**:

```python
from ztb.training.eval_gates import EvalGates, GateStatus

gates = EvalGates(enabled=True)

# Run gate checks
results = gates.evaluate_all(
    rewards=[0.1, 0.15, 0.12],
    steps=[100000, 200000, 300000],
    final_eval_reward=0.14
)

# Check if training should halt
should_halt, reason = gates.should_halt_training(results)
if should_halt:
    print(f"Auto-halt triggered: {reason}")
```

## PPO Trainer

**Role**: Proximal Policy Optimization implementation with integrated evaluation gates and auto-halt.

**Key Components**:

- `ztb.training.ppo_trainer.PPOTrainer`: Main PPO training orchestrator
- Integration with evaluation gates for quality control
- Automatic checkpointing and resume functionality

**Features**:

- **Auto-Halt**: Automatically stops training when gates fail
- **Progress Tracking**: Monitors training metrics and gate status
- **Checkpoint Management**: Saves and loads training state
- **Callback Integration**: Custom halt callbacks for training orchestration

**Training Flow**:

1. Initialize trainer with evaluation gates
2. Start training session
3. Update progress periodically (triggers gate checks)
4. Auto-halt if critical failures detected
5. Save checkpoints for resume capability

**Usage Example**:

```python
from ztb.training.ppo_trainer import PPOTrainer
from ztb.training.eval_gates import EvalGates

def halt_callback(reason: str):
    print(f"Training halted: {reason}")
    # Send notification, cleanup resources, etc.

trainer = PPOTrainer(
    eval_gates=EvalGates(enabled=True),
    halt_callback=halt_callback,
    checkpoint_interval=10000
)

trainer.start_training()

# Training loop
for step in range(1000000):
    # ... training logic ...
    reward = train_step()

    trainer.update_progress(step, reward)

    if not trainer.is_training:
        break  # Auto-halt triggered

# Save final checkpoint
trainer.save_checkpoint("final_checkpoint.json")
```

## Curriculum Learning

**Role**: Progressive learning approach to address persistent action distribution bias in trading policies.

**Key Components**:

- `ztb.training.curriculum_learning`: P0→P2 staged learning implementation
- `ztb.training.unified_trainer.UnifiedTrainer._train_curriculum()`: Integration with unified training interface
- Progressive difficulty stages with balance enforcement

**Curriculum Stages**:

- **P0 (Forced Balance)**: High entropy exploration (ent_coef=0.8) with forced action balancing
- **P1 (Balanced Transition)**: Moderate entropy (ent_coef=0.6) with balance penalty reduction
- **P2 (Full Curriculum)**: Low entropy (ent_coef=0.4) with full reward structure

**Features**:

- **Bias Correction**: Addresses persistent BUY/SELL imbalance through staged learning
- **Balance Penalties**: Dynamic penalties for action distribution deviation from target
- **Progressive Difficulty**: Gradual reduction of constraints as learning advances
- **Regime Evaluation**: Automatic evaluation after each stage using regime_analysis

**Training Flow**:

1. P0: Train with forced balance constraints (50k steps)
2. Evaluate action distribution across market regimes
3. P1: Relax constraints with balance penalties (100k steps)
4. Evaluate and validate improvement
5. P2: Full curriculum with minimal constraints (200k steps)
6. Final evaluation and model saving

**Usage Example**:

```python
from ztb.training.unified_trainer import UnifiedTrainer

# Configure for curriculum learning
config = {
    "algorithm": "curriculum",
    "data_path": "ml-dataset-enhanced.csv",
    "curriculum_stage": "full",  # Will be overridden by stages
    "total_timesteps": 200000,   # Will be overridden by stages
}

trainer = UnifiedTrainer(config)
result = trainer.train()  # Runs P0→P2 automatically
```

**Configuration**:

```json
{
  "algorithm": "curriculum",
  "data_path": "ml-dataset-enhanced.csv",
  "model_dir": "models",
  "checkpoint_dir": "checkpoints"
}
```

## Run Seal

**Role**: Ensures deterministic training runs with environment tracking and seed management.

**Key Components**:

- `ztb.training.run_seal.RunSeal`: Reproducibility metadata container
- `ztb.training.run_seal.RunSealManager`: Seal creation, loading, and validation
- Environment snapshots with Python version, git commit, dependencies

**Features**:

- Automatic seed generation or manual seed setting
- Environment validation against saved seals
- Config and metadata storage
- JSON serialization for persistence

**Usage Example**:

```python
from ztb.training import create_run_seal, get_run_seal_manager

# Create run seal at training start
seal = create_run_seal(
    seed=42,
    config={"learning_rate": 0.001},
    metadata={"experiment": "baseline"}
)

# Later, validate environment
manager = get_run_seal_manager()
validation = manager.validate_environment(seal)
if not all(validation.values()):
    print("Environment mismatch - reproducibility not guaranteed")
```

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

## Integration with Core Systems

The training components integrate with:

- **Trading Environment**: RL environment for strategy learning
- **Data Pipeline**: Historical data replay for training
- **Risk Management**: Position sizing and loss limits during training
- **Monitoring**: Training metrics and gate status tracking

## Configuration

Training parameters are configured through:

- `ZTB_CHECKPOINT_INTERVAL`: Checkpoint frequency during training
- `ZTB_MAX_MEMORY_GB`: Memory limits for training processes
- `ZTB_MAX_CONSECUTIVE_FAILURES`: Auto-halt failure threshold
- Training hyperparameters in `trade-config.json`
