# Configuration Guide

## Core Configuration Classes

### ZTBConfig
Central configuration management for all system components.

```python
from ztb.utils.config import ZTBConfig

config = ZTBConfig()
mem_profile = config.get('ZTB_MEM_PROFILE', False)
cuda_warn = config.get('ZTB_CUDA_WARN_GB', 0.0)
```

### Environment Variable Categories

1. **Observability**: `ZTB_MEM_PROFILE`, `ZTB_CUDA_WARN_GB`, `ZTB_LOG_LEVEL`
2. **Training**: `ZTB_CHECKPOINT_INTERVAL`, `ZTB_MAX_MEMORY_GB`
3. **Testing**: `ZTB_TEST_ISOLATION`, `ZTB_FLOAT_TOLERANCE`

## Configuration File Examples

### YAML Configuration File

```yaml
# config/trade-config.yaml
observability:
  mem_profile: true
  cuda_warn_gb: 8.0
  log_level: "INFO"

training:
  checkpoint_interval: 1000
  max_memory_gb: 16.0

testing:
  isolation: true
  float_tolerance: 0.01
```

### JSON Configuration File

```json
{
  "observability": {
    "mem_profile": true,
    "cuda_warn_gb": 8.0,
    "log_level": "INFO"
  },
  "training": {
    "checkpoint_interval": 1000,
    "max_memory_gb": 16.0
  },
  "testing": {
    "isolation": true,
    "float_tolerance": 0.01
  }
}
```

## Environment Variables vs Configuration Files

| Aspect | Environment Variables | Configuration Files |
|--------|----------------------|-------------------|
| **Priority** | High (overrides files) | Low (fallback) |
| **Use Case** | Secrets, runtime overrides | Default settings, complex configs |
| **Format** | String only | YAML/JSON (structured) |
| **Validation** | Type conversion with fallbacks | Schema validation |
| **Examples** | `ZTB_MEM_PROFILE=1` | `mem_profile: true` |

### Usage Patterns

**Development (Environment Variables):**

```bash
export ZTB_MEM_PROFILE=1
export ZTB_CUDA_WARN_GB=4.0
export ZTB_LOG_LEVEL=DEBUG
```

**Production (Configuration File):**

```yaml
# Load via ZTB_CONFIG_FILE=/path/to/config.yaml
observability:
  mem_profile: true
  cuda_warn_gb: 8.0
  log_level: "WARN"
```

**Hybrid Approach:**

```bash
# Base config from file
export ZTB_CONFIG_FILE=config/prod.yaml
# Runtime overrides
export ZTB_LOG_LEVEL=DEBUG
```

## Pydantic-Based Configuration Schema

The system now uses Pydantic models for type-safe configuration management.

### Schema Components

- **TrainingConfig**: Training parameters (timesteps, environments, learning rate)
- **CheckpointConfig**: Checkpoint management (compression, retention, async saving)
- **StreamingConfig**: Data streaming settings (batch size, buffer policy)
- **EvalConfig**: Evaluation parameters (DSR trials, bootstrap resampling)
- **GlobalConfig**: Container for all configuration sections

### Loading Configuration

```python
from ztb.config.loader import load_config

# Load from YAML file
config = load_config(config_path="config/example.yaml")

# Access typed configuration
print(f"Training timesteps: {config.training.total_timesteps}")
print(f"Checkpoint compression: {config.checkpoint.compress}")
```

### Effective Configuration Dump

To see the merged configuration from all sources:

```bash
python scripts/print_effective_config.py --config config/example.yaml
```

This shows the final configuration after applying priority: CLI > ENV > YAML > defaults.

### JSON Schema Export

The configuration schema can be exported to JSON:

```python
from ztb.config.schema import GlobalConfig
import json

schema = GlobalConfig.model_json_schema()
with open('schema/config_schema.json', 'w') as f:
    json.dump(schema, f, indent=2)
```

Or use the convenience script:

```bash
python scripts/dump_config_schema.py
```

## Default Configuration Values

### Trading Defaults

| Component | Parameter | Default Value | Description |
|-----------|-----------|---------------|-------------|
| **Symbol** | Trading pair | `BTC_JPY` | Primary trading instrument |
| **Venue** | Exchange | `coincheck` | Default trading venue |
| **Risk Profile** | Risk management | `aggressive` | Default risk profile (conservative/balanced/aggressive) |

### Risk Profile Settings (Aggressive)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Max Position Size | 1.0 | Maximum position size multiplier |
| Stop Loss % | 5.0% | Stop loss threshold |
| Take Profit % | 10.0% | Take profit threshold |
| Max Daily Loss % | 20.0% | Maximum daily loss limit |
| Circuit Breaker % | 15.0% | Circuit breaker threshold |

### Training Defaults

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| Total Timesteps | 1,000,000 | Total training steps |
| Environments | 4 | Number of parallel environments |
| Eval Frequency | 10,000 | Evaluation interval |
| Learning Rate | 3e-4 | PPO learning rate |
| Batch Size | 64 | Training batch size |
| Gamma | 0.99 | Discount factor |
| GAE Lambda | 0.95 | GAE lambda parameter |
| Clip Range | 0.2 | PPO clip range |
| Seed | 42 | Random seed |
