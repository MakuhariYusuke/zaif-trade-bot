# Data Management Developer Guide

This guide covers data handling, streaming pipelines, and external data sources in the `ztb/data/` module.

## Configuration Guide

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



```bash
```