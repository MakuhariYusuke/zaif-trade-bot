# v457 Scripts

This directory contains scripts for the v457 trading strategy development, following the project standards.

## Scripts

### `train.py`
Standardized training script based on `scripts/v456/train_v456_production.py`.
It uses the `ztb.trading.environment.factory_v456.EnvironmentFactory` to ensure consistent 88-dimensional feature engineering and environment initialization as required by the v457 Reset Playbook.

**Features:**
- Uses `FastIntradayEnvV456` via Factory.
- Automatically computes Base, MTF, and Regime features.
- Supports dummy data generation for quick pipeline verification.
- Uses `config/v457/base/config.yaml`.

**Usage:**

```bash
# Run with real data (if available at data/btc_jpy_real_dataset.csv)
python scripts/v457/train.py --steps 1000

# Run with dummy data (for quick verification)
python scripts/v457/train.py --steps 200 --use_dummy_data

# Specify custom config or data
python scripts/v457/train.py --config config/v457/custom.yaml --data data/my_data.csv
```
