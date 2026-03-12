# v457 Configuration

This folder structure resets the configuration management to the clean style of v450, while injecting the high-performance parameters from v451.

## Structure

- **`base/config.yaml`**: The **Single Source of Truth** for the v457 baseline.
    - Contains the "Legacy Assets" (Gamma=0.8, No Hold Penalty).
    - Used for reproducibility and as a parent for experiments.
- **`features.yaml`**: Explicit definition of the feature set. No more hidden "v451" strings.
- **`experiments/`**: Place for specific hypothesis testing configs.
    - Do NOT overwrite `base/config.yaml`. Create a new file here instead.
- **`templates/`**: JSON templates for automated tools.

## Key Parameters (The "Alpha")

Derived from `sac_v451_optimized.json`:
- **Gamma**: `0.80` (Focus on immediate PnL)
- **Hold Penalty**: `0.0` (Patience is allowed)
- **Loss Multiplier**: `1.2` (Asymmetric risk aversion)
