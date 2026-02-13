v450 configuration

This folder contains base configuration and templates for v450 experiments.

Files
- `base/config.yaml` - YAML baseline for quick tests and local use.
- `templates/sac_v450_template.json` - JSON template that can be copied and modified for experiments or for CI jobs.

Key differences from v449
- Adds `dynamic_threshold_mode` and `z_score` tunables
- Adds `regime_detection_config` with `use_relative` option
- Details `action_discovery` reward settings and sets `curriculum_stage` default to `action_discovery`.

Usage
- Copy and adapt `templates/sac_v450_template.json` into `experiments` configs or include it as baseline in CI.
