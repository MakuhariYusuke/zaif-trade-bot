## AB Test Runner & Parameter Search

This document explains how to use `tools/ab_test_runner.py` and the new `tools/ab_param_search.py` for multi-seed AB tests and small hyperparameter grid search.

Usage: AB test runner

```cmd
python tools\ab_test_runner.py --configs "config\v447\a.json" "config\v447\b.json" --seeds 3
```

- `--configs`: one or more config file paths (use double quotes in cmd.exe)
- `--seeds`: number of seeds
- `--jobs`: number of parallel workers to use (optional)

Usage: Parameter search

```cmd
python tools\ab_param_search.py --template config\v447\sac_v447_1m_multiframe_config.json --grid config\ab_grid.json --seeds 3 --jobs 2 --objective balance
```

- `--template`: baseline config to copy from
- `--grid`: JSON file mapping dotted config keys to value lists
- `--seeds`: number of seeds for AB runs
- `--jobs`: parallel workers (use `--jobs 1` for sequential)
- `--objective`: `balance` (min |BUY - SELL|) or `min_sell` (minimize SELL fraction)
- `--method`: optimization method passed to `UnifiedOptimizer` (`grid`, `bayesian`, `random`)

PowerShell notes:

- On PowerShell, caret (^) is NOT a line continuation marker; use the backtick ` for multi-line commands or put the command on a single line.

- Example PowerShell single-line (recommended):

```powershell
python tools\ab_test_runner.py --configs "config\v447\sac_v447_1m_multiframe_entropy_lr_lower.json" "config\v447\sac_v447_1m_multiframe_balance_shaping.json" --seeds 3 --jobs 1
```

- Example PowerShell multi-line using backtick (no whitespace after backtick):

```powershell
python tools\ab_test_runner.py --configs `
	"config\v447\sac_v447_1m_multiframe_entropy_lr_lower.json" `
	"config\v447\sac_v447_1m_multiframe_balance_shaping.json" `
	--seeds 3 --jobs 1
```

- Example Bash (Linux/macOS):

```bash
python tools/ab_test_runner.py --configs \
	"config/v447/sac_v447_1m_multiframe_entropy_lr_lower.json" \
	"config/v447/sac_v447_1m_multiframe_balance_shaping.json" \
	--seeds 3 --jobs 1
```

Output: a JSON summary file indicating average action_distribution and objective score for each grid point.

Tips:
- For quick experiments use `--seeds 0` to validate configs generation and script flow without running training.
- Start with small grids and `--jobs` matching CPU/GPU resources to avoid contention.

Balance shaping example:

The repository includes a `config/v447/sac_v447_1m_multiframe_balance_shaping.json` example that enables new balance shaping and action entropy shaping options in `BehavioralPenaltyCalculator`.

To run AB comparisons including the shaping config:

```cmd
python tools\ab_test_runner.py --configs "config\v447\sac_v447_1m_multiframe_entropy_lr_lower.json" "config\v447\sac_v447_1m_multiframe_balance_shaping.json" --seeds 3 --jobs 1
```

Analyze `reports/training_report_*.json` and use `tools/analysis/action_distribution_window.py` to aggregate action distribution snapshots for steps of interest.

Notes on avoiding overcorrection:
- `balance_shaping` rewards actions that move the action distribution toward configured targets (see `behavior_optimization.action_balance_target`); it is symmetric and designed to avoid flipping bias from SELL to BUY when small values are used.
- Reduce `balance_shaping_value` if you see oscillation; prefer small non-zero values (e.g., 0.2–0.6) and evaluate across multiple seeds.
- Use `action_entropy_shaping` to increase diversity and prevent the agent from collapsing into a single action (e.g., always SELL).

Notes:
- `tools/ab_param_search.py` integrates with the `UnifiedOptimizer`'s `GridOptimizer` and saves results to the standard optimization history. This ensures consistency with `ztb.training.unified_optimizer` and reuses existing reporting/persistence.
- To use the legacy per-config execution (no optimizer), set `use_unified_optimizer=False` inside `tools/ab_param_search.py`.
