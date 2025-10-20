# SAC v428 Multiple Optimization Runs Report

**Generated:** 2025-10-20 18:54:49
**Total Runs:** 2
**Trials per Run:** 2

## Best Overall Results
- **Best Score:** -999.0000
- **From Run:** 1

## Score Statistics Across Runs
- **Mean:** -999.0000
- **Standard Deviation:** 0.0000
- **Min:** -999.0000
- **Max:** -999.0000
- **Median:** -999.0000

## Parameter Consistency Analysis

| Parameter | Consistency | CV | Value Range |
|-----------|-------------|----|-------------|
| batch_size | low | .3f | .3f |
| tau | low | .3f | .3f |
| learning_rate | low | .3f | .3f |
| buffer_size | low | .3f | .3f |
| ent_coef | low | .3f | .3f |
| gamma | high | .3f | .3f |

## Individual Run Results

| Run | Best Score | Key Parameters |
|-----|------------|----------------|
| 1 | -999.0000 | .3f, 64, 500000, ... |
| 2 | -999.0000 | .3f, 512, 50000, ... |

## Recommendations

- **High Consistency:** Results are very consistent across runs. The optimization has likely converged to a stable solution.
- **Stable Parameters:** gamma show high consistency and are likely well-optimized.
- **Variable Parameters:** batch_size, tau, learning_rate, buffer_size, ent_coef show high variation. Consider focusing optimization efforts on these parameters.
