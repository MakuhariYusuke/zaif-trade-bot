# Weekly Feature Evaluation Report

## Summary

- **Total Features**: 7
- **Valid Results**: 0
- **Experimental Features**: 2
- **Re-evaluate Candidates**: 0
- **Success Rate**: 0.0%

## Delta Sharpe Distribution



## Delta Sharpe Results

### Trend Features

| Feature | Mean | Std | CI95 Low | CI95 High | Status | Runs | NaN Rate |
|---------|------|-----|----------|-----------|--------|------|----------|
| Lags | N/A | N/A | N/A | N/A | Insufficient Data | 3 | N/A |
| Donchian | N/A | N/A | N/A | N/A | Insufficient Data | 3 | N/A |
| KalmanFilter | N/A | N/A | N/A | N/A | Insufficient Data | 3 | N/A |
| MovingAverages | N/A | N/A | N/A | N/A | Insufficient Data | 3 | 0.0101 |

### Volume Features

| Feature | Mean | Std | CI95 Low | CI95 High | Status | Runs | NaN Rate |
|---------|------|-----|----------|-----------|--------|------|----------|
| MFI | N/A | N/A | N/A | N/A | Insufficient Data | 3 | N/A |
| VWAP | N/A | N/A | N/A | N/A | Insufficient Data | 3 | N/A |

### Experimental Features

| Feature | Mean | Std | CI95 Low | CI95 High | Status | Runs | NaN Rate |
|---------|------|-----|----------|-----------|--------|------|----------|
| GradientSign | N/A | N/A | N/A | N/A | Insufficient Data | 3 | 0.0000 |


## Experimental Features

| Feature | Duration (ms) | NaN Rate | Columns | Delta Sharpe |
|---------|---------------|----------|---------|--------------|
| MovingAverages | 6.0 | 0.0101 | 8 | N/A |
| GradientSign | 1.0 | 0.0000 | 1 | N/A |

## Notes
- **Re-evaluate**: delta_sharpe.mean > 0.05 and CI95_low > 0
- **Monitor**: delta_sharpe.mean > 0.01
- **Maintain**: Otherwise
- Experimental features are marked with ✓