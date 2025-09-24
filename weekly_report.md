# Weekly Feature Evaluation Report

## Summary
- Total features evaluated: 7
- Valid results: 0/7
- Experimental features: 2
- Re-evaluation candidates: 0

## Delta Sharpe Results

| Feature | Mean | Std | CI95 Low | CI95 High | Status | Experimental |
|---------|------|-----|----------|-----------|--------|--------------|
| MFI | N/A | N/A | N/A | N/A | Insufficient Data |  |
| Lags | N/A | N/A | N/A | N/A | Insufficient Data |  |
| VWAP | N/A | N/A | N/A | N/A | Insufficient Data |  |
| Donchian | N/A | N/A | N/A | N/A | Insufficient Data |  |
| KalmanFilter | N/A | N/A | N/A | N/A | Insufficient Data |  |
| MovingAverages | N/A | N/A | N/A | N/A | Insufficient Data | ✓ |
| GradientSign | N/A | N/A | N/A | N/A | Insufficient Data | ✓ |

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