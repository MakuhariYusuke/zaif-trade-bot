
# SAC v437 Reproducibility Analysis Report

## Overview
This report analyzes the reproducibility of SAC v437 training across different random seeds.

## Training Results Summary
 seed  HOLD   BUY  SELL  final_reward
   42 0.347 0.316 0.337           2.0
   43 0.343 0.321 0.336           2.0

## Statistical Analysis

### Action Distribution Variability
- HOLD: Mean = 0.3450, Std = 0.0020, CV = 0.0058
- BUY: Mean = 0.3185, Std = 0.0025, CV = 0.0078
- SELL: Mean = 0.3365, Std = 0.0005, CV = 0.0015

### Final Reward Statistics
- Mean: 2.0000
- Standard Deviation: 0.0000
- Coefficient of Variation: 0.0000

### Reproducibility Assessment
- Overall Reproducibility Score: 0.0025

## Interpretation
- Coefficient of Variation (CV) measures relative variability
- Lower CV values indicate better reproducibility
- Reproducibility Score combines action and reward variability
- Score < 0.1: Excellent reproducibility
- Score 0.1-0.2: Good reproducibility
- Score > 0.2: Needs investigation

## Conclusion
✅ Excellent reproducibility achieved across seeds.
