# Correlation Reduction Bug - Technical Debt

## Issue Summary
Paper trading with SAC model `sac_v420_hold_relaxed.zip` fails due to observation shape mismatch. The model expects (5,) observations but environment provides (21,).

## Background
- **Goal**: Verify `paper_trade.py` works with SAC model trained on reduced features (5 features)
- **Model**: `sac_v420_hold_relaxed.zip` expects 5-dimensional observations
- **Data**: `btc_jpy_featured_dataset.csv` has 21 features
- **Config**: `correlation_reduction: True` set for SAC compatibility

## Investigation History

### Initial Attempts
1. Fixed config key mismatch: `enable_correlation_reduction` → `correlation_reduction`
2. Updated `paper_trade.py` to set `correlation_reduction: True` for SAC models
3. Verified model loads correctly with (5,) space

### Root Cause Analysis
- **Code Location**: `ztb/trading/environment/heavy_env/mixins/initialization.py`
- **Issue**: Correlation reduction logic calls `FeatureRegistry.select_features_by_correlation()`
- **Problem**: This method selects from all registered features (66 features) based on analysis file, but environment uses only 21 features from CSV
- **Analysis File**: `reports/feature_analysis_20251003.json` contains correlations for features like 'sma_short', 'BB_Upper' etc.
- **CSV Features**: `btc_jpy_featured_dataset.csv` has features like 'sma_5', 'bb_upper' etc. (different naming)

### Current State
- Logs show "Applied correlation-based feature reduction" but no remaining count
- Environment still outputs 21 features
- No KeyError despite selecting features not in DataFrame
- Execution reaches observation creation but shape remains (21,)

### Unresolved Questions
1. **Why no KeyError?** `select_features_by_correlation` returns features not in `btc_jpy_featured_dataset.csv`, but `df[self.features]` doesn't fail
2. **Why logs show success?** "Applied correlation-based feature reduction" appears but `remaining` not logged
3. **Feature name mismatch**: Analysis file uses different naming convention than CSV

## Proposed Solutions

### Option 1: Fix select_features_by_correlation
- Modify to accept current features list and filter based on available features
- Update analysis file reading to match CSV feature names

### Option 2: Implement environment-specific reduction
- Add logic in initialization.py to filter features based on correlation within current dataset
- Use pandas correlation matrix directly instead of pre-computed analysis file

### Option 3: Update feature naming consistency
- Standardize feature names across training data, analysis files, and test datasets
- Ensure `btc_jpy_featured_dataset.csv` uses same names as training data

## Immediate Workaround
- Manually set `correlation_reduction: False` for SAC models until fixed
- Or hardcode feature selection to match model's expected 5 features

## Files Affected
- `ztb/trading/environment/heavy_env/mixins/initialization.py`
- `ztb/features/registry.py`
- `reports/feature_analysis_20251003.json`
- `data/btc_jpy_featured_dataset.csv`

## Priority
High - Blocks SAC model validation and paper trading functionality

## Next Steps
1. Debug why no KeyError occurs when selecting non-existent features
2. Implement Option 1 or 2 above
3. Test with SAC model after fix
4. Update feature naming conventions for consistency