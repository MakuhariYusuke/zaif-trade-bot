# AI Code Review Request: v456 Trading System Analysis

## Project Overview

**Name**: ZAIF Trade Bot v456  
**Objective**: High-frequency intraday BTC-JPY trading using Deep Reinforcement Learning (SAC)  
**Status**: Post-training evaluation phase  
**Date**: 2026-01-14

---

## Current Performance Metrics (50-Episode Evaluation)

### Financial KPIs
| Metric | Value | Status |
|--------|-------|--------|
| Avg PnL | -10,100 JPY | ❌ Severe Loss |
| Win Rate | 0.0% | ❌ Complete Failure |
| Max Drawdown | -10.10% to -12.90% | ❌ Unacceptable |
| Final Balance Avg | 89,900 JPY | ❌ 10% Loss |
| Sharpe Ratio | -40.56 | ❌ Disastrous |

### Action Distribution
- BUY: 57.9% ⚠️ Over-aggressive long bias
- HOLD: 24.6%
- SELL: 17.5% ⚠️ Insufficient selling

### Episode Characteristics
- All episodes: exactly 500 steps (constant length)
- No early termination triggered (concerning)
- Negative rewards across all steps

---

## Architecture Overview

### Environment (FastIntradayEnvV456)
```python
# Current Configuration:
initial_balance = 100,000 JPY (changed from 124.01)
max_position = 0.01 BTC
drawdown_limit = 0.30 (10%)
max_steps = 500
fee_rate = 0.001 (0.1%)
slippage_rate = 0.0005 (0.05%)
```

### Feature Space
- **Base Features**: 30 columns (OHLCV indicators)
- **MTF Features**: 27 columns (multi-timeframe)
- **Regime Features**: 13 columns (market regime)
- **Challenge**: Features 31-70 are **randomly generated** (data not available)

### Model Architecture
- **Algorithm**: Stable-Baselines3 SAC (Soft Actor-Critic)
- **Checkpoint**: `models/week4_fixed/sac_fixed_v456_20260114_001812.zip`
- **Training**: 30,000 timesteps
- **Network**: MLP (fully connected)

### Data Issues
```python
# Current data pipeline problem:
data_path = 'data/btc_jpy_1m_v454.csv'
# Only contains real features, so:
for col_list in [base_cols, mtf_cols, regime_cols]:
    for col in col_list:
        if col not in df.columns:
            df[col] = np.random.randn(len(df))  # ← RANDOM NOISE
```

---

## Known Issues & Questions

### 1. **Fundamental Data Problem**
- Why is 60% of feature space synthetic/random?
- Should we use real technical indicators instead?
- Is the 1-minute resolution appropriate for HFT?

### 2. **Model Training Concerns**
- No validation set during training → overfitting risk
- Reward function may be misaligned with trading objectives
- Why does SAC produce such extreme action bias (57.9% BUY)?

### 3. **Environment Design**
- `max_steps=500` fixed episode length suggests no early termination
- Drawdown limit (30%) never triggered despite -10% avg loss
- Is the reward function penalizing losses sufficiently?

### 4. **Evaluation Methodology**
- 50 episodes on same data as training → severe data leakage
- No out-of-sample testing
- No walk-forward validation

---

## Specific Questions for AI Reviewer

### Priority 1: Critical Issues
1. **Feature Engineering**: How should we replace 40 synthetic features with real market data?
   - Suggestion: RSI, MACD, Bollinger Bands, ATR, Volume Profile, etc.?
   - Or should we reduce feature space to only statistically significant features?

2. **Reward Function**: Current reward structure produces negative returns across all episodes
   - Is the reward function properly scaled?
   - Should we implement separate trade signal validation?
   - Consider: Sharpe ratio, Sortino ratio, or trade-by-trade PnL?

3. **Data Leakage**: Evaluating on training data is invalid
   - How to implement proper train/val/test split for time-series?
   - Walk-forward validation strategy recommendation?

### Priority 2: Design Decisions
4. **Action Space Design**: Why does the model output 57.9% BUY actions?
   - Should we use discrete action space instead of continuous [-1, 1]?
   - Need action normalization or clipping adjustment?

5. **Hyperparameter Optimization**:
   - Current settings: `initial_balance=100k, max_position=0.01, max_steps=500`
   - Are these values mathematically justified?
   - Should we use learning rate scheduling?

6. **Model Architecture**:
   - Is MLP sufficient for this task, or do we need LSTM/Transformer?
   - Network size: hidden layers, units per layer?

### Priority 3: Debugging Steps
7. **Diagnostic Approach**:
   - How to instrument the code to detect where the model is failing?
   - What metrics should we log during evaluation?
   - Suggestion: action distribution histograms, balance trajectory heatmap, etc.?

8. **Baseline Comparison**:
   - Should we implement a simple rule-based trader (e.g., RSI > 70 = SELL)?
   - What constitutes acceptable performance for this market?

---

## Code Quality Observations

### Current Strengths
- ✅ Clean separation of concerns (environment, model, evaluation)
- ✅ Proper state management (balance tracking, drawdown calculation)
- ✅ Comprehensive metrics collection

### Potential Improvements
- ⚠️ Hard-coded feature column counts (30, 27, 13) → brittle
- ⚠️ Random feature generation as fallback → should fail explicitly
- ⚠️ No logging during model predictions (black box evaluation)
- ⚠️ Single-threaded evaluation (could parallelize episodes)

---

## Request Summary

Please review the following and provide recommendations:

1. **Diagnosis**: What is the most likely cause of the 100% loss rate?
   - Feature quality issue? Reward function? Environment design? Data leakage?

2. **Prioritized Action Plan**: Which 3 changes would have the highest impact?
   - Include estimated implementation complexity (low/medium/high)
   - Provide code snippets if applicable

3. **Alternative Approaches**: 
   - Consider completely different architectures (DQN, PPO, QRDQN)?
   - Simpler baselines (supervised learning, rule-based)?

4. **Testing Strategy**: How to validate improvements before full re-training?
   - Unit tests for environment?
   - Synthetic market simulations?

5. **Documentation Gaps**: What critical information is missing from the codebase?
   - Trading logic explanation?
   - Hyperparameter justification?

---

## Attachments

- Evaluation metrics: `results/week4_evaluation_metrics.json`
- Training config: `scripts/v456/train_mlp_v456_fixed.py`
- Environment code: `ztb/trading/environment/fast_intraday_env_v456.py`
- Data sample: `data/btc_jpy_1m_v454.csv` (first 100 rows)

---

**Requestor**: Development Team  
**Urgency**: High (project requires fundamental fixes)  
**Response Format**: Structured analysis with code recommendations
