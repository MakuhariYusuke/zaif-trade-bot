# Project Proposal v458: "Lost Alpha" Integration & Stabilization

## 1. Executive Summary

**v458** is the direct successor to the v457 "Lost Alpha" diagnostic phase. While v457.4 demonstrated that recovering features from v451/v456 can yield high performance (Seed 42), it suffered from "Bimodal Instability" (Seed 123/777 failed) and contained critical implementation flaws (MTF lookahead, unscaled rewards).

**The primary objective of v458 represents the "Correction & Stabilization" phase:**
To validate that the **v457.5 patches** (Causal MTF, Scaled Trend Guidance, Curriculum Decay) successfully eliminate the training instability, producing a model that consistently outperforms previous baselines without relying on lucky seeds or future leakage.

## 2. Background & Motivation

### The "Lost Alpha" Journey (v457)
- **Problem**: Recent versions (v455+) lost the specific "edge" that v451 had.
- **Diagnosis**: We identified missing features (Cyclical Time, Explicit Trend Guidance) and removed complex, ineffective additions.
- **Result (v457.4)**:
  - **Success**: One seed (42) achieved massive returns, proving the features *can* work.
  - **Failure**: Other seeds collapsed, suggesting the reward landscape was too harsh or confusing (bimodality).
- **Audit (Doc 36)**: Revealed critical bugs:
  1.  **MTF Leakage**: Future data leaked into the observation.
  2.  **Unscaled Penalty**: Trend penalty was too weak/inconsistent.
  3.  **Static Guidance**: No decay meant the agent fought the "training wheels" forever.

### The Solution (v458)
v458 uses the patched **`FastIntradayEnvV456` (v457.5)** which fixes these issues. We introduce **Trend Guidance Decay** (Curriculum Learning) to guide early exploration and then fade out, preventing the "agent vs. guidance" conflict in late training.

## 3. Technical Specification

### 3.1 Environment Configuration
- **Class**: `FastIntradayEnvV456` (Patched v457.5)
- **Key Parameters**:
  - `action_space_type`: "1d_position" (Continous -1 to +1)
  - `initial_balance`: 10,000,000 (standard)
  - `reward_scale`: 1.0 (no internal scaling applied)
  - `reward_clip`: null (no clipping)
  - **New parameter**: `guidance_decay_steps = 50,000` (Guidance fades linearly from 100% to 0% over first 50k lifetime steps).

### 3.2 Feature Set (88 Dimensions)
1.  **Base (30)**: Price action, volume, volatility (standard OHLCV + technical indicators).
2.  **MTF (27)**: 5m, 15m, 1h indicators (RSI, Bollinger, Trend). **CRITICAL FIX**: Index shifted to ensure strict causality (no lookahead).
3.  **Cyclical (6)**: Sin/Cos of Minute, Hour, DayOfWeek. (Restored from v451).
4.  **Global (6)**: Market context (spread, returns, volatility from Binance if available).
5.  **Regime (13)**: One-hot encoded market regime states.
6.  **Account (6)**: Position, TTL, cost basis, balance, PnL, step count.

### 3.3 Reward Function Updates
The reward function is standard PnL-based, but with a **Curriculum-Based Penalty**:
$$ R_{total} = R_{pnl} - (W_{decay} \times P_{trend}) $$
- $P_{trend}$: Penalty applied when action opposes the Ichimoku Cloud baseline.
- $W_{decay}$: Weight starting at 1.0 and decaying to 0.0 at step 50,000.
- **Fix**: $P_{trend}$ is now normalized effectively (approx -0.05 impact) regardless of JPY scale.
- **Scaling**: `reward_scale=1.0` (no internal scaling), `reward_clip=null` (no clipping).

## 4. Implementation Plan

### Phase 1: Verification (Immediate)
- [ ] Run `tests/verification/test_v458_features.py` (New test suite).
  - Verify MTF causality (ensure t does not see t+5m).
  - Verify Guidance Decay (check penalty reduces over steps).
  - Verify Reward Scaling (check magnitude of penalty vs PnL).

### Phase 2: Training Run (v458_main)
- **Algorithm**: SAC (Soft Actor-Critic)
- **Duration**: 2,000,000 Steps (Standard) or until convergence.
- **Seeds**: Run 3 seeds (e.g., 42, 123, 777) to confirm stability.
- **Success Criteria**:
  - All 3 seeds must show positive PnL (No collapse).
  - "Evaluation" episodes during training should show improvement *after* step 50k (post-guidance).
- **Implementation**: Leverage existing `scripts/v457/train.py` as base, create `scripts/v458/train_v458_main.py` with updated environment configuration.

### Phase 3: Analysis
- Compare v458 vs v457.4 (Seed 42) -> Did we lose performance by fixing the leak? (Expected: Yes, slight drop, but more realistic).
- Compare v458 vs v456 -> Did we regain the "Lost Alpha"?
- **Tools**: Use existing `scripts/v457/backtest.py` and analysis scripts for comparison.

## 5. Hypothesis
By **removing future leakage** (MTF fix), individual step performance will drop compared to the broken v457.4. However, by **correctly scaling trend guidance** and **decaying it**, the agent will learn a robust, transferable strategy that does not collapse on unseen data, resolving the Bimodal Instability.

---
**Status**: MVP Ready for Testing
**Author**: GitHub Copilot
**Date**: 2026-01-18
**Updates**: 2026-01-20 - Fixed all critical issues: training script, test suite, config, environment params, and documentation.
