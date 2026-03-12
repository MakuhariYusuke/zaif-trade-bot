# v457.4 Seed Instability & Curriculum-Guided Trend Following

## 1. Problem Analysis
In v457.4 (Native 1D Action) experiments, we observed a **Bimodal Outcome**:
- **Seed 42**: Massive Success (+1.8 Billion JPY, 97% Buy). The agent found the "Bull Trend" strategy.
- **Seeds 123, 777**: Massive Failure (-20M to -260M JPY, 100% Sell). The agent fell into an "Inverse" trap (shorting the bull market).

The "Native 1D" action space is highly sensitive to initial exploration deviations. If the agent starts by randomly winning with Shorts (due to noise/scalping), it gets stuck in a Short-only loop because the "Buy" action space is never effectively explored or is initially punished by spread/volatility.

### 1.1 Historical Context & Evolution
By reviewing past documentation (`docs/v450`, `docs/v456`), we identified why previous attempts didn't solve this:
- **v450 (Curriculum Learning)**: Introduced `action_discovery` stage to fix "HOLD bias". However, it used **PnL sign** as the proxy for correctness. In v457.4, early PnL is noisy, leading to false reinforcement of Shorts (Inverse Learning).
- **v456 (Improvement Proposal)**: Proposed using "Ichimoku" for direction determination ("Phase 1: Direction"), but primarily as a data filter or feature. It did not strictly enforce *behavioral compliance* via the Reward Function.

**Evolution in v457.4**:
We will combine v450's **Curriculum Structure** with v456's **Ichimoku Signal**, but with a critical specific change: **Replace PnL-based discovery reward with Signal-based Imitation reward** during early stages.

## 2. Solution: Trend-Guided Curriculum
Instead of relying on luck (Random Seed) for the initial direction, we will use **Explicit Trend Guidance** provided by the existing `Ichimoku` implementation during the early Curriculum stages.

**Philosophy**: "Listen to the Teacher first, then develop your own style."

### 2.1 Existing Components
We have all the necessary components:
1.  **Teacher**: `IchimokuPatternRecognizer` / `SignalRewardIntegrator`.
    - `ztb/features/generators/technical/trend/ichimoku/ichimoku_ext.py` provides the raw signals (Cloud, TK Cross).
    - `ztb/trading/strategies/signal_reward_integrator.py` already has logic to `integrate_signal_reward` with `ichimoku_weight`.
    - Currently, this weight is static (~1.1).

2.  **Manager**: `BalanceCurriculumManager`.
    - `ztb/trading/environment/components/reward/balance_curriculum.py` manages stages: `action_discovery` -> `forced_balance` -> `balanced_transition` -> `trading_focused`.

3.  **Executor**: `RewardCalculator`.
    - `ztb/trading/environment/components/calculators/reward_calculator.py` calls the integrator.

### 2.2 Integration Plan (Wiring)
We need to make the **Signal Weights Dynamic** based on the **Curriculum Stage**.

**Logic to Implement in `RewardCalculator`:**
```python
def _update_signal_weights_based_on_curriculum(self):
    stage = self.curriculum_manager.current_stage
    
    if stage in ["action_discovery", "forced_balance"]:
        # Phase 1: STRICT OBEDIENCE
        # Force the agent to follow the trend. Punish deviations heavily.
        self.signal_integrator.ichimoku_weight = 5.0
        self.signal_integrator.signal_penalty_weight = 0.5 # Heavy penalty for fighting the trend
        
    elif stage == "balanced_transition":
        # Phase 2: RELAXATION
        # Still guide, but allow some counter-trend exploration.
        self.signal_integrator.ichimoku_weight = 1.0
        self.signal_integrator.signal_penalty_weight = 0.05
        
    else: # trading_focused, profit_optimized
        # Phase 3: FREE MARKET
        # Let the agent decide. The signal is just a "hint".
        self.signal_integrator.ichimoku_weight = 0.2
        self.signal_integrator.signal_penalty_weight = 0.0
```

## 3. Implementation Steps

1.  **Modify `ztb/trading/environment/components/calculators/reward_calculator.py`**:
    - Add `_update_dynamic_weights()` method.
    - Call this method inside `update()` or `calculate_step_reward()`.
    - Map `active_stage` to specific `ichimoku_weight` values in `SignalRewardIntegrator`.

2.  **Verify `ztb/trading/strategies/signal_reward_integrator.py`**:
    - Ensure `ichimoku_weight` is mutable and actually affects the output calculation (it should).

3.  **Config Update**:
    - Ensure `curriculum_learning.enabled = true` in the training config.
    - Ensure `signal_guidance.ichimoku_weight` is not hardcoded to override the dynamic changes (or ensure the dynamic change overrides the config).

## 4. Expected Outcome
- **Seeds 123 & 777** should now converge to the "Buy" strategy because the "Ichimoku" teacher (which correctly identifies the bull market) will punish the "Sell" exploration heavily in the first 100-200 steps.
- **Seed 42** should remain successful.
- The learning curve for "Buy" actions should start immediately, not after 100k steps of random walking.
