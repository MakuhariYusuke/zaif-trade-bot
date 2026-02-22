# v456 Plan Review (External)

## Scope
- `docs/v456/00_improvement_proposal.md`
- `docs/v456/01_technical_specification.md`
- `docs/v456/02_feature_engineering_spec.md`
- `docs/v456/03_implementation_checklist.md`
- `docs/v456/04_self_review.md`

## Critical Issues
1. **MTF resampling may leak future data**  
   The resampling step does not specify bar alignment or “closed-bar only” usage, so 5m/15m/1h aggregates risk leaking future 1m information into the current step. This will inflate backtests and break live parity.  
   References: `docs/v456/02_feature_engineering_spec.md:500`, `docs/v456/02_feature_engineering_spec.md:501`

2. **Normalization pipeline conflicts with spec and distorts categorical/time features**  
   The unified `OnlineScaler` is applied to the full concatenated vector, which will normalize one-hot regime flags and sin/cos time features even though the spec expects them to remain categorical/periodic. This changes semantics and can induce non-stationary inputs.  
   References: `docs/v456/02_feature_engineering_spec.md:545`, `docs/v456/02_feature_engineering_spec.md:591`, `docs/v456/01_technical_specification.md:179`, `docs/v456/01_technical_specification.md:185`

3. **Recurrent SAC details are missing for off-policy training**  
   GRU architecture and config are specified, but there is no sequence replay/burn-in design, episode boundary handling, or hidden-state storage policy. Off-policy RNNs are fragile without this.  
   References: `docs/v456/01_technical_specification.md:304`, `docs/v456/01_technical_specification.md:392`

## Major Issues
4. **Reward shaping can dominate PnL and cause reward hacking**  
   The MTF alignment bonus and balance enforcement penalty have fixed scales with no calibration to PnL magnitude or cost model. This risks learning “alignment” or “balance” instead of profitability.  
   References: `docs/v456/01_technical_specification.md:50`, `docs/v456/01_technical_specification.md:77`, `docs/v456/01_technical_specification.md:106`

5. **Global market features assume lead-lag and ignore FX/basis risk**  
   Using BTC/USDT returns as a predictor for BTC/JPY without an FX or basis adjustment can create spurious signals. The stated ~500ms lead is small relative to 1m bars and can vanish under bar alignment. The current fallback to `0.0` also hides “data missing” states.  
   References: `docs/v456/02_feature_engineering_spec.md:295`, `docs/v456/02_feature_engineering_spec.md:300`, `docs/v456/02_feature_engineering_spec.md:350`, `docs/v456/02_feature_engineering_spec.md:367`

6. **Action filtering/gating risks train‑live mismatch**  
   Soft Filter, signal fusion, and Calibration Gate are post-action modifiers. If training does not include these gates inside the environment step, the policy learns on a different transition function than live.  
   References: `docs/v456/01_technical_specification.md:207`, `docs/v456/01_technical_specification.md:240`

## Moderate Issues
7. **Trade-based PnL reward is sparse without variance control**  
   Switching to trade-based rewards can slow learning and increase variance unless paired with n‑step returns or hybrid shaping.  
   References: `docs/v456/00_improvement_proposal.md:293`

8. **Backtest splits lack embargo/purged validation**  
   The contiguous split and standard walk-forward are vulnerable to leakage in highly autocorrelated crypto regimes; a purged CV or embargo gap is recommended.  
   References: `docs/v456/01_technical_specification.md:470`, `docs/v456/01_technical_specification.md:501`

9. **Circuit breaker thresholds are fixed rather than volatility‑adaptive**  
   Daily loss 5% and max DD 10% may be too strict or too loose depending on volatility regime. Static thresholds can either over‑halt or fail to protect.  
   References: `docs/v456/01_technical_specification.md:517`

10. **KPI targets lack statistical grounding**  
   +5% return and 55% win rate are ambitious relative to prior results and lack confidence intervals or expected error bars.  
   References: `docs/v456/00_improvement_proposal.md:275`

## Minor Issues
11. **Feature count inconsistency**  
   The summary states ~82 features total, while the calculated dimension is 85 including account state. Align the wording to avoid confusion in implementation.  
   References: `docs/v456/02_feature_engineering_spec.md:21`, `docs/v456/02_feature_engineering_spec.md:595`

12. **Timezone handling is ambiguous for naive timestamps**  
   `timestamp.tzinfo` conditional may silently treat naive timestamps as JST; if the source data is UTC-naive, time features will be wrong.  
   References: `docs/v456/02_feature_engineering_spec.md:237`, `docs/v456/02_feature_engineering_spec.md:247`

## Answers to Specific Questions
**Q1 (85 dims for SAC / PCA?)**  
85 dims is not inherently too large for SAC, but the combination of MTF+GRU raises sample complexity. Prefer staged feature rollout and ablation. PCA can help but may harm interpretability and introduce drift if fit online; if used, freeze PCA on training data and validate stability.

**Q2 (GRU with SAC off-policy?)**  
Technically feasible but requires sequence replay with burn‑in, truncated BPTT, and hidden‑state handling. If this is not already implemented, consider frame‑stacking or TCN/attention before a full recurrent SAC.

**Q3 (Lead‑Lag persistence & fallback)**  
Monitor rolling cross‑correlation or Granger causality with statistical tests; disable or down‑weight global features when significance drops. Add FX/basis features (USDJPY, USDT premium) and a “global data stale” flag for fallback.

**Q4 (Cyclical time vs masking)**  
Cyclical features are better for learning soft time‑of‑day effects; hard masking is safer but may miss opportunities. A hybrid approach works best: cyclical features + soft risk multiplier on known danger windows.

**Q5 (Reward shaping risk)**  
Yes, strong bonuses/penalties can dominate PnL and cause reward hacking. Use small, clipped shaping terms, or use them as auxiliary losses or gating conditions instead of direct rewards.

**Q6 (Trade‑based PnL shift)**  
Trade‑level reward reduces the “high win / low profit” artifact but becomes sparse. Use hybrid rewards (step‑level cost/holding + trade‑close PnL), or n‑step returns to reduce variance.

**Q7 (Circuit breaker thresholds)**  
Use volatility‑adaptive limits (e.g., ATR‑based) or percentile‑based thresholds. Calibrate with historical distribution of daily PnL to set limits that trigger on tail risk, not normal variance.

**Q8 (Black swan scenarios)**  
Add exchange outage handling, API schema change detection, severe spread spikes, stablecoin depeg events, and local price feed corruption. Include a data‑quality kill switch and forced position flattening.

**Q9 (8‑week implementation order)**  
Prioritize data integrity (resampling alignment + feature normalization), then MTF/time features + backtest, then gating/Soft Filter. Defer GRU and complex signal fusion until a stable, profitable baseline exists.

**Q10 (Missing lessons from v448‑v455)**  
The “no‑trade bias” risk from strong penalties and filters remains under‑addressed. Add explicit trade‑frequency or opportunity‑cost terms and verify that filtering doesn’t collapse action diversity.

## Priority Recommendations (Practical)
1. Fix MTF alignment/leakage and feature normalization first.
2. Build a strong non‑GRU baseline with staged features and strict leakage tests.
3. Add global features only after FX/basis and staleness indicators are in place.
4. Introduce GRU last, only if baseline shows persistent contextual errors.

## Alternative Approaches
- Replace GRU with TCN or attention on fixed‑length context windows.
- Use a supervised direction/volatility model as a gating signal for the RL agent.
- Treat signal fusion as a separate policy layer and train RL on the filtered action space.
