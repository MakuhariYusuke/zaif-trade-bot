# Experiment v450 Updates Log

## Phase 6: HFT Fine-tuning (High Frequency Trading)

### Objective
Transition the model from a "Swing Trader" (Phase 5) to a "High Frequency Trader" (Phase 6) that executes multiple trades per day with high win rate.

### Iteration History

#### Iteration 7
- **Config:** Penalty -0.05, Ent 0.005, Gamma 0.99
- **Result:** 65 Trades, PF 1.33, +11% Return.
- **Verdict:** Too slow.

#### Iteration 8
- **Config:** Penalty -0.035, Ent 0.02, Gamma 0.80
- **Result:** 103 Trades, PF 1.59, +23% Return.
- **Verdict:** Profitable but volume still too low (~1.7/day).

#### Iteration 9 (Breakthrough)
- **Config:** Penalty -0.032, Ent 0.05, Gamma 0.80
- **Result:** 612 Trades (~10/day), PF 1.15, +15.6% Return, Win Rate 51.1%.
- **Verdict:** Achieved volume goal. Win rate needs improvement.

#### Iteration 10
- **Config:** Penalty -0.030, Ent 0.03, Gamma 0.75
- **Result:** 268 Trades, PF 1.19, Win Rate 48.5%.
- **Verdict:** Lower entropy/gamma hurt volume and win rate. Reverting direction.

#### Iteration 11
- **Config:** Penalty -0.028, Ent 0.04, Gamma 0.82
- **Result:** 523 Trades, PF 1.14, Win Rate 51.05%, +16.2% Return.
- **Verdict:** Similar to Iter 9. Win rate stuck at 51%.

#### Iteration 12 (Success)
- **Config:** Penalty -0.025, Loss Penalty 1.2x, Ent 0.05, Gamma 0.80
- **Result:** 694 Trades (~11.5/day), PF 1.15, Win Rate 52.16%, +16.3% Return.
- **Verdict:** Best model so far. Improved both Volume and Win Rate.

#### Iteration 13
- **Config:** Penalty -0.020, Loss Penalty 1.5x, Ent 0.05, Gamma 0.80
- **Result:** 490 Trades, PF 1.12, Win Rate 53.27%, +13.1% Return.
- **Verdict:** Loss penalty too harsh. Volume and Profit dropped.

### Final Conclusion (Phase 6)
**Iteration 12 is the optimal configuration.**
- **Configuration:**
    - **Reward Penalty:** -0.025 (Low barrier for entry).
    - **Asymmetric Reward:** Losses penalized 1.2x (Encourages quality).
    - **Entropy:** 0.05 (High exploration).
    - **Gamma:** 0.80 (Scalping focus).
- **Performance:**
    - **Trades:** 694 (~11.5/day).
    - **Win Rate:** 52.16%.
    - **Profit Factor:** 1.15.
    - **Total Return:** +16.3%.
- **Status:** Model restored to Iteration 12 state.
