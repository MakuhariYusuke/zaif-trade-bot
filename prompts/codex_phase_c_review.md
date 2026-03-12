# Codex Review Request: Phase C0+C1 — SAC Deterministic Policy Learn "Do Nothing"

## Context

We are building a BTC/JPY 1-minute intraday trading bot using SAC (Soft Actor-Critic) on a custom Gymnasium environment (`HeavyTradingEnv`). After running 14 controlled experiments across gamma, threshold, min_holding_period, and reward calculator variants, **every single experiment resulted in a deterministic policy that executes zero trades**.

The codebase is in `zaif-trade-bot`, a private repo with ~50K lines of Python. Our environment outputs 8 features (RSI×7 + ReturnStdDev) and uses continuous action space where action magnitude controls position sizing and sign controls direction (BUY/SELL), with a threshold parameter determining the HOLD zone (|action| < threshold → HOLD).

## The Problem

### Training Phase (stochastic policy, entropy-driven exploration):
- ~800-1,080 trades per 50K steps (BUY:SELL perfectly symmetric, e.g. 518:518)
- Net ROI: -14.96% to -15.27% (stable across all params → ≈ random trading fee drain)
- Gross PnL ranges from -3,176 to +1,977 JPY on 100K initial balance

### Evaluation Phase (deterministic=True):
- **0 trades in all 14 experiments** (50K evaluation steps each)
- All Gate 2 KPIs: 0.0 (Sharpe, WinRate, ROI, etc.)
- Balance unchanged from initial 100,000 JPY

### Experiments Tried (all failed):
1. **Gamma sweep**: 0.80, 0.90, 0.95, 0.99 → No effect on deterministic trades
2. **Threshold sweep**: 0.33, 0.50, 0.60, 0.70 → Higher threshold = fewer stochastic trades, still 0 deterministic
3. **Gamma × Threshold combos**: γ=0.80 × {0.50, 0.60, 0.70} → No effect
4. **min_holding_period**: 3, 5, 10, 15 → MHP=5 *increased* stochastic trades by 4% (反動取引), still 0 deterministic
5. **v451 Golden Era settings**: γ=0.80 + V457RewardCalculator + scale=1.0 → No improvement

## Our Root Cause Analysis

### 1. Reward Signal Too Weak
```
1-step BTC/JPY move:  ±0.01% (1-minute bar)
1 trade gross edge:    ±10 JPY (on 100K position)
Transaction cost:      100 JPY (0.1% maker fee on Zaif)
Reward for trading:    (10 - 100) / 100,000 × 100 = -0.09
Reward for HOLD:       0.0
```
→ SAC rationally learns HOLD as optimal. The entropy bonus $\alpha \mathcal{H}(\pi)$ dominates over the tiny reward signal during training, and the mean of the resulting symmetric distribution (≈0.0) falls below the threshold.

### 2. Action Space Design
Continuous action in [-1, 1] with threshold → deterministic policy outputs the mean of the learned Gaussian → mean ≈ 0.0 → always HOLD.

### 3. BUY:SELL Perfect Symmetry
All trades during training come from SAC's exploration noise, not learned behavior. The policy never develops directional bias.

## Questions for Codex

Please analyze critically, including unconventional perspectives:

1. **Is our root cause analysis correct?** Are there alternative explanations we might be missing? Could the problem be in the environment (observation normalization, reward clipping, info dict), the SAC hyperparameters (learning rate, buffer size, batch size, tau, ent_coef), or something else entirely?

2. **Reward design**: What is the recommended approach for making SAC learn to trade in a high-transaction-cost, low-edge environment? Specifically:
   - Should we add explicit trade incentives (positive reward for trading)?
   - Should we use asymmetric rewards (larger penalty for missed profitable moves than for losing trades)?
   - Should we use a risk-adjusted reward (Sharpe-based, PnL-centered) instead of step-by-step delta?
   - What about reward shaping with potential-based functions to avoid reward hacking?

3. **Action space**: Should we abandon continuous actions and switch to discrete (BUY/SELL/HOLD)? What's the trade-off? Would DQN/PPO with discrete actions be more appropriate for this problem?

4. **Entropy coefficient**: SAC's auto-tuned α may be too high, making exploration dominate. Should we:
   - Fix ent_coef to a small value (e.g., 0.01)?
   - Use a schedule that decays α over training?
   - Start with a pre-trained policy to skip the "random exploration" phase?

5. **Training scale**: 50K steps on 1.2M-row data means each episode sees only ~4% of the data. Is 50K fundamentally insufficient? If so, what's the minimum viable training budget for SAC on 1-minute financial data?

6. **Environment design**: Is `HeavyTradingEnv` with position-based trading (buy/sell/hold × position_size) fundamentally harder for SAC than:
   - Target portfolio weight approach (action = desired allocation %)
   - Signal-based approach (action = expected return signal, execution handled externally)
   - Multi-agent approach (separate timing and sizing agents)

7. **Priority recommendation**: Given limited compute (single GPU, ~25 min per 50K steps), which of these should we try FIRST?
   - **C3-A**: Trade incentive reward (add positive reward for trading, penalty for extended HOLD)
   - **C3-B**: Discrete action space (BUY/SELL/HOLD with DQN or PPO)
   - **C3-C**: Scale to 200K steps (4× compute cost)
   - **C-alt**: Switch to FastIntradayEnv (different env with 88 features, previously showed some success)
   - **C-D**: Manual ent_coef control (reduce α to force exploitation)
   - **C3-A + C-D combo**: reward_scale=1000 + ent_coef=0.01

8. **Fundamental viability**: Given BTC/JPY 1-minute data with 0.1% transaction cost, is RL-based trading even viable at this timeframe? Should we consider:
   - Longer timeframes (5min, 15min) to increase signal-to-noise?
   - Lower-cost exchanges?
   - Hybrid approach (ML for signal, rules-based for execution)?

## Key Files for Reference
- Environment: `src/ztb/envs/heavy_trading_env.py`
- Reward: `src/ztb/envs/rewards.py` (SimpleRewardCalculator, V457RewardCalculator)
- SAC config: `src/ztb/config/ppo_config.py` (shared config, despite the name)
- Phase C runner: `scripts/v459/run_phase_c.py`
- Results: `docs/v459/102_phase_c_experiment_log.md`
- Gate 2 KPI spec: `docs/v459/000_operations_manual.md` §5.2

## Constraints
- **Primary goal**: High short-term profitability system (project mandate)
- **Timeline**: Rapid iteration needed, can't afford multi-week exploration
- **Compute**: Single machine, ~25 min per 50K SAC steps
- **Exchange**: Zaif (Japan), 0.1% maker fee, BTC/JPY only
- **Current state**: All 14 controlled experiments failed → need a paradigm shift, not incremental tuning
