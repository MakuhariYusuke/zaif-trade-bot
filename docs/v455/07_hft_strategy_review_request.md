# HFT Strategy Feasibility Review Request

## 1. Current Situation
We are developing a crypto trading bot using Reinforcement Learning (Stable Baselines 3).
- **Current Model**: v454/v455 (PPO/SAC based)
- **Target**: High-Frequency Trading (HFT) / Scalping
- **Current Performance**:
    - Standard Config: 94 trades / 35 days (Avg Holding Time: ~16 hours) -> **Swing Trading**
    - Aggressive Config (Z-Score < 1.0): 485 trades / 35 days (Avg Holding Time: ~54 mins) -> **Day Trading**

## 2. Problem
The current model is too passive for HFT. It learns to hold positions to maximize trends rather than scalping small profits frequently.

## 3. Proposed Solutions
We are considering the following approaches:
1.  **Retraining**: Use a reward function that penalizes holding duration and rewards trade frequency.
2.  **Config Tuning**: Drastically lower transaction costs in simulation and tighten Take Profit / Stop Loss.
3.  **Architecture Change**: Switch to a simpler logic for HFT (e.g., Grid Trading, Market Making) and use RL only for parameter tuning.

## 4. Request for Review
Please review the above context and provide your expert opinion on:
1.  **Feasibility**: Can RL (PPO/SAC) effectively learn HFT strategies with standard OHLCV data (1m/5m candles)?
2.  **Strategy**: What specific changes to the Reward Function or Observation Space would you recommend for HFT?
3.  **Alternatives**: Would you recommend a different approach entirely?

---
*Please append your review below this line.*

## Expert Review (RL + HFT)

### Summary
With 1m/5m OHLCV only, PPO/SAC will almost always learn swing or at best day-trade behavior. The state is too coarse to reward true microstructure timing, and the reward is usually dominated by trend capture. You can push trade frequency up, but it will be fragile unless you change the reward, the action space, and the environment timing.

### Feasibility: Can PPO/SAC learn HFT from 1m/5m candles?
Short answer: not really, at least not in the HFT sense. You can approximate faster trading, but real HFT depends on sub-minute microstructure (spread, queue position, order book). With OHLCV, the agent sees a smoothed and delayed signal. That pushes it toward longer holds and trend capture. If you must stay on 1m/5m candles, aim for "fast swing" or "intraday momentum", not true scalping.

### Reward Function Design (Deep Dive)
Penalizing holding time alone will just create churn and fee bleed. You need a reward that:
1) Aligns with net PnL after costs.
2) Rewards speed when profit exists, but avoids flipping when edge is weak.

Recommended structure:
```
r_t = pnl_t - fee_t - slippage_t
      - alpha * abs(position_change)
      - beta  * holding_time
      - gamma * inventory_risk
      - k     * max(0, drawdown - dd_cap)
```
Notes:
- `pnl_t` should be mark-to-market each step to avoid sparse reward.
- `position_change` discourages overtrading without killing frequency.
- `holding_time` should be a soft penalty, not a hard clamp. Use a small beta and increase it after a warmup phase.
- `inventory_risk` can be `abs(position) * volatility` to discourage large exposure in volatile periods.

If you want truly short trades, use a time-discounted PnL:
```
reward = exp(-t_h / tau) * realized_pnl - costs
```
This makes fast exits more attractive without forcing churn.

### HFT Suitability Improvements
1) **Action space**: Use "target position" or "entry + time-to-live". If an agent can hold forever, it will.
2) **Forced exit**: Add max holding time (TTL) or forced flattening every N steps.
3) **Event-based steps**: Use time only when there is volume or volatility; fixed 1m steps are too slow for HFT behavior.
4) **Observation**: Add microstructure proxies:
   - range and range acceleration
   - volume imbalance (up/down volume if available)
   - VWAP deviation
   - short-term volatility regime

### Reward and Gate Consistency
If the Gate filters on EV after costs, but training reward ignores costs, you are teaching a policy that the Gate will reject. This is a guaranteed mismatch. Fix it by using the same cost model in both training and Gate.

### Alternatives (If HFT is non-negotiable)
1) **Market making / grid** with a control policy:
   - Rule-based core (spread, inventory bands)
   - RL only tunes spread width and skew
2) **Supervised direction + RL sizing**:
   - Classifier predicts short-term direction
   - RL only chooses size or whether to trade
3) **Bandit selection of strategies**:
   - Small set of hand-crafted scalping rules
   - Bandit chooses which rule to activate per regime

### Concrete Next Steps
1) Add cost to reward now (fees and slippage) and re-run.
2) Add a soft holding-time penalty and a hard max holding time.
3) Use target-position action to avoid "hold forever" behavior.
4) Do a sanity test: If you randomize action order, does performance remain? If yes, you are not learning signal.

If you want true HFT, you must move to sub-minute data or order book features. Without that, the best outcome is "faster swing", not actual scalping.
