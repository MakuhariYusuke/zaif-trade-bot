# HFT/Fast Intraday Implementation Details Request

## 1. Context & Decision
Thank you for the "Expert Review". We accept the conclusion that with **1m/5m OHLCV data**, we cannot achieve true microstructure HFT.
**Decision**: We will pivot our target from "True HFT" to **"Fast Intraday / Momentum Trading"** (holding time: minutes to hours, not seconds).

## 2. Request for Implementation Details
Based on your offer, please provide specific implementation details for the following areas. We use Python, Stable Baselines 3, and Gymnasium.

### A. Concrete Reward Function Implementation
Please provide a Python code snippet (or pseudo-code) for the reward function you proposed:
$$ r_t = pnl_t - fee_t - slippage_t - \alpha \cdot |position\_change| - \beta \cdot holding\_time - \gamma \cdot inventory\_risk $$
*   **Question**: How do we normalize these components? PnL can be large, while penalties might be small.
*   **Question**: How specifically should `inventory_risk` be calculated using 1m OHLCV volatility?

### B. Action Space Redesign (Target Position / TTL)
We currently use `Discrete` actions (Buy, Sell, Hold) or `Box` (Amount).
*   **Proposal**: You suggested "Target Position" or "Entry + TTL".
*   **Request**: Please describe the `action_space` definition and how to map the action to environment logic.
    *   *Example*: If action is `0.5` (Target 50% Long), how does the env handle the transition from current position?
    *   *TTL*: How do we implement "Time To Live" as an action?

### C. Microstructure Proxies from 1m Data
Since we lack tick data, we need the best possible proxies from 1m OHLCV.
*   **Request**: Please list 3-5 specific feature engineering formulas.
    *   *Volume Imbalance*: Can we approximate this from Open/Close/High/Low?
    *   *Volatility Regime*: Which specific indicator is most responsive for 1m data (ATR, StdDev, etc.)?

---
*Please append your detailed implementation guide below.*

## Implementation Guide (Appendix)

### A. Reward Function (Python Snippet + Normalization)
Below is a concrete reward implementation aligned with your formula. The key is **unitless normalization** to keep signal scale stable.

**Normalization Strategy**
- Use **return-scale**: `pnl / (price * max_position)` or `pnl / (ATR * max_position)`.
- Use **cost-scale** consistent with pnl: cost in JPY, normalize by same denominator.

```python
def compute_reward(
    price_prev: float,
    price_now: float,
    position_prev: float,
    position_now: float,
    atr: float,
    fee_paid: float,
    slippage_paid: float,
    holding_steps: int,
    max_position: float,
    alpha: float = 0.2,
    beta: float = 0.01,
    gamma: float = 0.5,
    eps: float = 1e-8,
):
    # Mark-to-market PnL (JPY)
    pnl = position_prev * (price_now - price_prev)

    # Normalizer: use ATR or price to make reward unitless
    denom = max(atr, price_now * 0.001, eps) * max_position

    pnl_norm = pnl / denom
    fee_norm = fee_paid / denom
    slip_norm = slippage_paid / denom

    # Position change penalty (churn)
    position_change = abs(position_now - position_prev)
    churn_penalty = alpha * position_change

    # Holding time penalty (soft)
    hold_penalty = beta * holding_steps

    # Inventory risk: larger position + higher vol = more penalty
    vol_ratio = atr / max(price_now, eps)
    inventory_risk = gamma * abs(position_now) * vol_ratio

    reward = (
        pnl_norm
        - fee_norm
        - slip_norm
        - churn_penalty
        - hold_penalty
        - inventory_risk
    )
    return reward
```

**Inventory Risk (1m OHLCV)**
- Recommended: `inventory_risk = |pos| * (ATR / price)`  
  This is unitless and increases risk when volatility expands.

---

### B. Action Space Redesign (Target Position + TTL)

#### Option 1: Target Position Only
Action space: continuous target position in `[-1, 1]`.
```python
action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
```
Mapping logic:
```python
target = float(action[0]) * max_position  # e.g. 0.5 -> 50% long
delta = target - current_position

# Optional: cap per-step change (risk control)
delta = np.clip(delta, -max_delta_per_step, max_delta_per_step)

execute_trade(delta)
current_position += delta
```

#### Option 2: Target Position + TTL (Recommended)
Action space: `[target_position, ttl_fraction]`
```python
action_space = gym.spaces.Box(low=np.array([-1.0, 0.0]),
                              high=np.array([ 1.0, 1.0]),
                              dtype=np.float32)
```
Mapping logic:
```python
target = float(action[0]) * max_position
ttl_steps = int(float(action[1]) * max_ttl_steps)

# When a new position is opened or reversed:
if sign(target) != sign(current_position):
    position_ttl = ttl_steps

# Each step:
position_ttl -= 1
if position_ttl <= 0:
    target = 0.0  # force flatten

delta = np.clip(target - current_position, -max_delta, max_delta)
execute_trade(delta)
current_position += delta
```

Notes:
- TTL acts as a **hard timeout** to stop swing-like behavior.
- `max_delta` prevents the agent from instant full reversals.

---

### C. Microstructure Proxies from 1m OHLCV
Since no order book exists, use robust candle/volume proxies.

1) **Close Location Value (CLV)**
```
CLV = (close - low) / (high - low + eps) * 2 - 1
```
Range: [-1, 1]. Positive = closes near high (buy pressure).

2) **Signed Volume Pressure**
```
vol_pressure = CLV * volume
```
Proxy for aggressive flow without tick data.

3) **Range-per-Volume (Impact Proxy)**
```
impact = (high - low) / max(volume, eps)
```
Higher impact implies poor liquidity / high slippage risk.

4) **Short-Term Volatility Regime**
```
vol_1m = ATR(14) / close
vol_fast = EMA(|log_return|, span=5)
```
Use both: ATR for baseline, EMA(abs returns) for fast shock detection.

5) **Trend Persistence**
```
trend = EMA(close, 5) - EMA(close, 20)
trend_strength = trend / ATR(14)
```
Detects short momentum vs mean-reversion.

---

## Quick Configuration Defaults (Starting Point)
- `max_position = 1.0`
- `max_ttl_steps = 30` (30 minutes on 1m bars)
- `max_delta_per_step = 0.2 * max_position`
- `alpha = 0.2`, `beta = 0.01`, `gamma = 0.5` (tune after leak fix)

If you want, I can turn the above into a concrete Gym environment patch with tests.
