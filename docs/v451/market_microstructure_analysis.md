# Phase 7 (v451) Deep Dive: Market Microstructure & Regime Analysis

## 1. The "Blind Agent" Hypothesis
The failure of v450 (Iter 12) to handle specific time windows (14:00, 17:00, 01:00) and volatility regimes (Med-Low) is not a random stochastic error. It is a structural deficiency in the agent's State Space.

### The Problem of Implicit Learning
Reinforcement Learning agents *can* theoretically learn time dependencies from price action sequences (e.g., "volatility usually increases after X steps"), but in financial time series:
1.  **Noise Ratio:** The signal-to-noise ratio is extremely low.
2.  **Context Switching:** A "flat" price history looks the same at 14:00 (Pre-London Lull) as it does at 04:00 (Asian Lull), but the *probability of a breakout* in the next 15 minutes is vastly different.
3.  **Blindness:** Without explicit time features, the agent treats these two situations as identical. It learns an "average" policy that might be slightly profitable overall but disastrous during specific structural shifts.

## 2. Market Microstructure Analysis (Time-of-Day Failures)
We identified three specific loss-making windows. Here is the deep-dive hypothesis for each, assuming JST (Japan Standard Time) based on the bot's context (`zaif-trade-bot`).

### A. 14:00 JST (05:00 UTC) - The "Pre-European Trap"
*   **Market State:** This is the late Asian session / pre-European morning.
*   **Characteristics:** Liquidity often thins out as Asian traders close positions. Volatility compresses.
*   **The Trap:** The v450 model, which loves "Low Volatility" (Mean Reversion), likely identifies this quiet period as a perfect scalping environment. It opens positions expecting mean reversion.
*   **The Failure:** As 15:00-16:00 approaches (European pre-market/open), institutional flows begin to position for the open. These are often *directional* moves that break the mean-reversion logic. The agent gets caught holding a mean-reversion bag into a directional breakout.

### B. 17:00 JST (08:00 UTC) - London Open Volatility
*   **Market State:** Official open of the London Stock Exchange and major European forex centers.
*   **Characteristics:** Massive injection of volume and volatility. "Fake-outs" (Stop hunts) are common in the first 15-30 minutes.
*   **The Failure:** The agent sees high volatility and might interpret it as a trend, but the initial moves are often noise/reversals. Or, it tries to fade the move (mean reversion) and gets run over by a sustained trend.
*   **v451 Solution:** Explicitly knowing it is "17:00" allows the agent to learn a specific policy: "Widen stops" or "Wait for confirmation" specifically at this hour.

### C. 01:00 JST (16:00 UTC) - London Close / NY Mid-day
*   **Market State:** The "London Fix" (16:00 London time).
*   **Characteristics:** A massive liquidity event where benchmark rates are set. Large flows occur to rebalance portfolios.
*   **The Failure:** Similar to the London Open, this is a structural liquidity event that defies standard technical analysis patterns. Price action becomes disjointed from typical momentum/reversion logic.

## 3. Regime Analysis (The "Med-Low" Valley of Death)
The analysis showed a "U-shaped" performance curve:
*   **Low Vol:** Profitable (Mean Reversion works).
*   **Med-Low Vol:** **Losses** (Transition Zone).
*   **High Vol:** Profitable (Trend Following works).

### The "Uncanny Valley" of Volatility
*   **Low Vol:** Price noise is Gaussian. Bollinger Bands work perfectly.
*   **High Vol:** Price moves are heavy-tailed (Trends). Momentum works.
*   **Med-Low Vol:** This is the *transition* phase. The market is waking up. It looks like a range (Low Vol), so the agent applies Mean Reversion. But it has just enough energy to *break* the range levels, triggering stop-losses, before reverting (or not).
*   **The Fix:** By providing `vol_rank` and `regime_med_low` explicitly, the agent can partition its policy:
    *   If `Regime == Low`: Aggressive Mean Reversion.
    *   If `Regime == Med-Low`: **Defensive Mode** (Tight stops, wait for breakout).
    *   If `Regime == High`: Trend Following.

## 4. v451 Architectural Solution
We are not changing the brain (SAC Algorithm); we are upgrading the eyes (Features).

### New Sensory Inputs
1.  **Cyclical Time:**
    *   $\sin(\frac{2\pi h}{24}), \cos(\frac{2\pi h}{24})$: Maps 23:00 and 00:00 as close neighbors.
    *   Allows the neural network to approximate functions like $f(t) = \text{Risk}(t)$.
2.  **Regime Context:**
    *   `vol_rank` (0.0 - 1.0): A normalized "fear gauge".
    *   `vol_ratio` (Short/Long): A "regime change" derivative.

### Expected Outcome
The v451 model should be able to learn:
> "When `hour_sin` corresponds to 14:00 AND `vol_rank` is rising (Transition), REDUCE position size or WIDEN stops."

This is a fundamental capability upgrade, not a heuristic rule.
