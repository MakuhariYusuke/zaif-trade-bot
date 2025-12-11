# Phase 7 (v451) Architecture Design

## 1. Overview
**Version:** v451
**Codename:** "Chronos" (Time & Regime Aware)
**Base Model:** SAC (Soft Actor-Critic)
**Predecessor:** v450 (Iter 12)

## 2. Core Philosophy
The v451 architecture addresses the "Context Blindness" of previous iterations. Instead of assuming a stationary environment where $P(State | Action)$ is constant, v451 explicitly acknowledges that market dynamics are functions of Time ($t$) and Volatility Regime ($V$).

$$ Policy(s) \rightarrow Policy(s, t, V) $$

## 3. Feature Engineering (The "Eyes")
The input state space is augmented with 7 critical dimensions:

### A. Cyclical Time Embeddings
To avoid the discontinuity problem of linear time (where 23:59 and 00:00 are far apart), we use trigonometric encoding:
*   **Hour:** $H_{sin} = \sin(2\pi \frac{h}{24})$, $H_{cos} = \cos(2\pi \frac{h}{24})$
*   **Minute:** $M_{sin} = \sin(2\pi \frac{m}{60})$, $M_{cos} = \cos(2\pi \frac{m}{60})$
*   **Day:** $D_{sin} = \sin(2\pi \frac{d}{7})$, $D_{cos} = \cos(2\pi \frac{d}{7})$

### B. Explicit Regime Embeddings
*   **Volatility Rank (`vol_rank`):** A percentile rank (0-1) of current volatility vs. the last 1000 steps. This normalizes volatility across different market eras (2020 vs 2025).
*   **Regime Flags:** One-hot encoding of the volatility quartile.

## 4. Network Architecture (The "Brain")
*   **Algorithm:** SAC (Standard)
*   **Policy Network:** MLP (Multi-Layer Perceptron)
    *   Input Layer: ~180 neurons (v427 features + v451 features)
    *   Hidden Layers: [256, 256]
    *   Output: Continuous Action (Buy/Sell/Hold)
*   **Critic Network:** Double Q-Learning (to reduce overestimation bias).

## 5. Reward Function (The "Motivation")
Retaining the successful "Asymmetric Loss Aversion" from v450 Iter 12.

$$ R = \begin{cases} PnL & \text{if } PnL > 0 \\ PnL \times 1.2 & \text{if } PnL < 0 \end{cases} $$

*   **Fee Penalty:** Implicit in PnL (Net PnL used).
*   **Entropy:** High ($\alpha=0.05$) to encourage exploration of the new state space dimensions.

## 6. Hypothesis for Improvement
By correlating `Reward` with `(State, Time, Regime)`, the Critic network ($Q(s,a)$) will learn to assign lower values to "Buy" actions during:
1.  14:00 JST (Pre-European Lull)
2.  Med-Low Volatility Transitions

Consequently, the Actor will learn to output "Hold" or reduce position sizes in these specific contexts, without human hard-coding.
