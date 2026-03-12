from ztb.trading.constants import ACTION_BUY, ACTION_SELL

from .base import RewardComponent, RewardContext

class TradingFocusedReward(RewardComponent):
    """
    Stage: Trading-focused reward that heavily penalizes HOLD and encourages trading.
    Ported from RewardCalculator._calculate_trading_focused_reward.
    """

    ACTION_BUY = ACTION_BUY
    ACTION_SELL = ACTION_SELL

    def get_name(self) -> str:
        return "trading_focused"

    def calculate(self, context: RewardContext) -> float:
        # 1. PnL Reward (Primary Driver)
        # Normalize PnL relative to portfolio size.
        # pnl_pct is in percentage points (e.g., 1.0 = 1%).
        pnl_pct = (context.pnl / max(context.initial_portfolio_value, 1.0)) * 100.0

        # We use pnl_pct directly. 1% gain = +1.0 reward.
        # Asymmetric Reward: Punish losses 1.2x more to encourage higher win rate.
        if pnl_pct < 0:
            reward = pnl_pct * 1.2
        else:
            reward = pnl_pct

        # 2. Fee Penalty (Simulated Hurdle)
        # We subtract 0.025 (representing 0.025% cost) for every trade.
        # This forces the agent to only trade if it expects > 0.025% profit.
        # Adjusted to 0.025 (Very Low) to maximize volume, while asymmetric reward handles quality.
        if context.action in [self.ACTION_BUY, self.ACTION_SELL]:
            reward -= 0.025

        return reward

        # 3. Volatility Scaler
        # If volatility is high, we expect larger PnL swings.
        # We might want to normalize reward by ATR to make learning stable across regimes.
        # OR, we want to encourage trading during high volatility.
        # Let's stick to simple PnL for now.

        # 4. Remove Balance Penalty
        # HFT doesn't care about 80% HOLD ratio.

        return reward
