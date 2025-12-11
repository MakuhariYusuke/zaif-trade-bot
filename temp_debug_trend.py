from ztb.trading.constants import ACTION_BUY
from ztb.trading.environment.components.reward_calculator import RewardCalculator
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings


class StubTrendDetector:
    def __init__(self, s):
        self._s = s

    def get_trend_signal(self):
        return self._s

    def update(self, price):
        pass

    def get_statistics(self):
        return {"samples": 1, "last_signal": self._s}


config = EnvironmentConfig()
cfg = RewardSettings()
rc = RewardCalculator(config, cfg, 1000.0)
rc.reward_settings.custom_reward_params = {
    "forced_balance_min_actions": 1,
    "forced_balance_exploration_reward": 0.0,
}
rc._action_counts = [0, 8, 0]
rc.behavioral_penalty_calculator.trend_detector = StubTrendDetector(0.6)
print("pos targets", rc.behavioral_penalty_calculator.get_target_ratios())
pos = rc._calculate_forced_balance_reward(action=ACTION_BUY, step=10)
print("pos reward", pos)
rc.behavioral_penalty_calculator.trend_detector = StubTrendDetector(-0.6)
print("neg targets", rc.behavioral_penalty_calculator.get_target_ratios())
neg = rc._calculate_forced_balance_reward(action=ACTION_BUY, step=10)
print("neg reward", neg)
print("neg last_details", rc.forced_balance_reward.last_reward_details)
print("pos last_details", rc.forced_balance_reward.last_reward_details)

from ztb.trading.environment.components.rewards.forced_balance import (
    ForcedBalanceReward,
)

f = ForcedBalanceReward()
context = rc._build_reward_context(action=ACTION_BUY)
target_ratios = context.target_ratios
print("context targets", target_ratios)
deviation_pos = 1.0 - 0.36
deviation_neg = 1.0 - 0.24
print("penalty pos", f._map_forced_balance_penalty(context, deviation_pos, 0.1))
print("penalty neg", f._map_forced_balance_penalty(context, deviation_neg, 0.1))
