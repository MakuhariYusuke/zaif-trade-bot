from ztb.trading.environment.components.behavioral_penalty_calculator import BehavioralPenaltyCalculator
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.trading.constants import ACTION_BUY, ACTION_SELL, ACTION_HOLD

cfg = {'behavior_optimization':{'balance_penalty_min_actions':1}}
calc = BehavioralPenaltyCalculator(cfg)
# No prior actions
print('initial counts', calc._get_recent_counts())
print('balance_penalty_min_actions', calc.balance_penalty_min_actions)
print('calc.calculate_balance_penalty(ACTION_BUY)', calc.calculate_balance_penalty(ACTION_BUY, action_bonus=0.0))
# Try with buy in history
calc.record_action(ACTION_BUY)
calc.record_action(ACTION_BUY)
print('after recents', calc._get_recent_counts())
print('calc.calculate_balance_penalty(ACTION_BUY)', calc.calculate_balance_penalty(ACTION_BUY, action_bonus=0.0))
print('calc.calculate_balance_penalty(ACTION_SELL)', calc.calculate_balance_penalty(ACTION_SELL, action_bonus=0.0))
