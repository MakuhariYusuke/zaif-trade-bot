from ztb.trading.environment.components.behavioral_penalty_calculator import BehavioralPenaltyCalculator

config = {"behavior":{"consistency_penalty":{"enabled":True, "value":0.1, "lookback":4}}}
calc = BehavioralPenaltyCalculator(config)
print('lookback', calc.lookback)
print('recent maxlen', calc.recent_actions.maxlen)
print('penalty_value', calc.penalty_value)
