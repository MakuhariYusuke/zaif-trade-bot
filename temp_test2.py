from ztb.trading.environment.components.behavioral_penalty_calculator import BehavioralPenaltyCalculator
from ztb.trading.constants import ACTION_BUY,ACTION_SELL,ACTION_HOLD
settings={'consistency_penalty_enabled': True,'consistency_penalty': 0.1,'consistency_lookback': 3,'behavior': {'consistency_penalty': {'enabled': True,'value': 0.1,'lookback': 3}}}
calc=BehavioralPenaltyCalculator(settings)
[calc.record_action(a) for a in [ACTION_BUY,ACTION_HOLD,ACTION_HOLD]]
calc.record_action(ACTION_SELL)
print('recent_actions', list(calc.recent_actions))
print('lookback', calc.lookback)
window=list(calc.recent_actions)[-calc.lookback:]
print('window', window)
last_action=None
prev_action=None
for a in reversed(window):
	if a != ACTION_HOLD:
		if last_action is None:
			last_action=a
		elif prev_action is None:
			prev_action=a
			break
print('last_action', last_action, 'prev_action', prev_action)
print('penalty_value', calc.penalty_value)
print('calc.calculate_consistency_penalty()', calc.calculate_consistency_penalty())
