from ztb.trading.environment.components.behavioral_penalty_calculator import BehavioralPenaltyCalculator

class DummyConfig:
    pass

cfg = DummyConfig()
cfg.reward_settings = {
    "behavior": {
        "emergency_intervention_enabled": True,
        "emergency_intervention_threshold": 0.1,
        "emergency_intervention_penalty": -250.0,
        "balance_penalty_min_actions": 5,
    }
}

calc = BehavioralPenaltyCalculator(cfg)
print('emergency_enabled', calc.emergency_intervention_enabled)
print('threshold', calc.emergency_intervention_threshold)
print('penalty', calc.emergency_intervention_penalty)
print('min_actions', calc.balance_penalty_min_actions)
