from ztb.trading.environment.components.reward.balance_curriculum import (
    BalanceCurriculumManager,
)
from ztb.trading.environment.utils.config import EnvironmentConfig


def test_stage_change_listener_called():
    # create minimal config stub
    cfg = type("C", (), {})()
    cfg.curriculum_stage = "forced_balance"
    bcm = BalanceCurriculumManager(cfg)

    calls = []

    def listener(**kwargs):
        calls.append(kwargs)

    bcm.add_stage_change_listener(listener)
    # Simulate progress
    bcm._progress_to_stage("balanced_transition", step=200)
    assert len(calls) == 1
    assert calls[0].get("new_stage") == "balanced_transition"

    # Simulate emergency revert
    prev_stage = bcm.current_stage
    bcm._revert_to_forced_balance()
    assert len(calls) >= 2
    assert calls[-1].get("new_stage") == "forced_balance"
    assert calls[-1].get("emergency") is True


def test_balance_curriculum_default_stage():
    config = EnvironmentConfig.from_dict({})
    bc = BalanceCurriculumManager(config, enabled=True)
    assert bc.current_stage == "action_discovery"
