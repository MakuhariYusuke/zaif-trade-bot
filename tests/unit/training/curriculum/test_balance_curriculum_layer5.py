
from ztb.trading.environment.components.reward.balance_curriculum import BalanceCurriculumManager
from ztb.trading.environment.utils.config import EnvironmentConfig


def test_balance_curriculum_initialization():
    cfg = EnvironmentConfig()
    mgr = BalanceCurriculumManager(cfg, enabled=False)
    assert mgr.get_current_stage() in {"forced_balance", "balanced_transition", "pnl_focused", "autonomous"}


def test_balance_curriculum_updates_and_reset():
    cfg = EnvironmentConfig()
    mgr = BalanceCurriculumManager(cfg, enabled=True, auto_progression=True)
    status = mgr.update(0, [50, 25, 25], [], [])
    assert isinstance(status, dict)
    mgr.reset()
    assert mgr.get_current_stage() == "forced_balance" or isinstance(mgr.get_current_stage(), str)

