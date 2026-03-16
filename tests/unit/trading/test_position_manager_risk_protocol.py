from types import SimpleNamespace

from ztb.trading.environment.components.position_manager import PositionManager
from ztb.trading.risk.interfaces import RiskManagerProtocol


def get_price():
    return 100.0


def test_position_manager_risk_manager_protocol():
    config = SimpleNamespace(
        risk_management={"min_signal_strength": 0.5},
        allow_reverse=True,
        transaction_cost=0.001,
    )
    pm = PositionManager(config, get_price)
    assert isinstance(pm.risk_manager, RiskManagerProtocol)
