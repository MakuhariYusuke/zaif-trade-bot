from types import SimpleNamespace

from ztb.trading.environment.components.position_manager import PositionManager


def test_position_manager_cost_breakdown_fields():
    config = SimpleNamespace(
        risk_management={},
        transaction_cost=0.001,
        max_position_size=1.0,
        initial_portfolio_value=1000.0,
    )
    manager = PositionManager(config, lambda: 100.0)
    manager.realized_pnl = -50.0
    manager.total_fees = 10.0
    manager.total_slippage = 5.0

    info = manager.get_position_info()

    assert info["gross_pnl"] == -35.0
    assert info["net_pnl"] == -50.0
    assert info["total_fees"] == 10.0
    assert info["total_slippage"] == 5.0
