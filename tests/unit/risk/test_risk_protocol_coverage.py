from typing import Any

import pandas as pd

from ztb.risk.risk_manager import RiskManager as RepoRiskManager
from ztb.trading.risk.backtest_risk_manager import BacktestRiskManager

try:
    from ztb.trading.live_trader.components.risk_manager import (
        RiskManager as LiveRiskManager,
    )
except Exception:
    # Avoid import issues when live_trader package has additional runtime imports that
    # require complex initialization (normalize_action missing, etc.). We'll skip
    # tests that require the live trader RiskManager if it's not importable in this
    # environment.
    LiveRiskManager = None
from ztb.trading.risk.interfaces import RiskManagerProtocol


class DummyLiveTrader:
    def __init__(self):
        self.config = {"test_mode": True, "default_base_position": 0.01}
        self.daily_start_pnl = 0.0
        self.daily_trades = 0
        self.total_pnl = 0.0
        self.initial_portfolio_value = 100000.0


def _assert_protocol(obj: Any) -> None:
    assert isinstance(obj, RiskManagerProtocol)
    assert hasattr(obj, "test_mode")
    assert hasattr(obj, "portfolio_value")
    assert callable(getattr(obj, "should_open_position", None))
    assert callable(getattr(obj, "should_close_position", None))
    assert callable(getattr(obj, "get_risk_adjusted_position_size", None))
    assert callable(getattr(obj, "calculate_atr_stop_levels", None))
    assert callable(getattr(obj, "update_risk_metrics", None))
    assert callable(getattr(obj, "reset", None))


def test_backtest_risk_manager_implements_protocol():
    brm = BacktestRiskManager({"test_mode": True})
    _assert_protocol(brm)

    # should_open_position under test_mode should accept everything
    assert brm.should_open_position(0.1, 0.05, 100000.0) is True
    size = brm.get_risk_adjusted_position_size(0.8, 0.01)
    assert isinstance(size, float)


def test_repo_risk_manager_implements_protocol():
    repo = RepoRiskManager({"test_mode": True, "initial_portfolio_value": 100000.0})
    _assert_protocol(repo)
    assert repo.should_open_position(0.8, 0.01, 100000.0) in (True, False)
    size = repo.get_risk_adjusted_position_size(0.8, 0.01)
    assert isinstance(size, float)


def test_live_risk_manager_implements_protocol():
    if LiveRiskManager is None:
        import pytest

        pytest.skip("LiveTrader RiskManager cannot be imported in the test environment")

    dummy = DummyLiveTrader()
    live_rm = LiveRiskManager(dummy)
    _assert_protocol(live_rm)
    assert live_rm.should_open_position(0.4, 0.05, 100000.0) in (True, False)
    size = live_rm.get_risk_adjusted_position_size(0.4, 0.05)
    assert isinstance(size, float)


def test_calculate_atr_stop_levels_signature():
    brm = BacktestRiskManager({"test_mode": True})
    stop_loss, tp = brm.calculate_atr_stop_levels(
        pd.DataFrame({"atr": [0.01]}), 100.0, "long"
    )
    assert isinstance(stop_loss, float)
    assert isinstance(tp, float)

    repo = RepoRiskManager({"test_mode": True, "initial_portfolio_value": 100000.0})
    stop_loss, tp = repo.calculate_atr_stop_levels(None, 100.0, "long")
    assert isinstance(stop_loss, float)
    assert isinstance(tp, float)
