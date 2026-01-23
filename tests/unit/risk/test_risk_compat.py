from ztb.trading.risk.backtest_risk_manager import BacktestRiskManager
from ztb.trading.risk.compat import ensure_risk_manager_protocol


def test_wrap_backtest_risk_manager():
    brm = BacktestRiskManager({"test_mode": True})
    wrapped = ensure_risk_manager_protocol(brm)
    assert wrapped.should_open_position(0.8, 0.01, 1.0) is True
    assert isinstance(wrapped.get_risk_adjusted_position_size(0.8, 0.01), float)


class LegacyRisk:
    def __init__(self):
        self.max_position_size = 0.05

    def can_open_position(self, signal_strength, market_volatility):
        return signal_strength > 0.5

    def get_position_size(self, signal_strength):
        return 0.02 if signal_strength > 0.5 else 0.01


def test_wrap_legacy_risk_manager():
    legacy = LegacyRisk()
    wrapped = ensure_risk_manager_protocol(legacy)
    assert wrapped.should_open_position(0.6, 0.1, 1.0) is True
    assert wrapped.should_open_position(0.4, 0.1, 1.0) is False
    assert wrapped.get_risk_adjusted_position_size(0.6, 0.1) <= legacy.max_position_size
