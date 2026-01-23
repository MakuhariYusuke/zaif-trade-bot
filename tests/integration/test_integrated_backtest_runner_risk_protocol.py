from ztb.trading.backtest.integrated_backtest_runner import IntegratedBacktestRunner
from ztb.trading.risk.interfaces import RiskManagerProtocol


def test_integrated_backtest_runner_risk_manager_protocol():
    cfg = {"risk_config": {"test_mode": True}, "n_iterations": 1}
    runner = IntegratedBacktestRunner(cfg)
    assert isinstance(runner.risk_manager, RiskManagerProtocol)
