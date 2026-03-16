from ztb.utils.env_metrics import extract_env_metrics


class DummyEnv:
    def __init__(self) -> None:
        self.balance = 110.0
        self.initial_balance = 100.0
        self.total_trades = 3
        self.gross_pnl = 12.0
        self.total_fees = 2.0
        self.total_slippage = 1.0
        self.net_pnl = 9.0


def test_extract_env_metrics_cost_fields():
    env = DummyEnv()
    metrics = extract_env_metrics(env)

    assert metrics["final_balance"] == 110.0
    assert metrics["initial_balance"] == 100.0
    assert metrics["total_trades"] == 3
    assert metrics["gross_pnl"] == 12.0
    assert metrics["total_fees"] == 2.0
    assert metrics["total_slippage"] == 1.0
    assert metrics["net_pnl"] == 9.0
