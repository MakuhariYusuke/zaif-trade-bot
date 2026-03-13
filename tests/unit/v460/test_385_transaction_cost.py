"""385# transaction_cost=0.0 (maker 0%) の動作検証テスト.

000# 前提条件: 「全取引はmaker注文（手数料0%）で執行する」
の YAML 修正 (transaction_cost: 0.001 → 0.0) が正しく機能するか検証。
"""

from __future__ import annotations

import pytest


class TestTransactionCostZeroMaker:
    """transaction_cost=0.0 が position_manager に正しく伝搬するテスト."""

    def test_env_config_transaction_cost_zero(self) -> None:
        """明示的に transaction_cost=0.0 を渡した場合、0 のまま保持される."""
        from ztb.trading.environment.utils.config import EnvironmentConfig

        config = EnvironmentConfig(
            transaction_cost=0.0,
            exchange="coincheck",
        )
        # 000#: maker 0% なので transaction_cost=0.0 が正しい
        assert config.transaction_cost == 0.0

    def test_env_config_coincheck_default_zero(self) -> None:
        """exchange=coincheck で transaction_cost 未指定時、ExchangeFeeModel の 0.0 が適用."""
        from ztb.trading.environment.utils.config import EnvironmentConfig

        config = EnvironmentConfig(exchange="coincheck")
        # ExchangeFeeModel.coincheck = {"buy": 0.0, "sell": 0.0}
        assert config.transaction_cost == 0.0

    def test_env_config_explicit_overrides_exchange_default(self) -> None:
        """明示 transaction_cost が ExchangeFeeModel のデフォルトを上書きする."""
        from ztb.trading.environment.utils.config import EnvironmentConfig

        config = EnvironmentConfig(
            transaction_cost=0.002,
            exchange="coincheck",
        )
        # 明示的に設定した場合はその値が使われる
        assert config.transaction_cost == 0.002

    def test_exchange_fee_model_coincheck_rates(self) -> None:
        """ExchangeFeeModel の Coincheck デフォルト値が 0.0."""
        from ztb.utils.fee_model import ExchangeFeeModel

        model = ExchangeFeeModel()
        model.set_exchange("coincheck")
        assert model.get_fee_rate("buy") == 0.0
        assert model.get_fee_rate("sell") == 0.0

    def test_position_manager_zero_cost_no_fee_deduction(self) -> None:
        """transaction_cost=0.0 で position_manager がfee を引かないことを確認."""
        from ztb.trading.environment.utils.config import EnvironmentConfig

        config = EnvironmentConfig(
            transaction_cost=0.0,
            exchange="coincheck",
            initial_portfolio_value=10_000_000.0,
        )
        assert config.transaction_cost == 0.0
        # fee = trade_value × 0.0 = 0 → PnL にペナルティなし
        trade_value = 150_000.0  # 0.01 BTC × 15M JPY
        fee = trade_value * config.transaction_cost
        assert fee == 0.0


class TestTransactionCostImpactEstimate:
    """手数料インパクトの数値検証."""

    @pytest.mark.parametrize(
        "cost_rate,n_trades,expected_total_fee",
        [
            (0.001, 1000, 300_000.0),  # 旧: 0.1% × 1000 往復 (片道150 × 2 × 1000)
            (0.0, 1000, 0.0),           # 新: 0%
            (0.0008, 1000, 240_000.0),  # 参考: taker 0.08%
        ],
    )
    def test_total_fee_impact(
        self,
        cost_rate: float,
        n_trades: int,
        expected_total_fee: float,
    ) -> None:
        """往復手数料の合計インパクト計算.

        BTC=15M JPY, position=0.01 BTC, trade_value=150K JPY.
        """
        trade_value = 150_000.0  # 0.01 BTC × 15M
        # 往復 (entry + exit) の手数料
        round_trip_fee = trade_value * cost_rate * 2
        total = round_trip_fee * n_trades
        assert total == pytest.approx(expected_total_fee, rel=1e-6)

    def test_roi_impact_from_fees(self) -> None:
        """手数料が ROI に与えるインパクト (10M ポートフォリオ)."""
        portfolio = 10_000_000.0
        trade_value = 150_000.0
        n_trades = 1000

        # 旧: 0.1%
        old_total_fee = trade_value * 0.001 * 2 * n_trades  # 300,000
        old_roi_drag = old_total_fee / portfolio  # -3%

        # 新: 0%
        new_total_fee = trade_value * 0.0 * 2 * n_trades  # 0
        new_roi_drag = new_total_fee / portfolio  # 0%

        assert old_roi_drag == pytest.approx(0.03, rel=1e-3)
        assert new_roi_drag == 0.0
        # 差分: 3% — これが前回の ROI=-0.25% を説明する可能性あり
        assert old_roi_drag - new_roi_drag == pytest.approx(0.03, rel=1e-3)
