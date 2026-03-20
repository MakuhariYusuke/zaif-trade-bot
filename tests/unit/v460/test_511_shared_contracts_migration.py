from __future__ import annotations

from scripts.v460.lib.maker_price import (
    ImbalanceResult as LegacyImbalanceResult,
    MakerPriceResult as LegacyMakerPriceResult,
    OrderbookProvider as LegacyOrderbookProvider,
)
from scripts.v460.lib.order_monitor import (
    ExchangeAdapter as LegacyExchangeAdapter,
    OrderLike as LegacyOrderLike,
    OrderStatusLike as LegacyOrderStatusLike,
    _SkipDecisionLike as LegacySkipDecisionLike,
    _SkipGateLike as LegacySkipGateLike,
)
from scripts.v460.lib.skip_gate_evaluator import (
    SkipGateAdapter as LegacySkipGateAdapter,
    _SkipDecisionLike as LegacyEvaluatorSkipDecisionLike,
    _SkipGateLike as LegacyEvaluatorSkipGateLike,
)
from ztb.ml.skip_gate_contracts import (
    SkipGateAdapter,
    SkipDecisionLike,
    SkipGateLike,
)
from ztb.trading.execution.contracts import ExchangeAdapter, OrderLike, OrderStatusLike
from ztb.trading.pricing.contracts import ImbalanceResult, MakerPriceResult, OrderbookProvider


class TestSharedContractsMigration:
    def test_maker_price_contracts_match_canonical(self) -> None:
        assert LegacyOrderbookProvider is OrderbookProvider
        assert LegacyMakerPriceResult is MakerPriceResult
        assert LegacyImbalanceResult is ImbalanceResult

    def test_order_monitor_contracts_match_canonical(self) -> None:
        assert LegacyOrderLike is OrderLike
        assert LegacyOrderStatusLike is OrderStatusLike
        assert LegacyExchangeAdapter is ExchangeAdapter
        assert LegacySkipGateLike is SkipGateLike
        assert LegacySkipDecisionLike is SkipDecisionLike

    def test_skip_gate_evaluator_contracts_match_canonical(self) -> None:
        assert LegacySkipGateAdapter is SkipGateAdapter
        assert LegacyEvaluatorSkipGateLike is SkipGateLike
        assert LegacyEvaluatorSkipDecisionLike is SkipDecisionLike
