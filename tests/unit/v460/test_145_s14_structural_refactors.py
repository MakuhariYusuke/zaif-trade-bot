"""145# §14 tests: §10.1 structural refactors.

§10.1-#1 CoincheckAdapter → BaseExchangeAdapter inheritance
§10.1-#2 FillTestRunner → AbstractCycleRunner inheritance  
§10.1-#3 MarketDataAccessor in ob_utils.py
"""
from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from scripts.v460.lib.abstract_cycle_runner import AbstractCycleRunner
from scripts.v460.lib.ob_utils import MarketDataAccessor
from scripts.v460.run_fill_test import FillTestRunner
from tests.unit.v460._fill_test_source import (
    FILL_CYCLE_EXECUTOR,
    read_class_method_source,
    read_fill_test_method_source,
    read_source_text,
)
from ztb.trading.live.exchanges.base.adapter import BaseExchangeAdapter
from ztb.trading.live.exchanges.base.broker_interfaces import IBroker
from ztb.trading.live.exchanges.coincheck import adapter as coincheck_adapter_module
from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter
from ztb.utils.errors import OrderNotFoundError

COINCHECK_ADAPTER_SOURCE = Path(coincheck_adapter_module.__file__).resolve()
_RUN_SINGLE_CYCLE_SOURCE = read_fill_test_method_source("run_single_cycle")
_MAKE_API_REQUEST_SOURCE = read_class_method_source(
    COINCHECK_ADAPTER_SOURCE, "CoincheckAdapter", "_make_api_request"
)
_CREATE_SIGNATURE_SOURCE = read_class_method_source(
    COINCHECK_ADAPTER_SOURCE, "CoincheckAdapter", "_create_signature"
)
_PLACE_ORDER_REAL_SOURCE = read_class_method_source(
    COINCHECK_ADAPTER_SOURCE, "CoincheckAdapter", "_place_order_real"
)
_RUNNER_INIT_SIG = inspect.signature(FillTestRunner.__init__)


# =====================================================================
# §10.1-#1: CoincheckAdapter → BaseExchangeAdapter
# =====================================================================

class TestCoincheckBaseExchangeInheritance:
    """CoincheckAdapter が BaseExchangeAdapter を正しく継承している."""

    def test_inherits_base_exchange_adapter(self) -> None:
        assert issubclass(CoincheckAdapter, BaseExchangeAdapter)

    def test_inherits_ibroker(self) -> None:
        assert issubclass(CoincheckAdapter, IBroker)

    def test_dry_run_place_order_uses_base_simulation(self) -> None:
        """dry_run order は BaseExchangeAdapter の _place_order_dry_run を使用."""

        adapter = CoincheckAdapter(dry_run=True, random_seed=42)
        order = asyncio.run(adapter.place_order(
            symbol="btc_jpy", side="buy", quantity=0.01,
            price=5000000, order_type="limit",
        ))
        assert order.order_id is not None
        assert order.symbol == "btc_jpy"

    def test_dry_run_get_balance(self) -> None:
        """dry_run balance は BaseExchangeAdapter の _get_balance_dry_run を使用."""

        adapter = CoincheckAdapter(dry_run=True)
        balances = asyncio.run(adapter.get_balance())
        currencies = {b.currency for b in balances}
        assert "JPY" in currencies
        assert "BTC" in currencies

    def test_dry_run_cancel_order_not_found(self) -> None:
        """dry_run cancel で存在しないオーダーは OrderNotFoundError."""

        adapter = CoincheckAdapter(dry_run=True)
        with pytest.raises(OrderNotFoundError):
            asyncio.run(adapter.cancel_order("nonexistent-id"))

    def test_has_all_real_methods(self) -> None:
        """7 つの _xxx_real 抽象メソッドが全て実装されている."""

        required = [
            "_place_order_real", "_cancel_order_real",
            "_get_order_status_real", "_get_open_orders_real",
            "_get_positions_real", "_get_balance_real",
            "_get_current_price_real",
        ]
        for method_name in required:
            method = getattr(CoincheckAdapter, method_name, None)
            assert method is not None, f"Missing: {method_name}"
            assert callable(method), f"Not callable: {method_name}"

    def test_generate_order_id_is_uuid(self) -> None:
        """CoincheckAdapter の _generate_order_id は UUID 形式."""

        adapter = CoincheckAdapter(dry_run=True)
        oid = adapter._generate_order_id()
        parts = oid.split("-")
        assert len(parts) == 5, f"UUID format expected, got: {oid}"

    def test_rate_limit_4_req_per_sec(self) -> None:
        """requests_per_second=4.0 が BaseExchangeAdapter に渡る."""

        adapter = CoincheckAdapter(dry_run=True)
        assert adapter.rate_limiter.config.requests_per_second == 4.0

    def test_get_current_price_overridden(self) -> None:
        """get_current_price は CoincheckAdapter でオーバーライドされている."""

        # CoincheckAdapter.get_current_price should NOT be the same as base
        cc_method = CoincheckAdapter.get_current_price
        base_method = BaseExchangeAdapter.get_current_price
        assert cc_method is not base_method

    def test_coincheck_specific_methods_preserved(self) -> None:
        """Coincheck 固有メソッドが保持されている."""

        assert hasattr(CoincheckAdapter, "_create_signature")
        assert hasattr(CoincheckAdapter, "_make_api_request")
        assert hasattr(CoincheckAdapter, "get_orderbook")
        assert hasattr(CoincheckAdapter, "get_recent_trades")


# =====================================================================
# §10.1-#2: FillTestRunner → AbstractCycleRunner
# =====================================================================

class TestFillTestRunnerAbstractInheritance:
    """FillTestRunner が AbstractCycleRunner を継承している."""

    def test_inherits_abstract_cycle_runner(self) -> None:
        assert issubclass(FillTestRunner, AbstractCycleRunner)

    def test_abstract_methods_implemented(self) -> None:
        """run_single_cycle, run_continuous が実装されている."""

        assert hasattr(FillTestRunner, "run_single_cycle")
        assert hasattr(FillTestRunner, "run_continuous")
        # Not abstract (no __isabstractmethod__)
        assert not getattr(FillTestRunner.run_single_cycle, "__isabstractmethod__", False)
        assert not getattr(FillTestRunner.run_continuous, "__isabstractmethod__", False)

    def test_abc_has_hook_methods(self) -> None:
        """AbstractCycleRunner にフック (on_cycle_start, should_skip_cycle) がある."""

        assert hasattr(AbstractCycleRunner, "on_cycle_start")
        assert hasattr(AbstractCycleRunner, "on_cycle_end")
        assert hasattr(AbstractCycleRunner, "should_skip_cycle")

    def test_abc_has_common_utilities(self) -> None:
        """AbstractCycleRunner に _new_cycle_id, _get_git_sha がある."""

        assert hasattr(AbstractCycleRunner, "_new_cycle_id")
        assert hasattr(AbstractCycleRunner, "_get_git_sha")

    def test_new_cycle_id_format(self) -> None:
        """_new_cycle_id は timestamp_uuid 形式."""

        cid = AbstractCycleRunner._new_cycle_id()
        parts = cid.split("_")
        assert len(parts) == 2
        assert parts[0].isdigit()

    def test_new_cycle_id_prefix(self) -> None:
        """_new_cycle_id(prefix) はプレフィクス付き."""

        cid = AbstractCycleRunner._new_cycle_id(prefix="test")
        assert cid.startswith("test_")

    def test_runner_still_has_init_params(self) -> None:
        """FillTestRunner.__init__ は adapter, config を受け取る."""
        params = list(_RUNNER_INIT_SIG.parameters.keys())
        assert "adapter" in params
        assert "config" in params


# =====================================================================
# §10.1-#3: MarketDataAccessor
# =====================================================================

class TestMarketDataAccessor:
    """ob_utils.py の MarketDataAccessor."""

    def test_import(self) -> None:
        assert MarketDataAccessor is not None

    def test_best_bid_ask_from_tuple_ob(self) -> None:
        """tuple 形式の OB から best bid/ask を取得."""

        mock_ob = MagicMock()
        mock_ob.bids = [(5000000.0, 0.1)]
        mock_ob.asks = [(5001000.0, 0.2)]

        mock_adapter = MagicMock()
        mock_adapter.get_orderbook = AsyncMock(return_value=mock_ob)

        accessor = MarketDataAccessor(mock_adapter, "btc_jpy")
        bid, ask = asyncio.run(accessor.best_bid_ask())
        assert bid == 5000000.0
        assert ask == 5001000.0

    def test_spread(self) -> None:
        """spread = ask - bid."""

        mock_ob = MagicMock()
        mock_ob.bids = [(5000000.0, 0.1)]
        mock_ob.asks = [(5001000.0, 0.2)]

        mock_adapter = MagicMock()
        mock_adapter.get_orderbook = AsyncMock(return_value=mock_ob)

        accessor = MarketDataAccessor(mock_adapter, "btc_jpy")
        spread = asyncio.run(accessor.spread())
        assert spread == 1000.0

    def test_mid_price(self) -> None:
        """mid_price = (bid + ask) / 2."""

        mock_ob = MagicMock()
        mock_ob.bids = [(5000000.0, 0.1)]
        mock_ob.asks = [(5002000.0, 0.2)]

        mock_adapter = MagicMock()
        mock_adapter.get_orderbook = AsyncMock(return_value=mock_ob)

        accessor = MarketDataAccessor(mock_adapter, "btc_jpy")
        mid = asyncio.run(accessor.mid_price())
        assert mid == 5001000.0

    def test_api_failure_returns_none(self) -> None:
        """API エラー時は (None, None)."""

        mock_adapter = MagicMock()
        mock_adapter.get_orderbook = AsyncMock(side_effect=Exception("API down"))

        accessor = MarketDataAccessor(mock_adapter, "btc_jpy")
        bid, ask = asyncio.run(accessor.best_bid_ask())
        assert bid is None
        assert ask is None

    def test_depth_volume_methods(self) -> None:
        """bid/ask depth volume が計算される."""

        mock_ob = MagicMock()
        mock_ob.bids = [(5000000.0, 0.1), (4999000.0, 0.2), (4998000.0, 0.3)]
        mock_ob.asks = [(5001000.0, 0.4), (5002000.0, 0.5)]

        mock_adapter = MagicMock()
        mock_adapter.get_orderbook = AsyncMock(return_value=mock_ob)

        accessor = MarketDataAccessor(mock_adapter, "btc_jpy")
        bid_vol = asyncio.run(accessor.bid_depth_volume(depth=3))
        ask_vol = asyncio.run(accessor.ask_depth_volume(depth=2))
        assert abs(bid_vol - 0.6) < 1e-9
        assert abs(ask_vol - 0.9) < 1e-9


# =====================================================================
# §10.1-#3: run_fill_test.py の OB inline 解消確認
# =====================================================================

class TestOBInlineElimination:
    """run_fill_test.py の板データ inline アクセスが ob_utils に委譲済み."""

    def test_no_hasattr_price_in_postonly_guard(self) -> None:
        """postonly guard に hasattr(... 'price') パターンが残っていない."""
        assert "hasattr(_pre_ob.bids[0], 'price')" not in _RUN_SINGLE_CYCLE_SOURCE, (
            "Inline dual-format access should be replaced with ob_utils.best_bid_ask()"
        )

    def test_best_bid_ask_import_in_source(self) -> None:
        """fill_cycle_executor 側で best_bid_ask import が維持されている."""
        assert "best_bid_ask" in read_source_text(FILL_CYCLE_EXECUTOR)


# =====================================================================
# §14: CoincheckAdapter テスト互換性 (145# §13 migration)
# =====================================================================

class TestCoincheckAdapterTestCompat:
    """既存テストの inspect.getsource 互換を確認."""

    def test_make_api_request_still_on_coincheck(self) -> None:
        """_make_api_request は CoincheckAdapter に残存."""
        assert "urlencode" in _MAKE_API_REQUEST_SOURCE  # C-3 signature fix

    def test_create_signature_still_on_coincheck(self) -> None:
        """_create_signature は CoincheckAdapter に残存."""
        assert "hmac" in _CREATE_SIGNATURE_SOURCE

    def test_place_order_real_has_post_only(self) -> None:
        """_place_order_real に post_only がある."""
        assert "post_only" in _PLACE_ORDER_REAL_SOURCE
