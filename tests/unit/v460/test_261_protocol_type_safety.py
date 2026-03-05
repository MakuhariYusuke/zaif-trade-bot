"""261# テスト — P2-1/P2-5/P2-6/P2-7 Protocol 化 + 型安全向上.

P2-1: OrderBookLevelLike Protocol (ob_utils/ob_recorder getattr 排除)
P2-5: OrderBookSnapshot Protocol (_last_ob_snapshot 型安全化)
P2-6: config_hot_reload getattr 排除 (_fast_fill_defense 直接参照)
P2-7: BalanceAdapterProtocol (balance_checker adapter: object → Protocol)
"""
from __future__ import annotations

import inspect
import typing

import pytest
from scripts.v460.lib import balance_checker
from scripts.v460.lib.balance_checker import BalanceAdapterProtocol, BalanceChecker, _BalanceLike
from scripts.v460.lib.config_hot_reload import ConfigHotReloader
from scripts.v460.lib.maker_price import MakerPriceCalculator, OrderBookSnapshot, OrderbookProvider
from scripts.v460.lib.ob_recorder import _normalize_levels
from scripts.v460.lib.ob_utils import OrderBookLevel, OrderBookLevelLike, extract_price, extract_size


# ======================================================================
# P2-1: OrderBookLevelLike Protocol
# ======================================================================


class TestOrderBookLevelLikeProtocol:
    """261# P2-1: ob_utils の OrderBookLevelLike Protocol."""

    def test_protocol_defined(self) -> None:
        """OrderBookLevelLike Protocol が定義されている."""
        assert hasattr(OrderBookLevelLike, "price")
        assert hasattr(OrderBookLevelLike, "quantity")

    def test_runtime_checkable(self) -> None:
        """OrderBookLevelLike は runtime_checkable."""

        class MockLevel:
            @property
            def price(self) -> float:
                return 100.0

            @property
            def quantity(self) -> float:
                return 0.5

        assert isinstance(MockLevel(), OrderBookLevelLike)

    def test_extract_price_no_getattr(self) -> None:
        """extract_price に getattr(level, 'price'...) が存在しない."""
        src = inspect.getsource(extract_price)
        assert 'getattr(level, "price"' not in src

    def test_extract_size_no_nested_getattr(self) -> None:
        """extract_size にネストした getattr(...getattr...) が存在しない."""
        src = inspect.getsource(extract_size)
        # 旧: getattr(level, "quantity", getattr(level, "size", 0.0))
        assert 'getattr(level, "quantity", getattr' not in src

    def test_ob_type_alias_not_bare_object(self) -> None:
        """OrderBookLevel TypeAlias が bare 'object' でない."""
        alias_str = str(OrderBookLevel)
        # "object" のみは NG、OrderBookLevelLike を含むべき
        assert "OrderBookLevelLike" in alias_str or "object" not in alias_str

    def test_extract_price_tuple(self) -> None:
        """tuple 入力で price を正しく抽出."""
        assert extract_price((12345.0, 0.5)) == 12345.0

    def test_extract_size_tuple(self) -> None:
        """tuple 入力で size を正しく抽出."""
        assert extract_size((12345.0, 0.5)) == 0.5

    def test_extract_price_object(self) -> None:
        """Protocol 準拠 object で price を正しく抽出."""

        class Lv:
            price = 99999.0
            quantity = 1.0

        assert extract_price(Lv()) == 99999.0  # type: ignore[arg-type]

    def test_extract_size_object(self) -> None:
        """Protocol 準拠 object で quantity を正しく抽出."""

        class Lv:
            price = 99999.0
            quantity = 1.0

        assert extract_size(Lv()) == 1.0  # type: ignore[arg-type]

    def test_ob_recorder_uses_protocol_check(self) -> None:
        """ob_recorder._normalize_levels が OrderBookLevelLike isinstance を使用."""
        src = inspect.getsource(_normalize_levels)
        assert "OrderBookLevelLike" in src
        assert "isinstance(level, OrderBookLevelLike)" in src


# ======================================================================
# P2-5: OrderBookSnapshot Protocol
# ======================================================================


class TestOrderBookSnapshotProtocol:
    """261# P2-5: maker_price の OrderBookSnapshot Protocol."""

    def test_protocol_defined(self) -> None:
        """OrderBookSnapshot Protocol が定義されている."""
        assert hasattr(OrderBookSnapshot, "bids")
        assert hasattr(OrderBookSnapshot, "asks")

    def test_last_ob_snapshot_typed(self) -> None:
        """_last_ob_snapshot が OrderBookSnapshot | None 型."""
        src = inspect.getsource(MakerPriceCalculator.__init__)
        assert "OrderBookSnapshot" in src
        assert "object | None" not in src

    def test_orderbook_provider_returns_snapshot(self) -> None:
        """OrderbookProvider.get_orderbook の戻り値が OrderBookSnapshot."""
        hints = typing.get_type_hints(OrderbookProvider.get_orderbook)
        ret = hints.get("return")
        # Coroutine[Any, Any, OrderBookSnapshot] の内部型を検査
        assert ret is not None
        # 戻り値の文字列表現に OrderBookSnapshot が含まれる
        assert "object" not in str(ret), f"Return should be OrderBookSnapshot, got {ret}"


# ======================================================================
# P2-6: config_hot_reload getattr 排除
# ======================================================================


class TestConfigHotReloadGetattr:
    """261# P2-6: _fast_fill_defense への直接アクセス."""

    def test_no_getattr_for_ffd(self) -> None:
        """config_hot_reload に getattr(runner, '_fast_fill_defense'...) がない."""
        src = inspect.getsource(ConfigHotReloader)
        assert 'getattr(runner, "_fast_fill_defense"' not in src

    def test_direct_ffd_access(self) -> None:
        """_do_reload 内で runner._fast_fill_defense を直接参照."""
        src = inspect.getsource(ConfigHotReloader._do_reload)
        assert "runner._fast_fill_defense" in src


# ======================================================================
# P2-7: BalanceAdapterProtocol
# ======================================================================


class TestBalanceAdapterProtocol:
    """261# P2-7: balance_checker の adapter: object → Protocol."""

    def test_protocol_defined(self) -> None:
        """BalanceAdapterProtocol が定義されている."""
        assert hasattr(BalanceAdapterProtocol, "get_balance")
        assert hasattr(BalanceAdapterProtocol, "get_current_price")

    def test_check_signature_typed(self) -> None:
        """check() の adapter 引数が BalanceAdapterProtocol."""
        hints = typing.get_type_hints(BalanceChecker.check)
        adapter_hint = hints.get("adapter")
        assert adapter_hint is not None
        assert "BalanceAdapterProtocol" in str(adapter_hint)

    def test_no_type_ignore_union_attr(self) -> None:
        """balance_checker に type: ignore[union-attr] が残っていない."""
        src = inspect.getsource(balance_checker)
        assert "type: ignore[union-attr]" not in src

    def test_balance_like_protocol(self) -> None:
        """_BalanceLike Protocol が free プロパティを持つ."""
        assert hasattr(_BalanceLike, "free")
