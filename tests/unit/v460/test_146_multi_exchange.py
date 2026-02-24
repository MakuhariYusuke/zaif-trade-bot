"""146# §15: マルチ取引所基盤テスト.

Tests:
  - BrokerRegistry 型安全性・登録・検索
  - BrokerRegistry credential resolution + create_adapter (146# §16)
  - run_fill_test / run_observation のレジストリ経由化 (146# §16)
  - BitFlyer adapter async _make_request (asyncio.to_thread)
  - シンボル正規化 (btc_jpy 統一)
  - __init__.py パッケージ構造
  - ZaifAdapter 除去確認
  - CoincheckSkeletonBroker 除去確認
  - 新規取引所追加パス検証
"""

from __future__ import annotations


import inspect
import os
from pathlib import Path
from unittest.mock import patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ---------------------------------------------------------------------------
# BrokerRegistry
# ---------------------------------------------------------------------------


class TestBrokerRegistry:
    """BrokerRegistry の型安全性・登録・検索."""

    def test_default_brokers_registered(self) -> None:
        """coincheck, bitflyer がデフォルト登録される."""
        from ztb.trading.live.registry.broker_registry import BrokerRegistry

        reg = BrokerRegistry()
        names = reg.list_brokers()
        assert "coincheck" in names
        assert "bitflyer" in names

    def test_no_skeleton_or_sim(self) -> None:
        """旧 CoincheckSkeletonBroker, SimBroker は登録されない."""
        from ztb.trading.live.registry.broker_registry import BrokerRegistry

        reg = BrokerRegistry()
        names = reg.list_brokers()
        assert "sim" not in names
        assert "coincheck_skeleton" not in names

    def test_get_broker_returns_ibroker(self) -> None:
        """get_broker() が IBroker インスタンスを返す."""
        from ztb.trading.live.exchanges.base.broker_interfaces import IBroker
        from ztb.trading.live.registry.broker_registry import BrokerRegistry

        reg = BrokerRegistry()
        adapter = reg.get_broker("coincheck", dry_run=True)
        assert isinstance(adapter, IBroker)

    def test_get_broker_unknown_raises(self) -> None:
        """未登録名で ValueError."""
        from ztb.trading.live.registry.broker_registry import BrokerRegistry

        reg = BrokerRegistry()
        with pytest.raises(ValueError, match="Unknown broker"):
            reg.get_broker("nonexistent")

    def test_register_non_ibroker_raises(self) -> None:
        """IBroker 非サブクラスの登録で TypeError."""
        from ztb.trading.live.registry.broker_registry import BrokerRegistry

        reg = BrokerRegistry()
        with pytest.raises(TypeError, match="not a subclass of IBroker"):
            reg.register_broker("bad", dict)  # type: ignore[arg-type]

    def test_has_broker(self) -> None:
        """has_broker() の動作確認."""
        from ztb.trading.live.registry.broker_registry import BrokerRegistry

        reg = BrokerRegistry()
        assert reg.has_broker("coincheck") is True
        assert reg.has_broker("nonexistent") is False

    def test_custom_adapter_registration(self) -> None:
        """新規取引所アダプタを動的に登録できる."""
        from ztb.trading.live.exchanges.base.adapter import BaseExchangeAdapter
        from ztb.trading.live.exchanges.base.broker_interfaces import (
            Balance,
            Order,
            Position,
        )
        from ztb.trading.live.registry.broker_registry import BrokerRegistry

        class DummyExchangeAdapter(BaseExchangeAdapter):
            async def _get_balance_real(self, currency=None):
                return []

            async def _place_order_real(self, symbol, side, quantity, price=None,
                                         order_type="market", client_order_id=None,
                                         sizing_reason=None, target_vol=None):
                return Order(order_id="d1", symbol=symbol, side=side, quantity=quantity)

            async def _cancel_order_real(self, order_id):
                return True

            async def _get_order_status_real(self, order_id):
                return None

            async def _get_open_orders_real(self, symbol=None):
                return []

            async def _get_positions_real(self):
                return []

            async def _get_current_price_real(self, symbol):
                return 1000.0

        reg = BrokerRegistry()
        reg.register_broker("dummy", DummyExchangeAdapter)
        assert reg.has_broker("dummy")
        adapter = reg.get_broker("dummy", dry_run=True)
        assert isinstance(adapter, BaseExchangeAdapter)


# ---------------------------------------------------------------------------
# BitFlyer adapter fixes
# ---------------------------------------------------------------------------


class TestBitFlyerAdapterFixes:
    """BitFlyer adapter の修正確認."""

    def test_make_request_uses_asyncio_to_thread(self) -> None:
        """_make_request が asyncio.to_thread を使用 (イベントループ非ブロック)."""
        from ztb.trading.live.exchanges.bitflyer.adapter import BitFlyerAdapter

        src = inspect.getsource(BitFlyerAdapter._make_request)
        assert "asyncio.to_thread" in src

    def test_make_request_raises_network_error(self) -> None:
        """_make_request の docstring が NetworkError を記載."""
        from ztb.trading.live.exchanges.bitflyer.adapter import BitFlyerAdapter

        src = inspect.getsource(BitFlyerAdapter._make_request)
        assert "NetworkError" in src
        # 旧い "Exception: For API errors" は除去
        assert "Exception: For API errors" not in src

    def test_default_prices_lowercase(self) -> None:
        """BitFlyer のデフォルト価格キーが小文字 (btc_jpy)."""
        from ztb.trading.live.exchanges.bitflyer.adapter import BitFlyerAdapter

        adapter = BitFlyerAdapter(dry_run=True)
        assert "btc_jpy" in adapter._current_prices
        assert "BTC_JPY" not in adapter._current_prices


# ---------------------------------------------------------------------------
# シンボル正規化
# ---------------------------------------------------------------------------


class TestSymbolNormalization:
    """内部シンボルが小文字統一であることの確認."""

    def test_normalize_symbol(self) -> None:
        from ztb.trading.live.exchanges.base.broker_interfaces import normalize_symbol

        assert normalize_symbol("BTC_JPY") == "btc_jpy"
        assert normalize_symbol("btc_jpy") == "btc_jpy"
        assert normalize_symbol("Eth_Jpy") == "eth_jpy"

    def test_coincheck_default_prices_lowercase(self) -> None:
        from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter

        adapter = CoincheckAdapter(dry_run=True)
        # BaseExchangeAdapter のデフォルトが btc_jpy
        assert "btc_jpy" in adapter._current_prices

    def test_bitflyer_api_calls_uppercase(self) -> None:
        """BitFlyer API 呼び出しではシンボルを .upper() する."""
        from ztb.trading.live.exchanges.bitflyer.adapter import BitFlyerAdapter

        src = inspect.getsource(BitFlyerAdapter._place_order_real)
        assert "symbol.upper()" in src


# ---------------------------------------------------------------------------
# __init__.py パッケージ構造
# ---------------------------------------------------------------------------


class TestPackageInit:
    """__init__.py が存在し、主要クラスが公開されている."""

    def test_exchanges_init(self) -> None:
        import ztb.trading.live.exchanges
        assert hasattr(ztb.trading.live.exchanges, "__doc__")

    def test_base_init_exports(self) -> None:
        from ztb.trading.live.exchanges.base import (
            BaseExchangeAdapter,
            BaseExchangeConfig,
            IBroker,
            Order,
            Balance,
            Position,
            OrderBookSnapshot,
            TradeRecord,
            MarketDataNotSupported,
            normalize_symbol,
        )
        assert BaseExchangeAdapter is not None
        assert IBroker is not None

    def test_coincheck_init_exports(self) -> None:
        from ztb.trading.live.exchanges.coincheck import (
            CoincheckAdapter,
            CoincheckConfig,
        )
        assert CoincheckAdapter is not None

    def test_bitflyer_init_exports(self) -> None:
        from ztb.trading.live.exchanges.bitflyer import (
            BitFlyerAdapter,
            BitflyerConfig,
        )
        assert BitFlyerAdapter is not None


# ---------------------------------------------------------------------------
# ZaifAdapter / CoincheckSkeletonBroker 除去
# ---------------------------------------------------------------------------


class TestLegacyCleanup:
    """旧スタブ・skeleton の除去確認."""

    def test_no_zaif_adapter_in_broker_interfaces(self) -> None:
        """ZaifAdapter が broker_interfaces.py から除去されている."""
        import ztb.trading.live.exchanges.base.broker_interfaces as bi

        assert not hasattr(bi, "ZaifAdapter")

    def test_no_zaif_in_reexport(self) -> None:
        """再エクスポートモジュールに ZaifAdapter がない."""
        import ztb.trading.live.broker_interfaces as bi

        assert "ZaifAdapter" not in dir(bi)

    def test_no_skeleton_in_registry(self) -> None:
        """CoincheckSkeletonBroker が registry から除去."""
        import ztb.trading.live.registry.broker_registry as br

        assert not hasattr(br, "CoincheckSkeletonBroker")

    def test_no_broker_protocol_in_registry(self) -> None:
        """旧 BrokerProtocol (同期/非同期不整合) が除去."""
        import ztb.trading.live.registry.broker_registry as br

        assert not hasattr(br, "BrokerProtocol")


# ---------------------------------------------------------------------------
# 両アダプタ共通: BaseExchangeAdapter 継承
# ---------------------------------------------------------------------------


class TestAdapterInheritance:
    """全アダプタが BaseExchangeAdapter を継承."""

    def test_coincheck_inherits_base(self) -> None:
        from ztb.trading.live.exchanges.base.adapter import BaseExchangeAdapter
        from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter

        assert issubclass(CoincheckAdapter, BaseExchangeAdapter)

    def test_bitflyer_inherits_base(self) -> None:
        from ztb.trading.live.exchanges.base.adapter import BaseExchangeAdapter
        from ztb.trading.live.exchanges.bitflyer.adapter import BitFlyerAdapter

        assert issubclass(BitFlyerAdapter, BaseExchangeAdapter)

    def test_both_have_7_real_methods(self) -> None:
        """7 つの _*_real() 抽象メソッドが実装されている."""
        from ztb.trading.live.exchanges.bitflyer.adapter import BitFlyerAdapter
        from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter

        real_methods = [
            "_get_balance_real",
            "_place_order_real",
            "_cancel_order_real",
            "_get_order_status_real",
            "_get_open_orders_real",
            "_get_positions_real",
            "_get_current_price_real",
        ]
        for method in real_methods:
            assert hasattr(CoincheckAdapter, method), f"Coincheck missing {method}"
            assert hasattr(BitFlyerAdapter, method), f"BitFlyer missing {method}"

    def test_both_have_market_data(self) -> None:
        """get_orderbook / get_recent_trades がオーバーライドされている."""
        from ztb.trading.live.exchanges.bitflyer.adapter import BitFlyerAdapter
        from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter

        for cls in [CoincheckAdapter, BitFlyerAdapter]:
            # Should not raise MarketDataNotSupported
            assert "get_orderbook" in dir(cls)
            assert "get_recent_trades" in dir(cls)
            # Should be overridden (not the base default)
            src = inspect.getsource(cls.get_orderbook)
            assert "MarketDataNotSupported" not in src


# ---------------------------------------------------------------------------
# 146# §16 BrokerRegistry — credential resolution + create_adapter
# ---------------------------------------------------------------------------


class TestBrokerRegistryCredentials:
    """credential env map と resolve_credentials テスト."""

    def test_credential_env_map_coincheck(self) -> None:
        from ztb.trading.live.registry.broker_registry import BrokerRegistry

        reg = BrokerRegistry()
        key_var, secret_var = reg.get_credential_env_vars("coincheck")
        assert key_var == "COINCHECK_API_KEY"
        assert secret_var == "COINCHECK_API_SECRET"

    def test_credential_env_map_bitflyer(self) -> None:
        from ztb.trading.live.registry.broker_registry import BrokerRegistry

        reg = BrokerRegistry()
        key_var, secret_var = reg.get_credential_env_vars("bitflyer")
        assert key_var == "BITFLYER_API_KEY"
        assert secret_var == "BITFLYER_API_SECRET"

    def test_resolve_credentials_from_env(self) -> None:
        from ztb.trading.live.registry.broker_registry import BrokerRegistry

        reg = BrokerRegistry()
        with patch.dict(
            os.environ,
            {"COINCHECK_API_KEY": "test_key", "COINCHECK_API_SECRET": "test_secret"},
        ):
            key, secret = reg.resolve_credentials("coincheck")
            assert key == "test_key"
            assert secret == "test_secret"

    def test_resolve_credentials_empty_returns_none(self) -> None:
        from ztb.trading.live.registry.broker_registry import BrokerRegistry

        reg = BrokerRegistry()
        with patch.dict(
            os.environ,
            {"COINCHECK_API_KEY": "", "COINCHECK_API_SECRET": ""},
            clear=False,
        ):
            key, secret = reg.resolve_credentials("coincheck")
            assert key is None
            assert secret is None

    def test_unknown_broker_credential_raises(self) -> None:
        from ztb.trading.live.registry.broker_registry import BrokerRegistry

        reg = BrokerRegistry()
        with pytest.raises(ValueError, match="No credential env mapping"):
            reg.get_credential_env_vars("zaif")

    def test_custom_credential_env_on_register(self) -> None:
        """register_broker(credential_env=...) でカスタム env 登録."""
        from ztb.trading.live.exchanges.base.adapter import BaseExchangeAdapter
        from ztb.trading.live.exchanges.base.broker_interfaces import Order
        from ztb.trading.live.registry.broker_registry import BrokerRegistry

        class _Stub(BaseExchangeAdapter):
            async def _get_balance_real(self, currency=None): return []
            async def _place_order_real(self, *a, **k): return Order("x","s","b",1)
            async def _cancel_order_real(self, oid): return True
            async def _get_order_status_real(self, oid): return None
            async def _get_open_orders_real(self, s=None): return []
            async def _get_positions_real(self): return []
            async def _get_current_price_real(self, s): return 1.0

        reg = BrokerRegistry()
        reg.register_broker("test_ex", _Stub, credential_env=("TE_KEY", "TE_SEC"))
        assert reg.get_credential_env_vars("test_ex") == ("TE_KEY", "TE_SEC")


class TestBrokerRegistryCreateAdapter:
    """create_adapter() のテスト."""

    def test_create_dry_run_no_credentials(self) -> None:
        from ztb.trading.live.registry.broker_registry import BrokerRegistry

        reg = BrokerRegistry()
        adapter = reg.create_adapter("coincheck", dry_run=True)
        assert adapter is not None
        assert adapter.dry_run  # type: ignore[attr-defined]

    def test_create_live_without_credentials_raises(self) -> None:
        from ztb.trading.live.registry.broker_registry import BrokerRegistry

        reg = BrokerRegistry()
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API credentials required"):
                reg.create_adapter("coincheck", dry_run=False)

    def test_create_live_with_explicit_credentials(self) -> None:
        from ztb.trading.live.registry.broker_registry import BrokerRegistry

        reg = BrokerRegistry()
        adapter = reg.create_adapter(
            "coincheck", dry_run=False,
            api_key="ek", api_secret="es",
        )
        assert not adapter.dry_run  # type: ignore[attr-defined]

    def test_create_bitflyer_dry_run(self) -> None:
        from ztb.trading.live.registry.broker_registry import BrokerRegistry

        reg = BrokerRegistry()
        adapter = reg.create_adapter("bitflyer", dry_run=True)
        assert adapter is not None

    def test_create_unknown_raises(self) -> None:
        from ztb.trading.live.registry.broker_registry import BrokerRegistry

        reg = BrokerRegistry()
        with pytest.raises(ValueError, match="Unknown broker"):
            reg.create_adapter("zaif")


# ---------------------------------------------------------------------------
# 146# §16 run_fill_test — レジストリ経由化 + --exchange 引数
# ---------------------------------------------------------------------------


class TestRunFillTestExchangeDecoupling:
    """run_fill_test.py から CoincheckAdapter 直接参照が除去されている."""

    def test_no_coincheck_adapter_import(self) -> None:
        src = (_PROJECT_ROOT / "scripts" / "v460" / "run_fill_test.py").read_text(
            encoding="utf-8"
        )
        assert "from ztb.trading.live.exchanges.coincheck.adapter import" not in src

    def test_uses_broker_registry(self) -> None:
        src = (_PROJECT_ROOT / "scripts" / "v460" / "run_fill_test.py").read_text(
            encoding="utf-8"
        )
        assert "get_broker_registry" in src

    def test_ibbroker_type_annotation(self) -> None:
        src = (_PROJECT_ROOT / "scripts" / "v460" / "run_fill_test.py").read_text(
            encoding="utf-8"
        )
        assert "adapter: IBroker" in src

    def test_exchange_cli_arg(self) -> None:
        # 158# P2-4: CLI は fill_test_cli.py に移動
        src = (_PROJECT_ROOT / "scripts" / "v460" / "lib" / "fill_test_cli.py").read_text(
            encoding="utf-8"
        )
        assert '"--exchange"' in src

    def test_create_adapter_call(self) -> None:
        # 158# P2-4: CLI は fill_test_cli.py に移動
        src = (_PROJECT_ROOT / "scripts" / "v460" / "lib" / "fill_test_cli.py").read_text(
            encoding="utf-8"
        )
        assert "registry.create_adapter(" in src


# ---------------------------------------------------------------------------
# 146# §16 run_observation — レジストリ経由化 + --exchange 引数
# ---------------------------------------------------------------------------


class TestRunObservationExchangeDecoupling:
    """run_observation.py の CoincheckAdapter 直接参照が除去されている."""

    def test_no_coincheck_adapter_import(self) -> None:
        src = (_PROJECT_ROOT / "scripts" / "v460" / "run_observation.py").read_text(
            encoding="utf-8"
        )
        assert "from ztb.trading.live.exchanges.coincheck.adapter import" not in src

    def test_uses_broker_registry(self) -> None:
        src = (_PROJECT_ROOT / "scripts" / "v460" / "run_observation.py").read_text(
            encoding="utf-8"
        )
        assert "get_broker_registry" in src

    def test_exchange_cli_arg(self) -> None:
        src = (_PROJECT_ROOT / "scripts" / "v460" / "run_observation.py").read_text(
            encoding="utf-8"
        )
        assert '"--exchange"' in src


# ---------------------------------------------------------------------------
# 146# §16 registry __init__.py
# ---------------------------------------------------------------------------


class TestRegistryInit:
    """registry パッケージの __init__.py が正しくエクスポートする."""

    def test_registry_importable(self) -> None:
        from ztb.trading.live.registry import BrokerRegistry, get_broker_registry

        assert BrokerRegistry is not None
        assert callable(get_broker_registry)


# ---------------------------------------------------------------------------
# 146# §11 review fixes
# ---------------------------------------------------------------------------


class TestS11ReviewFixes:
    """§11 レビュー指摘修正の回帰テスト."""

    def test_create_adapter_custom_broker_dry_run_without_credential_env(self) -> None:
        """§11 #1 (HIGH): credential_env 未登録のカスタム取引所で dry-run 生成可能."""
        from ztb.trading.live.exchanges.base.adapter import BaseExchangeAdapter
        from ztb.trading.live.exchanges.base.broker_interfaces import Order
        from ztb.trading.live.registry.broker_registry import BrokerRegistry

        class _NoCredStub(BaseExchangeAdapter):
            async def _get_balance_real(self, currency=None): return []
            async def _place_order_real(self, *a, **k): return Order("x","s","b",1)
            async def _cancel_order_real(self, oid): return True
            async def _get_order_status_real(self, oid): return None
            async def _get_open_orders_real(self, s=None): return []
            async def _get_positions_real(self): return []
            async def _get_current_price_real(self, s): return 1.0

        reg = BrokerRegistry()
        # credential_env を指定せずに登録
        reg.register_broker("custom_dry", _NoCredStub)
        # dry_run=True なら credential_env なしでも生成可能
        adapter = reg.create_adapter("custom_dry", dry_run=True)
        assert adapter is not None
        assert adapter.dry_run  # type: ignore[attr-defined]

    def test_create_adapter_custom_broker_live_without_creds_raises(self) -> None:
        """§11 #1 補完: credential_env 未登録 + live → ValueError."""
        from ztb.trading.live.exchanges.base.adapter import BaseExchangeAdapter
        from ztb.trading.live.exchanges.base.broker_interfaces import Order
        from ztb.trading.live.registry.broker_registry import BrokerRegistry

        class _NoCredStub2(BaseExchangeAdapter):
            async def _get_balance_real(self, currency=None): return []
            async def _place_order_real(self, *a, **k): return Order("x","s","b",1)
            async def _cancel_order_real(self, oid): return True
            async def _get_order_status_real(self, oid): return None
            async def _get_open_orders_real(self, s=None): return []
            async def _get_positions_real(self): return []
            async def _get_current_price_real(self, s): return 1.0

        reg = BrokerRegistry()
        reg.register_broker("custom_live", _NoCredStub2)
        with pytest.raises(ValueError, match="API credentials required"):
            reg.create_adapter("custom_live", dry_run=False)

    def test_run_observation_exchange_lowercase(self) -> None:
        """§11 #2: run_observation.py が exchange 引数を lowercase 正規化する."""
        src = (_PROJECT_ROOT / "scripts" / "v460" / "run_observation.py").read_text(
            encoding="utf-8"
        )
        # .lower() が呼ばれていること
        assert ".lower()" in src
        # has_broker() 事前チェック
        assert "has_broker" in src

    def test_run_observation_unknown_exchange_exit(self) -> None:
        """§11 #2: 未知 exchange 時に sys.exit(1)."""
        src = (_PROJECT_ROOT / "scripts" / "v460" / "run_observation.py").read_text(
            encoding="utf-8"
        )
        assert "sys.exit(1)" in src


# ---------------------------------------------------------------------------
# 146# P2-04/P3-04: daily_health_check
# ---------------------------------------------------------------------------


class TestDailyHealthCheck:
    """daily_health_check.py の構造テスト."""

    def test_module_importable(self) -> None:
        """daily_health_check.py がインポート可能."""
        import importlib
        mod = importlib.import_module("scripts.v460.daily_health_check")
        assert hasattr(mod, "run_daily_health_check")
        assert hasattr(mod, "main")

    def test_run_daily_health_check_signature(self) -> None:
        """run_daily_health_check の引数シグネチャ."""
        import inspect
        from scripts.v460.daily_health_check import run_daily_health_check
        sig = inspect.signature(run_daily_health_check)
        params = list(sig.parameters.keys())
        assert "results_dir" in params
        assert "skip_monte_carlo" in params
        assert "skip_oracle" in params
        assert "output_path" in params

    def test_trades_health_integration(self) -> None:
        """_run_trades_health が呼び出し可能."""
        from scripts.v460.daily_health_check import _run_trades_health
        result = _run_trades_health(days=1)
        assert "check" in result
        assert result["check"] == "trades_health"
        assert "healthy" in result
        assert "stale_hours" in result

    def test_feature_freshness_integration(self) -> None:
        """_run_feature_freshness が呼び出し可能."""
        from scripts.v460.daily_health_check import _run_feature_freshness
        result = _run_feature_freshness()
        assert "check" in result
        assert result["check"] == "feature_freshness"
        assert "fresh" in result
        assert "trades_stale_hours" in result

    def test_ps1_runner_exists(self) -> None:
        """ops/windows/daily_health_check.ps1 が存在."""
        ps1 = _PROJECT_ROOT / "ops" / "windows" / "daily_health_check.ps1"
        assert ps1.exists()
