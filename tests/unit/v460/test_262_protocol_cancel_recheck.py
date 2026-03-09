"""262# テスト — adaptation_engine Protocol化 + order_monitor cancel-recheck DRY.

変更内容:
- adaptation_engine: FastFillDefenseLike / LossCapAdapterProtocol / _LossCapBalanceLike Protocol 導入
  → 4 箇所の type: ignore 排除
- order_monitor: _try_cancel_with_fill_recheck ヘルパー抽出 (3重複→1)
  → _CancelFillCheck 結果クラス追加
"""
from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Optional, Sequence
from unittest.mock import AsyncMock, MagicMock

import pytest

import scripts.v460.lib.adaptation_engine as adaptation_engine_mod
from scripts.v460.lib.adaptation_engine import (
    AdaptationEngine,
    FastFillDefenseLike,
    LossCapAdapterProtocol,
    _LossCapBalanceLike,
)
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.fast_fill_defense import FastFillDefense, FastFillDefenseConfig
from scripts.v460.lib.order_monitor import OrderMonitor, _CancelFillCheck


# ======================================================================
# helpers
# ======================================================================


def _make_config(**overrides: object) -> FillTestConfig:
    defaults: dict[str, object] = dict(
        spread_offset_ratio=0.001,
        min_offset_jpy=1.0,
        max_offset_ratio=0.30,
        min_offset_ratio=0.01,
    )
    defaults.update(overrides)
    return FillTestConfig(**defaults)


# ======================================================================
# A) adaptation_engine Protocol 型安全性テスト
# ======================================================================


class TestAdaptationEngineProtocols:
    """262# adaptation_engine Protocol 定義の型安全性テスト."""

    def test_fast_fill_defense_like_protocol_exists(self) -> None:
        """FastFillDefenseLike Protocol がインポート可能."""
        assert hasattr(FastFillDefenseLike, "update_base_offsets")

    def test_loss_cap_adapter_protocol_exists(self) -> None:
        """LossCapAdapterProtocol がインポート可能."""
        # Protocol メソッドの存在確認
        assert hasattr(LossCapAdapterProtocol, "get_current_price")
        assert hasattr(LossCapAdapterProtocol, "get_balance")

    def test_loss_cap_balance_like_protocol_exists(self) -> None:
        """_LossCapBalanceLike Protocol がインポート可能."""
        assert hasattr(_LossCapBalanceLike, "currency")
        assert hasattr(_LossCapBalanceLike, "total")

    def test_try_auto_adapt_signature_uses_ffd_protocol(self) -> None:
        """try_auto_adapt の fast_fill_defense 引数が FastFillDefenseLike 型."""
        sig = inspect.signature(AdaptationEngine.try_auto_adapt)
        ffd_param = sig.parameters["fast_fill_defense"]
        ann = str(ffd_param.annotation)
        assert "FastFillDefenseLike" in ann
        assert "object" not in ann

    def test_update_dynamic_loss_cap_signature_uses_protocol(self) -> None:
        """update_dynamic_loss_cap の adapter 引数が LossCapAdapterProtocol 型."""
        sig = inspect.signature(AdaptationEngine.update_dynamic_loss_cap)
        adapter_param = sig.parameters["adapter"]
        ann = str(adapter_param.annotation)
        assert "LossCapAdapterProtocol" in ann
        assert "object" not in ann

    def test_no_type_ignore_in_adaptation_engine(self) -> None:
        """adaptation_engine.py にコード行の type: ignore が残っていないことを確認."""
        src = inspect.getsource(adaptation_engine_mod)
        # コメント行 (# で始まる行) やドキュメント文字列中の言及は除外
        code_lines = [
            line for line in src.splitlines()
            if line.strip() and not line.strip().startswith("#")
            and not line.strip().startswith('"""') and not line.strip().startswith("'''")
        ]
        for line in code_lines:
            # コードの末尾に # type: ignore がないことを確認
            code_part = line.split("#")[0] if "#" in line else line
            comment_part = line[len(code_part):] if "#" in line else ""
            assert "type: ignore" not in comment_part, f"Found type: ignore in: {line.strip()}"

    def test_fast_fill_defense_satisfies_protocol(self) -> None:
        """実際の FastFillDefense クラスが FastFillDefenseLike Protocol を満たす."""
        ffd = FastFillDefense(
            FastFillDefenseConfig(enabled=False),
            base_offset_ratio=0.05,
        )
        # Protocol の update_base_offsets が呼び出し可能
        assert callable(getattr(ffd, "update_base_offsets", None))

    def test_update_dynamic_loss_cap_with_mock_adapter(self) -> None:
        """LossCapAdapterProtocol を満たすモックで update_dynamic_loss_cap 呼び出し."""
        cfg = _make_config(
            loss_cap_ratio=0.5,
            min_loss_cap_jpy=1000.0,
            loss_cap_jpy=50000.0,
        )
        engine = AdaptationEngine(cfg, yaml_cfg={}, results_dir=Path("/tmp"))

        # モック adapter
        balance_jpy = MagicMock()
        balance_jpy.currency = "JPY"
        balance_jpy.total = 100000.0

        balance_btc = MagicMock()
        balance_btc.currency = "BTC"
        balance_btc.total = 0.01

        adapter = AsyncMock()
        adapter.get_current_price = AsyncMock(return_value=10_000_000.0)
        adapter.get_balance = AsyncMock(return_value=[balance_jpy, balance_btc])

        result = asyncio.run(
            engine.update_dynamic_loss_cap(adapter, "btc_jpy")
        )

        assert result is not None
        # 残高 = 100000 + 0.01*10000000 = 200000
        # cap = 200000 * 0.5 = 100000
        assert result == 100000.0
        adapter.get_current_price.assert_awaited_once_with("btc_jpy")
        adapter.get_balance.assert_awaited_once()

    def test_update_dynamic_loss_cap_returns_none_on_error(self) -> None:
        """adapter エラー時に None を返す."""
        cfg = _make_config(loss_cap_jpy=50000.0)
        engine = AdaptationEngine(cfg, yaml_cfg={}, results_dir=Path("/tmp"))

        adapter = AsyncMock()
        adapter.get_current_price = AsyncMock(side_effect=Exception("API error"))

        result = asyncio.run(
            engine.update_dynamic_loss_cap(adapter, "btc_jpy")
        )
        assert result is None


# ======================================================================
# B) order_monitor _CancelFillCheck + _try_cancel_with_fill_recheck テスト
# ======================================================================


class _FakeOrder:
    """テスト用注文オブジェクト."""

    def __init__(self, order_id: str = "test-123") -> None:
        self._order_id = order_id

    @property
    def order_id(self) -> str:
        return self._order_id


class _FakeStatus:
    """テスト用注文ステータス."""

    def __init__(self, status: str, price: float | None = None) -> None:
        self._status = status
        self._price = price

    @property
    def status(self) -> str:
        return self._status

    @property
    def price(self) -> float | None:
        return self._price


class TestCancelFillCheck:
    """262# _CancelFillCheck 結果クラスのテスト."""

    def test_default_values(self) -> None:
        chk = _CancelFillCheck()
        assert chk.was_filled is False
        assert chk.fill_price is None
        assert chk.t_fill is None
        assert chk.cancel_succeeded is True

    def test_filled_values(self) -> None:
        chk = _CancelFillCheck(
            was_filled=True,
            fill_price=15_000_000.0,
            t_fill=1234567890.0,
        )
        assert chk.was_filled is True
        assert chk.fill_price == 15_000_000.0
        assert chk.t_fill == 1234567890.0
        assert chk.cancel_succeeded is True

    def test_cancel_failed(self) -> None:
        chk = _CancelFillCheck(cancel_succeeded=False)
        assert chk.cancel_succeeded is False
        assert chk.was_filled is False

    def test_slots_defined(self) -> None:
        """__slots__ がメモリ効率のために定義されている."""
        assert hasattr(_CancelFillCheck, "__slots__")
        assert "was_filled" in _CancelFillCheck.__slots__
        assert "cancel_succeeded" in _CancelFillCheck.__slots__


class TestTryCancelWithFillRecheck:
    """262# _try_cancel_with_fill_recheck ヘルパーのテスト."""

    def test_cancel_success(self) -> None:
        """cancel_order 成功時: cancel_succeeded=True, was_filled=False."""
        adapter = AsyncMock()
        adapter.cancel_order = AsyncMock(return_value=None)

        result = asyncio.run(
            OrderMonitor._try_cancel_with_fill_recheck(
                adapter, "order-1", 15_000_000.0,
            )
        )
        assert result.cancel_succeeded is True
        assert result.was_filled is False
        adapter.cancel_order.assert_awaited_once_with("order-1")

    def test_cancel_failed_to_cancel_and_filled(self) -> None:
        """'Failed to cancel' エラー + fill 確認 → was_filled=True."""
        adapter = AsyncMock()
        adapter.cancel_order = AsyncMock(
            side_effect=Exception("Failed to cancel order")
        )
        adapter.get_order_status = AsyncMock(
            return_value=_FakeStatus("filled", price=15_500_000.0)
        )

        result = asyncio.run(
            OrderMonitor._try_cancel_with_fill_recheck(
                adapter, "order-2", 15_000_000.0,
            )
        )
        assert result.was_filled is True
        assert result.fill_price == 15_500_000.0
        assert result.t_fill is not None

    def test_cancel_not_found_and_filled(self) -> None:
        """'not found' エラー + fill 確認 → was_filled=True."""
        adapter = AsyncMock()
        adapter.cancel_order = AsyncMock(
            side_effect=Exception("Order not found")
        )
        adapter.get_order_status = AsyncMock(
            return_value=_FakeStatus("filled", price=14_900_000.0)
        )

        result = asyncio.run(
            OrderMonitor._try_cancel_with_fill_recheck(
                adapter, "order-3", 15_000_000.0,
            )
        )
        assert result.was_filled is True
        assert result.fill_price == 14_900_000.0

    def test_cancel_failed_to_cancel_but_not_filled(self) -> None:
        """'Failed to cancel' エラー + pending → cancel_succeeded=False."""
        adapter = AsyncMock()
        adapter.cancel_order = AsyncMock(
            side_effect=Exception("Failed to cancel")
        )
        adapter.get_order_status = AsyncMock(
            return_value=_FakeStatus("pending")
        )

        result = asyncio.run(
            OrderMonitor._try_cancel_with_fill_recheck(
                adapter, "order-4", 15_000_000.0,
            )
        )
        assert result.was_filled is False
        assert result.cancel_succeeded is False

    def test_cancel_unexpected_error(self) -> None:
        """予期しないエラー → cancel_succeeded=False, was_filled=False."""
        adapter = AsyncMock()
        adapter.cancel_order = AsyncMock(
            side_effect=Exception("Network timeout")
        )

        result = asyncio.run(
            OrderMonitor._try_cancel_with_fill_recheck(
                adapter, "order-5", 15_000_000.0,
            )
        )
        assert result.was_filled is False
        assert result.cancel_succeeded is False

    def test_cancel_failed_recheck_also_fails(self) -> None:
        """cancel + recheck 両方失敗 → cancel_succeeded=False."""
        adapter = AsyncMock()
        adapter.cancel_order = AsyncMock(
            side_effect=Exception("Failed to cancel")
        )
        adapter.get_order_status = AsyncMock(
            side_effect=Exception("Recheck also failed")
        )

        result = asyncio.run(
            OrderMonitor._try_cancel_with_fill_recheck(
                adapter, "order-6", 15_000_000.0,
            )
        )
        assert result.was_filled is False
        assert result.cancel_succeeded is False

    def test_cancel_filled_without_price_uses_fallback(self) -> None:
        """fill 確認時に price=None → fallback_price を使用."""
        adapter = AsyncMock()
        adapter.cancel_order = AsyncMock(
            side_effect=Exception("Failed to cancel order")
        )
        adapter.get_order_status = AsyncMock(
            return_value=_FakeStatus("filled", price=None)
        )

        result = asyncio.run(
            OrderMonitor._try_cancel_with_fill_recheck(
                adapter, "order-7", 15_000_000.0,
            )
        )
        assert result.was_filled is True
        assert result.fill_price == 15_000_000.0  # fallback

    def test_cancel_recheck_returns_none(self) -> None:
        """recheck が None を返す → was_filled=False."""
        adapter = AsyncMock()
        adapter.cancel_order = AsyncMock(
            side_effect=Exception("not found")
        )
        adapter.get_order_status = AsyncMock(return_value=None)

        result = asyncio.run(
            OrderMonitor._try_cancel_with_fill_recheck(
                adapter, "order-8", 15_000_000.0,
            )
        )
        assert result.was_filled is False
        assert result.cancel_succeeded is False


# ======================================================================
# C) order_monitor DRY 統合: cancel-recheck 重複排除の確認
# ======================================================================


class TestOrderMonitorCancelRecheckDRY:
    """262# order_monitor.py のキャンセル-再確認パターン重複排除テスト."""

    def test_no_duplicated_cancel_recheck_pattern(self) -> None:
        """'Failed to cancel' 文字列チェックが monitor() 内に残っていないことを確認.

        3箇所の重複パターンが _try_cancel_with_fill_recheck に統合されたため、
        monitor() メソッド本体には 'Failed to cancel' の文字列判定が無い。
        """
        src = inspect.getsource(OrderMonitor.monitor)
        assert '"Failed to cancel"' not in src
        assert "'Failed to cancel'" not in src

    def test_helper_method_exists(self) -> None:
        """_try_cancel_with_fill_recheck メソッドが存在する."""
        assert hasattr(OrderMonitor, "_try_cancel_with_fill_recheck")
        assert callable(OrderMonitor._try_cancel_with_fill_recheck)

    def test_cancel_fill_check_class_exists(self) -> None:
        """_CancelFillCheck クラスがインポート可能."""
        assert _CancelFillCheck is not None

    def test_monitor_uses_helper_for_cancel(self) -> None:
        """monitor() が _try_cancel_with_fill_recheck を使用している."""
        src = inspect.getsource(OrderMonitor.monitor)
        assert "_try_cancel_with_fill_recheck" in src

    def test_helper_is_static_method(self) -> None:
        """_try_cancel_with_fill_recheck が staticmethod (self 不要)."""
        # staticmethod なので __func__ を持たない
        method = OrderMonitor.__dict__.get("_try_cancel_with_fill_recheck")
        assert isinstance(method, staticmethod)


# ======================================================================
# D) order_monitor: except Exception 数の改善確認
# ======================================================================


class TestOrderMonitorExceptCount:
    """262# order_monitor の except Exception 数が削減されていることを確認."""

    def test_monitor_except_exception_reduced(self) -> None:
        """monitor() メソッド内の 'except Exception' 数が削減されている.

        262# 前: 11箇所 (monitor + _should_block_reprice_with_skip_gate)
        262# 後: 5箇所 (poll, mid_at_order, SkipGate, place_order, stale_check)
        ヘルパーに 2箇所 (cancel + recheck)
        """
        monitor_src = inspect.getsource(OrderMonitor.monitor)
        monitor_count = monitor_src.count("except Exception")
        # 3箇所のcancel-recheck (各2 except = 6) → helper (2) に統合
        # monitor() 本体: 11 - 6 = 5 (poll, mid_at_order, place_order, stale_check, SkipGate含まず)
        assert monitor_count <= 6, f"expect ≤6 except Exception in monitor(), got {monitor_count}"

    def test_helper_except_exception_count(self) -> None:
        """ヘルパー内の except Exception は 2箇所 (cancel + recheck)."""
        helper_src = inspect.getsource(
            OrderMonitor._try_cancel_with_fill_recheck
        )
        count = helper_src.count("except Exception")
        assert count == 2
