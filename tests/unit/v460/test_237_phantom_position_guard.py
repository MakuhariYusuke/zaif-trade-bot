"""237# PhantomPositionGuard テスト.

232# §1.6 [HIGH] 対応: status_unknown 後の phantom position 検出・再照合の
ユニットテスト。

238# セルフレビュー追加テスト:
  - TTL によるエントリ自動パージ
  - side veto (is_side_vetoed / tick_veto)
  - balance snapshot 受け渡し
  - Protocol 型安全化
  - get_metrics の side_veto_active フィールド

カバレッジ対象:
  - PhantomPositionGuard: register_unknown / reconcile / clear / metrics
  - FillRecord.pending_reconciliation フィールド
  - FillMonitorResult.order_id_for_reconciliation フィールド
  - fill_cycle_executor の pending_reconciliation 設定ロジック
  - fill_loop_orchestrator の reconcile 統合
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.v460.lib.phantom_position_guard import (
    PendingReconciliation,
    PhantomDetection,
    PhantomPositionGuard,
)


# ─────────────────────────────────────────────────────
# Fixtures & Helpers
# ─────────────────────────────────────────────────────

@dataclass
class _MockOrderStatus:
    status: str
    price: float | None = None


@dataclass
class _MockBalance:
    free: float


def _make_adapter(
    *,
    order_status: _MockOrderStatus | None = None,
    btc_free: float = 0.1,
    get_status_raises: bool = False,
    get_balance_raises: bool = False,
):
    """テスト用 adapter mock を生成."""
    adapter = AsyncMock()

    if get_status_raises:
        adapter.get_order_status = AsyncMock(side_effect=Exception("API error"))
    else:
        adapter.get_order_status = AsyncMock(return_value=order_status)

    if get_balance_raises:
        adapter.get_balance = AsyncMock(side_effect=Exception("Balance API error"))
    else:
        adapter.get_balance = AsyncMock(return_value=[_MockBalance(free=btc_free)])

    return adapter


@pytest.fixture
def guard():
    return PhantomPositionGuard()


# ─────────────────────────────────────────────────────
# A. PhantomPositionGuard 基本機能
# ─────────────────────────────────────────────────────

class TestPhantomPositionGuardBasic:
    """基本プロパティ・メソッドのテスト."""

    def test_initial_state(self, guard: PhantomPositionGuard):
        """初期状態: pending なし、phantom なし."""
        assert not guard.has_pending
        assert guard.pending_count == 0
        assert guard.phantom_detected_count == 0
        assert guard.total_reconciled == 0

    def test_register_unknown(self, guard: PhantomPositionGuard):
        """register_unknown で pending が増加."""
        guard.register_unknown(
            order_id="order_001",
            side="buy",
            quantity=0.001,
            price=10_000_000.0,
        )
        assert guard.has_pending
        assert guard.pending_count == 1

    def test_register_multiple(self, guard: PhantomPositionGuard):
        """複数登録可能."""
        for i in range(3):
            guard.register_unknown(
                order_id=f"order_{i}",
                side="buy",
                quantity=0.001,
                price=10_000_000.0,
            )
        assert guard.pending_count == 3

    def test_clear(self, guard: PhantomPositionGuard):
        """clear で全エントリ削除."""
        guard.register_unknown("o1", "buy", 0.001, 1e7)
        guard.register_unknown("o2", "sell", 0.002, 1e7)
        guard.clear()
        assert not guard.has_pending
        assert guard.pending_count == 0

    def test_metrics(self, guard: PhantomPositionGuard):
        """get_metrics が正しい dict を返す."""
        metrics = guard.get_metrics()
        assert "pending_count" in metrics
        assert "phantom_detected_count" in metrics
        assert "total_reconciled" in metrics


# ─────────────────────────────────────────────────────
# B. Reconcile — 注文ステータス再確認
# ─────────────────────────────────────────────────────

class TestReconcileOrderRecheck:
    """注文ステータス再確認による reconciliation テスト."""

    @pytest.mark.asyncio
    async def test_no_pending_returns_empty(self, guard: PhantomPositionGuard):
        """pending なしでは reconcile は空リスト."""
        adapter = _make_adapter()
        result = await guard.reconcile(adapter)
        assert result == []
        adapter.get_order_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_order_confirmed_cancelled_no_phantom(self, guard: PhantomPositionGuard):
        """注文がキャンセル確認 → ファントムなし."""
        guard.register_unknown("order_001", "buy", 0.001, 1e7)
        adapter = _make_adapter(
            order_status=_MockOrderStatus(status="cancelled"),
        )
        result = await guard.reconcile(adapter)
        assert len(result) == 0
        assert guard.phantom_detected_count == 0
        assert guard.total_reconciled == 1
        assert not guard.has_pending

    @pytest.mark.asyncio
    async def test_order_confirmed_filled_phantom_detected(self, guard: PhantomPositionGuard):
        """注文が約定確認 → ファントム検出."""
        guard.register_unknown("order_001", "buy", 0.001, 1e7)
        adapter = _make_adapter(
            order_status=_MockOrderStatus(status="filled", price=1e7),
        )
        result = await guard.reconcile(adapter)
        assert len(result) == 1
        assert result[0].detection_method == "order_recheck"
        assert result[0].order_id == "order_001"
        assert result[0].side == "buy"
        assert guard.phantom_detected_count == 1

    @pytest.mark.asyncio
    async def test_order_not_found_and_no_snapshot_no_phantom(self, guard: PhantomPositionGuard):
        """注文未検出 + バランス snapshot なし → ファントム検出不可 (not detected)."""
        guard.register_unknown("order_001", "buy", 0.001, 1e7)
        adapter = _make_adapter(order_status=None)
        result = await guard.reconcile(adapter)
        assert len(result) == 0
        assert guard.total_reconciled == 1

    @pytest.mark.asyncio
    async def test_order_status_api_error_no_crash(self, guard: PhantomPositionGuard):
        """API エラー時にクラッシュしない.

        251# 三値化: API 障害 + snapshot なし → INCONCLUSIVE (pending 保持)。
        旧動作 (即座クリア) からの意図的な変更。
        """
        guard.register_unknown("order_001", "buy", 0.001, 1e7)
        adapter = _make_adapter(get_status_raises=True)
        result = await guard.reconcile(adapter)
        # API エラー + balance snapshot なし → INCONCLUSIVE (検出せず pending 保持)
        assert len(result) == 0
        # 251# 三値化: INCONCLUSIVE は pending に残る (旧: total_reconciled == 1)
        assert guard.has_pending
        assert guard.total_reconciled == 0


# ─────────────────────────────────────────────────────
# C. Reconcile — 残高差分検出
# ─────────────────────────────────────────────────────

class TestReconcileBalanceDelta:
    """残高差分による phantom 検出テスト."""

    @pytest.mark.asyncio
    async def test_buy_balance_increase_detected(self, guard: PhantomPositionGuard):
        """buy 注文の status_unknown → BTC 残高増加 → phantom 検出."""
        guard.register_unknown(
            "order_001", "buy", 0.001, 1e7,
            balance_btc=0.100,  # 発注前 BTC
        )
        # 注文ステータスは不明のまま
        adapter = _make_adapter(
            order_status=None,
            btc_free=0.102,  # 0.002 増加 → phantom fill の兆候
        )
        result = await guard.reconcile(adapter)
        assert len(result) == 1
        assert result[0].detection_method == "balance_delta"
        assert result[0].balance_delta_btc is not None
        assert result[0].balance_delta_btc > 0

    @pytest.mark.asyncio
    async def test_sell_balance_decrease_detected(self, guard: PhantomPositionGuard):
        """sell 注文の status_unknown → BTC 残高減少 → phantom 検出."""
        guard.register_unknown(
            "order_001", "sell", 0.001, 1e7,
            balance_btc=0.100,  # 発注前
        )
        adapter = _make_adapter(
            order_status=None,
            btc_free=0.098,  # 0.002 減少 → phantom sell fill の兆候
        )
        result = await guard.reconcile(adapter)
        assert len(result) == 1
        assert result[0].detection_method == "balance_delta"

    @pytest.mark.asyncio
    async def test_balance_within_tolerance_no_phantom(self, guard: PhantomPositionGuard):
        """残高変動が tolerance 以内 → ファントムなし."""
        guard.register_unknown(
            "order_001", "buy", 0.001, 1e7,
            balance_btc=0.100,
        )
        adapter = _make_adapter(
            order_status=None,
            btc_free=0.10002,  # dust レベル → 無視
        )
        result = await guard.reconcile(adapter)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_both_method_detected(self, guard: PhantomPositionGuard):
        """注文 filled + 残高変動 → method="both"."""
        guard.register_unknown(
            "order_001", "buy", 0.001, 1e7,
            balance_btc=0.100,
        )
        adapter = _make_adapter(
            order_status=_MockOrderStatus(status="filled", price=1e7),
            btc_free=0.102,
        )
        result = await guard.reconcile(adapter)
        assert len(result) == 1
        assert result[0].detection_method == "both"

    @pytest.mark.asyncio
    async def test_balance_api_error_no_crash(self, guard: PhantomPositionGuard):
        """残高 API エラーでもクラッシュしない."""
        guard.register_unknown(
            "order_001", "buy", 0.001, 1e7,
            balance_btc=0.100,
        )
        class _Adapter:
            async def get_order_status(self, order_id: str) -> _MockOrderStatus | None:
                del order_id
                return None

            async def get_balance(self, currency: str) -> list[_MockBalance]:
                del currency
                raise Exception("Balance API error")

        adapter = _Adapter()
        result = await guard.reconcile(adapter)
        assert len(result) == 0


# ─────────────────────────────────────────────────────
# D. Rate Limiter
# ─────────────────────────────────────────────────────

class TestReconcileRateLimit:
    """reconcile の rate limit テスト."""

    @pytest.mark.asyncio
    async def test_rate_limit_blocks_rapid_calls(self, guard: PhantomPositionGuard):
        """短時間の連続呼出しは rate limit で抑制."""
        class _Adapter:
            async def get_order_status(self, order_id: str) -> _MockOrderStatus | None:
                del order_id
                return _MockOrderStatus(status="cancelled")

            async def get_balance(self, currency: str) -> list[_MockBalance]:
                del currency
                return [_MockBalance(free=0.1)]

        adapter = _Adapter()
        guard.register_unknown("o1", "buy", 0.001, 1e7)
        # 1回目: 成功
        await guard.reconcile(adapter)
        # 2回目の登録
        guard.register_unknown("o2", "buy", 0.001, 1e7)
        # 2回目の reconcile: 直後呼び出しのため rate limit で空リスト
        result = await guard.reconcile(adapter)
        assert result == []
        assert guard.has_pending  # まだ pending


# ─────────────────────────────────────────────────────
# E. FillRecord.pending_reconciliation フィールド
# ─────────────────────────────────────────────────────

class TestFillRecordPendingReconciliation:
    """FillRecord に pending_reconciliation フィールドが存在することを検証."""

    def test_field_default_none(self):
        from ztb.metrics.fill_quality import FillRecord

        record = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=1e7, order_quantity=0.001,
        )
        assert record.pending_reconciliation is None

    def test_field_set_true(self):
        from ztb.metrics.fill_quality import FillRecord

        record = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=1e7, order_quantity=0.001,
            pending_reconciliation=True,
        )
        assert record.pending_reconciliation is True

    def test_field_in_to_dict(self):
        from ztb.metrics.fill_quality import FillRecord

        record = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=1e7, order_quantity=0.001,
            pending_reconciliation=True,
        )
        d = record.to_dict()
        assert d["pending_reconciliation"] is True

    def test_field_from_dict(self):
        from ztb.metrics.fill_quality import FillRecord

        d = {
            "cycle_id": "test", "timestamp": 0.0, "side": "buy",
            "order_price": 1e7, "order_quantity": 0.001,
            "pending_reconciliation": True,
        }
        record = FillRecord.from_dict(d)
        assert record.pending_reconciliation is True


# ─────────────────────────────────────────────────────
# F. FillMonitorResult.order_id_for_reconciliation
# ─────────────────────────────────────────────────────

class TestFillMonitorResultOrderId:
    """FillMonitorResult に order_id_for_reconciliation が存在することを検証."""

    def test_default_none(self):
        from scripts.v460.lib.fill_config import FillMonitorResult

        r = FillMonitorResult()
        assert r.order_id_for_reconciliation is None

    def test_set_value(self):
        from scripts.v460.lib.fill_config import FillMonitorResult

        r = FillMonitorResult(order_id_for_reconciliation="order_xyz")
        assert r.order_id_for_reconciliation == "order_xyz"


# ─────────────────────────────────────────────────────
# G. FillTestState.phantom_guard_metrics
# ─────────────────────────────────────────────────────

class TestFillTestStatePhantomMetrics:
    """FillTestState に phantom_guard_metrics が永続化されることを検証."""

    def test_field_default_none(self):
        from scripts.v460.lib.resilience import FillTestState

        state = FillTestState()
        assert state.phantom_guard_metrics is None

    def test_field_set(self):
        from scripts.v460.lib.resilience import FillTestState

        state = FillTestState(
            phantom_guard_metrics={"pending_count": 0, "phantom_detected_count": 1, "total_reconciled": 2}
        )
        assert state.phantom_guard_metrics["phantom_detected_count"] == 1


# ─────────────────────────────────────────────────────
# H. Multiple 照合の累積カウントテスト
# ─────────────────────────────────────────────────────

class TestReconcileMultiple:
    """複数エントリの照合と累積メトリクス."""

    @pytest.mark.asyncio
    async def test_multiple_entries_all_resolved(self, guard: PhantomPositionGuard):
        """複数エントリを一括照合、すべて解消."""
        guard.register_unknown("o1", "buy", 0.001, 1e7)
        guard.register_unknown("o2", "sell", 0.002, 1e7)
        adapter = _make_adapter(
            order_status=_MockOrderStatus(status="cancelled"),
        )
        result = await guard.reconcile(adapter)
        assert len(result) == 0
        assert guard.total_reconciled == 2
        assert not guard.has_pending

    @pytest.mark.asyncio
    async def test_mixed_results_phantom_and_clean(self, guard: PhantomPositionGuard):
        """一括照合で一部 phantom, 一部 clean."""
        guard.register_unknown("o1", "buy", 0.001, 1e7)
        guard.register_unknown("o2", "sell", 0.002, 1e7)

        # o1 は filled → phantom, o2 は cancelled → clean
        call_count = 0

        async def _mock_get_status(order_id):
            nonlocal call_count
            call_count += 1
            if order_id == "o1":
                return _MockOrderStatus(status="filled", price=1e7)
            return _MockOrderStatus(status="cancelled")

        adapter = AsyncMock()
        adapter.get_order_status = _mock_get_status
        adapter.get_balance = AsyncMock(return_value=[_MockBalance(free=0.1)])

        result = await guard.reconcile(adapter)
        assert len(result) == 1
        assert result[0].order_id == "o1"
        assert guard.phantom_detected_count == 1
        assert guard.total_reconciled == 2


# ─────────────────────────────────────────────────────
# I. PendingReconciliation / PhantomDetection dataclass
# ─────────────────────────────────────────────────────

class TestDataclasses:
    """データクラスの基本検証."""

    def test_pending_reconciliation_fields(self):
        pr = PendingReconciliation(
            order_id="o1", side="buy", quantity=0.001, price=1e7,
            timestamp=time.time(),
        )
        assert pr.order_id == "o1"
        assert pr.balance_snapshot_btc is None

    def test_phantom_detection_fields(self):
        pd = PhantomDetection(
            order_id="o1", side="buy", quantity=0.001, price=1e7,
            detection_method="order_recheck",
        )
        assert pd.detection_method == "order_recheck"
        assert pd.balance_delta_btc is None

    def test_phantom_detection_with_balance(self):
        pd = PhantomDetection(
            order_id="o1", side="sell", quantity=0.002, price=9e6,
            detection_method="balance_delta",
            balance_delta_btc=-0.002,
        )
        assert pd.balance_delta_btc == -0.002


# ─────────────────────────────────────────────────────
# J. 238# TTL パージ
# ─────────────────────────────────────────────────────

class TestTTLPurge:
    """238# S-1: stale エントリの自動パージ."""

    @pytest.mark.asyncio
    async def test_stale_entries_purged(self, guard: PhantomPositionGuard):
        """TTL 超過エントリは reconcile 前にパージされる."""
        guard.register_unknown("o1", "buy", 0.001, 1e7)
        # timestamp を 500秒前に偽装 (TTL=300s)
        guard._pending[0].timestamp = time.time() - 500
        adapter = _make_adapter(order_status=_MockOrderStatus(status="cancelled"))
        result = await guard.reconcile(adapter)
        assert len(result) == 0
        assert not guard.has_pending
        # stale もカウント
        assert guard.total_reconciled == 1

    @pytest.mark.asyncio
    async def test_fresh_entries_not_purged(self, guard: PhantomPositionGuard):
        """TTL 以内のエントリはパージされない."""
        guard.register_unknown("o1", "buy", 0.001, 1e7)
        adapter = _make_adapter(order_status=_MockOrderStatus(status="cancelled"))
        result = await guard.reconcile(adapter)
        assert guard.total_reconciled == 1
        assert not guard.has_pending


# ─────────────────────────────────────────────────────
# K. 238# Side Veto
# ─────────────────────────────────────────────────────

class TestSideVeto:
    """238# S-2: phantom 検出後の同 side 一時拒否."""

    def test_initial_no_veto(self, guard: PhantomPositionGuard):
        """初期状態では veto なし."""
        assert not guard.is_side_vetoed("buy")
        assert not guard.is_side_vetoed("sell")

    @pytest.mark.asyncio
    async def test_phantom_sets_side_veto(self, guard: PhantomPositionGuard):
        """phantom 検出で同 side に veto が設定される."""
        guard.register_unknown("o1", "buy", 0.001, 1e7)
        adapter = _make_adapter(order_status=_MockOrderStatus(status="filled", price=1e7))
        await guard.reconcile(adapter)
        assert guard.is_side_vetoed("buy")
        assert not guard.is_side_vetoed("sell")

    def test_tick_veto_decrements(self, guard: PhantomPositionGuard):
        """tick_veto で veto カウンタが減算される."""
        guard._side_veto["buy"] = 3
        guard.tick_veto()
        assert guard._side_veto["buy"] == 2
        guard.tick_veto()
        assert guard._side_veto["buy"] == 1
        guard.tick_veto()
        assert not guard.is_side_vetoed("buy")

    def test_clear_resets_veto(self, guard: PhantomPositionGuard):
        """clear() で veto もリセットされる."""
        guard._side_veto["sell"] = 3
        guard.clear()
        assert not guard.is_side_vetoed("sell")


# ─────────────────────────────────────────────────────
# L. 238# Metrics 拡張
# ─────────────────────────────────────────────────────

class TestMetricsExtended:
    """238# get_metrics に side_veto_active が追加されたことを検証."""

    def test_metrics_includes_side_veto(self, guard: PhantomPositionGuard):
        metrics = guard.get_metrics()
        assert "side_veto_active" in metrics
        assert metrics["side_veto_active"] == 0

    def test_metrics_side_veto_count(self, guard: PhantomPositionGuard):
        guard._side_veto["buy"] = 2
        guard._side_veto["sell"] = 1
        metrics = guard.get_metrics()
        assert metrics["side_veto_active"] == 2


# ─────────────────────────────────────────────────────
# M. 238# BalanceChecker.last_btc_free
# ─────────────────────────────────────────────────────

class TestBalanceCheckerCache:
    """238# C-2: BalanceChecker の last_btc_free キャッシュ."""

    def test_initial_none(self):
        from scripts.v460.lib.balance_checker import BalanceChecker
        from scripts.v460.lib.fill_config import FillTestConfig

        config = FillTestConfig()
        bc = BalanceChecker(config)
        assert bc.last_btc_free is None

    @pytest.mark.asyncio
    async def test_cached_after_check_sell(self):
        from scripts.v460.lib.balance_checker import BalanceChecker
        from scripts.v460.lib.fill_config import FillTestConfig

        config = FillTestConfig()
        bc = BalanceChecker(config)

        adapter = AsyncMock()
        adapter.get_balance = AsyncMock(return_value=[_MockBalance(free=0.5)])
        await bc._check_sell(adapter, "btc_jpy")
        assert bc.last_btc_free == 0.5


# ─────────────────────────────────────────────────────
# N. 238# cancel_reasons 定数
# ─────────────────────────────────────────────────────

class TestCancelReasonPhantom:
    """238# PHANTOM_SIDE_VETO cancel reason."""

    def test_phantom_side_veto_exists(self):
        from scripts.v460.lib import cancel_reasons as CR

        assert CR.PHANTOM_SIDE_VETO == "phantom_side_veto"

    def test_phantom_side_veto_in_known(self):
        from scripts.v460.lib.cancel_reasons import AUDIT_CANCEL_REASONS

        assert "phantom_side_veto" in AUDIT_CANCEL_REASONS
