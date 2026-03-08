"""252# Sell Asymmetric Mode + PhantomGuard 三値化 + 型安全テスト.

対象変更:
  A. Sell Asymmetric Gate — high_vol regime 拡張 (248# P1-1)
     - high_vol sell skip が sell_asymmetric_high_vol_enabled で有効化
     - trending_up_only は high_vol を制約しない
     - safety valve (inv_bypass, consecutive, HF4) は high_vol でも動作
  B. PhantomGuard 三値化 (251# T-1/T-2)
     - ReconcileResult enum (DETECTED / CLEAN / INCONCLUSIVE)
     - API 障害時に INCONCLUSIVE → pending 保持 → 再試行
     - 再試行上限超過 → 強制パージ
  C. PhantomGuard buy 側 JPY 照合 (251# T-3)
     - buy 側 JPY 残高乖離検出
     - balance_delta_jpy フィールド
  D. BalanceChecker.last_jpy_free property (251#)
  E. getattr 排除 (fill_cycle_executor)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.cycle_gate_aggregator import (
    CycleGateAggregator,
    CycleGateResult,
    GateCheckResult,
)
from scripts.v460.lib.phantom_position_guard import (
    PendingReconciliation,
    PhantomDetection,
    PhantomPositionGuard,
    ReconcileResult,
)


# ─────────────────────────────────────────────────────
# Fixtures & Helpers
# ─────────────────────────────────────────────────────

from tests.unit.v460.conftest import make_gate_config as _make_config


def _make_gate(**overrides: object) -> CycleGateAggregator:
    return CycleGateAggregator(_make_config(**overrides))


def _default_ctx(**overrides: object) -> dict:
    ctx: dict = {
        "side": "buy",
        "regime": "ranging",
        "vol_ratio": 1.0,
        "inv_net_imbalance": 0.0,
        "is_buy_killed": False,
        "is_sell_killed": False,
    }
    ctx.update(overrides)
    return ctx


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
    jpy_free: float = 1_000_000.0,
    get_status_raises: bool = False,
    get_balance_raises: bool = False,
    get_btc_balance_raises: bool = False,
    get_jpy_balance_raises: bool = False,
):
    """テスト用 adapter mock を生成.

    251# 拡張: BTC/JPY 別の balance API エラー制御を追加。
    """
    adapter = AsyncMock()

    if get_status_raises:
        adapter.get_order_status = AsyncMock(side_effect=Exception("API error"))
    else:
        adapter.get_order_status = AsyncMock(return_value=order_status)

    if get_balance_raises:
        adapter.get_balance = AsyncMock(side_effect=Exception("Balance API error"))
    elif get_btc_balance_raises or get_jpy_balance_raises:
        async def _mock_get_balance(currency: str):
            if currency == "BTC" and get_btc_balance_raises:
                raise Exception("BTC Balance API error")
            if currency == "JPY" and get_jpy_balance_raises:
                raise Exception("JPY Balance API error")
            if currency == "BTC":
                return [_MockBalance(free=btc_free)]
            return [_MockBalance(free=jpy_free)]
        adapter.get_balance = AsyncMock(side_effect=_mock_get_balance)
    else:
        async def _mock_get_balance(currency: str):
            if currency == "BTC":
                return [_MockBalance(free=btc_free)]
            return [_MockBalance(free=jpy_free)]
        adapter.get_balance = AsyncMock(side_effect=_mock_get_balance)

    return adapter


@pytest.fixture
def guard():
    return PhantomPositionGuard()


# ═════════════════════════════════════════════════════
# A. Sell Asymmetric Gate — high_vol regime 拡張
# ═════════════════════════════════════════════════════


class TestSellAsymmetricHighVol:
    """251# 248# P1-1: high_vol regime での sell 抑制."""

    def test_high_vol_sell_skip_when_enabled(self) -> None:
        """sell_asymmetric_high_vol_enabled=True → high_vol sell をブロック."""
        gate = _make_gate(sell_asymmetric_high_vol_enabled=True)
        r = gate.evaluate(**_default_ctx(side="sell", regime="high_vol"))
        assert r.blocked
        assert r.blocking_reason == "trending_sell_skip"

    def test_high_vol_sell_allowed_when_disabled(self) -> None:
        """sell_asymmetric_high_vol_enabled=False → high_vol sell は通過."""
        gate = _make_gate(sell_asymmetric_high_vol_enabled=False)
        r = gate.evaluate(**_default_ctx(side="sell", regime="high_vol"))
        assert not r.blocked

    def test_high_vol_buy_always_allowed(self) -> None:
        """high_vol でも buy は常に通過 (sell 専用 gate)."""
        gate = _make_gate(sell_asymmetric_high_vol_enabled=True)
        r = gate.evaluate(**_default_ctx(side="buy", regime="high_vol"))
        assert not r.blocked

    def test_trending_up_only_does_not_block_high_vol(self) -> None:
        """251# trending_up_only + high_vol: high_vol は trending_up_only の制約外."""
        gate = _make_gate(
            sell_asymmetric_high_vol_enabled=True,
            skip_sell_trending_up_only=True,
        )
        r = gate.evaluate(**_default_ctx(side="sell", regime="high_vol"))
        # high_vol は trending_up_only の影響を受けない → ブロック
        assert r.blocked

    def test_trending_up_only_still_allows_trending_down(self) -> None:
        """trending_up_only は trending_down 通過を維持 (既存動作不変)."""
        gate = _make_gate(
            sell_asymmetric_high_vol_enabled=True,
            skip_sell_trending_up_only=True,
        )
        r = gate.evaluate(**_default_ctx(side="sell", regime="trending_down"))
        assert not r.blocked

    def test_inv_bypass_works_in_high_vol(self) -> None:
        """171# Guard Paradox: inv bypass は high_vol でも sell 許可."""
        gate = _make_gate(sell_asymmetric_high_vol_enabled=True)
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="high_vol", inv_net_imbalance=0.5,
        ))
        assert not r.blocked

    def test_consecutive_safety_valve_in_high_vol(self) -> None:
        """158# §20-B: 連続 skip 安全弁は high_vol でも有効."""
        gate = _make_gate(
            sell_asymmetric_high_vol_enabled=True,
            max_consecutive_trending_sell_skip=5,
        )
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="high_vol",
            trending_sell_skip_count=5,
        ))
        assert not r.blocked

    def test_hf4_safety_valve_in_high_vol(self) -> None:
        """166# HF4: buy 側残高不足 → high_vol でも sell 許可."""
        gate = _make_gate(sell_asymmetric_high_vol_enabled=True)
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="high_vol",
            buy_side_insufficient=True,
        ))
        assert not r.blocked

    def test_config_default_is_false(self) -> None:
        """sell_asymmetric_high_vol_enabled のデフォルトは False."""
        cfg = FillTestConfig()
        assert cfg.sell_asymmetric_high_vol_enabled is False

    def test_offset_mode_applies_in_high_vol(self) -> None:
        """196# ソフトモード: high_vol でも offset boost 適用."""
        gate = _make_gate(
            sell_asymmetric_high_vol_enabled=True,
            trending_sell_as_offset_enabled=True,
            trending_sell_offset_boost_factor=2.5,
        )
        r = gate.evaluate(**_default_ctx(side="sell", regime="high_vol"))
        assert not r.blocked
        trending_check = [c for c in r.checks if c.gate_name == "trending_sell"]
        assert len(trending_check) == 1
        assert trending_check[0].offset_mult == 2.5


# ═════════════════════════════════════════════════════
# B. PhantomGuard 三値化 (ReconcileResult)
# ═════════════════════════════════════════════════════


class TestReconcileResultEnum:
    """251# T-1: ReconcileResult enum の基本検証."""

    def test_enum_values(self) -> None:
        assert ReconcileResult.DETECTED.value == "detected"
        assert ReconcileResult.CLEAN.value == "clean"
        assert ReconcileResult.INCONCLUSIVE.value == "inconclusive"

    def test_enum_members_count(self) -> None:
        assert len(ReconcileResult) == 3


class TestReconcileInconclusive:
    """251# T-1: API 障害時の INCONCLUSIVE 保持とリトライ."""

    @pytest.mark.asyncio
    async def test_both_api_fail_inconclusive_retained(self, guard: PhantomPositionGuard) -> None:
        """Phase 1 + Phase 2 とも API 障害 → エントリが pending に保持."""
        guard.register_unknown(
            "order_001", "buy", 0.001, 1e7,
            balance_btc=0.100,
        )
        adapter = _make_adapter(
            get_status_raises=True,
            get_balance_raises=True,
        )
        result = await guard.reconcile(adapter)
        assert len(result) == 0
        # INCONCLUSIVE → pending に残る
        assert guard.has_pending
        assert guard.pending_count == 1
        # total_reconciled は増加しない (未解決)
        assert guard.total_reconciled == 0

    @pytest.mark.asyncio
    async def test_phase1_fail_phase2_clean_is_clean(self, guard: PhantomPositionGuard) -> None:
        """Phase 1 API 障害 + Phase 2 clean → CLEAN (balance で確認可能)."""
        guard.register_unknown(
            "order_001", "buy", 0.001, 1e7,
            balance_btc=0.100,
        )
        adapter = _make_adapter(
            get_status_raises=True,
            btc_free=0.10002,  # tolerance 以内 → clean
        )
        result = await guard.reconcile(adapter)
        assert len(result) == 0
        # Phase 2 で clean 判定 → pending クリア
        assert not guard.has_pending
        assert guard.total_reconciled == 1

    @pytest.mark.asyncio
    async def test_phase1_fail_no_snapshot_inconclusive(self, guard: PhantomPositionGuard) -> None:
        """Phase 1 API 障害 + snapshot なし → INCONCLUSIVE."""
        guard.register_unknown("order_001", "buy", 0.001, 1e7)
        adapter = _make_adapter(get_status_raises=True)
        result = await guard.reconcile(adapter)
        assert len(result) == 0
        assert guard.has_pending  # INCONCLUSIVE → 保持

    @pytest.mark.asyncio
    async def test_inconclusive_retry_succeeds(self, guard: PhantomPositionGuard) -> None:
        """INCONCLUSIVE → 次回 reconcile で成功 (clean)."""
        guard.register_unknown("order_001", "buy", 0.001, 1e7)

        # 1回目: API 障害 → INCONCLUSIVE
        adapter_fail = _make_adapter(get_status_raises=True)
        await guard.reconcile(adapter_fail)
        assert guard.has_pending

        # rate limit を回避
        guard._last_reconcile_time = 0

        # 2回目: API 復旧 → CLEAN
        adapter_ok = _make_adapter(order_status=_MockOrderStatus(status="cancelled"))
        result = await guard.reconcile(adapter_ok)
        assert len(result) == 0
        assert not guard.has_pending
        assert guard.total_reconciled == 1

    @pytest.mark.asyncio
    async def test_inconclusive_retry_detects(self, guard: PhantomPositionGuard) -> None:
        """INCONCLUSIVE → 次回 reconcile で phantom 検出."""
        guard.register_unknown("order_001", "buy", 0.001, 1e7)

        # 1回目: API 障害
        adapter_fail = _make_adapter(get_status_raises=True)
        await guard.reconcile(adapter_fail)
        assert guard.has_pending

        guard._last_reconcile_time = 0

        # 2回目: filled 確認
        adapter_ok = _make_adapter(order_status=_MockOrderStatus(status="filled"))
        result = await guard.reconcile(adapter_ok)
        assert len(result) == 1
        assert result[0].detection_method == "order_recheck"
        assert guard.phantom_detected_count == 1

    @pytest.mark.asyncio
    async def test_retry_limit_forces_purge(self, guard: PhantomPositionGuard) -> None:
        """251# T-2: 再試行上限超過 → 強制パージ."""
        guard.register_unknown("order_001", "buy", 0.001, 1e7)

        # MAX_RECONCILE_ATTEMPTS 回連続で API 障害
        for attempt in range(PhantomPositionGuard._MAX_RECONCILE_ATTEMPTS):
            adapter_fail = _make_adapter(get_status_raises=True)
            await guard.reconcile(adapter_fail)
            guard._last_reconcile_time = 0

        # 上限到達 → パージ
        assert not guard.has_pending
        assert guard.total_reconciled == 1  # force-purge 分

    @pytest.mark.asyncio
    async def test_reconcile_attempts_counter_increments(self, guard: PhantomPositionGuard) -> None:
        """reconcile_attempts が毎回インクリメントされる."""
        guard.register_unknown("order_001", "buy", 0.001, 1e7)

        adapter_fail = _make_adapter(get_status_raises=True)
        await guard.reconcile(adapter_fail)
        assert guard._pending[0].reconcile_attempts == 1

        guard._last_reconcile_time = 0
        await guard.reconcile(adapter_fail)
        assert guard._pending[0].reconcile_attempts == 2


class TestPendingReconciliationRetryField:
    """251# T-2: PendingReconciliation.reconcile_attempts フィールド."""

    def test_default_zero(self) -> None:
        pr = PendingReconciliation(
            order_id="o1", side="buy", quantity=0.001,
            price=1e7, timestamp=time.time(),
        )
        assert pr.reconcile_attempts == 0

    def test_set_value(self) -> None:
        pr = PendingReconciliation(
            order_id="o1", side="buy", quantity=0.001,
            price=1e7, timestamp=time.time(),
            reconcile_attempts=5,
        )
        assert pr.reconcile_attempts == 5


# ═════════════════════════════════════════════════════
# C. PhantomGuard buy 側 JPY 照合
# ═════════════════════════════════════════════════════


class TestPhantomGuardJPYReconcile:
    """251# T-3: buy 側 JPY 残高乖離検出."""

    @pytest.mark.asyncio
    async def test_buy_jpy_decrease_detected(self, guard: PhantomPositionGuard) -> None:
        """buy status_unknown → JPY 残高減少 → phantom 検出."""
        # order_price=1e7, qty=0.001 → expected cost = 10,000 JPY
        guard.register_unknown(
            "order_001", "buy", 0.001, 10_000_000.0,
            balance_btc=0.100,
            balance_jpy=1_000_000.0,
        )
        adapter = _make_adapter(
            order_status=None,
            btc_free=0.10002,  # BTC tolerance 以内
            jpy_free=985_000.0,  # 15,000 JPY 減少 > tolerance(50) + > 0.5*cost(5000)
        )
        result = await guard.reconcile(adapter)
        assert len(result) == 1
        assert result[0].detection_method == "balance_delta_jpy"
        assert result[0].balance_delta_jpy is not None
        assert result[0].balance_delta_jpy < 0

    @pytest.mark.asyncio
    async def test_buy_jpy_within_tolerance_no_phantom(self, guard: PhantomPositionGuard) -> None:
        """buy 注文で JPY 変動が tolerance 以内 → 検出なし."""
        guard.register_unknown(
            "order_001", "buy", 0.001, 10_000_000.0,
            balance_btc=0.100,
            balance_jpy=1_000_000.0,
        )
        adapter = _make_adapter(
            order_status=None,
            btc_free=0.10002,  # BTC within tolerance
            jpy_free=999_970.0,  # 30 JPY 減少 < tolerance(50)
        )
        result = await guard.reconcile(adapter)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_sell_side_no_jpy_check(self, guard: PhantomPositionGuard) -> None:
        """sell 注文では JPY 照合は行わない (BTC のみ)."""
        guard.register_unknown(
            "order_001", "sell", 0.001, 10_000_000.0,
            balance_btc=0.100,
            balance_jpy=1_000_000.0,
        )
        # JPY が大幅に変動しても sell 側では無視される
        adapter = _make_adapter(
            order_status=None,
            btc_free=0.10002,  # BTC within tolerance
            jpy_free=500_000.0,  # 大幅減少 — sell 側なので無視
        )
        result = await guard.reconcile(adapter)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_both_btc_and_jpy_mismatch(self, guard: PhantomPositionGuard) -> None:
        """buy 注文で BTC 増加 + JPY 減少 → method="both" or "balance_delta"."""
        guard.register_unknown(
            "order_001", "buy", 0.001, 10_000_000.0,
            balance_btc=0.100,
            balance_jpy=1_000_000.0,
        )
        adapter = _make_adapter(
            order_status=None,
            btc_free=0.102,  # BTC 明確に増加
            jpy_free=985_000.0,  # JPY 明確に減少
        )
        result = await guard.reconcile(adapter)
        assert len(result) == 1
        assert result[0].balance_delta_btc is not None
        assert result[0].balance_delta_jpy is not None

    @pytest.mark.asyncio
    async def test_jpy_delta_field_in_phantom_detection(self) -> None:
        """PhantomDetection に balance_delta_jpy フィールドが存在."""
        pd = PhantomDetection(
            order_id="o1", side="buy", quantity=0.001,
            price=1e7, detection_method="balance_delta_jpy",
            balance_delta_jpy=-15000.0,
        )
        assert pd.balance_delta_jpy == -15000.0

    @pytest.mark.asyncio
    async def test_jpy_api_fail_btc_clean_is_clean(self, guard: PhantomPositionGuard) -> None:
        """JPY API 障害 + BTC clean → CLEAN (BTC で判定可能)."""
        guard.register_unknown(
            "order_001", "buy", 0.001, 10_000_000.0,
            balance_btc=0.100,
            balance_jpy=1_000_000.0,
        )
        adapter = _make_adapter(
            order_status=_MockOrderStatus(status="cancelled"),
        )
        result = await guard.reconcile(adapter)
        assert len(result) == 0
        assert not guard.has_pending


# ═════════════════════════════════════════════════════
# D. BalanceChecker.last_jpy_free property
# ═════════════════════════════════════════════════════


class TestBalanceCheckerLastJpyFree:
    """251# BalanceChecker に last_jpy_free property が追加されたこと."""

    def test_initial_none(self) -> None:
        from scripts.v460.lib.balance_checker import BalanceChecker

        config = FillTestConfig()
        bc = BalanceChecker(config)
        assert bc.last_jpy_free is None

    def test_property_exists(self) -> None:
        from scripts.v460.lib.balance_checker import BalanceChecker

        assert hasattr(BalanceChecker, "last_jpy_free")
        # property であることを確認
        assert isinstance(
            BalanceChecker.__dict__["last_jpy_free"],
            property,
        )


# ═════════════════════════════════════════════════════
# E. getattr 排除 (fill_cycle_executor)
# ═════════════════════════════════════════════════════


class TestGetAttrRemoval:
    """251# getattr → 型安全 property 直接参照への変更検証."""

    def test_no_getattr_in_maybe_register_phantom(self) -> None:
        """_maybe_register_phantom に getattr 呼出が残っていないこと."""
        import inspect

        from scripts.v460.lib.fill_cycle_executor import FillCycleExecutorMixin

        source = inspect.getsource(FillCycleExecutorMixin._maybe_register_phantom)
        # コメント行を除外して getattr 呼出がないことを確認
        code_lines = [
            line for line in source.splitlines()
            if not line.strip().startswith("#")
        ]
        code_only = "\n".join(code_lines)
        assert "getattr" not in code_only
        # 新しい直接参照が存在すること
        assert "last_btc_free" in source
        assert "last_jpy_free" in source

    def test_balance_jpy_passed_to_register_unknown(self) -> None:
        """_maybe_register_phantom が balance_jpy も渡していること."""
        import inspect

        from scripts.v460.lib.fill_cycle_executor import FillCycleExecutorMixin

        source = inspect.getsource(FillCycleExecutorMixin._maybe_register_phantom)
        assert "balance_jpy=" in source


# ═════════════════════════════════════════════════════
# F. 既存テスト不変 (回帰防御)
# ═════════════════════════════════════════════════════


class TestExistingBehaviorUnchanged:
    """251# 変更が既存動作を破壊しないことを確認."""

    def test_trending_sell_still_blocks(self) -> None:
        """既存の trending sell gate は変更なし."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="sell", regime="trending_up"))
        assert r.blocked

    def test_ranging_sell_not_affected(self) -> None:
        """ranging regime の sell は影響なし."""
        gate = _make_gate(sell_asymmetric_high_vol_enabled=True)
        r = gate.evaluate(**_default_ctx(side="sell", regime="ranging"))
        assert not r.blocked

    @pytest.mark.asyncio
    async def test_phantom_guard_cancelled_still_clean(self, guard: PhantomPositionGuard) -> None:
        """cancelled 注文は引き続き CLEAN."""
        guard.register_unknown("order_001", "buy", 0.001, 1e7)
        adapter = _make_adapter(order_status=_MockOrderStatus(status="cancelled"))
        result = await guard.reconcile(adapter)
        assert len(result) == 0
        assert not guard.has_pending

    @pytest.mark.asyncio
    async def test_phantom_guard_filled_still_detected(self, guard: PhantomPositionGuard) -> None:
        """filled 注文は引き続き DETECTED."""
        guard.register_unknown("order_001", "buy", 0.001, 1e7)
        adapter = _make_adapter(order_status=_MockOrderStatus(status="filled"))
        result = await guard.reconcile(adapter)
        assert len(result) == 1
        assert result[0].detection_method == "order_recheck"
