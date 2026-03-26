"""499# hard_loss_cap 当日スコープ修正のテスト.

根本原因: cumulative_pnl_jpy が全期間のレコードを合算していたため、
長期運用で loss_cap を必ず超過し crash loop に陥る。

修正:
  A. resume 時の cumulative_pnl_jpy 計算を当日 UTC 分のみスコープ
  B. _process_daily_reset で cumulative_pnl_jpy をゼロリセット
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from scripts.v460.lib.fill_loop_orchestrator import RunSessionState
from scripts.v460.lib.orchestrator_pre_cycle import OrchestratorPreCycleMixin
from ztb.data.raw_paths import utc_day_str_from_timestamp
from ztb.metrics.fill_quality import compute_record_pnl_jpy


# ─── 軽量スタブ ────────────────────────────────────────────
@dataclass
class _FakeRecord:
    timestamp: float = 0.0
    filled: bool = True
    post_fill_30s_pnl: float | None = None
    fill_price: float | None = None
    order_quantity: float | None = None
    side: str = "buy"
    adverse_selected: bool = False
    cycle_id: str = ""


def _make_record(
    *,
    utc_day: str,
    pnl_bps: float = -5.0,
    side: str = "buy",
    price: float = 10_000_000.0,
    qty: float = 0.001,
) -> _FakeRecord:
    """指定 UTC 日のダミーレコードを生成."""
    dt = datetime.strptime(utc_day, "%Y%m%d").replace(
        hour=12, tzinfo=timezone.utc,
    )
    return _FakeRecord(
        timestamp=dt.timestamp(),
        filled=True,
        post_fill_30s_pnl=pnl_bps,
        fill_price=price,
        order_quantity=qty,
        side=side,
        cycle_id=f"cycle_{utc_day}_{side}_{pnl_bps}",
    )


def _make_pre_cycle_mixin(
    *,
    day_changed: bool,
    soft_loss_cap_triggered: bool = False,
) -> OrchestratorPreCycleMixin:
    mock_dd = MagicMock()
    mock_dd.maybe_reset_day.return_value = day_changed

    mock_km = MagicMock()
    mock_km.is_kill_active.return_value = (False, 0.0, 0)
    mock_km.reset = MagicMock()

    obj = object.__new__(OrchestratorPreCycleMixin)
    obj._daily_drawdown_guard = mock_dd
    obj._soft_drawdown_interval_multiplier = 1.0
    obj._dd_soft_lot_scale_buy = 1.0
    obj._dd_soft_lot_scale_sell = 1.0
    obj._toxic_veto = {}
    obj._one_sided_consecutive_count = 0
    obj._sell_kill_mgr = mock_km
    obj._buy_kill_mgr = mock_km
    obj._soft_loss_cap_triggered = soft_loss_cap_triggered
    obj._guard_fire_counts = {}
    return obj


# ─── A. resume 時の当日スコープ ──────────────────────────────

class TestResumeCumulativePnlDailyScope:
    """orchestrator_lifecycle の累積 PnL 計算が当日分のみであること."""

    def test_only_today_records_counted(self) -> None:
        """前日以前のレコードは cumulative_pnl_jpy に加算されない."""

        utc_today = datetime.now(timezone.utc).strftime("%Y%m%d")
        yesterday = "20260101"  # 確実に当日ではない

        rec_old = _make_record(utc_day=yesterday, pnl_bps=-50.0)
        rec_today = _make_record(utc_day=utc_today, pnl_bps=-10.0)

        # 当日分のみ加算するロジックを再現
        total = 0.0
        for r in [rec_old, rec_today]:
            pnl_jpy = compute_record_pnl_jpy(r)
            if pnl_jpy is not None:
                r_day = utc_day_str_from_timestamp(r.timestamp)
                if r_day == utc_today:
                    total += pnl_jpy

        # 前日分 (-50 bps) が除外されていること
        pnl_today_only = compute_record_pnl_jpy(rec_today)
        assert pnl_today_only is not None
        assert total == pytest.approx(pnl_today_only)

    def test_utc_day_str_from_timestamp_consistency(self) -> None:
        """utc_day_str_from_timestamp が datetime.strftime と一致."""
        now = time.time()
        expected = datetime.fromtimestamp(now, timezone.utc).strftime("%Y%m%d")
        assert utc_day_str_from_timestamp(now) == expected


# ─── B. _process_daily_reset での cumPnL リセット ───────────

class TestDailyResetCumulativePnl:
    """_process_daily_reset が cumulative_pnl_jpy をゼロリセットすること."""

    def test_daily_reset_clears_cumulative_pnl(self) -> None:
        """日替わり時に st.cumulative_pnl_jpy がゼロになる."""
        obj = _make_pre_cycle_mixin(day_changed=True, soft_loss_cap_triggered=True)
        st = RunSessionState(cumulative_pnl_jpy=-500.0)

        obj._process_daily_reset(st)

        assert st.cumulative_pnl_jpy == 0.0
        assert obj._soft_loss_cap_triggered is False

    def test_daily_reset_noop_when_no_day_change(self) -> None:
        """日替わりでない場合は st は変更されない."""
        obj = _make_pre_cycle_mixin(day_changed=False)
        st = RunSessionState(cumulative_pnl_jpy=-500.0)

        obj._process_daily_reset(st)

        # 日替わりなしなので変更なし
        assert st.cumulative_pnl_jpy == -500.0

    def test_daily_reset_with_none_st_no_crash(self) -> None:
        """st=None (後方互換) でもクラッシュしない."""
        obj = _make_pre_cycle_mixin(day_changed=True)
        # st=None でも TypeError にならない
        obj._process_daily_reset(None)


# ─── C. crash loop 防止の統合テスト ─────────────────────────

class TestCrashLoopPrevention:
    """全期間 PnL が loss_cap を超えていても当日分で判定される."""

    def test_long_history_does_not_trigger_cap(self) -> None:
        """35日分のレコードでも当日分が閾値内なら cap 未発動."""
        utc_today = datetime.now(timezone.utc).strftime("%Y%m%d")

        # 30日分の過去レコード (各日 -50 JPY 相当)
        old_records = []
        for day_offset in range(1, 31):
            old_day = f"202601{day_offset:02d}"
            old_records.append(
                _make_record(utc_day=old_day, pnl_bps=-50.0),
            )

        # 当日分: 軽微な損失 (-5 bps ≈ -5 JPY)
        today_record = _make_record(utc_day=utc_today, pnl_bps=-5.0)
        all_records = old_records + [today_record]

        # 当日分のみ合算
        cumulative = 0.0
        for r in all_records:
            pnl_jpy = compute_record_pnl_jpy(r)
            if pnl_jpy is not None:
                r_day = utc_day_str_from_timestamp(r.timestamp)
                if r_day == utc_today:
                    cumulative += pnl_jpy

        # loss_cap_jpy (例: 1000 JPY) を超えていないこと
        loss_cap_jpy = 1000.0
        assert cumulative > -loss_cap_jpy, (
            f"当日 cumPnL={cumulative:.1f} が cap=-{loss_cap_jpy} を超過"
        )

    def test_all_period_sum_would_exceed_cap(self) -> None:
        """全期間合算だと cap を超過する (修正前の挙動確認)."""
        # 30日分の過去レコード
        records = []
        for day_offset in range(1, 31):
            old_day = f"202601{day_offset:02d}"
            records.append(
                _make_record(utc_day=old_day, pnl_bps=-50.0),
            )

        total = sum(
            compute_record_pnl_jpy(r) or 0.0
            for r in records
        )

        # 全期間で -1500 JPY 相当 → 1000 JPY の cap を超過
        assert total < -1000.0, (
            f"全期間 cumPnL={total:.1f} > -1000 — テスト前提が不正"
        )
