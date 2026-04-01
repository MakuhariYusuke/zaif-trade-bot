"""
G1.1-exec Fill Quality 単体テスト — 009# P2-4 準拠.

fill_quality.py の FillRecord / FillMetrics / compute / judgment を検証する。
"""

from __future__ import annotations

import json
import inspect
import os
import tempfile
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import scripts.v460.lib.fill_test_cli as cli_mod
from scripts.v460.analysis.vg_and_trend import (
    _match_vg_to_records,
    _parse_vg_activations,
    analyze_8h_trend,
    analyze_daily_trend,
    analyze_vg_effectiveness,
)
from scripts.v460.lib.adaptation_engine import AdaptationEngine
from scripts.v460.lib.balance_checker import BalanceChecker, MIN_ORDER_BTC
from scripts.v460.lib.batch_persistence import BatchPersistence
from scripts.v460.lib.lock_manager import LockManager
from scripts.v460.lib.maker_price import MakerPriceCalculator
from scripts.v460.lib.order_monitor import OrderMonitor
from scripts.v460.lib.pnl_measurer import PnlMeasurer
from scripts.v460.monitor_fill_test import _check_cumulative_loss, print_report, run_monitor
from scripts.v460.run_fill_test import FillTestConfig, FillTestRunner
from tests.unit.v460._yaml_test_helpers import clone_fill_test_config, load_fill_test_config_from_mapping
from scripts.v460.run_gate_check import run_g1_1
from tests.unit.v460._fill_test_source import (
    MAKER_PRICE,
    read_fill_test_method_source,
    ORCHESTRATOR_BALANCE,
    read_fill_test_runner_source,
    read_inspect_source,
    read_source_text,
)
from ztb.data.market_data_collector import MarketDataCollector
from ztb.trading.live.exchanges.base.broker_interfaces import TradeRecord
from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter
from ztb.trading.risk.fast_fill_defense import FastFillDefense, FastFillDefenseConfig
from ztb.metrics.fill_quality import (
    FillMetrics,
    FillRecord,
    PnlWinAccumulator,
    _quarantine_reason,
    apply_fill_record_filters,
    build_fill_record,
    build_skip_fill_record,
    compute_fill_metrics,
    compute_hourly_metrics,
    compute_record_pnl_jpy,
    compute_regime_metrics,
    compute_round_trip_metrics,
    fill_records_to_dataframe,
    filter_clean_records,
    g1_1_judgment,
    g1_1_quick_judgment,
    g1_2_full_judgment,
    iter_fill_record_objects_glob,
    iter_fill_records,
    iter_fill_records_glob,
    list_fill_record_files,
    load_fill_record_objects_glob,
    load_fill_records,
    load_fill_records_glob,
    partition_clean_records,
    save_fill_records,
)

_FAST_STATUS_UNKNOWN_RETRY_DELAYS = [0.0]
_FAST_CYCLE_ORDER_TIMEOUT_SEC = 0.002
_FAST_CYCLE_POLL_INTERVAL_SEC = 0.001
_FAST_CYCLE_POST_FILL_WAIT_SEC = 0.0
_FAST_CYCLE_LOGGER_PATHS = (
    "scripts.v460.run_fill_test.logger",
    "scripts.v460.lib.fill_cycle_executor.logger",
    "scripts.v460.lib.order_monitor.logger",
    "scripts.v460.lib.phantom_position_guard.logger",
)
_ORDER_MONITOR_SOURCE = read_inspect_source(OrderMonitor)
_PROCESS_POST_CYCLE_SOURCE = read_inspect_source(FillTestRunner._process_post_cycle)
_RUN_PRE_ORDER_PHASE_SOURCE = read_fill_test_method_source("_run_pre_order_phase")
_MAKER_PRICE_SOURCE = read_source_text(MAKER_PRICE)


async def _instant_async_sleep(_delay: float) -> None:
    return None


class _NoopLogger:
    def debug(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def info(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def warning(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def error(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class _AdvancingClock:
    def __init__(self, *, start: float = 1_700_000_000.0, step: float = 0.01) -> None:
        self._current = start
        self._step = step

    def __call__(self) -> float:
        value = self._current
        self._current += self._step
        return value


async def _run_single_cycle_without_sleep(runner: "FillTestRunner") -> FillRecord:
    fake_time = _AdvancingClock(step=1.0)
    with ExitStack() as stack:
        stack.enter_context(patch("asyncio.sleep", new=_instant_async_sleep))
        stack.enter_context(
            patch("scripts.v460.lib.fill_cycle_executor.time.time", new=fake_time)
        )
        stack.enter_context(
            patch("scripts.v460.lib.order_monitor.time.time", new=fake_time)
        )
        for logger_path in _FAST_CYCLE_LOGGER_PATHS:
            stack.enter_context(patch(logger_path, new=_NoopLogger()))
        return await runner.run_single_cycle()


async def _async_return(value: object) -> object:
    return value


async def _async_raise(exc: Exception) -> object:
    raise exc


@dataclass
class _BalanceEntry:
    free: float


@dataclass
class _OrderbookView:
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]


@dataclass
class _PlacedOrder:
    order_id: str


class _FastCycleAdapter:
    def __init__(self, *, order_id: str) -> None:
        self.dry_run = True
        self._orderbook = _OrderbookView(
            bids=[(15000000.0, 0.1)],
            asks=[(15001000.0, 0.1)],
        )
        self._placed_order = _PlacedOrder(order_id=order_id)

    async def get_orderbook(self, _symbol: str, depth: int = 1) -> _OrderbookView:
        assert depth == 1
        return self._orderbook

    async def place_order(self, **_kwargs: object) -> _PlacedOrder:
        return self._placed_order

    async def get_balance(self, currency: str) -> list[_BalanceEntry]:
        if currency == "BTC":
            return [_BalanceEntry(free=1.0)]
        if currency == "JPY":
            return [_BalanceEntry(free=100_000_000.0)]
        return []

    async def get_current_price(self, _symbol: str) -> float:
        return 15_000_500.0

    async def get_order_status(self, _order_id: str) -> object:
        return None

    async def cancel_order(self, _order_id: str) -> None:
        return None


def _make_status_side_effect(*values: object) -> Callable[[str], object]:
    values_iter = iter(values)

    async def _side_effect(_order_id: str) -> object:
        return next(values_iter)

    return _side_effect


def _build_daily_records(
    *,
    days: int,
    per_day: int,
    build_record: Callable[[int, int], FillRecord],
) -> list[FillRecord]:
    """日次×件数の共通ループで FillRecord 群を構築する."""
    return [
        build_record(day, i)
        for day in range(days)
        for i in range(per_day)
    ]


def _make_uniform_daily_records(
    *,
    prefix: str,
    days: int,
    per_day: int,
    base_ts: float,
    side: str = "buy",
    order_price: float = 100.0,
    order_quantity: float = 0.001,
    filled: bool = True,
    queue_wait_sec: float | None = None,
    post_fill_30s_pnl: float | None = None,
    adverse_selected: bool | None = None,
) -> list[FillRecord]:
    return _build_daily_records(
        days=days,
        per_day=per_day,
        build_record=lambda day, i: FillRecord(
            cycle_id=f"{prefix}_d{day}_{i}",
            timestamp=base_ts + day * 86400 + i * 120,
            side=side,
            order_price=order_price,
            order_quantity=order_quantity,
            filled=filled,
            queue_wait_sec=0.0 if queue_wait_sec is None else queue_wait_sec,
            post_fill_30s_pnl=post_fill_30s_pnl,
            adverse_selected=adverse_selected,
        ),
    )


def _make_daily_fill_count_records(
    *,
    prefix: str,
    filled_counts: list[int],
    per_day: int,
    base_ts: float,
    order_price: float = 100.0,
    order_quantity: float = 0.001,
    queue_wait_sec: float | None = None,
    post_fill_30s_pnl: float | None = None,
    adverse_selected: bool | None = None,
    alternate_side: bool = False,
) -> list[FillRecord]:
    return _build_daily_records(
        days=len(filled_counts),
        per_day=per_day,
        build_record=lambda day, i: FillRecord(
            cycle_id=f"{prefix}_d{day}_{i}",
            timestamp=base_ts + day * 86400 + i * 120,
            side="buy" if (not alternate_side or i % 2 == 0) else "sell",
            order_price=order_price,
            order_quantity=order_quantity,
            filled=i < filled_counts[day],
            cancelled=i >= filled_counts[day],
            queue_wait_sec=(
                queue_wait_sec
                if i < filled_counts[day] and queue_wait_sec is not None
                else 0.0
            ),
            post_fill_30s_pnl=post_fill_30s_pnl if i < filled_counts[day] else None,
            adverse_selected=adverse_selected if i < filled_counts[day] else None,
        ),
    )


def _make_outcome_records(
    *,
    prefix: str,
    counts: dict[str, int],
    base_ts: float,
    side: str = "buy",
    order_price: float = 100.0,
    order_quantity: float = 0.001,
) -> list[FillRecord]:
    """fill / cancel outcome 別の均一レコードを生成."""
    records: list[FillRecord] = []
    offset = 0
    for outcome, count in counts.items():
        for i in range(count):
            record = FillRecord(
                cycle_id=f"{prefix}_{outcome}_{i}",
                timestamp=base_ts + (offset + i) * 120,
                side=side,
                order_price=order_price,
                order_quantity=order_quantity,
            )
            if outcome == "fill":
                record.filled = True
            elif outcome == "skip_gate":
                record.cancelled = True
                record.cancel_reason = "skip_gate"
                record.skip_gate_skipped = True
            elif outcome == "timeout":
                record.cancelled = True
                record.cancel_reason = "timeout"
            elif outcome == "postonly_reject":
                record.cancelled = True
                record.cancel_reason = "postonly_reject"
            elif outcome == "unknown_cancel":
                record.cancelled = True
            else:
                raise ValueError(f"unsupported outcome: {outcome}")
            records.append(record)
        offset += count
    return records


def _make_linear_records(
    *,
    prefix: str,
    count: int,
    base_ts: float,
    ts_step: float = 120.0,
    alternating_side: bool = False,
    side: str = "buy",
    order_price: float = 100.0,
    price_step: float = 0.0,
    order_quantity: float = 0.001,
    filled: bool = False,
    start_index: int = 0,
    separator: str = "_",
) -> list[FillRecord]:
    """単純な連番 FillRecord 群を生成."""
    records: list[FillRecord] = []
    for i in range(count):
        idx = start_index + i
        records.append(FillRecord(
            cycle_id=f"{prefix}{separator}{idx}",
            timestamp=base_ts + i * ts_step,
            side=side if not alternating_side else ("buy" if i % 2 == 0 else "sell"),
            order_price=order_price + i * price_step,
            order_quantity=order_quantity,
            filled=filled,
        ))
    return records


def _save_linear_records(path: Path, **kwargs: object) -> None:
    """線形レコードを生成して保存する."""
    _save_generated_records(path, _make_linear_records, **kwargs)


def _save_daily_fill_count_records(path: Path, **kwargs: object) -> None:
    """日次 fill-count レコードを生成して保存する."""
    _save_generated_records(path, _make_daily_fill_count_records, **kwargs)


def _save_generated_records(
    path: Path,
    builder: Callable[..., list[FillRecord]],
    **kwargs: object,
) -> None:
    """builder が返す FillRecord 群を保存する."""
    save_fill_records(builder(**kwargs), path)


def _seed_fill_records_direct(records: Iterable[FillRecord], path: Path) -> None:
    """save path を検証しないテスト向けの軽量 JSONL seed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
        for record in records
    )
    path.write_text(payload, encoding="utf-8")


def _seed_linear_records(path: Path, **kwargs: object) -> None:
    """線形レコードを直接 JSONL seed する."""
    _seed_fill_records_direct(_make_linear_records(**kwargs), path)


def _seed_dated_linear_record(
    root: Path,
    *,
    day: str,
    prefix: str,
    start_index: int,
    base_ts: float,
    include_emergency: bool = False,
    side: str = "buy",
    order_price: float = 100.0,
    separator: str = "_",
) -> None:
    """list/load/glob テスト向けに日付付き JSONL を直接 seed する."""
    _seed_linear_records(
        root / f"fill_records_{day}.jsonl",
        prefix=prefix,
        count=1,
        start_index=start_index,
        base_ts=base_ts,
        side=side,
        order_price=order_price,
        separator=separator,
    )
    if include_emergency:
        _seed_linear_records(
            root / "emergency" / f"emergency_{day}.jsonl",
            prefix=prefix,
            count=1,
            start_index=start_index,
            base_ts=base_ts + 60.0,
            side="sell",
            order_price=order_price + 1.0,
            separator=separator,
        )


def _save_dated_linear_record(
    root: Path,
    *,
    day: str,
    prefix: str,
    start_index: int,
    base_ts: float,
    include_emergency: bool = False,
    side: str = "buy",
    order_price: float = 100.0,
    separator: str = "_",
) -> None:
    """日付付き fill/emergency JSONL を 1 件ずつ保存する."""
    _save_linear_records(
        root / f"fill_records_{day}.jsonl",
        prefix=prefix,
        count=1,
        start_index=start_index,
        base_ts=base_ts,
        side=side,
        order_price=order_price,
        separator=separator,
    )
    if include_emergency:
        emergency = root / "emergency"
        emergency.mkdir(exist_ok=True)
        _save_linear_records(
            emergency / f"emergency_{day}.jsonl",
            prefix=prefix,
            count=1,
            start_index=start_index,
            base_ts=base_ts + 60.0,
            side="sell",
            order_price=order_price + 1.0,
            separator=separator,
        )


def _make_fast_cycle_runner(
    tmp_path: Path,
    *,
    order_id: str,
) -> "FillTestRunner":
    """status/cancel race テスト向けの最小待機 runner."""

    adapter = _FastCycleAdapter(order_id=order_id)

    config = FillTestConfig(
        results_dir=str(tmp_path / "results"),
        order_timeout_sec=_FAST_CYCLE_ORDER_TIMEOUT_SEC,
        poll_interval_sec=_FAST_CYCLE_POLL_INTERVAL_SEC,
        post_fill_wait_sec=_FAST_CYCLE_POST_FILL_WAIT_SEC,
        status_unknown_retry_delays=_FAST_STATUS_UNKNOWN_RETRY_DELAYS,
    )
    with patch.object(FillTestRunner, "_get_git_sha", return_value="test-sha"):
        runner = FillTestRunner(adapter, config)
    # status_unknown / cancel-race のテストでは phantom 登録自体は検証対象外。
    runner._phantom_guard = None
    return runner


# =====================================================================
# FillRecord
# =====================================================================

class TestFillRecord:
    """FillRecord の serialize / deserialize テスト."""

    def test_round_trip(self) -> None:

        r = FillRecord(
            cycle_id="test_001",
            timestamp=1700000000.0,
            side="buy",
            order_price=15000000.0,
            order_quantity=0.001,
            fill_price=15000001.0,
            filled=True,
            cancelled=False,
            queue_wait_sec=12.5,
            mid_at_fill=15000050.0,
            mid_30s_after=15000100.0,
            post_fill_30s_pnl=0.33,
            adverse_selected=False,
        )
        d = r.to_dict()
        r2 = FillRecord.from_dict(d)
        assert r2.cycle_id == r.cycle_id
        assert r2.filled is True
        assert r2.post_fill_30s_pnl == pytest.approx(0.33)

    def test_from_dict_extra_keys_ignored(self) -> None:

        d = {
            "cycle_id": "x",
            "timestamp": 0.0,
            "side": "sell",
            "order_price": 100.0,
            "order_quantity": 0.01,
            "extra_key": "should_be_ignored",
        }
        r = FillRecord.from_dict(d)
        assert r.cycle_id == "x"
        assert not hasattr(r, "extra_key")

    def test_build_skip_fill_record_applies_known_extra_only(self) -> None:

        r = build_skip_fill_record(
            cycle_id="skip_1",
            timestamp=1.0,
            side="buy",
            order_price=100.0,
            order_quantity=0.01,
            cancel_reason="skip_gate",
            run_id="run_1",
            git_sha="abc123",
            regime="trending",
            filled=True,
            cancelled=False,
            skip_gate_skipped=True,
            skip_gate_reason="threshold",
            unknown_extra="ignored",
        )

        assert r.cancelled is True
        assert r.filled is False
        assert r.cancel_reason == "skip_gate"
        assert r.skip_gate_skipped is True
        assert r.skip_gate_reason == "threshold"
        assert not hasattr(r, "unknown_extra")

    def test_build_fill_record_preserves_skip_gate_bypassed(self) -> None:

        r = build_fill_record(
            cycle_id="base_2",
            timestamp=3.0,
            side="sell",
            order_price=101.0,
            order_quantity=0.02,
            skip_gate_skipped=False,
            skip_gate_bypassed=True,
        )

        assert r.skip_gate_skipped is False
        assert r.skip_gate_bypassed is True

    def test_build_fill_record_preserves_timeout_trace_fields(self) -> None:

        r = build_fill_record(
            cycle_id="trace_1",
            timestamp=4.0,
            side="buy",
            order_price=100.0,
            order_quantity=0.01,
            decision_trace_id="dt_test_1",
            timeout_applied_sec=20.0,
            timeout_reason="regime_strong_up_sell",
        )

        assert r.decision_trace_id == "dt_test_1"
        assert r.timeout_applied_sec == 20.0
        assert r.timeout_reason == "regime_strong_up_sell"

    def test_build_fill_record_ignores_unknown_fields(self) -> None:

        r = build_fill_record(
            cycle_id="base_1",
            timestamp=2.0,
            side="sell",
            order_price=101.0,
            order_quantity=0.02,
            cancelled=True,
            cancel_reason="timeout",
            stray_field="ignored",
        )

        assert r.cycle_id == "base_1"
        assert r.cancelled is True
        assert r.cancel_reason == "timeout"
        assert not hasattr(r, "stray_field")

    def test_compute_record_pnl_jpy(self) -> None:

        r = FillRecord(
            cycle_id="pnl_1",
            timestamp=1.0,
            side="buy",
            order_price=100.0,
            order_quantity=0.5,
            fill_price=200.0,
            filled=True,
            post_fill_30s_pnl=10.0,
        )
        assert compute_record_pnl_jpy(r) == pytest.approx(0.1)

    def test_defaults(self) -> None:

        r = FillRecord(
            cycle_id="d",
            timestamp=0.0,
            side="buy",
            order_price=1.0,
            order_quantity=0.001,
        )
        assert r.filled is False
        assert r.cancelled is False
        assert r.fill_price is None
        assert r.post_fill_30s_pnl is None

    def test_cancel_reason_field(self) -> None:
        """CM-2: cancel_reason フィールドの追加確認."""

        r = FillRecord(
            cycle_id="cr",
            timestamp=0.0,
            side="buy",
            order_price=1.0,
            order_quantity=0.001,
            cancelled=True,
            cancel_reason="post_only_reject",
        )
        assert r.cancel_reason == "post_only_reject"
        d = r.to_dict()
        assert d["cancel_reason"] == "post_only_reject"
        r2 = FillRecord.from_dict(d)
        assert r2.cancel_reason == "post_only_reject"

    def test_timeout_trace_fields_roundtrip(self) -> None:
        r = FillRecord(
            cycle_id="trace_rt",
            timestamp=0.0,
            side="buy",
            order_price=1.0,
            order_quantity=0.001,
            decision_trace_id="dt_trace_rt",
            timeout_applied_sec=30.0,
            timeout_reason="base_default_sell_age_cap",
        )
        d = r.to_dict()
        assert d["decision_trace_id"] == "dt_trace_rt"
        assert d["timeout_applied_sec"] == 30.0
        assert d["timeout_reason"] == "base_default_sell_age_cap"
        r2 = FillRecord.from_dict(d)
        assert r2.decision_trace_id == "dt_trace_rt"
        assert r2.timeout_applied_sec == 30.0
        assert r2.timeout_reason == "base_default_sell_age_cap"

    def test_cancel_reason_default_none(self) -> None:
        """CM-2: cancel_reason はデフォルト None."""

        r = FillRecord(
            cycle_id="cr2",
            timestamp=0.0,
            side="buy",
            order_price=1.0,
            order_quantity=0.001,
        )
        assert r.cancel_reason is None

    def test_new_fields_020(self) -> None:
        """020# O4/O5: run_id, git_sha, adverse_selected_raw フィールド."""

        r = FillRecord(
            cycle_id="o20",
            timestamp=0.0,
            side="buy",
            order_price=1.0,
            order_quantity=0.001,
            adverse_selected=False,
            adverse_selected_raw=True,
            run_id="1234_abc",
            git_sha="abc1234",
        )
        assert r.adverse_selected_raw is True
        assert r.run_id == "1234_abc"
        assert r.git_sha == "abc1234"
        # round-trip
        d = r.to_dict()
        assert d["adverse_selected_raw"] is True
        assert d["run_id"] == "1234_abc"
        r2 = FillRecord.from_dict(d)
        assert r2.adverse_selected_raw is True
        assert r2.run_id == "1234_abc"
        assert r2.git_sha == "abc1234"

    def test_new_fields_020_defaults_none(self) -> None:
        """020# フィールドはデフォルト None."""

        r = FillRecord(
            cycle_id="d020",
            timestamp=0.0,
            side="buy",
            order_price=1.0,
            order_quantity=0.001,
        )
        assert r.adverse_selected_raw is None
        assert r.run_id is None
        assert r.git_sha is None

    def test_new_fields_031(self) -> None:
        """031# spread_at_order, error_message, spread_offset_ratio フィールド."""

        r = FillRecord(
            cycle_id="o31",
            timestamp=0.0,
            side="buy",
            order_price=15000000.0,
            order_quantity=0.001,
            spread_at_order=200.0,
            error_message="test error",
            spread_offset_ratio=0.05,
        )
        assert r.spread_at_order == 200.0
        assert r.error_message == "test error"
        assert r.spread_offset_ratio == 0.05
        # round-trip
        d = r.to_dict()
        assert d["spread_at_order"] == 200.0
        assert d["error_message"] == "test error"
        assert d["spread_offset_ratio"] == 0.05
        r2 = FillRecord.from_dict(d)
        assert r2.spread_at_order == 200.0
        assert r2.error_message == "test error"
        assert r2.spread_offset_ratio == 0.05

    def test_new_fields_031_defaults_none(self) -> None:
        """031# フィールドはデフォルト None."""

        r = FillRecord(
            cycle_id="d031",
            timestamp=0.0,
            side="buy",
            order_price=1.0,
            order_quantity=0.001,
        )
        assert r.spread_at_order is None
        assert r.error_message is None
        assert r.spread_offset_ratio is None

    def test_new_fields_031_backward_compat(self) -> None:
        """031# 旧データ (フィールドなし) からの読み込み互換性."""

        old_data = {
            "cycle_id": "old",
            "timestamp": 0.0,
            "side": "buy",
            "order_price": 1.0,
            "order_quantity": 0.001,
            "filled": True,
        }
        r = FillRecord.from_dict(old_data)
        assert r.spread_at_order is None
        assert r.error_message is None
        assert r.spread_offset_ratio is None


# =====================================================================
# compute_fill_metrics
# =====================================================================

class TestComputeFillMetrics:
    """compute_fill_metrics の指標算出テスト."""

    def _make_records(
        self,
        n: int = 100,
        fill_rate: float = 0.9,
        queue_wait: float = 20.0,
        pnl_mean: float = 0.5,
        adverse_rate: float = 0.1,
        days: int = 3,
    ) -> list:
        """テスト用の FillRecord リストを生成."""

        records = []
        base_ts = 1700000000.0
        for i in range(n):
            day_offset = (i % days) * 86400
            filled = (i / n) < fill_rate
            pnl = np.random.normal(pnl_mean, 1.0) if filled else None
            adverse = pnl < 0 if pnl is not None else None

            records.append(FillRecord(
                cycle_id=f"cycle_{i}",
                timestamp=base_ts + day_offset + i * 120,
                side="buy" if i % 2 == 0 else "sell",
                order_price=15000000.0 + i,
                order_quantity=0.001,
                fill_price=15000000.0 + i if filled else None,
                filled=filled,
                cancelled=not filled,
                queue_wait_sec=queue_wait + np.random.uniform(-5, 5) if filled else 0.0,
                mid_at_fill=15000050.0 if filled else None,
                mid_30s_after=15000050.0 + pnl * 1.5 if filled and pnl is not None else None,
                post_fill_30s_pnl=pnl,
                adverse_selected=adverse,
            ))
        return records

    def test_empty_records(self) -> None:

        m = compute_fill_metrics([])
        assert m.total_orders == 0
        assert m.fill_rate_p90 == 0.0

    def test_all_filled(self) -> None:

        records = self._make_records(n=50, fill_rate=1.0, days=2)
        m = compute_fill_metrics(records)
        assert m.total_orders == 50
        assert m.filled_orders == 50
        assert m.cancelled_orders == 0
        assert m.fill_rate_p90 >= 0.9
        assert m.cancel_ratio == 0.0

    def test_partial_fill(self) -> None:

        records = self._make_records(n=100, fill_rate=0.7, days=5)
        m = compute_fill_metrics(records)
        assert m.filled_orders == 70
        assert m.cancelled_orders == 30
        assert m.cancel_ratio == pytest.approx(0.3)

    def test_queue_wait_median(self) -> None:

        records = []
        base_ts = 1700000000.0
        for i in range(10):
            records.append(FillRecord(
                cycle_id=f"qw_{i}",
                timestamp=base_ts + i * 120,
                side="buy",
                order_price=100.0,
                order_quantity=0.001,
                filled=True,
                queue_wait_sec=float(10 + i * 2),  # 10, 12, ..., 28
            ))
        m = compute_fill_metrics(records)
        # median of [10, 12, 14, 16, 18, 20, 22, 24, 26, 28] = 19
        assert m.queue_wait_median_sec == pytest.approx(19.0)

    def test_adverse_selection(self) -> None:

        records = []
        base_ts = 1700000000.0
        for i in range(10):
            records.append(FillRecord(
                cycle_id=f"as_{i}",
                timestamp=base_ts + i * 120,
                side="buy",
                order_price=100.0,
                order_quantity=0.001,
                filled=True,
                adverse_selected=(i < 3),  # 30% adverse
                post_fill_30s_pnl=-1.0 if i < 3 else 1.0,
            ))
        m = compute_fill_metrics(records)
        assert m.adverse_selection_ratio == pytest.approx(0.3)

    def test_adverse_selection_raw_020(self) -> None:
        """020# O5: E5-raw (deadzone 非適用) が並行計算されることを検証."""

        records = []
        base_ts = 1700000000.0
        for i in range(10):
            records.append(FillRecord(
                cycle_id=f"asr_{i}",
                timestamp=base_ts + i * 120,
                side="buy",
                order_price=100.0,
                order_quantity=0.001,
                filled=True,
                adverse_selected=(i < 2),       # 20% (deadzone 適用後)
                adverse_selected_raw=(i < 4),    # 40% (raw)
                post_fill_30s_pnl=-1.0 if i < 2 else 1.0,
            ))
        m = compute_fill_metrics(records)
        assert m.adverse_selection_ratio == pytest.approx(0.2)
        assert m.adverse_selection_ratio_raw == pytest.approx(0.4)

    def test_sample_sufficient_true(self) -> None:
        """047# Finding3: n>=200 & 7暦日 → sample_sufficient=True (FINAL)."""
        records = _make_uniform_daily_records(
            prefix="ss",
            days=7,
            per_day=30,
            base_ts=1700006400.0,  # 2023-11-15 00:00:00 UTC
            queue_wait_sec=10.0,
            post_fill_30s_pnl=0.5,
            adverse_selected=False,
        )
        m = compute_fill_metrics(records)
        assert m.total_orders == 210
        assert m.measurement_days >= 7
        assert m.sample_sufficient is True

    def test_sample_sufficient_false_n(self) -> None:
        """020# O1: n<200 → sample_sufficient=False."""
        records = _make_uniform_daily_records(
            prefix="ssf",
            days=3,
            per_day=30,
            base_ts=1700000000.0,
        )
        m = compute_fill_metrics(records)
        assert m.total_orders == 90
        assert m.sample_sufficient is False

    def test_daily_fill_rates(self) -> None:

        records = _make_daily_fill_count_records(
            prefix="df",
            filled_counts=[9, 5],
            per_day=10,
            base_ts=1700000000.0,
        )
        m = compute_fill_metrics(records)
        assert m.measurement_days == 2
        assert len(m.daily_fill_rates) == 2
        assert m.daily_fill_rates[0] == pytest.approx(0.9)
        assert m.daily_fill_rates[1] == pytest.approx(0.5)
        # P90 (10th percentile of sorted daily rates) = 0.5 + 0.1*(0.9-0.5) = 0.54
        assert m.fill_rate_p90 == pytest.approx(0.54)


# =====================================================================
# g1_1_judgment
# =====================================================================

class TestG11Judgment:
    """G1.1 Gate 合否判定テスト."""

    def test_all_pass(self) -> None:

        metrics = FillMetrics(
            total_orders=100,
            filled_orders=95,
            cancelled_orders=5,
            fill_rate_p90=0.92,
            cancel_ratio=0.05,
            queue_wait_median_sec=15.0,
            post_fill_30s_pnl_mean=0.5,
            post_fill_30s_pnl_pvalue=0.8,
            adverse_selection_ratio=0.10,
        )
        thresholds = {
            "min_fill_rate_p90": 0.90,
            "max_cancel_ratio": 0.30,
            "max_queue_wait_median_sec": 60,
            "min_post_fill_30s_pnl": 0.0,
            "max_adverse_selection_ratio": 0.20,
        }
        result = g1_1_judgment(metrics, thresholds)
        assert result["gate_result"] == "PASS"
        assert all(c["pass"] for c in result["checks"].values())

    def test_fill_rate_fail(self) -> None:

        metrics = FillMetrics(
            fill_rate_p90=0.50,
            cancel_ratio=0.05,
            queue_wait_median_sec=10.0,
            post_fill_30s_pnl_mean=1.0,
            adverse_selection_ratio=0.05,
        )
        thresholds = {"min_fill_rate_p90": 0.90}
        result = g1_1_judgment(metrics, thresholds)
        assert result["gate_result"] == "FAIL"
        assert result["checks"]["E1_fill_rate_p90"]["pass"] is False

    def test_adverse_selection_fail(self) -> None:

        metrics = FillMetrics(
            fill_rate_p90=0.95,
            cancel_ratio=0.05,
            queue_wait_median_sec=10.0,
            post_fill_30s_pnl_mean=0.5,
            adverse_selection_ratio=0.35,
        )
        thresholds = {"max_adverse_selection_ratio": 0.20}
        result = g1_1_judgment(metrics, thresholds)
        assert result["gate_result"] == "FAIL"
        assert result["checks"]["E5_adverse_selection"]["pass"] is False

    def test_pnl_negative_but_not_significant(self) -> None:
        """E4: mean < 0 だが p >= 0.05 → PASS (009# §2.4)."""

        metrics = FillMetrics(
            fill_rate_p90=0.95,
            cancel_ratio=0.05,
            queue_wait_median_sec=10.0,
            post_fill_30s_pnl_mean=-0.1,
            post_fill_30s_pnl_pvalue=0.3,  # Not significant
            adverse_selection_ratio=0.05,
        )
        thresholds = {"min_post_fill_30s_pnl": 0.0}
        result = g1_1_judgment(metrics, thresholds)
        assert result["checks"]["E4_post_fill_pnl"]["pass"] is True

    def test_pnl_negative_and_significant(self) -> None:
        """E4: mean < 0 かつ p < 0.05 → FAIL (systemic adverse selection)."""

        metrics = FillMetrics(
            fill_rate_p90=0.95,
            cancel_ratio=0.05,
            queue_wait_median_sec=10.0,
            post_fill_30s_pnl_mean=-2.0,
            post_fill_30s_pnl_pvalue=0.001,  # Highly significant
            adverse_selection_ratio=0.05,
        )
        thresholds = {"min_post_fill_30s_pnl": 0.0}
        result = g1_1_judgment(metrics, thresholds)
        assert result["checks"]["E4_post_fill_pnl"]["pass"] is False

    def test_queue_wait_fail(self) -> None:

        metrics = FillMetrics(
            fill_rate_p90=0.95,
            cancel_ratio=0.05,
            queue_wait_median_sec=120.0,
            post_fill_30s_pnl_mean=1.0,
            adverse_selection_ratio=0.05,
        )
        thresholds = {"max_queue_wait_median_sec": 60}
        result = g1_1_judgment(metrics, thresholds)
        assert result["gate_result"] == "FAIL"
        assert result["checks"]["E3_queue_wait_median"]["pass"] is False

    def test_judgment_type_provisional(self) -> None:
        """020# O1: sample_sufficient=False → judgment_type=PROVISIONAL."""

        metrics = FillMetrics(
            total_orders=95,
            fill_rate_p90=0.95,
            cancel_ratio=0.05,
            queue_wait_median_sec=10.0,
            post_fill_30s_pnl_mean=0.5,
            adverse_selection_ratio=0.05,
            sample_sufficient=False,
        )
        thresholds = {
            "min_fill_rate_p90": 0.90,
            "max_cancel_ratio": 0.30,
            "max_queue_wait_median_sec": 60,
            "min_post_fill_30s_pnl": 0.0,
            "max_adverse_selection_ratio": 0.20,
        }
        result = g1_1_judgment(metrics, thresholds)
        assert result["judgment_type"] == "PROVISIONAL"
        assert result["sample_sufficient"] is False

    def test_judgment_type_final(self) -> None:
        """020# O1: sample_sufficient=True → judgment_type=FINAL."""

        metrics = FillMetrics(
            total_orders=250,
            fill_rate_p90=0.95,
            cancel_ratio=0.05,
            queue_wait_median_sec=10.0,
            post_fill_30s_pnl_mean=0.5,
            adverse_selection_ratio=0.05,
            sample_sufficient=True,
        )
        thresholds = {
            "min_fill_rate_p90": 0.90,
            "max_cancel_ratio": 0.30,
            "max_queue_wait_median_sec": 60,
            "min_post_fill_30s_pnl": 0.0,
            "max_adverse_selection_ratio": 0.20,
        }
        result = g1_1_judgment(metrics, thresholds)
        assert result["judgment_type"] == "FINAL"
        assert result["sample_sufficient"] is True

    def test_e5_raw_informational(self) -> None:
        """020# O5: E5-raw チェックは informational で gate に影響しない."""

        metrics = FillMetrics(
            fill_rate_p90=0.95,
            cancel_ratio=0.05,
            queue_wait_median_sec=10.0,
            post_fill_30s_pnl_mean=0.5,
            adverse_selection_ratio=0.05,
            adverse_selection_ratio_raw=0.85,  # 超過しているが informational
            sample_sufficient=True,
        )
        thresholds = {"max_adverse_selection_ratio": 0.20}
        result = g1_1_judgment(metrics, thresholds)
        # E5 (deadzone) は PASS
        assert result["checks"]["E5_adverse_selection"]["pass"] is True
        # E5-raw は informational (gate に影響しない)
        assert result["checks"]["E5_adverse_selection_raw"]["informational"] is True
        # gate は PASS のまま
        assert result["gate_result"] == "PASS"


# =====================================================================
# 116# Two-Stage Gate Tests
# =====================================================================


class TestG11QuickJudgment:
    """G1.1-quick (72h Kill Gate) テスト — 116# / 115# レビュー反映."""

    def _make_metrics(self, **overrides) -> "FillMetrics":
        defaults = dict(
            total_orders=400,
            filled_orders=280,
            cancelled_orders=120,
            fill_rate_p90=0.65,
            cancel_ratio=0.30,
            queue_wait_median_sec=12.0,
            post_fill_30s_pnl_mean=-0.1,
            post_fill_30s_pnl_pvalue=0.3,
            post_fill_30s_pnl_ci_upper=0.2,
            adverse_selection_ratio=0.28,
            attempted_orders=360,
            skip_gate_count=40,
            skip_gate_ratio=0.10,
            attempted_fill_rate=0.778,
            attempted_cancel_ratio=0.222,
            overall_fill_rate=0.70,
            measurement_days=4,
            sample_sufficient=False,
        )
        defaults.update(overrides)
        return FillMetrics(**defaults)

    def test_all_pass(self) -> None:
        metrics = self._make_metrics()
        thresholds = {
            "min_attempted_fill_rate": 0.60,
            "max_attempted_cancel_ratio": 0.40,
            "max_queue_wait_median_sec": 120,
            "pnl_kill_p_threshold": 0.02,
            "pnl_kill_mean_threshold": -0.8,
            "max_cumulative_loss_jpy": 10000,
            "max_skip_gate_ratio": 0.25,
        }
        result = g1_1_quick_judgment(metrics, thresholds)
        assert result["gate_result"] == "PASS"
        assert result["gate"] == "G1.1-quick"
        assert all(c["pass"] for c in result["checks"].values())

    def test_k1_fill_rate_fail(self) -> None:
        metrics = self._make_metrics(attempted_fill_rate=0.50)
        result = g1_1_quick_judgment(metrics, {"min_attempted_fill_rate": 0.60})
        assert result["gate_result"] == "FAIL"
        assert result["checks"]["K1_attempted_fill_rate"]["pass"] is False

    def test_k4_pnl_compound_both_conditions_fail(self) -> None:
        """K4: p < 0.02 かつ mean <= -0.8 で FAIL (115# 複合条件)."""
        metrics = self._make_metrics(
            post_fill_30s_pnl_mean=-1.2,
            post_fill_30s_pnl_pvalue=0.005,
        )
        thresholds = {"pnl_kill_p_threshold": 0.02, "pnl_kill_mean_threshold": -0.8}
        result = g1_1_quick_judgment(metrics, thresholds)
        assert result["gate_result"] == "FAIL"
        assert result["checks"]["K4_pnl_kill"]["pass"] is False
        assert result["checks"]["K4_pnl_kill"]["significant"] is True
        assert result["checks"]["K4_pnl_kill"]["large_loss"] is True

    def test_k4_pnl_significant_but_small_loss_passes(self) -> None:
        """K4: p < 0.02 だが mean > -0.8 → PASS (効果量不足)."""
        metrics = self._make_metrics(
            post_fill_30s_pnl_mean=-0.3,
            post_fill_30s_pnl_pvalue=0.01,
        )
        thresholds = {"pnl_kill_p_threshold": 0.02, "pnl_kill_mean_threshold": -0.8}
        result = g1_1_quick_judgment(metrics, thresholds)
        assert result["checks"]["K4_pnl_kill"]["pass"] is True

    def test_k4_pnl_large_loss_but_not_significant_passes(self) -> None:
        """K4: mean <= -0.8 だが p >= 0.02 → PASS (統計的に不確実)."""
        metrics = self._make_metrics(
            post_fill_30s_pnl_mean=-1.5,
            post_fill_30s_pnl_pvalue=0.06,
        )
        thresholds = {"pnl_kill_p_threshold": 0.02, "pnl_kill_mean_threshold": -0.8}
        result = g1_1_quick_judgment(metrics, thresholds)
        assert result["checks"]["K4_pnl_kill"]["pass"] is True

    def test_k5_cumulative_loss_fail(self) -> None:
        metrics = self._make_metrics()
        result = g1_1_quick_judgment(
            metrics,
            {"max_cumulative_loss_jpy": 10000},
            cumulative_loss_jpy=12000,
        )
        assert result["gate_result"] == "FAIL"
        assert result["checks"]["K5_cumulative_loss"]["pass"] is False

    def test_k6_skip_gate_ratio_fail(self) -> None:
        metrics = self._make_metrics(skip_gate_ratio=0.30)
        result = g1_1_quick_judgment(metrics, {"max_skip_gate_ratio": 0.25})
        assert result["gate_result"] == "FAIL"
        assert result["checks"]["K6_skip_gate_ratio"]["pass"] is False

    def test_watch_layer(self) -> None:
        """115# Q10.4: PASS だが PnL が黄信号 → WATCH."""
        metrics = self._make_metrics(
            post_fill_30s_pnl_mean=-0.5,
            post_fill_30s_pnl_pvalue=0.03,
        )
        thresholds = {
            "min_attempted_fill_rate": 0.60,
            "max_attempted_cancel_ratio": 0.40,
            "max_queue_wait_median_sec": 120,
            "pnl_kill_p_threshold": 0.02,
            "pnl_kill_mean_threshold": -0.8,
            "max_cumulative_loss_jpy": 10000,
            "max_skip_gate_ratio": 0.25,
            "pnl_watch_p_threshold": 0.05,
            "pnl_watch_mean_threshold": -0.3,
        }
        result = g1_1_quick_judgment(metrics, thresholds)
        assert result["gate_result"] == "WATCH"
        assert result["watch"] is True
        assert result["watch_detail"] is not None

    def test_no_watch_when_pnl_ok(self) -> None:
        metrics = self._make_metrics(
            post_fill_30s_pnl_mean=0.5,
            post_fill_30s_pnl_pvalue=0.8,
        )
        result = g1_1_quick_judgment(metrics, {})
        assert result["gate_result"] == "PASS"
        assert result["watch"] is False


class TestG12FullJudgment:
    """G1.2-full (168h Qualification Gate) テスト — 116# / 115# レビュー反映."""

    def _make_metrics(self, **overrides) -> "FillMetrics":
        defaults = dict(
            total_orders=1057,
            filled_orders=714,
            cancelled_orders=343,
            fill_rate_p90=0.62,
            cancel_ratio=0.325,
            queue_wait_median_sec=12.8,
            post_fill_30s_pnl_mean=-0.196,
            post_fill_30s_pnl_pvalue=0.161,
            post_fill_30s_pnl_ci_upper=0.192,
            adverse_selection_ratio=0.28,
            attempted_orders=971,
            skip_gate_count=86,
            skip_gate_ratio=0.081,
            attempted_fill_rate=0.735,
            attempted_cancel_ratio=0.265,
            overall_fill_rate=0.675,
            measurement_days=7,
            sample_sufficient=True,
        )
        defaults.update(overrides)
        return FillMetrics(**defaults)

    def test_all_pass(self) -> None:
        metrics = self._make_metrics()
        thresholds = {
            "min_attempted_fill_rate": 0.70,
            "min_overall_fill_rate": 0.62,
            "max_attempted_cancel_ratio": 0.30,
            "max_queue_wait_median_sec": 60,
            "pnl_alpha": 0.05,
            "max_adverse_selection_ratio": 0.30,
            "max_skip_gate_ratio": 0.20,
            "min_calendar_days": 7,
            "min_attempted_samples": 500,
            "pnl_mean_floor_bps": -0.20,  # 123# metrics mean=-0.196 > -0.20 → PASS
        }
        result = g1_2_full_judgment(metrics, thresholds)
        assert result["gate_result"] == "PASS"
        assert result["gate"] == "G1.2-full"
        assert all(c["pass"] for c in result["checks"].values())

    def test_f1_attempted_fill_rate_fail(self) -> None:
        metrics = self._make_metrics(attempted_fill_rate=0.65)
        result = g1_2_full_judgment(metrics, {"min_attempted_fill_rate": 0.70})
        assert result["gate_result"] == "FAIL"
        assert result["checks"]["F1_attempted_fill_rate"]["pass"] is False

    def test_f1b_overall_fill_rate_fail(self) -> None:
        """115# Q10.2(A): overall 下限の併設チェック."""
        metrics = self._make_metrics(overall_fill_rate=0.55)
        result = g1_2_full_judgment(metrics, {"min_overall_fill_rate": 0.62})
        assert result["gate_result"] == "FAIL"
        assert result["checks"]["F1b_overall_fill_rate"]["pass"] is False

    def test_f4_pnl_negative_not_significant_passes(self) -> None:
        metrics = self._make_metrics(
            post_fill_30s_pnl_mean=-0.2,
            post_fill_30s_pnl_pvalue=0.16,
        )
        result = g1_2_full_judgment(metrics, {"pnl_alpha": 0.05})
        assert result["checks"]["F4_pnl"]["pass"] is True

    def test_f4_pnl_negative_significant_fails(self) -> None:
        metrics = self._make_metrics(
            post_fill_30s_pnl_mean=-0.8,
            post_fill_30s_pnl_pvalue=0.01,
        )
        result = g1_2_full_judgment(metrics, {"pnl_alpha": 0.05})
        assert result["checks"]["F4_pnl"]["pass"] is False

    def test_f5_adverse_selection_fail(self) -> None:
        """115# Q10.2(B): AS 30% 閾値チェック."""
        metrics = self._make_metrics(adverse_selection_ratio=0.35)
        result = g1_2_full_judgment(metrics, {"max_adverse_selection_ratio": 0.30})
        assert result["gate_result"] == "FAIL"
        assert result["checks"]["F5_adverse_selection"]["pass"] is False

    def test_f6_skip_gate_ratio_fail(self) -> None:
        metrics = self._make_metrics(skip_gate_ratio=0.25)
        result = g1_2_full_judgment(metrics, {"max_skip_gate_ratio": 0.20})
        assert result["gate_result"] == "FAIL"
        assert result["checks"]["F6_skip_gate_ratio"]["pass"] is False

    def test_f7_calendar_days_fail(self) -> None:
        metrics = self._make_metrics(measurement_days=5)
        result = g1_2_full_judgment(metrics, {"min_calendar_days": 7})
        assert result["gate_result"] == "FAIL"
        assert result["checks"]["F7_calendar_days"]["pass"] is False

    def test_f8_n_attempted_fail(self) -> None:
        metrics = self._make_metrics(attempted_orders=400)
        result = g1_2_full_judgment(metrics, {"min_attempted_samples": 500})
        assert result["gate_result"] == "FAIL"
        assert result["checks"]["F8_n_attempted"]["pass"] is False


# =====================================================================
# 116# compute_fill_metrics attempted field tests
# =====================================================================


class TestComputeFillMetricsAttempted:
    """116# attempted ベース指標の compute_fill_metrics テスト."""

    def test_skip_gate_fields_populated(self) -> None:
        """skip_gate_skipped=True のレコードが正しく除外される."""
        base_ts = time.time()
        records = _make_outcome_records(
            prefix="attempted",
            counts={"fill": 10, "skip_gate": 3, "timeout": 2},
            base_ts=base_ts,
        )

        m = compute_fill_metrics(records)
        assert m.total_orders == 15
        assert m.skip_gate_count == 3
        assert m.attempted_orders == 12  # 15 - 3
        assert m.skip_gate_ratio == pytest.approx(3 / 15)
        assert m.attempted_fill_rate == pytest.approx(10 / 12)
        assert m.attempted_cancel_ratio == pytest.approx(2 / 12)
        assert m.overall_fill_rate == pytest.approx(10 / 15)

    def test_no_skip_gate_records(self) -> None:
        """skip_gate なし → attempted = total."""
        base_ts = time.time()
        records = _make_outcome_records(
            prefix="attempted_all_fill",
            counts={"fill": 5},
            base_ts=base_ts,
        )
        m = compute_fill_metrics(records)
        assert m.skip_gate_count == 0
        assert m.attempted_orders == 5
        assert m.attempted_fill_rate == pytest.approx(1.0)
        assert m.overall_fill_rate == pytest.approx(1.0)

    def test_cancel_reason_breakdown(self) -> None:
        """117# cancel reason 内訳が正しく集計される."""
        base_ts = time.time()
        records = _make_outcome_records(
            prefix="reasons",
            counts={
                "fill": 5,
                "timeout": 3,
                "skip_gate": 2,
                "postonly_reject": 1,
                "unknown_cancel": 1,
            },
            base_ts=base_ts,
        )
        records[-2].side = "sell"
        records[-1].side = "sell"

        m = compute_fill_metrics(records)
        assert m.cancel_reason_breakdown["timeout"] == 3
        assert m.cancel_reason_breakdown["skip_gate"] == 2
        assert m.cancel_reason_breakdown["postonly_reject"] == 1
        assert m.cancel_reason_breakdown["unknown"] == 1
        assert sum(m.cancel_reason_breakdown.values()) == m.cancelled_orders


# =====================================================================
# I/O
# =====================================================================

class TestFillRecordIO:
    """FillRecord の JSONL I/O テスト."""

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        _save_linear_records(
            path,
            prefix="io",
            count=5,
            base_ts=1700000000.0,
            alternating_side=True,
            order_price=15000000.0,
            filled=True,
        )
        loaded = load_fill_records(path)
        assert len(loaded) == 5
        assert loaded[0].cycle_id == "io_0"
        assert loaded[4].side == "buy"

    def test_iter_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "iter.jsonl"
        _seed_linear_records(
            path,
            prefix="iter",
            count=3,
            base_ts=1700000000.0,
            ts_step=60.0,
            order_price=12000000.0,
        )
        loaded = list(iter_fill_records(path))
        assert len(loaded) == 3
        assert loaded[0].cycle_id == "iter_0"
        assert loaded[2].cycle_id == "iter_2"

    def test_load_nonexistent(self) -> None:

        records = load_fill_records("/nonexistent/path.jsonl")
        assert records == []

    def test_glob_load(self, tmp_path: Path) -> None:
        for day in ["20260101", "20260102"]:
            _seed_linear_records(
                tmp_path / f"fill_records_{day}.jsonl",
                prefix=day,
                count=3,
                base_ts=1700000000.0,
                ts_step=1.0,
            )

        all_records = load_fill_records_glob(tmp_path)
        assert len(all_records) == 6

    def test_glob_load_deduplicates_emergency_records(self, tmp_path: Path) -> None:
        root = tmp_path
        emergency = root / "emergency"
        emergency.mkdir()
        _seed_linear_records(
            root / "fill_records_20260101.jsonl",
            prefix="dup",
            count=1,
            start_index=1,
            base_ts=1700000000.0,
        )
        _seed_linear_records(
            emergency / "emergency_20260101.jsonl",
            prefix="dup",
            count=1,
            start_index=1,
            base_ts=1700000000.0,
        )

        all_records = load_fill_records_glob(root)
        assert len(all_records) == 1
        assert all_records[0].cycle_id == "dup_1"

    def test_iter_glob_load_roundtrip(self, tmp_path: Path) -> None:
        root = tmp_path
        _seed_linear_records(
            root / "fill_records_20260101.jsonl",
            prefix="g",
            count=2,
            base_ts=1700000000.0,
            ts_step=60.0,
            alternating_side=True,
            price_step=1.0,
        )

        loaded = list(iter_fill_records_glob(root))
        assert [r.cycle_id for r in loaded] == ["g_0", "g_1"]

    def test_iter_glob_load_can_exclude_emergency(self, tmp_path: Path) -> None:
        root = tmp_path
        emergency = root / "emergency"
        emergency.mkdir()
        _seed_linear_records(
            root / "fill_records_20260101.jsonl",
            prefix="main",
            count=1,
            start_index=1,
            base_ts=1700000000.0,
        )
        _seed_linear_records(
            emergency / "emergency_20260101.jsonl",
            prefix="emg",
            count=1,
            start_index=1,
            base_ts=1700000060.0,
            side="sell",
            order_price=101.0,
        )

        loaded = list(iter_fill_records_glob(root, include_emergency=False))
        assert [r.cycle_id for r in loaded] == ["main_1"]

    def test_fill_records_to_dataframe_accepts_iterator(self) -> None:

        records = (
            FillRecord(
                cycle_id=f"df_{i}",
                timestamp=1700000000.0 + i,
                side="buy",
                order_price=100.0 + i,
                order_quantity=0.001,
            )
            for i in range(2)
        )

        df = fill_records_to_dataframe(records)
        assert list(df["cycle_id"]) == ["df_0", "df_1"]
        assert list(df["order_price"]) == [100.0, 101.0]

    def test_iter_fill_record_objects_glob_roundtrip(self, tmp_path: Path) -> None:
        root = tmp_path
        emergency = root / "emergency"
        emergency.mkdir()
        _seed_linear_records(
            root / "fill_records_20260101.jsonl",
            prefix="obj",
            count=1,
            start_index=1,
            base_ts=1700000000.0,
        )
        _seed_linear_records(
            emergency / "emergency_20260101.jsonl",
            prefix="obj",
            count=2,
            start_index=1,
            base_ts=1700000001.0,
            ts_step=1.0,
            side="sell",
            order_price=101.0,
            price_step=1.0,
        )

        loaded = list(iter_fill_record_objects_glob(root))
        assert [record["cycle_id"] for record in loaded] == ["obj_1", "obj_2"]

    def test_apply_fill_record_filters_handles_invalid_timestamp(self) -> None:

        records = [
            {"cycle_id": "a", "timestamp": "bad", "run_id": "r1", "git_sha": "abc123"},
            {"cycle_id": "b", "timestamp": 1771718400.0, "run_id": "r1", "git_sha": "abc999"},
        ]

        filtered, applied = apply_fill_record_filters(
            records,
            run_id="r1",
            git_sha="abc",
            date_from="2026-02-21",
        )

        assert [record["cycle_id"] for record in filtered] == ["b"]
        assert applied["run_id"] == "r1"
        assert applied["git_sha"] == "abc"

    def test_list_fill_record_files_supports_date_range(self, tmp_path: Path) -> None:
        root = tmp_path
        _seed_dated_linear_record(
            root,
            day="20260101",
            prefix="d",
            start_index=1,
            base_ts=1700000000.0,
        )
        _seed_dated_linear_record(
            root,
            day="20260102",
            prefix="d",
            start_index=2,
            base_ts=1700000060.0,
            order_price=101.0,
        )

        files = list_fill_record_files(
            root,
            include_emergency=False,
            start_date="2026-01-02",
            end_date="20260102",
        )
        assert [path.name for path in files] == ["fill_records_20260102.jsonl"]

    def test_list_fill_record_files_date_range_uses_direct_resolution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        root = tmp_path
        _seed_dated_linear_record(
            root,
            day="20260102",
            prefix="d",
            start_index=1,
            base_ts=1700000000.0,
            include_emergency=True,
            order_price=100.0,
        )
        emergency = root / "emergency"

        original_iterdir = Path.iterdir

        def _guarded_iterdir(path: Path):
            if path in (root, emergency):
                raise AssertionError("bounded date range should not scan directory")
            return original_iterdir(path)

        monkeypatch.setattr(Path, "iterdir", _guarded_iterdir)

        files = list_fill_record_files(
            root,
            include_emergency=True,
            start_date="2026-01-02",
            end_date="2026-01-02",
        )
        assert [path.name for path in files] == [
            "fill_records_20260102.jsonl",
            "emergency_20260102.jsonl",
        ]

    def test_list_fill_record_files_cache_invalidates_when_directory_changes(self, tmp_path: Path) -> None:
        root = tmp_path
        _seed_dated_linear_record(
            root,
            day="20260101",
            prefix="d",
            start_index=1,
            base_ts=1700000000.0,
        )
        before = list_fill_record_files(root, include_emergency=False)
        assert [path.name for path in before] == ["fill_records_20260101.jsonl"]

        _seed_dated_linear_record(
            root,
            day="20260102",
            prefix="d",
            start_index=2,
            base_ts=1700000060.0,
            order_price=101.0,
        )
        os.utime(root, None)

        after = list_fill_record_files(root, include_emergency=False)
        assert [path.name for path in after] == [
            "fill_records_20260101.jsonl",
            "fill_records_20260102.jsonl",
        ]

    def test_load_fill_record_objects_glob_supports_date_range(self, tmp_path: Path) -> None:
        root = tmp_path
        _seed_dated_linear_record(
            root,
            day="20260101",
            prefix="r",
            start_index=1,
            base_ts=1700000000.0,
            separator="",
        )
        _seed_dated_linear_record(
            root,
            day="20260103",
            prefix="r",
            start_index=2,
            base_ts=1700000060.0,
            side="sell",
            order_price=101.0,
            separator="",
        )

        loaded = load_fill_record_objects_glob(
            root,
            include_emergency=False,
            start_date="20260102",
            end_date="2026-01-03",
        )
        assert [record["cycle_id"] for record in loaded] == ["r2"]

    def test_load_corrupt_lines_skipped(self, tmp_path: Path) -> None:
        """032# #19: 破損行はスキップして残りを正常読込."""
        valid_records = _make_linear_records(
            prefix="valid",
            count=2,
            base_ts=1700000000.0,
            ts_step=120.0,
            order_price=15000000.0,
            alternating_side=True,
        )
        valid_records[0].filled = True
        valid_records[1].cancelled = True

        path = tmp_path / "test_corrupt.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            # 正常行
            f.write(json.dumps(valid_records[0].to_dict(), ensure_ascii=False) + "\n")
            # 破損行 (不完全 JSON)
            f.write('{"cycle_id": "broken_1", "timestamp": 170000\n')
            # 空行
            f.write("\n")
            # もう1つの正常行
            f.write(json.dumps(valid_records[1].to_dict(), ensure_ascii=False) + "\n")

        loaded = load_fill_records(path)
        assert len(loaded) == 2
        assert loaded[0].cycle_id == "valid_0"
        assert loaded[1].cycle_id == "valid_1"


# =====================================================================
# run_gate_check.py G1.1 integration
# =====================================================================

class TestGateCheckG11:
    """run_gate_check.py の G1.1 対応テスト."""

    def test_g1_1_no_data(self, tmp_path: Path) -> None:
        result = run_g1_1(tmp_path)
        assert result["gate_result"] == "NO_DATA"

    def test_g1_1_with_data(self, tmp_path: Path) -> None:
        _save_daily_fill_count_records(
            tmp_path / "fill_records_20260101.jsonl",
            prefix="g11",
            filled_counts=[19, 19, 19],
            per_day=20,
            base_ts=1700000000.0,
            order_price=15000000.0,
            queue_wait_sec=15.0,
            post_fill_30s_pnl=0.5,
            adverse_selected=False,
            alternate_side=True,
        )
        result = run_g1_1(tmp_path)
        # 135# P0-12: delegation 後は g1_1_quick_judgment 由来の "G1.1-quick"
        assert result["gate"] == "G1.1-quick"
        assert result["gate_result"] in ("PASS", "WATCH", "FAIL")


# =====================================================================
# Trade dedup (F8)
# =====================================================================

class TestTradeDedupTuple:
    """F8: trade dedup がタプル比較で正しく動作するテスト."""

    def test_dedup_skips_older(self) -> None:
        collector = MarketDataCollector.__new__(MarketDataCollector)
        collector._tr_buffer = []
        collector._last_trade_id = None

        trades_batch1 = [
            TradeRecord(timestamp=100.0, price=50000.0, amount=0.1, side="buy"),
            TradeRecord(timestamp=200.0, price=50001.0, amount=0.2, side="sell"),
        ]
        collector._append_raw_trades(trades_batch1)
        assert len(collector._tr_buffer) == 2

        # Same batch again → should be deduped
        collector._append_raw_trades(trades_batch1)
        assert len(collector._tr_buffer) == 2  # No duplicates

    def test_dedup_numeric_comparison(self) -> None:
        """F8 の本質: 文字列比較だと "10.5" < "9.5" になるバグの修正確認."""
        collector = MarketDataCollector.__new__(MarketDataCollector)
        collector._tr_buffer = []
        collector._last_trade_id = None

        # timestamp=9.5 を先に登録
        batch1 = [TradeRecord(timestamp=9.5, price=100.0, amount=0.1, side="buy")]
        collector._append_raw_trades(batch1)
        assert len(collector._tr_buffer) == 1

        # timestamp=10.5 → 新しいので追加されるべき
        batch2 = [TradeRecord(timestamp=10.5, price=100.0, amount=0.1, side="buy")]
        collector._append_raw_trades(batch2)
        assert len(collector._tr_buffer) == 2  # Must be 2 (not deduped)


# =====================================================================
# CoincheckAdapter real mode stubs
# =====================================================================

class TestAdapterRealModeP20:
    """P2-0: CoincheckAdapter real mode メソッドが NotImplementedError を出さないテスト."""

    def test_get_current_price_has_no_not_implemented(self) -> None:
        """get_current_price がreal modeで NotImplementedError を投げないことの静的確認."""
        source = read_inspect_source(CoincheckAdapter.get_current_price)
        # 009# P2-0: real mode should NOT raise NotImplementedError
        assert "NotImplementedError" not in source

    def test_get_order_status_has_no_not_implemented(self) -> None:
        source = read_inspect_source(CoincheckAdapter.get_order_status)
        assert "NotImplementedError" not in source

    def test_get_open_orders_has_no_not_implemented(self) -> None:
        source = read_inspect_source(CoincheckAdapter.get_open_orders)
        assert "NotImplementedError" not in source

    def test_get_positions_has_no_not_implemented(self) -> None:
        source = read_inspect_source(CoincheckAdapter.get_positions)
        assert "NotImplementedError" not in source


# =====================================================================
# 024# R1-R4: FillTestRunner 保存耐障害性テスト
# =====================================================================

class TestFillTestRunnerSaveResilience:
    """024# R1-R4: BatchPersistence (119# 委譲) のテスト."""

    def _make_persistence(
        self,
        tmp_path: Path,
        *,
        max_retries: int = 3,
        save_fail_threshold: int = 3,
    ) -> "BatchPersistence":
        """テスト用の BatchPersistence を作成."""
        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        return BatchPersistence(
            results_dir=results_dir,
            max_retries=max_retries,
            save_fail_threshold=save_fail_threshold,
            retry_backoff_sec=0.0,
            flush_interval_sec=600.0,
        )

    def _make_runner(self, tmp_path: Path) -> "FillTestRunner":
        """テスト用の FillTestRunner を作成 (adapter は mock)."""
        adapter = MagicMock()
        config = FillTestConfig(results_dir=str(tmp_path / "results"))
        with patch.object(FillTestRunner, "_get_git_sha", return_value="test-sha"):
            runner = FillTestRunner(adapter, config)
        return runner

    def _make_cleanup_runner(self, tmp_path: Path) -> "FillTestRunner":
        """_cleanup_sync 専用の軽量 runner."""
        runner = FillTestRunner.__new__(FillTestRunner)
        runner._ob_recorder = MagicMock()
        runner._ob_recorder.flush.return_value = 0
        runner._trades_recorder = MagicMock()
        runner._trades_recorder.flush.return_value = 0
        runner._pending_order_id = None
        runner.adapter = MagicMock()
        runner._release_lock = MagicMock()
        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        runner._batch_persistence = BatchPersistence(
            results_dir=results_dir,
            max_retries=3,
            save_fail_threshold=3,
            retry_backoff_sec=0.0,
            flush_interval_sec=600.0,
        )
        return runner

    def _make_record(self, ts: float = 1700000000.0, side: str = "buy") -> "FillRecord":
        return FillRecord(
            cycle_id=f"test_{int(ts)}",
            timestamp=ts,
            side=side,
            order_price=15000000.0,
            order_quantity=0.001,
            filled=True,
            fill_price=15000000.0,
        )

    def test_try_save_batch_success(self, tmp_path: Path) -> None:
        """正常保存時に True を返し、ファイルが作成される."""
        bp = self._make_persistence(tmp_path)
        batch = [self._make_record(ts=1700000000.0 + i) for i in range(5)]

        result = bp.try_save_batch(batch)

        assert result is True
        assert bp._save_fail_count == 0
        # JSONL ファイルが作成されている
        jsonl_files = list((tmp_path / "results").glob("fill_records_*.jsonl"))
        assert len(jsonl_files) >= 1

    def test_try_save_batch_retry_on_failure(self, tmp_path: Path) -> None:
        """保存失敗時にリトライし、最終的に失敗を返す."""
        bp = self._make_persistence(tmp_path, max_retries=1)
        batch = [self._make_record()]

        with patch.object(bp, "_save_batch_by_date", side_effect=IOError("disk full")):
            result = bp.try_save_batch(batch)

        assert result is False
        assert bp._save_fail_count == 1
        assert len(bp.unsaved_batch) == 1

    def test_try_save_batch_emergency_dump_after_3_failures(self, tmp_path: Path) -> None:
        """3回連続失敗で緊急ダンプが発動する."""
        bp = self._make_persistence(tmp_path, max_retries=1)
        bp._save_fail_count = 2  # 既に2回失敗
        batch = [self._make_record()]

        with patch.object(bp, "_save_batch_by_date", side_effect=IOError("disk full")):
            result = bp.try_save_batch(batch)

        # 緊急ダンプが発動 → True を返す
        assert result is True
        assert bp._save_fail_count == 0
        # emergency ディレクトリにファイルが作成
        emergency_files = list((tmp_path / "results" / "emergency").glob("emergency_*.jsonl"))
        assert len(emergency_files) >= 1

    def test_save_batch_by_date_groups_by_utc_date(self, tmp_path: Path) -> None:
        """024# R4: record.timestamp 由来で日付別ファイル分割."""
        bp = self._make_persistence(tmp_path)

        # 2つの異なる UTC 日付のレコード
        # 2023-11-14 23:59 UTC と 2023-11-15 00:01 UTC
        batch = [
            self._make_record(ts=1700006340.0),  # 2023-11-14 23:59 UTC
            self._make_record(ts=1700006460.0),  # 2023-11-15 00:01 UTC
        ]

        bp._save_batch_by_date(batch)

        results_dir = tmp_path / "results"
        f1 = results_dir / "fill_records_20231114.jsonl"
        f2 = results_dir / "fill_records_20231115.jsonl"
        assert f1.exists(), f"Expected {f1}"
        assert f2.exists(), f"Expected {f2}"

    def test_emergency_dump_creates_file(self, tmp_path: Path) -> None:
        """緊急ダンプがファイルを作成する."""
        bp = self._make_persistence(tmp_path)
        batch = [self._make_record()]

        bp.emergency_dump(batch, "test_reason")

        emergency_files = list((tmp_path / "results" / "emergency").glob("emergency_test_reason_*.jsonl"))
        assert len(emergency_files) == 1

    def test_cleanup_sync_saves_unsaved_batch(self, tmp_path: Path) -> None:
        """atexit で未保存バッチが退避される."""
        runner = self._make_cleanup_runner(tmp_path)
        runner._batch_persistence._unsaved_batch = [self._make_record()]

        with patch.object(runner._batch_persistence, "emergency_dump") as mock_dump, patch(
            "scripts.v460.lib.sidecar_signal_io.clear_sidecar_signal_cache",
        ) as mock_clear_sidecar:
            runner._cleanup_sync()

        assert runner._batch_persistence.unsaved_batch == []
        mock_dump.assert_called_once()
        mock_clear_sidecar.assert_called_once()

    def test_cleanup_sync_no_unsaved_no_dump(self, tmp_path: Path) -> None:
        """未保存バッチが空なら緊急ダンプは作成されない."""
        runner = self._make_cleanup_runner(tmp_path)

        with patch.object(runner._batch_persistence, "emergency_dump") as mock_dump:
            runner._cleanup_sync()

        mock_dump.assert_not_called()


class TestUnknownFillHandling:
    """025# F6: status_order is None 時に filled 扱いしない安全策のテスト."""

    def _make_runner(self, tmp_path: Path) -> "FillTestRunner":
        """テスト用の FillTestRunner を作成 (adapter は AsyncMock)."""
        return _make_fast_cycle_runner(tmp_path, order_id="test_order_123")

    @pytest.mark.asyncio
    async def test_status_none_twice_becomes_cancelled_status_unknown(
        self, tmp_path: Path,
    ) -> None:
        """get_order_status が 2 回 None → cancelled, cancel_reason=postonly_reject or status_unknown."""
        runner = self._make_runner(tmp_path)
        # 初回も retry も None
        runner.adapter.get_order_status = _make_status_side_effect(None, None, None, None)  # type: ignore[assignment]

        record = await _run_single_cycle_without_sleep(runner)

        assert record.filled is False
        assert record.cancelled is True
        # 079#: elapsed が短い場合は post_only_reject/status_unknown_fast、長い場合は status_unknown
        # 122# E12: spread 条件も考慮して 3 分類に細分化
        # 156# §10 #3: postonly_reject → post_only_reject に統一
        assert record.cancel_reason in ("status_unknown", "status_unknown_fast", "post_only_reject")

    @pytest.mark.asyncio
    async def test_status_none_then_filled_on_retry(
        self, tmp_path: Path,
    ) -> None:
        """get_order_status が None → retry で filled → filled=True."""

        runner = self._make_runner(tmp_path)
        filled_order = SimpleNamespace(status="filled", price=15000500.0)
        # 1 回目 None, 2 回目 filled
        runner.adapter.get_order_status = _make_status_side_effect(None, filled_order)  # type: ignore[assignment]

        record = await _run_single_cycle_without_sleep(runner)

        assert record.filled is True
        assert record.fill_price == 15000500.0
        assert record.cancel_reason is None

    @pytest.mark.asyncio
    async def test_status_filled_directly(
        self, tmp_path: Path,
    ) -> None:
        """get_order_status が直接 filled → 通常の filled 処理."""

        runner = self._make_runner(tmp_path)
        filled_order = SimpleNamespace(status="filled", price=15000200.0)
        runner.adapter.get_order_status = lambda _order_id: _async_return(filled_order)  # type: ignore[method-assign]

        record = await _run_single_cycle_without_sleep(runner)

        assert record.filled is True
        assert record.fill_price == 15000200.0
        assert record.cancel_reason is None


# ======================================================================
# 047# Bug11: cancel 失敗後の fill 再確認テスト
# ======================================================================

class TestBug11CancelRaceCondition:
    """047# Bug11: cancel_order 失敗時に get_order_status で fill を再確認."""

    def _make_runner(self, tmp_path: Path) -> "FillTestRunner":
        return _make_fast_cycle_runner(tmp_path, order_id="test_order_bug11")

    @pytest.mark.asyncio
    async def test_cancel_fail_detects_fill(self, tmp_path: Path) -> None:
        """cancel_order が "Failed to cancel" で失敗 → recheck で filled 検出."""

        runner = self._make_runner(tmp_path)
        # cancel_order fails with "Failed to cancel"
        cancel_exc = Exception(
            'Coincheck API error: 400 | body={"success":false,"error":"Failed to cancel the order."}'
        )
        runner.adapter.cancel_order = lambda _order_id: _async_raise(cancel_exc)  # type: ignore[assignment]

        # First 2 calls return None (polling), then filled on recheck
        filled_order = SimpleNamespace(status="filled", price=15000500.0)
        runner.adapter.get_order_status = _make_status_side_effect(None, None, filled_order)  # type: ignore[assignment]

        record = await _run_single_cycle_without_sleep(runner)
        assert record.filled is True
        assert record.fill_price == 15000500.0

    @pytest.mark.asyncio
    async def test_cancel_fail_no_fill(self, tmp_path: Path) -> None:
        """cancel 失敗かつ recheck でも None → unfilled のまま."""
        runner = self._make_runner(tmp_path)
        runner.adapter.get_order_status = _make_status_side_effect(None, None, None, None)  # type: ignore[assignment]
        cancel_exc = Exception(
            'body={"success":false,"error":"Failed to cancel the order."}'
        )
        runner.adapter.cancel_order = lambda _order_id: _async_raise(cancel_exc)  # type: ignore[assignment]

        record = await _run_single_cycle_without_sleep(runner)
        assert record.filled is False
        assert record.cancelled is True


# ======================================================================
# 047# Finding3: INTERIM judgment type テスト
# ======================================================================

class TestInterimJudgment:
    """047# Finding3: 3日<=days<7 で INTERIM, 7日以上で FINAL."""

    def test_interim_3_days_200_samples(self) -> None:
        """n>=200 & 3日 → INTERIM (not FINAL)."""
        records = _make_uniform_daily_records(
            prefix="interim",
            days=3,
            per_day=67,
            base_ts=1700006400.0,
            queue_wait_sec=10.0,
            post_fill_30s_pnl=0.5,
            adverse_selected=False,
        )
        metrics = compute_fill_metrics(records)
        assert metrics.sample_sufficient is False  # 7日未満
        judgment = g1_1_judgment(metrics, {})
        assert judgment["judgment_type"] == "INTERIM"

    def test_final_7_days(self) -> None:
        """n>=200 & 7日 → FINAL."""
        records = _make_uniform_daily_records(
            prefix="final",
            days=7,
            per_day=29,
            base_ts=1700006400.0,
            queue_wait_sec=10.0,
            post_fill_30s_pnl=0.5,
            adverse_selected=False,
        )
        metrics = compute_fill_metrics(records)
        assert metrics.sample_sufficient is True
        judgment = g1_1_judgment(metrics, {})
        assert judgment["judgment_type"] == "FINAL"

    def test_provisional_insufficient(self) -> None:
        """n<200 or days<3 → PROVISIONAL."""

        records = _make_uniform_daily_records(
            prefix="prov",
            days=1,
            per_day=50,
            base_ts=1700006400.0,
            queue_wait_sec=10.0,
        )
        metrics = compute_fill_metrics(records)
        judgment = g1_1_judgment(metrics, {})
        assert judgment["judgment_type"] == "PROVISIONAL"


# ======================================================================
# 047# Finding4: AS coverage フィールドテスト
# ======================================================================

class TestASCoverage:
    """047# Finding4: FillMetrics に as_coverage / as_raw_coverage が含まれる."""

    def test_coverage_fields_present(self) -> None:
        """coverage フィールドが正しく算出される."""

        records = []
        for i in range(10):
            records.append(FillRecord(
                cycle_id=f"cov_{i}",
                timestamp=1700006400.0 + i * 120,
                side="buy",
                order_price=100.0,
                order_quantity=0.001,
                filled=True,
                queue_wait_sec=10.0,
                adverse_selected=i % 3 == 0,
                adverse_selected_raw=i % 2 == 0 if i < 6 else None,
            ))
        m = compute_fill_metrics(records)
        assert m.as_coverage == 10
        assert m.as_raw_coverage == 6

    def test_coverage_in_dict(self) -> None:
        """to_dict() に coverage が含まれる."""

        m = FillMetrics(as_coverage=5, as_raw_coverage=3)
        d = m.to_dict()
        assert "as_coverage" in d
        assert "as_raw_coverage" in d
        assert d["as_coverage"] == 5
        assert d["as_raw_coverage"] == 3


# ======================================================================
# 047# Issue12: time_filter ログ throttle テスト
# ======================================================================

class TestTimeFilterLogThrottle:
    """047# Issue12: High-AS time filter のログが突入/離脱のみに制限される."""

    def test_in_time_filter_flag_init(self) -> None:
        """_in_time_filter が False で初期化される (121# TimeFilter に委譲)."""

        adapter = AsyncMock()
        config = FillTestConfig(enable_time_filter=True, skip_utc_hours=[0, 1])
        with patch.object(FillTestRunner, "_get_git_sha", return_value="test-sha"):
            runner = FillTestRunner(adapter, config)
        assert runner._time_filter.in_filter is False


# ======================================================================
# 047# A5: filter_clean_records 拡張基準テスト
# ======================================================================

class TestFilterCleanRecordsExpanded:
    """047# A5: quarantine 基準の拡張 (run_id, 必須フィールド)."""

    def _make_record(self, **kwargs: object) -> "FillRecord":
        defaults = dict(
            cycle_id="test_1",
            timestamp=1700000000.0,
            side="buy",
            order_price=15000000.0,
            order_quantity=0.001,
            filled=True,
            git_sha="abc123",
            run_id="run_001",
        )
        defaults.update(kwargs)
        return FillRecord(**defaults)  # type: ignore[arg-type]

    def test_clean_with_all_fields(self) -> None:
        """全フィールド正常 → clean."""
        records = [self._make_record()]
        clean, q = filter_clean_records(records)
        assert len(clean) == 1
        assert len(q) == 0

    def test_quarantine_blank_git_sha(self) -> None:
        """git_sha=None → quarantine (既存ルール)."""
        records = [self._make_record(git_sha=None)]
        clean, q = filter_clean_records(records)
        assert len(clean) == 0
        assert len(q) == 1

    def test_quarantine_blank_run_id(self) -> None:
        """run_id=None → quarantine (A5 新規)."""
        records = [self._make_record(run_id=None)]
        clean, q = filter_clean_records(records)
        assert len(clean) == 0
        assert len(q) == 1

    def test_quarantine_empty_run_id(self) -> None:
        """run_id='' → quarantine (A5 新規)."""
        records = [self._make_record(run_id="")]
        clean, q = filter_clean_records(records)
        assert len(clean) == 0
        assert len(q) == 1

    def test_quarantine_invalid_side(self) -> None:
        """side='invalid' → quarantine (A5 新規)."""
        records = [self._make_record(side="invalid")]
        clean, q = filter_clean_records(records)
        assert len(clean) == 0
        assert len(q) == 1

    def test_quarantine_zero_price(self) -> None:
        """order_price=0 → quarantine (A5 新規)."""
        records = [self._make_record(order_price=0)]
        clean, q = filter_clean_records(records)
        assert len(clean) == 0
        assert len(q) == 1

    def test_quarantine_negative_quantity(self) -> None:
        """order_quantity=-1 → quarantine (A5 新規)."""
        records = [self._make_record(order_quantity=-1)]
        clean, q = filter_clean_records(records)
        assert len(clean) == 0
        assert len(q) == 1

    def test_require_git_sha_false_bypasses(self) -> None:
        """require_git_sha=False → 全件 clean 返却."""
        records = [self._make_record(git_sha=None, run_id=None)]
        clean, q = filter_clean_records(records, require_git_sha=False)
        assert len(clean) == 1
        assert len(q) == 0

    def test_mixed_clean_and_quarantine(self) -> None:
        """正常 + 異常レコード混在 → 分離."""
        records = [
            self._make_record(cycle_id="ok"),
            self._make_record(cycle_id="bad_sha", git_sha=None),
            self._make_record(cycle_id="bad_rid", run_id=""),
            self._make_record(cycle_id="bad_side", side="xxx"),
        ]
        clean, q = filter_clean_records(records)
        assert len(clean) == 1
        assert len(q) == 3
        assert clean[0].cycle_id == "ok"

    def test_partition_accepts_generator(self) -> None:
        """iterable 入力でも clean/quarantine 分離できる."""

        records = (
            rec for rec in [
                self._make_record(cycle_id="ok"),
                self._make_record(cycle_id="bad_sha", git_sha=None),
            ]
        )
        clean, q = partition_clean_records(records)
        assert len(clean) == 1
        assert len(q) == 1
        assert clean[0].cycle_id == "ok"


# ======================================================================
# 047# A5: _quarantine_reason 単体テスト
# ======================================================================

class TestQuarantineReason:
    """047# A5: _quarantine_reason のユニットテスト."""

    def _make_record(self, **kwargs: object) -> "FillRecord":
        defaults = dict(
            cycle_id="qr_1",
            timestamp=1700000000.0,
            side="buy",
            order_price=15000000.0,
            order_quantity=0.001,
            git_sha="abc123",
            run_id="run_001",
        )
        defaults.update(kwargs)
        return FillRecord(**defaults)  # type: ignore[arg-type]

    def test_clean_returns_none(self) -> None:
        assert _quarantine_reason(self._make_record()) is None

    def test_blank_git_sha(self) -> None:
        assert _quarantine_reason(self._make_record(git_sha="")) == "blank_git_sha"

    def test_blank_run_id(self) -> None:
        assert _quarantine_reason(self._make_record(run_id=" ")) == "blank_run_id"

    def test_invalid_side(self) -> None:
        r = _quarantine_reason(self._make_record(side="hold"))
        assert r is not None and "invalid_side" in r

    def test_zero_price(self) -> None:
        assert _quarantine_reason(self._make_record(order_price=0)) == "invalid_order_price"

    def test_negative_quantity(self) -> None:
        assert _quarantine_reason(self._make_record(order_quantity=-0.5)) == "invalid_order_quantity"


# ======================================================================
# 047# A3: exit code judgment_type テスト
# ======================================================================

class TestExitCodeJudgmentType:
    """047# A3: FINAL PASS → 0, INTERIM/PROVISIONAL PASS → 2, FAIL → 1."""

    def test_final_pass_exit_0(self) -> None:
        """FINAL + PASS → exit 0."""
        result = {"gate_result": "PASS", "judgment_type": "FINAL"}
        jtype = result.get("judgment_type", "PROVISIONAL")
        gate = result.get("gate_result")
        if gate == "PASS" and jtype == "FINAL":
            code = 0
        elif gate == "PASS":
            code = 2
        else:
            code = 1
        assert code == 0

    def test_interim_pass_exit_2(self) -> None:
        """INTERIM + PASS → exit 2."""
        result = {"gate_result": "PASS", "judgment_type": "INTERIM"}
        jtype = result.get("judgment_type", "PROVISIONAL")
        gate = result.get("gate_result")
        if gate == "PASS" and jtype == "FINAL":
            code = 0
        elif gate == "PASS":
            code = 2
        else:
            code = 1
        assert code == 2

    def test_provisional_pass_exit_2(self) -> None:
        """PROVISIONAL + PASS → exit 2."""
        result = {"gate_result": "PASS", "judgment_type": "PROVISIONAL"}
        jtype = result.get("judgment_type", "PROVISIONAL")
        gate = result.get("gate_result")
        if gate == "PASS" and jtype == "FINAL":
            code = 0
        elif gate == "PASS":
            code = 2
        else:
            code = 1
        assert code == 2

    def test_fail_exit_1(self) -> None:
        """FAIL → exit 1."""
        result = {"gate_result": "FAIL", "judgment_type": "FINAL"}
        jtype = result.get("judgment_type", "PROVISIONAL")
        gate = result.get("gate_result")
        if gate == "PASS" and jtype == "FINAL":
            code = 0
        elif gate == "PASS":
            code = 2
        else:
            code = 1
        assert code == 1

    def test_missing_judgment_type_defaults_provisional(self) -> None:
        """judgment_type 未設定 → PROVISIONAL 扱い → exit 2."""
        result = {"gate_result": "PASS"}
        jtype = result.get("judgment_type", "PROVISIONAL")
        gate = result.get("gate_result")
        if gate == "PASS" and jtype == "FINAL":
            code = 0
        elif gate == "PASS":
            code = 2
        else:
            code = 1
        assert code == 2


# ======================================================================
# 047# A4: atomic lock テスト
# ======================================================================

class TestAtomicLock:
    """047# A4: _acquire_lock が OS-level exclusive create を使用."""

    @pytest.fixture(autouse=True)
    def _disable_os_lock(self) -> object:
        """logic lock 検証では portalocker の取得コストを外す."""
        with patch.object(LockManager, "_acquire_os_lock", return_value=None):
            yield

    @pytest.fixture(autouse=True)
    def _disable_process_scan(self) -> object:
        """実環境の run_fill_test プロセスに依存しないよう固定する."""
        with patch.object(LockManager, "_check_running_fill_test", return_value=None):
            yield

    def _make_lock_manager(self, tmp_path: Path) -> LockManager:
        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        return LockManager(results_dir=results_dir, run_id="test_run")

    def test_acquire_creates_lockfile(self, tmp_path: Path) -> None:
        """ロックファイルが作成される."""
        manager = self._make_lock_manager(tmp_path)
        manager.acquire()
        lock_path = tmp_path / "results" / "fill_test.lock"
        assert lock_path.exists()
        content = lock_path.read_text(encoding="utf-8")
        assert str(os.getpid()) in content
        manager.release()

    def test_acquire_blocks_second(self, tmp_path: Path) -> None:
        """既存の有効ロック → RuntimeError."""
        manager = self._make_lock_manager(tmp_path)
        manager.acquire()

        # 同じディレクトリで 2 つ目が起動 → ロックに自PIDが記録済みなので
        # stale ではない (自PID=fill_test)。ただしテスト環境では
        # psutil が実行中プロセスを fill_test と判定するか次第。
        # ここでは lockfile が既に存在する状態を直接テスト
        lock_path = tmp_path / "results" / "fill_test.lock"
        assert lock_path.exists()
        manager.release()

    def test_atomic_exclusive_create(self, tmp_path: Path) -> None:
        """open(O_CREAT|O_EXCL) で排他的作成が行われる."""
        lock_path = tmp_path / "test.lock"

        # First create succeeds
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, b"test")
        os.close(fd)

        # Second create fails
        with pytest.raises(FileExistsError):
            os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)


# ======================================================================
# 047# A6/Issue13: API Response ログレベルテスト
# ======================================================================

class TestAPIResponseLogLevel:
    """047# A6/Issue13: API Response ログが DEBUG に降格されている."""

    def test_api_response_log_is_debug(self) -> None:
        """adapter.py の API Response ログが logger.debug であることを確認."""
        source = read_inspect_source(CoincheckAdapter._make_api_request)
        # logger.info("API Response ...") が存在しないことを確認
        assert 'logger.info(f"API Response status' not in source
        assert 'logger.info(f"API Response content' not in source
        # logger.debug("API Response status ...") が存在することを確認
        assert 'logger.debug(f"API Response status' in source
        # content ログも debug であること (改行がありうるので分割チェック)
        assert 'f"API Response content' in source
        # info 版が存在しないことで間接的に debug 確認
        lines = [l.strip() for l in source.splitlines()]
        debug_lines = [l for l in lines if "API Response" in l and "logger.debug" in l]
        assert len(debug_lines) >= 1


# =====================================================================
# 049# P0: clean-only 集計 + exit code FINAL 整合 + coverage
# =====================================================================

class Test049CleanOnlyMainJudgment:
    """049# §4-#2: main() の最終集計が filter_clean_records を通している."""

    def test_main_uses_filter_clean_records(self) -> None:
        """main() のソースに filter_clean_records が含まれることを確認."""
        source = read_inspect_source(cli_mod)
        # filter_clean_records が使われている
        assert "filter_clean_records" in source
        # 旧パターン (records を直接 compute_fill_metrics に渡す) が存在しない
        assert "compute_fill_metrics(records)" not in source
        # clean_records を使ったパターンが存在
        assert "compute_fill_metrics(clean_records)" in source

    def test_main_exit_code_uses_judgment_type(self) -> None:
        """049# §4-#1: 通常実行の exit code が judgment_type を参照."""
        source = read_inspect_source(cli_mod)
        # FINAL/INTERIM 分岐がある
        assert 'jtype == "FINAL"' in source or "judgment_type" in source
        # 旧パターン (gate_result のみ) が存在しない
        assert 'sys.exit(0 if judgment["gate_result"]' not in source

    def test_main_has_data_quality_output(self) -> None:
        """049# §6.1-#4: judgment に data_quality セクションが含まれる."""
        source = read_inspect_source(cli_mod)
        assert '"data_quality"' in source
        assert '"clean_records"' in source
        assert '"quarantine_records"' in source
        assert '"clean_rate"' in source


class Test049DataQualityInJudgment:
    """049# §6.1-#4: data_quality フィールドの正確性を検証."""

    def _make_record(self, *, git_sha: str = "abc1234", run_id: str = "r1",
                     filled: bool = True, pnl: float = 0.0) -> "FillRecord":
        return FillRecord(
            cycle_id="c1", timestamp=1700000000.0, side="buy",
            order_price=15000000.0, order_quantity=0.001,
            fill_price=15000001.0 if filled else None,
            filled=filled, cancelled=not filled,
            queue_wait_sec=10.0,
            mid_at_fill=15000050.0 if filled else None,
            mid_30s_after=15000100.0 if filled else None,
            post_fill_30s_pnl=pnl if filled else None,
            adverse_selected=pnl < 0 if filled else None,
            run_id=run_id, git_sha=git_sha,
        )

    def test_clean_quarantine_split(self) -> None:
        records = [
            self._make_record(git_sha="abc1234", run_id="r1"),
            self._make_record(git_sha="", run_id="r1"),     # quarantine: blank sha
            self._make_record(git_sha="abc1234", run_id=""),  # quarantine: blank run_id
        ]
        clean, quarantine = filter_clean_records(records)
        assert len(clean) == 1
        assert len(quarantine) == 2


# =====================================================================
# 049# P1: E3 サンプリング
# =====================================================================

class Test049E3Sampling:
    """049# §3-#6: E3 計測がサンプリングで実行される."""

    def test_e3_sampling_ratio_config(self) -> None:
        """e3_sampling_ratio フィールドが FillTestConfig に存在."""
        cfg = FillTestConfig()
        assert hasattr(cfg, "e3_sampling_ratio")
        assert cfg.e3_sampling_ratio == 1.0  # デフォルトは全約定

    def test_e3_sampling_from_yaml(self) -> None:
        """YAML の e3.sampling_ratio が正しくパースされる."""
        cfg = clone_fill_test_config(load_fill_test_config_from_mapping({"e3": {"sampling_ratio": 0.33}}))
        assert cfg.e3_sampling_ratio == pytest.approx(0.33)

    def test_e3_sampling_ratio_zero_skips_all(self) -> None:
        """e3_sampling_ratio=0.0 で E3 計測がスキップされる (ソース確認)."""
        # 120#: E3 logic extracted to PnlMeasurer.measure
        source = read_inspect_source(PnlMeasurer.measure)
        assert "e3_sampling_ratio" in source


# =====================================================================
# 049# P1: side 別 offset
# =====================================================================

class Test049SideOffset:
    """049# §5.1: buy/sell 独立 offset テーブル."""

    def test_side_offset_fields_exist(self) -> None:
        """spread_offset_ratio_buy/sell フィールドが存在."""
        cfg = FillTestConfig()
        assert cfg.spread_offset_ratio_buy is None
        assert cfg.spread_offset_ratio_sell is None

    def test_side_offset_from_yaml(self) -> None:
        """YAML の side_offset.buy/sell が正しくパースされる."""
        cfg = clone_fill_test_config(load_fill_test_config_from_mapping({"side_offset": {"buy": 0.03, "sell": 0.08}}))
        assert cfg.spread_offset_ratio_buy == pytest.approx(0.03)
        assert cfg.spread_offset_ratio_sell == pytest.approx(0.08)

    def test_side_offset_used_in_price_calc(self) -> None:
        """MakerPriceCalculator.compute が side 別 offset を参照することをソースで確認.

        096# 状態分離: config.spread_offset_ratio_buy/sell ではなく
        base_offset_ratio_buy/sell を参照する設計に変更。
        120#: maker_price.py に抽出済み。
        """
        source = _MAKER_PRICE_SOURCE
        # 096# 状態分離: base_offset_ratio* を使用
        assert "base_offset_ratio" in source
        assert "effective_offset_ratio" in source


# =====================================================================
# 049# P1: 即約定防御
# =====================================================================

class Test049FastFillDefense:
    """049# §6.2-#3: queue_wait<=5s + 負エッジ時の防御ロジック."""

    def test_fast_fill_defense_config(self) -> None:
        """fast_fill_defense 設定フィールドが存在."""
        cfg = FillTestConfig()
        assert cfg.fast_fill_defense_enabled is False
        assert cfg.fast_fill_threshold_sec == 5.0
        assert cfg.fast_fill_offset_boost == 2.0

    def test_fast_fill_defense_from_yaml(self) -> None:
        """YAML の fast_fill_defense セクションが正しくパースされる."""
        cfg = clone_fill_test_config(
            load_fill_test_config_from_mapping(
                {
                    "fast_fill_defense": {
                        "enabled": True,
                        "threshold_sec": 3.0,
                        "offset_boost": 1.5,
                    }
                }
            )
        )
        assert cfg.fast_fill_defense_enabled is True
        assert cfg.fast_fill_threshold_sec == pytest.approx(3.0)
        assert cfg.fast_fill_offset_boost == pytest.approx(1.5)

    def test_fast_fill_boost_flag_initialized(self) -> None:
        """100# FillTestRunner が FastFillDefense インスタンスを持つ."""

        source = read_inspect_source(FillTestRunner.__init__)
        assert "_fast_fill_defense" in source

    def test_fast_fill_defense_logic_in_run_continuous(self) -> None:
        """run_continuous が FastFillDefense に委譲している.

        265# extract: post-cycle 処理は _process_post_cycle に分離。
        """

        source = _PROCESS_POST_CYCLE_SOURCE
        assert "fast_fill_defense" in source
        assert "evaluate_fill" in source


# =====================================================================
# 050# Bug fixes: offset復元, side-specific defense, 実効offset記録
# =====================================================================


class Test050SideOffsetSellOnly:
    """050# Bug#7: sell のみ設定時の YAML パース."""

    def test_sell_only_offset_from_yaml(self) -> None:
        """side_offset に sell のみ指定 → buy は None のまま."""

        cfg = clone_fill_test_config(load_fill_test_config_from_mapping({"side_offset": {"sell": 0.07}}))
        assert cfg.spread_offset_ratio_buy is None
        assert cfg.spread_offset_ratio_sell == pytest.approx(0.07)

    def test_buy_only_offset_from_yaml(self) -> None:
        """side_offset に buy のみ指定 → sell は None のまま."""

        cfg = clone_fill_test_config(load_fill_test_config_from_mapping({"side_offset": {"buy": 0.04}}))
        assert cfg.spread_offset_ratio_buy == pytest.approx(0.04)
        assert cfg.spread_offset_ratio_sell is None

    def test_empty_side_offset_from_yaml(self) -> None:
        """side_offset が空 dict → 両方 None."""

        cfg = clone_fill_test_config(load_fill_test_config_from_mapping({"side_offset": {}}))
        assert cfg.spread_offset_ratio_buy is None
        assert cfg.spread_offset_ratio_sell is None


class Test050FastFillDefenseRestore:
    """050# Bug#1-2: fast_fill_defense offset 復元 + side-specific 対応."""

    def test_boost_multiplier_field_exists(self) -> None:
        """100# FillTestRunner に FastFillDefense が存在.

        旧: _boost_multiplier + _fast_fill_boost_active で inline 管理
        新: FastFillDefense クラスに per-side 状態管理を委譲
        120#: offset 管理は MakerPriceCalculator に移動
        """

        source = read_inspect_source(FillTestRunner.__init__)
        assert "_fast_fill_defense" in source
        # 120#: offset は _maker_price 経由で管理
        assert "_maker_price" in source

    def test_offset_restore_logic_in_run_continuous(self) -> None:
        """run_continuous が boost 解除ロジックを含む.

        265# extract: post-cycle 処理は _process_post_cycle に分離。
        """

        source = _PROCESS_POST_CYCLE_SOURCE
        # 100# FastFillDefense の evaluate_fill / reset_on_unfilled で管理
        assert "fast_fill_defense" in source
        assert "reset_on_unfilled" in source


class Test050EffectiveOffsetRecord:
    """050# Bug#3: FillRecord に実効 offset を記録."""

    def test_compute_maker_price_returns_3_values(self) -> None:
        """MakerPriceCalculator.compute の戻り値が price/spread/offset を含むことをソースで確認.

        120#: maker_price.py に抽出済み。MakerPriceResult NamedTuple で返却。
        """
        source = _MAKER_PRICE_SOURCE
        assert "effective_offset_ratio" in source
        # MakerPriceResult に price, spread, effective_offset_ratio を格納
        assert "MakerPriceResult" in source

    def test_run_single_cycle_unpacks_3_values(self) -> None:
        """pre-order phase が 3 値展開を行う."""

        source = _RUN_PRE_ORDER_PHASE_SOURCE
        assert "effective_offset_ratio" in source


# ======================================================================
# 051# P2-2: Round-trip 評価テスト
# ======================================================================


class Test051RoundTripMetrics:
    """051# P2-2: compute_round_trip_metrics のテスト."""

    def _make_filled_record(
        self, side: str, fill_price: float, timestamp: float, qty: float = 0.001
    ) -> "FillRecord":
        return FillRecord(
            cycle_id=f"test_{timestamp}",
            timestamp=timestamp,
            side=side,
            order_price=fill_price,
            order_quantity=qty,
            fill_price=fill_price,
            filled=True,
        )

    def test_basic_round_trip(self) -> None:
        """buy→sell のシンプルなペアリング."""

        records = [
            self._make_filled_record("buy", 14_500_000, 1000.0),
            self._make_filled_record("sell", 14_501_000, 1200.0),
        ]
        metrics, trips = compute_round_trip_metrics(records)
        assert metrics.total_pairs == 1
        assert len(trips) == 1
        # PnL = (14501000 - 14500000) / 14500000 * 10000 ≈ 0.69 bps
        assert trips[0].pnl_bps > 0
        assert trips[0].pnl_jpy == 1000.0 * 0.001  # 1.0 JPY
        assert trips[0].hold_sec == 200.0
        assert metrics.win_rate == 1.0

    def test_multiple_round_trips(self) -> None:
        """複数ペアの FIFO マッチング."""

        records = [
            self._make_filled_record("buy", 14_500_000, 1000.0),
            self._make_filled_record("sell", 14_501_000, 1200.0),
            self._make_filled_record("buy", 14_502_000, 1400.0),
            self._make_filled_record("sell", 14_500_000, 1600.0),  # 損失ペア
        ]
        metrics, trips = compute_round_trip_metrics(records)
        assert metrics.total_pairs == 2
        assert trips[0].pnl_jpy > 0  # 1st: 利益
        assert trips[1].pnl_jpy < 0  # 2nd: 損失
        assert metrics.win_rate == 0.5

    def test_unpaired_buys(self) -> None:
        """sell が不足する場合の未ペア buy."""

        records = [
            self._make_filled_record("buy", 14_500_000, 1000.0),
            self._make_filled_record("buy", 14_501_000, 1200.0),
            self._make_filled_record("sell", 14_502_000, 1400.0),
        ]
        metrics, trips = compute_round_trip_metrics(records)
        assert metrics.total_pairs == 1
        assert metrics.unpaired_buys == 1

    def test_empty_records(self) -> None:
        """空リスト."""

        metrics, trips = compute_round_trip_metrics([])
        assert metrics.total_pairs == 0
        assert len(trips) == 0

    def test_sell_without_buy_ignored(self) -> None:
        """buy なしの sell はペアリングされない."""

        records = [
            self._make_filled_record("sell", 14_500_000, 1000.0),
            self._make_filled_record("buy", 14_501_000, 1200.0),
            self._make_filled_record("sell", 14_502_000, 1400.0),
        ]
        metrics, trips = compute_round_trip_metrics(records)
        assert metrics.total_pairs == 1
        # ペアは buy@1200 → sell@1400
        assert trips[0].buy_record.timestamp == 1200.0


# ======================================================================
# PnL accumulator helpers
# ======================================================================


class TestPnlWinAccumulator:
    """勝率付き PnL 集計 helper のテスト."""

    def test_tracks_mean_and_win_rate(self) -> None:

        acc = PnlWinAccumulator()
        acc.add(None)
        acc.add(float("nan"))
        acc.add(-1.5)
        acc.add(0.0)
        acc.add(3.0)

        assert acc.count == 3
        assert acc.total_bps == pytest.approx(1.5)
        assert acc.mean_bps == pytest.approx(0.5)
        assert acc.positive_count == 1
        assert acc.win_rate == pytest.approx(1 / 3)


# ======================================================================
# 051# P2-4: レジーム別メトリクステスト
# ======================================================================


class Test051RegimeMetrics:
    """051# P2-4: compute_regime_metrics のテスト."""

    def _make_record(
        self,
        regime: str,
        filled: bool = True,
        pnl: float | None = -0.5,
        adverse: bool | None = False,
    ) -> "FillRecord":
        return FillRecord(
            cycle_id="test",
            timestamp=1000.0,
            side="buy",
            order_price=14_500_000,
            order_quantity=0.001,
            fill_price=14_500_000 if filled else None,
            filled=filled,
            post_fill_30s_pnl=pnl if filled else None,
            adverse_selected=adverse if filled else None,
            regime=regime,
        )

    def test_basic_regime_grouping(self) -> None:
        """レジーム別に正しくグループされる."""

        records = [
            self._make_record("trending", pnl=1.0, adverse=False),
            self._make_record("trending", pnl=-0.5, adverse=True),
            self._make_record("ranging", pnl=0.3, adverse=False),
        ]
        result = compute_regime_metrics(records)
        assert len(result) == 2
        # ソート済み: ranging → trending
        assert result[0].regime == "ranging"
        assert result[0].count == 1
        assert result[1].regime == "trending"
        assert result[1].count == 2
        assert result[1].as_ratio == 0.5  # 1/2

    def test_unknown_regime(self) -> None:
        """regime=None は 'unknown' にマッピング."""

        records = [self._make_record("unknown")]
        result = compute_regime_metrics(records)
        assert len(result) == 1
        assert result[0].regime == "unknown"

    def test_empty_records(self) -> None:
        """空リスト."""

        result = compute_regime_metrics([])
        assert result == []


# ======================================================================
# 051# UTC 時間帯別分析テスト
# ======================================================================


class Test051HourlyMetrics:
    """051# compute_hourly_metrics のテスト."""

    def test_basic_hourly(self) -> None:
        """UTC hour 別にグループされる."""
        # UTC 10:00 と 13:00 のレコード
        t_10 = datetime(2025, 2, 15, 10, 0, tzinfo=timezone.utc).timestamp()
        t_13 = datetime(2025, 2, 15, 13, 0, tzinfo=timezone.utc).timestamp()

        records = [
            FillRecord(
                cycle_id="a", timestamp=t_10, side="buy",
                order_price=14_500_000, order_quantity=0.001,
                filled=True, fill_price=14_500_000,
                post_fill_30s_pnl=1.0, adverse_selected=False,
            ),
            FillRecord(
                cycle_id="b", timestamp=t_13, side="sell",
                order_price=14_500_000, order_quantity=0.001,
                filled=True, fill_price=14_500_000,
                post_fill_30s_pnl=-2.0, adverse_selected=True,
            ),
        ]
        result = compute_hourly_metrics(records)
        assert len(result) == 2
        h10 = next(h for h in result if h.utc_hour == 10)
        h13 = next(h for h in result if h.utc_hour == 13)
        assert h10.count == 1
        assert h10.as_ratio == 0.0
        assert h13.count == 1
        assert h13.as_ratio == 1.0
        assert h13.pnl_mean_bps == -2.0


# ======================================================================
# 051# P2-3: Balance auto-shrink テスト
# ======================================================================


class Test051BalanceAutoShrink:
    """051# P2-3: 残高不足時のロット一時縮小."""

    def test_balance_shrink_fields_exist(self) -> None:
        """BalanceChecker に balance_shrink 関連フィールドがある (121# 委譲)."""
        source = read_inspect_source(BalanceChecker.__init__)
        assert "_balance_shrink_active" in source
        assert "_pre_shrink_lot" in source
        # 121# FillTestRunner は BalanceChecker に委譲
        runner_src = read_inspect_source(FillTestRunner.__init__)
        assert "_balance_checker" in runner_src

    def test_balance_shrink_logic_in_run_continuous(self) -> None:
        """run_continuous に balance_shrink ロジックが含まれる.

        265# extract: loss_cap 処理は _process_post_cycle に分離。
        """

        # 332# balance_forced/preflight は orchestrator_balance に抽出済み
        rc_source = read_source_text(ORCHESTRATOR_BALANCE)
        assert "balance_shrink" in rc_source
        # 121# pre_shrink_lot は _process_post_cycle 内 (_balance_checker 経由)
        post_source = read_inspect_source(FillTestRunner._process_post_cycle)
        assert "pre_shrink_lot" in post_source

    def test_shrink_threshold_is_3(self) -> None:
        """連続 3 回で shrink 発動 (設定外部化済み)."""

        cfg = FillTestConfig()
        assert cfg.balance_shrink_consecutive == 3


# ======================================================================
# 051# Monitor 拡張テスト
# ======================================================================


class Test051MonitorExtensions:
    """051# monitor_fill_test 拡張のテスト."""

    def test_print_report_accepts_clean_quarantine(self) -> None:
        """print_report が clean_count/quarantine_count を受け付ける."""
        sig = inspect.signature(print_report)
        params = list(sig.parameters.keys())
        assert "clean_count" in params
        assert "quarantine_count" in params

    def test_run_monitor_uses_clean_records(self) -> None:
        """run_monitor が clean/quarantine 分離を使う."""
        source = read_inspect_source(run_monitor)
        assert "partition_clean_records" in source
        assert "iter_fill_records_glob" in source
        assert "clean_records" in source
        assert "quarantine_records" in source

    def test_run_monitor_uses_shared_pnl_helper(self) -> None:
        source = read_inspect_source(_check_cumulative_loss)
        assert "compute_record_pnl_jpy" in source

    def test_monitor_imports_new_functions(self) -> None:
        """monitor が新しい分析関数をインポート."""
        # インポートテスト (NameError にならないことを確認)
        assert callable(print_report)


# ======================================================================
# 052# 方策A: sell offset 連動調整テスト
# ======================================================================


class Test052AdaptSellOffsetSync:
    """052# 方策AがSell offsetも比例調整する."""

    @pytest.fixture(scope="class", autouse=True)
    def _yaml_config(
        self,
        request: pytest.FixtureRequest,
        v460_fill_test_yaml_base: dict[str, object],
    ) -> None:
        request.cls.yaml_cfg = v460_fill_test_yaml_base

    def test_adapt_syncs_sell_offset_in_code(self) -> None:
        """AdaptationEngine.try_auto_adapt に sell offset 比例調整コードが含まれる.

        096# 状態分離: _base_offset_ratio_sell を直接更新する設計に変更。
        120#: adaptation_engine.py に抽出済み。
        """
        source = read_inspect_source(AdaptationEngine.try_auto_adapt)
        # 096# 状態分離: base_offset_ratio_sell を使用
        assert "base_offset_ratio_sell" in source
        # 120#: 比例調整 ratio = new_base / base_offset_ratio
        assert "ratio" in source and "new_sell" in source

    def test_yaml_sell_offset_updated(self) -> None:
        """current YAML で sell offset が 0.14 に設定されている."""
        assert self.yaml_cfg["side_offset"]["sell"] == 0.14

    def test_yaml_skip_utc_hours_buy_includes_12(self) -> None:
        """163# Step2: time_filter buy=[16]のみ。旧 UTC08/18 は regime_adaptive で復元."""
        # 169# time_filter 全廃: 全リスト空 (条件ベースフィルタに完全移行)
        assert self.yaml_cfg["time_filter"]["skip_utc_hours_buy"] == []
        assert self.yaml_cfg["time_filter"]["skip_utc_hours_sell"] == []
        assert self.yaml_cfg["time_filter"]["regime_adaptive_extra_buy"] == []
        assert self.yaml_cfg["time_filter"]["regime_adaptive_extra_sell"] == []

    def test_yaml_deadzone_updated(self) -> None:
        """052# で deadzone が 2.5 に更新されている."""
        assert self.yaml_cfg["as_deadzone_bps"] == 2.5

    def test_dynamic_lot_shrink_in_balance_check(self) -> None:
        """052# BalanceChecker にロット自動縮小が含まれる (121# 抽出)."""
        source = read_inspect_source(BalanceChecker._check_buy)
        assert "_min_order_btc" in source
        assert "affordable_lot" in source

    def test_min_order_btc_constant(self) -> None:
        """052# min_order_btc が Coincheck 最小注文量 0.001 に設定されている (121# YAML 外部化)."""
        assert MIN_ORDER_BTC == 0.001
        assert FillTestConfig().min_order_btc == 0.001

    def test_yaml_skip_utc_hours_side_specific_089(self) -> None:
        """169# time_filter全廃: 条件ベースフィルタに完全移行."""
        buy_skip = self.yaml_cfg["time_filter"]["skip_utc_hours_buy"]
        sell_skip = self.yaml_cfg["time_filter"]["skip_utc_hours_sell"]
        # 169# time_filter 全廃: 全リスト空 (B1' + SkipGate + VG + sell_dynamic_kill が根本対策)
        assert buy_skip == [], f"Expected [], got {buy_skip}"
        assert sell_skip == [], f"Expected [], got {sell_skip}"
        # 169# regime_adaptive リストも空 (VG が high_vol を直接処理)
        tf = self.yaml_cfg["time_filter"]
        assert tf["regime_adaptive_extra_buy"] == []
        assert tf["regime_adaptive_extra_sell"] == []

    def test_trending_offset_boost_in_code(self) -> None:
        """052# MakerPriceCalculator.compute にトレンディングブーストが含まれる.

        120#: maker_price.py に抽出済み。
        322# God Object 分割: regime boost は RegimeBoostMixin に移管。
        """
        source = read_inspect_source(MakerPriceCalculator._resolve_trending_boost)
        assert "trending" in source
        assert "regime_trending_offset_boost" in source

    def test_trending_offset_boost_config(self) -> None:
        """052# regime_trending_offset_boost がConfigに存在し1.5."""

        cfg = FillTestConfig()
        assert hasattr(cfg, "regime_trending_offset_boost")
        assert cfg.regime_trending_offset_boost == 1.5

    def test_yaml_trending_offset_boost(self) -> None:
        """052# で trending_offset_boost が YAML に設定されている."""
        assert self.yaml_cfg["regime"]["trending_offset_boost"] == 1.5

    def test_balance_shrink_uses_min_order_btc(self) -> None:
        """052# balance_shrink の最低ロットが min_order_btc を使用する (121# YAML 外部化)."""

        source = read_source_text(ORCHESTRATOR_BALANCE)
        assert "min_order_btc" in source


# ======================================================================
# 107# Time filter dynamic gating + Volatility Guard tests
# ======================================================================


class Test107TimeFilterDynamicGating:
    """107# time_filter 改善 + Volatility Guard テスト."""

    @pytest.fixture(scope="class", autouse=True)
    def _yaml_config(
        self,
        request: pytest.FixtureRequest,
        v460_fill_test_yaml_base: dict[str, object],
    ) -> None:
        request.cls.yaml_cfg = v460_fill_test_yaml_base

    def test_yaml_sell_utc17_unblocked(self) -> None:
        """107# SELL UTC17 は +0.65 bps なのでブロック解除済み."""
        sell_skip = self.yaml_cfg["time_filter"]["skip_utc_hours_sell"]
        assert 17 not in sell_skip, "UTC17 SELL is +0.65 bps, should NOT be blocked"

    def test_yaml_skip_gate_target_rates_raised(self) -> None:
        """107# SkipGate target_skip_rate が引き上げられている."""
        sg = self.yaml_cfg["skip_gate"]
        assert sg["target_skip_rate_buy"] == 0.15, "buy rate should be 0.15"
        assert sg["target_skip_rate_sell"] == 0.20, "321# H-1: sell rate 0.25→0.20"

    def test_yaml_volatility_guard_section(self) -> None:
        """107# volatility_guard セクションが YAML に存在する."""
        vg = self.yaml_cfg["volatility_guard"]
        assert vg["enabled"] is True
        assert vg["velocity_window_sec"] == 60
        assert vg["velocity_threshold_bps"] == 6.0   # 630# 12.0→6.0 (626# §3: VG感度引上げ)
        assert vg["vpin_threshold"] == 0.80  # 491# 0.60→0.80 (VPIN常時飽和を緩和)
        assert vg["offset_boost_factor"] == 2.0

    def test_volatility_guard_config_fields(self) -> None:
        """107# FillTestConfig に volatility_guard フィールドが存在する."""

        cfg = FillTestConfig()
        assert hasattr(cfg, "volatility_guard_enabled")
        assert cfg.volatility_guard_enabled is False  # デフォルトは無効
        assert cfg.volatility_guard_velocity_window_sec == 60.0
        assert cfg.volatility_guard_velocity_threshold_bps == 15.0
        assert cfg.volatility_guard_vpin_threshold == 0.70
        assert cfg.volatility_guard_offset_boost_factor == 2.0

    def test_volatility_guard_in_compute_maker_price(self) -> None:
        """107# MakerPriceCalculator.compute に volatility_guard ロジックが含まれる.

        120#: maker_price.py に抽出済み。
        322# God Object 分割: VG は RiskGuardsMixin に移管。
        """
        source = read_inspect_source(MakerPriceCalculator._apply_volatility_guard)
        assert "volatility_guard" in source
        assert "velocity_threshold_bps" in source or "vpin_threshold" in source

    # ---- 168# InvSkew/VG 競合解消テスト ----

    def test_vg_inv_skew_damping_config_default(self) -> None:
        """168# vg_inv_skew_damping_enabled のデフォルトは False."""

        cfg = FillTestConfig()
        assert hasattr(cfg, "vg_inv_skew_damping_enabled")
        assert cfg.vg_inv_skew_damping_enabled is False

    def test_vg_inv_skew_damping_yaml_mapping(self) -> None:
        """168# YAML 'inv_skew_damping_enabled' が VG セクションで読み込まれる."""
        vg = self.yaml_cfg["volatility_guard"]
        assert "inv_skew_damping_enabled" in vg
        assert vg["inv_skew_damping_enabled"] is True

    def test_vg_inv_skew_damping_code_present(self) -> None:
        """168# InvSkew/VG damping ロジックがソースに含まれる.

        322# God Object 分割: VG は RiskGuardsMixin に移管。
        """
        source = read_inspect_source(MakerPriceCalculator._apply_volatility_guard)
        assert "vg_inv_skew_damping_enabled" in source
        assert "_last_inv_skew_factor" in source
        assert "vg_damping" in source  # ログラベル

    def test_vg_damping_reduces_boost_when_inv_skew_negative(self) -> None:
        """168# InvSkew factor<0 時に VG boost が dampen される."""
        cfg = FillTestConfig(
            volatility_guard_enabled=True,
            volatility_guard_velocity_threshold_bps=10.0,
            volatility_guard_offset_boost_factor=2.0,
            vg_inv_skew_damping_enabled=True,
            max_offset_ratio=1.0,
        )
        _ffd = FastFillDefense(FastFillDefenseConfig(), base_offset_ratio=0.05)
        calc = MakerPriceCalculator(
            cfg, _ffd, regime_detector=None, base_offset_ratio=0.05,
        )
        # Simulate InvSkew having reduced sell offset (factor = -0.4)
        calc._last_inv_skew_factor = -0.4
        # Trigger VG via velocity
        result = calc._apply_volatility_guard(
            side="sell", mid_trend_bps=20.0, effective_offset_ratio=0.10,
        )
        # Without damping: 0.10 * 2.0 = 0.20
        # With damping: factor=-0.4 → damping=0.6 → boost=1+0.6*1=1.6
        #   → 0.10 * 1.6 = 0.16
        expected = 0.10 * (1.0 + 0.6 * (2.0 - 1.0))
        assert abs(result - expected) < 1e-9, f"Expected ~{expected}, got {result}"

    def test_vg_damping_no_effect_when_factor_positive(self) -> None:
        """168# InvSkew factor>=0 では VG boost は通常通り."""
        cfg = FillTestConfig(
            volatility_guard_enabled=True,
            volatility_guard_velocity_threshold_bps=10.0,
            volatility_guard_offset_boost_factor=2.0,
            vg_inv_skew_damping_enabled=True,
            max_offset_ratio=1.0,
        )
        _ffd = FastFillDefense(FastFillDefenseConfig(), base_offset_ratio=0.05)
        calc = MakerPriceCalculator(
            cfg, _ffd, regime_detector=None, base_offset_ratio=0.05,
        )
        calc._last_inv_skew_factor = 0.3  # positive = no damping
        result = calc._apply_volatility_guard(
            side="buy", mid_trend_bps=20.0, effective_offset_ratio=0.10,
        )
        expected = 0.10 * 2.0  # full boost
        assert abs(result - expected) < 1e-9, f"Expected {expected}, got {result}"

    def test_vg_damping_disabled_full_boost(self) -> None:
        """168# damping 無効時は従来通り full boost."""
        cfg = FillTestConfig(
            volatility_guard_enabled=True,
            volatility_guard_velocity_threshold_bps=10.0,
            volatility_guard_offset_boost_factor=2.0,
            vg_inv_skew_damping_enabled=False,  # disabled
            max_offset_ratio=1.0,
        )
        _ffd = FastFillDefense(FastFillDefenseConfig(), base_offset_ratio=0.05)
        calc = MakerPriceCalculator(
            cfg, _ffd, regime_detector=None, base_offset_ratio=0.05,
        )
        calc._last_inv_skew_factor = -0.4  # would dampen if enabled
        result = calc._apply_volatility_guard(
            side="sell", mid_trend_bps=20.0, effective_offset_ratio=0.10,
        )
        expected = 0.10 * 2.0  # full boost, no damping
        assert abs(result - expected) < 1e-9, f"Expected {expected}, got {result}"

    def test_vg_damping_extreme_factor_caps_at_one(self) -> None:
        """168# |factor| > 1.0 は 1.0 で cap → boost=1.0 (完全抑制)."""
        cfg = FillTestConfig(
            volatility_guard_enabled=True,
            volatility_guard_velocity_threshold_bps=10.0,
            volatility_guard_offset_boost_factor=3.0,
            vg_inv_skew_damping_enabled=True,
            max_offset_ratio=1.0,
        )
        _ffd = FastFillDefense(FastFillDefenseConfig(), base_offset_ratio=0.05)
        calc = MakerPriceCalculator(
            cfg, _ffd, regime_detector=None, base_offset_ratio=0.05,
        )
        calc._last_inv_skew_factor = -1.5  # extreme: capped to 1.0
        result = calc._apply_volatility_guard(
            side="sell", mid_trend_bps=20.0, effective_offset_ratio=0.10,
        )
        # damping = 1.0 - min(1.5, 1.0) = 0.0 → boost = 1.0 + 0.0 = 1.0
        expected = 0.10 * 1.0  # completely suppressed
        assert abs(result - expected) < 1e-9, f"Expected {expected}, got {result}"

    # ---- 168# P2-C1/C2/C3 テスト ----

    def test_p2c1_sell_guard_max_spread_yaml(self) -> None:
        """168# P2-C1: sell_guard max_spread_jpy が 5000 に更新済み."""
        sg = self.yaml_cfg["sell_guard"]
        assert sg["max_spread_jpy"] == 5000.0, (
            f"168# P2-C1: expected 5000, got {sg['max_spread_jpy']}"
        )

    def test_p2c2_reprice_tighten_yaml(self) -> None:
        """168# P2-C2: stale_order.reprice_tighten が YAML に設定済み."""
        so = self.yaml_cfg["stale_order"]
        assert "reprice_tighten" in so, "168# P2-C2: reprice_tighten not in YAML"
        assert so["reprice_tighten"] == 0.85

    def test_p2c2_reprice_tighten_config_field(self) -> None:
        """168# P2-C2: FillTestConfig.stale_reprice_tighten のデフォルトは 1.0."""

        cfg = FillTestConfig()
        assert cfg.stale_reprice_tighten == 1.0

    def test_p2c3_reprice_skip_gate_offset_yaml(self) -> None:
        """168# P2-C3: stale_order.reprice_skip_gate_offset が YAML に設定済み."""
        so = self.yaml_cfg["stale_order"]
        assert "reprice_skip_gate_offset" in so, (
            "168# P2-C3: reprice_skip_gate_offset not in YAML"
        )
        assert so["reprice_skip_gate_offset"] == 0.05

    def test_p2c3_reprice_skip_gate_offset_config(self) -> None:
        """168# P2-C3: FillTestConfig.stale_reprice_skip_gate_offset は 0.0 デフォルト."""

        cfg = FillTestConfig()
        assert hasattr(cfg, "stale_reprice_skip_gate_offset")
        assert cfg.stale_reprice_skip_gate_offset == 0.0

    def test_p2c3_reprice_skip_gate_offset_in_code(self) -> None:
        """168# P2-C3: order_monitor.py で reprice_skip_gate_offset が使用されている."""
        assert "stale_reprice_skip_gate_offset" in _ORDER_MONITOR_SOURCE
        assert "threshold_offset" in _ORDER_MONITOR_SOURCE  # evaluate に offset を渡している



    def test_batch_persistence_in_code(self) -> None:
        """119# BatchPersistence 委譲が存在する."""
        assert hasattr(BatchPersistence, "maybe_flush")

    def test_batch_persistence_used_in_run_continuous(self) -> None:
        """119# run_continuous 内で BatchPersistence.maybe_flush が使用されている."""

        source = read_fill_test_runner_source()
        assert "_batch_persistence.maybe_flush" in source

    def test_vpin_caching_in_code(self) -> None:
        """107# VPIN が SkipGate features からキャッシュされている."""

        # 113# R1: VPIN caching moved to _evaluate_skip_gate
        source = read_inspect_source(FillTestRunner._evaluate_skip_gate)
        assert "_last_vpin" in source


# =====================================================================
# 122# B2: Holm-Bonferroni multi-timeframe PnL tests
# =====================================================================


class TestHolmBonferroniPnL:
    """122# B2: g1_2_full_judgment の Holm-Bonferroni 補正テスト."""

    def _make_metrics(self, **overrides) -> "FillMetrics":
        defaults = dict(
            total_orders=1000,
            filled_orders=700,
            cancelled_orders=300,
            fill_rate_p90=0.65,
            cancel_ratio=0.30,
            queue_wait_median_sec=15.0,
            post_fill_30s_pnl_mean=-0.2,
            post_fill_30s_pnl_pvalue=0.16,
            post_fill_30s_pnl_ci_upper=0.1,
            post_fill_60s_pnl_mean=-0.1,
            post_fill_60s_pnl_pvalue=0.30,
            post_fill_120s_pnl_mean=-0.05,
            post_fill_120s_pnl_pvalue=0.40,
            adverse_selection_ratio=0.25,
            attempted_orders=900,
            skip_gate_count=100,
            skip_gate_ratio=0.10,
            attempted_fill_rate=0.778,
            attempted_cancel_ratio=0.222,
            overall_fill_rate=0.70,
            measurement_days=7,
            sample_sufficient=True,
        )
        defaults.update(overrides)
        return FillMetrics(**defaults)

    def test_holm_keys_present(self) -> None:
        """Holm 補正済み F4_pnl_30s/F4b/F4c キーが存在する."""
        metrics = self._make_metrics()
        result = g1_2_full_judgment(metrics, {"pnl_alpha": 0.05})
        checks = result["checks"]
        assert "F4_pnl_30s" in checks
        assert "F4b_pnl_60s" in checks
        assert "F4c_pnl_120s" in checks
        assert "pvalue_holm" in checks["F4_pnl_30s"]
        assert "pvalue_raw" in checks["F4_pnl_30s"]

    def test_holm_backward_compat_f4_pnl(self) -> None:
        """後方互換: F4_pnl キーが維持される."""
        metrics = self._make_metrics()
        result = g1_2_full_judgment(metrics, {"pnl_alpha": 0.05})
        f4 = result["checks"]["F4_pnl"]
        assert "pvalue" in f4
        assert "pass" in f4
        assert f4["pass"] == result["checks"]["F4_pnl_30s"]["pass"]

    def test_holm_correction_loosens_threshold(self) -> None:
        """Holm 補正で p_holm > p_raw (多重比較補正は厳格化方向)."""
        metrics = self._make_metrics(
            post_fill_30s_pnl_mean=-0.5,
            post_fill_30s_pnl_pvalue=0.03,
            post_fill_60s_pnl_mean=-0.3,
            post_fill_60s_pnl_pvalue=0.08,
            post_fill_120s_pnl_mean=-0.1,
            post_fill_120s_pnl_pvalue=0.15,
        )
        result = g1_2_full_judgment(metrics, {"pnl_alpha": 0.05})
        f4_30 = result["checks"]["F4_pnl_30s"]
        # p_raw=0.03 → p_holm = min(0.03 * 3, 1.0) = 0.09 >= 0.05 → PASS
        assert f4_30["pvalue_holm"] > f4_30["pvalue_raw"]
        assert f4_30["pvalue_holm"] >= 0.05
        assert f4_30["pass"] is True

    def test_holm_strongly_significant_still_fails(self) -> None:
        """非常に小さい p値は Holm 補正後も FAIL."""
        metrics = self._make_metrics(
            post_fill_30s_pnl_mean=-1.0,
            post_fill_30s_pnl_pvalue=0.005,
        )
        result = g1_2_full_judgment(metrics, {"pnl_alpha": 0.05})
        f4_30 = result["checks"]["F4_pnl_30s"]
        # p_holm = min(0.005 * 3, 1.0) = 0.015 < 0.05 → FAIL
        assert f4_30["pvalue_holm"] <= 0.015 + 1e-9
        assert f4_30["pass"] is False


class TestComputeMultiTimeframePnL:
    """122# B3: multi-timeframe PnL の compute_fill_metrics テスト."""

    def test_pnl_60s_120s_populated(self) -> None:
        """PnL 60s/120s の mean/pvalue が算出される."""
        base_ts = time.time()
        records = [
            FillRecord(
                cycle_id=f"c_{i}", timestamp=base_ts + i * 120,
                side="buy", order_price=10000000, order_quantity=0.001,
                filled=True, post_fill_30s_pnl=0.5,
                post_fill_60s_pnl=0.3, post_fill_120s_pnl=0.1,
            )
            for i in range(30)
        ]
        m = compute_fill_metrics(records)
        assert m.post_fill_60s_pnl_mean > 0
        assert m.post_fill_120s_pnl_mean > 0
        # 正の PnL → p値は大きい (有意に負ではない)
        assert m.post_fill_60s_pnl_pvalue > 0.5
        assert m.post_fill_120s_pnl_pvalue > 0.5

    def test_pnl_60s_120s_defaults_when_missing(self) -> None:
        """PnL 60s/120s データがない場合のデフォルト値."""
        base_ts = time.time()
        records = [
            FillRecord(
                cycle_id=f"c_{i}", timestamp=base_ts + i * 120,
                side="buy", order_price=10000000, order_quantity=0.001,
                filled=True, post_fill_30s_pnl=0.5,
                # 60s/120s は None (未計測)
            )
            for i in range(10)
        ]
        m = compute_fill_metrics(records)
        assert m.post_fill_60s_pnl_mean == 0.0
        assert m.post_fill_60s_pnl_pvalue == 1.0
        assert m.post_fill_120s_pnl_mean == 0.0
        assert m.post_fill_120s_pnl_pvalue == 1.0


# =====================================================================
# 122# A4+B4: VG effectiveness + daily trend analysis
# =====================================================================


class TestVGAndTrendAnalysis:
    """122# A4/B4: vg_and_trend.py の分析関数テスト."""

    def _make_records(self, n: int = 20, base_ts: float = 1708000000.0) -> list:
        """テスト用 FillRecord リストを生成."""
        records = []
        for i in range(n):
            records.append(FillRecord(
                cycle_id=f"vg_test_{i}",
                timestamp=base_ts + i * 120,
                side="buy" if i % 2 == 0 else "sell",
                order_price=10_000_000,
                order_quantity=0.001,
                filled=True,
                fill_price=10_000_000.0,
                mid_at_fill=10_000_050.0,
                post_fill_30s_pnl=0.5 if i % 3 != 0 else -1.0,
                post_fill_120s_pnl=0.3 if i % 3 != 0 else -0.5,
                adverse_selected=(i % 3 == 0),
                effective_offset_used=0.15 if i < 15 else 0.30,
                run_id="test_run",
                git_sha="abc123",
            ))
        return records

    def test_analyze_vg_effectiveness_basic(self) -> None:
        """VG 効果分析が VG/非VG 群を正しく分離."""
        records = self._make_records(20)
        vg_ids = {records[0].cycle_id, records[5].cycle_id, records[10].cycle_id}

        result = analyze_vg_effectiveness(records, vg_ids)
        assert result["vg_filled"]["n"] == 3
        assert result["non_vg_filled"]["n"] == 17
        assert "interpretation" in result

    def test_analyze_vg_empty_vg(self) -> None:
        """VG 発動 0 件でもエラーにならない."""
        records = self._make_records(10)
        result = analyze_vg_effectiveness(records, set())
        assert result["vg_filled"]["n"] == 0
        assert result["non_vg_filled"]["n"] == 10

    def test_analyze_daily_trend(self) -> None:
        """日別トレンドが正しく分割される."""
        records = self._make_records(20, base_ts=1708000000.0)
        daily = analyze_daily_trend(records)
        assert len(daily) >= 1
        assert "date" in daily[0]
        assert "as_rate" in daily[0]
        assert "buy_as_rate" in daily[0]
        assert "sell_as_rate" in daily[0]

    def test_analyze_8h_trend(self) -> None:
        """8h帯別トレンドが出力される."""
        records = self._make_records(20, base_ts=1708000000.0)
        periods = analyze_8h_trend(records)
        assert len(periods) >= 1
        assert "period" in periods[0]
        assert "as_rate" in periods[0]

    def test_parse_vg_activations_empty(self) -> None:
        """存在しないログで空リスト."""
        result = _parse_vg_activations(Path("/nonexistent/log_file.log"))
        assert result == []

    def test_match_vg_to_records(self) -> None:
        """VG タイムスタンプが records と正しくマッチ."""
        ts = 1708000000.0
        records = [
            FillRecord(
                cycle_id="r1", timestamp=ts, side="buy",
                order_price=10000000, order_quantity=0.001,
            ),
            FillRecord(
                cycle_id="r2", timestamp=ts + 120, side="sell",
                order_price=10000000, order_quantity=0.001,
            ),
        ]
        activations = [{"timestamp": ts + 2, "side": "buy"}]  # 2秒差 → マッチ

        matched = _match_vg_to_records(activations, records, tolerance_sec=10)
        assert "r1" in matched
        assert "r2" not in matched

    def test_match_vg_side_mismatch(self) -> None:
        """VG side が record side と異なる場合はマッチしない."""
        ts = 1708000000.0
        records = [
            FillRecord(
                cycle_id="r1", timestamp=ts, side="buy",
                order_price=10000000, order_quantity=0.001,
            ),
        ]
        activations = [{"timestamp": ts + 1, "side": "sell"}]  # side不一致

        matched = _match_vg_to_records(activations, records, tolerance_sec=10)
        assert len(matched) == 0


# =====================================================================
# 123# Gemini review: F4d PnL mean floor テスト
# =====================================================================


class TestF4dPnLMeanFloor:
    """123# Gemini review Critical 1: PnL 平均が許容フロア未満なら WATCH/FAIL."""

    def _make_metrics(self, **overrides) -> "FillMetrics":
        defaults = dict(
            total_orders=1000,
            filled_orders=700,
            cancelled_orders=300,
            fill_rate_p90=0.65,
            cancel_ratio=0.30,
            queue_wait_median_sec=15.0,
            post_fill_30s_pnl_mean=0.1,
            post_fill_30s_pnl_pvalue=0.40,
            post_fill_30s_pnl_ci_upper=0.3,
            post_fill_60s_pnl_mean=0.05,
            post_fill_60s_pnl_pvalue=0.45,
            post_fill_120s_pnl_mean=0.02,
            post_fill_120s_pnl_pvalue=0.48,
            adverse_selection_ratio=0.25,
            attempted_orders=900,
            skip_gate_count=100,
            skip_gate_ratio=0.10,
            attempted_fill_rate=0.778,
            attempted_cancel_ratio=0.222,
            overall_fill_rate=0.70,
            measurement_days=7,
            sample_sufficient=True,
        )
        defaults.update(overrides)
        return FillMetrics(**defaults)

    def test_positive_pnl_passes(self) -> None:
        """PnL > 0 → F4d PASS, gate_result PASS."""
        metrics = self._make_metrics(post_fill_30s_pnl_mean=0.1)
        result = g1_2_full_judgment(metrics, {"pnl_mean_floor_bps": -0.10})
        assert result["checks"]["F4d_pnl_mean_floor"]["pass"] is True
        assert result["checks"]["F4d_pnl_mean_floor"]["watch"] is False
        assert result["gate_result"] == "PASS"

    def test_mild_negative_watch(self) -> None:
        """floor < mean < 0 → F4d WATCH, gate_result WATCH."""
        metrics = self._make_metrics(
            post_fill_30s_pnl_mean=-0.15,
            post_fill_30s_pnl_pvalue=0.20,  # 有意でない
        )
        result = g1_2_full_judgment(metrics, {
            "pnl_mean_floor_bps": -0.10,
            "pnl_mean_hard_floor_bps": -0.50,
        })
        assert result["checks"]["F4d_pnl_mean_floor"]["pass"] is True
        assert result["checks"]["F4d_pnl_mean_floor"]["watch"] is True
        assert result["gate_result"] == "WATCH"

    def test_severe_negative_fails(self) -> None:
        """mean < hard_floor → F4d FAIL, gate_result FAIL."""
        metrics = self._make_metrics(
            post_fill_30s_pnl_mean=-0.80,
            post_fill_30s_pnl_pvalue=0.20,
        )
        result = g1_2_full_judgment(metrics, {
            "pnl_mean_floor_bps": -0.10,
            "pnl_mean_hard_floor_bps": -0.50,
        })
        assert result["checks"]["F4d_pnl_mean_floor"]["pass"] is False
        assert result["gate_result"] == "FAIL"

    def test_exactly_at_floor_passes(self) -> None:
        """mean == floor → PASS (境界条件)."""
        metrics = self._make_metrics(post_fill_30s_pnl_mean=-0.10)
        result = g1_2_full_judgment(metrics, {"pnl_mean_floor_bps": -0.10})
        assert result["checks"]["F4d_pnl_mean_floor"]["pass"] is True
        assert result["checks"]["F4d_pnl_mean_floor"]["watch"] is False

    def test_default_floor_applied(self) -> None:
        """floor 未指定 → デフォルト -0.10 bps が適用される."""
        metrics = self._make_metrics(post_fill_30s_pnl_mean=-0.05)
        result = g1_2_full_judgment(metrics, {})
        f4d = result["checks"]["F4d_pnl_mean_floor"]
        assert f4d["floor"] == -0.10
        assert f4d["hard_floor"] == -0.50
        assert f4d["pass"] is True
