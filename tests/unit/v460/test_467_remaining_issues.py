"""467# テスト: config_hash, hour_ceiling_mult, status_unknown_fast連続検知, hour-matched comparison."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.fill_config_parser import parse_fill_config_yaml
from scripts.v460.lib.hour_rules import resolve_hour_float, resolve_optional_hour_float
from ztb.metrics.fill_quality import FillRecord


# ══════════════════════════════════════════════════════════════════════
# 1. FillRecord config_hash フィールド
# ══════════════════════════════════════════════════════════════════════


class TestFillRecordConfigHash:
    """462# 残課題: FillRecord に config_hash が追加されたことの検証."""

    def test_config_hash_default_none(self) -> None:
        """デフォルト値は None."""
        rec = FillRecord(
            cycle_id="c1", timestamp=time.time(), side="buy",
            order_price=1500000, order_quantity=0.001,
        )
        assert rec.config_hash is None

    def test_config_hash_setter(self) -> None:
        """config_hash を設定可能."""
        rec = FillRecord(
            cycle_id="c2", timestamp=time.time(), side="buy",
            order_price=1500000, order_quantity=0.001,
            config_hash="abc123def456",
        )
        assert rec.config_hash == "abc123def456"

    def test_config_hash_from_dict(self) -> None:
        """from_dict で config_hash を復元."""
        d = {
            "side": "sell",
            "timestamp": time.time(),
            "cycle_id": "c3",
            "order_price": 1500000,
            "order_quantity": 0.001,
            "config_hash": "x1y2z3",
        }
        rec = FillRecord.from_dict(d)
        assert rec.config_hash == "x1y2z3"

    def test_config_hash_from_dict_missing(self) -> None:
        """from_dict で config_hash がない場合は None."""
        d = {
            "side": "buy",
            "timestamp": time.time(),
            "cycle_id": "c4",
            "order_price": 1500000,
            "order_quantity": 0.001,
        }
        rec = FillRecord.from_dict(d)
        assert rec.config_hash is None


# ══════════════════════════════════════════════════════════════════════
# 2. hour_ceiling_mult: 時間帯別 ceiling 乗数
# ══════════════════════════════════════════════════════════════════════


class TestHourCeilingMult:
    """461# P0 deep-night ceiling 緩和機構テスト."""

    def test_default_empty(self) -> None:
        """デフォルトは空 dict."""
        cfg = FillTestConfig()
        assert cfg.hour_ceiling_mult == {}

    def test_resolve_offset_ceiling_no_hour(self) -> None:
        """utc_hour 指定なし → 既存動作と同一."""
        cfg = FillTestConfig(
            offset_ceiling_ratio=0.15,
            hour_ceiling_mult={13: 2.0},
        )
        assert cfg.resolve_offset_ceiling("buy") == 0.15

    def test_resolve_offset_ceiling_with_hour_match(self) -> None:
        """utc_hour 指定あり＋マッチ → ceiling × mult."""
        cfg = FillTestConfig(
            offset_ceiling_ratio=0.15,
            hour_ceiling_mult={13: 2.0, 14: 1.5},
        )
        assert cfg.resolve_offset_ceiling("buy", utc_hour=13) == pytest.approx(0.30)
        assert cfg.resolve_offset_ceiling("sell", utc_hour=14) == pytest.approx(0.225)

    def test_resolve_offset_ceiling_with_hour_no_match(self) -> None:
        """utc_hour 指定あり＋マッチなし → 基本値をそのまま返す."""
        cfg = FillTestConfig(
            offset_ceiling_ratio=0.15,
            hour_ceiling_mult={13: 2.0},
        )
        assert cfg.resolve_offset_ceiling("buy", utc_hour=10) == 0.15

    def test_resolve_offset_ceiling_side_specific_with_hour(self) -> None:
        """サイド別 ceiling + hour_ceiling_mult の組み合わせ."""
        cfg = FillTestConfig(
            offset_ceiling_ratio=0.15,
            offset_ceiling_ratio_sell=0.30,
            hour_ceiling_mult={21: 1.5},
        )
        # buy → 0.15 × 1.5 = 0.225
        assert cfg.resolve_offset_ceiling("buy", utc_hour=21) == pytest.approx(0.225)
        # sell → 0.30 × 1.5 = 0.45
        assert cfg.resolve_offset_ceiling("sell", utc_hour=21) == pytest.approx(0.45)

    def test_resolve_offset_ceiling_disabled(self) -> None:
        """ceiling=0.0 (無効) のときは hour_ceiling_mult も効かない."""
        cfg = FillTestConfig(
            offset_ceiling_ratio=0.0,
            hour_ceiling_mult={13: 2.0},
        )
        assert cfg.resolve_offset_ceiling("buy", utc_hour=13) == 0.0

    def test_resolve_offset_ceiling_backward_compat(self) -> None:
        """hour_ceiling_mult 空 + utc_hour 指定 → 基本値."""
        cfg = FillTestConfig(offset_ceiling_ratio=0.15)
        assert cfg.resolve_offset_ceiling("buy", utc_hour=13) == 0.15

    def test_hot_reloadable(self) -> None:
        """hour_ceiling_mult が HOT_RELOADABLE_FIELDS に含まれている."""
        from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS
        assert "hour_ceiling_mult" in _HOT_RELOADABLE_FIELDS

    # ---- 568# eDRC A/B Toggle Tests ----

    def test_resolve_offset_ceiling_edrc_disabled(self) -> None:
        """experimental_additive_pipeline=False → 従来ロジック."""
        cfg = FillTestConfig(
            offset_ceiling_ratio=0.15,
            experimental_additive_pipeline=False,
            edrc_c_base=0.40,
        )
        assert cfg.resolve_offset_ceiling("buy") == 0.15

    def test_resolve_offset_ceiling_edrc_enabled_zero_inputs(self) -> None:
        """eDRC 有効 + sigma=0, ofi=0 → c_base そのまま."""
        cfg = FillTestConfig(
            experimental_additive_pipeline=True,
            edrc_c_base=0.40,
            edrc_alpha=0.5,
            edrc_beta=0.3,
        )
        # exp(0) = 1.0 → 0.40
        assert cfg.resolve_offset_ceiling("buy", sigma=0.0, adverse_ofi=0.0) == pytest.approx(0.40)

    def test_resolve_offset_ceiling_edrc_with_sigma(self) -> None:
        """eDRC 有効 + sigma > 0 → ceiling 拡大."""
        from math import exp
        cfg = FillTestConfig(
            experimental_additive_pipeline=True,
            edrc_c_base=0.40,
            edrc_alpha=1.0,
            edrc_beta=0.0,
        )
        expected = 0.40 * exp(1.0 * 0.5)
        assert cfg.resolve_offset_ceiling("buy", sigma=0.5) == pytest.approx(expected)

    def test_resolve_offset_ceiling_edrc_with_hour(self) -> None:
        """eDRC 有効 + hour_ceiling_mult の組み合わせ."""
        cfg = FillTestConfig(
            experimental_additive_pipeline=True,
            edrc_c_base=0.40,
            edrc_alpha=0.0,
            edrc_beta=0.0,
            hour_ceiling_mult={13: 1.5},
        )
        # exp(0) = 1.0 → 0.40 × 1.5 = 0.60
        assert cfg.resolve_offset_ceiling("buy", utc_hour=13) == pytest.approx(0.60)

    def test_resolve_offset_ceiling_edrc_exponent_clip(self) -> None:
        """589# eDRC 指数クリップ: 極端な入力で exp() が発散しない."""
        from math import exp
        cfg = FillTestConfig(
            experimental_additive_pipeline=True,
            edrc_c_base=0.40,
            edrc_alpha=1.0,
            edrc_beta=0.1,
            edrc_hard_cap=1.0,
        )
        # sigma=100, ofi=500 → exponent=100+50=150 → clipped to 10
        result = cfg.resolve_offset_ceiling("buy", sigma=100.0, adverse_ofi=500.0)
        expected = min(0.40 * exp(10.0), 1.0)  # hard_cap
        assert result == pytest.approx(expected)
        assert result <= 1.0

    def test_resolve_offset_ceiling_edrc_exponent_no_clip(self) -> None:
        """589# 指数が閾値以下なら通常計算."""
        from math import exp
        cfg = FillTestConfig(
            experimental_additive_pipeline=True,
            edrc_c_base=0.40,
            edrc_alpha=0.020,
            edrc_beta=0.40,
        )
        # exponent = 0.020*15 + 0.40*0.6 = 0.54 < 10 → クリップなし
        result = cfg.resolve_offset_ceiling("buy", sigma=15.0, adverse_ofi=0.6)
        expected = 0.40 * exp(0.020 * 15.0 + 0.40 * 0.6)
        assert result == pytest.approx(expected)

    def test_parse_warns_when_execution_flag_mismatches_logic_flag(self) -> None:
        """598#: telemetry flag と logic flag の不一致は warning で明示."""
        yaml_cfg = {
            "execution_additive_enabled": False,
            "experimental_additive_pipeline": {
                "enabled": True,
                "alpha": 0.2,
                "beta": 0.1,
                "c_base": 0.4,
                "hard_cap": 1.0,
            },
        }
        with patch("scripts.v460.lib.fill_config_parser.logger.warning") as mock_warning:
            cfg = parse_fill_config_yaml(yaml_cfg)
        assert cfg.execution_additive_enabled is False
        assert cfg.experimental_additive_pipeline is True
        mock_warning.assert_called_once()
        assert "experimental_additive_pipeline" in str(mock_warning.call_args[0][0])

    def test_resolve_offset_ceiling_edrc_input_clip(self) -> None:
        """589# eDRC 入力クリップ: 極端な sigma/ofi が exp() を発散させない."""
        from math import exp
        cfg = FillTestConfig(
            experimental_additive_pipeline=True,
            edrc_c_base=0.40,
            edrc_alpha=1.0,
            edrc_beta=0.1,
            edrc_hard_cap=1.0,
        )
        # sigma=100 → clipped to 5.0, ofi=500 → clipped to 50.0
        result = cfg.resolve_offset_ceiling("buy", sigma=100.0, adverse_ofi=500.0)
        expected = min(0.40 * exp(1.0 * 5.0 + 0.1 * 50.0), 1.0)
        assert result == pytest.approx(expected)
        # 結果は hard_cap (1.0) で制限される
        assert result <= 1.0


# ══════════════════════════════════════════════════════════════════════
# 3. status_unknown_fast 連続検知
# ══════════════════════════════════════════════════════════════════════


class _MockMonitor(NamedTuple):
    """_FillMonitorResult の軽量模倣."""

    filled: bool = False
    cancel_reason: str | None = None
    order_id_for_reconciliation: str | None = None
    fill_price: float | None = None
    queue_wait: float = 0.0
    reprice_count: int = 0
    reprice_drift_bps: float | None = None
    final_order_price: float = 0.0
    effective_timeout: float | None = None
    cancel_failed_likely_filled: bool = False


class TestConsecutiveStatusUnknownFast:
    """461# P0: status_unknown_fast 連続検知テスト."""

    def _make_executor(self) -> MagicMock:
        """FillCycleExecutorMixin の必要最小属性をモック."""
        from scripts.v460.lib.fill_cycle_executor import FillCycleExecutorMixin
        executor = MagicMock(spec=FillCycleExecutorMixin)
        executor._consecutive_status_unknown_fast = 0
        executor._phantom_guard = None
        executor._balance_checker = MagicMock()
        executor._balance_checker.last_btc_free = 0.001
        executor._balance_checker.last_jpy_free = 100000.0
        # 実メソッドをバインド
        executor._maybe_register_phantom = (
            FillCycleExecutorMixin._maybe_register_phantom.__get__(executor)
        )
        return executor

    def test_counter_increments_on_status_unknown_fast(self) -> None:
        """status_unknown_fast で連続カウンタが増加."""
        ex = self._make_executor()
        monitor = _MockMonitor(
            cancel_reason="status_unknown_fast",
            order_id_for_reconciliation="ord123",
        )
        ex._maybe_register_phantom(monitor, "buy", 0.001, 15000000)
        assert ex._consecutive_status_unknown_fast == 1

    def test_counter_resets_on_filled(self) -> None:
        """fill 成功で連続カウンタがリセット."""
        ex = self._make_executor()
        ex._consecutive_status_unknown_fast = 2
        monitor = _MockMonitor(filled=True, fill_price=15000000)
        ex._maybe_register_phantom(monitor, "buy", 0.001, 15000000)
        assert ex._consecutive_status_unknown_fast == 0

    def test_counter_resets_on_timeout(self) -> None:
        """timeout/cancel でカウンタリセット."""
        ex = self._make_executor()
        ex._consecutive_status_unknown_fast = 2
        monitor = _MockMonitor(cancel_reason="timeout")
        ex._maybe_register_phantom(monitor, "buy", 0.001, 15000000)
        assert ex._consecutive_status_unknown_fast == 0

    def test_counter_resets_on_status_unknown_slow(self) -> None:
        """status_unknown (non-fast) でカウンタリセット."""
        ex = self._make_executor()
        ex._consecutive_status_unknown_fast = 2
        monitor = _MockMonitor(
            cancel_reason="status_unknown",
            order_id_for_reconciliation="ord456",
        )
        ex._maybe_register_phantom(monitor, "buy", 0.001, 15000000)
        assert ex._consecutive_status_unknown_fast == 0

    def test_warning_at_threshold(self, caplog: pytest.LogCaptureFixture) -> None:
        """3連続で WARNING ログ出力."""
        import logging
        ex = self._make_executor()
        for i in range(3):
            monitor = _MockMonitor(
                cancel_reason="status_unknown_fast",
                order_id_for_reconciliation=f"ord{i}",
            )
            with caplog.at_level(logging.WARNING):
                ex._maybe_register_phantom(monitor, "buy", 0.001, 15000000)
        assert ex._consecutive_status_unknown_fast == 3
        assert "consecutive status_unknown_fast" in caplog.text


# ══════════════════════════════════════════════════════════════════════
# 4. hour_matched_comparison ユーティリティ
# ══════════════════════════════════════════════════════════════════════


class TestHourMatchedComparison:
    """462# hour-matched comparison ユーティリティテスト."""

    def _make_record(
        self, sha: str, utc_hour: int, *, filled: bool = True, pnl: float = 1.0,
        side: str = "buy", as_flag: bool = False,
    ) -> dict[str, object]:
        """テスト用 fill_record dict を生成."""
        from datetime import datetime, timezone
        # UTC hour を反映した timestamp
        dt = datetime(2026, 3, 20, utc_hour, 30, 0, tzinfo=timezone.utc)
        return {
            "git_sha": sha,
            "side": side,
            "start_ts": dt.timestamp(),
            "cycle_id": f"c_{sha}_{utc_hour}_{id(self)}_{pnl}",
            "filled": filled,
            "post_fill_30s_pnl": pnl if filled else None,
            "adverse_selected": as_flag,
        }

    def test_basic_comparison(self) -> None:
        """基本: 同一 hour で 2 SHA を比較."""
        from scripts.v460.analysis.hour_matched_comparison import (
            run_hour_matched_comparison,
            _compute_bucket,
        )
        recs_a = [self._make_record("aaa1111", 10, pnl=2.0)]
        recs_b = [self._make_record("bbb2222", 10, pnl=4.0)]
        m_a = _compute_bucket(recs_a, "aaa1111", 10)
        m_b = _compute_bucket(recs_b, "bbb2222", 10)
        assert m_a["avg_pnl_bps"] == pytest.approx(2.0)
        assert m_b["avg_pnl_bps"] == pytest.approx(4.0)

    def test_compute_bucket_no_fill(self) -> None:
        """全 unfilled → fill_rate=0, as_rate=0."""
        from scripts.v460.analysis.hour_matched_comparison import _compute_bucket
        recs = [self._make_record("aaa", 5, filled=False)]
        m = _compute_bucket(recs, "aaa", 5)
        assert m["fill_rate"] == 0.0
        assert m["as_rate"] == 0.0

    def test_as_rate_calc(self) -> None:
        """AS 率計算."""
        from scripts.v460.analysis.hour_matched_comparison import _compute_bucket
        recs = [
            self._make_record("aaa", 5, as_flag=True, pnl=-5.0),
            self._make_record("aaa", 5, as_flag=False, pnl=3.0),
        ]
        m = _compute_bucket(recs, "aaa", 5)
        assert m["as_rate"] == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════════════════
# 5. config_hash 計算
# ══════════════════════════════════════════════════════════════════════


class TestComputeConfigHash:
    """manifest.compute_config_hash の基本動作."""

    def test_deterministic(self) -> None:
        from scripts.v460.lib.manifest import compute_config_hash
        cfg = {"a": 1, "b": "hello"}
        h1 = compute_config_hash(cfg)
        h2 = compute_config_hash(cfg)
        assert h1 == h2
        assert len(h1) == 16

    def test_order_independent(self) -> None:
        from scripts.v460.lib.manifest import compute_config_hash
        h1 = compute_config_hash({"a": 1, "b": 2})
        h2 = compute_config_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_config_different_hash(self) -> None:
        from scripts.v460.lib.manifest import compute_config_hash
        h1 = compute_config_hash({"a": 1})
        h2 = compute_config_hash({"a": 2})
        assert h1 != h2
