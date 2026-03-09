"""365# P3/P4/P5 テスト: SAC Sidecar types + signal I/O + cycle_gate injection.

カバレッジ対象:
  - sidecar_types.py: SidecarSignal, classify_bias, compute_sidecar_offset_bps
  - sidecar_signal_io.py: write/read atomic I/O, staleness, error handling
  - cycle_gate_aggregator.py: sidecar offset injection
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest


# ════════════════════════════════════════════════════════════════
# §1 sidecar_types テスト
# ════════════════════════════════════════════════════════════════


class TestSidecarDirection:
    """SidecarDirection enum."""

    def test_values(self) -> None:
        from scripts.v460.lib.sidecar_types import SidecarDirection

        assert SidecarDirection.BUY_BIAS == 1
        assert SidecarDirection.NEUTRAL == 0
        assert SidecarDirection.SELL_BIAS == -1


class TestClassifyBias:
    """classify_bias() — 閾値ベース分類."""

    def test_buy_bias(self) -> None:
        from scripts.v460.lib.sidecar_types import classify_bias, SidecarDirection

        assert classify_bias(0.5) == SidecarDirection.BUY_BIAS
        assert classify_bias(1.0) == SidecarDirection.BUY_BIAS

    def test_sell_bias(self) -> None:
        from scripts.v460.lib.sidecar_types import classify_bias, SidecarDirection

        assert classify_bias(-0.5) == SidecarDirection.SELL_BIAS
        assert classify_bias(-1.0) == SidecarDirection.SELL_BIAS

    def test_neutral(self) -> None:
        from scripts.v460.lib.sidecar_types import classify_bias, SidecarDirection

        assert classify_bias(0.0) == SidecarDirection.NEUTRAL
        assert classify_bias(0.3) == SidecarDirection.NEUTRAL
        assert classify_bias(-0.3) == SidecarDirection.NEUTRAL
        assert classify_bias(0.29) == SidecarDirection.NEUTRAL

    def test_boundary_above(self) -> None:
        from scripts.v460.lib.sidecar_types import classify_bias, SidecarDirection

        assert classify_bias(0.31) == SidecarDirection.BUY_BIAS
        assert classify_bias(-0.31) == SidecarDirection.SELL_BIAS

    def test_custom_threshold(self) -> None:
        from scripts.v460.lib.sidecar_types import classify_bias, SidecarDirection

        assert classify_bias(0.15, threshold=0.1) == SidecarDirection.BUY_BIAS
        assert classify_bias(0.15, threshold=0.2) == SidecarDirection.NEUTRAL


class TestSidecarSignal:
    """SidecarSignal dataclass."""

    def test_creation(self) -> None:
        from scripts.v460.lib.sidecar_types import SidecarSignal

        sig = SidecarSignal(timestamp="2026-03-10T12:00:00+09:00", directional_bias=0.42)
        assert sig.directional_bias == 0.42
        assert sig.confidence == 1.0
        assert sig.model_version == ""

    def test_direction_property(self) -> None:
        from scripts.v460.lib.sidecar_types import SidecarSignal, SidecarDirection

        sig = SidecarSignal(timestamp="t", directional_bias=0.5)
        assert sig.direction == SidecarDirection.BUY_BIAS

        sig2 = SidecarSignal(timestamp="t", directional_bias=-0.5)
        assert sig2.direction == SidecarDirection.SELL_BIAS

        sig3 = SidecarSignal(timestamp="t", directional_bias=0.0)
        assert sig3.direction == SidecarDirection.NEUTRAL

    def test_bias_validation_out_of_range(self) -> None:
        from scripts.v460.lib.sidecar_types import SidecarSignal

        with pytest.raises(ValueError, match="directional_bias"):
            SidecarSignal(timestamp="t", directional_bias=1.5)

        with pytest.raises(ValueError, match="directional_bias"):
            SidecarSignal(timestamp="t", directional_bias=-1.1)

    def test_confidence_validation(self) -> None:
        from scripts.v460.lib.sidecar_types import SidecarSignal

        with pytest.raises(ValueError, match="confidence"):
            SidecarSignal(timestamp="t", directional_bias=0.0, confidence=-0.1)
        with pytest.raises(ValueError, match="confidence"):
            SidecarSignal(timestamp="t", directional_bias=0.0, confidence=1.5)

    def test_frozen(self) -> None:
        from scripts.v460.lib.sidecar_types import SidecarSignal

        sig = SidecarSignal(timestamp="t", directional_bias=0.0)
        with pytest.raises(AttributeError):
            sig.directional_bias = 0.5  # type: ignore[misc]


class TestComputeSidecarOffsetBps:
    """compute_sidecar_offset_bps() — 非対称 offset 計算."""

    def test_buy_bias_buy_side(self) -> None:
        from scripts.v460.lib.sidecar_types import compute_sidecar_offset_bps

        offset = compute_sidecar_offset_bps(0.5, "buy")
        assert offset > 0  # 攻撃的

    def test_buy_bias_sell_side(self) -> None:
        from scripts.v460.lib.sidecar_types import compute_sidecar_offset_bps

        offset = compute_sidecar_offset_bps(0.5, "sell")
        assert offset < 0  # 保守的

    def test_sell_bias_sell_side(self) -> None:
        from scripts.v460.lib.sidecar_types import compute_sidecar_offset_bps

        offset = compute_sidecar_offset_bps(-0.5, "sell")
        assert offset > 0  # 攻撃的

    def test_sell_bias_buy_side(self) -> None:
        from scripts.v460.lib.sidecar_types import compute_sidecar_offset_bps

        offset = compute_sidecar_offset_bps(-0.5, "buy")
        assert offset < 0  # 保守的

    def test_neutral_returns_zero(self) -> None:
        from scripts.v460.lib.sidecar_types import compute_sidecar_offset_bps

        assert compute_sidecar_offset_bps(0.0, "buy") == 0.0
        assert compute_sidecar_offset_bps(0.0, "sell") == 0.0
        assert compute_sidecar_offset_bps(0.2, "buy") == 0.0

    def test_confidence_scaling(self) -> None:
        from scripts.v460.lib.sidecar_types import compute_sidecar_offset_bps

        full = compute_sidecar_offset_bps(0.5, "buy", confidence=1.0)
        half = compute_sidecar_offset_bps(0.5, "buy", confidence=0.5)
        assert abs(half) == pytest.approx(abs(full) * 0.5)

    def test_zero_confidence(self) -> None:
        from scripts.v460.lib.sidecar_types import compute_sidecar_offset_bps

        assert compute_sidecar_offset_bps(0.5, "buy", confidence=0.0) == 0.0

    def test_custom_boost(self) -> None:
        from scripts.v460.lib.sidecar_types import compute_sidecar_offset_bps

        offset = compute_sidecar_offset_bps(0.5, "buy", boost_bps=1.0)
        assert offset == pytest.approx(1.0)

    def test_symmetry(self) -> None:
        """BUY bias → buy offset と SELL bias → sell offset は同符号同値."""
        from scripts.v460.lib.sidecar_types import compute_sidecar_offset_bps

        buy_buy = compute_sidecar_offset_bps(0.5, "buy")
        sell_sell = compute_sidecar_offset_bps(-0.5, "sell")
        assert buy_buy == pytest.approx(sell_sell)


# ════════════════════════════════════════════════════════════════
# §2 sidecar_signal_io テスト
# ════════════════════════════════════════════════════════════════


class TestWriteAndReadSignal:
    """write_sidecar_signal / read_sidecar_signal の往復テスト."""

    def test_round_trip(self, tmp_path: Path) -> None:
        from scripts.v460.lib.sidecar_types import SidecarSignal
        from scripts.v460.lib.sidecar_signal_io import (
            write_sidecar_signal,
            read_sidecar_signal,
        )

        sig = SidecarSignal(
            timestamp="2026-03-10T12:00:00+09:00",
            directional_bias=0.42,
            model_version="test_v1",
            confidence=0.78,
            regime_hint="trending_up",
        )
        out = tmp_path / "signal.json"
        write_sidecar_signal(sig, out)

        loaded = read_sidecar_signal(out, ttl_sec=0)  # TTL 無効
        assert loaded is not None
        assert loaded.directional_bias == pytest.approx(0.42)
        assert loaded.confidence == pytest.approx(0.78)
        assert loaded.model_version == "test_v1"
        assert loaded.regime_hint == "trending_up"

    def test_atomic_overwrite(self, tmp_path: Path) -> None:
        from scripts.v460.lib.sidecar_types import SidecarSignal
        from scripts.v460.lib.sidecar_signal_io import (
            write_sidecar_signal,
            read_sidecar_signal,
        )

        out = tmp_path / "signal.json"
        sig1 = SidecarSignal(timestamp="t1", directional_bias=0.1)
        write_sidecar_signal(sig1, out)

        sig2 = SidecarSignal(timestamp="t2", directional_bias=-0.8)
        write_sidecar_signal(sig2, out)

        loaded = read_sidecar_signal(out, ttl_sec=0)
        assert loaded is not None
        assert loaded.directional_bias == pytest.approx(-0.8)


class TestReadSignalErrors:
    """read_sidecar_signal のエラーハンドリング."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        from scripts.v460.lib.sidecar_signal_io import read_sidecar_signal

        result = read_sidecar_signal(tmp_path / "nonexistent.json")
        assert result is None

    def test_invalid_json(self, tmp_path: Path) -> None:
        from scripts.v460.lib.sidecar_signal_io import read_sidecar_signal

        bad = tmp_path / "bad.json"
        bad.write_text("{invalid", encoding="utf-8")
        result = read_sidecar_signal(bad)
        assert result is None

    def test_missing_required_field(self, tmp_path: Path) -> None:
        from scripts.v460.lib.sidecar_signal_io import read_sidecar_signal

        p = tmp_path / "sig.json"
        p.write_text('{"timestamp": "t"}', encoding="utf-8")  # no directional_bias
        result = read_sidecar_signal(p, ttl_sec=0)
        assert result is None

    def test_invalid_bias_value(self, tmp_path: Path) -> None:
        from scripts.v460.lib.sidecar_signal_io import read_sidecar_signal

        p = tmp_path / "sig.json"
        data = {"timestamp": "t", "directional_bias": 5.0}
        p.write_text(json.dumps(data), encoding="utf-8")
        result = read_sidecar_signal(p, ttl_sec=0)
        assert result is None


class TestSignalStaleness:
    """TTL ベースの staleness チェック."""

    def test_fresh_signal(self, tmp_path: Path) -> None:
        from scripts.v460.lib.sidecar_types import SidecarSignal
        from scripts.v460.lib.sidecar_signal_io import (
            write_sidecar_signal,
            read_sidecar_signal,
            make_timestamp,
        )

        sig = SidecarSignal(
            timestamp=make_timestamp(),
            directional_bias=0.5,
        )
        out = tmp_path / "sig.json"
        write_sidecar_signal(sig, out)

        loaded = read_sidecar_signal(out, ttl_sec=600)
        assert loaded is not None

    def test_stale_signal(self, tmp_path: Path) -> None:
        from scripts.v460.lib.sidecar_types import SidecarSignal
        from scripts.v460.lib.sidecar_signal_io import (
            write_sidecar_signal,
            read_sidecar_signal,
        )

        old_ts = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        sig = SidecarSignal(timestamp=old_ts, directional_bias=0.5)
        out = tmp_path / "sig.json"
        write_sidecar_signal(sig, out)

        loaded = read_sidecar_signal(out, ttl_sec=60)
        assert loaded is None  # stale


class TestHelpers:
    """make_timestamp, create_neutral_signal."""

    def test_make_timestamp(self) -> None:
        from scripts.v460.lib.sidecar_signal_io import make_timestamp

        ts = make_timestamp()
        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None

    def test_create_neutral_signal(self) -> None:
        from scripts.v460.lib.sidecar_signal_io import create_neutral_signal

        sig = create_neutral_signal()
        assert sig.directional_bias == 0.0
        assert sig.confidence == 0.0
        assert sig.direction.name == "NEUTRAL"


# ════════════════════════════════════════════════════════════════
# §3 cycle_gate_aggregator sidecar injection テスト
# ════════════════════════════════════════════════════════════════


def _make_aggregator():
    """テスト用 CycleGateAggregator を生成."""
    from tests.unit.v460.conftest import make_gate_config
    from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator

    config = make_gate_config(
        skip_buy_unknown_regime=False,
        skip_ranging_buy_low_vol=False,
        skip_sell_trending=False,
        buy_dynamic_kill_enabled=False,
        sell_dynamic_kill_enabled=False,
        sell_velocity_skip_enabled=False,
        buy_velocity_skip_enabled=False,
        skip_sell_unknown_regime=False,
        narrow_spread_pause_enabled=False,
    )
    return CycleGateAggregator(config)


def _make_signal(bias: float, confidence: float = 1.0):
    from scripts.v460.lib.sidecar_types import SidecarSignal
    from scripts.v460.lib.sidecar_signal_io import make_timestamp

    return SidecarSignal(
        timestamp=make_timestamp(),
        directional_bias=bias,
        confidence=confidence,
    )


class TestCycleGateSidecarInjection:
    """365# P5: cycle_gate_aggregator への sidecar injection."""

    def test_no_sidecar_defaults(self) -> None:
        """sidecar_signal=None → offset=0, direction=neutral."""
        agg = _make_aggregator()
        result = agg.evaluate(
            side="buy", regime="ranging", vol_ratio=1.0,
            inv_net_imbalance=0.0, is_buy_killed=False, is_sell_killed=False,
        )
        assert result.sidecar_offset_bps == 0.0
        assert result.sidecar_direction == "neutral"

    def test_buy_bias_buy_side(self) -> None:
        """BUY bias → buy offset > 0 (攻撃的)."""
        agg = _make_aggregator()
        sig = _make_signal(0.5)
        result = agg.evaluate(
            side="buy", regime="ranging", vol_ratio=1.0,
            inv_net_imbalance=0.0, is_buy_killed=False, is_sell_killed=False,
            sidecar_signal=sig,
        )
        assert result.sidecar_offset_bps > 0
        assert result.sidecar_direction == "buy_bias"
        assert result.sidecar_bias == pytest.approx(0.5)

    def test_buy_bias_sell_side(self) -> None:
        """BUY bias → sell offset < 0 (保守的)."""
        agg = _make_aggregator()
        sig = _make_signal(0.5)
        result = agg.evaluate(
            side="sell", regime="ranging", vol_ratio=1.0,
            inv_net_imbalance=0.0, is_buy_killed=False, is_sell_killed=False,
            sidecar_signal=sig,
        )
        assert result.sidecar_offset_bps < 0

    def test_sell_bias(self) -> None:
        """SELL bias → sell を攻撃的に."""
        agg = _make_aggregator()
        sig = _make_signal(-0.5)
        result = agg.evaluate(
            side="sell", regime="ranging", vol_ratio=1.0,
            inv_net_imbalance=0.0, is_buy_killed=False, is_sell_killed=False,
            sidecar_signal=sig,
        )
        assert result.sidecar_offset_bps > 0
        assert result.sidecar_direction == "sell_bias"

    def test_neutral_no_offset(self) -> None:
        """NEUTRAL bias → offset = 0."""
        agg = _make_aggregator()
        sig = _make_signal(0.1)  # below threshold
        result = agg.evaluate(
            side="buy", regime="ranging", vol_ratio=1.0,
            inv_net_imbalance=0.0, is_buy_killed=False, is_sell_killed=False,
            sidecar_signal=sig,
        )
        assert result.sidecar_offset_bps == 0.0
        assert result.sidecar_direction == "neutral"

    def test_confidence_scaling(self) -> None:
        """confidence=0.5 → offset は半減."""
        agg = _make_aggregator()
        full = _make_signal(0.5, confidence=1.0)
        half = _make_signal(0.5, confidence=0.5)

        r_full = agg.evaluate(
            side="buy", regime="ranging", vol_ratio=1.0,
            inv_net_imbalance=0.0, is_buy_killed=False, is_sell_killed=False,
            sidecar_signal=full,
        )
        r_half = agg.evaluate(
            side="buy", regime="ranging", vol_ratio=1.0,
            inv_net_imbalance=0.0, is_buy_killed=False, is_sell_killed=False,
            sidecar_signal=half,
        )
        assert r_half.sidecar_offset_bps == pytest.approx(
            r_full.sidecar_offset_bps * 0.5,
        )

    def test_gate_blocked_no_sidecar(self) -> None:
        """Gate blocked → sidecar は適用されない (early return)."""
        from tests.unit.v460.conftest import make_gate_config
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator

        config = make_gate_config(skip_buy_unknown_regime=True)
        agg = CycleGateAggregator(config)
        sig = _make_signal(0.5)

        result = agg.evaluate(
            side="buy", regime=None, vol_ratio=1.0,
            inv_net_imbalance=0.0, is_buy_killed=False, is_sell_killed=False,
            sidecar_signal=sig,
        )
        assert result.blocked is True
        # Gate blocked → sidecar offset は未適用のまま 0
        assert result.sidecar_offset_bps == 0.0


class TestCycleGateResultSidecarFields:
    """CycleGateResult の sidecar フィールドのデフォルト値."""

    def test_defaults(self) -> None:
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateResult

        r = CycleGateResult()
        assert r.sidecar_offset_bps == 0.0
        assert r.sidecar_direction == "neutral"
        assert r.sidecar_bias == 0.0

    def test_audit_summary_includes_sidecar_info(self) -> None:
        """sidecar 情報が result に記録される."""
        agg = _make_aggregator()
        sig = _make_signal(0.6)
        result = agg.evaluate(
            side="buy", regime="ranging", vol_ratio=1.0,
            inv_net_imbalance=0.0, is_buy_killed=False, is_sell_killed=False,
            sidecar_signal=sig,
        )
        # 基本的なフィールドが設定されている
        assert result.sidecar_bias == pytest.approx(0.6)
        assert result.sidecar_direction == "buy_bias"
