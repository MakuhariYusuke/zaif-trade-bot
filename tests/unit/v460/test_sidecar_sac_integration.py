"""365# P3/P4/P5 テスト: SAC Sidecar types + signal I/O + cycle_gate injection.

カバレッジ対象:
  - sidecar_types.py: SidecarSignal, classify_bias, compute_sidecar_offset_bps
  - sidecar_signal_io.py: write/read atomic I/O, staleness, error handling
  - cycle_gate_aggregator.py: sidecar offset injection
  - 372# F2: _get_latest_obs() — env 末尾 observation 取得
  - 372# F1 Gap-3: sidecar bps offset → pricing 適用ロジック
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pytest

from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator, CycleGateResult
from scripts.v460.lib.sidecar_signal_io import (
    clear_sidecar_signal_cache,
    create_neutral_signal,
    get_sidecar_signal_cache_stats,
    make_timestamp,
    read_sidecar_signal,
    read_sidecar_signal_with_status,
    write_sidecar_signal,
)
from scripts.v460.lib.sidecar_types import (
    PPOSidecarSignal,
    SidecarDirection,
    SidecarSignal,
    classify_bias,
    compute_sidecar_offset_bps,
)
from scripts.v460.ml.sac_retrain_scheduler import SACRetrainConfig, _get_latest_obs
from tests.unit.v460.conftest import make_gate_config
from ztb.metrics.fill_quality import FillRecord


def _compute_confidence(
    roi: float,
    gate_threshold: float = 0.0,
    full_roi: float = 0.005,
) -> float:
    """sac_retrain_scheduler の confidence 計算ロジックを抽出。"""
    if full_roi <= gate_threshold:
        return 1.0
    if roi <= gate_threshold:
        return 0.0
    return min(1.0, (roi - gate_threshold) / (full_roi - gate_threshold))


# ════════════════════════════════════════════════════════════════
# §1 sidecar_types テスト
# ════════════════════════════════════════════════════════════════


class TestSidecarDirection:
    """SidecarDirection enum."""

    def test_values(self) -> None:
        assert SidecarDirection.BUY_BIAS == 1
        assert SidecarDirection.NEUTRAL == 0
        assert SidecarDirection.SELL_BIAS == -1


class TestClassifyBias:
    """classify_bias() — 閾値ベース分類."""

    def test_buy_bias(self) -> None:
        assert classify_bias(0.5) == SidecarDirection.BUY_BIAS
        assert classify_bias(1.0) == SidecarDirection.BUY_BIAS

    def test_sell_bias(self) -> None:
        assert classify_bias(-0.5) == SidecarDirection.SELL_BIAS
        assert classify_bias(-1.0) == SidecarDirection.SELL_BIAS

    def test_neutral(self) -> None:
        assert classify_bias(0.0) == SidecarDirection.NEUTRAL
        assert classify_bias(0.3) == SidecarDirection.NEUTRAL
        assert classify_bias(-0.3) == SidecarDirection.NEUTRAL
        assert classify_bias(0.29) == SidecarDirection.NEUTRAL

    def test_boundary_above(self) -> None:
        assert classify_bias(0.31) == SidecarDirection.BUY_BIAS
        assert classify_bias(-0.31) == SidecarDirection.SELL_BIAS

    def test_custom_threshold(self) -> None:
        assert classify_bias(0.15, threshold=0.1) == SidecarDirection.BUY_BIAS
        assert classify_bias(0.15, threshold=0.2) == SidecarDirection.NEUTRAL


class TestSidecarSignal:
    """SidecarSignal dataclass."""

    def test_creation(self) -> None:
        sig = SidecarSignal(timestamp="2026-03-10T12:00:00+09:00", directional_bias=0.42)
        assert sig.directional_bias == 0.42
        assert sig.confidence == 1.0
        assert sig.model_version == ""

    def test_direction_property(self) -> None:
        sig = SidecarSignal(timestamp="t", directional_bias=0.5)
        assert sig.direction == SidecarDirection.BUY_BIAS

        sig2 = SidecarSignal(timestamp="t", directional_bias=-0.5)
        assert sig2.direction == SidecarDirection.SELL_BIAS

        sig3 = SidecarSignal(timestamp="t", directional_bias=0.0)
        assert sig3.direction == SidecarDirection.NEUTRAL

    def test_bias_validation_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="directional_bias"):
            SidecarSignal(timestamp="t", directional_bias=1.5)

        with pytest.raises(ValueError, match="directional_bias"):
            SidecarSignal(timestamp="t", directional_bias=-1.1)

    def test_confidence_validation(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            SidecarSignal(timestamp="t", directional_bias=0.0, confidence=-0.1)
        with pytest.raises(ValueError, match="confidence"):
            SidecarSignal(timestamp="t", directional_bias=0.0, confidence=1.5)

    def test_frozen(self) -> None:
        sig = SidecarSignal(timestamp="t", directional_bias=0.0)
        with pytest.raises(AttributeError):
            sig.directional_bias = 0.5  # type: ignore[misc]


class TestComputeSidecarOffsetBps:
    """compute_sidecar_offset_bps() — 非対称 offset 計算."""

    def test_buy_bias_buy_side(self) -> None:
        offset = compute_sidecar_offset_bps(0.5, "buy")
        assert offset > 0  # 攻撃的

    def test_buy_bias_sell_side(self) -> None:
        offset = compute_sidecar_offset_bps(0.5, "sell")
        assert offset < 0  # 保守的

    def test_sell_bias_sell_side(self) -> None:
        offset = compute_sidecar_offset_bps(-0.5, "sell")
        assert offset > 0  # 攻撃的

    def test_sell_bias_buy_side(self) -> None:
        offset = compute_sidecar_offset_bps(-0.5, "buy")
        assert offset < 0  # 保守的

    def test_neutral_returns_zero(self) -> None:
        assert compute_sidecar_offset_bps(0.0, "buy") == 0.0
        assert compute_sidecar_offset_bps(0.0, "sell") == 0.0
        assert compute_sidecar_offset_bps(0.2, "buy") == 0.0

    def test_confidence_scaling(self) -> None:
        full = compute_sidecar_offset_bps(0.5, "buy", confidence=1.0)
        half = compute_sidecar_offset_bps(0.5, "buy", confidence=0.5)
        assert abs(half) == pytest.approx(abs(full) * 0.5)

    def test_zero_confidence(self) -> None:
        assert compute_sidecar_offset_bps(0.5, "buy", confidence=0.0) == 0.0

    def test_custom_boost(self) -> None:
        offset = compute_sidecar_offset_bps(0.5, "buy", boost_bps=1.0)
        assert offset == pytest.approx(1.0)

    def test_symmetry(self) -> None:
        """BUY bias → buy offset と SELL bias → sell offset は同符号同値."""
        buy_buy = compute_sidecar_offset_bps(0.5, "buy")
        sell_sell = compute_sidecar_offset_bps(-0.5, "sell")
        assert buy_buy == pytest.approx(sell_sell)


# ════════════════════════════════════════════════════════════════
# §2 sidecar_signal_io テスト
# ════════════════════════════════════════════════════════════════


class TestWriteAndReadSignal:
    """write_sidecar_signal / read_sidecar_signal の往復テスト."""

    def test_round_trip(self, tmp_path: Path) -> None:
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
        result = read_sidecar_signal(tmp_path / "nonexistent.json")
        assert result is None

    def test_invalid_json(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{invalid", encoding="utf-8")
        result = read_sidecar_signal(bad)
        assert result is None

    def test_missing_required_field(self, tmp_path: Path) -> None:
        p = tmp_path / "sig.json"
        p.write_text('{"timestamp": "t"}', encoding="utf-8")  # no directional_bias
        result = read_sidecar_signal(p, ttl_sec=0)
        assert result is None


class TestSidecarSignalCache:
    def test_cache_stats_and_clear(self, tmp_path: Path) -> None:
        clear_sidecar_signal_cache()
        out = tmp_path / "sig.json"
        write_sidecar_signal(create_neutral_signal(), out)
        assert read_sidecar_signal(out, ttl_sec=0) is not None

        stats = get_sidecar_signal_cache_stats()
        assert stats["entries"] == 1
        assert stats["max_entries"] >= 1

        clear_sidecar_signal_cache()
        assert get_sidecar_signal_cache_stats()["entries"] == 0

    def test_invalid_bias_value(self, tmp_path: Path) -> None:
        p = tmp_path / "sig.json"
        data = {"timestamp": "t", "directional_bias": 5.0}
        p.write_text(json.dumps(data), encoding="utf-8")
        result = read_sidecar_signal(p, ttl_sec=0)
        assert result is None


class TestSignalStaleness:
    """TTL ベースの staleness チェック."""

    def test_fresh_signal(self, tmp_path: Path) -> None:
        sig = SidecarSignal(
            timestamp=make_timestamp(),
            directional_bias=0.5,
        )
        out = tmp_path / "sig.json"
        write_sidecar_signal(sig, out)

        loaded = read_sidecar_signal(out, ttl_sec=600)
        assert loaded is not None

    def test_stale_signal(self, tmp_path: Path) -> None:
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        sig = SidecarSignal(timestamp=old_ts, directional_bias=0.5)
        out = tmp_path / "sig.json"
        write_sidecar_signal(sig, out)

        loaded = read_sidecar_signal(out, ttl_sec=60)
        assert loaded is None  # stale

    def test_stale_signal_twice_stays_stale(self, tmp_path: Path) -> None:
        """629# 回帰テスト: stale signal を 2 回読んでも stale のまま (error に化けない)."""
        clear_sidecar_signal_cache()
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        sig = SidecarSignal(timestamp=old_ts, directional_bias=0.3)
        out = tmp_path / "sig.json"
        write_sidecar_signal(sig, out)

        # 1 回目: stale
        r1, s1 = read_sidecar_signal_with_status(out, ttl_sec=60)
        assert s1 == "stale"
        assert r1 is None

        # 2 回目: キャッシュヒットでも stale のまま (修正前は "error" に化けていた)
        r2, s2 = read_sidecar_signal_with_status(out, ttl_sec=60)
        assert s2 == "stale", f"expected 'stale' but got '{s2}' (stale→error regression)"
        assert r2 is None


class TestHelpers:
    """make_timestamp, create_neutral_signal."""

    def test_make_timestamp(self) -> None:
        ts = make_timestamp()
        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None

    def test_create_neutral_signal(self) -> None:
        sig = create_neutral_signal()
        assert sig.directional_bias == 0.0
        assert sig.confidence == 0.0
        assert sig.direction.name == "NEUTRAL"


# ════════════════════════════════════════════════════════════════
# §3 cycle_gate_aggregator sidecar injection テスト
# ════════════════════════════════════════════════════════════════


def _make_aggregator():
    """テスト用 CycleGateAggregator を生成."""
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
    return SidecarSignal(
        timestamp=make_timestamp(),
        directional_bias=bias,
        confidence=confidence,
    )


def _make_ppo_signal(
    *,
    buy: float,
    sell: float,
    skip: float,
    model_version: str = "ppo_v1",
) -> PPOSidecarSignal:
    return PPOSidecarSignal.from_probabilities(
        timestamp=make_timestamp(),
        action_probabilities={"buy": buy, "sell": sell, "skip": skip},
        model_version=model_version,
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


class TestCycleGatePPOSidecarInjection:
    """675# PPO sidecar の safe veto / telemetry."""

    def test_none_signal_skips_gate(self) -> None:
        agg = _make_aggregator()

        result = agg.evaluate(
            side="buy",
            regime="ranging",
            vol_ratio=1.0,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=False,
            ppo_sidecar_signal=None,
        )

        assert result.blocked is False
        assert result.ppo_sidecar_action is None
        assert result.ppo_sidecar_override_active is False

    def test_skip_signal_blocks_cycle(self) -> None:
        agg = _make_aggregator()
        signal = _make_ppo_signal(buy=0.10, sell=0.15, skip=0.75)

        result = agg.evaluate(
            side="buy",
            regime="ranging",
            vol_ratio=1.0,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=False,
            ppo_sidecar_signal=signal,
        )

        assert result.blocked is True
        assert result.blocking_reason == "ppo_sidecar_skip"
        assert result.ppo_sidecar_action == "skip"
        assert result.ppo_sidecar_override_active is True

    def test_conflicting_side_blocks_cycle(self) -> None:
        agg = _make_aggregator()
        signal = _make_ppo_signal(buy=0.12, sell=0.76, skip=0.12)

        result = agg.evaluate(
            side="buy",
            regime="ranging",
            vol_ratio=1.0,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=False,
            ppo_sidecar_signal=signal,
        )

        assert result.blocked is True
        assert result.blocking_reason == "ppo_sidecar_side_conflict"
        assert result.ppo_sidecar_action == "sell"
        assert result.ppo_sidecar_override_active is True

    def test_below_threshold_is_observe_only(self) -> None:
        agg = _make_aggregator()
        signal = _make_ppo_signal(buy=0.52, sell=0.38, skip=0.10)

        result = agg.evaluate(
            side="buy",
            regime="ranging",
            vol_ratio=1.0,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=False,
            ppo_sidecar_signal=signal,
        )

        assert result.blocked is False
        assert result.ppo_sidecar_action == "buy"
        assert result.ppo_sidecar_override_active is False

    def test_below_margin_threshold_is_observe_only(self) -> None:
        agg = _make_aggregator()
        signal = PPOSidecarSignal(
            timestamp="2026-04-01T00:00:00+00:00",
            action="buy",
            action_probabilities={"buy": 0.51, "sell": 0.47, "skip": 0.02},
            confidence=0.80,
            model_version="ppo_v1",
        )

        result = agg.evaluate(
            side="buy",
            regime="ranging",
            vol_ratio=1.0,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=False,
            ppo_sidecar_signal=signal,
        )

        assert result.blocked is False
        assert result.ppo_sidecar_action == "buy"
        assert result.ppo_sidecar_action_margin == pytest.approx(0.04)
        assert result.ppo_sidecar_override_active is False

    def test_matching_side_passes_with_telemetry(self) -> None:
        agg = _make_aggregator()
        signal = _make_ppo_signal(buy=0.72, sell=0.18, skip=0.10)

        result = agg.evaluate(
            side="buy",
            regime="ranging",
            vol_ratio=1.0,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=False,
            ppo_sidecar_signal=signal,
        )

        assert result.blocked is False
        assert result.ppo_sidecar_action == "buy"
        assert result.ppo_sidecar_confidence == pytest.approx(0.72)
        assert result.ppo_sidecar_action_margin == pytest.approx(0.54)
        assert result.ppo_sidecar_model_version == "ppo_v1"
        assert result.ppo_sidecar_override_active is True


# ════════════════════════════════════════════════════════════════
# §5  372# F2: _get_latest_obs — env 末尾 observation 取得
# ════════════════════════════════════════════════════════════════


class _FakeLiteEnv:
    """LiteTradingEnv 相当の最小 stub."""

    def __init__(self, n_rows: int = 100, n_features: int = 12) -> None:
        self._feature_matrix = np.random.randn(n_rows, n_features).astype(
            np.float32,
        )
        self.current_step = 0

    def _get_observation(self):
        return self._feature_matrix[self.current_step].copy()

    def reset(self):
        self.current_step = 0
        return self._get_observation(), {}


class _FakeDfEnv:
    """HeavyTradingEnv 相当の最小 stub (df 属性あり)."""

    def __init__(self, n_rows: int = 200, n_features: int = 12) -> None:
        self.df = list(range(n_rows))  # len() 可能な何か
        self._data = np.random.randn(n_rows, n_features).astype(np.float32)
        self.current_step = 0

    def _get_observation(self):
        return self._data[self.current_step].copy()

    def reset(self):
        self.current_step = 0
        return self._get_observation(), {}


class TestGetLatestObs:
    """372# F2: _get_latest_obs() のテスト."""

    def test_lite_env_returns_last_row(self) -> None:
        """LiteTradingEnv パターン: 末尾行の observation を返す."""
        env = _FakeLiteEnv(n_rows=50, n_features=8)
        obs = _get_latest_obs(env)
        expected = env._feature_matrix[49]
        np.testing.assert_array_almost_equal(obs, expected)

    def test_lite_env_preserves_current_step(self) -> None:
        """_get_latest_obs 後に current_step が元に戻る."""
        env = _FakeLiteEnv(n_rows=50)
        env.current_step = 10
        _get_latest_obs(env)
        assert env.current_step == 10

    def test_df_env_returns_last_row(self) -> None:
        """HeavyTradingEnv パターン: df 末尾行の observation を返す."""
        env = _FakeDfEnv(n_rows=200, n_features=8)
        obs = _get_latest_obs(env)
        expected = env._data[199]
        np.testing.assert_array_almost_equal(obs, expected)

    def test_df_env_preserves_current_step(self) -> None:
        """HeavyTradingEnv パターン: current_step 復元."""
        env = _FakeDfEnv(n_rows=200)
        env.current_step = 42
        _get_latest_obs(env)
        assert env.current_step == 42

    def test_fallback_to_reset(self) -> None:
        """df も _feature_matrix もない env → reset() フォールバック."""
        class _PlainEnv:
            def __init__(self):
                self.current_step = 0
                self._obs = np.ones(4, dtype=np.float32)

            def reset(self):
                return self._obs.copy(), {}

        env = _PlainEnv()
        obs = _get_latest_obs(env)
        np.testing.assert_array_equal(obs, env._obs)


# ════════════════════════════════════════════════════════════════
# §6  372# F1 Gap-3: sidecar bps offset → pricing 適用ロジック
# ════════════════════════════════════════════════════════════════


class TestSidecarBpsOffset:
    """372# F1 Gap-3: sidecar_offset_bps → 価格調整の妥当性.

    fill_cycle_executor 内のロジックを再現して検証。
    """

    @staticmethod
    def _apply_sidecar_offset(
        side: str, order_price: float, sidecar_offset_bps: float,
    ) -> tuple[float, float]:
        """fill_cycle_executor.py 内のロジックを抽出 (テスト対象)."""
        if sidecar_offset_bps == 0.0 or order_price <= 0:
            return order_price, 0.0
        delta = round(sidecar_offset_bps / 10000.0 * order_price)
        if side == "buy":
            return round(order_price + delta), delta
        else:
            return round(order_price - delta), delta

    def test_positive_bps_buy_increases_price(self) -> None:
        """正bps + buy → 価格上昇 (mid に近づく = 攻撃的)."""
        new_price, delta = self._apply_sidecar_offset("buy", 15_000_000, 5.0)
        assert delta > 0
        assert new_price > 15_000_000

    def test_positive_bps_sell_decreases_price(self) -> None:
        """正bps + sell → 価格下降 (mid に近づく = 攻撃的)."""
        new_price, delta = self._apply_sidecar_offset("sell", 15_000_000, 5.0)
        assert delta > 0
        assert new_price < 15_000_000

    def test_negative_bps_buy_decreases_price(self) -> None:
        """負bps + buy → 価格下降 (mid から遠ざかる = 保守的)."""
        new_price, delta = self._apply_sidecar_offset("buy", 15_000_000, -5.0)
        assert delta < 0
        assert new_price < 15_000_000

    def test_negative_bps_sell_increases_price(self) -> None:
        """負bps + sell → 価格上昇 (mid から遠ざかる = 保守的)."""
        new_price, delta = self._apply_sidecar_offset("sell", 15_000_000, -5.0)
        assert delta < 0
        assert new_price > 15_000_000

    def test_zero_bps_no_change(self) -> None:
        """bps=0 → 価格不変."""
        new_price, delta = self._apply_sidecar_offset("buy", 15_000_000, 0.0)
        assert new_price == 15_000_000
        assert delta == 0.0

    def test_bps_magnitude_at_15m(self) -> None:
        """BTC 15M JPY で 5bps → 7500 JPY の調整."""
        _, delta = self._apply_sidecar_offset("buy", 15_000_000, 5.0)
        assert delta == 7500  # 5/10000 * 15_000_000

    def test_zero_price_no_change(self) -> None:
        """order_price=0 → 変化なし."""
        new_price, delta = self._apply_sidecar_offset("buy", 0, 5.0)
        assert new_price == 0
        assert delta == 0.0


# ════════════════════════════════════════════════════════════════
# §7  372# FillRecord sidecar フィールド + Deploy Gate
# ════════════════════════════════════════════════════════════════


class TestFillRecordSidecarFields:
    """372# FillRecord の sidecar_offset_bps / sidecar_bias フィールド."""

    def test_fields_exist(self) -> None:
        """FillRecord に sidecar 関連フィールドが定義されている。"""
        r = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
        )
        assert r.sidecar_offset_bps is None
        assert r.sidecar_bias is None

    def test_set_values(self) -> None:
        """sidecar フィールドに値を設定できる。"""
        r = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
            sidecar_offset_bps=3.5,
            sidecar_bias=0.42,
        )
        assert r.sidecar_offset_bps == pytest.approx(3.5)
        assert r.sidecar_bias == pytest.approx(0.42)

    def test_to_dict_includes_sidecar(self) -> None:
        """to_dict() に sidecar フィールドが含まれる。"""
        r = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
            sidecar_offset_bps=5.0,
        )
        d = r.to_dict()
        assert "sidecar_offset_bps" in d
        assert d["sidecar_offset_bps"] == pytest.approx(5.0)

    def test_round_trip_from_dict(self) -> None:
        """from_dict() で sidecar フィールドが復元される。"""
        r1 = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
            sidecar_offset_bps=2.5,
            sidecar_bias=-0.3,
        )
        r2 = FillRecord.from_dict(r1.to_dict())
        assert r2.sidecar_offset_bps == pytest.approx(2.5)
        assert r2.sidecar_bias == pytest.approx(-0.3)


class TestConfidenceDynamic:
    """372# confidence 動的計算のテスト."""

    def test_below_gate_zero(self) -> None:
        """ROI < gate → confidence=0."""
        assert _compute_confidence(-0.001) == 0.0


# ════════════════════════════════════════════════════════════════
# §8  487# P2 ログ改善テスト
# ════════════════════════════════════════════════════════════════


class TestRunSessionStateSidecarTracking:
    """487# P2: RunSessionState の sidecar / cancel_reason カウンタ."""

    def test_sidecar_counters_default_zero(self) -> None:
        from scripts.v460.lib.fill_loop_orchestrator import RunSessionState
        st = RunSessionState()
        assert st.sidecar_fresh_count == 0
        assert st.sidecar_stale_count == 0
        assert st.sidecar_missing_count == 0

    def test_sidecar_counters_increment(self) -> None:
        from scripts.v460.lib.fill_loop_orchestrator import RunSessionState
        st = RunSessionState()
        st.sidecar_fresh_count += 1
        st.sidecar_stale_count += 2
        st.sidecar_missing_count += 3
        assert st.sidecar_fresh_count == 1
        assert st.sidecar_stale_count == 2
        assert st.sidecar_missing_count == 3

    def test_cancel_reason_counts_default_empty(self) -> None:
        from scripts.v460.lib.fill_loop_orchestrator import RunSessionState
        st = RunSessionState()
        assert st.cancel_reason_counts == {}

    def test_cancel_reason_counts_tracking(self) -> None:
        from scripts.v460.lib.fill_loop_orchestrator import RunSessionState
        st = RunSessionState()
        for reason in ["timeout", "timeout", "spread_too_narrow", "timeout"]:
            st.cancel_reason_counts[reason] = st.cancel_reason_counts.get(reason, 0) + 1
        assert st.cancel_reason_counts["timeout"] == 3
        assert st.cancel_reason_counts["spread_too_narrow"] == 1

    def test_cancel_reason_counts_independent(self) -> None:
        """各インスタンスの cancel_reason_counts が独立."""
        from scripts.v460.lib.fill_loop_orchestrator import RunSessionState
        st1 = RunSessionState()
        st2 = RunSessionState()
        st1.cancel_reason_counts["x"] = 1
        assert "x" not in st2.cancel_reason_counts


class TestConfidenceDynamicCalc:
    """372# confidence 動的計算のテスト (§8 分離による継続)."""

    def test_at_gate_zero(self) -> None:
        """ROI == gate → confidence=0."""
        assert _compute_confidence(0.0) == 0.0

    def test_halfway(self) -> None:
        """ROI halfway between gate and full → confidence=0.5."""
        c = _compute_confidence(0.0025, gate_threshold=0.0, full_roi=0.005)
        assert c == pytest.approx(0.5)

    def test_at_full_roi(self) -> None:
        """ROI == full_roi → confidence=1.0."""
        c = _compute_confidence(0.005, gate_threshold=0.0, full_roi=0.005)
        assert c == pytest.approx(1.0)

    def test_above_full_roi_capped(self) -> None:
        """ROI > full_roi → confidence=1.0 (キャップ)."""
        c = _compute_confidence(0.01)
        assert c == 1.0

    def test_misconfigured_full_leq_gate(self) -> None:
        """full_roi <= gate → フォールバック 1.0."""
        c = _compute_confidence(0.001, gate_threshold=0.005, full_roi=0.005)
        assert c == 1.0


class TestDeployGateTradeCount:
    """372# Deploy Gate 強化: min_trade_count チェック."""

    def test_config_default(self) -> None:
        """min_trade_count のデフォルトは 3."""
        cfg = SACRetrainConfig()
        assert cfg.min_trade_count == 3

    def test_config_confidence_roi_full_default(self) -> None:
        """confidence_roi_full のデフォルトは 0.005."""
        cfg = SACRetrainConfig()
        assert cfg.confidence_roi_full == pytest.approx(0.005)


# ════════════════════════════════════════════════════════════════
# §487 read_sidecar_signal_with_status テスト
# ════════════════════════════════════════════════════════════════


class TestReadSidecarSignalWithStatus:
    """487# P0: read_sidecar_signal_with_status の 4 状態テスト."""

    def setup_method(self) -> None:
        clear_sidecar_signal_cache()

    def test_fresh(self, tmp_path: Path) -> None:
        """正常読込 → ("fresh", signal)."""
        from scripts.v460.lib.sidecar_signal_io import read_sidecar_signal_with_status

        sig = SidecarSignal(
            timestamp=make_timestamp(),
            directional_bias=0.5,
            model_version="test_v1",
            confidence=0.8,
        )
        p = tmp_path / "signal.json"
        write_sidecar_signal(sig, p)

        result, status = read_sidecar_signal_with_status(p)
        assert status == "fresh"
        assert result is not None
        assert result.directional_bias == pytest.approx(0.5)
        assert result.confidence == pytest.approx(0.8)
        assert result.model_version == "test_v1"

    def test_missing(self, tmp_path: Path) -> None:
        """ファイル不在 → ("missing", None)."""
        from scripts.v460.lib.sidecar_signal_io import read_sidecar_signal_with_status

        p = tmp_path / "nonexistent.json"
        result, status = read_sidecar_signal_with_status(p)
        assert status == "missing"
        assert result is None

    def test_stale(self, tmp_path: Path) -> None:
        """TTL 超過 → ("stale", None)."""
        from scripts.v460.lib.sidecar_signal_io import read_sidecar_signal_with_status

        old_ts = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        sig = SidecarSignal(
            timestamp=old_ts,
            directional_bias=0.3,
        )
        p = tmp_path / "signal.json"
        write_sidecar_signal(sig, p)

        result, status = read_sidecar_signal_with_status(p, ttl_sec=60)
        assert status == "stale"
        assert result is None

    def test_error_bad_json(self, tmp_path: Path) -> None:
        """JSON パースエラー → ("error", None)."""
        from scripts.v460.lib.sidecar_signal_io import read_sidecar_signal_with_status

        p = tmp_path / "signal.json"
        p.write_text("{invalid json", encoding="utf-8")

        result, status = read_sidecar_signal_with_status(p)
        assert status == "error"
        assert result is None

    def test_consistent_with_read_sidecar_signal(self, tmp_path: Path) -> None:
        """read_sidecar_signal と結果が一致する。"""
        from scripts.v460.lib.sidecar_signal_io import read_sidecar_signal_with_status

        sig = SidecarSignal(
            timestamp=make_timestamp(),
            directional_bias=-0.2,
        )
        p = tmp_path / "signal.json"
        write_sidecar_signal(sig, p)

        clear_sidecar_signal_cache()
        plain = read_sidecar_signal(p)
        clear_sidecar_signal_cache()
        with_status, status = read_sidecar_signal_with_status(p)

        assert status == "fresh"
        assert plain is not None
        assert with_status is not None
        assert plain.directional_bias == with_status.directional_bias


class TestFillRecordSidecarAttributionFields:
    """487# P0: FillRecord の sidecar attribution フィールド."""

    def test_new_fields_exist(self) -> None:
        """sidecar_confidence / model_version / signal_status が定義されている。"""
        r = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
        )
        assert r.sidecar_confidence is None
        assert r.sidecar_model_version is None
        assert r.sidecar_signal_status is None

    def test_set_values(self) -> None:
        """新フィールドに値を設定できる。"""
        r = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
            sidecar_confidence=0.85,
            sidecar_model_version="sac_v460_20260319",
            sidecar_signal_status="fresh",
        )
        assert r.sidecar_confidence == pytest.approx(0.85)
        assert r.sidecar_model_version == "sac_v460_20260319"
        assert r.sidecar_signal_status == "fresh"

    def test_round_trip(self) -> None:
        """to_dict/from_dict で新フィールドが復元される。"""
        r1 = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
            sidecar_confidence=0.7,
            sidecar_model_version="v1",
            sidecar_signal_status="stale",
        )
        r2 = FillRecord.from_dict(r1.to_dict())
        assert r2.sidecar_confidence == pytest.approx(0.7)
        assert r2.sidecar_model_version == "v1"
        assert r2.sidecar_signal_status == "stale"


class TestFillRecordPPOSidecarFields:
    """675# PPO sidecar attribution フィールド."""

    def test_new_fields_exist(self) -> None:
        r = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
        )
        assert r.ppo_sidecar_action is None
        assert r.ppo_sidecar_confidence is None
        assert r.ppo_sidecar_action_margin is None
        assert r.ppo_sidecar_model_version is None
        assert r.ppo_sidecar_signal_status is None
        assert r.ppo_sidecar_override_active is None

    def test_round_trip(self) -> None:
        r1 = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
            ppo_sidecar_action="skip",
            ppo_sidecar_confidence=0.81,
            ppo_sidecar_action_margin=0.47,
            ppo_sidecar_model_version="ppo_v461_20260401",
            ppo_sidecar_signal_status="fresh",
            ppo_sidecar_override_active=True,
        )
        r2 = FillRecord.from_dict(r1.to_dict())
        assert r2.ppo_sidecar_action == "skip"
        assert r2.ppo_sidecar_confidence == pytest.approx(0.81)
        assert r2.ppo_sidecar_action_margin == pytest.approx(0.47)
        assert r2.ppo_sidecar_model_version == "ppo_v461_20260401"
        assert r2.ppo_sidecar_signal_status == "fresh"
        assert r2.ppo_sidecar_override_active is True
