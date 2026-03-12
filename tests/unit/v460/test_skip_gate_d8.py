"""
D8: SkipGate evaluate / warm_start テストカバレッジ強化 — 122#.

118# D8 で指摘された evaluate/warm_start のテスト不足を補完:
- A3: side 別無効化 (buy_enabled/sell_enabled)
- A2: warm_start threshold 即収束 (quantile 直接計算)
- build_features_from_market_state() の基本テスト
- save/load round-trip
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scripts.v460.ml.skip_gate import (
    SkipGate,
    SkipGateConfig,
    SkipDecision,
    build_features_from_market_state,
    warm_start_skip_gate_thresholds,
)
from tests.unit.v460._skip_gate_test_helpers import save_and_load_skip_gate


class _PickleStub:
    """SkipGate save/load 用の最小 picklable object."""

    def __init__(self, name: str) -> None:
        self.name = name


class _CallableStub:
    """`return_value` を持つ軽量 callable."""

    def __init__(self, return_value: np.ndarray) -> None:
        self.return_value = return_value

    def __call__(self, *_args: object, **_kwargs: object) -> np.ndarray:
        return self.return_value


class _PipelineStub:
    """SkipGate evaluate 用の最小 pipeline stub."""

    steps: list[tuple[str, object]] = []

    def __init__(self, predict_prob: float) -> None:
        self.predict_proba = _CallableStub(
            np.array([[1.0 - predict_prob, predict_prob]], dtype=float)
        )
        self.predict = _CallableStub(np.array([0.5], dtype=float))

    def set_output(self, **_kwargs: object) -> "_PipelineStub":
        return self


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_gate(
    *,
    mode: str = "as",
    enabled: bool = True,
    buy_enabled: bool = True,
    sell_enabled: bool = True,
    as_threshold: float = 0.60,
    as_threshold_buy: float | None = 0.65,
    as_threshold_sell: float | None = 0.65,
    adaptive_threshold: bool = True,
    target_skip_rate_buy: float = 0.10,
    target_skip_rate_sell: float = 0.20,
    adaptive_min_samples: int = 20,
    adaptive_step: float = 0.05,
    predict_prob: float = 0.55,
) -> SkipGate:
    """テスト用 SkipGate ファクトリ."""
    config = SkipGateConfig(
        mode=mode,
        enabled=enabled,
        buy_enabled=buy_enabled,
        sell_enabled=sell_enabled,
        as_threshold=as_threshold,
        as_threshold_buy=as_threshold_buy,
        as_threshold_sell=as_threshold_sell,
        adaptive_threshold=adaptive_threshold,
        target_skip_rate_buy=target_skip_rate_buy,
        target_skip_rate_sell=target_skip_rate_sell,
        adaptive_min_samples=adaptive_min_samples,
        adaptive_step=adaptive_step,
    )
    mock_pipeline = _PipelineStub(predict_prob)
    gate = SkipGate(
        model=_PickleStub("model"),
        scaler=_PickleStub("scaler"),
        feature_cols=["spread_jpy", "offset_ratio", "regime_trending"],
        config=config,
        pipeline=mock_pipeline,
    )
    return gate


def _make_features() -> dict[str, float]:
    """テスト用特徴量辞書."""
    return {
        "spread_jpy": 3000.0,
        "offset_ratio": 0.05,
        "regime_trending": 0.0,
    }


def _make_fill_records(
    *,
    n_buy: int = 30,
    n_sell: int = 30,
    buy_probs: list[float] | None = None,
    sell_probs: list[float] | None = None,
    base_timestamp: float = 1770975573.0,
) -> list[dict[str, object]]:
    """warm_start 用の fill record dict を生成."""
    records: list[dict[str, object]] = []
    buy_p = buy_probs or [0.45 + i * 0.005 for i in range(n_buy)]
    sell_p = sell_probs or [0.50 + i * 0.005 for i in range(n_sell)]
    for i, p in enumerate(buy_p):
        records.append({
            "cycle_id": f"buy_{i:03d}",
            "side": "buy",
            "skip_gate_as_prob": p,
            "filled": True,
            "timestamp": base_timestamp + i * 120,
        })
    for i, p in enumerate(sell_p):
        records.append({
            "cycle_id": f"sell_{i:03d}",
            "side": "sell",
            "skip_gate_as_prob": p,
            "filled": True,
            "timestamp": base_timestamp + (n_buy + i) * 120,
        })
    return records


def _run_warm_start(
    gate: SkipGate,
    records: list[dict[str, object]],
    *,
    window: int,
) -> None:
    """warm_start を file I/O なしで実行する."""
    fake_file = Path("fill_records_20260220.jsonl")
    with patch(
        "ztb.ml.skip_gate.list_fill_record_files",
        return_value=[fake_file] if records else [],
    ), patch(
        "ztb.ml.skip_gate.iter_jsonl_objects",
        return_value=iter(records),
    ):
        warm_start_skip_gate_thresholds(gate, Path("."), window=window)


# =====================================================================
# A3: side 別無効化テスト (118# A3 sell 逆選別対策)
# =====================================================================

class TestSideEnableDisable:
    """118# A3: buy_enabled/sell_enabled パラメータのテスト."""

    def test_sell_disabled_skips_nothing(self) -> None:
        """sell_enabled=False → sell 側は常にスキップしない."""
        gate = _make_gate(sell_enabled=False, predict_prob=0.90)
        features = _make_features()
        decision = gate.evaluate(features, side="sell")
        assert decision.should_skip is False
        assert decision.reason == "sell_gate_disabled"

    def test_sell_disabled_buy_unaffected(self) -> None:
        """sell 無効化でも buy 側は通常判定."""
        gate = _make_gate(sell_enabled=False, predict_prob=0.90)
        features = _make_features()
        decision = gate.evaluate(features, side="buy")
        # P(AS) = 0.90 > threshold 0.65 → should_skip
        assert decision.should_skip is True

    def test_buy_disabled_skips_nothing(self) -> None:
        """buy_enabled=False → buy 側は常にスキップしない."""
        gate = _make_gate(buy_enabled=False, predict_prob=0.90)
        features = _make_features()
        decision = gate.evaluate(features, side="buy")
        assert decision.should_skip is False
        assert decision.reason == "buy_gate_disabled"

    def test_buy_disabled_sell_unaffected(self) -> None:
        """buy 無効化でも sell 側は通常判定."""
        gate = _make_gate(buy_enabled=False, predict_prob=0.90)
        features = _make_features()
        decision = gate.evaluate(features, side="sell")
        assert decision.should_skip is True

    def test_both_disabled(self) -> None:
        """両方無効 → 両方スキップなし."""
        gate = _make_gate(buy_enabled=False, sell_enabled=False, predict_prob=0.90)
        features = _make_features()
        buy_dec = gate.evaluate(features, side="buy")
        sell_dec = gate.evaluate(features, side="sell")
        assert buy_dec.should_skip is False
        assert sell_dec.should_skip is False

    def test_gate_disabled_takes_precedence(self) -> None:
        """enabled=False は buy_enabled/sell_enabled より優先."""
        gate = _make_gate(enabled=False, buy_enabled=True, sell_enabled=True)
        features = _make_features()
        decision = gate.evaluate(features, side="buy")
        assert decision.should_skip is False
        assert decision.reason == "gate_disabled"

    def test_skip_decision_fields_on_disabled(self) -> None:
        """無効化時の SkipDecision の各フィールドが正しい."""
        gate = _make_gate(sell_enabled=False)
        features = _make_features()
        dec = gate.evaluate(features, side="sell")
        assert dec.predicted_pnl_bps == 0.0
        assert dec.features_used == 0


# =====================================================================
# A2: warm_start 即収束テスト (118# A2)
# =====================================================================

class TestWarmStartImmediateConvergence:
    """118# A2: warm_start_skip_gate_thresholds の即収束テスト."""

    def test_immediate_threshold_convergence(self) -> None:
        """warm_start 後、閾値が YAML 初期値ではなく分位点に即座に設定される."""
        gate = _make_gate(
            as_threshold_buy=0.65,
            as_threshold_sell=0.65,
            adaptive_threshold=True,
            target_skip_rate_buy=0.10,
            target_skip_rate_sell=0.20,
            adaptive_min_samples=20,
        )
        _run_warm_start(
            gate,
            _make_fill_records(
                n_buy=30,
                n_sell=30,
                buy_probs=[0.42 + i * 0.005 for i in range(30)],
                sell_probs=[0.48 + i * 0.005 for i in range(30)],
            ),
            window=50,
        )

        # buy: P(AS) range [0.42, 0.565], target 10% → 90th percentile
        # q_idx = min(int(30 * 0.90), 29) = 27 → sorted[27] = 0.42 + 27*0.005 = 0.555
        assert gate.config.as_threshold_buy != 0.65  # 初期値から変更されている
        assert gate.config.as_threshold_buy is not None
        assert gate.config.as_threshold_buy == pytest.approx(0.555, abs=1e-6)

        # sell: P(AS) range [0.48, 0.625], target 20% → 80th percentile
        # q_idx = min(int(30 * 0.80), 29) = 24 → sorted[24] = 0.48 + 24*0.005 = 0.600
        assert gate.config.as_threshold_sell != 0.65
        assert gate.config.as_threshold_sell is not None
        assert gate.config.as_threshold_sell == pytest.approx(0.600, abs=1e-6)

    def test_warm_start_restores_history(self) -> None:
        """warm_start が P(AS) 履歴を正しく復元する."""
        gate = _make_gate(adaptive_threshold=True)
        _run_warm_start(gate, _make_fill_records(n_buy=12, n_sell=12), window=50)

        assert len(gate._pas_history_buy) == 12
        assert len(gate._pas_history_sell) == 12

    def test_warm_start_window_truncation(self) -> None:
        """window パラメータがヒストリーサイズを制限."""
        gate = _make_gate(adaptive_threshold=True)
        _run_warm_start(gate, _make_fill_records(n_buy=30, n_sell=30), window=20)

        assert len(gate._pas_history_buy) <= 20
        assert len(gate._pas_history_sell) <= 20

    def test_warm_start_insufficient_samples_no_calibration(self) -> None:
        """サンプル不足時は閾値を変更しない."""
        gate = _make_gate(
            as_threshold_buy=0.65,
            as_threshold_sell=0.65,
            adaptive_threshold=True,
            adaptive_min_samples=50,  # 高い要件
        )
        _run_warm_start(gate, _make_fill_records(n_buy=10, n_sell=10), window=50)

        # 10 < 50 → 閾値は変更されない
        assert gate.config.as_threshold_buy == 0.65
        assert gate.config.as_threshold_sell == 0.65

    def test_warm_start_empty_directory(self) -> None:
        """空ディレクトリでもクラッシュしない."""
        gate = _make_gate(adaptive_threshold=True)
        _run_warm_start(gate, [], window=50)
        assert gate._pas_history_buy == []
        assert gate._pas_history_sell == []

    def test_warm_start_non_adaptive_is_noop(self) -> None:
        """adaptive_threshold=False では閾値変更なし."""
        gate = _make_gate(
            adaptive_threshold=False,
            as_threshold_buy=0.65,
            as_threshold_sell=0.65,
        )
        _run_warm_start(gate, _make_fill_records(n_buy=12, n_sell=12), window=50)

        # 履歴は復元されるが閾値は変更されない
        assert len(gate._pas_history_buy) > 0
        assert gate.config.as_threshold_buy == 0.65

    def test_warm_start_filters_stale_records_by_trained_at(self) -> None:
        """124# モデル交代時: trained_at 以前のレコードが除外される."""
        from datetime import datetime as _dt

        gate = _make_gate(
            adaptive_threshold=True,
            as_threshold_buy=0.50,
            as_threshold_sell=0.50,
            adaptive_min_samples=5,
        )
        # trained_at をタイムスタンプ中間点から生成 (TZ 問題を回避)
        boundary_ts = 1771450000.0
        trained_at_str = _dt.fromtimestamp(boundary_ts).isoformat()
        gate.metadata = {
            "version": "v3_really_bad30",
            "trained_at": trained_at_str,
        }

        old_ts = boundary_ts - 100000.0  # clearly before trained_at
        new_ts = boundary_ts + 100000.0  # clearly after trained_at
        records = [
            {
                "side": "buy",
                "skip_gate_as_prob": 0.45 + i * 0.005,
                "timestamp": old_ts + i * 120,
            }
            for i in range(20)
        ] + [
            {
                "side": "buy",
                "skip_gate_as_prob": 0.30 + i * 0.01,
                "timestamp": new_ts + i * 120,
            }
            for i in range(5)
        ]
        _run_warm_start(gate, records, window=50)

        # 旧 20件がフィルタされ、新 5件のみ復元
        assert len(gate._pas_history_buy) == 5
        # 新レコードの値は 0.30〜0.34
        assert all(0.29 <= p <= 0.35 for p in gate._pas_history_buy)

    def test_warm_start_no_trained_at_uses_all_records(self) -> None:
        """metadata に trained_at がないモデルでは全レコードを使用 (後方互換)."""
        gate = _make_gate(adaptive_threshold=True, adaptive_min_samples=5)
        gate.metadata = {}  # trained_at なし

        _run_warm_start(gate, _make_fill_records(n_buy=6, n_sell=6), window=12)

        # 全レコードが使用される (後方互換)
        assert len(gate._pas_history_buy) == 6
        assert len(gate._pas_history_sell) == 6

    def test_warm_start_ignores_malformed_timestamp(self) -> None:
        """timestamp が壊れた行を含んでも warm_start が継続する."""
        gate = _make_gate(adaptive_threshold=True, adaptive_min_samples=3)
        gate.metadata = {"trained_at": "2026-02-01T00:00:00"}

        records = [
            {"side": "buy", "skip_gate_as_prob": 0.40, "timestamp": "bad"},
            {"side": "buy", "skip_gate_as_prob": 0.45, "timestamp": 1771451000.0},
            {"side": "buy", "skip_gate_as_prob": 0.50, "timestamp": 1771451200.0},
        ]
        _run_warm_start(gate, records, window=50)

        assert len(gate._pas_history_buy) == 2


# =====================================================================
# build_features_from_market_state テスト
# =====================================================================

class TestBuildFeaturesFromMarketState:
    """build_features_from_market_state() の基本テスト."""

    def test_basic_features_count(self) -> None:
        """use_ob_features=False → 16 特徴量."""
        features = build_features_from_market_state(
            side="buy",
            spread_jpy=3000.0,
            offset_ratio=0.05,
            regime="ranging",
        )
        assert len(features) == 16
        assert "spread_jpy" in features
        assert "side_buy" in features
        assert "regime_ranging" in features

    def test_ob_features_adds_3(self) -> None:
        """use_ob_features=True → +3 = 19 特徴量."""
        features = build_features_from_market_state(
            side="buy",
            spread_jpy=3000.0,
            offset_ratio=0.05,
            regime="trending",
            use_ob_features=True,
            best_bid=10_000_000.0,
            best_ask=10_003_000.0,
            bid_vol_5=1.5,
            ask_vol_5=2.0,
        )
        assert len(features) == 19
        assert "spread_bps_ob" in features
        assert "depth_imbalance_ob" in features
        assert "side_aligned_imbalance" in features

    def test_side_buy_flag(self) -> None:
        """side='buy' → side_buy = 1.0."""
        features = build_features_from_market_state(
            side="buy", spread_jpy=3000.0, offset_ratio=0.05, regime="ranging",
        )
        assert features["side_buy"] == 1.0

    def test_side_sell_flag(self) -> None:
        """side='sell' → side_buy = 0.0."""
        features = build_features_from_market_state(
            side="sell", spread_jpy=3000.0, offset_ratio=0.05, regime="ranging",
        )
        assert features["side_buy"] == 0.0

    def test_regime_encoding(self) -> None:
        """regime 引数が one-hot で正しくエンコードされる."""
        for regime, expected_col in [
            ("trending", "regime_trending"),
            ("ranging", "regime_ranging"),
            ("high_vol", "regime_high_vol"),
        ]:
            features = build_features_from_market_state(
                side="buy", spread_jpy=3000.0, offset_ratio=0.05, regime=regime,
            )
            assert features[expected_col] == 1.0
            # 他の regime フラグは 0
            for other in ["regime_trending", "regime_ranging", "regime_high_vol"]:
                if other != expected_col:
                    assert features[other] == 0.0

    def test_unknown_regime_all_zero(self) -> None:
        """未知 regime → 全 regime フラグが 0."""
        features = build_features_from_market_state(
            side="buy", spread_jpy=3000.0, offset_ratio=0.05, regime="unknown",
        )
        assert features["regime_trending"] == 0.0
        assert features["regime_ranging"] == 0.0
        assert features["regime_high_vol"] == 0.0

    def test_hour_features_sin_cos(self) -> None:
        """market_timestamp から hour_sin / hour_cos が計算される."""
        # 2026-02-20 12:00:00 (UTC) → hour=12 → sin(2π*12/24)=0, cos(2π*12/24)=1
        # Use explicit timestamp for noon
        features = build_features_from_market_state(
            side="buy", spread_jpy=3000.0, offset_ratio=0.05, regime="ranging",
            market_timestamp=1771502400.0,  # some fixed timestamp
        )
        assert "hour_sin" in features
        assert "hour_cos" in features
        # Values should be in [-1, 1]
        assert -1.0 <= features["hour_sin"] <= 1.0
        assert -1.0 <= features["hour_cos"] <= 1.0

    def test_trade_statistics_with_recent_trades(self) -> None:
        """recent_trades がある場合、取引統計が計算される."""
        ts_now = 1771502400.0
        trades = [
            {"ts": ts_now - 30, "price": 10_000_000.0, "amount": 0.1, "side": "buy"},
            {"ts": ts_now - 20, "price": 10_001_000.0, "amount": 0.2, "side": "sell"},
            {"ts": ts_now - 10, "price": 10_002_000.0, "amount": 0.15, "side": "buy"},
        ]
        features = build_features_from_market_state(
            side="buy", spread_jpy=3000.0, offset_ratio=0.05, regime="ranging",
            recent_trades=trades, market_timestamp=ts_now,
        )
        assert features["trade_count_60s"] == 3
        assert features["avg_trade_size"] > 0
        assert features["buy_ratio"] > 0
        assert features["vpin_60s"] > 0

    def test_no_trades_defaults(self) -> None:
        """recent_trades=None → デフォルト値."""
        features = build_features_from_market_state(
            side="buy", spread_jpy=3000.0, offset_ratio=0.05, regime="ranging",
        )
        assert features["trade_count_60s"] == 0.0
        assert features["buy_ratio"] == 0.5
        assert features["vpin_60s"] == 0.5

    def test_side_aligned_features(self) -> None:
        """side_aligned_tfi/velocity の符号が side に依存."""
        ts_now = 1771502400.0
        trades = [
            {"ts": ts_now - 30, "price": 10_000_000.0, "amount": 0.5, "side": "buy"},
            {"ts": ts_now - 10, "price": 10_010_000.0, "amount": 0.1, "side": "sell"},
        ]
        feat_buy = build_features_from_market_state(
            side="buy", spread_jpy=3000.0, offset_ratio=0.05, regime="ranging",
            recent_trades=trades, market_timestamp=ts_now,
        )
        feat_sell = build_features_from_market_state(
            side="sell", spread_jpy=3000.0, offset_ratio=0.05, regime="ranging",
            recent_trades=trades, market_timestamp=ts_now,
        )
        # TFI = (buy_vol - sell_vol) / total_vol > 0 (buy dominant)
        # buy: side_aligned_tfi = TFI * 1 > 0
        # sell: side_aligned_tfi = TFI * -1 < 0
        assert feat_buy["side_aligned_tfi"] > 0
        assert feat_sell["side_aligned_tfi"] < 0


# =====================================================================
# SkipGate save/load round-trip
# =====================================================================

class TestSkipGateSaveLoad:
    """SkipGate.save() / SkipGate.load() のラウンドトリップ."""

    @staticmethod
    def _make_picklable_gate(**kwargs: object) -> SkipGate:
        """pickle 可能な SkipGate (MagicMock は pickle 不可)."""
        config = SkipGateConfig(
            as_threshold_buy=kwargs.get("as_threshold_buy", 0.55),  # type: ignore[arg-type]
            sell_enabled=kwargs.get("sell_enabled", False),  # type: ignore[arg-type]
        )
        model = _PickleStub("model")
        scaler = _PickleStub("scaler")
        feature_cols = ["spread_jpy", "offset_ratio", "regime_trending"]
        return SkipGate(
            model=model, scaler=scaler, feature_cols=feature_cols,
            config=config, pipeline=None,
        )

    def test_save_load_roundtrip(self) -> None:
        """save → load で config/feature_cols が復元される."""
        gate = self._make_picklable_gate(as_threshold_buy=0.55, sell_enabled=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_gate.pkl"
            loaded = save_and_load_skip_gate(gate, path)

            assert loaded.config.as_threshold_buy == 0.55
            assert loaded.config.sell_enabled is False
            assert loaded.feature_cols == gate.feature_cols

    def test_save_creates_hash_file(self) -> None:
        """save 時に .sha256 ファイルが作成される."""
        gate = self._make_picklable_gate()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_gate.pkl"
            gate.save(path)
            hash_path = path.with_suffix(".pkl.sha256")
            assert hash_path.exists()
            assert len(hash_path.read_text().strip()) == 64  # SHA256 hex


# =====================================================================
# SkipGate evaluate edge cases
# =====================================================================

class TestSkipGateEvaluateEdgeCases:
    """evaluate() のエッジケーステスト."""

    def test_insufficient_features_returns_pass(self) -> None:
        """特徴量 < 3 件 → スキップしない."""
        gate = _make_gate()
        decision = gate.evaluate({"spread_jpy": 3000.0}, side="buy")
        # Only 1 feature provided → insufficient
        assert decision.should_skip is False
        assert "insufficient_features" in decision.reason

    def test_pnl_mode_uses_predict(self) -> None:
        """mode='pnl' では predict (回帰) が使われる."""
        gate = _make_gate(mode="pnl")
        gate._pipeline.predict.return_value = np.array([-0.5])
        features = _make_features()
        decision = gate.evaluate(features, side="buy")
        # pred_pnl = -0.5 < threshold_bps = 0.0 → skip
        assert decision.should_skip is True

    def test_as_mode_decision_fields(self) -> None:
        """mode='as' の SkipDecision に as_probability が設定される."""
        gate = _make_gate(mode="as", predict_prob=0.55)
        features = _make_features()
        decision = gate.evaluate(features, side="buy")
        assert decision.as_probability == pytest.approx(0.55)
        assert decision.threshold_used is not None

    def test_side_none_uses_base_threshold(self) -> None:
        """side=None → 共通 as_threshold が使われる."""
        gate = _make_gate(
            mode="as",
            as_threshold=0.50,
            as_threshold_buy=0.65,
            as_threshold_sell=0.65,
            predict_prob=0.55,
            adaptive_threshold=False,
        )
        features = _make_features()
        decision = gate.evaluate(features, side=None)
        # P(AS) = 0.55 >= shared threshold 0.50 → skip
        assert decision.should_skip is True

    def test_skip_rate_limit_force_pass(self) -> None:
        """連続スキップ率が max_skip_rate を超えると force-pass."""
        gate = _make_gate(predict_prob=0.90)  # 高確率で skip
        gate.config.max_skip_rate = 0.5
        features = _make_features()
        # 20 回 skip を履歴に積む → skip_rate = 100%
        gate._recent_skips_buy = [True] * 20
        decision = gate.evaluate(features, side="buy")
        assert decision.should_skip is False
        assert "skip_rate_limit" in decision.reason

    def test_invalid_feature_value_is_tolerated(self) -> None:
        """非数値特徴量が来てもクラッシュせず insufficient_features 扱い."""
        gate = _make_gate()
        decision = gate.evaluate(
            {"spread_jpy": "bad", "offset_ratio": None, "regime_trending": "x"},
            side="buy",
        )
        assert decision.should_skip is False
        assert "insufficient_features" in decision.reason


# =====================================================================
# 125# mode=pnl + adaptive threshold テスト
# =====================================================================

class TestPnlAdaptiveThreshold:
    """125# _calibrate_pnl_threshold のテスト."""

    def test_pnl_adaptive_warmup_uses_static(self) -> None:
        """adaptive_min_samples 未満では static threshold を使用."""
        gate = _make_gate(
            mode="pnl",
            adaptive_threshold=True,
            adaptive_min_samples=20,
        )
        gate.config.threshold_bps = -1.0
        gate._pipeline.predict.return_value = np.array([-0.5])  # > -1.0 → pass
        features = _make_features()
        # ウォームアップ中は threshold_bps = -1.0 が使われる
        decision = gate.evaluate(features, side="sell")
        assert decision.should_skip is False  # -0.5 > -1.0 → pass
        assert decision.threshold_used == pytest.approx(-1.0)

    def test_pnl_adaptive_calibrates_after_min_samples(self) -> None:
        """min_samples 到達後は分位点ベースの閾値に切り替わる."""
        gate = _make_gate(
            mode="pnl",
            adaptive_threshold=True,
            adaptive_min_samples=5,
        )
        gate.config.threshold_bps = 0.0
        gate.config.target_skip_rate_sell = 0.20
        gate.config.adaptive_step = 100.0  # 即収束させる
        features = _make_features()

        # 5件の PnL 予測を蓄積 (sell)
        pnl_values = [-3.0, -1.0, 0.0, 1.0, 2.0]
        for pnl in pnl_values:
            gate._pipeline.predict.return_value = np.array([pnl])
            gate.evaluate(features, side="sell")

        # 6件目: 較正が有効に (5 >= adaptive_min_samples=5)
        # 分布 [-3, -1, 0, 1, 2] の 20% 分位点
        # idx = int(5 * 0.20) = 1, sorted[1] = -1.0
        gate._pipeline.predict.return_value = np.array([0.5])
        decision = gate.evaluate(features, side="sell")
        # threshold は -1.0 付近に較正される (step=100 で即収束)
        assert decision.threshold_used is not None
        assert decision.threshold_used < 0.0  # 0.0 より下の分位点

    def test_pnl_adaptive_per_side_independence(self) -> None:
        """buy/sell 側の PnL 履歴が独立して管理される."""
        gate = _make_gate(
            mode="pnl",
            adaptive_threshold=True,
            adaptive_min_samples=3,
        )
        gate.config.threshold_bps = 0.0
        gate.config.adaptive_step = 100.0
        features = _make_features()

        # buy 側に正の PnL を蓄積
        for pnl in [1.0, 2.0, 3.0, 4.0]:
            gate._pipeline.predict.return_value = np.array([pnl])
            gate.evaluate(features, side="buy")

        # sell 側に負の PnL を蓄積
        for pnl in [-4.0, -3.0, -2.0, -1.0]:
            gate._pipeline.predict.return_value = np.array([pnl])
            gate.evaluate(features, side="sell")

        # buy/sell の閾値が異なることを確認
        assert gate._pnl_threshold_buy != gate._pnl_threshold_sell
        # buy 側はプラスの分位点, sell 側はマイナスの分位点
        assert gate._pnl_threshold_buy is not None
        assert gate._pnl_threshold_sell is not None
        assert gate._pnl_threshold_buy > gate._pnl_threshold_sell

    def test_pnl_adaptive_disabled_uses_fixed_threshold(self) -> None:
        """adaptive_threshold=False → 固定 threshold_bps を使用."""
        gate = _make_gate(
            mode="pnl",
            adaptive_threshold=False,
        )
        gate.config.threshold_bps = -2.0
        gate._pipeline.predict.return_value = np.array([-1.5])  # > -2.0 → pass
        features = _make_features()
        decision = gate.evaluate(features, side="sell")
        assert decision.should_skip is False
        assert decision.threshold_used == pytest.approx(-2.0)

    def test_pnl_adaptive_side_none_uses_fixed(self) -> None:
        """side=None + adaptive → 固定 threshold を使用."""
        gate = _make_gate(
            mode="pnl",
            adaptive_threshold=True,
        )
        gate.config.threshold_bps = 0.0
        gate._pipeline.predict.return_value = np.array([-0.5])
        features = _make_features()
        decision = gate.evaluate(features, side=None)
        # side=None → adaptive 不使用 → threshold_bps=0.0
        assert decision.threshold_used == pytest.approx(0.0)
        assert decision.should_skip is True  # -0.5 < 0.0

    def test_pnl_adaptive_step_gradual(self) -> None:
        """step が小さいと閾値は段階的に変化する."""
        gate = _make_gate(
            mode="pnl",
            adaptive_threshold=True,
            adaptive_min_samples=3,
        )
        gate.config.threshold_bps = 0.0
        gate.config.adaptive_step = 0.01  # 非常に小さいステップ
        gate.config.target_skip_rate_sell = 0.20
        features = _make_features()

        # 4件蓄積 (min_samples=3 到達)
        for pnl in [-5.0, -3.0, 1.0, 2.0]:
            gate._pipeline.predict.return_value = np.array([pnl])
            gate.evaluate(features, side="sell")

        # 5件目で較正
        gate._pipeline.predict.return_value = np.array([0.0])
        decision = gate.evaluate(features, side="sell")
        # step=0.01 × 複数回較正 → 0.0 から少し動く（大幅変動なし）
        assert decision.threshold_used is not None
        assert abs(decision.threshold_used - 0.0) <= 0.05  # step 近傍

    def test_pnl_lazy_init_backward_compat(self) -> None:
        """pickle 後方互換: _pred_pnl_history 属性が None でも動作."""
        gate = _make_gate(
            mode="pnl",
            adaptive_threshold=True,
            adaptive_min_samples=3,
        )
        # 属性を削除して pickle 後の状態をシミュレート
        gate._pred_pnl_history_buy = None  # type: ignore[assignment]
        gate._pred_pnl_history_sell = None  # type: ignore[assignment]
        gate._pnl_threshold_buy = None
        gate._pnl_threshold_sell = None

        gate.config.threshold_bps = 0.0
        gate.config.adaptive_step = 100.0
        features = _make_features()

        # 遅延初期化が正常に動作
        for pnl in [-2.0, 0.0, 1.0, 2.0]:
            gate._pipeline.predict.return_value = np.array([pnl])
            decision = gate.evaluate(features, side="buy")
            assert decision is not None
        # 履歴が蓄積されている
        assert gate._pred_pnl_history_buy is not None
        assert len(gate._pred_pnl_history_buy) == 4
