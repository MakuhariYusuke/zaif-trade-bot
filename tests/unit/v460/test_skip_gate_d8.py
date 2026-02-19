"""
D8: SkipGate evaluate / warm_start テストカバレッジ強化 — 122#.

118# D8 で指摘された evaluate/warm_start のテスト不足を補完:
- A3: side 別無効化 (buy_enabled/sell_enabled)
- A2: warm_start threshold 即収束 (quantile 直接計算)
- build_features_from_market_state() の基本テスト
- save/load round-trip
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from scripts.v460.ml.skip_gate import (
    SkipGate,
    SkipGateConfig,
    SkipDecision,
    build_features_from_market_state,
    warm_start_skip_gate_thresholds,
)


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
    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = np.array(
        [[1.0 - predict_prob, predict_prob]]
    )
    mock_pipeline.predict.return_value = np.array([0.5])  # PnL mode fallback
    gate = SkipGate(
        model=MagicMock(),
        scaler=MagicMock(),
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

    def _write_fill_records(
        self, tmpdir: Path, n_buy: int = 30, n_sell: int = 30,
        buy_probs: list[float] | None = None,
        sell_probs: list[float] | None = None,
    ) -> None:
        """テスト用 fill_records_*.jsonl を書き出す."""
        records: list[str] = []
        buy_p = buy_probs or [0.45 + i * 0.005 for i in range(n_buy)]
        sell_p = sell_probs or [0.50 + i * 0.005 for i in range(n_sell)]
        for i, p in enumerate(buy_p):
            records.append(json.dumps({
                "cycle_id": f"buy_{i:03d}",
                "side": "buy",
                "skip_gate_as_prob": p,
                "filled": True,
                "timestamp": 1770975573.0 + i * 120,
            }))
        for i, p in enumerate(sell_p):
            records.append(json.dumps({
                "cycle_id": f"sell_{i:03d}",
                "side": "sell",
                "skip_gate_as_prob": p,
                "filled": True,
                "timestamp": 1770975573.0 + (n_buy + i) * 120,
            }))
        (tmpdir / "fill_records_20260220.jsonl").write_text(
            "\n".join(records), encoding="utf-8",
        )

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
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            self._write_fill_records(
                tmpdir_path, n_buy=30, n_sell=30,
                buy_probs=[0.42 + i * 0.005 for i in range(30)],
                sell_probs=[0.48 + i * 0.005 for i in range(30)],
            )
            warm_start_skip_gate_thresholds(gate, tmpdir_path, window=50)

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
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            self._write_fill_records(tmpdir_path, n_buy=25, n_sell=25)
            warm_start_skip_gate_thresholds(gate, tmpdir_path, window=50)

            assert len(gate._pas_history_buy) == 25
            assert len(gate._pas_history_sell) == 25

    def test_warm_start_window_truncation(self) -> None:
        """window パラメータがヒストリーサイズを制限."""
        gate = _make_gate(adaptive_threshold=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            self._write_fill_records(tmpdir_path, n_buy=50, n_sell=50)
            warm_start_skip_gate_thresholds(gate, tmpdir_path, window=20)

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
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            self._write_fill_records(tmpdir_path, n_buy=10, n_sell=10)
            warm_start_skip_gate_thresholds(gate, tmpdir_path, window=50)

            # 10 < 50 → 閾値は変更されない
            assert gate.config.as_threshold_buy == 0.65
            assert gate.config.as_threshold_sell == 0.65

    def test_warm_start_empty_directory(self) -> None:
        """空ディレクトリでもクラッシュしない."""
        gate = _make_gate(adaptive_threshold=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            warm_start_skip_gate_thresholds(gate, Path(tmpdir), window=50)
            assert gate._pas_history_buy == []
            assert gate._pas_history_sell == []

    def test_warm_start_non_adaptive_is_noop(self) -> None:
        """adaptive_threshold=False では閾値変更なし."""
        gate = _make_gate(
            adaptive_threshold=False,
            as_threshold_buy=0.65,
            as_threshold_sell=0.65,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            self._write_fill_records(tmpdir_path, n_buy=30, n_sell=30)
            warm_start_skip_gate_thresholds(gate, tmpdir_path, window=50)

            # 履歴は復元されるが閾値は変更されない
            assert len(gate._pas_history_buy) > 0
            assert gate.config.as_threshold_buy == 0.65


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
        import time
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
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler as SkScaler

        config = SkipGateConfig(
            as_threshold_buy=kwargs.get("as_threshold_buy", 0.55),  # type: ignore[arg-type]
            sell_enabled=kwargs.get("sell_enabled", False),  # type: ignore[arg-type]
        )
        model = SGDClassifier()
        scaler = SkScaler()
        feature_cols = ["spread_jpy", "offset_ratio", "regime_trending"]
        pipeline = Pipeline([("scaler", SkScaler()), ("clf", SGDClassifier())])
        return SkipGate(
            model=model, scaler=scaler, feature_cols=feature_cols,
            config=config, pipeline=pipeline,
        )

    def test_save_load_roundtrip(self) -> None:
        """save → load で config/feature_cols が復元される."""
        gate = self._make_picklable_gate(as_threshold_buy=0.55, sell_enabled=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_gate.pkl"
            gate.save(path)
            loaded = SkipGate.load(path)

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
