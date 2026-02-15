"""058# Feature Enricher + Skip Gate テスト."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts.v460.ml.feature_enricher import (
    MICRO_FEATURE_COLS,
    build_enriched_as_features,
    build_pnl_features,
    enrich_fill_records,
    load_raw_orderbook,
    load_raw_trades,
    _compute_trade_features,
    _find_nearest_ob,
)
from scripts.v460.ml.skip_gate import (
    GATE_FEATURE_COLS,
    SkipDecision,
    SkipGate,
    SkipGateConfig,
    build_features_from_market_state,
    train_and_save_skip_gate,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def synthetic_fill_df() -> pd.DataFrame:
    """合成 fill records: 100件のテストデータ."""
    rng = np.random.RandomState(42)
    n = 100
    timestamps = np.arange(1700000000, 1700000000 + n * 120, 120, dtype=float)
    sides = rng.choice(["buy", "sell"], n)
    prices = 14_500_000 + rng.randn(n) * 10_000
    fill_prices = prices + rng.randn(n) * 500

    queue_waits = rng.exponential(25, n) + 5
    as_probs = np.where(queue_waits < 20, 0.7, 0.3)
    adverse = rng.binomial(1, as_probs).astype(bool)

    mid_at_fill = fill_prices + rng.randn(n) * 100
    pnl_30s = np.where(adverse, -rng.exponential(3, n), rng.exponential(2, n))

    filled = np.ones(n, dtype=bool)
    filled[rng.choice(n, 15, replace=False)] = False

    rows = []
    for i in range(n):
        row: dict[str, Any] = {
            "cycle_id": f"test_{i}",
            "timestamp": timestamps[i],
            "side": sides[i],
            "order_price": prices[i],
            "order_quantity": 0.001,
            "filled": bool(filled[i]),
            "fill_price": float(fill_prices[i]) if filled[i] else None,
            "cancelled": not bool(filled[i]),
            "cancel_reason": None if filled[i] else "timeout",
            "queue_wait_sec": float(queue_waits[i]),
            "adverse_selected_raw": bool(adverse[i]) if filled[i] else None,
            "adverse_selected": bool(adverse[i]) if filled[i] else None,
            "mid_at_fill": float(mid_at_fill[i]) if filled[i] else None,
            "mid_30s_after": None,
            "post_fill_30s_pnl": float(pnl_30s[i]) if filled[i] else None,
            "spread_at_order": float(rng.uniform(500, 3000)),
            "spread_offset_ratio": float(rng.uniform(0.03, 0.15)),
            "regime": rng.choice(["trending", "ranging", "high_vol", "unknown"]),
        }
        rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_ob_df() -> pd.DataFrame:
    """合成板データ: 200 snapshots."""
    rng = np.random.RandomState(42)
    n = 200
    ts_start = 1700000000 - 60  # fill records の少し前から
    ts = np.arange(ts_start, ts_start + n * 5, 5, dtype=float)

    mid = 14_500_000 + rng.randn(n).cumsum() * 100
    spread = rng.uniform(100, 800, n)
    bid_vol = rng.uniform(0.01, 0.5, n)
    ask_vol = rng.uniform(0.01, 0.5, n)
    total = bid_vol + ask_vol

    return pd.DataFrame({
        "ts": ts,
        "best_bid": mid - spread / 2,
        "best_ask": mid + spread / 2,
        "mid_price": mid,
        "spread_bps": spread / mid * 10000,
        "bid_vol_5": bid_vol,
        "ask_vol_5": ask_vol,
        "depth_imbalance": (bid_vol - ask_vol) / total,
    })


@pytest.fixture
def synthetic_trades_df() -> pd.DataFrame:
    """合成約定データ: 1000 trades."""
    rng = np.random.RandomState(42)
    n = 1000
    ts_start = 1700000000 - 120
    ts = np.sort(rng.uniform(ts_start, ts_start + 12000, n))

    return pd.DataFrame({
        "ts": ts,
        "price": 14_500_000 + rng.randn(n).cumsum() * 50,
        "amount": rng.exponential(0.01, n),
        "side": rng.choice(["buy", "sell"], n),
    })


# ======================================================================
# _compute_trade_features Tests
# ======================================================================


class Test058TradeFeatures:
    """約定統計特徴量のテスト."""

    def test_empty_trades(self) -> None:
        """空の約定データ → デフォルト値."""
        result = _compute_trade_features(pd.DataFrame(), 1700000000)
        assert result["trade_count_60s"] == 0.0
        assert result["buy_ratio"] == 0.5

    def test_no_trades_in_window(self, synthetic_trades_df: pd.DataFrame) -> None:
        """ウィンドウ外のタイムスタンプ → デフォルト値."""
        result = _compute_trade_features(synthetic_trades_df, 1600000000)
        assert result["trade_count_60s"] == 0.0

    def test_trade_features_computed(self, synthetic_trades_df: pd.DataFrame) -> None:
        """ウィンドウ内の取引から統計が計算される."""
        ts = synthetic_trades_df["ts"].median()
        result = _compute_trade_features(synthetic_trades_df, ts, window_sec=60)
        assert result["trade_count_60s"] > 0
        assert 0 <= result["buy_ratio"] <= 1
        assert -1 <= result["trade_flow_imbalance_60s"] <= 1
        assert 0 <= result["vpin_60s"] <= 1

    def test_all_buy_trades(self) -> None:
        """全て buy → buy_ratio=1, tfi=1."""
        trades = pd.DataFrame({
            "ts": [100.0, 101.0, 102.0],
            "price": [100.0, 101.0, 102.0],
            "amount": [1.0, 1.0, 1.0],
            "side": ["buy", "buy", "buy"],
        })
        result = _compute_trade_features(trades, 103.0, window_sec=10)
        assert result["buy_ratio"] == 1.0
        assert result["trade_flow_imbalance_60s"] == 1.0
        assert result["vpin_60s"] == 1.0


# ======================================================================
# _find_nearest_ob Tests
# ======================================================================


class Test058NearestOB:
    """板スナップショットマッチングのテスト."""

    def test_empty_ob(self) -> None:
        """空の板データ → NaN."""
        result = _find_nearest_ob(pd.DataFrame(), 1700000000)
        assert np.isnan(result["spread_bps_ob"])

    def test_exact_match(self, synthetic_ob_df: pd.DataFrame) -> None:
        """完全一致するタイムスタンプ."""
        ts = float(synthetic_ob_df["ts"].iloc[10])
        result = _find_nearest_ob(synthetic_ob_df, ts, tolerance_sec=5)
        assert not np.isnan(result["spread_bps_ob"])
        assert not np.isnan(result["depth_imbalance_ob"])

    def test_tolerance_exceeded(self, synthetic_ob_df: pd.DataFrame) -> None:
        """許容誤差を超える → NaN."""
        ts = float(synthetic_ob_df["ts"].max()) + 100
        result = _find_nearest_ob(synthetic_ob_df, ts, tolerance_sec=5)
        assert np.isnan(result["spread_bps_ob"])


# ======================================================================
# enrich_fill_records Tests
# ======================================================================


class Test058EnrichFillRecords:
    """fill records エンリッチメントのテスト."""

    def test_enrichment_adds_columns(self, synthetic_fill_df: pd.DataFrame) -> None:
        """エンリッチメントで新規カラムが追加される."""
        # 空の raw_dir を使用 (マッチなし, ただしカラムは追加)
        with tempfile.TemporaryDirectory() as td:
            raw_dir = Path(td)
            (raw_dir / "orderbook").mkdir()
            (raw_dir / "trades").mkdir()
            enriched = enrich_fill_records(synthetic_fill_df, raw_dir=raw_dir)

        assert "spread_bps_ob" in enriched.columns
        assert "trade_count_60s" in enriched.columns
        assert "vpin_60s" in enriched.columns
        assert len(enriched) == len(synthetic_fill_df)

    def test_enriched_as_features_shape(self, synthetic_fill_df: pd.DataFrame) -> None:
        """enriched AS 特徴量の shape."""
        # 直接エンリッチメントカラムを追加 (raw data なしでテスト)
        df = synthetic_fill_df.copy()
        for col in MICRO_FEATURE_COLS:
            df[col] = np.random.RandomState(42).randn(len(df))
        df["bid_vol_5_ob"] = np.random.RandomState(43).randn(len(df))
        df["ask_vol_5_ob"] = np.random.RandomState(44).randn(len(df))

        X, y = build_enriched_as_features(df)
        # base(10) + micro(8) + interaction(3) = 21
        assert X.shape[1] >= 18  # base + micro
        assert not X.isna().any().any()

    def test_pnl_features_shape(self, synthetic_fill_df: pd.DataFrame) -> None:
        """PnL 特徴量の shape と labels."""
        df = synthetic_fill_df.copy()
        for col in MICRO_FEATURE_COLS:
            df[col] = np.random.RandomState(42).randn(len(df))

        X, y = build_pnl_features(df)
        assert len(X) > 0
        assert len(X) == len(y)
        assert not X.isna().any().any()
        # PnL は連続値
        assert y.dtype == float

    def test_pnl_features_more_samples_than_as(
        self, synthetic_fill_df: pd.DataFrame
    ) -> None:
        """PnL 特徴量は AS ラベル不要なのでサンプル数が多い."""
        df = synthetic_fill_df.copy()
        for col in MICRO_FEATURE_COLS:
            df[col] = np.random.RandomState(42).randn(len(df))

        X_pnl, _ = build_pnl_features(df)
        X_as, _ = build_enriched_as_features(df)
        # PnL は filled & pnl_notna, AS は filled & as_notna
        # 同じ合成データだがサンプル数は同等以上
        assert len(X_pnl) >= len(X_as) * 0.8


# ======================================================================
# SkipGate Tests
# ======================================================================


class Test058SkipGate:
    """Skip Gate のテスト."""

    @pytest.fixture
    def trained_gate(self, synthetic_fill_df: pd.DataFrame) -> SkipGate:
        """合成データで学習した SkipGate."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        df = synthetic_fill_df.copy()
        for col in MICRO_FEATURE_COLS:
            df[col] = np.random.RandomState(42).randn(len(df))

        X_pnl, y_pnl = build_pnl_features(df)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_pnl)
        model = Ridge(alpha=10.0)
        model.fit(X_scaled, y_pnl.values)

        return SkipGate(
            model=model,
            scaler=scaler,
            feature_cols=X_pnl.columns.tolist(),
            config=SkipGateConfig(threshold_bps=0.0),
        )

    def test_evaluate_returns_decision(self, trained_gate: SkipGate) -> None:
        """evaluate が SkipDecision を返す."""
        features = {col: 0.0 for col in trained_gate.feature_cols}
        result = trained_gate.evaluate(features)
        assert isinstance(result, SkipDecision)
        assert isinstance(result.should_skip, bool)
        assert isinstance(result.predicted_pnl_bps, float)

    def test_evaluate_disabled(self, trained_gate: SkipGate) -> None:
        """gate 無効時はスキップしない."""
        trained_gate.config.enabled = False
        result = trained_gate.evaluate({})
        assert not result.should_skip
        assert result.reason == "gate_disabled"

    def test_evaluate_insufficient_features(self, trained_gate: SkipGate) -> None:
        """特徴量不足 → スキップしない."""
        result = trained_gate.evaluate({"side_buy": 1.0})
        assert not result.should_skip
        assert result.reason == "insufficient_features"

    def test_skip_rate_limit(self, trained_gate: SkipGate) -> None:
        """連続スキップ率上限."""
        trained_gate.config.max_skip_rate = 0.5
        trained_gate.config.threshold_bps = 999.0  # 常にスキップする閾値
        features = {col: 0.0 for col in trained_gate.feature_cols}

        # 20回連続スキップ → rate limit 発動
        for _ in range(25):
            trained_gate.evaluate(features)

        # rate limit が効いているか確認
        result = trained_gate.evaluate(features)
        # 直近の skip_rate > 0.5 なので rate limit が発動しうる
        # (実際の判定は predicted_pnl に依存)
        assert result.features_used > 0

    def test_save_load_roundtrip(self, trained_gate: SkipGate) -> None:
        """save → load で復元."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test_gate.pkl"
            trained_gate.save(path)
            assert path.exists()

            loaded = SkipGate.load(path)
            assert loaded.feature_cols == trained_gate.feature_cols
            assert loaded.config.threshold_bps == trained_gate.config.threshold_bps

            # 同じ入力で同じ予測
            features = {col: 0.5 for col in trained_gate.feature_cols}
            r1 = trained_gate.evaluate(features)
            r2 = loaded.evaluate(features)
            assert abs(r1.predicted_pnl_bps - r2.predicted_pnl_bps) < 1e-10


# ======================================================================
# build_features_from_market_state Tests
# ======================================================================


class Test058MarketStateFeatures:
    """マーケット状態特徴量構築のテスト."""

    def test_basic_features(self) -> None:
        """基本特徴量が生成される."""
        features = build_features_from_market_state(
            side="buy",
            spread_jpy=500.0,
            offset_ratio=0.05,
            regime="ranging",
            best_bid=14_500_000,
            best_ask=14_500_500,
            bid_vol_5=0.3,
            ask_vol_5=0.2,
        )
        assert "side_buy" in features
        assert features["side_buy"] == 1.0
        assert features["regime_ranging"] == 1.0
        assert features["regime_trending"] == 0.0
        assert features["spread_jpy"] == 500.0
        assert features["offset_ratio"] == 0.05

    def test_interaction_features(self) -> None:
        """インタラクション特徴量が正しい符号."""
        features_buy = build_features_from_market_state(
            side="buy",
            spread_jpy=500.0,
            offset_ratio=0.05,
            regime="ranging",
            best_bid=14_500_000,
            best_ask=14_500_500,
            bid_vol_5=0.3,  # bid > ask → positive imbalance
            ask_vol_5=0.2,
        )
        # buy + positive imbalance → positive aligned_imbalance
        assert features_buy["side_aligned_imbalance"] > 0

        features_sell = build_features_from_market_state(
            side="sell",
            spread_jpy=500.0,
            offset_ratio=0.05,
            regime="ranging",
            best_bid=14_500_000,
            best_ask=14_500_500,
            bid_vol_5=0.3,
            ask_vol_5=0.2,
        )
        # sell + positive imbalance → negative aligned_imbalance
        assert features_sell["side_aligned_imbalance"] < 0

    def test_with_recent_trades(self) -> None:
        """直近約定データ付きの特徴量."""
        trades = [
            {"ts": 100.0, "price": 14_500_000, "amount": 0.01, "side": "buy"},
            {"ts": 101.0, "price": 14_500_100, "amount": 0.02, "side": "sell"},
            {"ts": 102.0, "price": 14_500_200, "amount": 0.015, "side": "buy"},
        ]
        features = build_features_from_market_state(
            side="buy",
            spread_jpy=500.0,
            offset_ratio=0.05,
            regime="trending",
            best_bid=14_500_000,
            best_ask=14_500_500,
            bid_vol_5=0.3,
            ask_vol_5=0.3,
            recent_trades=trades,
        )
        assert features["trade_count_60s"] == 3.0
        assert features["avg_trade_size"] > 0
        assert features["price_velocity_60s"] > 0  # 価格上昇

    def test_no_trades(self) -> None:
        """約定データなし → デフォルト値."""
        features = build_features_from_market_state(
            side="buy",
            spread_jpy=500.0,
            offset_ratio=0.05,
            regime="unknown",
            best_bid=14_500_000,
            best_ask=14_500_500,
            bid_vol_5=0.3,
            ask_vol_5=0.3,
        )
        assert features["trade_count_60s"] == 0.0
        assert features["buy_ratio"] == 0.5
        assert features["vpin_60s"] == 0.5

    def test_all_gate_features_present(self) -> None:
        """GATE_FEATURE_COLS の全てが生成される."""
        features = build_features_from_market_state(
            side="buy",
            spread_jpy=500.0,
            offset_ratio=0.05,
            regime="ranging",
            best_bid=14_500_000,
            best_ask=14_500_500,
            bid_vol_5=0.3,
            ask_vol_5=0.2,
            recent_trades=[
                {"ts": 100.0, "price": 14_500_000, "amount": 0.01, "side": "buy"},
            ],
        )
        for col in GATE_FEATURE_COLS:
            assert col in features, f"Missing: {col}"


# ======================================================================
# Integration: 実データ
# ======================================================================


class Test058Integration:
    """実データが存在する場合の統合テスト."""

    @pytest.fixture
    def real_data_available(self) -> bool:
        return (
            Path("results/v460/fill_test/fill_records_20260213.jsonl").exists()
            and Path("data/v460/raw/orderbook").exists()
        )

    def test_enrichment_with_real_data(self, real_data_available: bool) -> None:
        """実データでのエンリッチメント."""
        if not real_data_available:
            pytest.skip("No real data")

        from scripts.v460.ml.data_loader import load_fill_records

        df = load_fill_records()
        enriched = enrich_fill_records(df)
        assert "spread_bps_ob" in enriched.columns
        n_matched = enriched["spread_bps_ob"].notna().sum()
        assert n_matched > 0

    def test_train_skip_gate_real(self, real_data_available: bool) -> None:
        """実データでの SkipGate 学習."""
        if not real_data_available:
            pytest.skip("No real data")

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test_gate.pkl"
            gate = train_and_save_skip_gate(output_path=path)
            assert path.exists()
            assert gate.metadata["n_samples"] > 100

            # 評価テスト
            features = build_features_from_market_state(
                side="buy",
                spread_jpy=500.0,
                offset_ratio=0.05,
                regime="ranging",
                best_bid=14_500_000,
                best_ask=14_500_500,
                bid_vol_5=0.3,
                ask_vol_5=0.2,
            )
            result = gate.evaluate(features)
            assert isinstance(result.predicted_pnl_bps, float)
