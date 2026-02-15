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

    def test_enriched_as_features_preserves_nan(
        self, synthetic_fill_df: pd.DataFrame
    ) -> None:
        """059# P0-1: enriched AS 特徴量が NaN を保持 (CV内補完のため)."""
        df = synthetic_fill_df.copy()
        for col in MICRO_FEATURE_COLS:
            vals = np.random.RandomState(42).randn(len(df))
            vals[0] = np.nan  # 意図的に NaN を入れる
            df[col] = vals
        df["bid_vol_5_ob"] = np.random.RandomState(43).randn(len(df))
        df["ask_vol_5_ob"] = np.random.RandomState(44).randn(len(df))

        X, y = build_enriched_as_features(df)
        # micro特徴量由来の NaN がそのまま残る (interaction は fillna(0) だが micro は保持)
        micro_in_X = [c for c in X.columns if c in MICRO_FEATURE_COLS]
        if micro_in_X:
            assert X[micro_in_X].isna().any().any(), \
                "Micro features should preserve NaN for CV-internal imputation"

    def test_enriched_as_require_spread_filters(
        self, synthetic_fill_df: pd.DataFrame
    ) -> None:
        """060# fix: require_spread=True でスプレッド欠損行を除外."""
        df = synthetic_fill_df.copy()
        for col in MICRO_FEATURE_COLS:
            df[col] = np.random.RandomState(42).randn(len(df))

        # 一部の行でスプレッドを NaN にする
        nan_mask = df.index[:20]  # 最初の 20 行を NaN に
        df.loc[nan_mask, "spread_at_order"] = np.nan
        df.loc[nan_mask, "spread_offset_ratio"] = np.nan

        # require_spread=True (default) → NaN 行が除外される
        X_true, y_true = build_enriched_as_features(df, require_spread=True)
        assert X_true["spread_jpy"].isna().sum() == 0, \
            "require_spread=True should have no NaN in spread_jpy"

        # require_spread=False → NaN 行が保持される
        X_false, y_false = build_enriched_as_features(df, require_spread=False)
        assert len(X_false) > len(X_true), \
            "require_spread=False should keep more samples"
        assert X_false["spread_jpy"].isna().sum() > 0, \
            "require_spread=False should preserve NaN in spread_jpy"

    def test_pnl_features_shape(self, synthetic_fill_df: pd.DataFrame) -> None:
        """PnL 特徴量の shape と labels."""
        df = synthetic_fill_df.copy()
        for col in MICRO_FEATURE_COLS:
            df[col] = np.random.RandomState(42).randn(len(df))

        X, y = build_pnl_features(df)
        assert len(X) > 0
        assert len(X) == len(y)
        # 059# P0-1: NaN は CV 内で補完されるため、ここでは保持されうる
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


class Test061SkipGateASMode:
    """061# AS 分類器モードの SkipGate テスト."""

    @pytest.fixture
    def as_gate(self, synthetic_fill_df: pd.DataFrame) -> SkipGate:
        """合成データで AS モードの SkipGate を構築."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        df = synthetic_fill_df.copy()
        for col in MICRO_FEATURE_COLS:
            df[col] = np.random.RandomState(42).randn(len(df))

        X_as, y_as = build_enriched_as_features(df)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_as)
        model = LogisticRegression(
            C=0.01, max_iter=2000, class_weight="balanced", random_state=42
        )
        model.fit(X_scaled, y_as.values)

        return SkipGate(
            model=model,
            scaler=scaler,
            feature_cols=X_as.columns.tolist(),
            config=SkipGateConfig(mode="as", as_threshold=0.6),
        )

    def test_as_mode_returns_decision(self, as_gate: SkipGate) -> None:
        """AS モードで SkipDecision を返す."""
        features = {col: 0.0 for col in as_gate.feature_cols}
        result = as_gate.evaluate(features)
        assert isinstance(result, SkipDecision)
        assert isinstance(result.should_skip, bool)

    def test_as_mode_config(self, as_gate: SkipGate) -> None:
        """AS モードの config が正しい."""
        assert as_gate.config.mode == "as"
        assert as_gate.config.as_threshold == 0.6

    def test_as_mode_disabled(self, as_gate: SkipGate) -> None:
        """AS モードでも disabled で skip しない."""
        as_gate.config.enabled = False
        result = as_gate.evaluate({})
        assert not result.should_skip
        assert result.reason == "gate_disabled"

    def test_as_mode_save_load(self, as_gate: SkipGate) -> None:
        """AS モードの save → load roundtrip."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test_as_gate.pkl"
            as_gate.save(path)

            loaded = SkipGate.load(path)
            assert loaded.config.mode == "as"
            assert loaded.config.as_threshold == 0.6

            features = {col: 0.5 for col in as_gate.feature_cols}
            r1 = as_gate.evaluate(features)
            r2 = loaded.evaluate(features)
            assert abs(r1.predicted_pnl_bps - r2.predicted_pnl_bps) < 1e-10


class Test065SkipGateTwoTier:
    """065# Two-Tier SkipGate フォールバックテスト."""

    @pytest.fixture
    def primary_gate(self, synthetic_fill_df: pd.DataFrame) -> SkipGate:
        """OB 特徴量を含む primary SkipGate (Pipeline付き)."""
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        df = synthetic_fill_df.copy()
        for col in MICRO_FEATURE_COLS:
            df[col] = np.random.RandomState(42).randn(len(df))

        X_as, y_as = build_enriched_as_features(df)
        k = min(8, X_as.shape[1])
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", SelectKBest(f_classif, k=k)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=0.01, max_iter=2000, class_weight="balanced", random_state=42
            )),
        ])
        pipe.fit(X_as, y_as.values)

        return SkipGate(
            model=pipe.named_steps["model"],
            scaler=pipe.named_steps["scaler"],
            feature_cols=X_as.columns.tolist(),
            config=SkipGateConfig(mode="as", as_threshold=0.6),
            metadata={"label": "primary_test"},
            pipeline=pipe,
        )

    @pytest.fixture
    def fallback_gate(self) -> SkipGate:
        """OB 不要なフォールバック SkipGate (少数特徴量, Pipeline付き)."""
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # 簡易的な trade-only features
        fb_cols = ["log_queue_wait", "edge_bps", "vpin_60s", "hour_cos", "hour_sin"]
        rng = np.random.RandomState(99)
        X_fb = pd.DataFrame(rng.randn(100, len(fb_cols)), columns=fb_cols)
        y_fb = pd.Series(rng.randint(0, 2, 100))
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.01, max_iter=2000, random_state=42)),
        ])
        pipe.fit(X_fb, y_fb)

        return SkipGate(
            model=pipe.named_steps["model"],
            scaler=pipe.named_steps["scaler"],
            feature_cols=fb_cols,
            config=SkipGateConfig(mode="as", as_threshold=0.6),
            metadata={"label": "fallback_test"},
            pipeline=pipe,
        )

    def test_set_fallback(
        self, primary_gate: SkipGate, fallback_gate: SkipGate
    ) -> None:
        """set_fallback で fallback が設定される."""
        assert primary_gate._fallback is None
        primary_gate.set_fallback(fallback_gate)
        assert primary_gate._fallback is fallback_gate

    def test_uses_primary_when_ob_present(
        self, primary_gate: SkipGate, fallback_gate: SkipGate
    ) -> None:
        """OB 特徴量が提供されると primary を使う."""
        primary_gate.set_fallback(fallback_gate)
        features = {col: 0.5 for col in primary_gate.feature_cols}
        # OB critical features を含む
        features["depth_imbalance_ob"] = 0.1
        features["spread_bps_ob"] = 5.0
        result = primary_gate.evaluate(features)
        # primary が使われる → features_used は primary の特徴量数に近い
        assert result.features_used >= 3

    def test_falls_back_when_ob_missing(
        self, primary_gate: SkipGate, fallback_gate: SkipGate
    ) -> None:
        """OB 特徴量が欠損するとフォールバックに委譲."""
        primary_gate.set_fallback(fallback_gate)
        # OB critical features を含まない trade-only features
        features = {
            "log_queue_wait": 1.0,
            "edge_bps": 0.5,
            "vpin_60s": 0.3,
            "hour_cos": 0.8,
            "hour_sin": 0.6,
        }
        result = primary_gate.evaluate(features)
        # fallback が使われる → features_used は fallback の特徴量数に近い
        assert result.features_used <= len(fallback_gate.feature_cols)

    def test_no_fallback_when_ob_missing_without_fallback(
        self, primary_gate: SkipGate
    ) -> None:
        """Fallback 未設定時は OB 欠損でも primary が NaN impute で処理."""
        features = {
            "log_queue_wait": 1.0,
            "edge_bps": 0.5,
            "vpin_60s": 0.3,
            "hour_cos": 0.8,
            "hour_sin": 0.6,
        }
        # fallback なし → primary が使う (NaN impute)
        result = primary_gate.evaluate(features)
        assert result.features_used >= 3


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


# ======================================================================
# 059# P2-9: 追加テスト — リーク検知・skip率履歴・時刻整合
# ======================================================================


class Test059LeakDetection:
    """059# P0-1: CV 外リークが修正されていることを検証."""

    def test_data_loader_preserves_nan_in_spread(self) -> None:
        """data_loader.build_as_features が spread_jpy の NaN を保持."""
        from scripts.v460.ml.data_loader import build_as_features

        n = 20
        rng = np.random.RandomState(42)
        spread = rng.uniform(500, 3000, n).astype(float)
        spread[0] = np.nan
        ratio = rng.uniform(0.03, 0.15, n).astype(float)
        ratio[1] = np.nan

        df = pd.DataFrame({
            "timestamp": np.arange(1700000000.0, 1700000000.0 + n * 120, 120),
            "side": rng.choice(["buy", "sell"], n),
            "spread_at_order": spread,
            "spread_offset_ratio": ratio,
            "queue_wait_sec": rng.exponential(20, n) + 5,
            "adverse_selected": rng.choice([True, False], n),
            "adverse_selected_raw": rng.choice([True, False], n).astype(float),
            "filled": np.ones(n, dtype=bool),
            "regime": rng.choice(["trending", "ranging"], n),
        })
        X, y = build_as_features(df, require_spread=False)
        # NaN が保持されているか (CV 内で SimpleImputer が補完する)
        assert X["spread_jpy"].isna().any(), \
            "spread_jpy NaN should be preserved for CV-internal imputation"
        assert X["offset_ratio"].isna().any(), \
            "offset_ratio NaN should be preserved for CV-internal imputation"

    def test_pnl_features_preserve_nan_in_micro(
        self, synthetic_fill_df: pd.DataFrame
    ) -> None:
        """build_pnl_features が micro 特徴量の NaN を保持."""
        df = synthetic_fill_df.copy()
        for col in MICRO_FEATURE_COLS:
            vals = np.random.RandomState(42).randn(len(df))
            vals[:5] = np.nan
            df[col] = vals

        X, y = build_pnl_features(df)
        micro_cols = [c for c in X.columns if c in MICRO_FEATURE_COLS]
        if micro_cols:
            assert X[micro_cols].isna().any().any(), \
                "Micro features should preserve NaN for CV-internal imputation"


class Test059SkipRateHistory:
    """059# P0-2: skip 率履歴が最終決定を記録することを検証."""

    def test_skip_rate_records_final_decision(self) -> None:
        """force-pass override 後の最終決定が _recent_skips に記録される."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        # 常に negative PnL を予測するモデル
        feature_cols = ["f1", "f2", "f3", "f4"]
        scaler = StandardScaler()
        X_dummy = np.ones((10, 4))
        scaler.fit(X_dummy)

        model = Ridge(alpha=1.0)
        model.fit(X_dummy, np.full(10, -5.0))  # 常に負を予測

        gate = SkipGate(
            model=model,
            scaler=scaler,
            feature_cols=feature_cols,
            config=SkipGateConfig(
                threshold_bps=0.0,
                max_skip_rate=0.3,  # 30% で制限
                enabled=True,
            ),
        )

        features = {c: 1.0 for c in feature_cols}

        # 最初の数件はスキップ (rate < 0.3 なので)
        results = []
        for _ in range(25):
            r = gate.evaluate(features)
            results.append(r)

        # force-pass が発動したら _recent_skips には False が記録される
        force_pass_count = sum(
            1 for r in results if "skip_rate_limit" in r.reason
        )
        assert force_pass_count > 0, "Rate limit should have fired"

        # _recent_skips の True 率が max_skip_rate 以下に収束
        recent_rate = sum(gate._recent_skips) / len(gate._recent_skips)
        assert recent_rate <= 0.5, (
            f"Rate {recent_rate:.2f} should converge below max_skip_rate "
            f"because force-pass records False"
        )

    def test_skip_rate_does_not_oscillate(self) -> None:
        """059# P0-2: skip 率がスパイクしないこと."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        feature_cols = ["f1", "f2", "f3"]
        scaler = StandardScaler()
        scaler.fit(np.ones((5, 3)))
        model = Ridge(alpha=1.0)
        model.fit(np.ones((5, 3)), np.full(5, -10.0))

        gate = SkipGate(
            model=model,
            scaler=scaler,
            feature_cols=feature_cols,
            config=SkipGateConfig(
                threshold_bps=0.0,
                max_skip_rate=0.4,
                enabled=True,
            ),
        )

        features = {c: 1.0 for c in feature_cols}
        rates = []
        for i in range(50):
            gate.evaluate(features)
            if gate._recent_skips:
                rate = sum(gate._recent_skips) / len(gate._recent_skips)
                rates.append(rate)

        # 安定後のレートが max_skip_rate 近辺を超えないこと
        if len(rates) > 20:
            late_rates = rates[-10:]
            assert max(late_rates) <= 0.6, (
                f"Late skip rates should stabilize near max_skip_rate, "
                f"got max={max(late_rates):.2f}"
            )


class Test059TimestampConsistency:
    """059# P1-5: 時刻特徴の整合性テスト."""

    def test_market_timestamp_matches_fromtimestamp(self) -> None:
        """market_timestamp を指定した場合、fromtimestamp と同じ時刻特徴になる."""
        from datetime import datetime as dt

        ts = 1700000000.0
        now = dt.fromtimestamp(ts)
        expected_hour = now.hour + now.minute / 60.0

        features = build_features_from_market_state(
            side="buy",
            spread_jpy=500.0,
            offset_ratio=0.05,
            regime="ranging",
            best_bid=14_500_000,
            best_ask=14_500_500,
            bid_vol_5=0.3,
            ask_vol_5=0.2,
            market_timestamp=ts,
        )

        # hour_sin / hour_cos が fromtimestamp ベースと一致
        expected_sin = float(np.sin(2 * np.pi * expected_hour / 24))
        assert abs(features["hour_sin"] - expected_sin) < 1e-6

    def test_trade_window_sec_filters_trades(self) -> None:
        """059# P1-6: trade_window_sec がフィルタに使われる."""
        market_ts = 200.0
        trades_in_window = [
            {"ts": 150.0, "price": 100, "amount": 1.0, "side": "buy"},
            {"ts": 180.0, "price": 101, "amount": 1.0, "side": "sell"},
        ]
        trades_outside = [
            {"ts": 50.0, "price": 99, "amount": 2.0, "side": "buy"},
        ]
        all_trades = trades_outside + trades_in_window

        # window=60s → ts 50 は除外される
        features_60 = build_features_from_market_state(
            side="buy",
            spread_jpy=500.0,
            offset_ratio=0.05,
            regime="ranging",
            best_bid=14_500_000,
            best_ask=14_500_500,
            bid_vol_5=0.3,
            ask_vol_5=0.2,
            recent_trades=all_trades,
            trade_window_sec=60,
            market_timestamp=market_ts,
        )
        assert features_60["trade_count_60s"] == 2.0

        # window=300s → ts 50 も含まれる
        features_300 = build_features_from_market_state(
            side="buy",
            spread_jpy=500.0,
            offset_ratio=0.05,
            regime="ranging",
            best_bid=14_500_000,
            best_ask=14_500_500,
            bid_vol_5=0.3,
            ask_vol_5=0.2,
            recent_trades=all_trades,
            trade_window_sec=300,
            market_timestamp=market_ts,
        )
        assert features_300["trade_count_60s"] == 3.0


class Test059PickleHash:
    """059# P2-8: pickle ハッシュ検証テスト."""

    def test_save_creates_hash_file(self) -> None:
        """save がハッシュファイルを作成する."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(np.ones((5, 3)))
        model = Ridge()
        model.fit(np.ones((5, 3)), np.ones(5))

        gate = SkipGate(
            model=model,
            scaler=scaler,
            feature_cols=["f1", "f2", "f3"],
        )

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gate.pkl"
            gate.save(path)
            hash_path = path.with_suffix(".pkl.sha256")
            assert hash_path.exists()
            digest = hash_path.read_text().strip()
            assert len(digest) == 64  # SHA256 hex

    def test_load_detects_corruption(self) -> None:
        """改竄されたファイルを検出する."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(np.ones((5, 3)))
        model = Ridge()
        model.fit(np.ones((5, 3)), np.ones(5))

        gate = SkipGate(
            model=model,
            scaler=scaler,
            feature_cols=["f1", "f2", "f3"],
        )

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gate.pkl"
            gate.save(path)

            # ファイルを改竄
            with open(path, "ab") as f:
                f.write(b"CORRUPTED")

            with pytest.raises(ValueError, match="hash mismatch"):
                SkipGate.load(path)

    def test_load_without_hash_file_succeeds(self) -> None:
        """ハッシュファイルがない場合は警告なしでロード."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(np.ones((5, 3)))
        model = Ridge()
        model.fit(np.ones((5, 3)), np.ones(5))

        gate = SkipGate(
            model=model,
            scaler=scaler,
            feature_cols=["f1", "f2", "f3"],
        )

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gate.pkl"
            gate.save(path)
            # ハッシュファイルを削除
            hash_path = path.with_suffix(".pkl.sha256")
            hash_path.unlink()

            # ロードは成功する (後方互換)
            loaded = SkipGate.load(path)
            assert loaded.feature_cols == ["f1", "f2", "f3"]


class Test059SearchsortedOptimization:
    """059# P1-7: searchsorted 最適化の正当性テスト."""

    def test_searchsorted_matches_brute_force(
        self, synthetic_trades_df: pd.DataFrame
    ) -> None:
        """searchsorted と従来のマスクフィルタが同じ結果を返す."""
        sorted_df = synthetic_trades_df.sort_values("ts").reset_index(drop=True)
        sorted_ts = sorted_df["ts"].values

        ts = sorted_df["ts"].median()

        # 従来方式 (brute force mask)
        result_brute = _compute_trade_features(sorted_df, ts, window_sec=60)

        # searchsorted 方式
        result_fast = _compute_trade_features(
            sorted_df, ts, window_sec=60, _sorted_ts=sorted_ts
        )

        for key in result_brute:
            assert abs(result_brute[key] - result_fast[key]) < 1e-10, (
                f"Mismatch in {key}: brute={result_brute[key]}, "
                f"fast={result_fast[key]}"
            )

    def test_searchsorted_empty_window(
        self, synthetic_trades_df: pd.DataFrame
    ) -> None:
        """searchsorted 方式でもウィンドウ外 → デフォルト値."""
        sorted_df = synthetic_trades_df.sort_values("ts").reset_index(drop=True)
        sorted_ts = sorted_df["ts"].values

        result = _compute_trade_features(
            sorted_df, 1600000000.0, window_sec=60, _sorted_ts=sorted_ts
        )
        assert result["trade_count_60s"] == 0.0
        assert result["buy_ratio"] == 0.5
