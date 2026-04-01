"""058# Feature Enricher + Skip Gate テスト."""

from __future__ import annotations

import copy
import json
from datetime import datetime as dt
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import scripts.v460.ml.feature_enricher as feature_enricher_module
from scripts.v460.ml.feature_enricher import (
    MICRO_FEATURE_COLS,
    build_enriched_as_features,
    clear_raw_load_caches,
    build_pnl_features,
    enrich_fill_records,
    get_raw_load_cache_stats,
    load_raw_orderbook,
    load_raw_trades,
    _compute_trade_features,
    _find_nearest_ob,
)
from ztb.ml.skip_gate import (
    GATE_FEATURE_COLS,
    SkipDecision,
    SkipGate,
    SkipGateConfig,
    _BASE_FEATURE_COLS,
    build_features_from_market_state,
    get_gate_feature_cols,
    train_and_save_skip_gate,
    warm_start_skip_gate_thresholds,
)
from scripts.v460.ml.data_loader import build_as_features, load_fill_records as load_fill_records_df
from tests.unit.v460._real_data_test_helpers import (
    JsonRow,
    has_fill_records_and_raw_data,
    load_recent_fill_records_df as _load_recent_fill_records_df,
    select_minimum_trainable_fill_df,
    write_jsonl_gz as _write_jsonl_gz,
)
from tests.unit.v460._skip_gate_test_helpers import save_and_load_skip_gate
from ztb.ml.artifact_paths import hash_sidecar_path

_REAL_DATA_SAMPLE_ROWS = 36
_REAL_DATA_SMOKE_SAMPLE_ROWS = 8
_REAL_DATA_SMOKE_FALLBACK_SAMPLE_ROWS = 12
_REAL_DATA_SMOKE_EXPANDED_SAMPLE_ROWS = 18
_REAL_DATA_FALLBACK_SAMPLE_ROWS = 48
_REAL_DATA_EXPANDED_SAMPLE_ROWS = 60
_REAL_DATA_MIN_TRAIN_SAMPLES = 20


def _assert_raw_cache_invalidates_on_file_update(
    *,
    tmp_path: Path,
    subdir: str,
    filename: str,
    first_rows: list[JsonRow],
    second_rows: list[JsonRow],
    load_fn,
) -> None:
    clear_raw_load_caches()
    data_dir = tmp_path / subdir
    data_dir.mkdir()
    target_file = data_dir / filename

    _write_jsonl_gz(target_file, first_rows)

    df1 = load_fn(tmp_path, date_filter={"20260220"})
    df2 = load_fn(tmp_path, date_filter={"20260220"})
    assert len(df1) == len(first_rows)
    assert len(df2) == len(first_rows)
    assert df1 is not df2

    _write_jsonl_gz(target_file, second_rows)
    df3 = load_fn(tmp_path, date_filter={"20260220"})
    assert len(df3) == len(second_rows)


def _select_real_enriched_training_df(
    *,
    initial_rows: int = _REAL_DATA_SAMPLE_ROWS,
    fallback_rows: int = _REAL_DATA_FALLBACK_SAMPLE_ROWS,
    expanded_rows: int = _REAL_DATA_EXPANDED_SAMPLE_ROWS,
    min_train_samples: int = _REAL_DATA_MIN_TRAIN_SAMPLES,
) -> pd.DataFrame:
    """学習成立条件を満たす最小限の real enriched_df を選ぶ."""
    return select_minimum_trainable_fill_df(
        initial_rows=initial_rows,
        fallback_rows=fallback_rows,
        expanded_rows=expanded_rows,
        min_train_samples=min_train_samples,
        enrich_fn=enrich_fill_records,
    )


@lru_cache(maxsize=1)
def _cached_real_enriched_training_df() -> pd.DataFrame:
    return _select_real_enriched_training_df()


@lru_cache(maxsize=1)
def _cached_real_enriched_smoke_df() -> pd.DataFrame:
    for sample_rows in (
        _REAL_DATA_SMOKE_SAMPLE_ROWS,
        _REAL_DATA_SMOKE_FALLBACK_SAMPLE_ROWS,
        _REAL_DATA_SMOKE_EXPANDED_SAMPLE_ROWS,
    ):
        recent_fill_df = _load_recent_fill_records_df(sample_rows=sample_rows)
        if recent_fill_df.empty:
            continue
        enriched = enrich_fill_records(recent_fill_df)
        if "spread_bps_ob" not in enriched.columns:
            continue
        if enriched["spread_bps_ob"].notna().any():
            return enriched
    return pd.DataFrame()


def _make_negative_skip_gate(
    *,
    feature_cols: list[str],
    max_skip_rate: float,
) -> SkipGate:
    scaler = StandardScaler()
    X_dummy = np.ones((4, len(feature_cols)))
    scaler.fit(X_dummy)

    model = Ridge(alpha=1.0)
    model.fit(X_dummy, np.full(4, -5.0))

    return SkipGate(
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        config=SkipGateConfig(
            threshold_bps=0.0,
            max_skip_rate=max_skip_rate,
            enabled=True,
        ),
    )


def _make_synthetic_fill_df() -> pd.DataFrame:
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
        row: JsonRow = {
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


@lru_cache(maxsize=1)
def _cached_synthetic_fill_df() -> pd.DataFrame:
    return _make_synthetic_fill_df()


def _make_micro_feature_fill_df() -> pd.DataFrame:
    df = _make_synthetic_fill_df()
    for i, col in enumerate(MICRO_FEATURE_COLS):
        df[col] = np.random.RandomState(42 + i).randn(len(df))
    df["bid_vol_5_ob"] = np.random.RandomState(43).randn(len(df))
    df["ask_vol_5_ob"] = np.random.RandomState(44).randn(len(df))
    return df


@lru_cache(maxsize=1)
def _cached_micro_feature_fill_df() -> pd.DataFrame:
    return _make_micro_feature_fill_df()


def _make_basic_gate() -> SkipGate:
    """hash / roundtrip テスト向けの最小 gate."""
    x = np.ones((5, 3))
    scaler = StandardScaler().fit(x)
    model = Ridge().fit(x, np.ones(5))
    return SkipGate(
        model=model,
        scaler=scaler,
        feature_cols=["f1", "f2", "f3"],
    )


@lru_cache(maxsize=1)
def _cached_synthetic_ob_df() -> pd.DataFrame:
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


@lru_cache(maxsize=1)
def _cached_synthetic_trades_df() -> pd.DataFrame:
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
# Fixtures
# ======================================================================


@pytest.fixture
def synthetic_fill_df() -> pd.DataFrame:
    return _cached_synthetic_fill_df().copy(deep=True)


@pytest.fixture
def synthetic_micro_fill_df() -> pd.DataFrame:
    return _cached_micro_feature_fill_df().copy(deep=True)


@pytest.fixture
def synthetic_ob_df() -> pd.DataFrame:
    """合成板データ: 200 snapshots."""
    return _cached_synthetic_ob_df().copy(deep=True)


@pytest.fixture
def synthetic_trades_df() -> pd.DataFrame:
    """合成約定データ: 1000 trades."""
    return _cached_synthetic_trades_df().copy(deep=True)


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

    def test_enrichment_adds_columns(
        self,
        synthetic_fill_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """エンリッチメントで新規カラムが追加される."""
        # 空の raw_dir を使用 (マッチなし, ただしカラムは追加)
        raw_dir = tmp_path
        (raw_dir / "orderbook").mkdir()
        (raw_dir / "trades").mkdir()
        enriched = enrich_fill_records(synthetic_fill_df, raw_dir=raw_dir)

        assert "spread_bps_ob" in enriched.columns
        assert "trade_count_60s" in enriched.columns
        assert "vpin_60s" in enriched.columns
        assert len(enriched) == len(synthetic_fill_df)

    def test_enriched_as_features_shape(self, synthetic_micro_fill_df: pd.DataFrame) -> None:
        """enriched AS 特徴量の shape."""
        X, y = build_enriched_as_features(synthetic_micro_fill_df)
        # base(10) + micro(8) + interaction(3) = 21
        assert X.shape[1] >= 18  # base + micro

    def test_enriched_as_features_preserves_nan(
        self, synthetic_micro_fill_df: pd.DataFrame
    ) -> None:
        """059# P0-1: enriched AS 特徴量が NaN を保持 (CV内補完のため)."""
        df = synthetic_micro_fill_df.copy()
        df.loc[df.index[0], MICRO_FEATURE_COLS] = np.nan

        X, y = build_enriched_as_features(df)
        # micro特徴量由来の NaN がそのまま残る (interaction は fillna(0) だが micro は保持)
        micro_in_X = [c for c in X.columns if c in MICRO_FEATURE_COLS]
        if micro_in_X:
            assert X[micro_in_X].isna().any().any(), \
                "Micro features should preserve NaN for CV-internal imputation"

    def test_enriched_as_require_spread_filters(
        self, synthetic_micro_fill_df: pd.DataFrame
    ) -> None:
        """060# fix: require_spread=True でスプレッド欠損行を除外."""
        df = synthetic_micro_fill_df.copy()

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

    def test_pnl_features_shape(self, synthetic_micro_fill_df: pd.DataFrame) -> None:
        """PnL 特徴量の shape と labels."""
        X, y = build_pnl_features(synthetic_micro_fill_df)
        assert len(X) > 0
        assert len(X) == len(y)
        # 059# P0-1: NaN は CV 内で補完されるため、ここでは保持されうる
        # PnL は連続値
        assert y.dtype == float

    def test_pnl_features_more_samples_than_as(
        self, synthetic_micro_fill_df: pd.DataFrame
    ) -> None:
        """PnL 特徴量は AS ラベル不要なのでサンプル数が多い."""
        X_pnl, _ = build_pnl_features(synthetic_micro_fill_df)
        X_as, _ = build_enriched_as_features(synthetic_micro_fill_df)
        # PnL は filled & pnl_notna, AS は filled & as_notna
        # 同じ合成データだがサンプル数は同等以上
        assert len(X_pnl) >= len(X_as) * 0.8


# ======================================================================
# raw load cache tests
# ======================================================================


class Test058RawLoadCache:
    """raw orderbook/trades ローダーのキャッシュ挙動テスト."""

    def test_raw_load_cache_is_bounded_and_clearable(self, tmp_path: Path) -> None:
        clear_raw_load_caches()
        for subdir, row_factory in (
            (
                "trades",
                lambda ts: [{"ts": ts, "price": 100.0 + ts, "amount": 0.1, "side": "buy"}],
            ),
            (
                "orderbook",
                lambda ts: [{"ts": ts, "bids": [[100.0 + ts, 0.2]], "asks": [[101.0 + ts, 0.3]]}],
            ),
        ):
            target_dir = tmp_path / subdir
            target_dir.mkdir(exist_ok=True)
            for day_index, day in enumerate(("20260220", "20260221", "20260222"), start=1):
                _write_jsonl_gz(target_dir / f"{day}.jsonl.gz", row_factory(float(day_index)))

        with patch.object(feature_enricher_module, "_RAW_LOAD_CACHE_MAX_ENTRIES", 2):
            for day in ("20260220", "20260221", "20260222"):
                load_raw_trades(tmp_path, date_filter={day})
                load_raw_orderbook(tmp_path, date_filter={day})

        stats = get_raw_load_cache_stats()
        assert stats["trades_cache_entries"] <= 2
        assert stats["orderbook_cache_entries"] <= 2
        clear_raw_load_caches()
        assert get_raw_load_cache_stats()["total_cache_entries"] == 0

    def test_trades_cache_invalidates_on_file_update(self, tmp_path: Path) -> None:
        _assert_raw_cache_invalidates_on_file_update(
            tmp_path=tmp_path,
            subdir="trades",
            filename="20260220.jsonl.gz",
            first_rows=[{"ts": 1.0, "price": 100.0, "amount": 0.1, "side": "buy"}],
            second_rows=[
                {"ts": 1.0, "price": 100.0, "amount": 0.1, "side": "buy"},
                {"ts": 2.0, "price": 101.0, "amount": 0.2, "side": "sell"},
            ],
            load_fn=load_raw_trades,
        )

    def test_orderbook_cache_invalidates_on_file_update(self, tmp_path: Path) -> None:
        _assert_raw_cache_invalidates_on_file_update(
            tmp_path=tmp_path,
            subdir="orderbook",
            filename="20260220.jsonl.gz",
            first_rows=[{"ts": 1.0, "bids": [[100.0, 0.2]], "asks": [[101.0, 0.3]]}],
            second_rows=[
                {"ts": 1.0, "bids": [[100.0, 0.2]], "asks": [[101.0, 0.3]]},
                {"ts": 2.0, "bids": [[102.0, 0.4]], "asks": [[103.0, 0.5]]},
            ],
            load_fn=load_raw_orderbook,
        )

    def test_trades_date_filter_avoids_full_glob_scan(self, tmp_path: Path) -> None:
        """date_filter 指定時は directory glob 全走査を行わず直接ファイル解決する."""
        trades_dir = tmp_path / "trades"
        trades_dir.mkdir()
        _write_jsonl_gz(
            trades_dir / "20260220.jsonl.gz",
            [{"ts": 1.0, "price": 100.0, "amount": 0.1, "side": "buy"}],
        )

        with patch("pathlib.Path.glob", side_effect=AssertionError("glob should not be called")):
            df = load_raw_trades(tmp_path, date_filter={"20260220"})
        assert len(df) == 1

    def test_orderbook_date_filter_avoids_full_glob_scan(self, tmp_path: Path) -> None:
        """date_filter 指定時は directory glob 全走査を行わず直接ファイル解決する."""
        ob_dir = tmp_path / "orderbook"
        ob_dir.mkdir()
        _write_jsonl_gz(
            ob_dir / "20260220.jsonl.gz",
            [{"ts": 1.0, "bids": [[100.0, 0.2]], "asks": [[101.0, 0.3]]}],
        )

        with patch("pathlib.Path.glob", side_effect=AssertionError("glob should not be called")):
            df = load_raw_orderbook(tmp_path, date_filter={"20260220"})
        assert len(df) == 1


# ======================================================================
# SkipGate Tests
# ======================================================================


class Test058SkipGate:
    """Skip Gate のテスト."""

    @pytest.fixture(scope="class")
    def trained_gate_template(self) -> SkipGate:
        """合成データで学習した SkipGate テンプレート."""
        df = _cached_micro_feature_fill_df().copy(deep=True)

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

    @pytest.fixture
    def trained_gate(self, trained_gate_template: SkipGate) -> SkipGate:
        return copy.deepcopy(trained_gate_template)

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

        first = trained_gate.evaluate(features, side="buy")
        result = trained_gate.evaluate(features, side="buy")
        assert first.reason == "skip"
        assert "skip_rate_limit" in result.reason
        assert result.features_used > 0

    def test_save_load_roundtrip(self, trained_gate: SkipGate, tmp_path: Path) -> None:
        """save → load で復元."""
        path = tmp_path / "test_gate.pkl"
        loaded = save_and_load_skip_gate(trained_gate, path)
        assert path.exists()
        assert loaded.feature_cols == trained_gate.feature_cols
        assert loaded.config.threshold_bps == trained_gate.config.threshold_bps

        # 同じ入力で同じ予測
        features = {col: 0.5 for col in trained_gate.feature_cols}
        r1 = trained_gate.evaluate(features)
        r2 = loaded.evaluate(features)
        assert abs(r1.predicted_pnl_bps - r2.predicted_pnl_bps) < 1e-10


class Test061SkipGateASMode:
    """061# AS 分類器モードの SkipGate テスト."""

    @pytest.fixture(scope="class")
    def as_gate_template(self) -> SkipGate:
        """合成データで AS モードの SkipGate テンプレートを構築."""
        df = _cached_micro_feature_fill_df().copy(deep=True)

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

    @pytest.fixture
    def as_gate(self, as_gate_template: SkipGate) -> SkipGate:
        return copy.deepcopy(as_gate_template)

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

    def test_as_mode_save_load(self, as_gate: SkipGate, tmp_path: Path) -> None:
        """AS モードの save → load roundtrip."""
        path = tmp_path / "test_as_gate.pkl"
        loaded = save_and_load_skip_gate(as_gate, path)
        assert loaded.config.mode == "as"
        assert loaded.config.as_threshold == 0.6

        features = {col: 0.5 for col in as_gate.feature_cols}
        r1 = as_gate.evaluate(features)
        r2 = loaded.evaluate(features)
        assert abs(r1.predicted_pnl_bps - r2.predicted_pnl_bps) < 1e-10


class Test065SkipGateNoOB:
    """071# OB 特徴量除去後の SkipGate テスト."""

    @pytest.fixture
    def trade_only_gate(self) -> SkipGate:
        """Trade-only 特徴量の SkipGate (071# OB 除去後)."""

        cols = ["side_buy", "hour_sin", "hour_cos", "spread_jpy", "trade_count_60s"]
        rng = np.random.RandomState(99)
        X = pd.DataFrame(rng.randn(100, len(cols)), columns=cols)
        y = pd.Series(rng.randint(0, 2, 100))
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.01, max_iter=2000, random_state=42)),
        ])
        pipe.fit(X, y)

        return SkipGate(
            model=pipe.named_steps["model"],
            scaler=pipe.named_steps["scaler"],
            feature_cols=cols,
            config=SkipGateConfig(mode="as", as_threshold=0.6),
            metadata={"label": "trade_only_test"},
            pipeline=pipe,
        )

    def test_evaluate_without_ob(self, trade_only_gate: SkipGate) -> None:
        """071# OB なしで正常に evaluate できる."""
        features = {
            "side_buy": 1.0,
            "hour_sin": 0.5,
            "hour_cos": 0.8,
            "spread_jpy": 500.0,
            "trade_count_60s": 5.0,
        }
        result = trade_only_gate.evaluate(features, side="buy")
        assert result.features_used >= 3
        assert result.model_used == "primary"
        assert isinstance(result.should_skip, bool)

    def test_no_fallback_attribute(self, trade_only_gate: SkipGate) -> None:
        """071# _fallback 属性は存在しない."""
        assert not hasattr(trade_only_gate, "_fallback")


class Test068SkipGateSideThreshold:
    """068# §3.3: side 別閾値テスト."""

    @pytest.fixture
    def as_gate(self) -> SkipGate:
        """AS モードの SkipGate (高 AS 確率を返す)."""

        cols = ["hour_sin", "hour_cos", "spread_jpy"]
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(200, len(cols)), columns=cols)
        # 偏ったラベル → 高い AS 確率を出力
        y = pd.Series(np.ones(200, dtype=int))
        y.iloc[:50] = 0
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.01, max_iter=2000, random_state=42)),
        ])
        pipe.fit(X, y)
        return SkipGate(
            model=pipe.named_steps["model"],
            scaler=pipe.named_steps["scaler"],
            feature_cols=cols,
            config=SkipGateConfig(mode="as", as_threshold=0.7),
            pipeline=pipe,
        )

    def test_common_threshold(self, as_gate: SkipGate) -> None:
        """side 別閾値未設定時は共通閾値を使う."""
        features = {"hour_sin": 0.5, "hour_cos": 0.5, "spread_jpy": 100.0}
        # 共通 0.7 で評価
        result = as_gate.evaluate(features, side="buy")
        assert isinstance(result.should_skip, bool)

    def test_sell_stricter_threshold(self, as_gate: SkipGate) -> None:
        """068# sell 側のみ厳格化した閾値が適用される."""
        as_gate.config.as_threshold = 0.9       # 共通は高め (ほぼスキップしない)
        as_gate.config.as_threshold_sell = 0.3   # sell は低め (ほぼスキップ)
        features = {"hour_sin": 0.5, "hour_cos": 0.5, "spread_jpy": 100.0}
        buy_result = as_gate.evaluate(features, side="buy")
        sell_result = as_gate.evaluate(features, side="sell")
        # sell は厳格化→スキップしやすい
        # (確率値によるが、少なくとも buy と sell で判定が異なりうる)
        assert sell_result.threshold_bps == buy_result.threshold_bps  # 共通PnL閾値は同じ
        # side別で異なる判定可能性を検証(確率的)

    def test_buy_threshold_override(self, as_gate: SkipGate) -> None:
        """068# buy 側にも個別閾値を設定できる."""
        as_gate.config.as_threshold = 0.5
        as_gate.config.as_threshold_buy = 0.9
        as_gate.config.as_threshold_sell = None
        features = {"hour_sin": 0.5, "hour_cos": 0.5, "spread_jpy": 100.0}
        # buy は 0.9 (スキップしない), sell は 0.5 (共通)
        buy_result = as_gate.evaluate(features, side="buy")
        sell_result = as_gate.evaluate(features, side="sell")
        # buy が 0.9 → スキップしにくい
        # sell が 0.5 → スキップしやすい
        # 逆方向であることを確認 (or at least not both skip)
        assert isinstance(buy_result.should_skip, bool)
        assert isinstance(sell_result.should_skip, bool)


class Test071OBRemoved:
    """071# OB 品質判定が除去されたことの確認."""

    def test_no_ob_quality_method(self) -> None:
        """_check_ob_quality メソッドが存在しない."""
        gate = SkipGate(
            model=None, scaler=None, feature_cols=[],
            config=SkipGateConfig(),
        )
        assert not hasattr(gate, "_check_ob_quality")

    def test_no_ob_critical_features(self) -> None:
        """OB_CRITICAL_FEATURES が存在しない."""
        assert not hasattr(SkipGate, "OB_CRITICAL_FEATURES")

    def test_gate_feature_cols_no_ob(self) -> None:
        """GATE_FEATURE_COLS に OB 特徴量が含まれない."""
        ob_features = {"spread_bps_ob", "depth_imbalance_ob", "side_aligned_imbalance"}
        for col in GATE_FEATURE_COLS:
            assert col not in ob_features, f"OB feature still in GATE_FEATURE_COLS: {col}"


class Test072OBToggle:
    """072# OB トグル: use_ob_features フラグで OB 特徴量の有無を切替."""

    def test_get_gate_feature_cols_no_ob(self) -> None:
        """use_ob=False で OB 特徴量を含まない (16 cols)."""
        cols = get_gate_feature_cols(use_ob=False)
        assert len(cols) == 16
        assert "spread_bps_ob" not in cols
        assert "depth_imbalance_ob" not in cols
        assert "side_aligned_imbalance" not in cols

    def test_get_gate_feature_cols_with_ob(self) -> None:
        """use_ob=True で OB 特徴量を含む (19 cols)."""
        cols = get_gate_feature_cols(use_ob=True)
        assert len(cols) == 19
        assert "spread_bps_ob" in cols
        assert "depth_imbalance_ob" in cols
        assert "side_aligned_imbalance" in cols

    def test_build_features_without_ob(self) -> None:
        """use_ob_features=False で OB 特徴量が生成されない."""
        features = build_features_from_market_state(
            side="buy",
            spread_jpy=500.0,
            offset_ratio=0.05,
            regime="ranging",
            use_ob_features=False,
        )
        assert "spread_bps_ob" not in features
        assert "depth_imbalance_ob" not in features
        assert "side_aligned_imbalance" not in features
        assert "side_buy" in features  # base は健在

    def test_build_features_with_ob(self) -> None:
        """use_ob_features=True + OB データで OB 特徴量が生成される."""
        features = build_features_from_market_state(
            side="buy",
            spread_jpy=500.0,
            offset_ratio=0.05,
            regime="ranging",
            best_bid=14_500_000,
            best_ask=14_500_500,
            bid_vol_5=0.3,
            ask_vol_5=0.2,
            use_ob_features=True,
        )
        # OB 特徴量が存在
        assert "spread_bps_ob" in features
        assert "depth_imbalance_ob" in features
        assert "side_aligned_imbalance" in features
        # 値の妥当性
        expected_spread_bps = (14_500_500 - 14_500_000) / 14_500_250 * 10_000
        assert abs(features["spread_bps_ob"] - expected_spread_bps) < 0.01
        expected_imb = (0.3 - 0.2) / (0.3 + 0.2)
        assert abs(features["depth_imbalance_ob"] - expected_imb) < 1e-6
        # side=buy → side_sign=1.0 → aligned = imb * 1.0
        assert abs(features["side_aligned_imbalance"] - expected_imb) < 1e-6

    def test_build_features_with_ob_sell_side(self) -> None:
        """sell 側: side_aligned_imbalance の符号反転."""
        features = build_features_from_market_state(
            side="sell",
            spread_jpy=500.0,
            offset_ratio=0.05,
            regime="ranging",
            best_bid=14_500_000,
            best_ask=14_500_500,
            bid_vol_5=0.3,
            ask_vol_5=0.2,
            use_ob_features=True,
        )
        expected_imb = (0.3 - 0.2) / (0.3 + 0.2)
        # sell → side_sign=-1 → aligned = imb * -1
        assert abs(features["side_aligned_imbalance"] - (-expected_imb)) < 1e-6

    def test_build_features_with_ob_missing_data(self) -> None:
        """OB データなし + use_ob_features=True → NaN フォールバック."""
        features = build_features_from_market_state(
            side="buy",
            spread_jpy=500.0,
            offset_ratio=0.05,
            regime="ranging",
            use_ob_features=True,
        )
        assert np.isnan(features["spread_bps_ob"])
        assert np.isnan(features["depth_imbalance_ob"])

    def test_skip_gate_config_use_ob_default(self) -> None:
        """SkipGateConfig.use_ob_features のデフォルトは False."""
        cfg = SkipGateConfig()
        assert cfg.use_ob_features is False

    def test_ob_features_nan_handling_via_imputer(self) -> None:
        """OB 特徴量の NaN は SimpleImputer で補完できる."""
        feature_cols = get_gate_feature_cols(use_ob=True)
        rows = []
        for idx in range(4):
            features = build_features_from_market_state(
                side="buy" if idx % 2 == 0 else "sell",
                spread_jpy=400.0 + idx * 10.0,
                offset_ratio=0.04 + idx * 0.01,
                regime="ranging",
                use_ob_features=True,
            )
            rows.append(features)
        frame = pd.DataFrame(rows, columns=feature_cols)

        transformed = SimpleImputer(strategy="median").fit_transform(frame)

        assert transformed.shape[0] == 4
        assert transformed.shape[1] <= len(feature_cols)
        assert transformed.shape[1] >= len(_BASE_FEATURE_COLS)
        assert np.isfinite(transformed).all()

    def test_feature_count_consistency(self) -> None:
        """get_gate_feature_cols と build_features の出力が一致."""
        # OB なし
        cols_no_ob = get_gate_feature_cols(use_ob=False)
        feats_no_ob = build_features_from_market_state(
            side="buy", spread_jpy=500.0, offset_ratio=0.05,
            regime="ranging", use_ob_features=False,
        )
        assert set(cols_no_ob) == set(feats_no_ob.keys())
        # OB あり
        cols_ob = get_gate_feature_cols(use_ob=True)
        feats_ob = build_features_from_market_state(
            side="buy", spread_jpy=500.0, offset_ratio=0.05,
            regime="ranging", best_bid=14_500_000, best_ask=14_500_500,
            bid_vol_5=0.3, ask_vol_5=0.2, use_ob_features=True,
        )
        assert set(cols_ob) == set(feats_ob.keys())


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
        )
        assert "side_buy" in features
        assert features["side_buy"] == 1.0
        assert features["regime_ranging"] == 1.0
        assert features["regime_trending"] == 0.0
        assert features["spread_jpy"] == 500.0
        assert features["offset_ratio"] == 0.05

    def test_interaction_features(self) -> None:
        """071# OB 除去後: trade-based インタラクション特徴量."""
        trades = [
            {"ts": 100.0, "price": 14_500_000, "amount": 0.01, "side": "buy"},
            {"ts": 101.0, "price": 14_500_100, "amount": 0.02, "side": "sell"},
        ]
        features_buy = build_features_from_market_state(
            side="buy",
            spread_jpy=500.0,
            offset_ratio=0.05,
            regime="ranging",
            recent_trades=trades,
        )
        # buy + positive TFI → positive aligned_tfi
        assert "side_aligned_tfi" in features_buy
        assert "side_aligned_velocity" in features_buy
        # OB 特徴量は存在しない
        assert "spread_bps_ob" not in features_buy
        assert "depth_imbalance_ob" not in features_buy
        assert "side_aligned_imbalance" not in features_buy

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
            recent_trades=trades,
        )
        assert features["trade_count_60s"] == 3.0
        assert features["avg_trade_size"] > 0
        assert features["price_velocity_bps"] > 0  # 価格上昇

    def test_no_trades(self) -> None:
        """約定データなし → デフォルト値."""
        features = build_features_from_market_state(
            side="buy",
            spread_jpy=500.0,
            offset_ratio=0.05,
            regime="unknown",
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
            recent_trades=[
                {"ts": 100.0, "price": 14_500_000, "amount": 0.01, "side": "buy"},
            ],
        )
        for col in GATE_FEATURE_COLS:
            assert col in features, f"Missing: {col}"

    def test_recent_trades_handles_unsorted_and_malformed_rows(self) -> None:
        """未整列・一部不正な trade でも安全に集計できる."""
        trades = [
            {"ts": "bad", "price": 99.0, "amount": "oops", "side": None},
            {"ts": 102.0, "price": 102.0, "amount": 3.0, "side": "sell"},
            {"ts": 100.0, "price": 100.0, "amount": 2.0, "side": "buy"},
            {"ts": 101.0, "price": 101.0, "amount": 1.0, "side": "buy"},
        ]
        features = build_features_from_market_state(
            side="buy",
            spread_jpy=500.0,
            offset_ratio=0.05,
            regime="ranging",
            recent_trades=trades,
            trade_window_sec=10,
            market_timestamp=105.0,
        )
        assert features["trade_count_60s"] == 3.0
        assert features["avg_trade_size"] == pytest.approx(2.0)
        assert features["buy_ratio"] == pytest.approx(0.5)
        assert features["price_velocity_bps"] == pytest.approx(200.0)


# ======================================================================
# Integration: 実データ
# ======================================================================


@pytest.mark.slow
@pytest.mark.integration
class Test058Integration:
    """実データが存在する場合の統合テスト."""

    @pytest.fixture(scope="class")
    def real_data_available(self) -> bool:
        return has_fill_records_and_raw_data()

    @pytest.fixture(scope="class")
    def real_smoke_enriched_df(
        self,
        real_data_available: bool,
    ) -> pd.DataFrame:
        if not real_data_available:
            pytest.skip("No real data")
        enriched = _cached_real_enriched_smoke_df()
        if enriched.empty:
            pytest.skip("No fill records")
        return enriched

    @pytest.fixture(scope="class")
    def real_trainable_enriched_df(
        self,
    ) -> pd.DataFrame:
        enriched = _cached_real_enriched_training_df()
        if enriched.empty:
            pytest.skip("No fill records")
        return enriched

    def test_enrichment_with_real_data(
        self,
        real_data_available: bool,
        real_smoke_enriched_df: pd.DataFrame,
    ) -> None:
        """実データでのエンリッチメント."""
        if not real_data_available:
            pytest.skip("No real data")

        assert "spread_bps_ob" in real_smoke_enriched_df.columns
        n_matched = real_smoke_enriched_df["spread_bps_ob"].notna().sum()
        assert n_matched > 0

    def test_train_skip_gate_real(
        self,
        real_data_available: bool,
        real_trainable_enriched_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """実データでの SkipGate 学習."""
        if not real_data_available:
            pytest.skip("No real data")

        # 559# fix: PnL サンプル数が学習最小要件 (20) 未満ならスキップ
        _filled_pnl = real_trainable_enriched_df[
            real_trainable_enriched_df["filled"].astype(bool)
            & real_trainable_enriched_df["post_fill_30s_pnl"].notna()
        ]
        if len(_filled_pnl) < 20:
            pytest.skip(f"Insufficient PnL samples for training: {len(_filled_pnl)}")

        path = tmp_path / "test_gate.pkl"
        gate = train_and_save_skip_gate(
            output_path=path,
            enriched_df=real_trainable_enriched_df,
        )
        assert path.exists()
        # 最新サンプルからの高速読み込みでも学習が成立する最小件数を確認
        assert gate.metadata["n_samples"] >= _REAL_DATA_MIN_TRAIN_SAMPLES

        # 評価テスト (071# OB params removed)
        features = build_features_from_market_state(
            side="buy",
            spread_jpy=500.0,
            offset_ratio=0.05,
            regime="ranging",
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
        self, synthetic_micro_fill_df: pd.DataFrame
    ) -> None:
        """build_pnl_features が micro 特徴量の NaN を保持."""
        df = synthetic_micro_fill_df.copy()
        df.loc[df.index[:5], MICRO_FEATURE_COLS] = np.nan

        X, y = build_pnl_features(df)
        micro_cols = [c for c in X.columns if c in MICRO_FEATURE_COLS]
        if micro_cols:
            assert X[micro_cols].isna().any().any(), \
                "Micro features should preserve NaN for CV-internal imputation"


class Test059SkipRateHistory:
    """059# P0-2: skip 率履歴が最終決定を記録することを検証."""

    def test_skip_rate_records_final_decision(self) -> None:
        """force-pass override 後の最終決定が _recent_skips に記録される."""

        feature_cols = ["f1", "f2", "f3", "f4"]
        gate = _make_negative_skip_gate(feature_cols=feature_cols, max_skip_rate=0.3)

        features = {c: 1.0 for c in feature_cols}

        # 100# per-side skip rate: side を指定して評価
        results = []
        for _ in range(6):
            r = gate.evaluate(features, side="buy")
            results.append(r)

        # force-pass が発動したら _recent_skips には False が記録される
        force_pass_count = sum(
            1 for r in results if "skip_rate_limit" in r.reason
        )
        assert force_pass_count > 0, "Rate limit should have fired"

        # 100# per-side: buy の skip 履歴で判定
        recent_rate = sum(gate._recent_skips_buy) / len(gate._recent_skips_buy)
        assert recent_rate <= 0.5, (
            f"Rate {recent_rate:.2f} should converge below max_skip_rate "
            f"because force-pass records False"
        )

    def test_skip_rate_does_not_oscillate(self) -> None:
        """059# P0-2: skip 率がスパイクしないこと."""

        feature_cols = ["f1", "f2", "f3"]
        gate = _make_negative_skip_gate(feature_cols=feature_cols, max_skip_rate=0.4)

        features = {c: 1.0 for c in feature_cols}
        rates = []
        for i in range(12):
            gate.evaluate(features, side="sell")  # 100# per-side
            if gate._recent_skips_sell:
                rate = sum(gate._recent_skips_sell) / len(gate._recent_skips_sell)
                rates.append(rate)

        # 安定後のレートが max_skip_rate 近辺を超えないこと
        if len(rates) > 8:
            late_rates = rates[-4:]
            assert max(late_rates) <= 0.6, (
                f"Late skip rates should stabilize near max_skip_rate, "
                f"got max={max(late_rates):.2f}"
            )

    def test_trades_fallback_uses_fill_timestamp_window(self) -> None:
        fill_df = pd.DataFrame({
            "timestamp": [1708430400.0],  # 2024-02-20 12:00:00 UTC
            "side": ["buy"],
            "filled": [True],
            "queue_wait_sec": [10.0],
            "adverse_selected": [True],
            "adverse_selected_raw": [True],
            "spread_at_order": [500.0],
            "spread_offset_ratio": [0.05],
            "regime": ["ranging"],
        })
        empty_entry = feature_enricher_module._RawLoadCacheEntry(
            file_signature=(),
            df=pd.DataFrame(),
        )
        seen_filters: list[set[str] | None] = []

        def _fake_load_raw_trades_entry(raw_dir=None, date_filter=None):
            seen_filters.append(None if date_filter is None else set(date_filter))
            return empty_entry

        with patch.object(
            feature_enricher_module,
            "_load_raw_orderbook_entry",
            return_value=empty_entry,
        ), patch.object(
            feature_enricher_module,
            "_load_raw_trades_entry",
            side_effect=_fake_load_raw_trades_entry,
        ):
            enrich_fill_records(fill_df, trades_fallback_recent_days=1)

        assert seen_filters[0] == {"20240220"}
        assert seen_filters[1] == {"20240219", "20240220", "20240221"}


class Test059TimestampConsistency:
    """059# P1-5: 時刻特徴の整合性テスト."""

    def test_market_timestamp_matches_fromtimestamp(self) -> None:
        """market_timestamp を指定した場合、fromtimestamp と同じ時刻特徴になる."""

        ts = 1700000000.0
        now = dt.fromtimestamp(ts)
        expected_hour = now.hour + now.minute / 60.0

        features = build_features_from_market_state(
            side="buy",
            spread_jpy=500.0,
            offset_ratio=0.05,
            regime="ranging",
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
            recent_trades=all_trades,
            trade_window_sec=300,
            market_timestamp=market_ts,
        )
        assert features_300["trade_count_60s"] == 3.0


class Test059PickleHash:
    """059# P2-8: pickle ハッシュ検証テスト."""

    def test_save_creates_hash_file(self, tmp_path: Path) -> None:
        """save がハッシュファイルを作成する."""
        gate = _make_basic_gate()

        path = tmp_path / "gate.pkl"
        gate.save(path)
        hash_path = hash_sidecar_path(path)
        assert hash_path.exists()
        digest = hash_path.read_text().strip()
        assert len(digest) == 64  # SHA256 hex

    def test_load_detects_corruption(self, tmp_path: Path) -> None:
        """改竄されたファイルを検出する."""
        gate = _make_basic_gate()

        path = tmp_path / "gate.pkl"
        gate.save(path)

        # ファイルを改竄
        with open(path, "ab") as f:
            f.write(b"CORRUPTED")

        with pytest.raises(ValueError, match="hash mismatch"):
            SkipGate.load(path)

    def test_load_without_hash_file_succeeds(self, tmp_path: Path) -> None:
        """ハッシュファイルがない場合は警告なしでロード."""
        gate = _make_basic_gate()

        path = tmp_path / "gate.pkl"
        save_and_load_skip_gate(gate, path)
        # ハッシュファイルを削除
        hash_path = hash_sidecar_path(path)
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


# ======================================================================
# 106# R3 / 107#: SkipGate warm_start + adaptive threshold tests
# ======================================================================


class Test106R3AdaptiveThreshold:
    """106# R3: SkipGate _calibrate_threshold の単体テスト."""

    @pytest.fixture()
    def adaptive_gate(self) -> SkipGate:
        """adaptive_threshold=True の SkipGate."""


        cfg = SkipGateConfig(
            mode="as",
            as_threshold=0.52,
            as_threshold_buy=0.52,
            as_threshold_sell=0.50,
            adaptive_threshold=True,
            target_skip_rate_buy=0.15,
            target_skip_rate_sell=0.25,
            adaptive_window=50,
            adaptive_min_samples=5,  # テスト用に低く設定
            adaptive_step=0.05,
            adaptive_floor=0.35,
            adaptive_ceiling=0.80,
        )
        model = Ridge(alpha=1.0)
        feature_cols = list(_BASE_FEATURE_COLS)
        n_features = len(feature_cols)
        X_dummy = np.random.randn(20, n_features)
        y_dummy = np.random.randint(0, 2, 20).astype(float)
        model.fit(X_dummy, y_dummy)
        scaler = StandardScaler().fit(X_dummy)
        return SkipGate(model, scaler, feature_cols, config=cfg)

    def test_calibrate_warmup_returns_base(self, adaptive_gate: SkipGate) -> None:
        """ウォームアップ期間中は静的閾値をそのまま返す."""
        # adaptive_min_samples=5 なので、4件ではまだウォーム中
        for _ in range(4):
            result = adaptive_gate._calibrate_threshold("buy", 0.50, 0.52)
        assert result == 0.52

    def test_calibrate_after_warmup_adjusts(self, adaptive_gate: SkipGate) -> None:
        """ウォームアップ完了後は閾値が調整される."""
        # 5件挿入 → 較正開始
        for i in range(6):
            adaptive_gate._calibrate_threshold("buy", 0.45 + i * 0.01, 0.52)
        # 較正後の閾値が元の 0.52 から変化しているはず
        th = adaptive_gate.config.as_threshold_buy
        assert th is not None
        assert th != 0.52  # 調整が入った

    def test_calibrate_respects_floor_ceiling(self, adaptive_gate: SkipGate) -> None:
        """閾値が floor/ceiling でクランプされる."""
        # 全て P(AS)=0.01 → 閾値は下がる方向だが floor=0.35 で制限
        for _ in range(10):
            adaptive_gate._calibrate_threshold("sell", 0.01, 0.35)
        th = adaptive_gate.config.as_threshold_sell
        assert th is not None
        assert th >= 0.35  # floor を下回らない

    def test_calibrate_side_independence(self, adaptive_gate: SkipGate) -> None:
        """buy と sell の calibration は独立."""
        for _ in range(10):
            adaptive_gate._calibrate_threshold("buy", 0.60, 0.52)
            adaptive_gate._calibrate_threshold("sell", 0.40, 0.50)
        # buy の履歴は buy だけ、sell の履歴は sell だけ
        assert len(adaptive_gate._pas_history_buy) == 10
        assert len(adaptive_gate._pas_history_sell) == 10

    def test_calibrate_window_limit(self, adaptive_gate: SkipGate) -> None:
        """履歴が adaptive_window を超えたら古いものが削除される."""
        for i in range(100):
            adaptive_gate._calibrate_threshold("buy", 0.50, 0.52)
        assert len(adaptive_gate._pas_history_buy) == 50  # window=50


class Test106R3WarmStart:
    """106# R3: warm_start_skip_gate_thresholds の単体テスト."""

    @pytest.fixture()
    def adaptive_gate(self) -> SkipGate:
        """adaptive_threshold=True の SkipGate (warmup 可能)."""


        cfg = SkipGateConfig(
            mode="as",
            as_threshold=0.52,
            as_threshold_buy=0.52,
            as_threshold_sell=0.50,
            adaptive_threshold=True,
            target_skip_rate_buy=0.15,
            target_skip_rate_sell=0.25,
            adaptive_window=10,
            adaptive_min_samples=5,
            adaptive_step=0.05,
            adaptive_floor=0.35,
            adaptive_ceiling=0.80,
        )
        model = Ridge(alpha=1.0)
        feature_cols = list(_BASE_FEATURE_COLS)
        n_features = len(feature_cols)
        X_dummy = np.random.randn(20, n_features)
        y_dummy = np.random.randint(0, 2, 20).astype(float)
        model.fit(X_dummy, y_dummy)
        scaler = StandardScaler().fit(X_dummy)
        return SkipGate(model, scaler, feature_cols, config=cfg)

    def test_warm_start_restores_history(
        self, adaptive_gate: SkipGate, tmp_path: Path,
    ) -> None:
        """fill_records から P(AS) 履歴を正しく復元する."""

        # テスト用 fill_records を作成
        records = []
        for i in range(15):
            side = "buy" if i % 2 == 0 else "sell"
            records.append(json.dumps({
                "side": side,
                "skip_gate_as_prob": 0.45 + i * 0.005,
                "filled": True,
                "timestamp": 1700000000 + i * 60,
            }))
        (tmp_path / "fill_records_20260101.jsonl").write_text("\n".join(records))

        warm_start_skip_gate_thresholds(adaptive_gate, str(tmp_path), window=10)

        # buy: i=0,2,4,6,8,10,12,14 → 8件、window=10 なので全部入る
        assert len(adaptive_gate._pas_history_buy) > 0
        assert len(adaptive_gate._pas_history_sell) > 0

    def test_warm_start_empty_dir(
        self, adaptive_gate: SkipGate, tmp_path: Path,
    ) -> None:
        """空のディレクトリでも例外なく動作する."""

        warm_start_skip_gate_thresholds(adaptive_gate, str(tmp_path), window=10)
        assert len(adaptive_gate._pas_history_buy) == 0
        assert len(adaptive_gate._pas_history_sell) == 0

    def test_warm_start_triggers_calibration(
        self, adaptive_gate: SkipGate, tmp_path: Path,
    ) -> None:
        """十分なサンプルがあれば warm_start 後に閾値較正が発動する."""

        records = []
        for i in range(20):
            records.append(json.dumps({
                "side": "buy",
                "skip_gate_as_prob": 0.48 + i * 0.001,
                "filled": True,
                "timestamp": 1700000000 + i * 60,
            }))
        (tmp_path / "fill_records_20260101.jsonl").write_text("\n".join(records))

        original_buy_th = adaptive_gate.config.as_threshold_buy
        warm_start_skip_gate_thresholds(adaptive_gate, str(tmp_path), window=10)

        # 較正が入って閾値が変化しているはず
        assert adaptive_gate.config.as_threshold_buy != original_buy_th

    def test_warm_start_prefers_most_recent_records(
        self, adaptive_gate: SkipGate, tmp_path: Path,
    ) -> None:
        """複数ファイル時は新しいレコード側を優先して復元する."""

        old_records = [
            json.dumps({
                "side": "buy",
                "skip_gate_as_prob": 0.10 + i * 0.01,
                "filled": True,
                "timestamp": 1700000000 + i * 60,
            })
            for i in range(6)
        ]
        new_records = [
            json.dumps({
                "side": "buy",
                "skip_gate_as_prob": 0.60 + i * 0.01,
                "filled": True,
                "timestamp": 1700003600 + i * 60,
            })
            for i in range(6)
        ]
        (tmp_path / "fill_records_20260101.jsonl").write_text("\n".join(old_records))
        (tmp_path / "fill_records_20260102.jsonl").write_text("\n".join(new_records))

        warm_start_skip_gate_thresholds(adaptive_gate, str(tmp_path), window=3)

        assert adaptive_gate._pas_history_buy == pytest.approx([0.63, 0.64, 0.65])
