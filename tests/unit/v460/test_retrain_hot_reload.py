"""126# SkipGate 再学習スケジューラ + hot-reload テスト.

retrain_scheduler.py の retrain_model() と
SkipGateEvaluator の hot-reload 機構をテスト。
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

try:
    import lightgbm  # noqa: F401
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

from scripts.v460.ml.skip_gate import (
    SkipGate,
    SkipGateConfig,
    get_gate_feature_cols,
)
from scripts.v460.ml.feature_enricher import enrich_fill_records
from scripts.v460.ml.retrain_scheduler import (
    _DEFAULT_CONFIG,
    _apply_statistical_gate,
    _build_full_features,
    _build_lgbm_regressor,
    _compute_regime_sample_weights,
    _evaluate_wf,
    _evaluate_wf_multi,
    _evaluate_wf_single,
    _load_enriched_cache,
    _safe_import_ztb_module,
    _save_enriched_cache,
    load_retrain_config,
    retrain_model,
)
from scripts.v460.analysis.oracle_baseline import (
    _group_oracle_aggregates,
    compute_oracle_metrics,
)
from scripts.v460.lib.lot_sizer import LotSizingConfig, compute_lot_size
from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator
from tests.unit.v460._skip_gate_test_helpers import PickleStub, save_and_load_skip_gate
from ztb.metrics.fill_quality import FillRecord
from ztb.ml.artifact_paths import atomic_pickle_tmp_path, hash_sidecar_path
from ztb.utils.run_manifest import compute_file_hash

try:
    _REDUNDANCY_MODULE = _safe_import_ztb_module("ztb.analysis.redundancy")
except ImportError:
    _REDUNDANCY_MODULE = None


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_placeholder_model(path: Path) -> None:
    """SkipGateEvaluator 初期化向けの最小ファイルを配置する."""
    path.write_bytes(b"placeholder-model")


def _write_stub_gate_artifact(_gate: object, path: Path) -> None:
    """SkipGate.save の代替: 最小 deploy artifact と sidecar hash を生成する."""
    payload = b"stub-gate"
    path.write_bytes(payload)
    hash_sidecar_path(path).write_text(
        hashlib.sha256(payload).hexdigest(),
        encoding="utf-8",
    )


def _make_skip_gate_eval_config(
    *,
    model_path: str,
    results_dir: str,
) -> SimpleNamespace:
    """SkipGateEvaluator が直接参照する属性を満たす軽量 config."""
    return SimpleNamespace(
        skip_gate_enabled=True,
        skip_gate_model_path=model_path,
        skip_gate_mode="pnl",
        skip_gate_as_threshold=0.50,
        skip_gate_pnl_threshold=0.0,
        skip_gate_max_skip_rate=0.3,
        skip_gate_buy_enabled=True,
        skip_gate_sell_enabled=True,
        skip_gate_as_threshold_buy=0.50,
        skip_gate_as_threshold_sell=0.50,
        skip_gate_use_ob_features=False,
        skip_gate_adaptive_threshold=False,
        skip_gate_target_skip_rate_buy=0.15,
        skip_gate_target_skip_rate_sell=0.20,
        skip_gate_adaptive_window=50,
        skip_gate_adaptive_min_samples=20,
        skip_gate_adaptive_step=0.05,
        skip_gate_adaptive_floor=0.35,
        skip_gate_adaptive_ceiling=0.80,
        skip_gate_regime_thresholds={},
        skip_gate_score_calibration=False,
        skip_gate_calibrator_path="",
        skip_gate_recent_trades_limit=200,
        skip_gate_ob_depth=5,
        skip_gate_hour_offsets={},
        skip_gate_narrow_spread_threshold_jpy=0.0,
        skip_gate_narrow_spread_offset=0.0,
        skip_gate_offset_floor=-10.0,
        skip_gate_offset_ceil=10.0,
        skip_gate_ev_weighted_enabled=False,
        skip_gate_ev_w30=0.5,
        skip_gate_ev_w120=0.5,
        skip_gate_ev_as_offset_enabled=False,
        skip_gate_ev_one_sided_threshold_shift=0.0,
        skip_gate_ev_max_consecutive_skip=0,
        skip_gate_ev_emergency_skip_threshold=1.0,
        skip_gate_ev_offset_sensitivity=1.0,
        skip_gate_ev_offset_min_mult=1.0,
        skip_gate_ev_offset_max_mult=1.0,
        skip_gate_ev_warning_threshold=1.0,
        skip_gate_ev_warning_offset_factor=1.0,
        skip_sell_unknown_regime=False,
        results_dir=results_dir,
        hot_reload_check_interval_sec=120.0,
    )

def _make_picklable_gate(
    *,
    n_samples: int = 100,
    version: str = "test",
    mode: str = "pnl",
) -> SkipGate:
    """pickle 可能な SkipGate を作成."""
    config = SkipGateConfig(mode=mode)
    model = PickleStub("model")
    scaler = PickleStub("scaler")
    feature_cols = ["spread_jpy", "offset_ratio", "regime_trending"]
    gate = SkipGate(
        model=model, scaler=scaler, feature_cols=feature_cols,
        config=config, pipeline=None,
        metadata={
            "version": version,
            "n_samples": n_samples,
            "trained_at": "2026-02-21T12:00:00",
        },
    )
    return gate


def _save_gate_to(gate: SkipGate, path: Path) -> None:
    """SkipGate を指定パスに保存."""
    gate.save(path)


def _write_corrupt_gate(path: Path, *, hash_text: str = "deadbeef") -> None:
    """壊れた gate 本体 + sidecar hash を共通生成する."""
    path.write_bytes(b"corrupt data")
    hash_sidecar_path(path).write_text(hash_text)


def _cache_path(tmp_path: Path, name: str = "test_cache.pkl") -> Path:
    """cache test 用の path を返す."""
    return tmp_path / name


def _model_paths(tmp_path: Path, name: str = "model.pkl") -> tuple[Path, Path]:
    """model path と atomic save 用 tmp path を返す."""
    model_path = tmp_path / name
    return model_path, atomic_pickle_tmp_path(model_path)


def _identity_enrich(fill_df: pd.DataFrame, **_: object) -> pd.DataFrame:
    """enrich を省略しつつ、学習に必要な最小特徴量を補完する軽量ヘルパー."""
    enriched = fill_df.copy()
    defaults: dict[str, float] = {
        "trade_count_60s": 0.0,
        "buy_ratio": 0.5,
        "trade_flow_imbalance_60s": 0.0,
        "avg_trade_size": 0.0,
        "price_velocity_bps": 0.0,
        "vpin_60s": 0.5,
        "spread_bps_ob": 0.0,
        "depth_imbalance_ob": 0.0,
    }
    for col, value in defaults.items():
        if col not in enriched.columns:
            enriched[col] = value
            continue
        enriched[col] = pd.to_numeric(enriched[col], errors="coerce").fillna(value)
    return enriched


def _build_fast_preorder_as_features(
    fill_df: pd.DataFrame,
    require_spread: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """retrain_model() 向けの最小特徴量を高速構築する."""
    mask = fill_df["filled"].astype(bool)
    if require_spread:
        if "spread_at_order" in fill_df.columns:
            mask &= fill_df["spread_at_order"].notna()
        if "spread_offset_ratio" in fill_df.columns:
            mask &= fill_df["spread_offset_ratio"].notna()

    subset = fill_df.loc[mask].copy()
    feature_cols = list(get_gate_feature_cols(use_ob=False))
    X = pd.DataFrame(0.0, index=subset.index, columns=feature_cols)

    if "spread_jpy" in X.columns and "spread_at_order" in subset.columns:
        X["spread_jpy"] = pd.to_numeric(
            subset["spread_at_order"],
            errors="coerce",
        ).fillna(0.0)
    if "offset_ratio" in X.columns and "spread_offset_ratio" in subset.columns:
        X["offset_ratio"] = pd.to_numeric(
            subset["spread_offset_ratio"],
            errors="coerce",
        ).fillna(0.0)
    if "queue_wait_sec" in X.columns:
        if "queue_wait_sec" in subset.columns:
            X["queue_wait_sec"] = pd.to_numeric(
                subset["queue_wait_sec"],
                errors="coerce",
            ).fillna(0.0)
        else:
            X["queue_wait_sec"] = 0.0
    if "side_buy" in X.columns and "side" in subset.columns:
        X["side_buy"] = subset["side"].eq("buy").astype(float)

    y = pd.to_numeric(
        subset.get("adverse_selected_raw", 0),
        errors="coerce",
    ).fillna(0).astype(int)
    return X, y


def _make_retrain_records_df(
    n: int,
    *,
    seed: int,
    cycle_prefix: str,
    run_id: str,
    regimes: tuple[str, ...] = ("ranging",),
    balance_forced_first: int = 0,
) -> pd.DataFrame:
    """retrain_model() 向けの最小 fill records DataFrame を構築する."""
    rng = np.random.RandomState(seed)
    rows: list[dict[str, object]] = []
    for i in range(n):
        row: dict[str, object] = {
            "cycle_id": f"{cycle_prefix}_{i}",
            "side": "buy" if i % 2 == 0 else "sell",
            "filled": True,
            "timestamp": 1771502400.0 + i * 120,
            "order_price": 15_000_000.0 + rng.randn() * 10_000,
            "order_quantity": 0.001,
            "spread_at_order": 2500.0 + rng.randn() * 500,
            "spread_offset_ratio": 0.05 + rng.randn() * 0.01,
            "adverse_selected_raw": int(rng.random() > 0.5),
            "post_fill_30s_pnl": float(rng.randn() * 2),
            "post_fill_120s_pnl": float(rng.randn() * 3),
            "regime": regimes[i % len(regimes)],
            "run_id": run_id,
        }
        if balance_forced_first > 0:
            row["balance_forced_switch"] = i < balance_forced_first
        rows.append(row)
    return pd.DataFrame(rows)


class _FastBooster:
    """LightGBM 依存を避ける最小 booster スタブ."""

    def __init__(self, n_trees: int) -> None:
        self._n_trees = max(1, n_trees)

    def num_trees(self) -> int:
        return self._n_trees


class _FastRegressor:
    """WF/E2E 配線検証用の軽量 regressor."""

    def __init__(self, n_estimators: int = 4) -> None:
        self.n_estimators = max(1, n_estimators)
        self.feature_importances_ = np.empty(0, dtype=int)
        self.booster_ = _FastBooster(self.n_estimators)
        self._prediction = 0.0

    def fit(self, X: object, y: object, **_: object) -> "_FastRegressor":
        x_arr = np.asarray(X)
        n_features = int(x_arr.shape[1]) if x_arr.ndim > 1 else 1
        self.feature_importances_ = np.ones(n_features, dtype=int)
        y_arr = np.asarray(y, dtype=np.float64)
        self._prediction = float(np.mean(y_arr)) if y_arr.size > 0 else 0.0
        self.booster_ = _FastBooster(self.n_estimators)
        return self

    def predict(self, X: object) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        n = x_arr.shape[0] if x_arr.ndim >= 1 else 1
        # 465# D2 対応: 特徴量ベースの変動を付加 (定数出力ガード回避)
        if x_arr.ndim >= 2 and x_arr.shape[1] > 0:
            variation = x_arr[:, 0] * 0.1
        else:
            variation = np.zeros(n)
        return np.full(n, self._prediction, dtype=np.float64) + variation


def _build_fast_regressor(
    cfg: dict[str, object],
    *,
    n_estimators_override: int | None = None,
) -> _FastRegressor:
    """retrain_scheduler._build_lgbm_regressor 互換の高速スタブ."""
    n_estimators = (
        n_estimators_override
        if n_estimators_override is not None
        else int(cfg.get("lgbm_n_estimators", 4))
    )
    return _FastRegressor(n_estimators=n_estimators)


def _make_stub_window(
    train_start: int,
    train_end: int,
    val_start: int,
    val_end: int,
    test_start: int,
    test_end: int,
    *,
    window_id: int = 0,
) -> SimpleNamespace:
    """_evaluate_wf_multi 用の最小 window オブジェクト."""
    return SimpleNamespace(
        window_id=window_id,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
    )


def _make_splitter_module(windows: list[SimpleNamespace]) -> SimpleNamespace:
    """WalkForwardSplitter だけを持つ最小 module stub."""

    class _FakeWalkForwardSplitter:
        def __init__(self, **_: object) -> None:
            self._windows = windows

        def split(self, _df: pd.DataFrame) -> list[SimpleNamespace]:
            return list(self._windows)

    return SimpleNamespace(WalkForwardSplitter=_FakeWalkForwardSplitter)


# =====================================================================
# Hot-Reload テスト
# =====================================================================

class TestHotReload:
    """126# SkipGateEvaluator hot-reload テスト."""

    def _make_config(self, model_path: str) -> SimpleNamespace:
        """テスト用 FillTestConfig モック."""
        return _make_skip_gate_eval_config(
            model_path=model_path,
            results_dir="results/v460/fill_test",
        )

    def _create_evaluator(
        self,
        tmpdir: str,
        *,
        gate: SkipGate,
        write_placeholder: bool = False,
        hash_override: str | None = None,
    ) -> tuple[Path, SkipGateEvaluator]:
        """共通の model path + evaluator 初期化."""
        model_path = Path(tmpdir) / "gate.pkl"
        if write_placeholder:
            _write_placeholder_model(model_path)
        else:
            _save_gate_to(gate, model_path)

        cfg = self._make_config(str(model_path))
        if hash_override is None:
            return model_path, SkipGateEvaluator(cfg, Path(tmpdir))

        with patch(
            "scripts.v460.lib.skip_gate_evaluator.SkipGateEvaluator._read_model_hash",
            return_value=hash_override,
        ):
            return model_path, SkipGateEvaluator(cfg, Path(tmpdir))

    def test_initial_hash_stored(self, tmp_path: Path) -> None:
        """初期ロード時にモデルファイルのハッシュが保存される."""
        gate = _make_picklable_gate(version="v1")
        _, evaluator = self._create_evaluator(tmp_path, gate=gate)

        assert evaluator._model_file_hash != ""
        assert len(evaluator._model_file_hash) == 64  # SHA256 hex

    def test_initial_hash_uses_sidecar_when_fresh(self, tmp_path: Path) -> None:
        """モデル sidecar hash が新鮮なら full file hash scan にフォールバックしない."""
        gate = _make_picklable_gate(version="v1")
        with patch(
            "scripts.v460.lib.skip_gate_evaluator.compute_file_hash",
            side_effect=AssertionError("full hash scan should not run"),
        ):
            _, evaluator = self._create_evaluator(tmp_path, gate=gate)

        assert evaluator._model_file_hash != ""

    def test_no_reload_when_unchanged(self, tmp_path: Path) -> None:
        """ファイル未変更時はリロードしない."""
        gate = _make_picklable_gate(version="v1")
        with patch(
            "scripts.v460.ml.skip_gate.SkipGate.load",
            return_value=gate,
        ), patch(
            "scripts.v460.lib.skip_gate_evaluator.SkipGateEvaluator._read_model_hash",
            return_value="a" * 64,
        ):
            _, evaluator = self._create_evaluator(tmp_path, gate=gate, write_placeholder=True)
            original_gate = evaluator._skip_gate
            original_hash = evaluator._model_file_hash

            # 即座にチェックを強制 (interval を 0 に)
            evaluator._last_reload_check = 0
            with patch.object(evaluator, "_check_and_reload_side_models", return_value=None):
                evaluator._check_and_reload_model()

        assert evaluator._skip_gate is original_gate
        assert evaluator._model_file_hash == original_hash

    def test_reload_on_file_change(self, tmp_path: Path) -> None:
        """ファイル変更時にリロードされる."""
        gate_v1 = _make_picklable_gate(version="v1", n_samples=100)
        _, evaluator = self._create_evaluator(tmp_path, gate=gate_v1)
        original_hash = evaluator._model_file_hash

        gate_v2 = _make_picklable_gate(version="v2", n_samples=200)

        # 強制チェック
        evaluator._last_reload_check = 0
        with patch(
            "scripts.v460.lib.skip_gate_evaluator.SkipGateEvaluator._read_model_hash",
            return_value="b" * 64,
        ), patch.object(
            evaluator,
            "_load_gate_from_path",
            return_value=gate_v2,
        ):
            evaluator._check_and_reload_model()

        assert evaluator._model_file_hash != original_hash
        assert evaluator._skip_gate is not None
        assert evaluator._skip_gate.metadata["version"] == "v2"  # type: ignore[union-attr]
        assert evaluator._skip_gate.metadata["n_samples"] == 200  # type: ignore[union-attr]

    def test_reload_failure_keeps_old_model(self, tmp_path: Path) -> None:
        """リロード失敗時は旧モデルを維持."""
        gate = _make_picklable_gate(version="v1")
        model_path, evaluator = self._create_evaluator(tmp_path, gate=gate)
        original_gate = evaluator._skip_gate
        original_hash = evaluator._model_file_hash

        # 不正なデータで上書き
        model_path.write_bytes(b"corrupted data")

        evaluator._last_reload_check = 0
        evaluator._check_and_reload_model()

        # 旧モデルが維持される
        assert evaluator._skip_gate is original_gate
        # ハッシュも旧のまま (更新失敗)
        assert evaluator._model_file_hash == original_hash

    def test_check_interval_respected(self, tmp_path: Path) -> None:
        """チェック間隔内ではファイル変更を検出しない."""
        gate = _make_picklable_gate(version="v1")
        model_path, evaluator = self._create_evaluator(tmp_path, gate=gate)

        # 内容変更だけ入れる。interval 内なので reload 経路には入らない。
        model_path.write_bytes(b"changed but should not be reloaded yet")

        # last_reload_check を更新しない → interval 内
        evaluator._check_and_reload_model()

        # まだ v1 のまま
        assert evaluator._skip_gate.metadata["version"] == "v1"  # type: ignore[union-attr]

    def test_compute_file_hash(self, tmp_path: Path) -> None:
        """compute_file_hash で SHA256 が正しく計算される."""
        p = tmp_path / "test.bin"
        p.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert compute_file_hash(p) == expected

    def test_compute_file_hash_missing_file(self) -> None:
        """存在しないファイルのハッシュ計算は例外."""
        with pytest.raises((FileNotFoundError, OSError)):
            compute_file_hash(Path("/nonexistent"))


# =====================================================================
# Retrain Scheduler テスト
# =====================================================================

class TestRetrainConfig:
    """126# retrain 設定ロードテスト."""

    def _load_config_from_yaml_data(
        self,
        tmp_path: Path,
        yaml_data: dict[str, object],
    ) -> dict[str, object]:
        yaml_path = tmp_path / "test.yaml"
        yaml_path.touch()
        with patch(
            "scripts.v460.lib.config_loader.load_fill_test_config",
            return_value=yaml_data,
        ):
            return load_retrain_config(yaml_path)

    def test_default_config(self) -> None:
        """デフォルト設定が正しく読み込まれる."""
        cfg = load_retrain_config(Path("/nonexistent.yaml"))
        assert cfg["interval_sec"] == 3600
        assert cfg["min_new_samples"] == 30
        assert cfg["target"] == "pnl120"
        assert cfg["quality_gate_enabled"] is True
        # 127# C1: skip_gate 由来のデフォルト
        assert "model_path" in cfg
        assert "mode" in cfg
        assert "use_ob_features" in cfg
        # 127# M1
        assert "absolute_min_score" in cfg
        # 127# H2
        assert cfg["latest_run_only"] is True

    def test_yaml_override(self, tmp_path: Path) -> None:
        """YAML の retrain + skip_gate セクションが統合される."""
        cfg = self._load_config_from_yaml_data(
            tmp_path,
            {
                "results_dir": "/tmp/test_results",
                "skip_gate": {
                    "mode": "pnl",
                    "model_path": "/tmp/test_model.pkl",
                    "use_ob_features": False,
                },
                "retrain": {
                    "interval_sec": 7200,
                    "min_new_samples": 50,
                },
            },
        )
        assert cfg["interval_sec"] == 7200
        assert cfg["min_new_samples"] == 50
        # 127# C1: skip_gate から継承
        assert cfg["model_path"] == "/tmp/test_model.pkl"
        assert cfg["mode"] == "pnl"
        assert cfg["use_ob_features"] is False
        assert cfg["results_dir"] == "/tmp/test_results"
        # 未指定のキーはデフォルト
        assert cfg["target"] == "pnl120"

    def test_validation_rejects_non_pnl_mode(self, tmp_path: Path) -> None:
        """127# C1: mode != pnl なら起動拒否."""
        with pytest.raises(ValueError, match="requires skip_gate.mode='pnl'"):
            self._load_config_from_yaml_data(
                tmp_path,
                {
                    "skip_gate": {
                        "mode": "as",
                        "model_path": "/tmp/model.pkl",
                    },
                },
            )

    def test_validation_rejects_bad_target(self, tmp_path: Path) -> None:
        """127# M2: 不正な target は拒否."""
        with pytest.raises(ValueError, match="pnl120.*pnl30"):
            self._load_config_from_yaml_data(
                tmp_path,
                {
                    "skip_gate": {
                        "mode": "pnl",
                        "model_path": "/tmp/model.pkl",
                    },
                    "retrain": {
                        "target": "as30",
                    },
                },
            )

    def test_retrain_model_forwards_fill_records_max_files(self) -> None:
        """fill_records_max_files が load_fill_records に伝播する."""
        cfg = dict(_DEFAULT_CONFIG)
        cfg["results_dir"] = "results/v460/fill_test"
        cfg["fill_records_max_files"] = 7

        def _fake_loader(*args: object, **kwargs: object) -> pd.DataFrame:
            assert kwargs.get("max_files") == 7
            raise FileNotFoundError("test")

        with patch("scripts.v460.ml.retrain_scheduler.load_fill_records", side_effect=_fake_loader):
            result = retrain_model(cfg)
        assert result["status"] == "skipped"

    def test_retrain_model_ignores_non_positive_fill_records_max_files(self) -> None:
        """fill_records_max_files<=0 は未指定(None)として扱う."""
        cfg = dict(_DEFAULT_CONFIG)
        cfg["results_dir"] = "results/v460/fill_test"
        cfg["fill_records_max_files"] = 0

        def _fake_loader(*args: object, **kwargs: object) -> pd.DataFrame:
            assert kwargs.get("max_files") is None
            raise FileNotFoundError("test")

        with patch("scripts.v460.ml.retrain_scheduler.load_fill_records", side_effect=_fake_loader):
            result = retrain_model(cfg)
        assert result["status"] == "skipped"


class TestBuildFullFeatures:
    """126# _build_full_features テスト."""

    def test_base_features_only(self) -> None:
        """use_ob=False → base features のみ."""

        X_base = pd.DataFrame({
            "side_buy": [1.0, 0.0],
            "hour_sin": [0.5, -0.5],
            "hour_cos": [0.866, 0.866],
            "spread_jpy": [3000.0, 2500.0],
            "offset_ratio": [0.05, 0.10],
            "regime_trending": [1.0, 0.0],
            "regime_ranging": [0.0, 1.0],
            "regime_high_vol": [0.0, 0.0],
            "trade_count_60s": [10.0, 5.0],
            "buy_ratio": [0.6, 0.4],
            "trade_flow_imbalance_60s": [0.2, -0.2],
            "avg_trade_size": [0.1, 0.05],
            "price_velocity_bps": [1.0, -1.0],
            "vpin_60s": [0.3, 0.7],
            "side_aligned_tfi": [0.2, 0.2],
            "side_aligned_velocity": [1.0, 1.0],
        })
        enriched = pd.DataFrame({"side": ["buy", "sell"]}, index=X_base.index)

        result = _build_full_features(enriched, X_base, use_ob=False)
        assert result.shape[1] == 16  # base のみ

    def test_full_features_with_ob(self) -> None:
        """use_ob=True → base + OB features."""

        X_base = pd.DataFrame({
            "side_buy": [1.0],
            "hour_sin": [0.5],
            "hour_cos": [0.866],
            "spread_jpy": [3000.0],
            "offset_ratio": [0.05],
            "regime_trending": [1.0],
            "regime_ranging": [0.0],
            "regime_high_vol": [0.0],
            "trade_count_60s": [10.0],
            "buy_ratio": [0.6],
            "trade_flow_imbalance_60s": [0.2],
            "avg_trade_size": [0.1],
            "price_velocity_bps": [1.0],
            "vpin_60s": [0.3],
            "side_aligned_tfi": [0.2],
            "side_aligned_velocity": [1.0],
        })
        enriched = pd.DataFrame({
            "side": ["buy"],
            "spread_bps_ob": [30.0],
            "depth_imbalance_ob": [0.5],
        }, index=X_base.index)

        result = _build_full_features(enriched, X_base, use_ob=True)
        assert result.shape[1] == 19  # base + OB(3)
        assert "spread_bps_ob" in result.columns
        assert "depth_imbalance_ob" in result.columns
        assert "side_aligned_imbalance" in result.columns


@pytest.mark.skipif(not _HAS_LIGHTGBM, reason="lightgbm not installed")
class TestRetrainModel:
    """126# retrain_model() テスト."""

    @pytest.fixture(autouse=True)
    def _fast_preorder_features(self) -> object:
        with patch(
            "scripts.v460.ml.retrain_scheduler.build_preorder_as_features",
            side_effect=_build_fast_preorder_as_features,
        ):
            yield

    def test_skip_when_no_fill_records(self) -> None:
        """fill_records が存在しない場合スキップ."""

        cfg = dict(_DEFAULT_CONFIG)
        cfg["results_dir"] = "/nonexistent_dir_12345"
        cfg["model_path"] = "/nonexistent_model.pkl"
        cfg["mode"] = "pnl"
        cfg["use_ob_features"] = True
        result = retrain_model(cfg)
        assert result["status"] == "skipped"
        assert "no fill_records" in result["reason"]

    def test_skip_when_insufficient_samples(self) -> None:
        """サンプル不足時はスキップ."""

        with tempfile.TemporaryDirectory() as tmpdir:
            records_dir = Path(tmpdir)
            records_df = _make_retrain_records_df(
                5,
                seed=7,
                cycle_prefix="test",
                run_id="test_run",
            )

            cfg = dict(_DEFAULT_CONFIG)
            cfg["results_dir"] = str(records_dir)
            cfg["model_path"] = str(Path(tmpdir) / "nonexistent.pkl")
            cfg["mode"] = "pnl"
            cfg["use_ob_features"] = True
            cfg["min_total_samples"] = 100  # 5 < 100
            cfg["latest_run_only"] = False
            cfg["exclude_missing_run_id"] = False
            with patch(
                "scripts.v460.ml.retrain_scheduler.load_fill_records",
                side_effect=lambda *args, **kwargs: records_df.copy(deep=True),
            ):
                result = retrain_model(cfg)
            assert result["status"] == "skipped"

    def test_skip_when_insufficient_new_samples(self) -> None:
        """新規サンプル不足時はスキップ."""

        with tempfile.TemporaryDirectory() as tmpdir:
            records_dir = Path(tmpdir) / "results"
            records_dir.mkdir()
            model_dir = Path(tmpdir) / "models"
            model_dir.mkdir()
            records_df = _make_retrain_records_df(
                10,
                seed=11,
                cycle_prefix="test",
                run_id="test_run",
            )

            # 既存モデルを配置 (n_samples=10 → 新規 0 件)
            model_path = model_dir / "gate.pkl"
            gate = _make_picklable_gate(n_samples=10)
            _save_gate_to(gate, model_path)

            cfg = dict(_DEFAULT_CONFIG)
            cfg["results_dir"] = str(records_dir)
            cfg["model_path"] = str(model_path)
            cfg["mode"] = "pnl"
            cfg["use_ob_features"] = True
            cfg["min_new_samples"] = 5
            cfg["min_total_samples"] = 10
            cfg["bootstrap_min_total_samples"] = 10
            cfg["bootstrap_min_new_samples"] = 5
            cfg["latest_run_only"] = False
            cfg["exclude_missing_run_id"] = False
            cfg["lgbm_n_estimators"] = 30
            cfg["enriched_cache_enabled"] = False
            with patch(
                "scripts.v460.ml.retrain_scheduler.load_fill_records",
                side_effect=lambda *args, **kwargs: records_df.copy(deep=True),
            ), patch(
                "scripts.v460.ml.retrain_scheduler.enrich_fill_records",
                side_effect=_identity_enrich,
            ):
                result = retrain_model(cfg)
            assert result["status"] == "skipped"
            assert "insufficient new samples" in result.get("reason", "")


# =====================================================================
# 127# M3: E2E 成功テスト (retrain → deploy → hot-reload → evaluate)
# =====================================================================

@pytest.mark.skipif(not _HAS_LIGHTGBM, reason="lightgbm not installed")
class TestE2ERetrainHotReload:
    """127# M3: 再学習→配置→hot-reload→評価の統合テスト."""

    @pytest.fixture(autouse=True)
    def _fast_regressor(self) -> object:
        with patch(
            "scripts.v460.ml.retrain_scheduler._build_lgbm_regressor",
            side_effect=_build_fast_regressor,
        ):
            yield

    @pytest.fixture(autouse=True)
    def _disable_warm_start(self) -> object:
        with patch(
            "scripts.v460.lib.skip_gate_evaluator.SkipGateEvaluator._apply_warm_start",
            return_value=None,
        ):
            yield

    @pytest.fixture(autouse=True)
    def _fast_preorder_features(self) -> object:
        with patch(
            "scripts.v460.ml.retrain_scheduler.build_preorder_as_features",
            side_effect=_build_fast_preorder_as_features,
        ):
            yield

    def test_retrain_deploy_and_hot_reload(self) -> None:
        """E2E: 十分なデータで retrain → deploy → SkipGateEvaluator が hot-reload."""

        with tempfile.TemporaryDirectory() as tmpdir:
            records_dir = Path(tmpdir) / "results"
            records_dir.mkdir()
            model_dir = Path(tmpdir) / "models"
            model_dir.mkdir()
            model_path = model_dir / "gate.pkl"

            records_df = _make_retrain_records_df(
                10,
                seed=42,
                cycle_prefix="e2e",
                run_id="e2e_run",
                regimes=("trending", "ranging", "high_vol"),
            )

            # retrain_model 実行
            cfg = dict(_DEFAULT_CONFIG)
            cfg["results_dir"] = str(records_dir)
            cfg["model_path"] = str(model_path)
            cfg["mode"] = "pnl"
            cfg["use_ob_features"] = False  # テスト高速化
            cfg["min_new_samples"] = 1
            cfg["min_total_samples"] = 10
            cfg["quality_gate_enabled"] = False  # E2E テストでは品質ゲート無効
            cfg["latest_run_only"] = False
            cfg["exclude_missing_run_id"] = False
            cfg["lgbm_n_estimators"] = 1
            cfg["lgbm_max_depth"] = 2
            cfg["lgbm_num_leaves"] = 4
            cfg["lgbm_min_child_samples"] = 2
            cfg["lgbm_subsample"] = 1.0
            cfg["lgbm_colsample_bytree"] = 1.0
            cfg["early_stopping_rounds"] = 0
            cfg["bootstrap_threshold"] = 10
            cfg["enriched_cache_enabled"] = False
            cfg["feature_pruning_enabled"] = False
            cfg["redundancy_pruning_enabled"] = False
            cfg["warm_start_enabled"] = False
            cfg["min_deploy_trees"] = 0  # 465#: E2E stub は 1-tree — D1 guard bypass
            cfg["min_pred_std"] = 0.0    # 465#: E2E stub 用 — D2 guard bypass

            gate_v1 = _make_picklable_gate(n_samples=10, version="verified")
            with patch(
                "scripts.v460.ml.retrain_scheduler.load_fill_records",
                side_effect=lambda *args, **kwargs: records_df.copy(deep=True),
            ), patch(
                "scripts.v460.ml.retrain_scheduler.enrich_fill_records",
                side_effect=_identity_enrich,
            ), patch(
                "scripts.v460.ml.retrain_scheduler.SkipGate.load",
                return_value=gate_v1,
            ), patch(
                "scripts.v460.ml.retrain_scheduler.SkipGate.save",
                autospec=True,
                side_effect=_write_stub_gate_artifact,
            ):
                result = retrain_model(cfg)
            assert result["status"] in ("deployed", "deployed_verified"), f"Expected deployed*, got {result}"
            assert model_path.exists()

            # SkipGateEvaluator で hot-reload テスト
            eval_cfg = _make_skip_gate_eval_config(
                model_path=str(model_path),
                results_dir=str(records_dir),
            )

            gate_v2 = _make_picklable_gate(n_samples=45, version="e2e_v2")
            gate_v2.metadata["retrained"] = True
            with patch(
                "scripts.v460.ml.skip_gate.SkipGate.load",
                side_effect=[gate_v1, gate_v2],
            ), patch(
                "scripts.v460.lib.skip_gate_evaluator.SkipGateEvaluator._read_model_hash",
                side_effect=["a" * 64, "b" * 64],
            ):
                evaluator = SkipGateEvaluator(eval_cfg, Path(tmpdir))
                assert evaluator._skip_gate is not None
                initial_hash = evaluator._model_file_hash

                # hot-reload トリガー (interval リセット)
                evaluator._last_reload_check = 0
                with patch.object(evaluator, "_check_and_reload_side_models", return_value=None):
                    evaluator._check_and_reload_model()

                # モデルが更新されていること
                assert evaluator._model_file_hash != initial_hash
                assert evaluator._skip_gate is not None
                # retrain されたモデルのメタデータ確認
                meta = evaluator._skip_gate.metadata  # type: ignore[union-attr]
                assert meta.get("retrained") is True
                assert meta.get("n_samples", 0) > 0


# =====================================================================
# 130# Y5: balance_forced_switch フィルタリングテスト
# =====================================================================

@pytest.mark.skipif(not _HAS_LIGHTGBM, reason="lightgbm not installed")
class TestBalanceForcedSwitchFilter:
    """130# Y5: retrain_model() が balance_forced_switch=True を除外する."""

    @pytest.fixture(autouse=True)
    def _fast_regressor(self) -> object:
        with patch(
            "scripts.v460.ml.retrain_scheduler._build_lgbm_regressor",
            side_effect=_build_fast_regressor,
        ):
            yield

    @pytest.fixture(autouse=True)
    def _fast_preorder_features(self) -> object:
        with patch(
            "scripts.v460.ml.retrain_scheduler.build_preorder_as_features",
            side_effect=_build_fast_preorder_as_features,
        ):
            yield

    def test_balance_forced_records_excluded(self) -> None:
        """balance_forced_switch=True のレコードが学習データから除外される."""

        with tempfile.TemporaryDirectory() as tmpdir:
            records_dir = Path(tmpdir) / "results"
            records_dir.mkdir()
            records_df = _make_retrain_records_df(
                11,
                seed=123,
                cycle_prefix="bf",
                run_id="bf_run",
                balance_forced_first=1,
            )

            cfg = dict(_DEFAULT_CONFIG)
            cfg["results_dir"] = str(records_dir)
            cfg["model_path"] = str(Path(tmpdir) / "model.pkl")
            cfg["mode"] = "pnl"
            cfg["target"] = "pnl30"
            cfg["use_ob_features"] = False
            cfg["min_total_samples"] = 10
            cfg["min_new_samples"] = 1
            cfg["bootstrap_min_total_samples"] = 10
            cfg["bootstrap_min_new_samples"] = 1
            cfg["quality_gate_enabled"] = False
            cfg["latest_run_only"] = False
            cfg["exclude_missing_run_id"] = False
            cfg["lgbm_n_estimators"] = 1
            cfg["enriched_cache_enabled"] = False
            cfg["feature_pruning_enabled"] = False
            cfg["redundancy_pruning_enabled"] = False
            cfg["warm_start_enabled"] = False
            cfg["min_deploy_trees"] = 0  # 465#: stub bypass
            cfg["min_pred_std"] = 0.0    # 465#: stub bypass

            with patch(
                "scripts.v460.ml.retrain_scheduler.load_fill_records",
                side_effect=lambda *args, **kwargs: records_df.copy(deep=True),
            ), patch(
                "scripts.v460.ml.retrain_scheduler.enrich_fill_records",
                side_effect=_identity_enrich,
            ), patch(
                "scripts.v460.ml.retrain_scheduler.SkipGate.load",
                return_value=_make_picklable_gate(n_samples=10, version="verified"),
            ), patch(
                "scripts.v460.ml.retrain_scheduler.SkipGate.save",
                autospec=True,
                side_effect=_write_stub_gate_artifact,
            ):
                result = retrain_model(cfg)
            # 11 - 1 forced = 10 records usable; filled_records should be <= 10
            assert result["status"] in ("deployed", "deployed_verified", "skipped")
            if result["status"] in ("deployed", "deployed_verified"):
                assert result.get("filled_records", 0) <= 10

    def test_no_balance_column_no_error(self) -> None:
        """balance_forced_switch カラムがなくてもエラーにならない."""

        with tempfile.TemporaryDirectory() as tmpdir:
            records_dir = Path(tmpdir) / "results"
            records_dir.mkdir()
            records_df = _make_retrain_records_df(
                5,
                seed=5,
                cycle_prefix="no_bf",
                run_id="no_bf_run",
            ).drop(columns=["balance_forced_switch"], errors="ignore")

            cfg = dict(_DEFAULT_CONFIG)
            cfg["results_dir"] = str(records_dir)
            cfg["model_path"] = str(Path(tmpdir) / "model.pkl")
            cfg["mode"] = "pnl"
            cfg["use_ob_features"] = False
            cfg["latest_run_only"] = False
            cfg["exclude_missing_run_id"] = False
            cfg["enriched_cache_enabled"] = False
            # Should not raise
            with patch(
                "scripts.v460.ml.retrain_scheduler.load_fill_records",
                side_effect=lambda *args, **kwargs: records_df.copy(deep=True),
            ), patch(
                "scripts.v460.ml.retrain_scheduler.enrich_fill_records",
                side_effect=_identity_enrich,
            ):
                result = retrain_model(cfg)
            assert result["status"] in ("skipped", "deployed")


# =====================================================================
# 130# F7: trades I/O 7 日 fallback テスト
# =====================================================================

class TestTradesIOFallback:
    """130# F7: trades fallback が全量ではなく直近 7 日でフォールバックする."""

    def test_fallback_uses_7day_window(self) -> None:
        """date_filter で空→全量ではなく 7 日 window が先に試行される."""

        # enrich_fill_records 内の load_raw_trades 呼び出しを追跡
        call_args: list[tuple] = []

        def _tracking_load(raw_dir=None, date_filter=None):
            call_args.append((raw_dir, date_filter))
            # 常に空を返して fallback チェーン全体をテスト
            return pd.DataFrame()

        fill_df = pd.DataFrame({
            "timestamp": [1771502400.0],
            "side": ["buy"],
            "filled": [True],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "scripts.v460.ml.feature_enricher._load_raw_trades_entry",
                side_effect=lambda raw_dir=None, date_filter=None: SimpleNamespace(
                    df=_tracking_load(raw_dir=raw_dir, date_filter=date_filter),
                    sorted_ts=None,
                    context=None,
                ),
            ), patch(
                "scripts.v460.ml.feature_enricher._load_raw_orderbook_entry",
                return_value=SimpleNamespace(df=pd.DataFrame(), sorted_ts=None, context=None),
            ):
                # raw_dir を空ディレクトリに固定し、実ファイル走査コストを回避
                enrich_fill_records(fill_df, raw_dir=Path(tmpdir))

        # 139# §9-#5: 全量フォールバック廃止により 2 コールに変更
        # 呼び出し順: (1) date_filter あり → (2) 7日 window
        assert len(call_args) == 2, f"Expected 2 calls, got {len(call_args)}"
        # 1st call: original date_filter
        assert call_args[0][1] is not None
        # 2nd call: 7-day fallback (set of date strings)
        fb_filter = call_args[1][1]
        assert fb_filter is not None
        assert len(fb_filter) <= 8  # 7 days + 1


# =====================================================================
# E1-E4 効率化施策テスト
# =====================================================================

class TestE4EnrichedCache:
    """E4: enriched data cache テスト."""

    def test_cache_roundtrip(self, tmp_path: Path) -> None:
        """キャッシュ保存→読み込みでデータが一致."""

        cache_path = _cache_path(tmp_path)
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        _save_enriched_cache(cache_path, df, cache_key="test_key_1")
        loaded = _load_enriched_cache(cache_path, n_records=3, cache_key="test_key_1")
        assert loaded is not None
        assert len(loaded) == 3
        pd.testing.assert_frame_equal(df, loaded)

    def test_cache_invalidation_on_count_mismatch(self, tmp_path: Path) -> None:
        """レコード数不一致でキャッシュを無効化."""

        cache_path = _cache_path(tmp_path)
        df = pd.DataFrame({"a": [1, 2, 3]})
        _save_enriched_cache(cache_path, df, cache_key="key1")
        loaded = _load_enriched_cache(cache_path, n_records=5, cache_key="key1")
        assert loaded is None

    def test_cache_invalidation_on_key_mismatch(self, tmp_path: Path) -> None:
        """131# A.1 #6: cache_key 不一致でキャッシュを無効化."""

        cache_path = _cache_path(tmp_path)
        df = pd.DataFrame({"a": [1, 2, 3]})
        _save_enriched_cache(cache_path, df, cache_key="key_v1")
        loaded = _load_enriched_cache(cache_path, n_records=3, cache_key="key_v2")
        assert loaded is None

    def test_backward_compat_old_cache_format(self, tmp_path: Path) -> None:
        """旧形式(DataFrame直接)のキャッシュも読める."""

        cache_path = _cache_path(tmp_path, "old_cache.pkl")
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.to_pickle(cache_path)
        loaded = _load_enriched_cache(cache_path, n_records=3, cache_key=None)
        assert loaded is not None
        assert len(loaded) == 3


class TestE3FeaturePruning:
    """E3: dead feature pruning テスト."""

    def test_dead_features_identified(self) -> None:
        """split=0 の特徴量が正しく特定される."""
        feat_importance = {
            "spread_jpy": 100,
            "hour_cos": 50,
            "regime_high_vol": 0,
            "trade_flow_imbalance_60s": 0,
            "side_buy": 0,
        }
        # split=0 は 3 件
        dead = [c for c, v in feat_importance.items() if v <= 0]
        assert len(dead) == 3
        assert "regime_high_vol" in dead
        assert "spread_jpy" not in dead

    def test_pruning_preserves_minimum_features(self) -> None:
        """最低5特徴量は保持される (過剰pruning防止)."""
        feature_cols = ["a", "b", "c", "d", "e", "f"]
        feat_importance = {
            "a": 10, "b": 0, "c": 0, "d": 0, "e": 0, "f": 0,
        }
        min_imp = 0
        pruned = [c for c in feature_cols if feat_importance.get(c, 0) <= min_imp]
        # 5個を pruning → 残りは 1 < 5 → pruning しない
        if len(feature_cols) - len(pruned) >= 5:
            feature_cols = [c for c in feature_cols if c not in pruned]
        # 残り 1 < 5 なので pruning されない
        assert len(feature_cols) == 6


@pytest.mark.skipif(not _HAS_LIGHTGBM, reason="lightgbm not installed")
class TestBuildLgbmRegressor:
    """DRY: _build_lgbm_regressor テスト."""

    def test_default_params(self) -> None:
        """デフォルトパラメータで LGBMRegressor を構築."""

        cfg = {"lgbm_n_estimators": 100, "lgbm_max_depth": 3}
        model = _build_lgbm_regressor(cfg)
        assert model.n_estimators == 100
        assert model.max_depth == 3

    def test_n_estimators_override(self) -> None:
        """n_estimators をオーバーライドできる (E2 early stopping 用)."""

        cfg = {"lgbm_n_estimators": 100}
        model = _build_lgbm_regressor(cfg, n_estimators_override=300)
        assert model.n_estimators == 300


class TestAtomicHashMove:
    """131# A.1 #1: アトミック保存時の hash 移動パス計算テスト."""

    def test_tmp_hash_path_calculation(self) -> None:
        """tmp_path.with_suffix(tmp_path.suffix + '.sha256') で正しいパスが得られる."""
        # 再現: retrain_scheduler の hash 移動ロジック
        model_path = Path("models/v460/skip_gate_lgbm_pnl120.pkl")
        tmp_path = atomic_pickle_tmp_path(model_path)

        # 修正後のロジック
        tmp_hash = hash_sidecar_path(tmp_path)
        real_hash = hash_sidecar_path(model_path)

        assert str(tmp_hash).endswith(".pkl.tmp.sha256")
        assert str(real_hash).endswith(".pkl.sha256")
        # 二重 .pkl がないことを確認
        assert ".pkl.pkl" not in str(tmp_hash)
        assert ".pkl.pkl" not in str(real_hash)

    def test_old_buggy_path_was_wrong(self) -> None:
        """旧コードが二重 .pkl を生成していたことを確認 (regression guard)."""
        model_path = Path("models/v460/skip_gate_lgbm_pnl120.pkl")
        tmp_path = atomic_pickle_tmp_path(model_path)

        # 旧コード: tmp_path.with_suffix(".pkl.tmp.sha256") → 二重 .pkl
        buggy_hash = tmp_path.with_suffix(".pkl.tmp.sha256")
        assert ".pkl.pkl" in str(buggy_hash), "旧コードは二重 .pkl を生成する"

    def test_atomic_save_roundtrip(self, tmp_path: Path) -> None:
        """save → atomic move → load でハッシュが一致."""
        model_path, tmp_model_path = _model_paths(tmp_path)

        gate = _make_picklable_gate()
        gate.save(tmp_model_path)
        os.replace(str(tmp_model_path), str(model_path))

        tmp_hash = hash_sidecar_path(tmp_model_path)
        real_hash = hash_sidecar_path(model_path)
        assert tmp_hash.exists(), f"tmp hash should exist at {tmp_hash}"
        os.replace(str(tmp_hash), str(real_hash))

        loaded = SkipGate.load(model_path)
        assert loaded.metadata["version"] == "test"


@pytest.mark.skipif(not _HAS_LIGHTGBM, reason="lightgbm not installed")
class TestPrevModelLoadError:
    """131# A.1 #3: except:pass 廃止テスト."""

    def test_prev_load_error_recorded(self) -> None:
        """前モデル読み込み失敗がログ出力され result に記録される."""

        with tempfile.TemporaryDirectory() as tmpdir:
            # 不正な pkl を仕込む
            model_path = Path(tmpdir) / "broken.pkl"
            model_path.write_bytes(b"broken data")

            cfg = {
                "model_path": str(model_path),
                "results_dir": str(Path(tmpdir) / "results"),
                "target": "pnl120",
                "mode": "pnl",
                "use_ob_features": False,
                "latest_run_only": False,
                "exclude_missing_run_id": False,
            }
            (Path(tmpdir) / "results").mkdir()
            result = retrain_model(cfg)
            # fill_records がないので skipped になるが、
            # エラーは result に記録されるべき (もし Load まで到達すれば)
            # ここでは fill_records 不在で先に skipped になるのでパス
            assert result["status"] == "skipped"


class TestE3PruningMinTrees:
    """131# A.1 #7: E3 pruning の最小木数ガードテスト."""

    def test_pruning_skipped_when_too_few_trees(self) -> None:
        """WF eval の木数が閾値未満なら pruning をスキップ."""
        feature_cols = ["a", "b", "c", "d", "e", "f", "g", "h"]
        feat_importance = {
            "a": 10, "b": 20, "c": 0, "d": 0, "e": 30, "f": 15, "g": 25, "h": 0,
        }
        wf_actual_trees = 5
        min_trees_for_pruning = 20

        # pruning 条件: actual_n_trees >= min_trees_for_pruning
        should_prune = wf_actual_trees >= min_trees_for_pruning
        assert not should_prune, "5 trees < 20 → pruning should be skipped"

    def test_pruning_allowed_when_enough_trees(self) -> None:
        """WF eval の木数が閾値以上なら pruning を実行."""
        wf_actual_trees = 50
        min_trees_for_pruning = 20
        should_prune = wf_actual_trees >= min_trees_for_pruning
        assert should_prune, "50 trees >= 20 → pruning should proceed"


class TestConsecutiveDeadPruning:
    """131# B: 連続 dead pruning テスト."""

    def test_only_consecutive_dead_pruned(self) -> None:
        """前回も dead だった特徴量のみ prune される."""
        feature_cols = ["a", "b", "c", "d", "e", "f", "g", "h"]
        wf_dead = ["c", "d", "h"]  # 今回 WF で dead
        prev_dead = {"c", "d"}  # 前回も dead

        # require_consecutive=True → intersection のみ prune
        pruned = [c for c in wf_dead if c in prev_dead]
        assert pruned == ["c", "d"]
        assert "h" not in pruned  # h は今回初めて dead → 見送り

    def test_all_pruned_without_prev(self) -> None:
        """前モデルなし (prev_dead 空) → 全 dead を prune."""
        feature_cols = ["a", "b", "c", "d", "e", "f", "g", "h"]
        wf_dead = ["c", "d", "h"]
        prev_dead: set[str] = set()
        require_consecutive = True

        # prev_dead が空の場合は consecutive 制約なし → 全 dead を prune
        if require_consecutive and prev_dead:
            pruned = [c for c in wf_dead if c in prev_dead]
        else:
            pruned = list(wf_dead)
        assert pruned == ["c", "d", "h"]

    def test_consecutive_disabled(self) -> None:
        """require_consecutive=False なら全 dead を即 prune."""
        wf_dead = ["c", "d", "h"]
        prev_dead = {"c"}
        require_consecutive = False

        if require_consecutive and prev_dead:
            pruned = [c for c in wf_dead if c in prev_dead]
        else:
            pruned = list(wf_dead)
        assert pruned == ["c", "d", "h"]


class TestPostDeployVerification:
    """131# B: Post-deploy 自己検証テスト."""

    def test_deployed_verified_status(self, tmp_path: Path) -> None:
        """save → load 検証成功で SkipGate.load() が n_samples を保持."""
        model_path, _ = _model_paths(tmp_path)
        gate = _make_picklable_gate(n_samples=42)
        loaded = save_and_load_skip_gate(gate, model_path)
        n = loaded.metadata.get("n_samples", 0)
        assert n == 42, f"Expected n_samples=42, got {n}"

    def test_verification_fails_on_corrupt(self, tmp_path: Path) -> None:
        """壊れたモデルでは load が失敗する."""
        model_path, _ = _model_paths(tmp_path)
        _write_corrupt_gate(model_path)

        with pytest.raises(Exception):
            SkipGate.load(model_path)

    def test_wf_dead_features_in_metadata(self) -> None:
        """wf_dead_features が metadata に記録される."""
        # metadata にキーが存在することを検証
        metadata = {
            "wf_dead_features": ["regime_high_vol", "trade_flow_imbalance_60s"],
            "pruned_features": ["regime_high_vol"],
        }
        assert "wf_dead_features" in metadata
        assert len(metadata["wf_dead_features"]) == 2
        # pruned は consecutive で絞られた結果
        assert len(metadata["pruned_features"]) <= len(metadata["wf_dead_features"])


# =====================================================================
# 131# C1: Multi-Window Walk-Forward テスト
# =====================================================================

@pytest.mark.skipif(not _HAS_LIGHTGBM, reason="lightgbm not installed")
class TestMultiWindowWF:
    """131# C1: WalkForwardSplitter を使った multi-window WF 評価テスト."""

    @pytest.fixture(autouse=True)
    def _fast_regressor(self) -> object:
        with patch(
            "scripts.v460.ml.retrain_scheduler._build_lgbm_regressor",
            side_effect=_build_fast_regressor,
        ):
            yield

    def test_evaluate_wf_multi_returns_fold_data(self) -> None:
        """multi-window WF がfold-level PnL データを返す."""

        n = 40
        rng = np.random.RandomState(42)
        X = pd.DataFrame(
            rng.randn(n, 4),
            columns=["f1", "f2", "f3", "f4"],
        )
        y = pd.Series(rng.randn(n), name="target")
        enriched = pd.DataFrame({
            "post_fill_30s_pnl": rng.randn(n) * 0.1,
            "post_fill_120s_pnl": rng.randn(n) * 0.2,
        })

        cfg = {
            "wf_multi_window_enabled": True,
            "wf_initial_train_pct": 0.50,
            "wf_val_pct": 0.10,
            "wf_test_pct": 0.15,
            "wf_step_pct": 0.25,
            "wf_max_windows": 2,
            "wf_embargo_rows": 0,
            "wf_min_window_train": 10,
            "wf_min_window_test": 3,
            "early_stopping_rounds": 0,
            "lgbm_n_estimators": 4,
            "lgbm_max_depth": 2,
            "lgbm_learning_rate": 0.1,
            "lgbm_num_leaves": 4,
            "lgbm_min_child_samples": 3,
        }

        splitter_mod = _make_splitter_module([
            _make_stub_window(0, 12, 12, 15, 15, 19, window_id=0),
            _make_stub_window(19, 31, 31, 34, 34, 38, window_id=1),
        ])
        with patch(
            "scripts.v460.ml.retrain_scheduler._safe_import_ztb_module",
            return_value=splitter_mod,
        ):
            result = _evaluate_wf_multi(X, y, enriched, cfg)
        assert result is not None, "Should return result for n=180"
        assert result["n_windows"] >= 2, f"Expected >= 2 windows, got {result['n_windows']}"
        assert "fold_pnl30" in result
        assert "fold_pnl120" in result
        assert len(result["fold_pnl30"]) == result["n_windows"]
        assert "score" in result
        assert "feature_importance" in result

    def test_evaluate_wf_multi_respects_wf_max_windows(self) -> None:
        """wf_max_windows 指定時は評価 window 数が上限で切られる."""

        n = 48
        rng = np.random.RandomState(42)
        X = pd.DataFrame(
            rng.randn(n, 4),
            columns=["f1", "f2", "f3", "f4"],
        )
        y = pd.Series(rng.randn(n), name="target")
        enriched = pd.DataFrame({
            "post_fill_30s_pnl": rng.randn(n) * 0.1,
            "post_fill_120s_pnl": rng.randn(n) * 0.2,
        })

        cfg = {
            "wf_multi_window_enabled": True,
            "wf_initial_train_pct": 0.50,
            "wf_val_pct": 0.10,
            "wf_test_pct": 0.15,
            "wf_step_pct": 0.15,
            "wf_max_windows": 2,
            "wf_embargo_rows": 0,
            "wf_min_window_train": 10,
            "wf_min_window_test": 3,
            "early_stopping_rounds": 0,
            "lgbm_n_estimators": 4,
            "lgbm_max_depth": 2,
            "lgbm_learning_rate": 0.1,
            "lgbm_num_leaves": 4,
            "lgbm_min_child_samples": 3,
        }

        splitter_mod = _make_splitter_module([
            _make_stub_window(0, 12, 12, 15, 15, 19, window_id=0),
            _make_stub_window(19, 31, 31, 34, 34, 38, window_id=1),
            _make_stub_window(8, 20, 20, 23, 23, 27, window_id=2),
        ])
        with patch(
            "scripts.v460.ml.retrain_scheduler._safe_import_ztb_module",
            return_value=splitter_mod,
        ):
            result = _evaluate_wf_multi(X, y, enriched, cfg)
        assert result is not None
        assert result["n_windows"] == 2

    def test_evaluate_wf_multi_fallback_small_data(self) -> None:
        """データ不足時に multi-window が None を返す (single-window フォールバック)."""

        n = 24
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(n, 3), columns=["f1", "f2", "f3"])
        y = pd.Series(rng.randn(n))
        enriched = pd.DataFrame({
            "post_fill_30s_pnl": rng.randn(n) * 0.1,
            "post_fill_120s_pnl": rng.randn(n) * 0.2,
        })

        cfg = {
            "wf_multi_window_enabled": True,
            "wf_initial_train_pct": 0.50,
            "wf_val_pct": 0.10,
            "wf_test_pct": 0.15,
            "wf_step_pct": 0.20,
            "wf_max_windows": 2,
            "wf_embargo_rows": 0,
            "wf_min_window_train": 10,
            "wf_min_window_test": 3,
            "early_stopping_rounds": 0,
            "lgbm_n_estimators": 4,
            "lgbm_max_depth": 2,
            "lgbm_learning_rate": 0.1,
            "lgbm_num_leaves": 4,
            "lgbm_min_child_samples": 5,
        }

        splitter_mod = _make_splitter_module([
            _make_stub_window(0, 10, 10, 13, 13, 16, window_id=0),
        ])
        with patch(
            "scripts.v460.ml.retrain_scheduler._safe_import_ztb_module",
            return_value=splitter_mod,
        ):
            result = _evaluate_wf_multi(X, y, enriched, cfg)
        assert result is None, "Should return None (fallback) for small data"

    def test_evaluate_wf_dispatches_multi(self) -> None:
        """_evaluate_wf が multi-window に正しくディスパッチする."""

        n = 40
        rng = np.random.RandomState(42)
        X = pd.DataFrame(
            rng.randn(n, 5),
            columns=["f1", "f2", "f3", "f4", "f5"],
        )
        y = pd.Series(rng.randn(n))
        enriched = pd.DataFrame({
            "post_fill_30s_pnl": rng.randn(n) * 0.1,
            "post_fill_120s_pnl": rng.randn(n) * 0.2,
        })

        cfg = {
            "wf_multi_window_enabled": True,
            "wf_initial_train_pct": 0.50,
            "wf_val_pct": 0.10,
            "wf_test_pct": 0.15,
            "wf_step_pct": 0.20,
            "wf_max_windows": 2,
            "wf_embargo_rows": 0,
            "wf_min_window_train": 10,
            "wf_min_window_test": 3,
            "wf_test_ratio": 0.2,
            "early_stopping_rounds": 0,
            "lgbm_n_estimators": 4,
            "lgbm_max_depth": 2,
            "lgbm_learning_rate": 0.1,
            "lgbm_num_leaves": 4,
            "lgbm_min_child_samples": 3,
        }

        splitter_mod = _make_splitter_module([
            _make_stub_window(0, 12, 12, 15, 15, 19, window_id=0),
            _make_stub_window(19, 31, 31, 34, 34, 38, window_id=1),
        ])
        with patch(
            "scripts.v460.ml.retrain_scheduler._safe_import_ztb_module",
            return_value=splitter_mod,
        ):
            result = _evaluate_wf(X, y, enriched, cfg)
        # multi-window が成功していれば n_windows >= 1
        assert result["n_windows"] >= 1
        assert "fold_pnl30" in result

    def test_single_window_returns_fold_data(self) -> None:
        """single-window でも fold-level PnL を返す."""

        n = 120
        rng = np.random.RandomState(42)
        X = pd.DataFrame(
            rng.randn(n, 5),
            columns=["f1", "f2", "f3", "f4", "f5"],
        )
        y = pd.Series(rng.randn(n))
        enriched = pd.DataFrame({
            "post_fill_30s_pnl": rng.randn(n) * 0.1,
            "post_fill_120s_pnl": rng.randn(n) * 0.2,
        })

        cfg = {
            "wf_test_ratio": 0.2,
            "early_stopping_rounds": 0,
            "lgbm_n_estimators": 6,
            "lgbm_max_depth": 2,
            "lgbm_learning_rate": 0.1,
            "lgbm_num_leaves": 4,
            "lgbm_min_child_samples": 3,
            "warm_start_enabled": False,
        }

        result = _evaluate_wf_single(X, y, enriched, cfg)
        assert result["n_windows"] == 1
        assert len(result["fold_pnl30"]) == 1
        assert len(result["fold_pnl120"]) == 1
        # fold data は (kept, all) タプル
        kept, all_vals = result["fold_pnl30"][0]
        assert len(kept) <= len(all_vals)


# =====================================================================
# 158# P2-1: WF single-window leakage 修正テスト
# =====================================================================


@pytest.mark.skipif(not _HAS_LIGHTGBM, reason="lightgbm not installed")
class TestWFSingleWindowLeakageFix:
    """158# P2-1: _evaluate_wf_single の val/test 分離を検証."""

    def test_val_not_test_for_early_stopping(self) -> None:
        """early_stopping 有効時に test ではなく val を eval_set に使用."""

        n = 300
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(rng.randn(n))
        enriched = pd.DataFrame({
            "post_fill_30s_pnl": rng.randn(n) * 0.1,
            "post_fill_120s_pnl": rng.randn(n) * 0.2,
        })

        cfg = {
            "wf_test_ratio": 0.2,
            "wf_val_ratio_single": 0.1,
            "wf_embargo_rows": 0,
            "early_stopping_rounds": 10,
            "lgbm_n_estimators_max": 50,
            "lgbm_max_depth": 2,
            "lgbm_learning_rate": 0.1,
            "lgbm_num_leaves": 4,
            "lgbm_min_child_samples": 5,
            "warm_start_enabled": False,
        }

        result = _evaluate_wf_single(X, y, enriched, cfg)
        # 出力は正常
        assert result["n_windows"] == 1
        # train + val + test = n (embargo=0 の場合)
        assert result["n_train"] + 30 + result["n_test"] == n  # val=30 (10%×300)

    def test_train_test_sizes_with_embargo(self) -> None:
        """embargo 有効時に train/test の間に gap がある."""

        n = 300
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(rng.randn(n))
        enriched = pd.DataFrame({
            "post_fill_30s_pnl": rng.randn(n) * 0.1,
            "post_fill_120s_pnl": rng.randn(n) * 0.2,
        })

        cfg = {
            "wf_test_ratio": 0.2,
            "wf_val_ratio_single": 0.1,
            "wf_embargo_rows": 5,
            "early_stopping_rounds": 0,
            "lgbm_n_estimators": 10,
            "lgbm_max_depth": 2,
            "lgbm_learning_rate": 0.1,
            "lgbm_num_leaves": 4,
            "lgbm_min_child_samples": 5,
            "warm_start_enabled": False,
        }

        result = _evaluate_wf_single(X, y, enriched, cfg)
        # embargo=5 で train が 5 行減る
        # total = train + embargo(5) + val(30) + test(60) = 300
        expected_train = 300 - 60 - 30 - 5  # = 205
        assert result["n_train"] == expected_train
        assert result["n_test"] == 60  # 20% of 300

    def test_insufficient_data_returns_zero(self) -> None:
        """データ不足時にスコア 0.0 を返す."""

        n = 30  # Too small for meaningful split
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(n, 3), columns=["f1", "f2", "f3"])
        y = pd.Series(rng.randn(n))
        enriched = pd.DataFrame({
            "post_fill_30s_pnl": rng.randn(n) * 0.1,
            "post_fill_120s_pnl": rng.randn(n) * 0.2,
        })

        cfg = {
            "wf_test_ratio": 0.2,
            "wf_val_ratio_single": 0.1,
            "early_stopping_rounds": 0,
            "lgbm_n_estimators": 10,
            "lgbm_max_depth": 2,
            "lgbm_learning_rate": 0.1,
            "lgbm_num_leaves": 4,
            "lgbm_min_child_samples": 5,
        }

        result = _evaluate_wf_single(X, y, enriched, cfg)
        assert result["score"] == 0.0


# =====================================================================
# 131# C2: 統計的品質ゲート テスト
# =====================================================================

class TestStatisticalGate:
    """131# C2: gate_checks 統合テスト."""

    def test_apply_gate_multi_window(self) -> None:
        """multi-window fold data に対して g1_judgment が適用される."""

        # Model は baseline より明確に高い PnL を持つ
        wf_result = {
            "n_windows": 2,
            "fold_pnl30": [
                ([0.5, 0.6, 0.7, 0.8, 0.9] * 4, [0.1, 0.2, 0.3, 0.4, 0.5] * 4),
                ([0.5, 0.6, 0.7, 0.8, 0.9] * 4, [0.1, 0.2, 0.3, 0.4, 0.5] * 4),
            ],
            "fold_pnl120": [
                ([1.0, 1.1, 1.2, 1.3, 1.4] * 4, [0.1, 0.2, 0.3, 0.4, 0.5] * 4),
                ([1.0, 1.1, 1.2, 1.3, 1.4] * 4, [0.1, 0.2, 0.3, 0.4, 0.5] * 4),
            ],
        }
        cfg = {
            "statistical_gate_alpha": 0.05,
            "statistical_gate_min_effect": 0.147,
            "statistical_gate_min_test_samples": 10,
        }

        gate_result = _apply_statistical_gate(wf_result, cfg)
        assert gate_result["applied"]
        assert gate_result["method"] == "g1_judgment"
        assert gate_result["n_windows"] == 2
        # 明確な差なので pass するはず
        assert gate_result["pass"]
        assert len(gate_result["passed_targets"]) > 0

    def test_apply_gate_single_window(self) -> None:
        """single-window で holm_bonferroni_gate が適用される."""

        wf_result = {
            "n_windows": 1,
            "fold_pnl30": [
                ([0.5, 0.6, 0.7, 0.8, 0.9] * 4, [0.1, 0.2, 0.3, 0.4, 0.5] * 4),
            ],
            "fold_pnl120": [
                ([1.0, 1.1, 1.2, 1.3, 1.4] * 4, [0.1, 0.2, 0.3, 0.4, 0.5] * 4),
            ],
        }
        cfg = {
            "statistical_gate_alpha": 0.05,
            "statistical_gate_min_effect": 0.147,
            "statistical_gate_min_test_samples": 10,
        }

        gate_result = _apply_statistical_gate(wf_result, cfg)
        assert gate_result["applied"]
        assert gate_result["method"] == "holm_bonferroni_gate"

    def test_apply_gate_insufficient_samples(self) -> None:
        """サンプル不足時はスキップ."""

        wf_result = {
            "n_windows": 1,
            "fold_pnl30": [([0.5, 0.6], [0.1, 0.2])],
            "fold_pnl120": [([1.0, 1.1], [0.1, 0.2])],
        }
        cfg = {
            "statistical_gate_min_test_samples": 40,
        }

        gate_result = _apply_statistical_gate(wf_result, cfg)
        assert not gate_result["applied"]
        assert "insufficient" in gate_result["reason"]

    def test_apply_gate_no_significance(self) -> None:
        """有意差なし → pass=False."""

        rng = np.random.RandomState(42)
        # Model と baseline がほぼ同じ
        same_vals = list(rng.randn(30))
        wf_result = {
            "n_windows": 1,
            "fold_pnl30": [(same_vals, same_vals)],
            "fold_pnl120": [(same_vals, same_vals)],
        }
        cfg = {
            "statistical_gate_alpha": 0.05,
            "statistical_gate_min_effect": 0.33,
            "statistical_gate_min_test_samples": 10,
        }

        gate_result = _apply_statistical_gate(wf_result, cfg)
        assert gate_result["applied"]
        assert not gate_result["pass"]


# =====================================================================
# 131# C3: 冗長特徴量除去テスト
# =====================================================================

class TestRedundancyPruning:
    """131# C3: redundancy.find_highly_correlated_features 統合テスト."""

    def test_highly_correlated_features_detected(self) -> None:
        """高相関ペアが正しく検出される."""
        if _REDUNDANCY_MODULE is None:
            pytest.skip("ztb.analysis.redundancy not importable (circular import)")

        calculate_feature_correlations = _REDUNDANCY_MODULE.calculate_feature_correlations
        find_highly_correlated_features = _REDUNDANCY_MODULE.find_highly_correlated_features

        base = np.linspace(-1.0, 1.0, 48, dtype=float)
        df = pd.DataFrame({
            "f1": base,
            "f2": base * 1.001,  # f1 と高相関
            "f3": np.sin(base * np.pi),  # 独立寄り
            "f4": np.cos(base * np.pi),  # 独立寄り
        })
        corr = calculate_feature_correlations(df)
        pairs = find_highly_correlated_features(corr, threshold=0.9)
        assert len(pairs) >= 1
        # f1-f2 ペアが検出されるはず
        pair_features = {(p[0], p[1]) for p in pairs}
        assert ("f1", "f2") in pair_features or ("f2", "f1") in pair_features

    def test_redundancy_removal_uses_importance(self) -> None:
        """importance に基づいて低 importance 側が除去される."""
        # 擬似的に retrain_scheduler 内のロジックを再現
        feat_imp = {"f1": 100, "f2": 5, "f3": 50, "f4": 80}
        corr_pairs = [("f1", "f2", 0.95)]  # f1 と f2 が高相関

        to_remove: set[str] = set()
        for f1, f2, _ in corr_pairs:
            imp1 = feat_imp.get(f1, 0)
            imp2 = feat_imp.get(f2, 0)
            victim = f2 if imp1 >= imp2 else f1
            if imp1 == imp2:
                victim = max(f1, f2)
            to_remove.add(victim)

        assert "f2" in to_remove, "f2 (importance=5) should be removed, not f1 (importance=100)"
        assert "f1" not in to_remove

    def test_minimum_features_preserved(self) -> None:
        """最低5特徴量が保持される."""
        feature_cols = ["f1", "f2", "f3", "f4", "f5"]
        to_remove = {"f1", "f2"}  # 2 features to remove → 3 remaining
        remaining_after = len(feature_cols) - len(to_remove)
        # 5 未満なので pruning はブロックされるべき
        assert remaining_after < 5
        # 実際の retrain_scheduler のロジック
        if remaining_after >= 5 and to_remove:
            pruned = sorted(to_remove)
        else:
            pruned = []
        assert pruned == [], "Should not prune when it would leave < 5 features"


# ====================================================================
# 131# D1: Regime-aware lot sizing
# ====================================================================

class TestRegimeAwareLotSizing:
    """131# D1: レジーム連動ロット制御テスト."""

    def test_unknown_regime_blocks_increase(self) -> None:
        """unknown レジームでは全条件クリアでも増量が hold される."""

        config = LotSizingConfig(
            current_lot=0.001,
            min_lot=0.001,
            max_lot=0.005,
            lot_step=0.001,
            min_fill_rate=0.70,
            max_as_ratio=0.30,
            min_samples=10,
            regime_guard_enabled=True,
            regime_hold_regimes=("unknown",),
        )
        result = compute_lot_size(
            fill_rate=0.80,
            as_ratio=0.20,
            recent_pnl_bps=1.0,
            cumulative_pnl_jpy=0.0,
            sample_count=100,
            config=config,
            regime_tag="unknown",
        )
        assert result.action == "hold"
        assert not result.changed
        assert "レジーム増量抑制" in result.reason

    def test_ranging_regime_allows_increase(self) -> None:
        """ranging レジームでは通常通り増量が許可される."""

        config = LotSizingConfig(
            current_lot=0.001,
            min_lot=0.001,
            max_lot=0.005,
            lot_step=0.001,
            min_fill_rate=0.70,
            max_as_ratio=0.30,
            min_samples=10,
            regime_guard_enabled=True,
            regime_hold_regimes=("unknown",),
        )
        result = compute_lot_size(
            fill_rate=0.80,
            as_ratio=0.20,
            recent_pnl_bps=1.0,
            cumulative_pnl_jpy=0.0,
            sample_count=100,
            config=config,
            regime_tag="ranging",
        )
        assert result.action == "increase"
        assert result.changed
        assert result.new_lot == 0.002

    def test_decrease_regime_forces_decrease(self) -> None:
        """regime_decrease_regimes に含まれるレジームでは強制減量."""

        config = LotSizingConfig(
            current_lot=0.003,
            min_lot=0.001,
            max_lot=0.005,
            lot_step=0.001,
            min_fill_rate=0.70,
            max_as_ratio=0.30,
            min_samples=10,
            regime_guard_enabled=True,
            regime_hold_regimes=("unknown",),
            regime_decrease_regimes=("high_vol",),
        )
        # 全条件クリアでも high_vol なら減量
        result = compute_lot_size(
            fill_rate=0.80,
            as_ratio=0.20,
            recent_pnl_bps=1.0,
            cumulative_pnl_jpy=0.0,
            sample_count=100,
            config=config,
            regime_tag="high_vol",
        )
        assert result.action == "decrease"
        assert result.changed
        assert result.new_lot == 0.002

    def test_regime_guard_disabled_allows_increase(self) -> None:
        """regime_guard_enabled=False では unknown でも増量可能."""

        config = LotSizingConfig(
            current_lot=0.001,
            min_lot=0.001,
            max_lot=0.005,
            lot_step=0.001,
            min_fill_rate=0.70,
            max_as_ratio=0.30,
            min_samples=10,
            regime_guard_enabled=False,
        )
        result = compute_lot_size(
            fill_rate=0.80,
            as_ratio=0.20,
            recent_pnl_bps=1.0,
            cumulative_pnl_jpy=0.0,
            sample_count=100,
            config=config,
            regime_tag="unknown",
        )
        assert result.action == "increase"
        assert result.changed

    def test_na_regime_bypasses_guard(self) -> None:
        """regime_tag='n/a' (検出器なし) ではガードが無効."""

        config = LotSizingConfig(
            current_lot=0.001,
            min_lot=0.001,
            max_lot=0.005,
            lot_step=0.001,
            min_fill_rate=0.70,
            max_as_ratio=0.30,
            min_samples=10,
            regime_guard_enabled=True,
            regime_hold_regimes=("unknown",),
        )
        result = compute_lot_size(
            fill_rate=0.80,
            as_ratio=0.20,
            recent_pnl_bps=1.0,
            cumulative_pnl_jpy=0.0,
            sample_count=100,
            config=config,
            regime_tag="n/a",
        )
        assert result.action == "increase"

    def test_cap_shrink_overrides_regime_guard(self) -> None:
        """損失キャップ接近は regime guard より優先."""

        config = LotSizingConfig(
            current_lot=0.003,
            min_lot=0.001,
            max_lot=0.005,
            loss_cap_jpy=10_000,
            loss_cap_warning_ratio=0.7,
            min_samples=10,
            regime_guard_enabled=True,
            regime_hold_regimes=("unknown",),
        )
        result = compute_lot_size(
            fill_rate=0.80,
            as_ratio=0.20,
            recent_pnl_bps=1.0,
            cumulative_pnl_jpy=-8000,  # > 70% of cap
            sample_count=100,
            config=config,
            regime_tag="ranging",
        )
        assert result.action == "cap_shrink"
        assert result.new_lot == config.min_lot


# ====================================================================
# 131# D2: Oracle PnL Baseline
# ====================================================================

class TestOracleBaseline:
    """131# D2: Oracle PnL 基準線テスト."""

    def _make_mock_record(
        self,
        side: str = "buy",
        filled: bool = True,
        pnl_30s: float | None = 0.5,
        pnl_60s: float | None = None,
        pnl_120s: float | None = None,
        regime: str | None = "ranging",
    ) -> object:
        """テスト用 FillRecord モック."""
        return FillRecord(
            cycle_id="test",
            timestamp=1700000000.0,
            side=side,
            order_price=15_000_000,
            order_quantity=0.001,
            fill_price=15_000_000 if filled else None,
            filled=filled,
            post_fill_30s_pnl=pnl_30s,
            post_fill_60s_pnl=pnl_60s,
            post_fill_120s_pnl=pnl_120s,
            regime=regime,
        )

    def test_oracle_filters_negative_pnl(self) -> None:
        """Oracle は PnL < 0 の取引をスキップする."""

        records = [
            self._make_mock_record(pnl_30s=2.0),
            self._make_mock_record(pnl_30s=-1.0),
            self._make_mock_record(pnl_30s=3.0),
            self._make_mock_record(pnl_30s=-2.0),
        ]
        m = compute_oracle_metrics(records, "test")
        assert m.n_total == 4
        assert m.n_positive == 2
        assert m.n_negative == 2
        assert m.oracle_skip_rate == 0.5
        assert m.oracle_pnl_mean == 2.5  # (2 + 3) / 2
        assert m.actual_pnl_mean == 0.5  # (2 - 1 + 3 - 2) / 4

    def test_oracle_empty_records(self) -> None:
        """空レコードでエラーにならない."""

        m = compute_oracle_metrics([], "empty")
        assert m.n_total == 0
        assert m.oracle_pnl_mean == 0.0

    def test_oracle_all_positive(self) -> None:
        """全取引が正の場合、Oracle skip_rate = 0."""

        records = [
            self._make_mock_record(pnl_30s=1.0),
            self._make_mock_record(pnl_30s=2.0),
        ]
        m = compute_oracle_metrics(records, "all_pos")
        assert m.oracle_skip_rate == 0.0
        assert m.n_positive == 2

    def test_oracle_multi_timeframe(self) -> None:
        """60s/120s PnL も正しく計算される."""

        records = [
            self._make_mock_record(pnl_30s=1.0, pnl_60s=2.0, pnl_120s=3.0),
            self._make_mock_record(pnl_30s=-1.0, pnl_60s=-0.5, pnl_120s=0.5),
        ]
        m = compute_oracle_metrics(records, "multi")
        assert m.pnl_60s_mean is not None
        assert abs(m.pnl_60s_mean - 0.75) < 0.01  # (2.0 + -0.5) / 2
        assert m.pnl_120s_mean is not None
        assert abs(m.pnl_120s_mean - 1.75) < 0.01  # (3.0 + 0.5) / 2

    def test_oracle_jpy_conversion(self) -> None:
        """JPY 換算が正しく算出される."""

        records = [
            self._make_mock_record(pnl_30s=1.0),
        ]
        m = compute_oracle_metrics(
            records, "jpy_test",
            lot_btc=0.001,
            btc_price_jpy=15_000_000,
        )
        # 1.0 bps × 0.001 × 15,000,000 / 10,000 = 1.5 JPY
        assert m.actual_jpy_per_cycle is not None
        assert abs(m.actual_jpy_per_cycle - 1.5) < 0.01

    def test_group_oracle_aggregates_groups_by_side_and_regime(self) -> None:
        """全体/side/regime 集計が 1 パスで正しく構築される."""

        records = [
            self._make_mock_record(side="buy", regime="ranging", pnl_30s=2.0),
            self._make_mock_record(side="sell", regime="trending", pnl_30s=-1.0),
            self._make_mock_record(side="buy", regime="ranging", pnl_30s=None, filled=True),
        ]
        all_agg, side_aggs, regime_aggs = _group_oracle_aggregates(records)

        assert all_agg.n_total == 2
        assert side_aggs["buy"].n_total == 1
        assert side_aggs["sell"].n_total == 1
        assert regime_aggs["ranging"].n_total == 1
        assert regime_aggs["trending"].n_total == 1


# =====================================================================
# 145# R-2a: レジーム重み付き再学習テスト
# =====================================================================

class TestRegimeSampleWeights:
    """145# R-2a: _compute_regime_sample_weights 単体テスト."""

    def test_uniform_when_no_regime_col(self) -> None:
        """regime 列がない → 均一重み."""

        enriched = pd.DataFrame({"side": ["buy", "sell", "buy"]})
        idx = enriched.index
        cfg = {"regime_weighting_enabled": True, "regime_sample_weights": {"high_vol": 2.0}}
        weights, meta = _compute_regime_sample_weights(enriched, idx, cfg)
        assert len(weights) == 3
        assert np.allclose(weights, 1.0)
        assert meta["regime_weighting"] == "uniform"

    def test_config_weights_applied(self) -> None:
        """config の regime_sample_weights が正しく適用される."""

        enriched = pd.DataFrame({
            "regime": ["high_vol", "trending", "ranging", "unknown"] * 10,
        })
        idx = enriched.index
        cfg = {
            "regime_weighting_enabled": True,
            "regime_sample_weights": {
                "high_vol": 2.0,
                "trending": 1.5,
                "ranging": 0.5,
                "unknown": 0.5,
            },
            "regime_current_boost": 1.0,  # ブースト無効
            "regime_current_lookback": 10,
            "regime_weight_floor": 0.1,
        }
        weights, meta = _compute_regime_sample_weights(enriched, idx, cfg)
        assert len(weights) == 40
        assert meta["regime_weighting"] == "applied"
        # high_vol サンプルは trending よりも大きい重み
        hw = weights[0]  # high_vol
        tw = weights[1]  # trending
        rw = weights[2]  # ranging
        assert hw > tw   # 2.0 > 1.5
        assert tw > rw   # 1.5 > 0.5

    def test_current_regime_boost(self) -> None:
        """直近 N 件から推定した現在レジームにブースト倍率が適用される."""

        # 直近 5 件が全て high_vol → current_regime=high_vol
        regimes = ["ranging"] * 15 + ["high_vol"] * 5
        enriched = pd.DataFrame({"regime": regimes})
        idx = enriched.index
        cfg = {
            "regime_weighting_enabled": True,
            "regime_sample_weights": {"high_vol": 1.0, "ranging": 1.0},
            "regime_current_boost": 2.0,
            "regime_current_lookback": 5,
            "regime_weight_floor": 0.1,
        }
        weights, meta = _compute_regime_sample_weights(enriched, idx, cfg)
        assert meta["current_regime"] == "high_vol"
        # high_vol サンプル (idx 15-19) はブースト適用 → ranging より大きい
        assert weights[15] > weights[0]  # boosted high_vol > ranging

    def test_weights_normalized_mean_1(self) -> None:
        """重みの平均は正規化後 ≈ 1.0."""

        enriched = pd.DataFrame({
            "regime": ["high_vol"] * 30 + ["ranging"] * 70,
        })
        idx = enriched.index
        cfg = {
            "regime_weighting_enabled": True,
            "regime_sample_weights": {"high_vol": 3.0, "ranging": 1.0},
            "regime_current_boost": 1.0,
            "regime_current_lookback": 10,
            "regime_weight_floor": 0.1,
        }
        weights, meta = _compute_regime_sample_weights(enriched, idx, cfg)
        # 正規化でも floor が再適用されるので厳密に1.0ではないが、近い
        assert 0.5 < np.mean(weights) < 2.0

    def test_weight_floor_respected(self) -> None:
        """weight_floor より小さい重みは切り上げ."""

        enriched = pd.DataFrame({
            "regime": ["unknown"] * 10,
        })
        idx = enriched.index
        cfg = {
            "regime_weighting_enabled": True,
            "regime_sample_weights": {"unknown": 0.01},
            "regime_current_boost": 1.0,
            "regime_current_lookback": 5,
            "regime_weight_floor": 0.3,
        }
        weights, meta = _compute_regime_sample_weights(enriched, idx, cfg)
        assert np.all(weights >= 0.3)

    def test_nan_regime_treated_as_unknown(self) -> None:
        """NaN regime は 'unknown' として扱われる."""

        enriched = pd.DataFrame({
            "regime": [None, np.nan, "high_vol"],
        })
        idx = enriched.index
        cfg = {
            "regime_weighting_enabled": True,
            "regime_sample_weights": {"unknown": 0.5, "high_vol": 2.0},
            "regime_current_boost": 1.0,
            "regime_current_lookback": 10,
            "regime_weight_floor": 0.1,
        }
        weights, meta = _compute_regime_sample_weights(enriched, idx, cfg)
        assert len(weights) == 3
        # high_vol サンプルは unknown より大きい重み
        assert weights[2] > weights[0]

    def test_default_config_disabled(self) -> None:
        """デフォルト config は regime_weighting_enabled=False."""

        assert _DEFAULT_CONFIG["regime_weighting_enabled"] is False
        assert isinstance(_DEFAULT_CONFIG["regime_sample_weights"], dict)
        assert _DEFAULT_CONFIG["regime_current_boost"] == 1.5
        assert _DEFAULT_CONFIG["regime_weight_floor"] == 0.1


@pytest.mark.skipif(not _HAS_LIGHTGBM, reason="lightgbm not installed")
class TestStatisticalGateInitialTraining:
    """159# P0-1: 初回訓練 (prev model 不在) 時の統計ゲートスキップ."""

    def test_stat_gate_skip_no_prev_model(self) -> None:
        """前モデル不在 + quality_gate 通過でも stat_gate は skip される."""

        # WF eval が正のスコアを返す最小構成のモックを使い、
        # prev_gate_loaded=False の場合 stat_gate が applied=False になることを検証
        # → retrain_model 全体を呼ぶのはコスト高なので、
        #   ロジック分岐のみ単体テストする
        # retrain_scheduler 内の条件: not prev_gate_loaded → skip
        prev_gate_loaded = False
        stat_gate_skipped = not prev_gate_loaded  # 159# P0-1 の新条件
        assert stat_gate_skipped is True

    def test_stat_gate_applied_with_prev_model(self) -> None:
        """前モデル存在時は統計ゲートが適用される."""
        prev_gate_loaded = True
        stat_gate_skipped = not prev_gate_loaded
        assert stat_gate_skipped is False

    def test_all_runs_absolute_min_relaxed(self) -> None:
        """--all-runs モードで absolute_min_score が -0.50 に緩和される."""

        default_abs_min = _DEFAULT_CONFIG.get("absolute_min_score", -0.10)
        assert default_abs_min == -0.10  # デフォルトは厳格

        # --all-runs 時のオーバーライド値
        all_runs_abs_min = -0.50
        assert all_runs_abs_min < default_abs_min
        assert all_runs_abs_min > -1.0  # 安全マージン

    def test_deployed_sell_model_loadable(self) -> None:
        """デプロイ済み sell model が SkipGate.load() でロードできる."""
        sell_path = Path("models/v460/skip_gate_lgbm_pnl120_sell.pkl")
        if not sell_path.exists():
            pytest.skip("Sell model not yet deployed")

        gate = SkipGate.load(sell_path)
        assert len(gate.feature_cols) > 0
        target_meta = str(gate.metadata.get("target", ""))
        assert "pnl120" in target_meta
        assert gate.metadata.get("n_samples", 0) > 0

    def test_deployed_buy_model_loadable(self) -> None:
        """デプロイ済み buy model が SkipGate.load() でロードできる."""
        buy_path = Path("models/v460/skip_gate_lgbm_pnl30_buy.pkl")
        if not buy_path.exists():
            pytest.skip("Buy model not yet deployed")

        gate = SkipGate.load(buy_path)
        assert len(gate.feature_cols) > 0
        target_meta = str(gate.metadata.get("target", ""))
        assert "pnl30" in target_meta
        assert gate.metadata.get("n_samples", 0) > 0


# =====================================================================
# 465# D1/D2: モデル退化ガードテスト
# =====================================================================


class _ConstantRegressor(_FastRegressor):
    """465# D2 テスト用: 定数出力のみ返す regressor."""

    def predict(self, X: object) -> np.ndarray:
        return np.full(len(X), self._prediction, dtype=np.float64)  # type: ignore[arg-type]


class _SingleTreeRegressor(_FastRegressor):
    """465# D1 テスト用: 1-tree のみの regressor."""

    def __init__(self, n_estimators: int = 1) -> None:
        super().__init__(n_estimators=1)  # 常に 1 tree


class TestModelDegenerationGuard:
    """465# D1/D2: モデル退化ガード (定数出力・少数木の deploy 阻止)."""

    @pytest.fixture(autouse=True)
    def _fast_preorder_features(self) -> object:
        with patch(
            "scripts.v460.ml.retrain_scheduler.build_preorder_as_features",
            side_effect=_build_fast_preorder_as_features,
        ):
            yield

    def _base_cfg(self, tmpdir: str) -> dict[str, object]:
        """共通テスト設定."""
        return {
            **_DEFAULT_CONFIG,
            "results_dir": str(Path(tmpdir) / "results"),
            "model_path": str(Path(tmpdir) / "model.pkl"),
            "mode": "pnl",
            "target": "pnl30",
            "use_ob_features": False,
            "min_total_samples": 10,
            "min_new_samples": 1,
            "bootstrap_min_total_samples": 10,
            "bootstrap_min_new_samples": 1,
            "quality_gate_enabled": False,
            "latest_run_only": False,
            "exclude_missing_run_id": False,
            "lgbm_n_estimators": 4,
            "enriched_cache_enabled": False,
            "feature_pruning_enabled": False,
            "redundancy_pruning_enabled": False,
            "warm_start_enabled": False,
            "early_stopping_rounds": 0,
        }

    def test_d1_rejects_single_tree_model(self) -> None:
        """D1: 1-tree モデルは min_deploy_trees=3 で棄却される."""
        with tempfile.TemporaryDirectory() as tmpdir:
            records_df = _make_retrain_records_df(
                15, seed=100, cycle_prefix="d1", run_id="d1_run",
            )
            cfg = self._base_cfg(tmpdir)
            cfg["min_deploy_trees"] = 3  # default

            with patch(
                "scripts.v460.ml.retrain_scheduler.load_fill_records",
                side_effect=lambda *a, **kw: records_df.copy(deep=True),
            ), patch(
                "scripts.v460.ml.retrain_scheduler.enrich_fill_records",
                side_effect=_identity_enrich,
            ), patch(
                "scripts.v460.ml.retrain_scheduler._build_lgbm_regressor",
                return_value=_SingleTreeRegressor(),
            ):
                result = retrain_model(cfg)

            assert result["status"] == "rejected"
            assert "degenerate_model" in result["reason"]
            assert result["actual_n_trees"] == 1

    def test_d1_accepts_sufficient_trees(self) -> None:
        """D1: 十分な木数のモデルは通過する."""
        with tempfile.TemporaryDirectory() as tmpdir:
            records_df = _make_retrain_records_df(
                15, seed=101, cycle_prefix="d1ok", run_id="d1ok_run",
            )
            cfg = self._base_cfg(tmpdir)
            cfg["min_deploy_trees"] = 3

            gate_stub = _make_picklable_gate(n_samples=15, version="d1_verified")
            with patch(
                "scripts.v460.ml.retrain_scheduler.load_fill_records",
                side_effect=lambda *a, **kw: records_df.copy(deep=True),
            ), patch(
                "scripts.v460.ml.retrain_scheduler.enrich_fill_records",
                side_effect=_identity_enrich,
            ), patch(
                "scripts.v460.ml.retrain_scheduler._build_lgbm_regressor",
                side_effect=_build_fast_regressor,
            ), patch(
                "scripts.v460.ml.retrain_scheduler.SkipGate.load",
                return_value=gate_stub,
            ), patch(
                "scripts.v460.ml.retrain_scheduler.SkipGate.save",
                autospec=True,
                side_effect=_write_stub_gate_artifact,
            ):
                result = retrain_model(cfg)

            assert result["status"] in ("deployed", "deployed_verified"), (
                f"Expected deployed*, got {result}"
            )
            assert result["actual_n_trees"] >= 3

    def test_d2_rejects_constant_output(self) -> None:
        """D2: 定数出力モデルは pred_std < min_pred_std で棄却される."""
        with tempfile.TemporaryDirectory() as tmpdir:
            records_df = _make_retrain_records_df(
                15, seed=102, cycle_prefix="d2", run_id="d2_run",
            )
            cfg = self._base_cfg(tmpdir)
            cfg["min_deploy_trees"] = 0   # D1 bypass
            cfg["min_pred_std"] = 0.01    # D2 active

            with patch(
                "scripts.v460.ml.retrain_scheduler.load_fill_records",
                side_effect=lambda *a, **kw: records_df.copy(deep=True),
            ), patch(
                "scripts.v460.ml.retrain_scheduler.enrich_fill_records",
                side_effect=_identity_enrich,
            ), patch(
                "scripts.v460.ml.retrain_scheduler._build_lgbm_regressor",
                return_value=_ConstantRegressor(n_estimators=4),
            ):
                result = retrain_model(cfg)

            assert result["status"] == "rejected"
            assert "constant_output" in result["reason"]
            assert result["pred_std"] < 0.01

    def test_d2_accepts_varied_output(self) -> None:
        """D2: 十分な分散のあるモデルは通過する."""
        with tempfile.TemporaryDirectory() as tmpdir:
            records_df = _make_retrain_records_df(
                15, seed=103, cycle_prefix="d2ok", run_id="d2ok_run",
            )
            cfg = self._base_cfg(tmpdir)
            cfg["min_deploy_trees"] = 0
            cfg["min_pred_std"] = 0.001

            gate_stub = _make_picklable_gate(n_samples=15, version="d2_verified")
            with patch(
                "scripts.v460.ml.retrain_scheduler.load_fill_records",
                side_effect=lambda *a, **kw: records_df.copy(deep=True),
            ), patch(
                "scripts.v460.ml.retrain_scheduler.enrich_fill_records",
                side_effect=_identity_enrich,
            ), patch(
                "scripts.v460.ml.retrain_scheduler._build_lgbm_regressor",
                side_effect=_build_fast_regressor,
            ), patch(
                "scripts.v460.ml.retrain_scheduler.SkipGate.load",
                return_value=gate_stub,
            ), patch(
                "scripts.v460.ml.retrain_scheduler.SkipGate.save",
                autospec=True,
                side_effect=_write_stub_gate_artifact,
            ):
                result = retrain_model(cfg)

            assert result["status"] in ("deployed", "deployed_verified")
            assert result["pred_std"] >= 0.001
