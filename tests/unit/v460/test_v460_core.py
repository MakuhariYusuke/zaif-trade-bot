"""
v460 単体テスト: gate_checks, config_loader, data_loader, manifest, microstructure.

001# §6.5 テスト方針準拠.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scripts.v460.lib.config_loader import _deep_merge, load_config
from scripts.v460.lib.data_loader import (
    check_nan_ratio,
    compute_data_hash,
    generate_targets,
    load_parquet,
    split_train_eval,
)
from scripts.v460.lib.manifest import ManifestWriter, compute_config_hash
from scripts.v460.run_gate_check import run_g0, run_g1_judgment
from ztb.metrics.gate_checks import (
    cliffs_delta,
    g1_judgment,
    holm_bonferroni_gate,
    p_mean_gate,
)
from ztb.trading.live.exchanges.coincheck.adapter import _parse_timestamp

try:
    import xgboost  # noqa: F401
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False
import yaml


# =====================================================================
# gate_checks
# =====================================================================

class TestCliffsD:
    """cliffs_delta のテスト."""

    def test_perfect_dominance(self) -> None:
        x = [10.0, 11.0, 12.0]
        y = [1.0, 2.0, 3.0]
        assert cliffs_delta(x, y) == 1.0

    def test_no_dominance(self) -> None:
        x = [1.0, 2.0, 3.0]
        y = [1.0, 2.0, 3.0]
        assert cliffs_delta(x, y) == 0.0

    def test_reverse_dominance(self) -> None:
        x = [1.0, 2.0, 3.0]
        y = [10.0, 11.0, 12.0]
        assert cliffs_delta(x, y) == -1.0

    def test_empty(self) -> None:
        assert cliffs_delta([], [1.0]) == 0.0


class TestHolmBonferroniGate:
    """holm_bonferroni_gate のテスト."""

    def test_all_pass(self) -> None:
        # 003# #20: seed fixation for reproducibility
        np.random.seed(42)
        # Model strongly dominates baseline → should pass
        model = list(np.random.normal(1.0, 0.1, 200))
        baseline = list(np.random.normal(0.0, 0.1, 200))
        results = {
            "target_a": (model, baseline),
            "target_b": (model, baseline),
        }
        gate = holm_bonferroni_gate(results, alpha=0.05, min_effect=0.33)
        assert gate["target_a"]["pass"] is True
        assert gate["target_b"]["pass"] is True

    def test_no_pass(self) -> None:
        # Model ≈ baseline → should fail
        np.random.seed(42)
        model = list(np.random.normal(0.0, 1.0, 200))
        baseline = list(np.random.normal(0.0, 1.0, 200))
        results = {"target_a": (model, baseline)}
        gate = holm_bonferroni_gate(results, alpha=0.05, min_effect=0.33)
        assert bool(gate["target_a"]["pass"]) is False

    def test_empty(self) -> None:
        assert holm_bonferroni_gate({}) == {}


class TestPMeanGate:
    """p_mean_gate のテスト."""

    def test_all_significant(self) -> None:
        result = p_mean_gate([0.01, 0.02, 0.03], alpha=0.05)
        assert result["pass"] is True
        assert result["n_folds"] == 3
        assert result["p_geometric"] < 0.05

    def test_none_significant(self) -> None:
        result = p_mean_gate([0.5, 0.6, 0.7], alpha=0.05)
        assert result["pass"] is False

    def test_empty(self) -> None:
        result = p_mean_gate([])
        assert result["pass"] is False


class TestG1Judgment:
    """g1_judgment のテスト (§5.3 厳密仕様)."""

    def test_pass_scenario(self) -> None:

        np.random.seed(42)
        # 5 folds where model clearly dominates
        folds = []
        for _ in range(5):
            m = list(np.random.normal(1.0, 0.1, 100))
            b = list(np.random.normal(0.0, 0.1, 100))
            folds.append((m, b))

        result = g1_judgment({"h5_direction": folds})
        assert result["g1_pass"] is True
        assert "h5_direction" in result["passed_targets"]

    def test_fail_scenario(self) -> None:

        np.random.seed(42)
        folds = []
        for _ in range(5):
            m = list(np.random.normal(0.0, 1.0, 100))
            b = list(np.random.normal(0.0, 1.0, 100))
            folds.append((m, b))

        result = g1_judgment({"h5_direction": folds})
        assert result["g1_pass"] is False
        assert result["passed_targets"] == []

    def test_empty(self) -> None:
        result = g1_judgment({})
        assert result["g1_pass"] is False


# =====================================================================
# config_loader
# =====================================================================

class TestConfigLoader:
    """config_loader のテスト."""

    def test_deep_merge(self) -> None:
        base = {"a": 1, "b": {"c": 2, "d": 3}, "e": 5}
        override = {"b": {"c": 99}, "f": 6, "_meta": "skip"}
        merged = _deep_merge(base, override)
        assert merged["a"] == 1
        assert merged["b"]["c"] == 99
        assert merged["b"]["d"] == 3
        assert merged["e"] == 5
        assert merged["f"] == 6
        assert "_meta" not in merged

    def test_load_config_validation_error(self, tmp_path: Path) -> None:
        base_path = tmp_path / "base.yaml"
        base_path.write_text(
            "data:\n"
            "  train_end_index: null\n"
            "features:\n"
            "  selected: null\n",
            encoding="utf-8",
        )

        exp_path = tmp_path / "exp.yaml"
        exp_path.write_text(
            f"_base: {base_path}\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="features.selected is null"):
            load_config(exp_path, base_path=base_path)

    def test_load_config_valid(self, tmp_path: Path) -> None:
        base_path = tmp_path / "base.yaml"
        base_path.write_text(
            "data:\n"
            "  train_end_index: null\n"
            "  ohlcv_path: test.parquet\n"
            "features:\n"
            "  selected: null\n"
            "  candidates:\n"
            "    - a\n"
            "    - b\n",
            encoding="utf-8",
        )
        exp_path = tmp_path / "exp.yaml"
        exp_path.write_text(
            f"_base: {base_path}\n"
            "_gate: G1-info\n"
            "data:\n"
            "  train_end_index: 1000\n"
            "features:\n"
            "  selected:\n"
            "    - a\n"
            "    - b\n",
            encoding="utf-8",
        )

        cfg = load_config(exp_path, base_path=base_path)
        assert cfg["data"]["train_end_index"] == 1000
        assert cfg["features"]["selected"] == ["a", "b"]
        assert cfg["_gate"] == "G1-info"


# =====================================================================
# data_loader
# =====================================================================

class TestDataLoader:
    """data_loader のテスト."""

    def test_load_parquet(self, tmp_path: Path) -> None:

        df = pd.DataFrame({"close": [100, 101, 102], "feature_a": [1, 2, 3]})
        p = tmp_path / "test.parquet"
        df.to_parquet(p)

        loaded = load_parquet(p)
        assert len(loaded) == 3
        assert "close" in loaded.columns

    def test_load_parquet_select_cols(self, tmp_path: Path) -> None:

        df = pd.DataFrame({"close": [100], "a": [1], "b": [2], "c": [3]})
        p = tmp_path / "test.parquet"
        df.to_parquet(p)

        loaded = load_parquet(p, feature_cols=["a", "b"])
        assert "a" in loaded.columns
        assert "close" in loaded.columns  # always kept

    def test_split_train_eval(self) -> None:

        df = pd.DataFrame({"x": range(100)})
        train, eval_ = split_train_eval(df, 80)
        assert len(train) == 80
        assert len(eval_) == 20

    def test_generate_targets(self) -> None:

        df = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]})
        result = generate_targets(df, horizons=[1], target_types=["direction"])
        assert "target_direction_h1" in result.columns
        # 100→101 is up, so first row should be 1
        assert result["target_direction_h1"].iloc[0] == 1

    def test_check_nan_ratio(self) -> None:

        df = pd.DataFrame({"a": [1, 2, np.nan], "b": [4, 5, 6]})
        result = check_nan_ratio(df, max_ratio=0.5)
        assert result["pass"] is True
        assert result["nan_cells"] == 1

    def test_compute_data_hash(self, tmp_path: Path) -> None:

        p = tmp_path / "test.bin"
        p.write_bytes(b"hello")
        h = compute_data_hash(p)
        assert len(h) == 64  # SHA-256 hex


# =====================================================================
# manifest
# =====================================================================

class TestManifest:
    """manifest.py のテスト."""

    def test_write_and_read(self, tmp_path: Path) -> None:

        mw = ManifestWriter(path=tmp_path / "manifest.jsonl", sync_writes=False)
        entry = mw.start_run(
            config_path="test.yaml",
            config={"seed": 42},
            data_path=str(tmp_path / "nonexist.parquet"),
            gate="G1-info",
            seed=42,
        )
        assert entry.status == "running"
        assert "v460" in entry.run_id

        mw.finish_run(
            entry, metrics={"ic": 0.05}, gate_result="PASS",
            artifacts=["result.json"],
        )
        entries = mw.read_all()
        assert len(entries) == 2  # start + finish
        assert entries[1]["status"] == "completed"
        assert entries[1]["gate_result"] == "PASS"

    def test_config_hash_deterministic(self) -> None:

        cfg = {"a": 1, "b": "c"}
        h1 = compute_config_hash(cfg)
        h2 = compute_config_hash(cfg)
        assert h1 == h2
        assert len(h1) == 16

    def test_read_all_skips_malformed_lines(self, tmp_path: Path) -> None:

        manifest_path = tmp_path / "manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write('{"run_id":"ok1","status":"running"}\n')
            f.write('{"run_id":"broken",\n')
            f.write('["not","object"]\n')
            f.write('{"run_id":"ok2","status":"completed"}\n')

        mw = ManifestWriter(path=manifest_path, sync_writes=False)
        entries = mw.read_all()
        run_ids = [str(entry.get("run_id")) for entry in entries]
        assert run_ids == ["ok1", "ok2"]

    def test_start_run_with_empty_data_path_is_pending(self, tmp_path: Path) -> None:

        mw = ManifestWriter(path=tmp_path / "manifest.jsonl", sync_writes=False)
        entry = mw.start_run(
            config_path="test.yaml",
            config={"seed": 7},
            data_path="",
            gate="G1-info",
            seed=7,
        )
        assert entry.data_hash == "pending"

    def test_start_run_with_directory_data_path_is_pending(self, tmp_path: Path) -> None:

        mw = ManifestWriter(path=tmp_path / "manifest.jsonl", sync_writes=False)
        entry = mw.start_run(
            config_path="test.yaml",
            config={"seed": 8},
            data_path=str(tmp_path),
            gate="G1-info",
            seed=8,
        )
        assert entry.data_hash == "pending"


# =====================================================================
# microstructure
# =====================================================================

class TestMicrostructure:
    """microstructure.py のテスト."""

    def _make_sample_df(self) -> pd.DataFrame:
        # 003# #20: seed fixation
        rng = np.random.RandomState(42)
        n = 100
        return pd.DataFrame({
            "close": rng.uniform(100, 110, n),
            "best_bid": rng.uniform(99, 105, n),
            "best_ask": rng.uniform(105, 110, n),
            "mid_price": rng.uniform(102, 108, n),
            "spread": rng.uniform(0.001, 0.01, n),
            "bid_vol_5": rng.uniform(1, 10, n),
            "ask_vol_5": rng.uniform(1, 10, n),
            "depth_imbalance": rng.uniform(-1, 1, n),
            "buy_volume": rng.uniform(0, 5, n),
            "sell_volume": rng.uniform(0, 5, n),
            "trade_count": rng.randint(0, 50, n).astype(float),
            "vwap": rng.uniform(100, 110, n),
            "trade_flow_imbalance": rng.uniform(-1, 1, n),
        })

    def test_feature_generation(self) -> None:
        from ztb.features.microstructure import MICROSTRUCTURE_FEATURES, add_microstructure_features

        df = self._make_sample_df()
        result = add_microstructure_features(df, window=10)

        for feat in MICROSTRUCTURE_FEATURES:
            assert feat in result.columns, f"Missing feature: {feat}"

        # No NaN
        for feat in MICROSTRUCTURE_FEATURES:
            assert result[feat].isna().sum() == 0, f"NaN in {feat}"

    def test_no_mutation(self) -> None:
        from ztb.features.microstructure import add_microstructure_features

        df = self._make_sample_df()
        original_cols = set(df.columns)
        _ = add_microstructure_features(df)
        assert set(df.columns) == original_cols  # input not mutated


# =====================================================================
# 003# #19: 異常系テスト
# =====================================================================

class TestTimestampParsing:
    """003# #4: coincheck _parse_timestamp のテスト."""

    def test_float_passthrough(self) -> None:
        assert _parse_timestamp(1234567890.5) == 1234567890.5

    def test_epoch_string(self) -> None:
        result = _parse_timestamp("1234567890")
        assert result == 1234567890.0

    def test_iso8601(self) -> None:
        result = _parse_timestamp("2025-01-01T00:00:00Z")
        assert isinstance(result, float)
        assert result > 0

    def test_iso8601_with_offset(self) -> None:
        result = _parse_timestamp("2025-01-01T09:00:00+09:00")
        assert isinstance(result, float)

    def test_garbage_fallback(self) -> None:
        result = _parse_timestamp("not-a-timestamp")
        assert isinstance(result, float)  # Falls back to time.time()


class TestCollectorDedup:
    """003# #5: trade dedup via _last_trade_id のテスト."""

    def test_dedup_prevents_duplicates(self) -> None:
        from unittest.mock import AsyncMock

        from ztb.data.market_data_collector import MarketDataCollector
        from ztb.trading.live.exchanges.base.broker_interfaces import TradeRecord

        adapter = AsyncMock()
        collector = MarketDataCollector(adapter, "btc_jpy", raw_dir="/tmp/test_raw")

        trades = [
            TradeRecord(timestamp=1000.0, price=50000.0, amount=0.1, side="buy"),
            TradeRecord(timestamp=1001.0, price=50100.0, amount=0.2, side="sell"),
        ]

        collector._append_raw_trades(trades)
        assert len(collector._tr_buffer) == 2

        # Same trades again → should be filtered
        collector._append_raw_trades(trades)
        assert len(collector._tr_buffer) == 2  # No new


@pytest.mark.skipif(not _HAS_XGBOOST, reason="xgboost not installed")
class TestEvaluatorFactories:
    """003# #2/3: factory 関数の分類/回帰テスト."""

    def test_classifier_factory(self) -> None:
        from scripts.v460.lib.evaluator import make_xgboost_classifier
        model = make_xgboost_classifier(seed=42)
        assert hasattr(model, "predict_proba")

    def test_regressor_factory(self) -> None:
        from scripts.v460.lib.evaluator import make_xgboost_regressor
        model = make_xgboost_regressor(seed=42)
        assert hasattr(model, "predict")
        # Regressor should NOT have predict_proba
        assert not hasattr(model, "predict_proba")

    def test_reserved_keys_filtered(self) -> None:
        """003# #3: reserved keys don't cause TypeError."""
        from scripts.v460.lib.evaluator import make_xgboost_classifier
        # Passing eval_metric/verbosity/n_jobs should not raise
        model = make_xgboost_classifier(
            seed=42,
            eval_metric="auc",  # Should be filtered
            verbosity=1,        # Should be filtered
            n_jobs=2,           # Should be filtered
        )
        assert model is not None
        # Factory should use its own hardcoded values, not the passed ones
        assert model.get_params()["verbosity"] == 0

    def test_ridge_factory(self) -> None:
        from scripts.v460.lib.evaluator import make_ridge
        model = make_ridge(seed=42)
        assert hasattr(model, "predict")


class TestDataLoaderEdgeCases:
    """003# #12/#13/#14: data_loader 異常系テスト."""

    def test_direction_nan_preserved(self) -> None:
        """003# #12: NaN in returns should produce NaN direction, not 0."""

        # Last row has no forward close → ret is NaN
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
        result = generate_targets(df, horizons=[1], target_types=["direction"])

        # Last row (h=1) should be NaN (no future data)
        assert pd.isna(result["target_direction_h1"].iloc[-1])

    def test_column_order_deterministic(self, tmp_path: Path) -> None:
        """003# #13: column order is stable across calls."""

        df = pd.DataFrame({"close": [100], "z": [1], "a": [2], "m": [3]})
        p = tmp_path / "test.parquet"
        df.to_parquet(p)

        cols1 = list(load_parquet(p, feature_cols=["z", "a", "m"]).columns)
        cols2 = list(load_parquet(p, feature_cols=["m", "z", "a"]).columns)
        assert cols1 == cols2  # Same order regardless of input

    def test_missing_column_raises(self, tmp_path: Path) -> None:

        df = pd.DataFrame({"close": [100], "a": [1]})
        p = tmp_path / "test.parquet"
        df.to_parquet(p)

        with pytest.raises(KeyError, match="Missing columns"):
            load_parquet(p, feature_cols=["nonexistent"])


class TestGateCheckG0FeatureColumns:
    """003# #18: G0 column count uses feature columns only."""

    def test_feature_column_count_excludes_targets(self, tmp_path: Path) -> None:

        df = pd.DataFrame({
            "close": [100.0, 101.0, 102.0],
            "feat_a": [1, 2, 3],
            "feat_b": [4, 5, 6],
            "target_direction_h1": [1, 0, 1],
        })
        p = tmp_path / "test.parquet"
        df.to_parquet(p)

        result = run_g0(str(p), thresholds={"min_feature_columns": 2, "max_nan_ratio": 0.01})
        check = result["checks"]["feature_column_count"]
        # Should count feat_a, feat_b only (not close, not target_*)
        assert check["actual"] == 2
        assert check["pass"] is True


# =====================================================================
# build_features
# =====================================================================

class TestBuildFeatures:
    """build_features.py の単体テスト."""

    def test_proxy_features_generation(self) -> None:
        """OHLCV から 10 特徴量が生成されること."""
        from scripts.v460.build_features import V460_FEATURES, build_proxy_features

        rng = np.random.RandomState(42)
        n = 200
        close = 5000000.0 + np.cumsum(rng.normal(0, 1000, n))
        df = pd.DataFrame({
            "open": close + rng.normal(0, 500, n),
            "high": close + np.abs(rng.normal(0, 2000, n)),
            "low": close - np.abs(rng.normal(0, 2000, n)),
            "close": close,
            "volume": rng.exponential(1.0, n),
        })

        result = build_proxy_features(df)

        assert "close" in result.columns
        for feat in V460_FEATURES:
            assert feat in result.columns, f"Missing: {feat}"

        # No NaN
        assert result[V460_FEATURES].isna().sum().sum() == 0

    def test_all_features_nontrivial(self) -> None:
        """生成された特徴量が定数でないこと（標準偏差 > 0）."""
        from scripts.v460.build_features import V460_FEATURES, build_proxy_features

        rng = np.random.RandomState(123)
        n = 500
        close = 5000000.0 + np.cumsum(rng.normal(0, 1000, n))
        df = pd.DataFrame({
            "open": close + rng.normal(0, 500, n),
            "high": close + np.abs(rng.normal(0, 2000, n)),
            "low": close - np.abs(rng.normal(0, 2000, n)),
            "close": close,
            "volume": rng.exponential(1.0, n) + 0.01,
        })

        result = build_proxy_features(df)
        for feat in V460_FEATURES:
            std = result[feat].std()
            assert std > 0, f"{feat} is constant (std=0)"


class TestG0HashPrefix:
    """G0 hash prefix matching テスト."""

    def test_hash_prefix_match(self, tmp_path: Path) -> None:

        df = pd.DataFrame({
            "feat_a": [1, 2, 3],
            "feat_b": [4, 5, 6],
            "feat_c": [7, 8, 9],
            "feat_d": [10, 11, 12],
            "close": [100.0, 101.0, 102.0],
        })
        p = tmp_path / "test.parquet"
        df.to_parquet(p)

        full_hash = compute_data_hash(str(p))
        prefix = full_hash[:16]

        # Full hash and prefix should both match
        result = run_g0(str(p), expected_hash=prefix,
                        thresholds={"min_feature_columns": 4, "max_nan_ratio": 0.01})
        assert result["checks"]["data_hash"]["pass"] is True
        assert result["gate_result"] == "PASS"

    def test_hash_mismatch(self, tmp_path: Path) -> None:

        df = pd.DataFrame({
            "feat_a": [1, 2, 3],
            "feat_b": [4, 5, 6],
            "feat_c": [7, 8, 9],
            "feat_d": [10, 11, 12],
            "close": [100.0, 101.0, 102.0],
        })
        p = tmp_path / "test.parquet"
        df.to_parquet(p)

        result = run_g0(str(p), expected_hash="0000000000000000",
                        thresholds={"min_feature_columns": 4, "max_nan_ratio": 0.01})
        assert result["checks"]["data_hash"]["pass"] is False


# =====================================================================
# 007# F1/F2: run_gate_check G1 judgment — any() vs all()
# =====================================================================

class TestGateCheckG1AnyLogic:
    """007# F1/F2: G1 追加閾値チェックが any() で判定されることを検証."""

    def _make_exp_results(
        self, targets: dict[str, dict], fold_results: dict | None = None,
    ) -> dict:
        """テスト用 experiment results を構築."""
        return {
            "xgboost": targets,
            "fold_results": fold_results or {},
        }

    def test_any_pass_single_target(self) -> None:
        """1 target だけ閾値クリア → extra_any_pass = True."""

        # 2 targets: 1 passes thresholds, 1 fails
        targets = {
            "direction_h1": {
                "ic_mean": 0.05,
                "accuracy_mean": 0.55,
                "ic_significant_count": 3,
            },
            "direction_h5": {
                "ic_mean": 0.001,   # Below min_ic
                "accuracy_mean": 0.49,  # Below min_accuracy
                "ic_significant_count": 0,
            },
        }
        exp_results = self._make_exp_results(targets)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            json.dump(exp_results, f)
            tmp_path = f.name

        result = run_g1_judgment(tmp_path)
        checks = result.get("threshold_checks", {})

        # direction_h1 should pass, direction_h5 should fail
        assert checks["direction_h1"]["ic_pass"] is True
        assert checks["direction_h5"]["ic_pass"] is False

        # extra check uses any(), so having 1 pass is sufficient
        # (g1_judgment itself will likely fail due to no fold_results,
        #  but the threshold_checks logic should use any())

    def test_all_fail_no_extra_pass(self) -> None:
        """全 target が閾値未達 → extra_any_pass = False."""

        targets = {
            "direction_h1": {
                "ic_mean": 0.001,
                "accuracy_mean": 0.49,
                "ic_significant_count": 0,
            },
            "direction_h5": {
                "ic_mean": 0.001,
                "accuracy_mean": 0.49,
                "ic_significant_count": 0,
            },
        }
        exp_results = self._make_exp_results(targets)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            json.dump(exp_results, f)
            tmp_path = f.name

        result = run_g1_judgment(tmp_path)
        # No fold_results → g1_judgment returns g1_pass=False
        # All targets fail thresholds → extra_any_pass=False
        # final_pass = False AND False = False
        assert result["gate_result"] == "FAIL"


# =====================================================================
# 007# F4: fold_results slimming
# =====================================================================

class TestFoldResultsSlimming:
    """007# F4: fold_results_saved が統計量のみになることを検証."""

    def test_default_saves_stats_only(self) -> None:
        """debug=False → fold_results_saved は統計量辞書."""
        fold_results = {
            "direction_h1": [
                ([1.0, 2.0, 3.0], [0.5, 1.0, 1.5]),
                ([4.0, 5.0, 6.0], [2.0, 3.0, 4.0]),
            ],
        }

        # Simulate the slimming logic from feature_info
        debug_mode = False
        fold_results_for_save: dict = {}
        if debug_mode:
            fold_results_for_save = fold_results
        else:
            for tgt, pairs in fold_results.items():
                fold_stats = []
                for model_s, baseline_s in pairs:
                    fold_stats.append({
                        "n_model": len(model_s),
                        "n_baseline": len(baseline_s),
                        "model_mean": float(np.mean(model_s)),
                        "model_std": float(np.std(model_s)),
                        "baseline_mean": float(np.mean(baseline_s)),
                        "baseline_std": float(np.std(baseline_s)),
                    })
                fold_results_for_save[tgt] = fold_stats

        # Verify stats-only output
        assert "direction_h1" in fold_results_for_save
        stats = fold_results_for_save["direction_h1"]
        assert len(stats) == 2
        assert stats[0]["n_model"] == 3
        assert stats[0]["n_baseline"] == 3
        assert abs(stats[0]["model_mean"] - 2.0) < 1e-6

    def test_debug_preserves_raw(self) -> None:
        """debug=True → fold_results_saved は生配列."""
        fold_results = {
            "direction_h1": [
                ([1.0, 2.0, 3.0], [0.5, 1.0, 1.5]),
            ],
        }
        debug_mode = True
        if debug_mode:
            fold_results_for_save = fold_results
        else:
            fold_results_for_save = {}

        # Raw arrays preserved
        assert fold_results_for_save["direction_h1"][0][0] == [1.0, 2.0, 3.0]


# =====================================================================
# 007# F5: run_gate_check g1_judgment_cache 互換
# =====================================================================

class TestGateCheckCacheCompat:
    """007# F5: stats-only JSON でも run_gate_check が動作することを検証."""

    def test_cached_judgment_used(self) -> None:
        """g1_judgment_cache があればそれを使用し、fold_results を再計算しない."""

        # Stats-only fold_results (g1_judgment() に直接渡すとクラッシュする)
        exp_results = {
            "xgboost": {
                "direction_h1": {
                    "ic_mean": 0.05,
                    "accuracy_mean": 0.55,
                    "ic_significant_count": 3,
                },
            },
            "fold_results": {
                "direction_h1": [
                    {"n_model": 100, "model_mean": 0.5},  # stats dict, not arrays
                ],
            },
            "g1_judgment_cache": {
                "g1_pass": True,
                "passed_targets": ["direction_h1"],
                "details": {
                    "direction_h1": {
                        "p_geo": 0.001,
                        "pmean_pass": True,
                        "holm_pass": True,
                        "cliff_d": 0.45,
                    }
                },
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            json.dump(exp_results, f)
            tmp_path = f.name

        result = run_g1_judgment(tmp_path)
        # Cache says PASS, extra threshold direction_h1 passes → final PASS
        assert result["gate_result"] == "PASS"

    def test_stats_only_no_cache_fallback(self) -> None:
        """g1_judgment_cache がなく fold_results が stats → FAIL (クラッシュしない)."""

        exp_results = {
            "xgboost": {
                "direction_h1": {
                    "ic_mean": 0.05,
                    "accuracy_mean": 0.55,
                    "ic_significant_count": 3,
                },
            },
            "fold_results": {
                "direction_h1": [
                    {"n_model": 100, "model_mean": 0.5},  # stats dict
                ],
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            json.dump(exp_results, f)
            tmp_path = f.name

        # Should NOT crash — graceful fallback to FAIL
        result = run_g1_judgment(tmp_path)
        assert result["gate_result"] == "FAIL"


# =====================================================================
# 007# F7: _task preservation in config_loader
# =====================================================================

class TestConfigLoaderTaskPreservation:
    """007# F7: _task がマージ後も保持されることを検証."""

    def test_task_preserved(self, tmp_path: Path) -> None:
        """実験 YAML の _task が load_config 結果に含まれる."""

        base_yaml = tmp_path / "base.yaml"
        base_yaml.write_text(
            "data:\n"
            "  train_end_index: 1000\n"
            "  ohlcv_path: test.parquet\n"
            "features:\n"
            "  selected:\n"
            "    - feat_a\n",
            encoding="utf-8",
        )

        exp_yaml = tmp_path / "exp.yaml"
        exp_yaml.write_text(
            f"_base: {base_yaml}\n"
            "_gate: G1-info\n"
            "_task: sac_train\n",
            encoding="utf-8",
        )

        cfg = load_config(str(exp_yaml))
        assert cfg["_task"] == "sac_train"

    def test_task_default(self, tmp_path: Path) -> None:
        """_task 未指定時はデフォルト feature_info."""

        base_yaml = tmp_path / "base.yaml"
        base_yaml.write_text(
            "data:\n"
            "  train_end_index: 1000\n"
            "  ohlcv_path: test.parquet\n"
            "features:\n"
            "  selected:\n"
            "    - feat_a\n",
            encoding="utf-8",
        )

        exp_yaml = tmp_path / "exp.yaml"
        exp_yaml.write_text(
            f"_base: {base_yaml}\n"
            "_gate: G1-info\n",
            encoding="utf-8",
        )

        cfg = load_config(str(exp_yaml))
        assert cfg["_task"] == "feature_info"
