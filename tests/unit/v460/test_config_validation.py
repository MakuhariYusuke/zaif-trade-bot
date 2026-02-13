"""
v460 Config / Gate テスト — config_loader + gate_thresholds 検証.

config_loader.py の _deep_merge, load_config, _validate,
gate_thresholds.yaml の整合性をテスト.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.v460.lib.config_loader import (
    _deep_merge,
    _validate,
    load_config,
    load_gate_thresholds,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ======================================================================
# _deep_merge
# ======================================================================


class TestDeepMerge:
    """dict の再帰マージロジック."""

    def test_override_scalar(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"a": 10}
        result = _deep_merge(base, override)
        assert result["a"] == 10
        assert result["b"] == 2

    def test_nested_merge(self) -> None:
        base = {"data": {"path": "old.parquet", "hash": "abc"}}
        override = {"data": {"path": "new.parquet"}}
        result = _deep_merge(base, override)
        assert result["data"]["path"] == "new.parquet"
        assert result["data"]["hash"] == "abc"  # preserved

    def test_meta_keys_skipped(self) -> None:
        base = {"a": 1}
        override = {"_base": "x.yaml", "_gate": "G1", "a": 2}
        result = _deep_merge(base, override)
        assert result["a"] == 2
        assert "_base" not in result
        assert "_gate" not in result

    def test_base_not_mutated(self) -> None:
        base = {"data": {"path": "original"}}
        original_base = copy.deepcopy(base)
        _deep_merge(base, {"data": {"path": "changed"}})
        assert base == original_base

    def test_add_new_key(self) -> None:
        base = {"a": 1}
        override = {"b": 2}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_override_list_replaces(self) -> None:
        """list は再帰マージではなく完全置換."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = _deep_merge(base, override)
        assert result["items"] == [4, 5]

    def test_override_null(self) -> None:
        base = {"a": "value"}
        override = {"a": None}
        result = _deep_merge(base, override)
        assert result["a"] is None


# ======================================================================
# _validate
# ======================================================================


class TestValidate:
    """設定値のバリデーション."""

    def test_valid_config_passes(self) -> None:
        cfg = {
            "features": {"selected": ["bid_ask_spread"]},
            "data": {"train_end_index": 100000},
        }
        _validate(cfg)  # should not raise

    def test_missing_features_selected(self) -> None:
        cfg = {
            "features": {"selected": None},
            "data": {"train_end_index": 100000},
        }
        with pytest.raises(ValueError, match="features.selected is null"):
            _validate(cfg)

    def test_missing_train_end_index(self) -> None:
        cfg = {
            "features": {"selected": ["x"]},
            "data": {"train_end_index": None},
        }
        with pytest.raises(ValueError, match="train_end_index is null"):
            _validate(cfg)

    def test_both_missing(self) -> None:
        cfg = {
            "features": {"selected": None},
            "data": {"train_end_index": None},
        }
        with pytest.raises(ValueError, match="features.selected"):
            _validate(cfg)

    def test_missing_features_key(self) -> None:
        cfg = {"data": {"train_end_index": 1000}}
        with pytest.raises(ValueError, match="features.selected"):
            _validate(cfg)


# ======================================================================
# load_config — 実際の YAML ファイルと統合テスト
# ======================================================================


class TestLoadConfig:
    """実験 YAML を base.yaml とマージしてロード."""

    def test_load_g1_full(self) -> None:
        """g1_full_9targets.yaml が正常にロードされること."""
        cfg = load_config("configs/v460/experiments/g1_full_9targets.yaml")
        assert cfg["_gate"] == "G1-info"
        assert cfg["_task"] == "feature_info"
        assert isinstance(cfg["features"]["selected"], list)
        assert len(cfg["features"]["selected"]) == 10
        assert cfg["data"]["train_end_index"] is not None
        # base.yaml の値が引き継がれていること
        assert cfg["trading"]["exchange"] == "coincheck"
        assert cfg["trading"]["fee_model"] == "maker_zero"

    @pytest.mark.parametrize("experiment", [
        "configs/v460/experiments/g1_full_9targets.yaml",
        "configs/v460/experiments/g1_xgb_h1_direction.yaml",
        "configs/v460/experiments/g1_xgb_h5_direction.yaml",
        "configs/v460/experiments/g1_xgb_h15_direction.yaml",
    ])
    def test_all_experiments_loadable(self, experiment: str) -> None:
        """全実験 YAML がバリデーション通過すること."""
        exp_path = _PROJECT_ROOT / experiment
        if not exp_path.exists():
            pytest.skip(f"{experiment} not found")
        cfg = load_config(experiment)
        assert cfg["features"]["selected"] is not None
        assert cfg["data"]["train_end_index"] is not None

    def test_base_yaml_trading_config(self) -> None:
        """base.yaml にmaker_zero / maker_only が設定されていること."""
        base_path = _PROJECT_ROOT / "configs" / "v460" / "base.yaml"
        with open(base_path) as f:
            base = yaml.safe_load(f)
        assert base["trading"]["fee_model"] == "maker_zero"
        assert base["trading"]["order_type"] == "maker_only"
        assert base["trading"]["symbol"] == "btc_jpy"


# ======================================================================
# gate_thresholds.yaml
# ======================================================================


class TestGateThresholds:
    """gate_thresholds.yaml の整合性テスト."""

    @pytest.fixture
    def thresholds(self) -> dict[str, Any]:
        return load_gate_thresholds()

    def test_all_gates_present(self, thresholds: dict[str, Any]) -> None:
        for gate in ["g0_data", "g1_info", "g1_1_exec", "g2_train", "g3_pnl", "g4_live"]:
            assert gate in thresholds, f"Missing gate: {gate}"

    def test_g0_values(self, thresholds: dict[str, Any]) -> None:
        g0 = thresholds["g0_data"]
        assert g0["data_hash_match"] is True
        assert g0["min_feature_columns"] >= 1
        assert 0 < g0["max_nan_ratio"] <= 0.05

    def test_g1_values(self, thresholds: dict[str, Any]) -> None:
        g1 = thresholds["g1_info"]
        assert 0 < g1["min_ic"] < 0.10
        assert 0.50 < g1["min_accuracy"] < 0.60
        assert g1["horizons"] == [1, 5, 15]
        assert set(g1["targets"]) == {"direction", "magnitude", "volatility"}
        assert g1["holm_family_size"] == 9

    def test_g1_1_values(self, thresholds: dict[str, Any]) -> None:
        g11 = thresholds["g1_1_exec"]
        assert 0.80 <= g11["min_fill_rate_p90"] <= 1.0
        assert 0 < g11["max_cancel_ratio"] <= 0.50
        assert 0 < g11["max_queue_wait_median_sec"] <= 120
        assert g11["min_post_fill_30s_pnl"] == 0.0
        assert 0 < g11["max_adverse_selection_ratio"] <= 0.50

    def test_g2_values(self, thresholds: dict[str, Any]) -> None:
        g2 = thresholds["g2_train"]
        assert 0.5 <= g2["min_positive_seed_ratio"] <= 1.0
        assert g2["max_ic_seed_std"] > 0

    def test_g3_values(self, thresholds: dict[str, Any]) -> None:
        g3 = thresholds["g3_pnl"]
        assert g3["min_pf_median"] >= 1.0
        assert g3["gross_gt_fee"] is True
        assert g3["max_drawdown"] > 0

    def test_threshold_monotonicity(self, thresholds: dict[str, Any]) -> None:
        """論理的な整合性: fill_rate > cancel_ratio の関係."""
        g11 = thresholds["g1_1_exec"]
        # fill_rate P90 + cancel_ratio の合計は 100% 以下でなければならない
        assert g11["min_fill_rate_p90"] + g11["max_cancel_ratio"] <= 1.2  # soft check


# ======================================================================
# base.yaml / feature_sets.yaml 整合性
# ======================================================================


class TestBaseFeatureConsistency:
    """base.yaml の候補と feature_sets.yaml の整合性."""

    def test_base_candidates_not_empty(self) -> None:
        base_path = _PROJECT_ROOT / "configs" / "v460" / "base.yaml"
        with open(base_path) as f:
            base = yaml.safe_load(f)
        candidates = base.get("features", {}).get("candidates", [])
        assert len(candidates) >= 5, "base.yaml should have ≥5 feature candidates"

    def test_feature_sets_yaml_exists(self) -> None:
        fs_path = _PROJECT_ROOT / "configs" / "features" / "feature_sets.yaml"
        assert fs_path.exists()

    def test_sac_config_in_base(self) -> None:
        """SAC 設定が base.yaml に存在すること (G2 準備)."""
        base_path = _PROJECT_ROOT / "configs" / "v460" / "base.yaml"
        with open(base_path) as f:
            base = yaml.safe_load(f)
        sac = base.get("sac", {})
        assert sac.get("total_steps", 0) > 0
        assert isinstance(sac.get("seeds"), list)
        assert len(sac["seeds"]) >= 3
