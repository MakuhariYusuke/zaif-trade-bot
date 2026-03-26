"""NormLoader ユニットテスト (621#)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from ztb.features.norm_loader import NormLoader


@pytest.fixture()
def norm_json(tmp_path: Path) -> Path:
    """テスト用 norm.json を生成."""
    data = {
        "feature_stats": {
            "price_velocity": {"mean": 0.0, "std": 1.0, "min": -3.0, "max": 3.0},
            "micro_trend": {"mean": 0.5, "std": 2.0, "min": -5.0, "max": 5.0},
            "volume_surge": {"mean": 1.0, "std": 0.5, "min": 0.0, "max": 10.0},
        },
        "metadata": {
            "generated_at": "2026-03-25T00:00:00Z",
            "n_features": 3,
            "n_rows": 1000,
        },
    }
    p = tmp_path / "norm.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


class TestNormLoaderInit:
    def test_load_success(self, norm_json: Path) -> None:
        nl = NormLoader(norm_json)
        assert nl.is_loaded
        assert nl.feature_names == ["price_velocity", "micro_trend", "volume_surge"]

    def test_load_missing_file(self, tmp_path: Path) -> None:
        nl = NormLoader(tmp_path / "nonexistent.json")
        assert not nl.is_loaded

    def test_load_empty_stats(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.json"
        p.write_text(json.dumps({"feature_stats": {}}), encoding="utf-8")
        nl = NormLoader(p)
        assert not nl.is_loaded


class TestNormalize:
    def test_basic_zscore(self, norm_json: Path) -> None:
        nl = NormLoader(norm_json)
        result = nl.normalize({"price_velocity": 1.0, "micro_trend": 2.5, "volume_surge": 2.0})
        # price_velocity: (1.0 - 0.0) / (1.0 + 1e-10) = 1.0
        # micro_trend: (2.5 - 0.5) / (2.0 + 1e-10) = 1.0
        # volume_surge: (2.0 - 1.0) / (0.5 + 1e-10) = 2.0
        np.testing.assert_allclose(result, [1.0, 1.0, 2.0], atol=1e-6)

    def test_nan_imputation(self, norm_json: Path) -> None:
        nl = NormLoader(norm_json)
        result = nl.normalize({"price_velocity": float("nan"), "micro_trend": 0.5})
        # price_velocity: NaN → mean=0.0, z-score = 0.0
        # micro_trend: (0.5 - 0.5) / (2.0 + eps) = 0.0
        # volume_surge: missing → mean=1.0, z-score = 0.0
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-6)

    def test_missing_feature_imputation(self, norm_json: Path) -> None:
        nl = NormLoader(norm_json)
        result = nl.normalize({"price_velocity": 1.0})
        # micro_trend: missing → mean=0.5, z=(0.5-0.5)/2.0 = 0.0
        # volume_surge: missing → mean=1.0, z=(1.0-1.0)/0.5 = 0.0
        assert result[0] == pytest.approx(1.0, abs=1e-6)
        assert result[1] == pytest.approx(0.0, abs=1e-6)
        assert result[2] == pytest.approx(0.0, abs=1e-6)

    def test_clipping(self, norm_json: Path) -> None:
        nl = NormLoader(norm_json, clip=5.0)
        result = nl.normalize({"price_velocity": 100.0})
        assert result[0] == pytest.approx(5.0)

    def test_negative_clipping(self, norm_json: Path) -> None:
        nl = NormLoader(norm_json, clip=5.0)
        result = nl.normalize({"price_velocity": -100.0})
        assert result[0] == pytest.approx(-5.0)

    def test_output_dtype(self, norm_json: Path) -> None:
        nl = NormLoader(norm_json)
        result = nl.normalize({"price_velocity": 1.0})
        assert result.dtype == np.float64

    def test_empty_features(self, tmp_path: Path) -> None:
        nl = NormLoader(tmp_path / "nonexistent.json")
        result = nl.normalize({"x": 1.0})
        assert len(result) == 0


class TestReload:
    def test_reload_detects_change(self, norm_json: Path) -> None:
        nl = NormLoader(norm_json)
        assert nl.is_loaded
        # 内容を書き換え
        data = json.loads(norm_json.read_text(encoding="utf-8"))
        data["feature_stats"]["new_feat"] = {"mean": 0.0, "std": 1.0}
        norm_json.write_text(json.dumps(data), encoding="utf-8")
        stat = norm_json.stat()
        os.utime(norm_json, (stat.st_atime, stat.st_mtime + 1.0))
        changed = nl.reload_if_changed()
        assert changed
        assert "new_feat" in nl.feature_names

    def test_reload_no_change(self, norm_json: Path) -> None:
        nl = NormLoader(norm_json)
        changed = nl.reload_if_changed()
        assert not changed
