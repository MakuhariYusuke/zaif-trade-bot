"""
v460 Gate Check Runner 単体テスト.

run_gate_check.py の run_g0 / run_g1_judgment / run_g1_1 / run_g2_judgment
/ run_g3_judgment / run_g4_judgment を検証する。
外部ファイル依存はすべて mock で遮断。
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# テスト対象を遅延 import (sys.path 挿入済み前提)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import sys

sys.path.insert(0, str(_PROJECT_ROOT))


# =====================================================================
# Helpers
# =====================================================================


def _make_feature_df(n: int = 100, n_features: int = 10) -> pd.DataFrame:
    """テスト用の feature DataFrame を生成."""
    rng = np.random.RandomState(42)
    data = {"close": rng.uniform(14_000_000, 16_000_000, n)}
    for i in range(n_features):
        data[f"feat_{i}"] = rng.randn(n)
    return pd.DataFrame(data)


def _default_thresholds_g0() -> dict:
    return {"min_feature_columns": 4, "max_nan_ratio": 0.01}


def _default_thresholds_g1() -> dict:
    return {
        "min_ic": 0.02,
        "min_accuracy": 0.51,
        "min_significant_folds": 2,
        "p_alpha": 0.05,
        "min_cliff_d": 0.33,
    }


def _default_thresholds_g1_1() -> dict:
    return {
        "min_fill_rate_p90": 0.90,
        "max_cancel_ratio": 0.30,
        "max_queue_wait_median_sec": 60,
        "min_post_fill_30s_pnl": 0.0,
        "max_adverse_selection_ratio": 0.20,
    }


def _make_fill_records(
    n: int = 50,
    fill_rate: float = 0.95,
    pnl_mean: float = 0.1,
    as_ratio: float = 0.10,
    queue_wait: float = 15.0,
) -> list[dict]:
    """fill_records JSONL 用の dict リストを生成."""
    rng = np.random.RandomState(42)
    base_ts = 1700000000.0
    records = []
    for i in range(n):
        filled = rng.random() < fill_rate
        adverse = rng.random() < as_ratio if filled else None
        pnl = rng.normal(pnl_mean, 0.5) if filled else None
        records.append({
            "cycle_id": f"cycle_{i:04d}",
            "timestamp": base_ts + i * 120.0,  # 2 min intervals
            "side": "buy" if i % 2 == 0 else "sell",
            "order_price": 15_000_000.0 + rng.randint(-1000, 1000),
            "order_quantity": 0.001,
            "fill_price": 15_000_001.0 if filled else None,
            "filled": filled,
            "cancelled": not filled,
            "queue_wait_sec": rng.uniform(5, queue_wait * 2) if filled else rng.uniform(50, 120),
            "mid_at_fill": 15_000_050.0 if filled else None,
            "mid_30s_after": 15_000_100.0 if filled else None,
            "post_fill_30s_pnl": pnl,
            "adverse_selected": adverse,
        })
    return records


# =====================================================================
# G0-data テスト
# =====================================================================


class TestRunG0:
    """run_g0 のテスト."""

    @patch("scripts.v460.run_gate_check.ManifestWriter")
    @patch("scripts.v460.run_gate_check.check_nan_ratio")
    @patch("scripts.v460.run_gate_check.load_parquet")
    @patch("scripts.v460.run_gate_check.compute_data_hash")
    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g0_pass_all(
        self, mock_thresh, mock_hash, mock_load, mock_nan, mock_manifest,
    ) -> None:
        """全チェック PASS のケース."""
        from scripts.v460.run_gate_check import run_g0

        mock_thresh.return_value = {"g0_data": _default_thresholds_g0()}
        mock_hash.return_value = "abcdef1234567890abcdef1234567890"
        mock_load.return_value = _make_feature_df(100, 10)
        mock_nan.return_value = {"overall_nan_ratio": 0.001, "pass": True}
        mw_inst = MagicMock()
        mw_inst.path = MagicMock()
        mw_inst.path.exists.return_value = True
        mw_inst.path.__str__ = lambda self: "results/v460/manifest.jsonl"
        mock_manifest.return_value = mw_inst

        result = run_g0("dummy.parquet", "abcdef1234567890")
        assert result["gate"] == "G0-data"
        assert result["gate_result"] == "PASS"
        assert result["checks"]["data_hash"]["pass"] is True

    @patch("scripts.v460.run_gate_check.ManifestWriter")
    @patch("scripts.v460.run_gate_check.check_nan_ratio")
    @patch("scripts.v460.run_gate_check.load_parquet")
    @patch("scripts.v460.run_gate_check.compute_data_hash")
    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g0_hash_mismatch(
        self, mock_thresh, mock_hash, mock_load, mock_nan, mock_manifest,
    ) -> None:
        """ハッシュ不一致 → FAIL."""
        from scripts.v460.run_gate_check import run_g0

        mock_thresh.return_value = {"g0_data": _default_thresholds_g0()}
        mock_hash.return_value = "aaaa1111bbbb2222"
        mock_load.return_value = _make_feature_df(100, 10)
        mock_nan.return_value = {"overall_nan_ratio": 0.0, "pass": True}
        mw_inst = MagicMock()
        mw_inst.path = MagicMock()
        mw_inst.path.exists.return_value = True
        mw_inst.path.__str__ = lambda self: "results/v460/manifest.jsonl"
        mock_manifest.return_value = mw_inst

        result = run_g0("dummy.parquet", "zzzz9999yyyy8888")
        assert result["gate_result"] == "FAIL"
        assert result["checks"]["data_hash"]["pass"] is False

    @patch("scripts.v460.run_gate_check.ManifestWriter")
    @patch("scripts.v460.run_gate_check.check_nan_ratio")
    @patch("scripts.v460.run_gate_check.load_parquet")
    @patch("scripts.v460.run_gate_check.compute_data_hash")
    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g0_no_expected_hash_passes(
        self, mock_thresh, mock_hash, mock_load, mock_nan, mock_manifest,
    ) -> None:
        """expected_hash=None → hash チェック skip (PASS)."""
        from scripts.v460.run_gate_check import run_g0

        mock_thresh.return_value = {"g0_data": _default_thresholds_g0()}
        mock_hash.return_value = "abcdef1234567890"
        mock_load.return_value = _make_feature_df(100, 10)
        mock_nan.return_value = {"overall_nan_ratio": 0.0, "pass": True}
        mw_inst = MagicMock()
        mw_inst.path = MagicMock()
        mw_inst.path.exists.return_value = True
        mw_inst.path.__str__ = lambda self: "results/v460/manifest.jsonl"
        mock_manifest.return_value = mw_inst

        result = run_g0("dummy.parquet", None)
        assert result["checks"]["data_hash"]["pass"] is True

    @patch("scripts.v460.run_gate_check.ManifestWriter")
    @patch("scripts.v460.run_gate_check.check_nan_ratio")
    @patch("scripts.v460.run_gate_check.load_parquet")
    @patch("scripts.v460.run_gate_check.compute_data_hash")
    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g0_too_few_columns(
        self, mock_thresh, mock_hash, mock_load, mock_nan, mock_manifest,
    ) -> None:
        """特徴量カラム < 4 → FAIL."""
        from scripts.v460.run_gate_check import run_g0

        mock_thresh.return_value = {"g0_data": _default_thresholds_g0()}
        mock_hash.return_value = "abcdef1234567890"
        mock_load.return_value = _make_feature_df(100, 2)  # close + 2 feats = 2 feature cols
        mock_nan.return_value = {"overall_nan_ratio": 0.0, "pass": True}
        mw_inst = MagicMock()
        mw_inst.path = MagicMock()
        mw_inst.path.exists.return_value = True
        mw_inst.path.__str__ = lambda self: "results/v460/manifest.jsonl"
        mock_manifest.return_value = mw_inst

        result = run_g0("dummy.parquet")
        assert result["checks"]["feature_column_count"]["pass"] is False
        assert result["gate_result"] == "FAIL"

    @patch("scripts.v460.run_gate_check.ManifestWriter")
    @patch("scripts.v460.run_gate_check.check_nan_ratio")
    @patch("scripts.v460.run_gate_check.load_parquet")
    @patch("scripts.v460.run_gate_check.compute_data_hash")
    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g0_high_nan_ratio(
        self, mock_thresh, mock_hash, mock_load, mock_nan, mock_manifest,
    ) -> None:
        """NaN 比率超過 → FAIL."""
        from scripts.v460.run_gate_check import run_g0

        mock_thresh.return_value = {"g0_data": _default_thresholds_g0()}
        mock_hash.return_value = "abcdef1234567890"
        mock_load.return_value = _make_feature_df(100, 10)
        mock_nan.return_value = {"overall_nan_ratio": 0.05, "pass": False}
        mw_inst = MagicMock()
        mw_inst.path = MagicMock()
        mw_inst.path.exists.return_value = True
        mw_inst.path.__str__ = lambda self: "results/v460/manifest.jsonl"
        mock_manifest.return_value = mw_inst

        result = run_g0("dummy.parquet")
        assert result["checks"]["nan_ratio"]["pass"] is False
        assert result["gate_result"] == "FAIL"

    @patch("scripts.v460.run_gate_check.ManifestWriter")
    @patch("scripts.v460.run_gate_check.check_nan_ratio")
    @patch("scripts.v460.run_gate_check.load_parquet")
    @patch("scripts.v460.run_gate_check.compute_data_hash")
    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g0_no_manifest(
        self, mock_thresh, mock_hash, mock_load, mock_nan, mock_manifest,
    ) -> None:
        """manifest 不在 → FAIL."""
        from scripts.v460.run_gate_check import run_g0

        mock_thresh.return_value = {"g0_data": _default_thresholds_g0()}
        mock_hash.return_value = "abcdef1234567890"
        mock_load.return_value = _make_feature_df(100, 10)
        mock_nan.return_value = {"overall_nan_ratio": 0.0, "pass": True}
        mw_inst = MagicMock()
        mw_inst.path = MagicMock()
        mw_inst.path.exists.return_value = False
        mw_inst.path.__str__ = lambda self: "results/v460/manifest.jsonl"
        mock_manifest.return_value = mw_inst

        result = run_g0("dummy.parquet")
        assert result["checks"]["manifest_exists"]["pass"] is False
        assert result["gate_result"] == "FAIL"


# =====================================================================
# G1-info テスト
# =====================================================================


class TestRunG1Judgment:
    """run_g1_judgment のテスト."""

    def _make_g1_results(
        self,
        ic_mean: float = 0.05,
        acc_mean: float = 0.55,
        sig_count: int = 3,
        g1_pass: bool = True,
        use_cache: bool = True,
    ) -> dict:
        """G1 experiment results JSON を構築."""
        result: dict = {
            "xgboost": {
                "target_5m": {
                    "ic_mean": ic_mean,
                    "accuracy_mean": acc_mean,
                    "ic_significant_count": sig_count,
                }
            },
        }
        if use_cache:
            result["g1_judgment_cache"] = {
                "g1_pass": g1_pass,
                "passed_targets": ["target_5m"] if g1_pass else [],
                "details": {},
            }
        return result

    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g1_pass(self, mock_thresh) -> None:
        """G1 全パス."""
        from scripts.v460.run_gate_check import run_g1_judgment

        mock_thresh.return_value = {"g1_info": _default_thresholds_g1()}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(self._make_g1_results(), f)
            f.flush()
            result = run_g1_judgment(f.name)

        assert result["gate"] == "G1-info"
        assert result["gate_result"] == "PASS"

    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g1_judgment_cache_fail(self, mock_thresh) -> None:
        """g1_judgment_cache が FAIL → 全体 FAIL."""
        from scripts.v460.run_gate_check import run_g1_judgment

        mock_thresh.return_value = {"g1_info": _default_thresholds_g1()}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(self._make_g1_results(g1_pass=False), f)
            f.flush()
            result = run_g1_judgment(f.name)

        assert result["gate_result"] == "FAIL"

    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g1_low_ic(self, mock_thresh) -> None:
        """IC 不足 → threshold_checks FAIL → 全体 FAIL."""
        from scripts.v460.run_gate_check import run_g1_judgment

        mock_thresh.return_value = {"g1_info": _default_thresholds_g1()}
        data = self._make_g1_results(ic_mean=0.005)  # below 0.02
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            f.flush()
            result = run_g1_judgment(f.name)

        tc = result["threshold_checks"]["target_5m"]
        assert tc["ic_pass"] is False
        assert result["gate_result"] == "FAIL"

    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g1_low_accuracy(self, mock_thresh) -> None:
        """accuracy 不足 → FAIL."""
        from scripts.v460.run_gate_check import run_g1_judgment

        mock_thresh.return_value = {"g1_info": _default_thresholds_g1()}
        data = self._make_g1_results(acc_mean=0.48)  # below 0.51
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            f.flush()
            result = run_g1_judgment(f.name)

        tc = result["threshold_checks"]["target_5m"]
        assert tc["accuracy_pass"] is False
        assert result["gate_result"] == "FAIL"

    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g1_insufficient_sig_folds(self, mock_thresh) -> None:
        """sig_folds 不足 → FAIL."""
        from scripts.v460.run_gate_check import run_g1_judgment

        mock_thresh.return_value = {"g1_info": _default_thresholds_g1()}
        data = self._make_g1_results(sig_count=1)  # below 2
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            f.flush()
            result = run_g1_judgment(f.name)

        tc = result["threshold_checks"]["target_5m"]
        assert tc["sig_folds_pass"] is False
        assert result["gate_result"] == "FAIL"

    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g1_no_cache_no_raw_folds(self, mock_thresh) -> None:
        """cache なし + raw fold_results なし → FAIL (stats-only)."""
        from scripts.v460.run_gate_check import run_g1_judgment

        mock_thresh.return_value = {"g1_info": _default_thresholds_g1()}
        data = {
            "xgboost": {
                "target_5m": {
                    "ic_mean": 0.05,
                    "accuracy_mean": 0.55,
                    "ic_significant_count": 3,
                }
            },
            "fold_results": {"target_5m": {"ic_mean": 0.05}},  # stats dict, not raw
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            f.flush()
            result = run_g1_judgment(f.name)

        # judgment g1_pass = False (stats-only fallback)
        assert result["gate_result"] == "FAIL"

    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g1_no_xgboost_no_extra_checks(self, mock_thresh) -> None:
        """xgboost ブロック空 → extra_any_pass=False → FAIL."""
        from scripts.v460.run_gate_check import run_g1_judgment

        mock_thresh.return_value = {"g1_info": _default_thresholds_g1()}
        data = {
            "xgboost": {},
            "g1_judgment_cache": {"g1_pass": True, "passed_targets": [], "details": {}},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            f.flush()
            result = run_g1_judgment(f.name)

        assert result["gate_result"] == "FAIL"


# =====================================================================
# G1.1-exec テスト
# =====================================================================


class TestRunG1_1:
    """run_g1_1 のテスト."""

    def _write_fill_records(self, records: list[dict], tmp_dir: Path) -> None:
        """JSONL ファイルに write."""
        path = tmp_dir / "fill_records_test.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def test_g1_1_pass(self) -> None:
        """良好な fill_records → PASS."""
        from scripts.v460.run_gate_check import run_g1_1

        records = _make_fill_records(
            n=50, fill_rate=0.98, pnl_mean=0.2, as_ratio=0.05, queue_wait=20.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fill_records(records, Path(tmp))
            result = run_g1_1(tmp, thresholds=_default_thresholds_g1_1())

        assert result["gate"] == "G1.1-exec"
        assert result["gate_result"] == "PASS"

    def test_g1_1_low_fill_rate(self) -> None:
        """低 fill_rate → FAIL."""
        from scripts.v460.run_gate_check import run_g1_1

        records = _make_fill_records(n=50, fill_rate=0.50)
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fill_records(records, Path(tmp))
            result = run_g1_1(tmp, thresholds=_default_thresholds_g1_1())

        assert result["gate_result"] == "FAIL"
        assert result["checks"]["E1_fill_rate_p90"]["pass"] is False

    def test_g1_1_high_adverse_selection(self) -> None:
        """高 AS ratio → FAIL."""
        from scripts.v460.run_gate_check import run_g1_1

        records = _make_fill_records(n=50, fill_rate=0.98, as_ratio=0.80)
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fill_records(records, Path(tmp))
            result = run_g1_1(tmp, thresholds=_default_thresholds_g1_1())

        assert result["checks"]["E5_adverse_selection"]["pass"] is False

    def test_g1_1_no_data(self) -> None:
        """データなし → NO_DATA."""
        from scripts.v460.run_gate_check import run_g1_1

        with tempfile.TemporaryDirectory() as tmp:
            result = run_g1_1(tmp, thresholds=_default_thresholds_g1_1())

        assert result["gate_result"] == "NO_DATA"

    def test_g1_1_negative_pnl(self) -> None:
        """大幅負 PnL → E4 FAIL."""
        from scripts.v460.run_gate_check import run_g1_1

        records = _make_fill_records(
            n=100, fill_rate=0.98, pnl_mean=-2.0, as_ratio=0.05,
        )
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fill_records(records, Path(tmp))
            result = run_g1_1(tmp, thresholds=_default_thresholds_g1_1())

        assert result["checks"]["E4_post_fill_pnl"]["pass"] is False

    def test_g1_1_custom_thresholds(self) -> None:
        """カスタム閾値で判定."""
        from scripts.v460.run_gate_check import run_g1_1

        loose_thresholds = {
            "min_fill_rate_p90": 0.50,
            "max_cancel_ratio": 0.80,
            "max_queue_wait_median_sec": 300,
            "min_post_fill_30s_pnl": -10.0,
            "max_adverse_selection_ratio": 0.90,
        }
        records = _make_fill_records(n=50, fill_rate=0.60, pnl_mean=-1.0, as_ratio=0.40)
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fill_records(records, Path(tmp))
            result = run_g1_1(tmp, thresholds=loose_thresholds)

        assert result["gate_result"] == "PASS"


# =====================================================================
# G2-train テスト (031# F7)
# =====================================================================


def _make_g2_results(
    n_seeds: int = 5,
    gross_roi_mean: float = 0.03,
    ic_mean: float = 0.05,
    ic_std: float = 0.01,
    roi_var_pct: float = 2.0,
) -> dict:
    """G2 テスト用の results JSON を生成."""
    rng = np.random.RandomState(42)
    seed_results = []
    for i in range(n_seeds):
        seed_results.append({
            "seed": i,
            "gross_roi": rng.normal(gross_roi_mean, 0.01),
            "ic_mean": rng.normal(ic_mean, ic_std),
        })
    return {
        "seed_results": seed_results,
        "convergence": {"roi_variance_pct_after_30k": roi_var_pct},
    }


def _default_thresholds_g2() -> dict:
    return {
        "min_positive_seed_ratio": 0.75,
        "max_ic_seed_std": 0.03,
        "max_roi_variance_pct": 5.0,
        "worst_seed_min_roi": -0.02,
    }


def _make_g3_results(
    n_seeds: int = 5,
    pf_mean: float = 1.3,
    sharpe_mean: float = 1.5,
    max_dd: float = 0.08,
    gross: float = 0.005,
    fee: float = 0.002,
) -> dict:
    """G3 テスト用の results JSON を生成."""
    rng = np.random.RandomState(42)
    seed_metrics = []
    for i in range(n_seeds):
        seed_metrics.append({
            "seed": i,
            "pf": rng.normal(pf_mean, 0.1),
            "sharpe_annual": rng.normal(sharpe_mean, 0.2),
            "max_drawdown": rng.uniform(max_dd * 0.5, max_dd),
            "avg_gross_per_trade": gross,
            "avg_fee_per_trade": fee,
        })
    return {"seed_metrics": seed_metrics}


def _default_thresholds_g3() -> dict:
    return {
        "min_pf_median": 1.05,
        "min_pf_worst": 0.95,
        "gross_gt_fee": True,
        "max_drawdown": 0.15,
        "min_sharpe_annual": 0.8,
    }


def _make_g4_results(
    uptime_days: float = 10.0,
    downtime_ratio: float = 0.003,
    circuit_breaker_tested: bool = True,
    g3_maintained: bool = True,
    emergency_stop_sec: float = 0.3,
) -> dict:
    """G4 テスト用の results JSON を生成."""
    return {
        "uptime_days": uptime_days,
        "downtime_ratio": downtime_ratio,
        "circuit_breaker_tested": circuit_breaker_tested,
        "g3_maintained": g3_maintained,
        "emergency_stop_response_sec": emergency_stop_sec,
    }


def _default_thresholds_g4() -> dict:
    return {
        "min_paper_days": 7,
        "max_downtime_ratio": 0.01,
        "max_emergency_stop_sec": 1.0,
    }


class TestG2Train:
    """run_g2_judgment のテスト."""

    def _write_results(self, data: dict, tmp_dir: Path) -> str:
        path = tmp_dir / "g2_results.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def test_g2_pass(self) -> None:
        """全 seed 良好 → PASS."""
        from scripts.v460.run_gate_check import run_g2_judgment

        data = _make_g2_results(n_seeds=5, gross_roi_mean=0.05, ic_std=0.005, roi_var_pct=2.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(data, Path(tmp))
            result = run_g2_judgment(path, thresholds=_default_thresholds_g2())

        assert result["gate"] == "G2-train"
        assert result["gate_result"] == "PASS"
        assert all(c["pass"] for c in result["checks"].values())

    def test_g2_no_data(self) -> None:
        """seed_results 空 → NO_DATA."""
        from scripts.v460.run_gate_check import run_g2_judgment

        data = {"seed_results": [], "convergence": {}}
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(data, Path(tmp))
            result = run_g2_judgment(path, thresholds=_default_thresholds_g2())

        assert result["gate_result"] == "NO_DATA"

    def test_g2_low_positive_ratio(self) -> None:
        """positive seed 比率不足 → FAIL."""
        from scripts.v460.run_gate_check import run_g2_judgment

        data = _make_g2_results(n_seeds=5, gross_roi_mean=-0.03)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(data, Path(tmp))
            result = run_g2_judgment(path, thresholds=_default_thresholds_g2())

        assert result["checks"]["positive_seed_ratio"]["pass"] is False

    def test_g2_high_ic_std(self) -> None:
        """IC seed 間標準偏差が大きい → FAIL."""
        from scripts.v460.run_gate_check import run_g2_judgment

        data = _make_g2_results(n_seeds=5, ic_std=0.10)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(data, Path(tmp))
            result = run_g2_judgment(path, thresholds=_default_thresholds_g2())

        assert result["checks"]["ic_seed_std"]["pass"] is False

    def test_g2_poor_convergence(self) -> None:
        """ROI variance 高い → FAIL."""
        from scripts.v460.run_gate_check import run_g2_judgment

        data = _make_g2_results(n_seeds=5, roi_var_pct=15.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(data, Path(tmp))
            result = run_g2_judgment(path, thresholds=_default_thresholds_g2())

        assert result["checks"]["convergence"]["pass"] is False

    def test_g2_worst_seed_bad(self) -> None:
        """worst seed の ROI が閾値以下 → FAIL."""
        from scripts.v460.run_gate_check import run_g2_judgment

        # 手動で1つの seed を -5% にする
        data = _make_g2_results(n_seeds=5, gross_roi_mean=0.05)
        data["seed_results"][0]["gross_roi"] = -0.05
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(data, Path(tmp))
            result = run_g2_judgment(path, thresholds=_default_thresholds_g2())

        assert result["checks"]["worst_seed_roi"]["pass"] is False


class TestG3Pnl:
    """run_g3_judgment のテスト."""

    def _write_results(self, data: dict, tmp_dir: Path) -> str:
        path = tmp_dir / "g3_results.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def test_g3_pass(self) -> None:
        """全指標良好 → PASS."""
        from scripts.v460.run_gate_check import run_g3_judgment

        data = _make_g3_results(pf_mean=1.4, sharpe_mean=1.5, max_dd=0.08)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(data, Path(tmp))
            result = run_g3_judgment(path, thresholds=_default_thresholds_g3())

        assert result["gate"] == "G3-pnl"
        assert result["gate_result"] == "PASS"
        assert all(c["pass"] for c in result["checks"].values())

    def test_g3_no_data(self) -> None:
        """seed_metrics 空 → NO_DATA."""
        from scripts.v460.run_gate_check import run_g3_judgment

        data = {"seed_metrics": []}
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(data, Path(tmp))
            result = run_g3_judgment(path, thresholds=_default_thresholds_g3())

        assert result["gate_result"] == "NO_DATA"

    def test_g3_low_pf_median(self) -> None:
        """PF 中央値低い → FAIL."""
        from scripts.v460.run_gate_check import run_g3_judgment

        data = _make_g3_results(pf_mean=0.8)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(data, Path(tmp))
            result = run_g3_judgment(path, thresholds=_default_thresholds_g3())

        assert result["checks"]["pf_median"]["pass"] is False

    def test_g3_low_pf_worst(self) -> None:
        """PF worst が閾値以下 → FAIL."""
        from scripts.v460.run_gate_check import run_g3_judgment

        data = _make_g3_results(pf_mean=1.2)
        data["seed_metrics"][0]["pf"] = 0.5  # worst = 0.5 < 0.95
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(data, Path(tmp))
            result = run_g3_judgment(path, thresholds=_default_thresholds_g3())

        assert result["checks"]["pf_worst"]["pass"] is False

    def test_g3_fee_exceeds_gross(self) -> None:
        """gross < fee → FAIL."""
        from scripts.v460.run_gate_check import run_g3_judgment

        data = _make_g3_results(gross=0.001, fee=0.005)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(data, Path(tmp))
            result = run_g3_judgment(path, thresholds=_default_thresholds_g3())

        assert result["checks"]["gross_gt_fee"]["pass"] is False

    def test_g3_high_drawdown(self) -> None:
        """Max DD > 15% → FAIL."""
        from scripts.v460.run_gate_check import run_g3_judgment

        data = _make_g3_results(max_dd=0.25)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(data, Path(tmp))
            result = run_g3_judgment(path, thresholds=_default_thresholds_g3())

        assert result["checks"]["max_drawdown"]["pass"] is False

    def test_g3_low_sharpe(self) -> None:
        """Sharpe annual 低い → FAIL."""
        from scripts.v460.run_gate_check import run_g3_judgment

        data = _make_g3_results(sharpe_mean=0.3)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(data, Path(tmp))
            result = run_g3_judgment(path, thresholds=_default_thresholds_g3())

        assert result["checks"]["sharpe_annual"]["pass"] is False


class TestG4Live:
    """run_g4_judgment のテスト."""

    def _write_results(self, data: dict, tmp_dir: Path) -> str:
        path = tmp_dir / "g4_results.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def test_g4_pass(self) -> None:
        """全指標良好 → PASS."""
        from scripts.v460.run_gate_check import run_g4_judgment

        data = _make_g4_results()
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(data, Path(tmp))
            result = run_g4_judgment(path, thresholds=_default_thresholds_g4())

        assert result["gate"] == "G4-live"
        assert result["gate_result"] == "PASS"
        assert all(c["pass"] for c in result["checks"].values())

    def test_g4_low_uptime(self) -> None:
        """稼働日数不足 → FAIL."""
        from scripts.v460.run_gate_check import run_g4_judgment

        data = _make_g4_results(uptime_days=3.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(data, Path(tmp))
            result = run_g4_judgment(path, thresholds=_default_thresholds_g4())

        assert result["checks"]["uptime_days"]["pass"] is False

    def test_g4_high_downtime(self) -> None:
        """ダウンタイム比率超過 → FAIL."""
        from scripts.v460.run_gate_check import run_g4_judgment

        data = _make_g4_results(downtime_ratio=0.05)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(data, Path(tmp))
            result = run_g4_judgment(path, thresholds=_default_thresholds_g4())

        assert result["checks"]["downtime_ratio"]["pass"] is False

    def test_g4_no_circuit_breaker(self) -> None:
        """Circuit Breaker 未テスト → FAIL."""
        from scripts.v460.run_gate_check import run_g4_judgment

        data = _make_g4_results(circuit_breaker_tested=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(data, Path(tmp))
            result = run_g4_judgment(path, thresholds=_default_thresholds_g4())

        assert result["checks"]["circuit_breaker"]["pass"] is False

    def test_g4_g3_not_maintained(self) -> None:
        """G3 指標未維持 → FAIL."""
        from scripts.v460.run_gate_check import run_g4_judgment

        data = _make_g4_results(g3_maintained=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(data, Path(tmp))
            result = run_g4_judgment(path, thresholds=_default_thresholds_g4())

        assert result["checks"]["g3_maintained"]["pass"] is False

    def test_g4_slow_emergency_stop(self) -> None:
        """緊急停止応答遅延 → FAIL."""
        from scripts.v460.run_gate_check import run_g4_judgment

        data = _make_g4_results(emergency_stop_sec=5.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(data, Path(tmp))
            result = run_g4_judgment(path, thresholds=_default_thresholds_g4())

        assert result["checks"]["emergency_stop"]["pass"] is False

    def test_g4_multiple_failures(self) -> None:
        """複数指標 FAIL → FAIL + 全チェック記録."""
        from scripts.v460.run_gate_check import run_g4_judgment

        data = _make_g4_results(
            uptime_days=2.0,
            downtime_ratio=0.05,
            circuit_breaker_tested=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(data, Path(tmp))
            result = run_g4_judgment(path, thresholds=_default_thresholds_g4())

        assert result["gate_result"] == "FAIL"
        failed = [k for k, v in result["checks"].items() if not v["pass"]]
        assert len(failed) >= 3


# =====================================================================
# CLI テスト (G2/G3/G4 追加)
# =====================================================================


class TestCLI:
    """main() の CLI 引数テスト."""

    @patch("scripts.v460.run_gate_check.run_g0")
    def test_cli_g0(self, mock_run_g0) -> None:
        """--gate G0 で run_g0 が呼ばれる."""
        from scripts.v460.run_gate_check import main

        mock_run_g0.return_value = {"gate": "G0-data", "gate_result": "PASS", "checks": {}}

        with patch("sys.argv", ["prog", "--gate", "G0", "--data-path", "test.parquet"]):
            with patch("scripts.v460.run_gate_check.sys.exit") as mock_exit:
                main()
                mock_exit.assert_called_once_with(0)

        mock_run_g0.assert_called_once()

    @patch("scripts.v460.run_gate_check.run_g1_judgment")
    def test_cli_g1(self, mock_run_g1) -> None:
        """--gate G1 で run_g1_judgment が呼ばれる."""
        from scripts.v460.run_gate_check import main

        mock_run_g1.return_value = {"gate": "G1-info", "gate_result": "FAIL", "checks": {}}

        with patch("sys.argv", ["prog", "--gate", "G1", "--results-path", "g1.json"]):
            with patch("scripts.v460.run_gate_check.sys.exit") as mock_exit:
                main()
                mock_exit.assert_called_once_with(1)  # FAIL → exit 1

        mock_run_g1.assert_called_once()

    @patch("scripts.v460.run_gate_check.run_g1_1")
    def test_cli_g1_1(self, mock_run_g1_1) -> None:
        """--gate G1.1 で run_g1_1 が呼ばれる."""
        from scripts.v460.run_gate_check import main

        mock_run_g1_1.return_value = {
            "gate": "G1.1-exec",
            "gate_result": "PASS",
            "checks": {},
            "metrics_summary": {},
        }

        with patch("sys.argv", ["prog", "--gate", "G1.1", "--results-dir", "/tmp/fill"]):
            with patch("scripts.v460.run_gate_check.sys.exit") as mock_exit:
                main()
                mock_exit.assert_called_once_with(0)

        mock_run_g1_1.assert_called_once()

    @patch("scripts.v460.run_gate_check.run_g0")
    def test_cli_output_file(self, mock_run_g0) -> None:
        """--output で結果を JSON 出力."""
        from scripts.v460.run_gate_check import main

        mock_run_g0.return_value = {"gate": "G0-data", "gate_result": "PASS", "checks": {}}

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "result.json"
            with patch(
                "sys.argv",
                ["prog", "--gate", "G0", "--data-path", "test.parquet", "--output", str(out_path)],
            ):
                with patch("scripts.v460.run_gate_check.sys.exit"):
                    main()

            assert out_path.exists()
            loaded = json.loads(out_path.read_text(encoding="utf-8"))
            assert loaded["gate_result"] == "PASS"

    @patch("scripts.v460.run_gate_check.run_g2_judgment")
    def test_cli_g2(self, mock_run_g2) -> None:
        """--gate G2 で run_g2_judgment が呼ばれる."""
        from scripts.v460.run_gate_check import main

        mock_run_g2.return_value = {"gate": "G2-train", "gate_result": "PASS", "checks": {}}

        with patch("sys.argv", ["prog", "--gate", "G2", "--results-path", "g2.json"]):
            with patch("scripts.v460.run_gate_check.sys.exit") as mock_exit:
                main()
                mock_exit.assert_called_once_with(0)

        mock_run_g2.assert_called_once()

    @patch("scripts.v460.run_gate_check.run_g3_judgment")
    def test_cli_g3(self, mock_run_g3) -> None:
        """--gate G3 で run_g3_judgment が呼ばれる."""
        from scripts.v460.run_gate_check import main

        mock_run_g3.return_value = {"gate": "G3-pnl", "gate_result": "FAIL", "checks": {}}

        with patch("sys.argv", ["prog", "--gate", "G3", "--results-path", "g3.json"]):
            with patch("scripts.v460.run_gate_check.sys.exit") as mock_exit:
                main()
                mock_exit.assert_called_once_with(1)  # FAIL → exit 1

        mock_run_g3.assert_called_once()

    @patch("scripts.v460.run_gate_check.run_g4_judgment")
    def test_cli_g4(self, mock_run_g4) -> None:
        """--gate G4 で run_g4_judgment が呼ばれる."""
        from scripts.v460.run_gate_check import main

        mock_run_g4.return_value = {"gate": "G4-live", "gate_result": "PASS", "checks": {}}

        with patch("sys.argv", ["prog", "--gate", "G4", "--results-path", "g4.json"]):
            with patch("scripts.v460.run_gate_check.sys.exit") as mock_exit:
                main()
                mock_exit.assert_called_once_with(0)

        mock_run_g4.assert_called_once()
