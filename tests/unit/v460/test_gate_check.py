"""
v460 Gate Check Runner 単体テスト.

run_gate_check.py の run_g0 / run_g1_judgment / run_g1_1 / run_g2_judgment
/ run_g3_judgment / run_g4_judgment を検証する。
外部ファイル依存はすべて mock で遮断。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from tests.unit.v460._real_data_test_helpers import write_jsonl_sample

# テスト対象を遅延 import (sys.path 挿入済み前提)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import sys

sys.path.insert(0, str(_PROJECT_ROOT))
from scripts.v460.run_gate_check import (
    main,
    run_g0,
    run_g1_1,
    run_g1_judgment,
    run_g2_judgment,
    run_g3_judgment,
    run_g4_judgment,
)


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


_FEATURE_DF_10 = _make_feature_df(16, 10)
_FEATURE_DF_2 = _make_feature_df(16, 2)


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


class _ManifestPathStub:
    def __init__(self, *, exists: bool) -> None:
        self._exists = exists

    def exists(self) -> bool:
        return self._exists

    def __str__(self) -> str:
        return "results/v460/manifest.jsonl"


def _manifest_writer_stub(*, exists: bool = True) -> SimpleNamespace:
    return SimpleNamespace(path=_ManifestPathStub(exists=exists))


def _make_fill_records(
    n: int = 50,
    fill_rate: float = 0.95,
    pnl_mean: float = 0.1,
    as_ratio: float = 0.10,
    queue_wait: float = 15.0,
) -> list[dict]:
    """fill_records JSONL 用の dict リストを生成.

    135# P0-12: run_g1_1() delegation 後は filter_clean_records を通るため
    run_id / git_sha を含める必要がある。
    """
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
            "run_id": "test_run_001",
            "git_sha": "abc123def456",
        })
    return records


def _write_gate_results(tmp_dir: Path, filename: str, data: dict) -> str:
    path = tmp_dir / filename
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


# =====================================================================
# G0-data テスト
# =====================================================================


class TestRunG0:
    """run_g0 のテスト."""

    @patch("scripts.v460.run_gate_check.ManifestWriter")
    @patch("scripts.v460.run_gate_check.check_nan_ratio")
    @patch("scripts.v460.run_gate_check.load_parquet")
    @patch("scripts.v460.run_gate_check.count_feature_columns")
    @patch("scripts.v460.run_gate_check.compute_data_hash")
    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g0_pass_all(
        self, mock_thresh, mock_hash, mock_count, mock_load, mock_nan, mock_manifest,
    ) -> None:
        """全チェック PASS のケース."""

        mock_thresh.return_value = {"g0_data": _default_thresholds_g0()}
        mock_hash.return_value = "abcdef1234567890abcdef1234567890"
        mock_count.return_value = 10
        mock_load.return_value = _FEATURE_DF_10
        mock_nan.return_value = {"overall_nan_ratio": 0.001, "pass": True}
        mock_manifest.return_value = _manifest_writer_stub(exists=True)

        result = run_g0("dummy.parquet", "abcdef1234567890")
        assert result["gate"] == "G0-data"
        assert result["gate_result"] == "PASS"
        assert result["checks"]["data_hash"]["pass"] is True

    @patch("scripts.v460.run_gate_check.ManifestWriter")
    @patch("scripts.v460.run_gate_check.check_nan_ratio")
    @patch("scripts.v460.run_gate_check.load_parquet")
    @patch("scripts.v460.run_gate_check.count_feature_columns")
    @patch("scripts.v460.run_gate_check.compute_data_hash")
    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g0_hash_mismatch(
        self, mock_thresh, mock_hash, mock_count, mock_load, mock_nan, mock_manifest,
    ) -> None:
        """ハッシュ不一致 → FAIL."""

        mock_thresh.return_value = {"g0_data": _default_thresholds_g0()}
        mock_hash.return_value = "aaaa1111bbbb2222"
        mock_count.return_value = 10
        mock_load.return_value = _FEATURE_DF_10
        mock_nan.return_value = {"overall_nan_ratio": 0.0, "pass": True}
        mock_manifest.return_value = _manifest_writer_stub(exists=True)

        result = run_g0("dummy.parquet", "zzzz9999yyyy8888")
        assert result["gate_result"] == "FAIL"
        assert result["checks"]["data_hash"]["pass"] is False

    @patch("scripts.v460.run_gate_check.ManifestWriter")
    @patch("scripts.v460.run_gate_check.check_nan_ratio")
    @patch("scripts.v460.run_gate_check.load_parquet")
    @patch("scripts.v460.run_gate_check.count_feature_columns")
    @patch("scripts.v460.run_gate_check.compute_data_hash")
    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g0_no_expected_hash_passes(
        self, mock_thresh, mock_hash, mock_count, mock_load, mock_nan, mock_manifest,
    ) -> None:
        """expected_hash=None → hash チェック skip (PASS)."""

        mock_thresh.return_value = {"g0_data": _default_thresholds_g0()}
        mock_hash.return_value = "abcdef1234567890"
        mock_count.return_value = 10
        mock_load.return_value = _FEATURE_DF_10
        mock_nan.return_value = {"overall_nan_ratio": 0.0, "pass": True}
        mock_manifest.return_value = _manifest_writer_stub(exists=True)

        result = run_g0("dummy.parquet", None)
        assert result["checks"]["data_hash"]["pass"] is True

    @patch("scripts.v460.run_gate_check.ManifestWriter")
    @patch("scripts.v460.run_gate_check.check_nan_ratio")
    @patch("scripts.v460.run_gate_check.load_parquet")
    @patch("scripts.v460.run_gate_check.count_feature_columns")
    @patch("scripts.v460.run_gate_check.compute_data_hash")
    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g0_too_few_columns(
        self, mock_thresh, mock_hash, mock_count, mock_load, mock_nan, mock_manifest,
    ) -> None:
        """特徴量カラム < 4 → FAIL."""

        mock_thresh.return_value = {"g0_data": _default_thresholds_g0()}
        mock_hash.return_value = "abcdef1234567890"
        mock_count.return_value = 2
        mock_load.return_value = _FEATURE_DF_2
        mock_nan.return_value = {"overall_nan_ratio": 0.0, "pass": True}
        mock_manifest.return_value = _manifest_writer_stub(exists=True)

        result = run_g0("dummy.parquet")
        assert result["checks"]["feature_column_count"]["pass"] is False
        assert result["gate_result"] == "FAIL"

    @patch("scripts.v460.run_gate_check.ManifestWriter")
    @patch("scripts.v460.run_gate_check.check_nan_ratio")
    @patch("scripts.v460.run_gate_check.load_parquet")
    @patch("scripts.v460.run_gate_check.count_feature_columns")
    @patch("scripts.v460.run_gate_check.compute_data_hash")
    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g0_high_nan_ratio(
        self, mock_thresh, mock_hash, mock_count, mock_load, mock_nan, mock_manifest,
    ) -> None:
        """NaN 比率超過 → FAIL."""

        mock_thresh.return_value = {"g0_data": _default_thresholds_g0()}
        mock_hash.return_value = "abcdef1234567890"
        mock_count.return_value = 10
        mock_load.return_value = _FEATURE_DF_10
        mock_nan.return_value = {"overall_nan_ratio": 0.05, "pass": False}
        mock_manifest.return_value = _manifest_writer_stub(exists=True)

        result = run_g0("dummy.parquet")
        assert result["checks"]["nan_ratio"]["pass"] is False
        assert result["gate_result"] == "FAIL"

    @patch("scripts.v460.run_gate_check.ManifestWriter")
    @patch("scripts.v460.run_gate_check.check_nan_ratio")
    @patch("scripts.v460.run_gate_check.load_parquet")
    @patch("scripts.v460.run_gate_check.count_feature_columns")
    @patch("scripts.v460.run_gate_check.compute_data_hash")
    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g0_no_manifest(
        self, mock_thresh, mock_hash, mock_count, mock_load, mock_nan, mock_manifest,
    ) -> None:
        """manifest 不在 → FAIL."""

        mock_thresh.return_value = {"g0_data": _default_thresholds_g0()}
        mock_hash.return_value = "abcdef1234567890"
        mock_count.return_value = 10
        mock_load.return_value = _FEATURE_DF_10
        mock_nan.return_value = {"overall_nan_ratio": 0.0, "pass": True}
        mock_manifest.return_value = _manifest_writer_stub(exists=False)

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

    def _run_g1_judgment(self, payload: dict) -> dict:
        with patch(
            "scripts.v460.run_gate_check._load_results_payload",
            return_value=payload,
        ):
            return run_g1_judgment("dummy.json")

    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g1_pass(self, mock_thresh) -> None:
        """G1 全パス."""

        mock_thresh.return_value = {"g1_info": _default_thresholds_g1()}
        result = self._run_g1_judgment(self._make_g1_results())

        assert result["gate"] == "G1-info"
        assert result["gate_result"] == "PASS"

    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g1_judgment_cache_fail(self, mock_thresh) -> None:
        """g1_judgment_cache が FAIL → 全体 FAIL."""

        mock_thresh.return_value = {"g1_info": _default_thresholds_g1()}
        result = self._run_g1_judgment(self._make_g1_results(g1_pass=False))

        assert result["gate_result"] == "FAIL"

    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g1_low_ic(self, mock_thresh) -> None:
        """IC 不足 → threshold_checks FAIL → 全体 FAIL."""

        mock_thresh.return_value = {"g1_info": _default_thresholds_g1()}
        data = self._make_g1_results(ic_mean=0.005)  # below 0.02
        result = self._run_g1_judgment(data)

        tc = result["threshold_checks"]["target_5m"]
        assert tc["ic_pass"] is False
        assert result["gate_result"] == "FAIL"

    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g1_low_accuracy(self, mock_thresh) -> None:
        """accuracy 不足 → FAIL."""

        mock_thresh.return_value = {"g1_info": _default_thresholds_g1()}
        data = self._make_g1_results(acc_mean=0.48)  # below 0.51
        result = self._run_g1_judgment(data)

        tc = result["threshold_checks"]["target_5m"]
        assert tc["accuracy_pass"] is False
        assert result["gate_result"] == "FAIL"

    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g1_insufficient_sig_folds(self, mock_thresh) -> None:
        """sig_folds 不足 → FAIL."""

        mock_thresh.return_value = {"g1_info": _default_thresholds_g1()}
        data = self._make_g1_results(sig_count=1)  # below 2
        result = self._run_g1_judgment(data)

        tc = result["threshold_checks"]["target_5m"]
        assert tc["sig_folds_pass"] is False
        assert result["gate_result"] == "FAIL"

    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g1_no_cache_no_raw_folds(self, mock_thresh) -> None:
        """cache なし + raw fold_results なし → FAIL (stats-only)."""

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
        result = self._run_g1_judgment(data)

        # judgment g1_pass = False (stats-only fallback)
        assert result["gate_result"] == "FAIL"

    @patch("scripts.v460.run_gate_check.load_gate_thresholds")
    def test_g1_no_xgboost_no_extra_checks(self, mock_thresh) -> None:
        """xgboost ブロック空 → extra_any_pass=False → FAIL."""

        mock_thresh.return_value = {"g1_info": _default_thresholds_g1()}
        data = {
            "xgboost": {},
            "g1_judgment_cache": {"g1_pass": True, "passed_targets": [], "details": {}},
        }
        result = self._run_g1_judgment(data)

        assert result["gate_result"] == "FAIL"


# =====================================================================
# G1.1-exec テスト (135# P0-12: gate_judgment 委譲後)
# =====================================================================


class TestRunG1_1:
    """run_g1_1 のテスト.

    135# P0-12: run_g1_1() は内部的に gate_judgment.run_gate_judgment() へ委譲。
    返却形式が G1.1-exec (旧) → G1.1-quick (新) に変更。
    チェックキーも E1-E5 → K1-K6 に変更。
    """

    def _write_fill_records(self, records: list[dict], tmp_dir: Path) -> None:
        """JSONL ファイルに write."""
        path = tmp_dir / "fill_records_test.jsonl"
        write_jsonl_sample(path, records)

    def test_g1_1_pass(self, tmp_path: Path) -> None:
        """良好な fill_records → PASS."""

        records = _make_fill_records(
            n=50, fill_rate=0.98, pnl_mean=0.2, as_ratio=0.05, queue_wait=20.0,
        )
        self._write_fill_records(records, tmp_path)
        result = run_g1_1(tmp_path)

        # 135# P0-12: delegation 後は g1_1_quick_judgment 由来の "G1.1-quick"
        assert result["gate"] == "G1.1-quick"
        assert result["gate_result"] in ("PASS", "WATCH")

    def test_g1_1_low_fill_rate(self, tmp_path: Path) -> None:
        """低 fill_rate → FAIL (K1_attempted_fill_rate)."""

        records = _make_fill_records(n=50, fill_rate=0.50)
        self._write_fill_records(records, tmp_path)
        result = run_g1_1(tmp_path)

        assert result["gate_result"] == "FAIL"
        assert result["checks"]["K1_attempted_fill_rate"]["pass"] is False

    def test_g1_1_no_data(self, tmp_path: Path) -> None:
        """データなし → NO_DATA."""
        result = run_g1_1(tmp_path)

        assert result["gate_result"] == "NO_DATA"

    def test_g1_1_negative_pnl(self, tmp_path: Path) -> None:
        """大幅負 PnL → K4_pnl_kill FAIL."""

        records = _make_fill_records(
            n=100, fill_rate=0.98, pnl_mean=-2.0, as_ratio=0.05,
        )
        self._write_fill_records(records, tmp_path)
        result = run_g1_1(tmp_path)

        # K4 は pValue & mean の複合条件 — 大幅負 PnL + 有意ならFAIL
        assert result["checks"]["K4_pnl_kill"]["pass"] is False

    def test_g1_1_deprecation_warning(self, tmp_path: Path) -> None:
        """DeprecationWarning が発行されること."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_g1_1(tmp_path)
        dep_warns = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warns) >= 1
        assert "gate_judgment" in str(dep_warns[0].message)


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
    reward_profit_corr: float = 0.2,
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
            "reward_profit_corr": rng.normal(reward_profit_corr, 0.05),
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

    def test_g2_pass(self, tmp_path: Path) -> None:
        """全 seed 良好 → PASS."""

        data = _make_g2_results(n_seeds=5, gross_roi_mean=0.05, ic_std=0.005, roi_var_pct=2.0)
        path = _write_gate_results(tmp_path, "g2_results.json", data)
        result = run_g2_judgment(path, thresholds=_default_thresholds_g2())

        assert result["gate"] == "G2-train"
        assert result["gate_result"] == "PASS"
        assert all(c["pass"] for c in result["checks"].values())

    def test_g2_no_data(self, tmp_path: Path) -> None:
        """seed_results 空 → NO_DATA."""

        data = {"seed_results": [], "convergence": {}}
        path = _write_gate_results(tmp_path, "g2_results.json", data)
        result = run_g2_judgment(path, thresholds=_default_thresholds_g2())

        assert result["gate_result"] == "NO_DATA"

    def test_g2_low_positive_ratio(self, tmp_path: Path) -> None:
        """positive seed 比率不足 → FAIL."""

        data = _make_g2_results(n_seeds=5, gross_roi_mean=-0.03)
        path = _write_gate_results(tmp_path, "g2_results.json", data)
        result = run_g2_judgment(path, thresholds=_default_thresholds_g2())

        assert result["checks"]["positive_seed_ratio"]["pass"] is False

    def test_g2_high_ic_std(self, tmp_path: Path) -> None:
        """IC seed 間標準偏差が大きい → FAIL."""

        data = _make_g2_results(n_seeds=5, ic_std=0.10)
        path = _write_gate_results(tmp_path, "g2_results.json", data)
        result = run_g2_judgment(path, thresholds=_default_thresholds_g2())

        assert result["checks"]["ic_seed_std"]["pass"] is False

    def test_g2_poor_convergence(self, tmp_path: Path) -> None:
        """ROI variance 高い → FAIL."""

        data = _make_g2_results(n_seeds=5, roi_var_pct=15.0)
        path = _write_gate_results(tmp_path, "g2_results.json", data)
        result = run_g2_judgment(path, thresholds=_default_thresholds_g2())

        assert result["checks"]["convergence"]["pass"] is False

    def test_g2_worst_seed_bad(self, tmp_path: Path) -> None:
        """worst seed の ROI が閾値以下 → FAIL."""

        # 手動で1つの seed を -5% にする
        data = _make_g2_results(n_seeds=5, gross_roi_mean=0.05)
        data["seed_results"][0]["gross_roi"] = -0.05
        path = _write_gate_results(tmp_path, "g2_results.json", data)
        result = run_g2_judgment(path, thresholds=_default_thresholds_g2())

        assert result["checks"]["worst_seed_roi"]["pass"] is False


class TestG3Pnl:
    """run_g3_judgment のテスト."""

    def test_g3_pass(self, tmp_path: Path) -> None:
        """全指標良好 → PASS."""

        data = _make_g3_results(pf_mean=1.4, sharpe_mean=1.5, max_dd=0.08)
        path = _write_gate_results(tmp_path, "g3_results.json", data)
        result = run_g3_judgment(path, thresholds=_default_thresholds_g3())

        assert result["gate"] == "G3-pnl"
        assert result["gate_result"] == "PASS"
        assert all(c["pass"] for c in result["checks"].values())

    def test_g3_no_data(self, tmp_path: Path) -> None:
        """seed_metrics 空 → NO_DATA."""

        data = {"seed_metrics": []}
        path = _write_gate_results(tmp_path, "g3_results.json", data)
        result = run_g3_judgment(path, thresholds=_default_thresholds_g3())

        assert result["gate_result"] == "NO_DATA"

    def test_g3_low_pf_median(self, tmp_path: Path) -> None:
        """PF 中央値低い → FAIL."""

        data = _make_g3_results(pf_mean=0.8)
        path = _write_gate_results(tmp_path, "g3_results.json", data)
        result = run_g3_judgment(path, thresholds=_default_thresholds_g3())

        assert result["checks"]["pf_median"]["pass"] is False

    def test_g3_low_pf_worst(self, tmp_path: Path) -> None:
        """PF worst が閾値以下 → FAIL."""

        data = _make_g3_results(pf_mean=1.2)
        data["seed_metrics"][0]["pf"] = 0.5  # worst = 0.5 < 0.95
        path = _write_gate_results(tmp_path, "g3_results.json", data)
        result = run_g3_judgment(path, thresholds=_default_thresholds_g3())

        assert result["checks"]["pf_worst"]["pass"] is False

    def test_g3_fee_exceeds_gross(self, tmp_path: Path) -> None:
        """gross < fee → FAIL."""

        data = _make_g3_results(gross=0.001, fee=0.005)
        path = _write_gate_results(tmp_path, "g3_results.json", data)
        result = run_g3_judgment(path, thresholds=_default_thresholds_g3())

        assert result["checks"]["gross_gt_fee"]["pass"] is False

    def test_g3_high_drawdown(self, tmp_path: Path) -> None:
        """Max DD > 15% → FAIL."""

        data = _make_g3_results(max_dd=0.25)
        path = _write_gate_results(tmp_path, "g3_results.json", data)
        result = run_g3_judgment(path, thresholds=_default_thresholds_g3())

        assert result["checks"]["max_drawdown"]["pass"] is False

    def test_g3_low_sharpe(self, tmp_path: Path) -> None:
        """Sharpe annual 低い → FAIL."""

        data = _make_g3_results(sharpe_mean=0.3)
        path = _write_gate_results(tmp_path, "g3_results.json", data)
        result = run_g3_judgment(path, thresholds=_default_thresholds_g3())

        assert result["checks"]["sharpe_annual"]["pass"] is False


class TestG4Live:
    """run_g4_judgment のテスト."""

    def test_g4_pass(self, tmp_path: Path) -> None:
        """全指標良好 → PASS."""

        data = _make_g4_results()
        path = _write_gate_results(tmp_path, "g4_results.json", data)
        result = run_g4_judgment(path, thresholds=_default_thresholds_g4())

        assert result["gate"] == "G4-live"
        assert result["gate_result"] == "PASS"
        assert all(c["pass"] for c in result["checks"].values())

    def test_g4_low_uptime(self, tmp_path: Path) -> None:
        """稼働日数不足 → FAIL."""

        data = _make_g4_results(uptime_days=3.0)
        path = _write_gate_results(tmp_path, "g4_results.json", data)
        result = run_g4_judgment(path, thresholds=_default_thresholds_g4())

        assert result["checks"]["uptime_days"]["pass"] is False

    def test_g4_high_downtime(self, tmp_path: Path) -> None:
        """ダウンタイム比率超過 → FAIL."""

        data = _make_g4_results(downtime_ratio=0.05)
        path = _write_gate_results(tmp_path, "g4_results.json", data)
        result = run_g4_judgment(path, thresholds=_default_thresholds_g4())

        assert result["checks"]["downtime_ratio"]["pass"] is False

    def test_g4_no_circuit_breaker(self, tmp_path: Path) -> None:
        """Circuit Breaker 未テスト → FAIL."""

        data = _make_g4_results(circuit_breaker_tested=False)
        path = _write_gate_results(tmp_path, "g4_results.json", data)
        result = run_g4_judgment(path, thresholds=_default_thresholds_g4())

        assert result["checks"]["circuit_breaker"]["pass"] is False

    def test_g4_g3_not_maintained(self, tmp_path: Path) -> None:
        """G3 指標未維持 → FAIL."""

        data = _make_g4_results(g3_maintained=False)
        path = _write_gate_results(tmp_path, "g4_results.json", data)
        result = run_g4_judgment(path, thresholds=_default_thresholds_g4())

        assert result["checks"]["g3_maintained"]["pass"] is False

    def test_g4_slow_emergency_stop(self, tmp_path: Path) -> None:
        """緊急停止応答遅延 → FAIL."""

        data = _make_g4_results(emergency_stop_sec=5.0)
        path = _write_gate_results(tmp_path, "g4_results.json", data)
        result = run_g4_judgment(path, thresholds=_default_thresholds_g4())

        assert result["checks"]["emergency_stop"]["pass"] is False

    def test_g4_multiple_failures(self, tmp_path: Path) -> None:
        """複数指標 FAIL → FAIL + 全チェック記録."""

        data = _make_g4_results(
            uptime_days=2.0,
            downtime_ratio=0.05,
            circuit_breaker_tested=False,
        )
        path = _write_gate_results(tmp_path, "g4_results.json", data)
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

        mock_run_g0.return_value = {"gate": "G0-data", "gate_result": "PASS", "checks": {}}

        with patch("sys.argv", ["prog", "--gate", "G0", "--data-path", "test.parquet"]):
            with patch("scripts.v460.run_gate_check.sys.exit") as mock_exit:
                main()
                mock_exit.assert_called_once_with(0)

        mock_run_g0.assert_called_once()

    @patch("scripts.v460.run_gate_check.run_g1_judgment")
    def test_cli_g1(self, mock_run_g1) -> None:
        """--gate G1 で run_g1_judgment が呼ばれる."""

        mock_run_g1.return_value = {"gate": "G1-info", "gate_result": "FAIL", "checks": {}}

        with patch("sys.argv", ["prog", "--gate", "G1", "--results-path", "g1.json"]):
            with patch("scripts.v460.run_gate_check.sys.exit") as mock_exit:
                main()
                mock_exit.assert_called_once_with(1)  # FAIL → exit 1

        mock_run_g1.assert_called_once()

    @patch("scripts.v460.run_gate_check.run_g1_1")
    def test_cli_g1_1(self, mock_run_g1_1) -> None:
        """--gate G1.1 で run_g1_1 が呼ばれる."""

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
    def test_cli_output_file(self, mock_run_g0, tmp_path: Path) -> None:
        """--output で結果を JSON 出力."""

        mock_run_g0.return_value = {"gate": "G0-data", "gate_result": "PASS", "checks": {}}

        out_path = tmp_path / "result.json"
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

        mock_run_g2.return_value = {"gate": "G2-train", "gate_result": "PASS", "checks": {}}

        with patch("sys.argv", ["prog", "--gate", "G2", "--results-path", "g2.json"]):
            with patch("scripts.v460.run_gate_check.sys.exit") as mock_exit:
                main()
                mock_exit.assert_called_once_with(0)

        mock_run_g2.assert_called_once()

    @patch("scripts.v460.run_gate_check.run_g3_judgment")
    def test_cli_g3(self, mock_run_g3) -> None:
        """--gate G3 で run_g3_judgment が呼ばれる."""

        mock_run_g3.return_value = {"gate": "G3-pnl", "gate_result": "FAIL", "checks": {}}

        with patch("sys.argv", ["prog", "--gate", "G3", "--results-path", "g3.json"]):
            with patch("scripts.v460.run_gate_check.sys.exit") as mock_exit:
                main()
                mock_exit.assert_called_once_with(1)  # FAIL → exit 1

        mock_run_g3.assert_called_once()

    @patch("scripts.v460.run_gate_check.run_g4_judgment")
    def test_cli_g4(self, mock_run_g4) -> None:
        """--gate G4 で run_g4_judgment が呼ばれる."""

        mock_run_g4.return_value = {"gate": "G4-live", "gate_result": "PASS", "checks": {}}

        with patch("sys.argv", ["prog", "--gate", "G4", "--results-path", "g4.json"]):
            with patch("scripts.v460.run_gate_check.sys.exit") as mock_exit:
                main()
                mock_exit.assert_called_once_with(0)

        mock_run_g4.assert_called_once()
