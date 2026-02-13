"""
v460 Gate Check Runner 単体テスト.

run_gate_check.py の run_g0 / run_g1_judgment / run_g1_1 を検証する。
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
# CLI テスト
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
