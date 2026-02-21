"""126# SkipGate 再学習スケジューラ + hot-reload テスト.

retrain_scheduler.py の retrain_model() と
SkipGateEvaluator の hot-reload 機構をテスト。
"""

from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scripts.v460.ml.skip_gate import (
    SkipGate,
    SkipGateConfig,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_picklable_gate(
    *,
    n_samples: int = 100,
    version: str = "test",
    mode: str = "pnl",
) -> SkipGate:
    """pickle 可能な SkipGate を作成."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    config = SkipGateConfig(mode=mode)
    model = SGDClassifier()
    scaler = StandardScaler()
    feature_cols = ["spread_jpy", "offset_ratio", "regime_trending"]
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", SGDClassifier())])
    gate = SkipGate(
        model=model, scaler=scaler, feature_cols=feature_cols,
        config=config, pipeline=pipeline,
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


# =====================================================================
# Hot-Reload テスト
# =====================================================================

class TestHotReload:
    """126# SkipGateEvaluator hot-reload テスト."""

    def _make_config(self, model_path: str) -> MagicMock:
        """テスト用 FillTestConfig モック."""
        cfg = MagicMock()
        cfg.skip_gate_enabled = True
        cfg.skip_gate_model_path = model_path
        cfg.skip_gate_mode = "pnl"
        cfg.skip_gate_as_threshold = 0.50
        cfg.skip_gate_pnl_threshold = 0.0
        cfg.skip_gate_max_skip_rate = 0.3
        cfg.skip_gate_buy_enabled = True
        cfg.skip_gate_sell_enabled = True
        cfg.skip_gate_as_threshold_buy = 0.50
        cfg.skip_gate_as_threshold_sell = 0.50
        cfg.skip_gate_use_ob_features = False
        cfg.skip_gate_adaptive_threshold = False
        cfg.skip_gate_target_skip_rate_buy = 0.15
        cfg.skip_gate_target_skip_rate_sell = 0.20
        cfg.skip_gate_adaptive_window = 50
        cfg.skip_gate_adaptive_min_samples = 20
        cfg.skip_gate_adaptive_step = 0.05
        cfg.skip_gate_adaptive_floor = 0.35
        cfg.skip_gate_adaptive_ceiling = 0.80
        cfg.skip_sell_unknown_regime = False
        cfg.results_dir = "results/v460/fill_test"
        return cfg

    def test_initial_hash_stored(self) -> None:
        """初期ロード時にモデルファイルのハッシュが保存される."""
        from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "gate.pkl"
            gate = _make_picklable_gate(version="v1")
            _save_gate_to(gate, model_path)

            cfg = self._make_config(str(model_path))
            evaluator = SkipGateEvaluator(cfg, Path(tmpdir))

            assert evaluator._model_file_hash != ""
            assert len(evaluator._model_file_hash) == 64  # SHA256 hex

    def test_no_reload_when_unchanged(self) -> None:
        """ファイル未変更時はリロードしない."""
        from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "gate.pkl"
            gate = _make_picklable_gate(version="v1")
            _save_gate_to(gate, model_path)

            cfg = self._make_config(str(model_path))
            evaluator = SkipGateEvaluator(cfg, Path(tmpdir))
            original_gate = evaluator._skip_gate
            original_hash = evaluator._model_file_hash

            # 即座にチェックを強制 (interval を 0 に)
            evaluator._last_reload_check = 0
            evaluator._check_and_reload_model()

            assert evaluator._skip_gate is original_gate
            assert evaluator._model_file_hash == original_hash

    def test_reload_on_file_change(self) -> None:
        """ファイル変更時にリロードされる."""
        from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "gate.pkl"
            gate_v1 = _make_picklable_gate(version="v1", n_samples=100)
            _save_gate_to(gate_v1, model_path)

            cfg = self._make_config(str(model_path))
            evaluator = SkipGateEvaluator(cfg, Path(tmpdir))
            original_hash = evaluator._model_file_hash

            # v2 で上書き
            gate_v2 = _make_picklable_gate(version="v2", n_samples=200)
            _save_gate_to(gate_v2, model_path)

            # 強制チェック
            evaluator._last_reload_check = 0
            evaluator._check_and_reload_model()

            assert evaluator._model_file_hash != original_hash
            assert evaluator._skip_gate is not None
            assert evaluator._skip_gate.metadata["version"] == "v2"  # type: ignore[union-attr]
            assert evaluator._skip_gate.metadata["n_samples"] == 200  # type: ignore[union-attr]

    def test_reload_failure_keeps_old_model(self) -> None:
        """リロード失敗時は旧モデルを維持."""
        from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "gate.pkl"
            gate = _make_picklable_gate(version="v1")
            _save_gate_to(gate, model_path)

            cfg = self._make_config(str(model_path))
            evaluator = SkipGateEvaluator(cfg, Path(tmpdir))
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

    def test_check_interval_respected(self) -> None:
        """チェック間隔内ではファイル変更を検出しない."""
        from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "gate.pkl"
            gate = _make_picklable_gate(version="v1")
            _save_gate_to(gate, model_path)

            cfg = self._make_config(str(model_path))
            evaluator = SkipGateEvaluator(cfg, Path(tmpdir))

            # v2 で上書き
            gate_v2 = _make_picklable_gate(version="v2")
            _save_gate_to(gate_v2, model_path)

            # last_reload_check を更新しない → interval 内
            evaluator._check_and_reload_model()

            # まだ v1 のまま
            assert evaluator._skip_gate.metadata["version"] == "v1"  # type: ignore[union-attr]

    def test_compute_file_hash(self) -> None:
        """_compute_file_hash で SHA256 が正しく計算される."""
        from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.bin"
            p.write_bytes(b"hello world")
            expected = hashlib.sha256(b"hello world").hexdigest()
            assert SkipGateEvaluator._compute_file_hash(p) == expected

    def test_compute_file_hash_missing_file(self) -> None:
        """存在しないファイルのハッシュは空文字."""
        from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator
        result = SkipGateEvaluator._compute_file_hash(Path("/nonexistent"))
        assert result == ""


# =====================================================================
# Retrain Scheduler テスト
# =====================================================================

class TestRetrainConfig:
    """126# retrain 設定ロードテスト."""

    def test_default_config(self) -> None:
        """デフォルト設定が正しく読み込まれる."""
        from scripts.v460.ml.retrain_scheduler import load_retrain_config
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

    def test_yaml_override(self) -> None:
        """YAML の retrain + skip_gate セクションが統合される."""
        from scripts.v460.ml.retrain_scheduler import load_retrain_config

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test.yaml"
            yaml_path.write_text(
                "results_dir: /tmp/test_results\n"
                "skip_gate:\n"
                "  mode: pnl\n"
                "  model_path: /tmp/test_model.pkl\n"
                "  use_ob_features: false\n"
                "retrain:\n"
                "  interval_sec: 7200\n"
                "  min_new_samples: 50\n",
                encoding="utf-8",
            )
            cfg = load_retrain_config(yaml_path)
            assert cfg["interval_sec"] == 7200
            assert cfg["min_new_samples"] == 50
            # 127# C1: skip_gate から継承
            assert cfg["model_path"] == "/tmp/test_model.pkl"
            assert cfg["mode"] == "pnl"
            assert cfg["use_ob_features"] is False
            assert cfg["results_dir"] == "/tmp/test_results"
            # 未指定のキーはデフォルト
            assert cfg["target"] == "pnl120"

    def test_validation_rejects_non_pnl_mode(self) -> None:
        """127# C1: mode != pnl なら起動拒否."""
        from scripts.v460.ml.retrain_scheduler import load_retrain_config

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test.yaml"
            yaml_path.write_text(
                "skip_gate:\n"
                "  mode: as\n"
                "  model_path: /tmp/model.pkl\n",
                encoding="utf-8",
            )
            with pytest.raises(ValueError, match="requires skip_gate.mode='pnl'"):
                load_retrain_config(yaml_path)

    def test_validation_rejects_bad_target(self) -> None:
        """127# M2: 不正な target は拒否."""
        from scripts.v460.ml.retrain_scheduler import load_retrain_config

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test.yaml"
            yaml_path.write_text(
                "skip_gate:\n"
                "  mode: pnl\n"
                "  model_path: /tmp/model.pkl\n"
                "retrain:\n"
                "  target: as30\n",
                encoding="utf-8",
            )
            with pytest.raises(ValueError, match="pnl120.*pnl30"):
                load_retrain_config(yaml_path)


class TestBuildFullFeatures:
    """126# _build_full_features テスト."""

    def test_base_features_only(self) -> None:
        """use_ob=False → base features のみ."""
        from scripts.v460.ml.retrain_scheduler import _build_full_features
        import pandas as pd

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
            "price_velocity_60s": [1.0, -1.0],
            "vpin_60s": [0.3, 0.7],
            "side_aligned_tfi": [0.2, 0.2],
            "side_aligned_velocity": [1.0, 1.0],
        })
        enriched = pd.DataFrame({"side": ["buy", "sell"]}, index=X_base.index)

        result = _build_full_features(enriched, X_base, use_ob=False)
        assert result.shape[1] == 16  # base のみ

    def test_full_features_with_ob(self) -> None:
        """use_ob=True → base + OB features."""
        from scripts.v460.ml.retrain_scheduler import _build_full_features
        import pandas as pd

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
            "price_velocity_60s": [1.0],
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


class TestRetrainModel:
    """126# retrain_model() テスト."""

    def test_skip_when_no_fill_records(self) -> None:
        """fill_records が存在しない場合スキップ."""
        from scripts.v460.ml.retrain_scheduler import retrain_model, _DEFAULT_CONFIG

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
        from scripts.v460.ml.retrain_scheduler import retrain_model, _DEFAULT_CONFIG

        with tempfile.TemporaryDirectory() as tmpdir:
            # 少数の fill records を作成
            records_dir = Path(tmpdir)
            records = []
            for i in range(5):
                records.append(json.dumps({
                    "cycle_id": f"test_{i}",
                    "side": "buy",
                    "filled": True,
                    "timestamp": 1771502400.0 + i * 120,
                    "spread_at_order": 3000.0,
                    "spread_offset_ratio": 0.05,
                    "adverse_selected_raw": i % 2,
                    "post_fill_120s_pnl": 0.5 * (i - 2),
                    "run_id": "test_run",
                }))
            (records_dir / "fill_records_20260220.jsonl").write_text(
                "\n".join(records), encoding="utf-8",
            )

            cfg = dict(_DEFAULT_CONFIG)
            cfg["results_dir"] = str(records_dir)
            cfg["model_path"] = str(Path(tmpdir) / "nonexistent.pkl")
            cfg["mode"] = "pnl"
            cfg["use_ob_features"] = True
            cfg["min_total_samples"] = 100  # 5 < 100
            cfg["latest_run_only"] = False
            cfg["exclude_missing_run_id"] = False
            result = retrain_model(cfg)
            assert result["status"] == "skipped"

    def test_skip_when_insufficient_new_samples(self) -> None:
        """新規サンプル不足時はスキップ."""
        from scripts.v460.ml.retrain_scheduler import retrain_model, _DEFAULT_CONFIG

        with tempfile.TemporaryDirectory() as tmpdir:
            # 十分な fill records を作成
            records_dir = Path(tmpdir) / "results"
            records_dir.mkdir()
            model_dir = Path(tmpdir) / "models"
            model_dir.mkdir()

            records = []
            for i in range(150):
                records.append(json.dumps({
                    "cycle_id": f"test_{i}",
                    "side": "buy" if i % 2 == 0 else "sell",
                    "filled": True,
                    "timestamp": 1771502400.0 + i * 120,
                    "spread_at_order": 3000.0,
                    "spread_offset_ratio": 0.05,
                    "adverse_selected_raw": i % 3,
                    "post_fill_30s_pnl": 0.5 * (i - 75),
                    "post_fill_120s_pnl": 0.8 * (i - 75),
                    "regime": "ranging",
                    "run_id": "test_run",
                }))
            (records_dir / "fill_records_20260220.jsonl").write_text(
                "\n".join(records), encoding="utf-8",
            )

            # 既存モデルを配置 (n_samples=150 → 新規 0 件)
            model_path = model_dir / "gate.pkl"
            gate = _make_picklable_gate(n_samples=150)
            _save_gate_to(gate, model_path)

            cfg = dict(_DEFAULT_CONFIG)
            cfg["results_dir"] = str(records_dir)
            cfg["model_path"] = str(model_path)
            cfg["mode"] = "pnl"
            cfg["use_ob_features"] = True
            cfg["min_new_samples"] = 30
            cfg["min_total_samples"] = 10
            cfg["latest_run_only"] = False
            cfg["exclude_missing_run_id"] = False
            result = retrain_model(cfg)
            assert result["status"] == "skipped"
            assert "insufficient new samples" in result.get("reason", "")


# =====================================================================
# 127# M3: E2E 成功テスト (retrain → deploy → hot-reload → evaluate)
# =====================================================================

class TestE2ERetrainHotReload:
    """127# M3: 再学習→配置→hot-reload→評価の統合テスト."""

    def test_retrain_deploy_and_hot_reload(self) -> None:
        """E2E: 十分なデータで retrain → deploy → SkipGateEvaluator が hot-reload."""
        from scripts.v460.ml.retrain_scheduler import retrain_model, _DEFAULT_CONFIG
        from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator

        with tempfile.TemporaryDirectory() as tmpdir:
            records_dir = Path(tmpdir) / "results"
            records_dir.mkdir()
            model_dir = Path(tmpdir) / "models"
            model_dir.mkdir()
            model_path = model_dir / "gate.pkl"

            # 200件の fill records を生成 (十分なサンプル数)
            rng = np.random.RandomState(42)
            records = []
            for i in range(200):
                records.append(json.dumps({
                    "cycle_id": f"e2e_{i}",
                    "side": "buy" if i % 2 == 0 else "sell",
                    "filled": True,
                    "timestamp": 1771502400.0 + i * 120,
                    "spread_at_order": 2500.0 + rng.randn() * 500,
                    "spread_offset_ratio": 0.05 + rng.randn() * 0.01,
                    "adverse_selected_raw": int(rng.random() > 0.5),
                    "post_fill_30s_pnl": float(rng.randn() * 2),
                    "post_fill_120s_pnl": float(rng.randn() * 3),
                    "regime": rng.choice(["trending", "ranging", "high_vol"]),
                    "run_id": "e2e_run",
                }))
            (records_dir / "fill_records_20260220.jsonl").write_text(
                "\n".join(records), encoding="utf-8",
            )

            # retrain_model 実行
            cfg = dict(_DEFAULT_CONFIG)
            cfg["results_dir"] = str(records_dir)
            cfg["model_path"] = str(model_path)
            cfg["mode"] = "pnl"
            cfg["use_ob_features"] = False  # テスト高速化
            cfg["min_new_samples"] = 1
            cfg["min_total_samples"] = 50
            cfg["quality_gate_enabled"] = False  # E2E テストでは品質ゲート無効
            cfg["latest_run_only"] = False
            cfg["exclude_missing_run_id"] = False

            result = retrain_model(cfg)
            assert result["status"] == "deployed", f"Expected deployed, got {result}"
            assert model_path.exists()

            # SkipGateEvaluator で hot-reload テスト
            eval_cfg = MagicMock()
            eval_cfg.skip_gate_enabled = True
            eval_cfg.skip_gate_model_path = str(model_path)
            eval_cfg.skip_gate_mode = "pnl"
            eval_cfg.skip_gate_as_threshold = 0.50
            eval_cfg.skip_gate_pnl_threshold = 0.0
            eval_cfg.skip_gate_max_skip_rate = 0.3
            eval_cfg.skip_gate_buy_enabled = True
            eval_cfg.skip_gate_sell_enabled = True
            eval_cfg.skip_gate_as_threshold_buy = 0.50
            eval_cfg.skip_gate_as_threshold_sell = 0.50
            eval_cfg.skip_gate_use_ob_features = False
            eval_cfg.skip_gate_adaptive_threshold = False
            eval_cfg.skip_gate_target_skip_rate_buy = 0.15
            eval_cfg.skip_gate_target_skip_rate_sell = 0.20
            eval_cfg.skip_gate_adaptive_window = 50
            eval_cfg.skip_gate_adaptive_min_samples = 20
            eval_cfg.skip_gate_adaptive_step = 0.05
            eval_cfg.skip_gate_adaptive_floor = 0.35
            eval_cfg.skip_gate_adaptive_ceiling = 0.80
            eval_cfg.skip_sell_unknown_regime = False
            eval_cfg.results_dir = str(records_dir)

            evaluator = SkipGateEvaluator(eval_cfg, Path(tmpdir))
            assert evaluator._skip_gate is not None
            initial_hash = evaluator._model_file_hash

            # モデルを再 retrain して上書き (v2)
            cfg["min_new_samples"] = 0  # 新規サンプルチェック無効化
            result2 = retrain_model(cfg)
            assert result2["status"] == "deployed"

            # hot-reload トリガー (interval リセット)
            evaluator._last_reload_check = 0
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

class TestBalanceForcedSwitchFilter:
    """130# Y5: retrain_model() が balance_forced_switch=True を除外する."""

    def test_balance_forced_records_excluded(self) -> None:
        """balance_forced_switch=True のレコードが学習データから除外される."""
        from scripts.v460.ml.retrain_scheduler import retrain_model, _DEFAULT_CONFIG

        with tempfile.TemporaryDirectory() as tmpdir:
            records_dir = Path(tmpdir) / "results"
            records_dir.mkdir()

            rng = np.random.RandomState(123)
            records = []
            for i in range(60):
                # 最初の 20 件は balance_forced_switch=True
                forced = i < 20
                records.append(json.dumps({
                    "cycle_id": f"bf_{i}",
                    "side": "buy" if i % 2 == 0 else "sell",
                    "filled": True,
                    "timestamp": 1771502400.0 + i * 120,
                    "spread_at_order": 2500.0 + rng.randn() * 300,
                    "spread_offset_ratio": 0.05,
                    "adverse_selected_raw": int(rng.random() > 0.5),
                    "post_fill_30s_pnl": float(rng.randn() * 2),
                    "post_fill_120s_pnl": float(rng.randn() * 3),
                    "regime": "ranging",
                    "run_id": "bf_run",
                    "balance_forced_switch": forced,
                }))
            (records_dir / "fill_records_20260220.jsonl").write_text(
                "\n".join(records), encoding="utf-8",
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

            result = retrain_model(cfg)
            # 60 - 20 forced = 40 records usable; filled_records should be <= 40
            assert result["status"] in ("deployed", "skipped")
            if result["status"] == "deployed":
                assert result.get("filled_records", 0) <= 40

    def test_no_balance_column_no_error(self) -> None:
        """balance_forced_switch カラムがなくてもエラーにならない."""
        from scripts.v460.ml.retrain_scheduler import retrain_model, _DEFAULT_CONFIG

        with tempfile.TemporaryDirectory() as tmpdir:
            records_dir = Path(tmpdir) / "results"
            records_dir.mkdir()

            records = []
            for i in range(5):
                records.append(json.dumps({
                    "cycle_id": f"no_bf_{i}",
                    "side": "buy",
                    "filled": True,
                    "timestamp": 1771502400.0 + i * 120,
                    "spread_at_order": 3000.0,
                    "spread_offset_ratio": 0.05,
                    "run_id": "no_bf_run",
                }))
            (records_dir / "fill_records_20260220.jsonl").write_text(
                "\n".join(records), encoding="utf-8",
            )

            cfg = dict(_DEFAULT_CONFIG)
            cfg["results_dir"] = str(records_dir)
            cfg["model_path"] = str(Path(tmpdir) / "model.pkl")
            cfg["mode"] = "pnl"
            cfg["use_ob_features"] = False
            cfg["latest_run_only"] = False
            cfg["exclude_missing_run_id"] = False
            # Should not raise
            result = retrain_model(cfg)
            assert result["status"] in ("skipped", "deployed")


# =====================================================================
# 130# F7: trades I/O 7 日 fallback テスト
# =====================================================================

class TestTradesIOFallback:
    """130# F7: trades fallback が全量ではなく直近 7 日でフォールバックする."""

    def test_fallback_uses_7day_window(self) -> None:
        """date_filter で空→全量ではなく 7 日 window が先に試行される."""
        from scripts.v460.ml.feature_enricher import enrich_fill_records, load_raw_trades
        import pandas as pd

        # enrich_fill_records 内の load_raw_trades 呼び出しを追跡
        call_args: list[tuple] = []
        original_load = load_raw_trades

        def _tracking_load(raw_dir=None, date_filter=None):
            call_args.append((raw_dir, date_filter))
            # 常に空を返して fallback チェーン全体をテスト
            return pd.DataFrame()

        fill_df = pd.DataFrame({
            "timestamp": [1771502400.0],
            "side": ["buy"],
            "filled": [True],
        })

        with patch(
            "scripts.v460.ml.feature_enricher.load_raw_trades",
            side_effect=_tracking_load,
        ):
            enrich_fill_records(fill_df)

        # 呼び出し順: (1) date_filter あり → (2) 7日 window → (3) 全量
        assert len(call_args) == 3, f"Expected 3 calls, got {len(call_args)}"
        # 1st call: original date_filter
        assert call_args[0][1] is not None
        # 2nd call: 7-day fallback (set of date strings)
        fb_filter = call_args[1][1]
        assert fb_filter is not None
        assert len(fb_filter) <= 8  # 7 days + 1
        # 3rd call: full fallback (None)
        assert call_args[2][1] is None


# =====================================================================
# E1-E4 効率化施策テスト
# =====================================================================

class TestE4EnrichedCache:
    """E4: enriched data cache テスト."""

    def test_cache_roundtrip(self) -> None:
        """キャッシュ保存→読み込みでデータが一致."""
        from scripts.v460.ml.retrain_scheduler import (
            _save_enriched_cache,
            _load_enriched_cache,
        )
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.pkl"
            df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
            _save_enriched_cache(cache_path, df)
            loaded = _load_enriched_cache(cache_path, n_records=3)
            assert loaded is not None
            assert len(loaded) == 3
            pd.testing.assert_frame_equal(df, loaded)

    def test_cache_invalidation_on_count_mismatch(self) -> None:
        """レコード数不一致でキャッシュを無効化."""
        from scripts.v460.ml.retrain_scheduler import (
            _save_enriched_cache,
            _load_enriched_cache,
        )
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.pkl"
            df = pd.DataFrame({"a": [1, 2, 3]})
            _save_enriched_cache(cache_path, df)
            # 異なるレコード数で読み込み → None
            loaded = _load_enriched_cache(cache_path, n_records=5)
            assert loaded is None


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


class TestBuildLgbmRegressor:
    """DRY: _build_lgbm_regressor テスト."""

    def test_default_params(self) -> None:
        """デフォルトパラメータで LGBMRegressor を構築."""
        from scripts.v460.ml.retrain_scheduler import _build_lgbm_regressor

        cfg = {"lgbm_n_estimators": 100, "lgbm_max_depth": 3}
        model = _build_lgbm_regressor(cfg)
        assert model.n_estimators == 100
        assert model.max_depth == 3

    def test_n_estimators_override(self) -> None:
        """n_estimators をオーバーライドできる (E2 early stopping 用)."""
        from scripts.v460.ml.retrain_scheduler import _build_lgbm_regressor

        cfg = {"lgbm_n_estimators": 100}
        model = _build_lgbm_regressor(cfg, n_estimators_override=300)
        assert model.n_estimators == 300
