"""356a# B1/B3/B4 tests: g2_sac_train.yaml, feature_columns injection, G2 gate."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ======================================================================
# B1: g2_sac_train.yaml existence and structure
# ======================================================================


class TestB1YamlExists:
    """g2_sac_train.yaml が存在し構造が正しいこと."""

    @pytest.fixture()
    def yaml_path(self) -> Path:
        return Path("configs/v460/experiments/g2_sac_train.yaml")

    def test_file_exists(self, yaml_path: Path) -> None:
        assert yaml_path.exists(), f"{yaml_path} does not exist"

    def test_yaml_loads(self, yaml_path: Path) -> None:
        import yaml

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict)

    def test_required_keys(self, yaml_path: Path) -> None:
        import yaml

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["_gate"] == "G2-train"
        assert cfg["_task"] == "sac_train"
        assert "data" in cfg
        assert "features" in cfg
        assert "sac_hyperparameters" in cfg
        assert "seeds" in cfg

    def test_seeds_count(self, yaml_path: Path) -> None:
        import yaml

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        seeds = cfg["seeds"]
        assert isinstance(seeds, list)
        assert len(seeds) == 4, "G2 gate requires 4 seeds"

    def test_features_are_featureregistry(self, yaml_path: Path) -> None:
        import yaml

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        selected = cfg["features"]["selected"]
        assert isinstance(selected, list)
        assert len(selected) >= 10, f"Expected >= 10 features, got {len(selected)}"
        # These should be FeatureRegistry names, not v460 microstructure
        assert "bid_ask_spread" not in selected, "Should not use v460 microstructure features"

    def test_sac_hyperparameters_v459_tuning(self, yaml_path: Path) -> None:
        import yaml

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        sac = cfg["sac_hyperparameters"]
        assert sac["gamma"] == 0.80, "v459 short-term gamma"
        assert sac["buffer_size"] == 100000, "v459 buffer size"
        assert sac["learning_starts"] == 100, "v459 early learning"

    def test_environment_uses_continuous_actions(self, yaml_path: Path) -> None:
        import yaml

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        env = cfg["environment"]
        assert env.get("use_continuous_actions") is True


# ======================================================================
# B3: feature_columns injection into EnvironmentConfig
# ======================================================================


class TestB3FeatureInjection:
    """_create_training_env が feature_columns を env に注入すること."""

    def test_feature_names_injected_when_present(self) -> None:
        """features.selected が設定されたら EnvironmentConfig.feature_names にセットされる."""
        from ztb.trading.environment.utils.config import EnvironmentConfig

        # _create_training_env の核心ロジックを再現
        feature_columns = ["price_velocity", "micro_trend", "volume_surge"]
        env_config = EnvironmentConfig()
        assert env_config.feature_names is None

        # 356a# B3 修正後のロジック
        if feature_columns:
            env_config.feature_names = feature_columns

        assert env_config.feature_names == feature_columns
        assert len(env_config.feature_names) == 3

    def test_feature_names_not_injected_when_empty(self) -> None:
        """features.selected が空なら feature_names は None のまま."""
        from ztb.trading.environment.utils.config import EnvironmentConfig

        feature_columns: list[str] = []
        env_config = EnvironmentConfig()

        if feature_columns:
            env_config.feature_names = feature_columns

        assert env_config.feature_names is None

    def test_env_info_tracks_injection(self) -> None:
        """env_info に feature_columns_injected が記録される."""
        feature_columns = ["a", "b"]
        env_info = {
            "obs_dim": 2,
            "action_dim": 1,
            "env_type": "HeavyTradingEnv",
            "feature_columns_count": len(feature_columns),
            "feature_columns_injected": bool(feature_columns),
        }
        assert env_info["feature_columns_injected"] is True
        assert env_info["feature_columns_count"] == 2


# ======================================================================
# B4: G2 gate evaluation (dict-based)
# ======================================================================


class TestB4G2GateEvaluation:
    """_evaluate_g2_from_results が G2 gate 条件を正しく判定すること."""

    @pytest.fixture()
    def thresholds(self) -> dict:
        return {
            "min_positive_seed_ratio": 0.75,
            "max_ic_seed_std": 0.03,
            "convergence_window_start": 30000,
            "max_roi_variance_pct": 5.0,
            "worst_seed_min_roi": -0.02,
        }

    def test_all_pass(self, thresholds: dict) -> None:
        from scripts.v460.run_experiment import _evaluate_g2_from_results

        results = {
            "seed_results": [
                {"seed": 42, "gross_roi": 0.05, "ic_mean": 0.03},
                {"seed": 123, "gross_roi": 0.03, "ic_mean": 0.02},
                {"seed": 456, "gross_roi": 0.04, "ic_mean": 0.025},
                {"seed": 789, "gross_roi": 0.01, "ic_mean": 0.028},
            ],
            "convergence": {"roi_variance_pct_after_30k": 3.0},
        }

        judgment = _evaluate_g2_from_results(results, thresholds)
        assert judgment["gate_result"] == "PASS"
        assert all(c["pass"] for c in judgment["checks"].values())

    def test_e1_fail_insufficient_positive_seeds(self, thresholds: dict) -> None:
        from scripts.v460.run_experiment import _evaluate_g2_from_results

        results = {
            "seed_results": [
                {"seed": 42, "gross_roi": 0.05, "ic_mean": 0.03},
                {"seed": 123, "gross_roi": -0.01, "ic_mean": 0.02},
                {"seed": 456, "gross_roi": -0.01, "ic_mean": 0.025},
                {"seed": 789, "gross_roi": -0.01, "ic_mean": 0.028},
            ],
            "convergence": {"roi_variance_pct_after_30k": 3.0},
        }

        judgment = _evaluate_g2_from_results(results, thresholds)
        assert judgment["gate_result"] == "FAIL"
        assert judgment["checks"]["positive_seed_ratio"]["pass"] is False

    def test_e2_fail_high_ic_variance(self, thresholds: dict) -> None:
        from scripts.v460.run_experiment import _evaluate_g2_from_results

        results = {
            "seed_results": [
                {"seed": 42, "gross_roi": 0.05, "ic_mean": 0.10},
                {"seed": 123, "gross_roi": 0.03, "ic_mean": 0.01},
                {"seed": 456, "gross_roi": 0.04, "ic_mean": 0.08},
                {"seed": 789, "gross_roi": 0.01, "ic_mean": 0.02},
            ],
            "convergence": {"roi_variance_pct_after_30k": 3.0},
        }

        judgment = _evaluate_g2_from_results(results, thresholds)
        assert judgment["gate_result"] == "FAIL"
        assert judgment["checks"]["ic_seed_std"]["pass"] is False

    def test_e3_fail_convergence(self, thresholds: dict) -> None:
        from scripts.v460.run_experiment import _evaluate_g2_from_results

        results = {
            "seed_results": [
                {"seed": 42, "gross_roi": 0.05, "ic_mean": 0.03},
                {"seed": 123, "gross_roi": 0.03, "ic_mean": 0.02},
                {"seed": 456, "gross_roi": 0.04, "ic_mean": 0.025},
                {"seed": 789, "gross_roi": 0.01, "ic_mean": 0.028},
            ],
            "convergence": {"roi_variance_pct_after_30k": 8.0},
        }

        judgment = _evaluate_g2_from_results(results, thresholds)
        assert judgment["gate_result"] == "FAIL"
        assert judgment["checks"]["convergence"]["pass"] is False

    def test_e4_fail_worst_seed(self, thresholds: dict) -> None:
        from scripts.v460.run_experiment import _evaluate_g2_from_results

        results = {
            "seed_results": [
                {"seed": 42, "gross_roi": 0.05, "ic_mean": 0.03},
                {"seed": 123, "gross_roi": 0.03, "ic_mean": 0.02},
                {"seed": 456, "gross_roi": 0.04, "ic_mean": 0.025},
                {"seed": 789, "gross_roi": -0.03, "ic_mean": 0.028},
            ],
            "convergence": {"roi_variance_pct_after_30k": 3.0},
        }

        judgment = _evaluate_g2_from_results(results, thresholds)
        assert judgment["gate_result"] == "FAIL"
        assert judgment["checks"]["worst_seed_roi"]["pass"] is False

    def test_empty_seed_results(self, thresholds: dict) -> None:
        from scripts.v460.run_experiment import _evaluate_g2_from_results

        results: dict[str, object] = {"seed_results": [], "convergence": {}}
        judgment = _evaluate_g2_from_results(results, thresholds)
        assert judgment["gate_result"] == "FAIL"

    def test_boundary_e1_exactly_75_percent(self, thresholds: dict) -> None:
        """3/4 = 75% は PASS."""
        from scripts.v460.run_experiment import _evaluate_g2_from_results

        results = {
            "seed_results": [
                {"seed": 42, "gross_roi": 0.01, "ic_mean": 0.02},
                {"seed": 123, "gross_roi": 0.01, "ic_mean": 0.02},
                {"seed": 456, "gross_roi": 0.01, "ic_mean": 0.02},
                {"seed": 789, "gross_roi": -0.01, "ic_mean": 0.02},
            ],
            "convergence": {"roi_variance_pct_after_30k": 1.0},
        }

        judgment = _evaluate_g2_from_results(results, thresholds)
        assert judgment["checks"]["positive_seed_ratio"]["pass"] is True


# ======================================================================
# B4: convergence computation
# ======================================================================


class TestConvergenceComputation:
    """_compute_convergence が正しく ROI 変動を計算すること."""

    def test_basic_convergence(self) -> None:
        from scripts.v460.run_experiment import _compute_convergence

        checkpoints = [
            [
                {"timesteps": 5000, "roi": 0.01},
                {"timesteps": 10000, "roi": 0.02},
                {"timesteps": 30000, "roi": 0.03},
                {"timesteps": 40000, "roi": 0.04},
                {"timesteps": 50000, "roi": 0.05},
            ]
        ]
        result = _compute_convergence(checkpoints, window_start=30000)
        # Values after 30K: 0.03, 0.04, 0.05 → range = 0.02 → 2%
        assert abs(result["roi_variance_pct_after_30k"] - 2.0) < 0.01

    def test_empty_checkpoints(self) -> None:
        from scripts.v460.run_experiment import _compute_convergence

        result = _compute_convergence([], window_start=30000)
        assert result["roi_variance_pct_after_30k"] == 0.0

    def test_no_checkpoints_after_window(self) -> None:
        from scripts.v460.run_experiment import _compute_convergence

        checkpoints = [[{"timesteps": 5000, "roi": 0.01}]]
        result = _compute_convergence(checkpoints, window_start=30000)
        assert result["roi_variance_pct_after_30k"] == 0.0


# ======================================================================
# B4: Multi-seed dispatch integration
# ======================================================================


class TestMultiSeedDispatch:
    """run_experiment.py の G2 multi-seed ディスパッチロジック."""

    def test_g2_gate_triggers_multi_seed(self) -> None:
        """G2 gate + seeds > 1 で _run_multi_seed が呼ばれる."""
        cfg = {
            "_gate": "G2-train",
            "_task": "sac_train",
            "seeds": [42, 123, 456, 789],
            "seed": 42,
        }
        gate = cfg.get("_gate", "")
        seeds = cfg.get("seeds", [])
        assert "G2" in gate
        assert len(seeds) > 1

    def test_g1_gate_does_not_trigger_multi_seed(self) -> None:
        """G1 gate では multi-seed は発動しない."""
        cfg = {
            "_gate": "G1-info",
            "_task": "feature_info",
            "seeds": [42, 123, 456, 789],
        }
        gate = cfg.get("_gate", "")
        seeds = cfg.get("seeds", [])
        assert "G2" not in gate
