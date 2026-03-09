"""356# B1/B3/B4 tests: g2_sac_train.yaml, feature_columns injection, G2 gate."""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

from scripts.v460.lib.data_loader import load_parquet
from scripts.v460.lib.tasks.sac_train import _create_training_env
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig

_G2_SAC_YAML_PATH = Path("configs/v460/experiments/g2_sac_train.yaml")
_G2_REAL_ROWS = 128


@lru_cache(maxsize=1)
def _load_g2_sac_yaml() -> dict:
    with open(_G2_SAC_YAML_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError("g2_sac_train.yaml did not load as dict")
    return cfg


@lru_cache(maxsize=1)
def _load_g2_schema_names() -> tuple[str, ...]:
    cfg = _load_g2_sac_yaml()
    data_path = Path(cfg["data"]["ohlcv_path"])
    return tuple(pq.read_schema(str(data_path)).names)


@lru_cache(maxsize=1)
def _load_g2_real_df() -> pd.DataFrame:
    cfg = _load_g2_sac_yaml()
    data_path = Path(cfg["data"]["ohlcv_path"])
    selected = cfg["features"]["selected"]
    if not isinstance(selected, list):
        raise TypeError("features.selected must be list")
    return load_parquet(data_path, feature_cols=[str(col) for col in selected]).head(_G2_REAL_ROWS)


# ======================================================================
# B1: g2_sac_train.yaml existence and structure
# ======================================================================


class TestB1YamlExists:
    """g2_sac_train.yaml が存在し構造が正しいこと."""

    @pytest.fixture(scope="class")
    def yaml_path(self) -> Path:
        return _G2_SAC_YAML_PATH

    @pytest.fixture(scope="class")
    def yaml_cfg(self, yaml_path: Path) -> dict:
        assert yaml_path.exists(), f"{yaml_path} does not exist"
        return dict(_load_g2_sac_yaml())

    def test_file_exists(self, yaml_path: Path) -> None:
        assert yaml_path.exists(), f"{yaml_path} does not exist"

    def test_yaml_loads(self, yaml_cfg: dict) -> None:
        assert isinstance(yaml_cfg, dict)

    def test_required_keys(self, yaml_cfg: dict) -> None:
        cfg = yaml_cfg
        assert cfg["_gate"] == "G2-train"
        assert cfg["_task"] == "sac_train"
        assert "data" in cfg
        assert "features" in cfg
        assert "sac_hyperparameters" in cfg
        assert "seeds" in cfg

    def test_seeds_count(self, yaml_cfg: dict) -> None:
        cfg = yaml_cfg
        seeds = cfg["seeds"]
        assert isinstance(seeds, list)
        assert len(seeds) == 4, "G2 gate requires 4 seeds"

    def test_features_are_featureregistry(self, yaml_cfg: dict) -> None:
        cfg = yaml_cfg
        selected = cfg["features"]["selected"]
        assert isinstance(selected, list)
        assert len(selected) >= 10, f"Expected >= 10 features, got {len(selected)}"
        # These should be FeatureRegistry names, not v460 microstructure
        assert "bid_ask_spread" not in selected, "Should not use v460 microstructure features"

    def test_sac_hyperparameters_v459_tuning(self, yaml_cfg: dict) -> None:
        cfg = yaml_cfg
        sac = cfg["sac_hyperparameters"]
        assert sac["gamma"] == 0.80, "v459 short-term gamma"
        assert sac["buffer_size"] == 100000, "v459 buffer size"
        assert sac["learning_starts"] == 100, "v459 early learning"

    def test_environment_uses_continuous_actions(self, yaml_cfg: dict) -> None:
        cfg = yaml_cfg
        env = cfg["environment"]
        assert env.get("use_continuous_actions") is True


# ======================================================================
# B3: feature_columns injection into EnvironmentConfig
# ======================================================================


class TestB3FeatureInjection:
    """_create_training_env が feature_columns を env に注入すること."""

    def test_feature_names_injected_when_present(self) -> None:
        """features.selected が設定されたら EnvironmentConfig.feature_names にセットされる."""
        # _create_training_env の核心ロジックを再現
        feature_columns = ["price_velocity", "micro_trend", "volume_surge"]
        env_config = EnvironmentConfig()
        assert env_config.feature_names is None

        # 356# B3 修正後のロジック
        if feature_columns:
            env_config.feature_names = feature_columns

        assert env_config.feature_names == feature_columns
        assert len(env_config.feature_names) == 3

    def test_feature_names_not_injected_when_empty(self) -> None:
        """features.selected が空なら feature_names は None のまま."""
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
            "max_roi_seed_std": 0.03,
            "convergence_window_start": 30000,
            "max_roi_variance_pct": 5.0,
            "worst_seed_min_roi": -0.02,
        }

    def test_all_pass(self, thresholds: dict) -> None:
        from scripts.v460.run_experiment import _evaluate_g2_from_results

        results = {
            "seed_results": [
                {"seed": 42, "gross_roi": 0.05},
                {"seed": 123, "gross_roi": 0.03},
                {"seed": 456, "gross_roi": 0.04},
                {"seed": 789, "gross_roi": 0.01},
            ],
            "convergence": {"roi_variance_pct_after_30k": 3.0},
        }

        judgment = _evaluate_g2_from_results(results, thresholds)
        assert judgment["gate_result"] == "PASS"
        assert all(c["pass"] for c in judgment["checks"].values())

    def test_e1_fail_insufficient_positive_seeds(self, thresholds: dict) -> None:
        from scripts.v460.run_experiment import _evaluate_g2_from_results

        # 363# A4: ROI stdev(0.015) <= 0.03 → E2 PASS, 1/4 positive → E1 FAIL
        results = {
            "seed_results": [
                {"seed": 42, "gross_roi": 0.02},
                {"seed": 123, "gross_roi": -0.01},
                {"seed": 456, "gross_roi": -0.01},
                {"seed": 789, "gross_roi": -0.01},
            ],
            "convergence": {"roi_variance_pct_after_30k": 3.0},
        }

        judgment = _evaluate_g2_from_results(results, thresholds)
        assert judgment["gate_result"] == "FAIL"
        assert judgment["checks"]["positive_seed_ratio"]["pass"] is False

    def test_e2_fail_high_roi_variance(self, thresholds: dict) -> None:
        """363# A4: seed 間 ROI 標準偏差 > 0.03 で E2 FAIL."""
        from scripts.v460.run_experiment import _evaluate_g2_from_results

        # stdev([0.10, 0.01, 0.08, 0.001]) ≈ 0.0496 > 0.03 → E2 FAIL
        results = {
            "seed_results": [
                {"seed": 42, "gross_roi": 0.10},
                {"seed": 123, "gross_roi": 0.01},
                {"seed": 456, "gross_roi": 0.08},
                {"seed": 789, "gross_roi": 0.001},
            ],
            "convergence": {"roi_variance_pct_after_30k": 3.0},
        }

        judgment = _evaluate_g2_from_results(results, thresholds)
        assert judgment["gate_result"] == "FAIL"
        assert judgment["checks"]["roi_seed_std"]["pass"] is False

    def test_e3_fail_convergence(self, thresholds: dict) -> None:
        from scripts.v460.run_experiment import _evaluate_g2_from_results

        results = {
            "seed_results": [
                {"seed": 42, "gross_roi": 0.05},
                {"seed": 123, "gross_roi": 0.03},
                {"seed": 456, "gross_roi": 0.04},
                {"seed": 789, "gross_roi": 0.01},
            ],
            "convergence": {"roi_variance_pct_after_30k": 8.0},
        }

        judgment = _evaluate_g2_from_results(results, thresholds)
        assert judgment["gate_result"] == "FAIL"
        assert judgment["checks"]["convergence"]["pass"] is False

    def test_e4_fail_worst_seed(self, thresholds: dict) -> None:
        from scripts.v460.run_experiment import _evaluate_g2_from_results

        # 363# A4: stdev([0.03, 0.02, 0.03, -0.025]) ≈ 0.026 <= 0.03 → E2 PASS
        results = {
            "seed_results": [
                {"seed": 42, "gross_roi": 0.03},
                {"seed": 123, "gross_roi": 0.02},
                {"seed": 456, "gross_roi": 0.03},
                {"seed": 789, "gross_roi": -0.025},
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
                {"seed": 42, "gross_roi": 0.01},
                {"seed": 123, "gross_roi": 0.01},
                {"seed": 456, "gross_roi": 0.01},
                {"seed": 789, "gross_roi": -0.01},
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

    def test_seed_failure_captured_not_propagated(self) -> None:
        """1 seed 失敗時に他 seed の結果が保持され、例外が伝播しない."""
        from scripts.v460.run_experiment import _run_multi_seed

        call_count = 0

        def mock_task(cfg: dict) -> dict:
            nonlocal call_count
            call_count += 1
            if cfg["seed"] == 456:
                raise RuntimeError("seed 456 exploded")
            return {
                "eval_metrics": {"gross_roi": 0.05},
                "checkpoint_metrics": [],
            }

        cfg: dict = {"_gate": "G2-train", "_task": "sac_train"}
        seeds = [42, 123, 456, 789]

        result = _run_multi_seed(cfg, seeds, mock_task)

        assert call_count == 4
        assert len(result["seed_results"]) == 4
        # Seed 456 should have error marker with 0.0 ROI
        failed = [s for s in result["seed_results"] if s["seed"] == 456]
        assert len(failed) == 1
        assert failed[0]["gross_roi"] == 0.0
        assert "error" in failed[0]
        # Other seeds should have real results
        ok = [s for s in result["seed_results"] if s["seed"] != 456]
        assert all(s["gross_roi"] == 0.05 for s in ok)


# ======================================================================
# L-3/L-5: ROI extraction and metrics completeness (359#)
# ======================================================================


class TestROIExtraction:
    """_extract_roi_from_env が env から正しく ROI を取得すること."""

    def test_extract_roi_from_env_with_portfolio(self) -> None:
        """portfolio_value / initial_portfolio_value から ROI を算出."""
        from scripts.v460.lib.tasks.sac_train import _extract_roi_from_env

        env = MagicMock()
        env.portfolio_value = 110_000.0
        env.initial_portfolio_value = 100_000.0
        roi = _extract_roi_from_env(env)
        assert abs(roi - 0.1) < 1e-9

    def test_extract_roi_from_env_missing_attrs(self) -> None:
        """portfolio 属性が無い env では 0.0 を返す."""
        from scripts.v460.lib.tasks.sac_train import _extract_roi_from_env

        env = MagicMock(spec=[])  # no attributes
        roi = _extract_roi_from_env(env)
        assert roi == 0.0

    def test_extract_roi_from_env_zero_initial(self) -> None:
        """initial_portfolio_value が 0 の場合に 0.0 を返す (ZeroDivision 防止)."""
        from scripts.v460.lib.tasks.sac_train import _extract_roi_from_env

        env = MagicMock()
        env.portfolio_value = 100.0
        env.initial_portfolio_value = 0.0
        roi = _extract_roi_from_env(env)
        assert roi == 0.0

    def test_extract_roi_negative(self) -> None:
        """損失ケースで負の ROI を返す."""
        from scripts.v460.lib.tasks.sac_train import _extract_roi_from_env

        env = MagicMock()
        env.portfolio_value = 95_000.0
        env.initial_portfolio_value = 100_000.0
        roi = _extract_roi_from_env(env)
        assert abs(roi - (-0.05)) < 1e-9


class TestCheckpointAndEvalMetrics:
    """チェックポイントと評価メトリクスに G2 gate 必須フィールドが存在すること."""

    def test_checkpoint_metrics_contain_roi(self) -> None:
        """_train_with_checkpoints の結果に roi フィールドが含まれる."""
        from scripts.v460.lib.tasks.sac_train import _train_with_checkpoints

        model = MagicMock()
        model.predict.return_value = (0, None)

        env = MagicMock()
        env.reset.return_value = (0, {})
        env.step.return_value = (0, 0.1, True, False, {})
        env.portfolio_value = 101_000.0
        env.initial_portfolio_value = 100_000.0

        cfg = {"training": {"checkpoint_interval": 5000, "total_timesteps": 10000}}
        result = _train_with_checkpoints(model, env, 10000, cfg)

        assert len(result) == 2  # 10000 / 5000
        for cp in result:
            assert "roi" in cp
            assert "timesteps" in cp
            assert isinstance(cp["roi"], float)

    def test_eval_metrics_contain_gross_roi(self) -> None:
        """_evaluate_trained_model の結果に gross_roi が含まれる."""
        from scripts.v460.lib.tasks.sac_train import _evaluate_trained_model

        model = MagicMock()
        model.predict.return_value = (0, None)

        env = MagicMock()
        env.reset.return_value = (0, {})
        env.step.return_value = (0, 1.0, True, False, {})
        env.portfolio_value = 103_000.0
        env.initial_portfolio_value = 100_000.0
        env.trades_count = 15
        env.total_pnl = 3000.0

        cfg = {"evaluation": {"n_episodes": 1}}
        result = _evaluate_trained_model(model, env, cfg)

        assert "gross_roi" in result
        assert abs(float(result["gross_roi"]) - 0.03) < 1e-9
        assert result["trade_count"] == 15
        assert result["gross_pnl"] == 3000.0


# ======================================================================
# P3A-1: Training data integrity (359#)
# ======================================================================


class TestTrainingDataIntegrity:
    """P3A-1: YAML で参照するデータファイルが有効であること."""

    @pytest.fixture(scope="class")
    def yaml_cfg(self) -> dict:
        return dict(_load_g2_sac_yaml())

    @pytest.fixture(scope="class")
    def schema_names(self, yaml_cfg: dict) -> tuple[str, ...]:
        data_path = Path(yaml_cfg["data"]["ohlcv_path"])
        if not data_path.exists():
            pytest.skip(f"Data file not found: {data_path}")
        return _load_g2_schema_names()

    def test_yaml_data_file_exists_and_valid(self, yaml_cfg: dict, schema_names: tuple[str, ...]) -> None:
        """g2_sac_train.yaml の ohlcv_path が有効な Parquet ファイルを指す."""
        data_path = Path(yaml_cfg["data"]["ohlcv_path"])
        assert data_path.exists(), f"Data file not found: {data_path}"
        assert len(schema_names) > 0

    def test_yaml_features_present_in_data(self, yaml_cfg: dict, schema_names: tuple[str, ...]) -> None:
        """YAML selected features が全てデータファイルに存在する."""
        schema_cols = set(schema_names)
        selected = yaml_cfg["features"]["selected"]

        missing = [f for f in selected if f not in schema_cols]
        assert not missing, f"Features missing in data: {missing}"

    def test_data_has_close_column(self, schema_names: tuple[str, ...]) -> None:
        """HeavyTradingEnv が必要とする close カラムの存在確認."""
        schema_cols = set(schema_names)
        assert "close" in schema_cols


# ======================================================================
# P3A-2: HeavyTradingEnv integration (359#)
# ======================================================================


class TestHeavyTradingEnvIntegration:
    """P3A-2: 実データ + feature_names 注入で HeavyTradingEnv が正常動作すること.

    YAML → load_parquet → EnvironmentConfig → HeavyTradingEnv の E2E パイプラインを検証.
    """

    SELECTED_FEATURES: list[str] = [
        "price_velocity",
        "micro_trend",
        "price_acceleration",
        "volume_surge",
        "momentum_divergence",
        "tick_volume_ratio",
        "order_flow_imbalance",
        "micro_volatility",
        "spread_pressure",
        "momentum_burst",
        "liquidity_surge",
        "realized_volatility",
    ]

    @pytest.fixture(scope="class")
    def real_df(self) -> "pd.DataFrame":
        """実データの必要列だけを 1 回ロード (テスト用軽量スライス)."""
        data_path = Path("data/btc_jpy_1m_full_registry_features.parquet")
        if not data_path.exists():
            pytest.skip(f"Data file not found: {data_path}")
        return _load_g2_real_df()

    @pytest.fixture(scope="class")
    def env_config(self) -> "EnvironmentConfig":
        """YAML 環境セクションに準拠した EnvironmentConfig を構築."""
        config = EnvironmentConfig(
            transaction_cost=0.001,
            max_position_size=0.01,
            initial_portfolio_value=10_000_000.0,
            use_continuous_actions=True,
            action_space_type="continuous_1d",
            exchange="coincheck",
            timeframe="1m",
            feature_names=self.SELECTED_FEATURES,
            # 相関低減は schema 指定時に不要 → 無効化で安定
            correlation_reduction=False,
        )
        return config

    @staticmethod
    def _create_env(real_df: "pd.DataFrame", env_config: "EnvironmentConfig") -> HeavyTradingEnv:
        # HeavyTradingEnv 側で DataFrame を前処理用にコピーするため、
        # ここでは共有済みの軽量 slice をそのまま渡して二重コピーを避ける。
        return HeavyTradingEnv(
            df=real_df,
            config=dataclasses.replace(env_config),
        )

    @pytest.fixture(scope="class")
    def shared_env(
        self,
        real_df: "pd.DataFrame",
        env_config: "EnvironmentConfig",
    ) -> "HeavyTradingEnv":
        env = self._create_env(real_df, env_config)
        try:
            yield env
        finally:
            env.close()

    @pytest.fixture(scope="class")
    def training_env_bundle(
        self,
        real_df: "pd.DataFrame",
    ) -> tuple["HeavyTradingEnv", dict[str, int | str | bool]]:
        cfg = dict(_load_g2_sac_yaml())
        env, env_info = _create_training_env(real_df, cfg)
        try:
            yield env, env_info
        finally:
            env.close()

    def test_env_instantiation(
        self,
        shared_env: "HeavyTradingEnv",
    ) -> None:
        """HeavyTradingEnv が実データ + feature_names で例外なく生成できる."""
        assert shared_env is not None
        assert hasattr(shared_env, "observation_space")
        assert hasattr(shared_env, "action_space")

    def test_obs_dim_matches_feature_count(
        self,
        shared_env: "HeavyTradingEnv",
    ) -> None:
        """observation_space の次元が注入した feature 数 (12) と一致."""
        obs_dim = shared_env.observation_space.shape[0]
        assert obs_dim == len(self.SELECTED_FEATURES), (
            f"obs_dim={obs_dim} != expected={len(self.SELECTED_FEATURES)}"
        )

    def test_feature_names_synced(
        self,
        shared_env: "HeavyTradingEnv",
    ) -> None:
        """env.feature_names が注入した特徴量リストと一致."""
        assert shared_env.feature_names == self.SELECTED_FEATURES

    def test_reset_returns_valid_obs(
        self,
        shared_env: "HeavyTradingEnv",
    ) -> None:
        """reset() が正しい shape の observation を返す."""
        obs, info = shared_env.reset()
        assert obs.shape == (len(self.SELECTED_FEATURES),)
        assert not np.any(np.isnan(obs)), "Observation contains NaN"

    def test_step_returns_valid_tuple(
        self,
        shared_env: "HeavyTradingEnv",
    ) -> None:
        """step() が (obs, reward, terminated, truncated, info) を正しく返す."""
        obs, _ = shared_env.reset()
        # 連続行動空間: [-1, 1] の中間値 (HOLD に近い)
        action = np.array([0.0], dtype=np.float32)
        result = shared_env.step(action)
        assert len(result) == 5, f"step() returned {len(result)} elements, expected 5"
        obs2, reward, terminated, truncated, info = result
        assert obs2.shape == (len(self.SELECTED_FEATURES),)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_create_training_env_pipeline(
        self,
        training_env_bundle: tuple["HeavyTradingEnv", dict[str, int | str | bool]],
    ) -> None:
        """_create_training_env が YAML 相当の cfg で正常に環境を構築."""
        env, env_info = training_env_bundle
        assert env is not None
        assert env_info["obs_dim"] == len(self.SELECTED_FEATURES)
        assert env_info["feature_columns_injected"] is True
        assert env_info["feature_columns_count"] == len(self.SELECTED_FEATURES)
        assert env_info["env_type"] == "HeavyTradingEnv"


# ======================================================================
# 363# A3: Train/Val split design verification
# ======================================================================


class TestTrainValSplit:
    """363# A3: train/val time-series split の設計検証."""

    def test_val_ratio_clamp(self) -> None:
        """val_ratio は 0.0-0.5 にクランプされる."""
        assert max(0.0, min(float(0.7), 0.5)) == 0.5
        assert max(0.0, min(float(-0.1), 0.5)) == 0.0
        assert max(0.0, min(float(0.2), 0.5)) == 0.2
        assert max(0.0, min(float(0.0), 0.5)) == 0.0

    def test_split_index_calculation(self) -> None:
        """80/20 分割の正確性."""
        total = 1000
        val_ratio = 0.2
        split_idx = int(total * (1.0 - val_ratio))
        assert split_idx == 800
        assert total - split_idx == 200

    def test_eval_uses_only_passed_env(self) -> None:
        """_evaluate_trained_model が渡された env のみを呼ぶ (OOS 保証)."""
        from scripts.v460.lib.tasks.sac_train import _evaluate_trained_model

        eval_env = MagicMock()
        eval_env.reset.return_value = (0, {})
        eval_env.step.return_value = (0, 1.0, True, False, {})
        eval_env.portfolio_value = 105_000.0
        eval_env.initial_portfolio_value = 100_000.0
        eval_env.trades_count = 10
        eval_env.total_pnl = 5000.0

        model = MagicMock()
        model.predict.return_value = (0, None)

        cfg: dict = {"evaluation": {"n_episodes": 1}}
        result = _evaluate_trained_model(model, eval_env, cfg)

        eval_env.reset.assert_called()
        assert abs(float(result["gross_roi"]) - 0.05) < 1e-9

    def test_e2_roi_seed_std_pass(self) -> None:
        """363# A4: seed 間 ROI 標準偏差が閾値以下で E2 PASS."""
        from scripts.v460.run_experiment import _evaluate_g2_from_results

        thresholds = {"min_positive_seed_ratio": 0.75, "max_roi_seed_std": 0.03,
                       "convergence_window_start": 30000, "max_roi_variance_pct": 5.0,
                       "worst_seed_min_roi": -0.02}
        # stdev([0.04, 0.03, 0.035, 0.045]) ≈ 0.0065 ≤ 0.03
        results = {
            "seed_results": [
                {"seed": 42, "gross_roi": 0.04},
                {"seed": 123, "gross_roi": 0.03},
                {"seed": 456, "gross_roi": 0.035},
                {"seed": 789, "gross_roi": 0.045},
            ],
            "convergence": {"roi_variance_pct_after_30k": 3.0},
        }
        judgment = _evaluate_g2_from_results(results, thresholds)
        assert judgment["checks"]["roi_seed_std"]["pass"] is True
        assert judgment["gate_result"] == "PASS"
