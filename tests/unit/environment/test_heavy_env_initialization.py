# mypy: disable-error-code="untyped-decorator"

import pandas as pd
import pytest
from gymnasium import spaces
from unittest.mock import patch

from tests.helpers import (
    make_schema_feature_env_config,
    make_stub_multi_timeframe_features,
)
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv


@pytest.fixture(scope="module")
def base_ohlcv_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=48, freq="1min"),
            "open": [100.0] * 48,
            "high": [101.0] * 48,
            "low": [99.0] * 48,
            "close": [100.5] * 48,
            "volume": [1000.0] * 48,
        }
    )


@pytest.fixture(scope="module")
def schema_feature_env(base_ohlcv_df: pd.DataFrame) -> HeavyTradingEnv:
    env = HeavyTradingEnv(
        df=base_ohlcv_df.copy(),
        config=make_schema_feature_env_config(base_ohlcv_df),
    )
    yield env
    env.close()


@pytest.fixture(scope="module")
def mtf_env(base_ohlcv_df: pd.DataFrame) -> HeavyTradingEnv:
    mtf_stub = make_stub_multi_timeframe_features(base_ohlcv_df, columns=2)
    with patch(
        "ztb.features.multi_timeframe.MultiTimeframeFeatureSystem.process_multi_timeframe_data",
        autospec=True,
        return_value=mtf_stub,
    ):
        env = HeavyTradingEnv(
            df=base_ohlcv_df.copy(),
            config=make_schema_feature_env_config(
                base_ohlcv_df,
                include_feature_names=False,
                feature_set="full",
                correlation_reduction=False,
            ),
        )
    yield env
    env.close()


class TestHeavyTradingEnvInitialization:
    """Test cases for HeavyTradingEnv initialization."""

    def test_protocol_implementation(self) -> None:
        """Test that HeavyTradingEnv exposes the TradingEnvironment surface."""
        assert hasattr(HeavyTradingEnv, 'reset')
        assert hasattr(HeavyTradingEnv, 'step')
        assert hasattr(HeavyTradingEnv, 'render')
        assert hasattr(HeavyTradingEnv, 'close')
        assert hasattr(HeavyTradingEnv, "get_legal_actions")

    def test_observation_space_dimension_consistency(
        self, schema_feature_env: HeavyTradingEnv
    ) -> None:
        """Test that observation space dimension matches feature matrix columns and features list length."""
        env = schema_feature_env

        # Check consistency
        obs_dim = env.observation_space.shape[0]
        feature_matrix_cols = env._feature_matrix.shape[1]
        features_len = len(env.features)

        # 379# P3-A: env_tracker adds internal features (inventory_pressure,
        # loss_risk, time_in_market) to observation_space beyond feature_matrix cols
        n_internal = 0
        if getattr(env, "env_tracker", None) is not None:
            n_internal = len(env.env_tracker.get_feature_vector())

        assert (
            obs_dim == feature_matrix_cols + n_internal
        ), f"Observation space dim {obs_dim} != feature matrix cols {feature_matrix_cols} + internal {n_internal}"
        assert (
            obs_dim == features_len + n_internal
        ), f"Observation space dim {obs_dim} != features len {features_len} + internal {n_internal}"
        assert (
            feature_matrix_cols == features_len
        ), f"Feature matrix cols {feature_matrix_cols} != features len {features_len}"

    def test_multi_timeframe_features_merged_into_df(
        self, mtf_env: HeavyTradingEnv
    ) -> None:
        """Test that multi-timeframe features are properly merged into self.df."""
        env = mtf_env

        for column in ("mtf_stub_0", "mtf_stub_1"):
            assert column in env.df.columns

    def test_schema_scaler_skips_full_data_scaler_computation(self) -> None:
        """Schema-provided scaler should bypass full feature-matrix scan."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=64, freq="1min"),
                "open": [100.0] * 64,
                "high": [101.0] * 64,
                "low": [99.0] * 64,
                "close": [100.5] * 64,
                "volume": [1000.0] * 64,
            }
        )
        feature_names = ["open", "high", "low", "close", "volume"]
        scaler_mean = [100.0, 101.0, 99.0, 100.5, 1000.0]
        scaler_std = [1.0, 1.0, 1.0, 1.0, 10.0]

        with patch.object(
            HeavyTradingEnv,
            "_compute_scaler_from_data",
            autospec=True,
        ) as mock_compute_scaler:
            env = HeavyTradingEnv(
                df=df,
                config={
                    "feature_names": feature_names,
                    "scaler_mean": scaler_mean,
                    "scaler_std": scaler_std,
                },
            )

        mock_compute_scaler.assert_not_called()
        assert env.scaler_mean is not None
        assert env.scaler_std is not None
        assert env.scaler_mean.tolist() == scaler_mean
        assert env.scaler_std.tolist() == scaler_std

    def test_discrete_action_space_when_continuous_actions_disabled(
        self, base_ohlcv_df: pd.DataFrame
    ) -> None:
        """PPO/discrete mode should expose Discrete(3)."""
        env = HeavyTradingEnv(
            df=base_ohlcv_df.copy(),
            config=make_schema_feature_env_config(
                base_ohlcv_df,
                use_continuous_actions=False,
            ),
        )
        try:
            assert isinstance(env.action_space, spaces.Discrete)
            assert env.action_space.n == 3
        finally:
            env.close()

    def test_discrete_action_space_when_action_space_type_is_discrete(
        self, base_ohlcv_df: pd.DataFrame
    ) -> None:
        """Legacy discrete selector should still force PPO-compatible actions."""
        env = HeavyTradingEnv(
            df=base_ohlcv_df.copy(),
            config=make_schema_feature_env_config(
                base_ohlcv_df,
                action_space_type="discrete",
            ),
        )
        try:
            assert isinstance(env.action_space, spaces.Discrete)
            assert env.action_space.n == 3
        finally:
            env.close()

    def test_continuous_action_space_when_enabled(
        self, base_ohlcv_df: pd.DataFrame
    ) -> None:
        """SAC/continuous mode should expose the Box action space."""
        env = HeavyTradingEnv(
            df=base_ohlcv_df.copy(),
            config=make_schema_feature_env_config(
                base_ohlcv_df,
                use_continuous_actions=True,
            ),
        )
        try:
            assert isinstance(env.action_space, spaces.Box)
            assert env.action_space.shape == (1,)
        finally:
            env.close()
