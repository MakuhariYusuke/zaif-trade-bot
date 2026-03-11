import pandas as pd

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv


class TestHeavyTradingEnvInitialization:
    """Test cases for HeavyTradingEnv initialization."""

    def test_protocol_implementation(self):
        """Test that HeavyTradingEnv exposes the TradingEnvironment surface."""
        assert hasattr(HeavyTradingEnv, 'reset')
        assert hasattr(HeavyTradingEnv, 'step')
        assert hasattr(HeavyTradingEnv, 'render')
        assert hasattr(HeavyTradingEnv, 'close')
        assert hasattr(HeavyTradingEnv, "get_legal_actions")

    def test_observation_space_dimension_consistency(self):
        """Test that observation space dimension matches feature matrix columns and features list length."""
        # Create sample data
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="1min"),
                "open": [100] * 100,
                "high": [101] * 100,
                "low": [99] * 100,
                "close": [100.5] * 100,
                "volume": [1000] * 100,
            }
        )

        # Config with multi-timeframe features
        config = {
            "feature_set": "full",
            "multi_timeframe": {"enabled": True, "timeframes": ["5m", "15m"]},
        }

        # Initialize environment
        env = HeavyTradingEnv(df=df, config=config)

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

    def test_multi_timeframe_features_merged_into_df(self):
        """Test that multi-timeframe features are properly merged into self.df."""
        # Create sample data
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="1min"),
                "open": [100] * 100,
                "high": [101] * 100,
                "low": [99] * 100,
                "close": [100.5] * 100,
                "volume": [1000] * 100,
            }
        )

        config = {
            "feature_set": "full",
            "multi_timeframe": {"enabled": True, "timeframes": ["5m"]},
        }

        env = HeavyTradingEnv(df=df, config=config)

        # Check that df has additional columns from multi-timeframe features
        original_cols = 5  # timestamp, open, high, low, close, volume
        assert (
            len(env.df.columns) > original_cols
        ), "Multi-timeframe features should be merged into df"
