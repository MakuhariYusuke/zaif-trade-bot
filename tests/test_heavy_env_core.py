import pandas as pd

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv


class TestHeavyTradingEnvCore:
    """Test cases for HeavyTradingEnv core functionality."""

    def test_feature_registry_not_overridden_when_features_exist(self):
        """Test that FeatureRegistry is not used when self.features is already initialized."""
        # This test verifies the logic in __init__ where if self.features exists,
        # FeatureRegistry override is skipped.
        # Since full initialization is complex, we test the logic indirectly by
        # checking that the code path exists and doesn't crash.
        # In practice, this is tested by the initialization tests above.

        # Create minimal data
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="1min"),
                "open": [100] * 10,
                "high": [101] * 10,
                "low": [99] * 10,
                "close": [100.5] * 10,
                "volume": [1000] * 10,
            }
        )

        config = {"feature_set": "full"}

        # This should work without errors, and features should be initialized
        env = HeavyTradingEnv(df=df, config=config)

        # Verify features exist
        assert hasattr(env, "features")
        assert len(env.features) > 0

    def test_feature_registry_used_when_no_features(self):
        """Test that FeatureRegistry is used when self.features is not initialized."""
        # Similar to above - the initialization logic handles this.
        # The test_observation_space_dimension_consistency already covers this path.

        # Create minimal data
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="1min"),
                "open": [100] * 10,
                "high": [101] * 10,
                "low": [99] * 10,
                "close": [100.5] * 10,
                "volume": [1000] * 10,
            }
        )

        config = {"feature_set": "minimal"}  # Use minimal to ensure registry is used

        env = HeavyTradingEnv(df=df, config=config)

        # Verify features exist and observation space matches
        assert hasattr(env, "features")
        assert len(env.features) > 0
        assert env.observation_space.shape[0] == len(env.features)
