from unittest.mock import Mock, patch

import pandas as pd

from ztb.features.multi_timeframe.engine import MultiTimeframeFeatureEngineer
from ztb.features.timeframe import Timeframe


class TestMultiTimeframeFeatureEngineer:
    """Test cases for MultiTimeframeFeatureEngineer."""

    def test_integrate_timeframe_features_uses_concat(self):
        """Test that _integrate_timeframe_features uses pd.concat instead of incremental assignment."""
        # Create mock data
        base_df = pd.DataFrame(
            {"timestamp": pd.date_range("2023-01-01", periods=100, freq="1min")}
        )
        mtf_data = {
            "5m": pd.DataFrame({"feature1": [1] * 100, "feature2": [2] * 100}),
            "15m": pd.DataFrame({"feature3": [3] * 100, "feature4": [4] * 100}),
        }

        # Create instance
        engineer = MultiTimeframeFeatureEngineer()

        # Mock the feature engineers
        with patch(
            "ztb.features.multi_timeframe.engine.SACv427FeatureEngineer"
        ) as mock_fe:
            mock_fe_instance = Mock()
            mock_fe.return_value = mock_fe_instance
            mock_fe_instance.generate_features.side_effect = lambda df, tf: mtf_data[tf]

            # Call the method
            with patch("pandas.concat") as mock_concat:
                mock_concat.return_value = pd.DataFrame()  # Mock return

                result = engineer._integrate_timeframe_features(
                    timeframe_features={
                        Timeframe.M5: mtf_data["5m"],
                        Timeframe.M15: mtf_data["15m"],
                    },
                    base_df=base_df,
                    include_timeframe_indicators=False,
                )

                # Assert pd.concat was called
                mock_concat.assert_called_once()
                # Check that it was called with axis=1
                call_args = mock_concat.call_args
                assert call_args[1]["axis"] == 1

                # Check that dfs_to_concat was built correctly
                dfs_to_concat = call_args[0][0]
                assert len(dfs_to_concat) == 2  # Two timeframes
                assert all(isinstance(df, pd.DataFrame) for df in dfs_to_concat)
