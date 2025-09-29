"""
Auto-generated feature tests for AutoFeatureGenerator validation.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ztb.evaluation.auto_feature_generator import AutoFeatureGenerator


class BaseFeatureTest:
    """Base test class with common fixtures and utilities"""

    @pytest.fixture
    def sample_ohlc_data(self):
        """Create sample OHLC data for testing"""
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        np.random.seed(42)

        # Create realistic price data with trends
        base_price = 100
        trend = np.linspace(0, 20, 200)  # Upward trend
        noise = np.random.randn(200) * 2
        prices = base_price + trend + noise

        return pd.DataFrame(
            {
                "open": prices + np.random.randn(200) * 0.5,
                "high": prices + np.abs(np.random.randn(200)) * 1.5,
                "low": prices - np.abs(np.random.randn(200)) * 1.5,
                "close": prices,
                "volume": np.random.randint(1000, 10000, 200),
            },
            index=dates,
        )

    @pytest.fixture
    def auto_generator(self):
        """Create AutoFeatureGenerator instance"""
        return AutoFeatureGenerator()

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger"""
        return MagicMock()

    def assert_feature_data_structure(self, features: dict):
        """Assert that feature data has correct structure"""
        assert isinstance(features, dict)
        for name, data in features.items():
            assert isinstance(name, str)
            assert isinstance(data, pd.DataFrame)
            assert len(data) > 0
            assert not data.empty

    def assert_feature_names_follow_convention(
        self, feature_names: list, prefix: str = "auto_"
    ):
        """Assert that feature names follow naming convention"""
        for name in feature_names:
            assert name.startswith(prefix), (
                f"Feature name {name} doesn't start with {prefix}"
            )


class ParameterizedFeatureTestMixin:
    """Mixin for parameterized feature testing"""

    def assert_feature_data_structure(self, features: dict):
        """Assert that feature data has correct structure"""
        assert isinstance(features, dict)
        for name, data in features.items():
            assert isinstance(name, str)
            assert isinstance(data, pd.DataFrame)
            assert len(data) > 0
            assert not data.empty

    def assert_feature_names_follow_convention(
        self, feature_names: list, prefix: str = "auto_"
    ):
        """Assert that feature names follow naming convention"""
        for name in feature_names:
            assert name.startswith(prefix), (
                f"Feature name {name} doesn't start with {prefix}"
            )

    def run_parameterized_test(self, generator_method, param_combinations, ohlc_data):
        """Run test with multiple parameter combinations"""
        for params in param_combinations:
            try:
                features = generator_method(ohlc_data, **params)
                self.assert_feature_data_structure(features)

                # Check that features were generated
                assert len(features) > 0, f"No features generated for params: {params}"

                # Check naming convention
                feature_names = list(features.keys())
                self.assert_feature_names_follow_convention(feature_names)

            except Exception as e:
                pytest.fail(f"Test failed for params {params}: {e}")


class TestAutoFeatureGeneration(BaseFeatureTest, ParameterizedFeatureTestMixin):
    def test_ema_cross_feature_generation(self, auto_generator, sample_ohlc_data):
        """Test EMA cross feature generation"""
        features = auto_generator.generate_ema_cross_features(
            sample_ohlc_data, fast_periods=[5, 8], slow_periods=[20, 25]
        )

        # Should generate features for all combinations
        expected_features = [
            "auto_ma_ema_cross_5_20",
            "auto_ma_ema_cross_5_25",
            "auto_ma_ema_cross_8_20",
            "auto_ma_ema_cross_8_25",
        ]

        for feature_name in expected_features:
            assert feature_name in features, f"Missing feature: {feature_name}"
            assert isinstance(features[feature_name], pd.DataFrame)
            assert len(features[feature_name]) == len(sample_ohlc_data)

    def test_kama_feature_generation(self, auto_generator, sample_ohlc_data):
        """Test KAMA feature generation"""
        features = auto_generator.generate_kama_features(
            sample_ohlc_data,
            fast_periods=[10, 20],
            slow_periods=[30],
            efficiency_periods=[30],
        )

        expected_features = ["auto_ma_kama_10_30_30", "auto_ma_kama_20_30_30"]

        for feature_name in expected_features:
            assert feature_name in features, f"Missing feature: {feature_name}"
            assert isinstance(features[feature_name], pd.DataFrame)

    def test_feature_naming_convention(self, auto_generator, sample_ohlc_data):
        """Test that generated features follow naming convention"""
        features = auto_generator.generate_ema_cross_features(sample_ohlc_data)

        for feature_name in features.keys():
            assert feature_name.startswith("auto_"), (
                f"Feature {feature_name} doesn't start with 'auto_'"
            )
            parts = feature_name.split("_")
            assert len(parts) >= 3, (
                f"Feature {feature_name} has invalid naming structure"
            )

    def test_promotion_criteria_validation(self, auto_generator):
        """Test promotion criteria validation"""
        # Test good results that should be promoted
        good_results = {
            "feature_col": {"sharpe_ratio": 0.5, "win_rate": 0.6, "max_drawdown": -0.1}
        }

        # Mock results that should not be promoted
        bad_results = {
            "feature_col": {
                "sharpe_ratio": 0.2,  # < 0.3
                "win_rate": 0.5,  # < 0.55
                "max_drawdown": -0.2,  # < -0.15
            }
        }

        assert auto_generator._check_promotion_criteria(good_results) == True
        assert auto_generator._check_promotion_criteria(bad_results) == False

    def test_evaluate_and_promote_workflow(self, auto_generator, sample_ohlc_data):
        """Test the complete evaluate and promote workflow"""
        # Generate some features
        features = auto_generator.generate_ema_cross_features(
            sample_ohlc_data, fast_periods=[5], slow_periods=[20]
        )

        # Mock the evaluation process
        with (
            patch.object(auto_generator, "_calculate_basic_metrics") as mock_metrics,
            patch.object(auto_generator, "_check_promotion_criteria") as mock_promote,
        ):
            # Setup mocks
            mock_metrics.return_value = {
                "sharpe_ratio": 0.4,
                "win_rate": 0.6,
                "max_drawdown": -0.1,
                "sample_size": 150,
            }
            mock_promote.return_value = True

            # Run evaluation
            result = auto_generator.evaluate_and_promote_features(
                features, sample_ohlc_data
            )

            # Check result structure
            assert isinstance(result, dict)
            assert "promoted" in result
            assert "temporary" in result
            assert "summary" in result

            # Check summary
            summary = result["summary"]
            assert "total_generated" in summary
            assert "promoted_count" in summary
            assert "temporary_count" in summary
            assert "promotion_rate" in summary

    def test_parameter_expansion_logic(self, auto_generator, sample_ohlc_data):
        """Test that parameter expansion creates diverse features"""
        # Generate features with limited parameters
        features = auto_generator.generate_ema_cross_features(
            sample_ohlc_data, fast_periods=[5, 8, 12], slow_periods=[20, 25, 30]
        )

        # Should have 3 * 3 = 9 features
        assert len(features) == 9

        # Check that all combinations are present
        feature_names = list(features.keys())
        for fast in [5, 8, 12]:
            for slow in [20, 25, 30]:
                expected_name = f"auto_ma_ema_cross_{fast}_{slow}"
                assert expected_name in feature_names

    def test_feature_data_integrity(self, auto_generator, sample_ohlc_data):
        """Test that generated feature data maintains integrity"""
        features = auto_generator.generate_ema_cross_features(
            sample_ohlc_data, fast_periods=[5], slow_periods=[20]
        )

        feature_data = features["auto_ma_ema_cross_5_20"]

        # Should have same index as input data
        pd.testing.assert_index_equal(feature_data.index, sample_ohlc_data.index)

        # Should not have NaN values (after initial periods)
        non_nan_data = feature_data.dropna()
        assert len(non_nan_data) > 0, "Feature data should have valid values"

        # Values should be reasonable numeric values (not extreme outliers)
        unique_values = feature_data.stack().unique()
        valid_values = [val for val in unique_values if not np.isnan(val)]
        assert len(valid_values) > 0, "Should have valid numeric values"
        # Check that values are within reasonable bounds (not extreme outliers)
        assert all(-10 <= val <= 10 for val in valid_values), (
            f"Values out of reasonable range: {valid_values}"
        )

    def test_promotion_rules_edge_cases(self, auto_generator):
        """Test promotion rules with edge cases"""
        # Test boundary conditions
        boundary_cases = [
            # Just meets criteria
            {
                "sharpe_ratio": 0.3,
                "win_rate": 0.55,
                "max_drawdown": -0.15,
                "expected": True,
            },
            # Just below criteria
            {
                "sharpe_ratio": 0.29,
                "win_rate": 0.54,
                "max_drawdown": -0.16,
                "expected": False,
            },
            # Mixed (good sharpe, bad win rate)
            {
                "sharpe_ratio": 0.5,
                "win_rate": 0.4,
                "max_drawdown": -0.1,
                "expected": False,
            },
            # All criteria met except max_drawdown
            {
                "sharpe_ratio": 0.5,
                "win_rate": 0.6,
                "max_drawdown": -0.05,
                "expected": True,
            },
        ]

        for case in boundary_cases:
            expected = case.pop("expected")
            # Convert to the format expected by _check_promotion_criteria
            formatted_case = {"feature_col": case}
            assert (
                auto_generator._check_promotion_criteria(formatted_case) == expected
            ), f"Failed for case: {case}"

    def test_auto_generation_error_handling(self, auto_generator):
        """Test error handling in auto generation"""
        # Test with invalid data
        invalid_data = pd.DataFrame({"close": []})  # Empty data

        # Should handle gracefully
        features = auto_generator.generate_ema_cross_features(invalid_data)

        # Should return empty dict or handle error
        assert isinstance(features, dict)

        # Test with insufficient data
        short_data = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [102, 103],
                "low": [98, 99],
                "close": [101, 102],
                "volume": [1000, 1100],
            }
        )

        features = auto_generator.generate_ema_cross_features(short_data)
        # Should still work but may have limited features

    def test_config_parameter_loading(self, auto_generator):
        """Test that AutoFeatureGenerator loads parameters from config file"""
        # Check that params are loaded
        assert hasattr(auto_generator, "params")
        assert "ema" in auto_generator.params
        assert "kama" in auto_generator.params

        # Check EMA parameters
        ema_params = auto_generator.params["ema"]
        assert "periods" in ema_params
        assert isinstance(ema_params["periods"], list)

        # Check KAMA parameters
        kama_params = auto_generator.params["kama"]
        assert "fast_periods" in kama_params
        assert "slow_periods" in kama_params
        assert "efficiency_periods" in kama_params

    def test_naming_convention_enforcement(self, auto_generator, sample_ohlc_data):
        """Test that all generated features follow auto_ naming convention"""
        # Generate EMA features
        ema_features = auto_generator.generate_ema_cross_features(sample_ohlc_data)

        # All feature names should start with 'auto_'
        for feature_name in ema_features.keys():
            assert feature_name.startswith("auto_"), (
                f"Feature name {feature_name} doesn't start with 'auto_'"
            )

        # Generate KAMA features
        kama_features = auto_generator.generate_kama_features(sample_ohlc_data)

        # All feature names should start with 'auto_'
        for feature_name in kama_features.keys():
            assert feature_name.startswith("auto_"), (
                f"Feature name {feature_name} doesn't start with 'auto_'"
            )

    def test_feature_name_uniqueness(self, auto_generator, sample_ohlc_data):
        """Test that generated feature names are unique"""
        # Generate features
        ema_features = auto_generator.generate_ema_cross_features(sample_ohlc_data)
        kama_features = auto_generator.generate_kama_features(sample_ohlc_data)

        # Combine all feature names
        all_names = list(ema_features.keys()) + list(kama_features.keys())

        # Check uniqueness
        assert len(all_names) == len(set(all_names)), "Feature names are not unique"

    def test_max_combinations_limit(self, auto_generator, sample_ohlc_data):
        """Test that max_combinations limit is enforced"""
        # Set a very low limit
        auto_generator.params["max_combinations"] = 2

        # Generate features with many combinations
        features = auto_generator.generate_ema_cross_features(
            sample_ohlc_data, fast_periods=[5, 8, 12, 20], slow_periods=[25, 30, 40, 50]
        )

        # Should be limited to max_combinations
        assert len(features) <= 2, f"Generated {len(features)} features, expected <= 2"
