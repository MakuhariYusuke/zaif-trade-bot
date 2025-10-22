"""Tests for Real Data Validation System component."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from ztb.trading.real_data_validator import (
    AnomalyDetector,
    CrossValidator,
    DataIntegrityChecker,
    DataQualityMetrics,
    DataSource,
    DataValidationConfig,
    RealDataValidationSystem,
    StatisticalValidator,
    ValidationResult,
)


@pytest.fixture
def mock_integration_manager():
    """Mock V433 Integration Manager"""
    manager = Mock()
    manager.component_manager = Mock()
    manager.component_manager.v433_system = Mock()
    manager.component_manager.v433_system.update_market_data = Mock(return_value=None)
    manager.component_manager.position_manager = Mock()
    manager.component_manager.position_manager.submit_signal = Mock(return_value=None)
    return manager


@pytest.fixture
def validation_system(mock_integration_manager):
    """Real Data Validation System instance"""
    return RealDataValidationSystem(mock_integration_manager)


@pytest.fixture
def sample_validation_config():
    """Sample validation configuration"""
    return DataValidationConfig(
        data_sources=["zaif", "binance", "bitflyer"],
        validation_window_days=30,
        min_data_points=1000,
        max_missing_data_pct=0.01,
        outlier_threshold_std=3.0,
        correlation_threshold=0.8,
        stationarity_test_p_value=0.05,
        cross_validation_folds=5,
    )


@pytest.fixture
def sample_market_data():
    """Sample market data for validation"""
    dates = pd.date_range("2023-01-01", periods=1000, freq="1min")
    # Generate realistic price data with some noise
    base_price = 1500000
    price_changes = np.random.normal(0, 0.01, 1000)  # 1% volatility
    prices = base_price * np.cumprod(1 + price_changes)

    data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices * (1 + np.random.normal(0, 0.002, 1000)),
            "high": prices * (1 + np.random.normal(0, 0.005, 1000)),
            "low": prices * (1 + np.random.normal(0, 0.005, 1000)),
            "close": prices,
            "volume": np.random.uniform(100, 10000, 1000),
        }
    )

    # Ensure high >= max(open, close) and low <= min(open, close)
    data["high"] = np.maximum(data[["open", "close", "high"]].max(axis=1), data["high"])
    data["low"] = np.minimum(data[["open", "close", "low"]].min(axis=1), data["low"])

    return data


@pytest.fixture
def sample_data_sources():
    """Sample data sources"""
    return [
        DataSource(
            name="zaif",
            url="https://api.zaif.jp/api/1/ticker/btc_jpy",
            data_format="json",
            update_frequency="1min",
            reliability_score=0.95,
        ),
        DataSource(
            name="binance",
            url="https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
            data_format="json",
            update_frequency="1min",
            reliability_score=0.98,
        ),
    ]


class TestRealDataValidationSystemInitialization:
    """Initialization tests for Real Data Validation System"""

    def test_initialization(
        self, validation_system: RealDataValidationSystem, mock_integration_manager
    ):
        """Test successful initialization"""
        assert validation_system.integration_manager == mock_integration_manager
        assert isinstance(validation_system.integrity_checker, DataIntegrityChecker)
        assert isinstance(validation_system.statistical_validator, StatisticalValidator)
        assert isinstance(validation_system.anomaly_detector, AnomalyDetector)
        assert isinstance(validation_system.cross_validator, CrossValidator)
        assert validation_system.validation_results == []
        assert validation_system.is_validating is False

    def test_initialization_with_config(
        self, mock_integration_manager, sample_validation_config
    ):
        """Test initialization with configuration"""
        system = RealDataValidationSystem(
            mock_integration_manager, sample_validation_config
        )

        assert system.config == sample_validation_config
        assert system.integrity_checker.config == sample_validation_config


class TestRealDataValidationSystemOperations:
    """Operation tests for Real Data Validation System"""

    def test_run_comprehensive_validation(
        self,
        validation_system: RealDataValidationSystem,
        sample_market_data,
        sample_data_sources,
    ):
        """Test comprehensive validation execution"""
        with patch.object(
            validation_system.integrity_checker, "check_data_integrity"
        ) as mock_integrity, patch.object(
            validation_system.statistical_validator, "run_statistical_tests"
        ) as mock_statistical, patch.object(
            validation_system.anomaly_detector, "detect_anomalies"
        ) as mock_anomaly, patch.object(
            validation_system.cross_validator, "perform_cross_validation"
        ) as mock_cross:
            # Mock validation results
            mock_integrity.return_value = ValidationResult(
                data_source="zaif",
                validation_type="integrity",
                passed=True,
                score=0.95,
                issues=[],
                recommendations=[],
            )

            mock_statistical.return_value = ValidationResult(
                data_source="zaif",
                validation_type="statistical",
                passed=True,
                score=0.88,
                issues=[],
                recommendations=[],
            )

            mock_anomaly.return_value = ValidationResult(
                data_source="zaif",
                validation_type="anomaly",
                passed=True,
                score=0.92,
                issues=[],
                recommendations=[],
            )

            mock_cross.return_value = ValidationResult(
                data_source="zaif",
                validation_type="cross_validation",
                passed=True,
                score=0.85,
                issues=[],
                recommendations=[],
            )

            result = validation_system.run_comprehensive_validation(
                sample_market_data, sample_data_sources
            )

            assert "overall_score" in result
            assert "validation_results" in result
            assert "data_quality_report" in result
            assert "recommendations" in result
            assert result["overall_score"] >= 0 and result["overall_score"] <= 1
            assert len(validation_system.validation_results) == 4

    def test_validate_data_sources(
        self, validation_system: RealDataValidationSystem, sample_data_sources
    ):
        """Test data source validation"""
        with patch.object(
            validation_system, "run_comprehensive_validation"
        ) as mock_validate:
            mock_validate.return_value = {
                "overall_score": 0.9,
                "validation_results": [],
                "data_quality_report": {},
                "recommendations": [],
            }

            results = validation_system.validate_data_sources(
                sample_data_sources, pd.DataFrame()
            )

            assert len(results) == len(sample_data_sources)
            assert all("overall_score" in r for r in results)

    def test_get_validation_report(self, validation_system: RealDataValidationSystem):
        """Test getting validation report"""
        # Add mock validation results
        validation_system.validation_results = [
            ValidationResult("zaif", "integrity", True, 0.95, [], []),
            ValidationResult("zaif", "statistical", True, 0.88, [], []),
            ValidationResult(
                "binance", "integrity", False, 0.7, ["Missing data"], ["Fix data gaps"]
            ),
        ]

        report = validation_system.get_validation_report()

        assert "summary" in report
        assert "data_sources_status" in report
        assert "validation_timeline" in report
        assert "quality_trends" in report
        assert report["summary"]["total_validations"] == 3
        assert report["summary"]["passed_validations"] == 2
        assert report["summary"]["failed_validations"] == 1

    def test_monitor_data_quality(
        self, validation_system: RealDataValidationSystem, sample_data_sources
    ):
        """Test data quality monitoring"""
        with patch.object(
            validation_system, "run_comprehensive_validation"
        ) as mock_validate:
            mock_validate.return_value = {
                "overall_score": 0.85,
                "validation_results": [],
                "data_quality_report": {"data_completeness": 0.98},
                "recommendations": [],
            }

            result = validation_system.monitor_data_quality(
                sample_data_sources, pd.DataFrame()
            )

            assert result is True
            assert validation_system.is_validating is False  # Should complete

    def test_validate_real_time_data(
        self, validation_system: RealDataValidationSystem, sample_market_data
    ):
        """Test real-time data validation"""
        with patch.object(
            validation_system.integrity_checker, "check_real_time_integrity"
        ) as mock_integrity, patch.object(
            validation_system.anomaly_detector, "detect_real_time_anomalies"
        ) as mock_anomaly:
            mock_integrity.return_value = {"is_valid": True, "issues": []}
            mock_anomaly.return_value = {
                "anomalies_detected": False,
                "anomaly_score": 0.1,
            }

            result = validation_system.validate_real_time_data(
                sample_market_data.iloc[-1:]
            )

            assert "real_time_valid" in result
            assert "integrity_check" in result
            assert "anomaly_check" in result
            assert result["real_time_valid"] is True


class TestDataIntegrityChecker:
    """Tests for DataIntegrityChecker"""

    def test_initialization(self, mock_integration_manager, sample_validation_config):
        """Test DataIntegrityChecker initialization"""
        checker = DataIntegrityChecker(
            mock_integration_manager, sample_validation_config
        )

        assert checker.integration_manager == mock_integration_manager
        assert checker.config == sample_validation_config

    def test_check_data_integrity(
        self, mock_integration_manager, sample_validation_config, sample_market_data
    ):
        """Test data integrity checking"""
        checker = DataIntegrityChecker(
            mock_integration_manager, sample_validation_config
        )

        result = checker.check_data_integrity(sample_market_data, "zaif")

        assert isinstance(result, ValidationResult)
        assert result.data_source == "zaif"
        assert result.validation_type == "integrity"
        assert isinstance(result.passed, bool)
        assert result.score >= 0 and result.score <= 1

    def test_check_missing_data(
        self, mock_integration_manager, sample_validation_config
    ):
        """Test missing data detection"""
        checker = DataIntegrityChecker(
            mock_integration_manager, sample_validation_config
        )

        # Create data with missing values
        data_with_missing = pd.DataFrame(
            {
                "open": [1, 2, None, 4, 5],
                "close": [1, 2, 3, None, 5],
                "volume": [100, 200, 300, 400, None],
            }
        )

        missing_pct = checker._check_missing_data(data_with_missing)

        assert missing_pct["open"] == 0.2  # 1 out of 5
        assert missing_pct["close"] == 0.2
        assert missing_pct["volume"] == 0.2

    def test_check_data_types(
        self, mock_integration_manager, sample_validation_config, sample_market_data
    ):
        """Test data type validation"""
        checker = DataIntegrityChecker(
            mock_integration_manager, sample_validation_config
        )

        type_issues = checker._check_data_types(sample_market_data)

        assert isinstance(type_issues, list)
        # Should have no issues for properly typed data

    def test_check_data_ranges(
        self, mock_integration_manager, sample_validation_config
    ):
        """Test data range validation"""
        checker = DataIntegrityChecker(
            mock_integration_manager, sample_validation_config
        )

        # Create data with invalid ranges
        invalid_data = pd.DataFrame(
            {
                "open": [1, 2, -100, 4, 5],  # Negative price
                "high": [1, 2, 3, 4, 5],
                "low": [1, 2, 3, 4, 5],
                "close": [1, 2, 3, 4, 5],
                "volume": [100, 200, -50, 400, 500],  # Negative volume
            }
        )

        range_issues = checker._check_data_ranges(invalid_data)

        assert len(range_issues) > 0
        assert any("negative" in issue.lower() for issue in range_issues)


class TestStatisticalValidator:
    """Tests for StatisticalValidator"""

    def test_initialization(self, mock_integration_manager, sample_validation_config):
        """Test StatisticalValidator initialization"""
        validator = StatisticalValidator(
            mock_integration_manager, sample_validation_config
        )

        assert validator.integration_manager == mock_integration_manager
        assert validator.config == sample_validation_config

    def test_run_statistical_tests(
        self, mock_integration_manager, sample_validation_config, sample_market_data
    ):
        """Test statistical validation"""
        validator = StatisticalValidator(
            mock_integration_manager, sample_validation_config
        )

        result = validator.run_statistical_tests(sample_market_data, "zaif")

        assert isinstance(result, ValidationResult)
        assert result.data_source == "zaif"
        assert result.validation_type == "statistical"
        assert isinstance(result.passed, bool)
        assert result.score >= 0 and result.score <= 1

    def test_test_normality(self, mock_integration_manager, sample_validation_config):
        """Test normality testing"""
        validator = StatisticalValidator(
            mock_integration_manager, sample_validation_config
        )

        # Normal data
        normal_data = np.random.normal(0, 1, 1000)
        normal_result = validator._test_normality(normal_data)

        # Non-normal data
        non_normal_data = np.random.exponential(1, 1000)
        non_normal_result = validator._test_normality(non_normal_data)

        assert "statistic" in normal_result
        assert "p_value" in normal_result
        assert "is_normal" in normal_result
        assert isinstance(normal_result["is_normal"], bool)

    def test_test_stationarity(
        self, mock_integration_manager, sample_validation_config
    ):
        """Test stationarity testing"""
        validator = StatisticalValidator(
            mock_integration_manager, sample_validation_config
        )

        # Stationary data (white noise)
        stationary_data = np.random.normal(0, 1, 1000)
        stationary_result = validator._test_stationarity(stationary_data)

        # Non-stationary data (random walk)
        non_stationary_data = np.cumsum(np.random.normal(0, 1, 1000))
        non_stationary_result = validator._test_stationarity(non_stationary_data)

        assert "adf_statistic" in stationary_result
        assert "p_value" in stationary_result
        assert "is_stationary" in stationary_result
        assert isinstance(stationary_result["is_stationary"], bool)

    def test_calculate_volatility(
        self, mock_integration_manager, sample_validation_config, sample_market_data
    ):
        """Test volatility calculation"""
        validator = StatisticalValidator(
            mock_integration_manager, sample_validation_config
        )

        volatility = validator._calculate_volatility(sample_market_data["close"])

        assert isinstance(volatility, float)
        assert volatility >= 0


class TestAnomalyDetector:
    """Tests for AnomalyDetector"""

    def test_initialization(self, mock_integration_manager, sample_validation_config):
        """Test AnomalyDetector initialization"""
        detector = AnomalyDetector(mock_integration_manager, sample_validation_config)

        assert detector.integration_manager == mock_integration_manager
        assert detector.config == sample_validation_config

    def test_detect_anomalies(
        self, mock_integration_manager, sample_validation_config, sample_market_data
    ):
        """Test anomaly detection"""
        detector = AnomalyDetector(mock_integration_manager, sample_validation_config)

        result = detector.detect_anomalies(sample_market_data, "zaif")

        assert isinstance(result, ValidationResult)
        assert result.data_source == "zaif"
        assert result.validation_type == "anomaly"
        assert isinstance(result.passed, bool)
        assert result.score >= 0 and result.score <= 1

    def test_isolation_forest_detection(
        self, mock_integration_manager, sample_validation_config
    ):
        """Test isolation forest anomaly detection"""
        detector = AnomalyDetector(mock_integration_manager, sample_validation_config)

        # Create normal data with some outliers
        normal_data = np.random.normal(100, 10, 1000)
        outlier_data = np.append(normal_data, [500, -200, 600])  # Add extreme outliers

        anomalies = detector._isolation_forest_detection(outlier_data.reshape(-1, 1))

        assert isinstance(anomalies, np.ndarray)
        assert len(anomalies) == len(outlier_data)
        # Should detect some anomalies
        assert np.sum(anomalies) > 0

    def test_zscore_detection(self, mock_integration_manager, sample_validation_config):
        """Test Z-score anomaly detection"""
        detector = AnomalyDetector(mock_integration_manager, sample_validation_config)

        # Create data with outliers
        data = np.array([1, 2, 3, 4, 5, 100, 6, 7, 8, 9])  # 100 is outlier

        anomalies = detector._zscore_detection(data)

        assert isinstance(anomalies, list)
        assert len(anomalies) >= 1  # Should detect at least the outlier

    def test_mad_detection(self, mock_integration_manager, sample_validation_config):
        """Test MAD (Median Absolute Deviation) anomaly detection"""
        detector = AnomalyDetector(mock_integration_manager, sample_validation_config)

        # Create data with outliers
        data = np.array([1, 2, 3, 4, 5, 100, 6, 7, 8, 9])

        anomalies = detector._mad_detection(data)

        assert isinstance(anomalies, list)
        assert len(anomalies) >= 1


class TestCrossValidator:
    """Tests for CrossValidator"""

    def test_initialization(self, mock_integration_manager, sample_validation_config):
        """Test CrossValidator initialization"""
        validator = CrossValidator(mock_integration_manager, sample_validation_config)

        assert validator.integration_manager == mock_integration_manager
        assert validator.config == sample_validation_config

    def test_perform_cross_validation(
        self, mock_integration_manager, sample_validation_config, sample_data_sources
    ):
        """Test cross validation"""
        validator = CrossValidator(mock_integration_manager, sample_validation_config)

        # Mock data for different sources
        data_dict = {
            "zaif": pd.DataFrame({"close": np.random.uniform(1000000, 2000000, 100)}),
            "binance": pd.DataFrame(
                {"close": np.random.uniform(1000000, 2000000, 100)}
            ),
        }

        result = validator.perform_cross_validation(data_dict, "zaif")

        assert isinstance(result, ValidationResult)
        assert result.data_source == "zaif"
        assert result.validation_type == "cross_validation"
        assert isinstance(result.passed, bool)
        assert result.score >= 0 and result.score <= 1

    def test_calculate_correlation_matrix(
        self, mock_integration_manager, sample_validation_config
    ):
        """Test correlation matrix calculation"""
        validator = CrossValidator(mock_integration_manager, sample_validation_config)

        data_dict = {
            "source1": pd.DataFrame({"price": [1, 2, 3, 4, 5]}),
            "source2": pd.DataFrame(
                {"price": [1.1, 2.1, 3.1, 4.1, 5.1]}
            ),  # Highly correlated
            "source3": pd.DataFrame({"price": [5, 4, 3, 2, 1]}),  # Inversely correlated
        }

        correlation_matrix = validator._calculate_correlation_matrix(data_dict)

        assert isinstance(correlation_matrix, pd.DataFrame)
        assert correlation_matrix.shape[0] == 3
        assert correlation_matrix.shape[1] == 3
        # Check that source1 and source2 have high correlation
        assert abs(correlation_matrix.loc["source1", "source2"]) > 0.9

    def test_detect_data_discrepancies(
        self, mock_integration_manager, sample_validation_config
    ):
        """Test data discrepancy detection"""
        validator = CrossValidator(mock_integration_manager, sample_validation_config)

        data_dict = {
            "source1": pd.DataFrame({"price": [100, 105, 110, 115, 120]}),
            "source2": pd.DataFrame({"price": [100, 105, 110, 115, 120]}),  # Identical
            "source3": pd.DataFrame(
                {"price": [100, 200, 110, 115, 120]}
            ),  # One outlier
        }

        discrepancies = validator._detect_data_discrepancies(data_dict)

        assert isinstance(discrepancies, dict)
        assert "source1_vs_source2" in discrepancies
        assert "source1_vs_source3" in discrepancies
        # source3 should have higher discrepancy due to outlier
        assert (
            discrepancies["source1_vs_source3"]["max_difference"]
            > discrepancies["source1_vs_source2"]["max_difference"]
        )


class TestDataValidationConfig:
    """Tests for DataValidationConfig dataclass"""

    def test_initialization(self):
        """Test DataValidationConfig initialization"""
        config = DataValidationConfig()

        assert config.data_sources == []
        assert config.validation_window_days == 30
        assert config.min_data_points == 1000
        assert config.max_missing_data_pct == 0.01

    def test_custom_initialization(self, sample_validation_config):
        """Test DataValidationConfig with custom values"""
        assert sample_validation_config.validation_window_days == 30
        assert sample_validation_config.min_data_points == 1000
        assert sample_validation_config.outlier_threshold_std == 3.0


class TestValidationResult:
    """Tests for ValidationResult dataclass"""

    def test_initialization(self):
        """Test ValidationResult initialization"""
        result = ValidationResult(
            data_source="zaif",
            validation_type="integrity",
            passed=True,
            score=0.95,
            issues=[],
            recommendations=[],
        )

        assert result.data_source == "zaif"
        assert result.validation_type == "integrity"
        assert result.passed is True
        assert result.score == 0.95
        assert result.issues == []
        assert result.recommendations == []

    def test_result_summary(self):
        """Test result summary property"""
        result = ValidationResult(
            data_source="zaif",
            validation_type="integrity",
            passed=False,
            score=0.7,
            issues=["Missing data", "Invalid ranges"],
            recommendations=["Fix data gaps", "Validate ranges"],
        )

        summary = result.result_summary

        assert "FAILED" in summary
        assert "0.70" in summary
        assert "Missing data" in summary
        assert "Invalid ranges" in summary


class TestDataSource:
    """Tests for DataSource dataclass"""

    def test_initialization(self):
        """Test DataSource initialization"""
        source = DataSource(
            name="zaif",
            url="https://api.zaif.jp/api/1/ticker/btc_jpy",
            data_format="json",
            update_frequency="1min",
            reliability_score=0.95,
        )

        assert source.name == "zaif"
        assert source.url == "https://api.zaif.jp/api/1/ticker/btc_jpy"
        assert source.data_format == "json"
        assert source.update_frequency == "1min"
        assert source.reliability_score == 0.95

    def test_is_reliable(self):
        """Test reliability check"""
        reliable_source = DataSource(
            name="test",
            url="",
            data_format="json",
            update_frequency="1min",
            reliability_score=0.9,
        )
        unreliable_source = DataSource(
            name="test",
            url="",
            data_format="json",
            update_frequency="1min",
            reliability_score=0.3,
        )

        assert reliable_source.is_reliable() is True
        assert unreliable_source.is_reliable() is False


class TestDataQualityMetrics:
    """Tests for DataQualityMetrics dataclass"""

    def test_initialization(self):
        """Test DataQualityMetrics initialization"""
        metrics = DataQualityMetrics()

        assert metrics.completeness_score == 0.0
        assert metrics.accuracy_score == 0.0
        assert metrics.consistency_score == 0.0
        assert metrics.timeliness_score == 0.0

    def test_overall_quality_score(self):
        """Test overall quality score calculation"""
        metrics = DataQualityMetrics(
            completeness_score=0.9,
            accuracy_score=0.8,
            consistency_score=0.7,
            timeliness_score=0.6,
        )

        overall_score = metrics.overall_quality_score

        assert overall_score == 0.75  # Average of all scores
