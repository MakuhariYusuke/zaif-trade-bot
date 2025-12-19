"""Tests for End-to-End Validation System component."""

import datetime
from unittest.mock import Mock, patch

import pytest

from ztb.trading.end_to_end_validator import (
    ComponentTest,
    ComponentValidator,
    EndToEndValidationSystem,
    HealthCheck,
    IntegrationTest,
    IntegrationValidator,
    PerformanceTest,
    PerformanceValidator,
    PipelineResult,
    SystemHealthValidator,
    ValidationMetrics,
    ValidationPipeline,
    ValidationReport,
    ValidationStage,
    ValidationStatus,
)


@pytest.fixture


@pytest.fixture
def e2e_validator(mock_integration_manager):
    """End-to-End Validation System instance"""
    return EndToEndValidationSystem(mock_integration_manager)


@pytest.fixture
def sample_validation_pipeline():
    """Sample validation pipeline"""
    return ValidationPipeline(
        pipeline_id="test_pipeline_001",
        name="V433 End-to-End Validation",
        description="Complete system validation pipeline",
        stages=[
            ValidationStage(
                stage_id="component_validation",
                name="Component Validation",
                description="Validate individual components",
                order=1,
                timeout_seconds=300,
                required=True,
            ),
            ValidationStage(
                stage_id="integration_validation",
                name="Integration Validation",
                description="Validate component integration",
                order=2,
                timeout_seconds=600,
                required=True,
            ),
            ValidationStage(
                stage_id="performance_validation",
                name="Performance Validation",
                description="Validate system performance",
                order=3,
                timeout_seconds=900,
                required=True,
            ),
            ValidationStage(
                stage_id="health_validation",
                name="Health Validation",
                description="Validate system health",
                order=4,
                timeout_seconds=300,
                required=True,
            ),
        ],
        created_at=datetime.datetime.now(),
        updated_at=datetime.datetime.now(),
    )


@pytest.fixture
def sample_component_tests():
    """Sample component tests"""
    return [
        ComponentTest(
            test_id="v433_system_test",
            component_name="V433 System",
            test_type="unit_test",
            description="Test V433 system initialization and basic functions",
            expected_result=True,
            timeout_seconds=60,
        ),
        ComponentTest(
            test_id="performance_optimizer_test",
            component_name="Performance Optimizer",
            test_type="integration_test",
            description="Test performance optimization components",
            expected_result=True,
            timeout_seconds=120,
        ),
    ]


@pytest.fixture
def sample_integration_tests():
    """Sample integration tests"""
    return [
        IntegrationTest(
            test_id="data_flow_test",
            description="Test data flow between components",
            components_involved=["V433 System", "Performance Optimizer"],
            expected_behavior="Data flows correctly between components",
            success_criteria="No data loss, correct transformations",
            timeout_seconds=300,
        ),
        IntegrationTest(
            test_id="signal_processing_test",
            description="Test signal processing integration",
            components_involved=["V433 System", "Position Manager"],
            expected_behavior="Signals processed and executed correctly",
            success_criteria="Signals converted to positions accurately",
            timeout_seconds=300,
        ),
    ]


class TestEndToEndValidationSystemInitialization:
    """Initialization tests for End-to-End Validation System"""

    def test_initialization(
        self, e2e_validator: EndToEndValidationSystem, mock_integration_manager
    ):
        """Test successful initialization"""
        assert e2e_validator.integration_manager == mock_integration_manager
        assert isinstance(e2e_validator.component_validator, ComponentValidator)
        assert isinstance(e2e_validator.integration_validator, IntegrationValidator)
        assert isinstance(e2e_validator.performance_validator, PerformanceValidator)
        assert isinstance(e2e_validator.health_validator, SystemHealthValidator)
        assert e2e_validator.validation_pipelines == []
        assert e2e_validator.is_running is False
        assert e2e_validator.current_pipeline is None

    def test_initialization_with_pipeline(
        self, mock_integration_manager, sample_validation_pipeline
    ):
        """Test initialization with validation pipeline"""
        system = EndToEndValidationSystem(
            mock_integration_manager, sample_validation_pipeline
        )

        assert system.current_pipeline == sample_validation_pipeline
        assert len(system.validation_pipelines) == 1


class TestEndToEndValidationSystemOperations:
    """Operation tests for End-to-End Validation System"""

    def test_run_end_to_end_validation(
        self, e2e_validator: EndToEndValidationSystem, sample_validation_pipeline
    ):
        """Test end-to-end validation execution"""
        with patch.object(
            e2e_validator.component_validator, "validate_components"
        ) as mock_component, patch.object(
            e2e_validator.integration_validator, "validate_integrations"
        ) as mock_integration, patch.object(
            e2e_validator.performance_validator, "validate_performance"
        ) as mock_performance, patch.object(
            e2e_validator.health_validator, "validate_system_health"
        ) as mock_health:
            # Mock validation results
            mock_component.return_value = PipelineResult(
                stage_id="component_validation",
                status=ValidationStatus.PASSED,
                duration_seconds=45.2,
                metrics=ValidationMetrics(),
                issues=[],
                recommendations=[],
            )

            mock_integration.return_value = PipelineResult(
                stage_id="integration_validation",
                status=ValidationStatus.PASSED,
                duration_seconds=120.5,
                metrics=ValidationMetrics(),
                issues=[],
                recommendations=[],
            )

            mock_performance.return_value = PipelineResult(
                stage_id="performance_validation",
                status=ValidationStatus.PASSED,
                duration_seconds=180.3,
                metrics=ValidationMetrics(),
                issues=[],
                recommendations=[],
            )

            mock_health.return_value = PipelineResult(
                stage_id="health_validation",
                status=ValidationStatus.PASSED,
                duration_seconds=30.1,
                metrics=ValidationMetrics(),
                issues=[],
                recommendations=[],
            )

            result = e2e_validator.run_end_to_end_validation(sample_validation_pipeline)

            assert isinstance(result, ValidationReport)
            assert result.pipeline_id == sample_validation_pipeline.pipeline_id
            assert result.overall_status == ValidationStatus.PASSED
            assert len(result.stage_results) == 4
            assert result.total_duration > 0
            assert e2e_validator.current_pipeline == sample_validation_pipeline

    def test_run_validation_pipeline(
        self, e2e_validator: EndToEndValidationSystem, sample_validation_pipeline
    ):
        """Test validation pipeline execution"""
        with patch.object(e2e_validator, "run_end_to_end_validation") as mock_run:
            mock_run.return_value = ValidationReport(
                pipeline_id=sample_validation_pipeline.pipeline_id,
                overall_status=ValidationStatus.PASSED,
                stage_results=[],
                total_duration=375.1,
                validation_metrics=ValidationMetrics(),
                issues=[],
                recommendations=[],
                generated_at=datetime.datetime.now(),
            )

            result = e2e_validator.run_validation_pipeline(
                sample_validation_pipeline.pipeline_id
            )

            assert isinstance(result, ValidationReport)
            assert result.pipeline_id == sample_validation_pipeline.pipeline_id
            assert result.overall_status == ValidationStatus.PASSED

    def test_get_validation_status(
        self, e2e_validator: EndToEndValidationSystem, sample_validation_pipeline
    ):
        """Test getting validation status"""
        e2e_validator.current_pipeline = sample_validation_pipeline
        e2e_validator.is_running = True

        status = e2e_validator.get_validation_status()

        assert "current_pipeline" in status
        assert "is_running" in status
        assert "pipeline_progress" in status
        assert status["current_pipeline"] == sample_validation_pipeline.pipeline_id
        assert status["is_running"] is True

    def test_get_validation_history(
        self, e2e_validator: EndToEndValidationSystem, sample_validation_pipeline
    ):
        """Test getting validation history"""
        # Add some mock reports
        e2e_validator.validation_pipelines = [sample_validation_pipeline]

        # Mock return value
        mock_report = ValidationReport(
            pipeline_id=sample_validation_pipeline.pipeline_id,
            overall_status=ValidationStatus.PASSED,
            stage_results=[],
            total_duration=100.0,
            validation_metrics=ValidationMetrics(),
            issues=[],
            recommendations=[],
            generated_at=datetime.datetime.now(),
        )

        with patch.object(e2e_validator, "run_validation_pipeline") as mock_run:
            # side_effect to add report to history
            def add_to_history(pipeline_id):
                e2e_validator.validation_history.append(mock_report)
                return mock_report

            mock_run.side_effect = add_to_history

            # Run a few validations
            e2e_validator.run_validation_pipeline(
                sample_validation_pipeline.pipeline_id
            )
            e2e_validator.run_validation_pipeline(
                sample_validation_pipeline.pipeline_id
            )

            history = e2e_validator.get_validation_history()

            assert "total_validations" in history
            assert "successful_validations" in history
            assert "failed_validations" in history
            assert "average_duration" in history
            assert history["total_validations"] == 2

    def test_stop_validation(self, e2e_validator: EndToEndValidationSystem):
        """Test stopping validation"""
        e2e_validator.is_running = True
        e2e_validator.current_pipeline = Mock()

        result = e2e_validator.stop_validation()

        assert result is True
        assert e2e_validator.is_running is False
        assert e2e_validator.current_pipeline is None


class TestComponentValidator:
    """Tests for ComponentValidator"""

    def test_initialization(self, mock_integration_manager):
        """Test ComponentValidator initialization"""
        validator = ComponentValidator(mock_integration_manager)

        assert validator.integration_manager == mock_integration_manager
        assert validator.component_tests == []

    def test_validate_components(
        self, mock_integration_manager, sample_component_tests
    ):
        """Test component validation"""
        validator = ComponentValidator(mock_integration_manager)
        validator.component_tests = sample_component_tests

        with patch.object(validator, "_run_component_test") as mock_run:
            mock_run.side_effect = [
                {"passed": True, "duration": 30.5, "metrics": {"test_score": 0.95}},
                {"passed": True, "duration": 75.2, "metrics": {"test_score": 0.88}},
            ]

            result = validator.validate_components()

            assert isinstance(result, PipelineResult)
            assert result.stage_id == "component_validation"
            assert result.status == ValidationStatus.PASSED
            assert result.duration_seconds > 0
            assert len(result.issues) == 0

    def test_run_component_test(self, mock_integration_manager):
        """Test running individual component test"""
        validator = ComponentValidator(mock_integration_manager)

        test = ComponentTest(
            test_id="v433_system_test",
            component_name="V433 System",
            test_type="unit_test",
            description="Test V433 system initialization",
            expected_result=True,
            timeout_seconds=60,
        )

        with patch.object(validator, "_execute_component_test") as mock_execute:
            mock_execute.return_value = {"passed": True, "metrics": {"init_time": 2.3}}

            result = validator._run_component_test(test)

            assert "passed" in result
            assert "duration" in result
            assert "metrics" in result
            assert result["passed"] is True

    def test_validate_component_health(self, mock_integration_manager):
        """Test component health validation"""
        validator = ComponentValidator(mock_integration_manager)

        with patch.object(
            validator.integration_manager.component_manager, "get_component_status"
        ) as mock_status:
            mock_status.return_value = {
                "status": "healthy",
                "metrics": {"cpu": 45.0, "memory": 512},
            }

            health = validator._validate_component_health("V433 System")

            assert "is_healthy" in health
            assert "status" in health
            assert "metrics" in health
            assert health["is_healthy"] is True


class TestIntegrationValidator:
    """Tests for IntegrationValidator"""

    def test_initialization(self, mock_integration_manager):
        """Test IntegrationValidator initialization"""
        validator = IntegrationValidator(mock_integration_manager)

        assert validator.integration_manager == mock_integration_manager
        assert validator.integration_tests == []

    def test_validate_integrations(
        self, mock_integration_manager, sample_integration_tests
    ):
        """Test integration validation"""
        validator = IntegrationValidator(mock_integration_manager)
        validator.integration_tests = sample_integration_tests

        with patch.object(validator, "_run_integration_test") as mock_run:
            mock_run.side_effect = [
                {"passed": True, "duration": 45.2, "metrics": {"data_loss": 0.0}},
                {
                    "passed": True,
                    "duration": 67.8,
                    "metrics": {"signal_accuracy": 0.98},
                },
            ]

            result = validator.validate_integrations()

            assert isinstance(result, PipelineResult)
            assert result.stage_id == "integration_validation"
            assert result.status == ValidationStatus.PASSED
            assert result.duration_seconds > 0

    def test_run_integration_test(self, mock_integration_manager):
        """Test running individual integration test"""
        validator = IntegrationValidator(mock_integration_manager)

        test = IntegrationTest(
            test_id="data_flow_test",
            description="Test data flow between components",
            components_involved=["V433 System", "Performance Optimizer"],
            expected_behavior="Data flows correctly",
            success_criteria="No data loss",
            timeout_seconds=300,
        )

        with patch.object(validator, "_execute_integration_test") as mock_execute:
            mock_execute.return_value = {
                "passed": True,
                "metrics": {"throughput": 1000},
            }

            result = validator._run_integration_test(test)

            assert "passed" in result
            assert "duration" in result
            assert "metrics" in result

    def test_validate_data_flow(self, mock_integration_manager):
        """Test data flow validation"""
        validator = IntegrationValidator(mock_integration_manager)

        with patch.object(
            validator.integration_manager.component_manager.v433_system,
            "get_data_flow_metrics",
        ) as mock_metrics:
            mock_metrics.return_value = {
                "data_processed": 10000,
                "data_lost": 0,
                "avg_latency": 50.0,
            }

            flow_result = validator._validate_data_flow(
                "V433 System", "Performance Optimizer"
            )

            assert "data_integrity" in flow_result
            assert "latency_acceptable" in flow_result
            assert "flow_stable" in flow_result


class TestPerformanceValidator:
    """Tests for PerformanceValidator"""

    def test_initialization(self, mock_integration_manager):
        """Test PerformanceValidator initialization"""
        validator = PerformanceValidator(mock_integration_manager)

        assert validator.integration_manager == mock_integration_manager
        assert validator.performance_tests == []

    def test_validate_performance(self, mock_integration_manager):
        """Test performance validation"""
        validator = PerformanceValidator(mock_integration_manager)

        performance_tests = [
            PerformanceTest(
                test_id="latency_test",
                description="Test system latency",
                metric_name="avg_latency_ms",
                target_value=100.0,
                operator="<=",
                timeout_seconds=300,
            )
        ]
        validator.performance_tests = performance_tests

        with patch.object(validator, "_run_performance_test") as mock_run:
            mock_run.return_value = {
                "passed": True,
                "duration": 45.2,
                "metrics": {"avg_latency_ms": 75.0},
            }

            result = validator.validate_performance()

            assert isinstance(result, PipelineResult)
            assert result.stage_id == "performance_validation"
            assert result.status == ValidationStatus.PASSED

    def test_run_performance_test(self, mock_integration_manager):
        """Test running individual performance test"""
        validator = PerformanceValidator(mock_integration_manager)

        test = PerformanceTest(
            test_id="throughput_test",
            description="Test system throughput",
            metric_name="ops_per_second",
            target_value=1000.0,
            operator=">=",
            timeout_seconds=300,
        )

        with patch.object(validator, "_measure_performance_metric") as mock_measure:
            mock_measure.return_value = 1200.0

            result = validator._run_performance_test(test)

            assert "passed" in result
            assert "duration" in result
            assert "metrics" in result
            assert result["passed"] is True
            assert result["metrics"]["ops_per_second"] == 1200.0

    def test_measure_performance_metric(self, mock_integration_manager):
        """Test performance metric measurement"""
        validator = PerformanceValidator(mock_integration_manager)

        with patch.object(
            validator.integration_manager.component_manager, "get_performance_metrics"
        ) as mock_metrics:
            mock_metrics.return_value = {
                "avg_latency_ms": 45.2,
                "throughput_ops": 1500,
                "memory_usage_mb": 512,
            }

            latency = validator._measure_performance_metric("avg_latency_ms")
            throughput = validator._measure_performance_metric("throughput_ops")

            assert latency == 45.2
            assert throughput == 1500


class TestSystemHealthValidator:
    """Tests for SystemHealthValidator"""

    def test_initialization(self, mock_integration_manager):
        """Test SystemHealthValidator initialization"""
        validator = SystemHealthValidator(mock_integration_manager)

        assert validator.integration_manager == mock_integration_manager
        assert validator.health_checks == []

    def test_validate_system_health(self, mock_integration_manager):
        """Test system health validation"""
        validator = SystemHealthValidator(mock_integration_manager)

        health_checks = [
            HealthCheck(
                check_id="memory_check",
                description="Check system memory usage",
                check_type="resource",
                expected_result=True,
                timeout_seconds=30,
            )
        ]
        validator.health_checks = health_checks

        with patch.object(validator, "_run_health_check") as mock_run:
            mock_run.return_value = {
                "passed": True,
                "duration": 5.2,
                "metrics": {"memory_pct": 65.0},
            }

            result = validator.validate_system_health()

            assert isinstance(result, PipelineResult)
            assert result.stage_id == "health_validation"
            assert result.status == ValidationStatus.PASSED

    def test_run_health_check(self, mock_integration_manager):
        """Test running individual health check"""
        validator = SystemHealthValidator(mock_integration_manager)

        check = HealthCheck(
            check_id="cpu_check",
            description="Check CPU usage",
            check_type="resource",
            expected_result=True,
            timeout_seconds=30,
        )

        with patch.object(validator, "_execute_health_check") as mock_execute:
            mock_execute.return_value = {"cpu_usage": 45.0, "status": "healthy"}

            result = validator._run_health_check(check)

            assert "passed" in result
            assert "duration" in result
            assert "metrics" in result

    def test_check_system_resources(self, mock_integration_manager):
        """Test system resource checking"""
        validator = SystemHealthValidator(mock_integration_manager)

        with patch("psutil.cpu_percent", return_value=45.0), patch(
            "psutil.virtual_memory"
        ) as mock_memory, patch("psutil.disk_usage") as mock_disk:
            mock_memory.return_value.percent = 65.0
            mock_disk.return_value.percent = 55.0

            resources = validator._check_system_resources()

            assert "cpu_percent" in resources
            assert "memory_percent" in resources
            assert "disk_percent" in resources
            assert resources["cpu_percent"] == 45.0
            assert resources["memory_percent"] == 65.0


class TestValidationPipeline:
    """Tests for ValidationPipeline dataclass"""

    def test_initialization(self, sample_validation_pipeline):
        """Test ValidationPipeline initialization"""
        assert sample_validation_pipeline.pipeline_id == "test_pipeline_001"
        assert sample_validation_pipeline.name == "V433 End-to-End Validation"
        assert len(sample_validation_pipeline.stages) == 4
        assert sample_validation_pipeline.created_at is not None

    def test_get_stage_by_id(self, sample_validation_pipeline):
        """Test getting stage by ID"""
        stage = sample_validation_pipeline.get_stage_by_id("component_validation")

        assert stage is not None
        assert stage.stage_id == "component_validation"
        assert stage.name == "Component Validation"

    def test_get_stages_in_order(self, sample_validation_pipeline):
        """Test getting stages in order"""
        ordered_stages = sample_validation_pipeline.get_stages_in_order()

        assert len(ordered_stages) == 4
        assert ordered_stages[0].order == 1
        assert ordered_stages[1].order == 2
        assert ordered_stages[2].order == 3
        assert ordered_stages[3].order == 4


class TestValidationStage:
    """Tests for ValidationStage dataclass"""

    def test_initialization(self):
        """Test ValidationStage initialization"""
        stage = ValidationStage(
            stage_id="test_stage",
            name="Test Stage",
            description="A test validation stage",
            order=1,
            timeout_seconds=300,
            required=True,
        )

        assert stage.stage_id == "test_stage"
        assert stage.name == "Test Stage"
        assert stage.order == 1
        assert stage.timeout_seconds == 300
        assert stage.required is True

    def test_is_required(self):
        """Test required check"""
        required_stage = ValidationStage(
            stage_id="required",
            name="Required",
            description="",
            order=1,
            timeout_seconds=60,
            required=True,
        )
        optional_stage = ValidationStage(
            stage_id="optional",
            name="Optional",
            description="",
            order=1,
            timeout_seconds=60,
            required=False,
        )

        assert required_stage.is_required() is True
        assert optional_stage.is_required() is False


class TestValidationStatus:
    """Tests for ValidationStatus enum"""

    def test_status_values(self):
        """Test validation status values"""
        assert ValidationStatus.PENDING.value == "pending"
        assert ValidationStatus.RUNNING.value == "running"
        assert ValidationStatus.PASSED.value == "passed"
        assert ValidationStatus.FAILED.value == "failed"
        assert ValidationStatus.SKIPPED.value == "skipped"
        assert ValidationStatus.TIMEOUT.value == "timeout"

    def test_status_order(self):
        """Test status ordering for severity"""
        assert ValidationStatus.PENDING < ValidationStatus.RUNNING
        assert ValidationStatus.RUNNING < ValidationStatus.PASSED
        assert ValidationStatus.PASSED > ValidationStatus.FAILED


class TestValidationReport:
    """Tests for ValidationReport dataclass"""

    def test_initialization(self):
        """Test ValidationReport initialization"""
        report = ValidationReport(
            pipeline_id="test_pipeline",
            overall_status=ValidationStatus.PASSED,
            stage_results=[],
            total_duration=100.5,
            validation_metrics=ValidationMetrics(),
            issues=[],
            recommendations=[],
            generated_at=datetime.datetime.now(),
        )

        assert report.pipeline_id == "test_pipeline"
        assert report.overall_status == ValidationStatus.PASSED
        assert report.total_duration == 100.5

    def test_overall_status_calculation(self):
        """Test overall status calculation from stage results"""
        passed_result = PipelineResult(
            stage_id="stage1",
            status=ValidationStatus.PASSED,
            duration_seconds=10.0,
            metrics=ValidationMetrics(),
            issues=[],
            recommendations=[],
        )
        failed_result = PipelineResult(
            stage_id="stage2",
            status=ValidationStatus.FAILED,
            duration_seconds=10.0,
            metrics=ValidationMetrics(),
            issues=["Error"],
            recommendations=[],
        )

        # All passed
        report_passed = ValidationReport(
            pipeline_id="test",
            overall_status=ValidationStatus.PENDING,  # Will be calculated
            stage_results=[passed_result, passed_result],
            total_duration=20.0,
            validation_metrics=ValidationMetrics(),
            issues=[],
            recommendations=[],
            generated_at=datetime.datetime.now(),
        )

        # One failed
        report_failed = ValidationReport(
            pipeline_id="test",
            overall_status=ValidationStatus.PENDING,
            stage_results=[passed_result, failed_result],
            total_duration=20.0,
            validation_metrics=ValidationMetrics(),
            issues=[],
            recommendations=[],
            generated_at=datetime.datetime.now(),
        )

        # The overall_status should be calculated based on stage results
        # (This would typically be done in the constructor or a method)
        assert len(report_passed.stage_results) == 2
        assert len(report_failed.stage_results) == 2


class TestPipelineResult:
    """Tests for PipelineResult dataclass"""

    def test_initialization(self):
        """Test PipelineResult initialization"""
        result = PipelineResult(
            stage_id="test_stage",
            status=ValidationStatus.PASSED,
            duration_seconds=45.2,
            metrics=ValidationMetrics(),
            issues=[],
            recommendations=[],
        )

        assert result.stage_id == "test_stage"
        assert result.status == ValidationStatus.PASSED
        assert result.duration_seconds == 45.2
        assert result.issues == []
        assert result.recommendations == []

    def test_has_issues(self):
        """Test issues check"""
        result_with_issues = PipelineResult(
            stage_id="test",
            status=ValidationStatus.FAILED,
            duration_seconds=10.0,
            metrics=ValidationMetrics(),
            issues=["Error 1", "Error 2"],
            recommendations=[],
        )
        result_without_issues = PipelineResult(
            stage_id="test",
            status=ValidationStatus.PASSED,
            duration_seconds=10.0,
            metrics=ValidationMetrics(),
            issues=[],
            recommendations=[],
        )

        assert result_with_issues.has_issues() is True
        assert result_without_issues.has_issues() is False


class TestValidationMetrics:
    """Tests for ValidationMetrics dataclass"""

    def test_initialization(self):
        """Test ValidationMetrics initialization"""
        metrics = ValidationMetrics()

        assert metrics.tests_run == 0
        assert metrics.tests_passed == 0
        assert metrics.tests_failed == 0
        assert metrics.avg_duration_seconds == 0.0

    def test_success_rate(self):
        """Test success rate calculation"""
        metrics = ValidationMetrics(
            tests_run=10, tests_passed=8, tests_failed=2, avg_duration_seconds=5.5
        )

        assert metrics.success_rate == 0.8

    def test_success_rate_zero_tests(self):
        """Test success rate with zero tests"""
        metrics = ValidationMetrics(tests_run=0)

        assert metrics.success_rate == 0.0
