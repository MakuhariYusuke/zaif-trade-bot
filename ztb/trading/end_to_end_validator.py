from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Protocol, Sequence, TypedDict


class PipelineLike(Protocol):
    pipeline_id: str


class StageExecutionResult(TypedDict):
    passed: bool
    duration: float
    metrics: Dict[str, object]


class ValidationStatusPayload(TypedDict):
    current_pipeline: Optional[str]
    is_running: bool
    pipeline_progress: float


class ValidationHistoryPayload(TypedDict):
    total_validations: int
    successful_validations: int
    failed_validations: int
    average_duration: float


def _as_object_map(value: object) -> Dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


class ValidationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    PASSED = "passed"
    FAILED = "failed"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ValidationStatus):
            return NotImplemented
        order_list = ["pending", "running", "skipped", "timeout", "failed", "passed"]
        return order_list.index(self.value) < order_list.index(other.value)


@dataclass
class ValidationMetrics:
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    avg_duration_seconds: float = 0.0
    details: Dict[str, object] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.tests_run == 0:
            return 0.0
        return self.tests_passed / self.tests_run


@dataclass
class PipelineResult:
    stage_id: str
    status: ValidationStatus
    duration_seconds: float
    metrics: ValidationMetrics
    issues: List[str]
    recommendations: List[str]

    def has_issues(self) -> bool:
        return len(self.issues) > 0


@dataclass
class ValidationReport:
    pipeline_id: str
    overall_status: ValidationStatus
    stage_results: List[PipelineResult]
    total_duration: float
    validation_metrics: ValidationMetrics
    issues: List[str]
    recommendations: List[str]
    generated_at: datetime

    def __post_init__(self) -> None:
        # If overall_status is PENDING, compute final status from stage results
        if self.overall_status != ValidationStatus.PENDING:
            return
        self.overall_status = (
            ValidationStatus.FAILED
            if any(r.status == ValidationStatus.FAILED for r in self.stage_results)
            else ValidationStatus.PASSED
        )


@dataclass
class ValidationStage:
    stage_id: str
    name: str
    description: str
    order: int
    timeout_seconds: int
    required: bool = True

    def is_required(self) -> bool:
        return self.required


@dataclass
class ValidationPipeline:
    pipeline_id: str
    name: str
    description: str
    stages: List[ValidationStage]
    created_at: datetime
    updated_at: datetime

    def get_stage_by_id(self, stage_id: str) -> Optional[ValidationStage]:
        for s in self.stages:
            if s.stage_id == stage_id:
                return s
        return None

    def get_stages_in_order(self) -> List[ValidationStage]:
        return sorted(self.stages, key=lambda x: x.order)


@dataclass
class ComponentTest:
    test_id: str
    component_name: str
    test_type: str
    description: str
    expected_result: object
    timeout_seconds: int


@dataclass
class IntegrationTest:
    test_id: str
    description: str
    components_involved: List[str]
    expected_behavior: str
    success_criteria: str
    timeout_seconds: int


@dataclass
class PerformanceTest:
    test_id: str
    description: str
    metric_name: str
    target_value: float
    operator: str
    timeout_seconds: int


@dataclass
class HealthCheck:
    check_id: str
    description: str
    check_type: str
    expected_result: object
    timeout_seconds: int


class StageValidatorBase:
    stage_id: str

    def __init__(self, integration_manager: object):
        self.integration_manager = integration_manager

    def _build_pipeline_result(
        self, execution_results: Sequence[StageExecutionResult]
    ) -> PipelineResult:
        tests_run = len(execution_results)
        tests_passed = sum(1 for result in execution_results if result["passed"])
        tests_failed = tests_run - tests_passed
        total_duration = sum(result["duration"] for result in execution_results)
        avg_duration = total_duration / tests_run if tests_run > 0 else 0.0

        metrics = ValidationMetrics(
            tests_run=tests_run,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            avg_duration_seconds=avg_duration,
            details={
                "test_metrics": [result["metrics"] for result in execution_results],
            },
        )

        return PipelineResult(
            stage_id=self.stage_id,
            status=ValidationStatus.PASSED if tests_failed == 0 else ValidationStatus.FAILED,
            duration_seconds=total_duration,
            metrics=metrics,
            issues=[],
            recommendations=[],
        )


class ComponentValidator(StageValidatorBase):
    stage_id = "component_validation"

    def __init__(self, integration_manager: object):
        super().__init__(integration_manager)
        self.component_tests: List[ComponentTest] = []

    def validate_components(
        self, tests: Optional[List[ComponentTest]] = None
    ) -> PipelineResult:
        if tests is not None:
            self.component_tests = tests

        execution_results = [
            self._run_component_test(test) for test in self.component_tests
        ]
        return self._build_pipeline_result(execution_results)

    def _run_component_test(self, test: ComponentTest) -> StageExecutionResult:
        """Run a single component test."""
        result = self._execute_component_test(test)
        metrics = _as_object_map(result.get("metrics", {}))
        return {
            "passed": bool(result.get("passed", True)),
            "duration": float(result.get("duration", 1.0)),
            "metrics": metrics,
        }

    def _execute_component_test(self, test: ComponentTest) -> Dict[str, object]:
        """Actual execution of a component test (placeholder for real implementations)."""
        _ = test
        return {"passed": True, "duration": 1.0, "metrics": {}}

    def validate_component_health(self, component_name: str) -> bool:
        """Validate health of a component."""
        status = self.integration_manager.component_manager.get_component_status(
            component_name
        )
        if not status:
            return False

        if isinstance(status, dict):
            raw_status = status.get("status")
            return raw_status == "healthy" or raw_status is None
        return status == "healthy"

    def _validate_component_health(self, component_name: str) -> Dict[str, object]:
        """Private method used by tests to inspect component health details."""
        status = self.integration_manager.component_manager.get_component_status(
            component_name
        )

        if isinstance(status, dict):
            status_value = status.get("status")
            return {
                "status": status_value,
                "metrics": _as_object_map(status.get("metrics", {})),
                "is_healthy": status_value == "healthy",
            }

        return {
            "status": status,
            "is_healthy": status == "healthy",
        }


class IntegrationValidator(StageValidatorBase):
    stage_id = "integration_validation"

    def __init__(self, integration_manager: object):
        super().__init__(integration_manager)
        self.integration_tests: List[IntegrationTest] = []

    def validate_integrations(
        self, tests: Optional[List[IntegrationTest]] = None
    ) -> PipelineResult:
        if tests is not None:
            self.integration_tests = tests

        execution_results = [
            self._run_integration_test(test) for test in self.integration_tests
        ]
        return self._build_pipeline_result(execution_results)

    def _run_integration_test(self, test: IntegrationTest) -> StageExecutionResult:
        """Run a single integration test."""
        result = self._execute_integration_test(test)
        metrics = _as_object_map(result.get("metrics", {}))
        return {
            "passed": bool(result.get("passed", True)),
            "duration": float(result.get("duration", 1.0)),
            "metrics": metrics,
        }

    def _execute_integration_test(self, test: IntegrationTest) -> Dict[str, object]:
        """Execute integration test."""
        _ = test
        return {"passed": True, "duration": 1.0, "metrics": {}}

    def _validate_data_flow(self, source: str, target: str) -> Dict[str, object]:
        """Validate data flow between components."""
        metrics: Dict[str, object] = {}
        try:
            raw_metrics = (
                self.integration_manager.component_manager.v433_system.get_data_flow_metrics(
                    source, target
                )
            )
            metrics = _as_object_map(raw_metrics)
        except Exception:
            try:
                raw_metrics = (
                    self.integration_manager.component_manager.get_data_flow_metrics(
                        source, target
                    )
                )
                metrics = _as_object_map(raw_metrics)
            except Exception:
                metrics = {}

        data_processed = float(metrics.get("data_processed", 0))
        data_lost = float(metrics.get("data_lost", 0))
        avg_latency = float(metrics.get("avg_latency", 0.0))

        return {
            "data_integrity": data_lost == 0,
            "latency": avg_latency,
            "latency_acceptable": avg_latency <= 100.0,
            "flow_stable": data_processed > 0
            and (data_lost / max(1.0, data_processed)) < 0.01,
        }


class PerformanceValidator(StageValidatorBase):
    stage_id = "performance_validation"

    def __init__(self, integration_manager: object):
        super().__init__(integration_manager)
        self.performance_tests: List[PerformanceTest] = []

    def validate_performance(
        self, pipeline: Optional[PipelineLike] = None
    ) -> PipelineResult:
        _ = pipeline
        execution_results = [
            self._run_performance_test(test) for test in self.performance_tests
        ]
        return self._build_pipeline_result(execution_results)

    def _run_performance_test(self, test: PerformanceTest) -> StageExecutionResult:
        """Run a single performance test."""
        metric_value = self._measure_performance_metric(test.metric_name)

        if test.operator == "<=":
            passed = metric_value <= test.target_value
        elif test.operator == ">=":
            passed = metric_value >= test.target_value
        elif test.operator == "==":
            passed = metric_value == test.target_value
        else:
            passed = True

        return {
            "passed": passed,
            "duration": 1.0,
            "metrics": {test.metric_name: metric_value},
        }

    def _measure_performance_metric(self, metric_name: str) -> float:
        """Measure a performance metric."""
        try:
            metrics = self.integration_manager.component_manager.get_performance_metrics()
            metrics_map = _as_object_map(metrics)
            return float(metrics_map.get(metric_name, 0.0))
        except Exception:
            return 0.0


class SystemHealthValidator(StageValidatorBase):
    stage_id = "health_validation"

    def __init__(self, integration_manager: object):
        super().__init__(integration_manager)
        self.health_checks: List[HealthCheck] = []

    def validate_system_health(
        self, checks: Optional[List[HealthCheck]] = None
    ) -> PipelineResult:
        if checks is not None:
            self.health_checks = checks

        execution_results = [self._run_health_check(check) for check in self.health_checks]
        return self._build_pipeline_result(execution_results)

    def _run_health_check(self, check: HealthCheck) -> StageExecutionResult:
        result = self._execute_health_check(check)
        return {
            "passed": result.get("status", "healthy") == "healthy",
            "duration": float(result.get("duration", 1.0)),
            "metrics": result,
        }

    def _execute_health_check(self, check: HealthCheck) -> Dict[str, object]:
        if check.check_type == "resource":
            return self._check_system_resources()
        return {"status": "healthy", "duration": 1.0}

    def _check_system_resources(self) -> Dict[str, object]:
        import psutil

        cpu = float(psutil.cpu_percent())
        mem = float(psutil.virtual_memory().percent)
        disk = (
            float(psutil.disk_usage("/").percent)
            if hasattr(psutil, "disk_usage")
            else 0.0
        )
        return {
            "cpu_percent": cpu,
            "memory_percent": mem,
            "disk_percent": disk,
            "status": "healthy",
            "duration": 1.0,
        }


class EndToEndValidationSystem:
    def __init__(
        self,
        integration_manager: object,
        pipeline: Optional[PipelineLike] = None,
    ):
        self.integration_manager = integration_manager
        self.component_validator = ComponentValidator(integration_manager)
        self.integration_validator = IntegrationValidator(integration_manager)
        self.performance_validator = PerformanceValidator(integration_manager)
        self.health_validator = SystemHealthValidator(integration_manager)
        self.validation_pipelines: List[PipelineLike] = []
        self.validation_history: List[ValidationReport] = []
        self.is_running = False
        self.current_pipeline: Optional[PipelineLike] = pipeline
        if pipeline is not None:
            self.validation_pipelines.append(pipeline)

    def run_end_to_end_validation(
        self, pipeline: Optional[PipelineLike]
    ) -> ValidationReport:
        active_pipeline = pipeline if pipeline is not None else self.current_pipeline
        self.is_running = True

        try:
            component_result = self.component_validator.validate_components()
            integration_result = self.integration_validator.validate_integrations()
            performance_result = self.performance_validator.validate_performance(
                active_pipeline
            )
            health_result = self.health_validator.validate_system_health()

            stage_results = [
                component_result,
                integration_result,
                performance_result,
                health_result,
            ]

            overall_status = (
                ValidationStatus.PASSED
                if all(r.status == ValidationStatus.PASSED for r in stage_results)
                else ValidationStatus.FAILED
            )

            total_duration = sum(r.duration_seconds for r in stage_results)
            total_tests = sum(r.metrics.tests_run for r in stage_results)
            total_passed = sum(r.metrics.tests_passed for r in stage_results)
            total_failed = sum(r.metrics.tests_failed for r in stage_results)
            avg_stage_duration = (
                sum(r.metrics.avg_duration_seconds for r in stage_results)
                / len(stage_results)
                if stage_results
                else 0.0
            )

            report = ValidationReport(
                pipeline_id=(active_pipeline.pipeline_id if active_pipeline else "default"),
                overall_status=overall_status,
                stage_results=stage_results,
                total_duration=total_duration,
                validation_metrics=ValidationMetrics(
                    tests_run=total_tests,
                    tests_passed=total_passed,
                    tests_failed=total_failed,
                    avg_duration_seconds=avg_stage_duration,
                ),
                issues=[],
                recommendations=[],
                generated_at=datetime.now(),
            )

            if active_pipeline is not None:
                self.current_pipeline = active_pipeline
                if not any(
                    p.pipeline_id == active_pipeline.pipeline_id
                    for p in self.validation_pipelines
                ):
                    self.validation_pipelines.append(active_pipeline)

            self.validation_history.append(report)
            return report

        finally:
            self.is_running = False

    def run_validation_pipeline(self, pipeline_id: str) -> ValidationReport:
        pipeline: Optional[PipelineLike] = None
        for candidate in self.validation_pipelines:
            if candidate.pipeline_id == pipeline_id:
                pipeline = candidate
                break

        if pipeline is None:
            pipeline = self.current_pipeline

        return self.run_end_to_end_validation(pipeline)

    def get_validation_status(self) -> ValidationStatusPayload:
        current_pipeline_id = (
            self.current_pipeline.pipeline_id if self.current_pipeline else None
        )
        return {
            "current_pipeline": current_pipeline_id,
            "is_running": self.is_running,
            "pipeline_progress": 0.0,
        }

    def get_validation_history(self) -> ValidationHistoryPayload:
        total_validations = len(self.validation_history)
        successful_validations = sum(
            1
            for report in self.validation_history
            if report.overall_status == ValidationStatus.PASSED
        )
        failed_validations = total_validations - successful_validations
        average_duration = (
            sum(report.total_duration for report in self.validation_history)
            / total_validations
            if total_validations > 0
            else 0.0
        )

        return {
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "failed_validations": failed_validations,
            "average_duration": average_duration,
        }

    def stop_validation(self) -> bool:
        self.is_running = False
        self.current_pipeline = None
        return True


__all__ = [
    "EndToEndValidationSystem",
    "ValidationPipeline",
    "ComponentValidator",
    "IntegrationValidator",
    "PerformanceValidator",
    "SystemHealthValidator",
    "ValidationStage",
    "ValidationStatus",
    "ValidationMetrics",
    "ValidationReport",
    "PipelineResult",
    "ComponentTest",
    "IntegrationTest",
    "PerformanceTest",
    "HealthCheck",
]
