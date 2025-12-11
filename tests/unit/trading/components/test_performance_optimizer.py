"""Tests for Performance Optimization System component."""

import time
from unittest.mock import Mock, patch

import pytest

from ztb.trading.performance_optimizer import (
    CPUOptimizer,
    LatencyOptimizer,
    MemoryOptimizer,
    OptimizationResult,
    PerformanceOptimizationSystem,
    PerformanceTarget,
    SystemPerformanceMetrics,
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
def performance_optimizer(mock_integration_manager):
    """Performance Optimization System instance"""
    return PerformanceOptimizationSystem(mock_integration_manager)


@pytest.fixture
def sample_performance_target():
    """Sample performance target"""
    return PerformanceTarget(
        latency_ms=100.0, memory_gb=4.0, cpu_percent=80.0, throughput_ops=1000
    )


@pytest.fixture
def sample_performance_metrics():
    """Sample performance metrics"""
    return SystemPerformanceMetrics(
        avg_latency_ms=50.0,
        p95_latency_ms=75.0,
        p99_latency_ms=90.0,
        max_latency_ms=95.0,
        memory_usage_gb=2.5,
        cpu_usage_percent=60.0,
        operations_per_second=800.0,
    )


class TestPerformanceOptimizationSystemInitialization:
    """Initialization tests for Performance Optimization System"""

    def test_initialization(
        self,
        performance_optimizer: PerformanceOptimizationSystem,
        mock_integration_manager,
    ):
        """Test successful initialization"""
        assert performance_optimizer.integration_manager == mock_integration_manager
        assert isinstance(performance_optimizer.latency_optimizer, LatencyOptimizer)
        assert isinstance(performance_optimizer.memory_optimizer, MemoryOptimizer)
        assert isinstance(performance_optimizer.cpu_optimizer, CPUOptimizer)
        assert performance_optimizer.monitoring_thread is None
        assert performance_optimizer.is_monitoring is False
        assert performance_optimizer.optimization_results == []
        assert performance_optimizer.targets.latency_ms == 100.0
        assert performance_optimizer.targets.memory_gb == 4.0
        assert performance_optimizer.targets.cpu_percent == 80.0


class TestPerformanceOptimizationSystemOperations:
    """Operation tests for Performance Optimization System"""

    def test_run_comprehensive_optimization(
        self, performance_optimizer: PerformanceOptimizationSystem
    ):
        """Test comprehensive optimization execution"""
        with patch.object(
            performance_optimizer.latency_optimizer, "optimize_data_processing"
        ) as mock_latency, patch.object(
            performance_optimizer.latency_optimizer, "optimize_signal_processing"
        ) as mock_signal, patch.object(
            performance_optimizer.memory_optimizer, "optimize_memory_allocation"
        ) as mock_memory, patch.object(
            performance_optimizer.cpu_optimizer, "optimize_cpu_utilization"
        ) as mock_cpu, patch.object(
            performance_optimizer.latency_optimizer, "profile_critical_path"
        ) as mock_profile:
            # Mock optimization results
            mock_latency.return_value = OptimizationResult(
                optimization_type="data_processing",
                before_metrics=SystemPerformanceMetrics(avg_latency_ms=100.0),
                after_metrics=SystemPerformanceMetrics(avg_latency_ms=80.0),
                improvement_percent=20.0,
                success=True,
            )

            mock_signal.return_value = OptimizationResult(
                optimization_type="signal_processing",
                before_metrics=SystemPerformanceMetrics(avg_latency_ms=120.0),
                after_metrics=SystemPerformanceMetrics(avg_latency_ms=90.0),
                improvement_percent=25.0,
                success=True,
            )

            mock_memory.return_value = {
                "optimizations_applied": ["pool_optimization"],
                "success": True,
            }
            mock_cpu.return_value = {
                "optimizations_applied": ["thread_pool"],
                "success": True,
            }
            mock_profile.return_value = {"bottlenecks": [], "avg_latency_ms": 85.0}

            result = performance_optimizer.run_comprehensive_optimization()

            assert "results" in result
            assert "summary" in result
            assert result["summary"]["successful_optimizations"] == 2
            assert result["summary"]["total_optimizations"] == 5
            assert len(performance_optimizer.optimization_results) == 2

    def test_start_performance_monitoring(
        self, performance_optimizer: PerformanceOptimizationSystem
    ):
        """Test starting performance monitoring"""
        result = performance_optimizer.start_performance_monitoring()

        assert result is True
        assert performance_optimizer.is_monitoring is True
        assert performance_optimizer.monitoring_thread is not None
        assert performance_optimizer.monitoring_thread.is_alive()

        # Clean up
        performance_optimizer.stop_performance_monitoring()

    def test_stop_performance_monitoring(
        self, performance_optimizer: PerformanceOptimizationSystem
    ):
        """Test stopping performance monitoring"""
        performance_optimizer.start_performance_monitoring()
        assert performance_optimizer.is_monitoring is True

        result = performance_optimizer.stop_performance_monitoring()

        assert result is True
        assert performance_optimizer.is_monitoring is False

    def test_get_performance_report(
        self, performance_optimizer: PerformanceOptimizationSystem
    ):
        """Test getting performance report"""
        # Add some mock optimization results
        performance_optimizer.optimization_results = [
            OptimizationResult(
                optimization_type="latency",
                before_metrics=SystemPerformanceMetrics(avg_latency_ms=100.0),
                after_metrics=SystemPerformanceMetrics(avg_latency_ms=80.0),
                improvement_percent=20.0,
                success=True,
            )
        ]

        with patch.object(
            performance_optimizer.memory_optimizer,
            "analyze_memory_usage",
            return_value={"current_memory": {"rss_gb": 2.5}},
        ), patch.object(
            performance_optimizer.cpu_optimizer,
            "analyze_cpu_usage",
            return_value={"current_cpu": {"cpu_percent": 60.0}},
        ):
            report = performance_optimizer.get_performance_report()

            assert "current_performance" in report
            assert "optimization_summary" in report
            assert "targets" in report
            assert "optimization_details" in report
            assert report["optimization_summary"]["successful_optimizations"] == 1
            assert report["optimization_summary"]["avg_improvement_percent"] == 20.0

    def test_benchmark_system_performance(
        self, performance_optimizer: PerformanceOptimizationSystem
    ):
        """Test system performance benchmarking"""
        with patch.object(
            performance_optimizer.latency_optimizer, "measure_operation_latency"
        ) as mock_measure, patch(
            "time.time", side_effect=[0, 10]
        ):  # 10 second benchmark
            mock_measure.return_value = (50.0, None)  # 50ms latency

            result = performance_optimizer.benchmark_system_performance()

            assert "latency" in result
            assert "throughput" in result
            assert "memory" in result
            assert "cpu" in result
            assert "target_achievement" in result

            # Verify latency benchmark
            assert result["latency"]["avg_latency_ms"] == 50.0
            assert result["latency"]["within_target"] is True  # 50ms < 100ms target

            # Verify throughput benchmark
            assert (
                result["throughput"]["throughput_ops_sec"] == 0.1
            )  # 1 operation per 10 seconds
            assert result["throughput"]["within_target"] is False  # 0.1 < 1000 target


class TestLatencyOptimizer:
    """Tests for LatencyOptimizer"""

    def test_initialization(self, mock_integration_manager):
        """Test LatencyOptimizer initialization"""
        optimizer = LatencyOptimizer(mock_integration_manager)

        assert optimizer.integration_manager == mock_integration_manager
        assert optimizer.latency_measurements == []
        assert optimizer.profiler is not None

    def test_measure_operation_latency(self, mock_integration_manager):
        """Test operation latency measurement"""
        optimizer = LatencyOptimizer(mock_integration_manager)

        def test_operation():
            time.sleep(0.01)  # 10ms operation
            return "result"

        latency, result = optimizer.measure_operation_latency(test_operation)

        assert latency > 0  # Should measure some latency
        assert result == "result"
        assert len(optimizer.latency_measurements) == 1
        assert optimizer.latency_measurements[0] == latency

    def test_profile_critical_path(self, mock_integration_manager):
        """Test critical path profiling"""
        optimizer = LatencyOptimizer(mock_integration_manager)

        # Mock the critical operations
        with patch.object(optimizer, "_execute_critical_operations"), patch(
            "io.StringIO"
        ) as mock_stringio, patch("pstats.Stats") as mock_stats:
            mock_stats_instance = Mock()
            mock_stats_instance.print_stats.return_value = None
            mock_stats.return_value = mock_stats_instance

            mock_stringio_instance = Mock()
            mock_stringio_instance.getvalue.return_value = "profile output"
            mock_stringio.return_value = mock_stringio_instance

            result = optimizer.profile_critical_path()

            assert "profile_output" in result
            assert "bottlenecks" in result
            assert "total_measurements" in result

    def test_optimize_data_processing(self, mock_integration_manager):
        """Test data processing optimization"""
        optimizer = LatencyOptimizer(mock_integration_manager)

        with patch.object(
            optimizer, "_measure_data_processing_performance"
        ) as mock_measure, patch.object(
            optimizer, "_optimize_async_processing", return_value=True
        ), patch.object(
            optimizer, "_optimize_data_structures", return_value=True
        ), patch.object(
            optimizer, "_optimize_memory_pool", return_value=True
        ), patch.object(optimizer, "_optimize_caching", return_value=True):
            mock_measure.side_effect = [
                SystemPerformanceMetrics(avg_latency_ms=100.0),  # Before
                SystemPerformanceMetrics(avg_latency_ms=80.0),  # After
            ]

            result = optimizer.optimize_data_processing()

            assert isinstance(result, OptimizationResult)
            assert result.optimization_type == "data_processing"
            assert result.improvement_percent == 20.0
            assert result.success is True
            assert "async_processing" in result.details["optimizations_applied"]

    def test_optimize_signal_processing(self, mock_integration_manager):
        """Test signal processing optimization"""
        optimizer = LatencyOptimizer(mock_integration_manager)

        with patch.object(
            optimizer, "_measure_signal_processing_performance"
        ) as mock_measure, patch.object(
            optimizer, "_optimize_parallel_processing", return_value=True
        ), patch.object(
            optimizer, "_optimize_algorithms", return_value=True
        ), patch.object(optimizer, "_optimize_io_operations", return_value=True):
            mock_measure.side_effect = [
                SystemPerformanceMetrics(avg_latency_ms=120.0),  # Before
                SystemPerformanceMetrics(avg_latency_ms=90.0),  # After
            ]

            result = optimizer.optimize_signal_processing()

            assert isinstance(result, OptimizationResult)
            assert result.optimization_type == "signal_processing"
            assert result.improvement_percent == 25.0
            assert result.success is True

    def test_optimize_memory_usage(self, mock_integration_manager):
        """Test memory usage optimization"""
        optimizer = LatencyOptimizer(mock_integration_manager)

        with patch.object(
            optimizer, "_measure_memory_performance"
        ) as mock_measure, patch.object(
            optimizer, "_optimize_garbage_collection", return_value=True
        ), patch.object(optimizer, "_optimize_memory_structures", return_value=True):
            mock_measure.side_effect = [
                SystemPerformanceMetrics(memory_usage_gb=3.0),  # Before
                SystemPerformanceMetrics(memory_usage_gb=2.5),  # After
            ]

            result = optimizer.optimize_memory_usage()

            assert isinstance(result, OptimizationResult)
            assert result.optimization_type == "memory_usage"
            assert result.improvement_percent == 16.67  # (3.0-2.5)/3.0 * 100
            assert result.success is True


class TestMemoryOptimizer:
    """Tests for MemoryOptimizer"""

    def test_initialization(self, mock_integration_manager):
        """Test MemoryOptimizer initialization"""
        optimizer = MemoryOptimizer(mock_integration_manager)

        assert optimizer.integration_manager == mock_integration_manager
        assert optimizer.memory_snapshots == []
        assert optimizer.object_refs is not None

    def test_analyze_memory_usage(self, mock_integration_manager):
        """Test memory usage analysis"""
        optimizer = MemoryOptimizer(mock_integration_manager)

        with patch("psutil.Process") as mock_process:
            mock_process_instance = Mock()
            mock_process_instance.memory_info.return_value = Mock(
                rss=2 * 1024**3, vms=3 * 1024**3
            )  # 2GB RSS, 3GB VMS
            mock_process_instance.cpu_percent.return_value = 50.0
            mock_process_instance.num_threads.return_value = 8
            mock_process_instance.open_files.return_value = []
            mock_process_instance.connections.return_value = []
            mock_process.return_value = mock_process_instance

            result = optimizer.analyze_memory_usage()

            assert "current_memory" in result
            assert result["current_memory"]["rss_gb"] == 2.0
            assert result["current_memory"]["vms_gb"] == 3.0
            assert result["current_memory"]["cpu_percent"] == 50.0
            assert len(optimizer.memory_snapshots) == 1

    def test_optimize_memory_allocation(self, mock_integration_manager):
        """Test memory allocation optimization"""
        optimizer = MemoryOptimizer(mock_integration_manager)

        with patch.object(
            optimizer, "_optimize_object_pools", return_value=True
        ), patch.object(
            optimizer, "_reduce_memory_fragmentation", return_value=True
        ), patch.object(
            optimizer, "_optimize_cache_memory", return_value=True
        ), patch.object(
            optimizer, "_optimize_data_structure_memory", return_value=True
        ):
            result = optimizer.optimize_memory_allocation()

            assert "optimizations_applied" in result
            assert "success" in result
            assert len(result["optimizations_applied"]) == 4
            assert result["success"] is True

    def test_implement_memory_monitoring(self, mock_integration_manager):
        """Test memory monitoring implementation"""
        optimizer = MemoryOptimizer(mock_integration_manager)

        result = optimizer.implement_memory_monitoring()

        assert result is True
        # Note: In a real test, we would verify the thread is created and running


class TestCPUOptimizer:
    """Tests for CPUOptimizer"""

    def test_initialization(self, mock_integration_manager):
        """Test CPUOptimizer initialization"""
        optimizer = CPUOptimizer(mock_integration_manager)

        assert optimizer.integration_manager == mock_integration_manager
        assert optimizer.cpu_measurements == []

    def test_analyze_cpu_usage(self, mock_integration_manager):
        """Test CPU usage analysis"""
        optimizer = CPUOptimizer(mock_integration_manager)

        with patch("psutil.Process") as mock_process, patch(
            "psutil.cpu_count", return_value=8
        ):
            mock_process_instance = Mock()
            mock_process_instance.cpu_percent.return_value = 65.0
            mock_process_instance.cpu_times.return_value = Mock(user=100.0, system=50.0)
            mock_process_instance.num_threads.return_value = 12
            mock_process.return_value = mock_process_instance

            result = optimizer.analyze_cpu_usage()

            assert "current_cpu" in result
            assert result["current_cpu"]["cpu_percent"] == 65.0
            assert result["current_cpu"]["num_threads"] == 12
            assert len(optimizer.cpu_measurements) == 1

    def test_optimize_cpu_utilization(self, mock_integration_manager):
        """Test CPU utilization optimization"""
        optimizer = CPUOptimizer(mock_integration_manager)

        with patch.object(
            optimizer, "_optimize_thread_pools", return_value=True
        ), patch.object(
            optimizer, "_parallelize_computations", return_value=True
        ), patch.object(
            optimizer, "_optimize_cpu_affinity", return_value=True
        ), patch.object(optimizer, "_reduce_idle_cpu", return_value=True):
            result = optimizer.optimize_cpu_utilization()

            assert "optimizations_applied" in result
            assert "success" in result
            assert len(result["optimizations_applied"]) == 4
            assert result["success"] is True


class TestPerformanceTarget:
    """Tests for PerformanceTarget dataclass"""

    def test_initialization(self):
        """Test PerformanceTarget initialization"""
        target = PerformanceTarget()

        assert target.latency_ms == 100.0
        assert target.memory_gb == 4.0
        assert target.cpu_percent == 80.0
        assert target.throughput_ops == 1000

    def test_custom_initialization(self):
        """Test PerformanceTarget with custom values"""
        target = PerformanceTarget(
            latency_ms=50.0, memory_gb=8.0, cpu_percent=90.0, throughput_ops=2000
        )

        assert target.latency_ms == 50.0
        assert target.memory_gb == 8.0
        assert target.cpu_percent == 90.0
        assert target.throughput_ops == 2000


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass"""

    def test_initialization(self):
        """Test PerformanceMetrics initialization"""
        metrics = SystemPerformanceMetrics()

        assert metrics.avg_latency_ms == 0.0
        assert metrics.memory_usage_gb == 0.0
        assert metrics.cpu_usage_percent == 0.0
        assert metrics.operations_per_second == 0.0

    def test_custom_initialization(self, sample_performance_metrics):
        """Test PerformanceMetrics with custom values"""
        assert sample_performance_metrics.avg_latency_ms == 50.0
        assert sample_performance_metrics.memory_usage_gb == 2.5
        assert sample_performance_metrics.cpu_usage_percent == 60.0
        assert sample_performance_metrics.operations_per_second == 800.0


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass"""

    def test_initialization(self):
        """Test OptimizationResult initialization"""
        before = SystemPerformanceMetrics(avg_latency_ms=100.0)
        after = SystemPerformanceMetrics(avg_latency_ms=80.0)

        result = OptimizationResult(
            optimization_type="test",
            before_metrics=before,
            after_metrics=after,
            improvement_percent=20.0,
            success=True,
        )

        assert result.optimization_type == "test"
        assert result.improvement_percent == 20.0
        assert result.success is True
        assert result.improvement_description == "Improved by 20.0%"

    def test_improvement_description_improved(self):
        """Test improvement description for improvement"""
        result = OptimizationResult(
            optimization_type="test",
            before_metrics=SystemPerformanceMetrics(),
            after_metrics=SystemPerformanceMetrics(),
            improvement_percent=15.5,
            success=True,
        )

        assert "Improved by 15.5%" in result.improvement_description

    def test_improvement_description_degraded(self):
        """Test improvement description for degradation"""
        result = OptimizationResult(
            optimization_type="test",
            before_metrics=SystemPerformanceMetrics(),
            after_metrics=SystemPerformanceMetrics(),
            improvement_percent=-10.0,
            success=False,
        )

        assert "Degraded by 10.0%" in result.improvement_description
