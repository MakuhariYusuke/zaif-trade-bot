#!/usr/bin/env python3
"""
Tests for Common Analysis Components

Tests for data loaders, analysis interfaces, and path management components.
"""

import json
from typing import Any, Dict

import pytest

from ztb.analysis.common.analysis_interfaces import (
    AnalysisPipeline,
    AnalysisSummary,
    AnalysisValidator,
    BaseAnalyzer,
    ComparativeAnalysisResult,
)
from ztb.analysis.common.data_loaders import (
    AnalysisDataLoader,
    BacktestDataLoader,
    DataLoadError,
    TrainingDataLoader,
)
from ztb.analysis.common.path_manager import (
    AnalysisPathManager,
    PathManagerError,
    get_path_manager,
)
from ztb.analysis.common.types import (
    ExtendedRiskStatus,
    PerformanceMonitorProtocol,
    PositionMonitorResult,
    RiskProfile,
    RiskProfileLimits,
    RiskStatusReport,
    ThresholdManagerProtocol,
    TriggerStatus,
)
from ztb.analysis.reporting.display_manager import AnalysisDisplayManager


class TestBacktestDataLoader:
    """Tests for BacktestDataLoader."""

    def test_load_latest_backtest_results_success(self, tmp_path):
        """Test successful loading of backtest results."""
        # Create test directory structure
        experiment_dir = tmp_path / "backtest_experiments" / "test_experiment"
        result_dir = experiment_dir / "result_001"
        result_dir.mkdir(parents=True)

        # Create test data files
        backtest_data = {"total_return": 0.15, "sharpe_ratio": 1.2}
        with open(result_dir / "backtest_results.json", "w") as f:
            json.dump(backtest_data, f)

        portfolio_data = "step,portfolio_value\n1,100000\n2,101000\n"
        with open(result_dir / "portfolio_values.csv", "w") as f:
            f.write(portfolio_data)

        trades_data = "episode,step,action,reward\n1,1,1,0.01\n"
        with open(result_dir / "trades_history.csv", "w") as f:
            f.write(trades_data)

        loader = BacktestDataLoader(tmp_path / "backtest_experiments")
        results = loader.load_latest_backtest_results("test_experiment")

        assert "backtestresults" in results
        assert results["backtestresults"]["total_return"] == 0.15
        assert "portfoliovalues" in results
        assert "tradeshistory" in results

    def test_load_latest_backtest_results_no_directory(self, tmp_path):
        """Test loading when experiment directory doesn't exist."""
        loader = BacktestDataLoader(tmp_path / "backtest_experiments")

        with pytest.raises(DataLoadError):
            loader.load_latest_backtest_results("nonexistent_experiment")

    def test_load_backtest_results_from_path_file(self, tmp_path):
        """Test loading from a specific file path."""
        test_data = {"test": "data"}
        test_file = tmp_path / "test_results.json"
        with open(test_file, "w") as f:
            json.dump(test_data, f)

        loader = BacktestDataLoader()
        results = loader.load_backtest_results_from_path(test_file)

        assert results == test_data


class TestTrainingDataLoader:
    """Tests for TrainingDataLoader."""

    def test_load_training_results_success(self, tmp_path):
        """Test successful loading of training results."""
        training_data = {
            "training_stats": {"total_timesteps": 100000},
            "final_reward": -0.5,
        }

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        results_file = results_dir / "training_report_v441.json"
        with open(results_file, "w") as f:
            json.dump(training_data, f)

        loader = TrainingDataLoader(results_dir)
        results = loader.load_training_results("v441")

        assert results["training_stats"]["total_timesteps"] == 100000
        assert results["final_reward"] == -0.5

    def test_load_training_results_not_found(self, tmp_path):
        """Test loading when training results file doesn't exist."""
        loader = TrainingDataLoader(tmp_path / "results")

        with pytest.raises(DataLoadError):
            loader.load_training_results("nonexistent_version")


class TestAnalysisDataLoader:
    """Tests for AnalysisDataLoader."""

    def test_load_analysis_results_success(self, tmp_path):
        """Test successful loading of analysis results."""
        analysis_data = {"performance": {"accuracy": 0.95}}

        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        results_file = analysis_dir / "test_analysis_results.json"
        with open(results_file, "w") as f:
            json.dump(analysis_data, f)

        loader = AnalysisDataLoader(analysis_dir)
        results = loader.load_analysis_results("test_analysis")

        assert results["performance"]["accuracy"] == 0.95


class MockAnalyzer(BaseAnalyzer[Dict[str, Any], Dict[str, Any]]):
    """Mock analyzer for testing."""

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": data.get("input", "default")}


class TestAnalysisInterfaces:
    """Tests for analysis interfaces."""

    def test_base_analyzer_abstract(self):
        """Test that BaseAnalyzer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAnalyzer()

    def test_mock_analyzer(self):
        """Test mock analyzer implementation."""
        analyzer = MockAnalyzer("test_analyzer")
        result = analyzer.analyze({"input": "test"})

        assert result["result"] == "test"

    def test_analysis_pipeline(self):
        """Test analysis pipeline execution."""
        analyzer1 = MockAnalyzer("analyzer1")
        analyzer2 = MockAnalyzer("analyzer2")

        pipeline = AnalysisPipeline([analyzer1, analyzer2])
        results = pipeline.execute({"input": "pipeline_test"})

        assert "step_0_MockAnalyzer" in results
        assert "step_1_MockAnalyzer" in results
        assert results["step_0_MockAnalyzer"]["result"] == "pipeline_test"

    def test_analysis_validator_input_validation(self):
        """Test input validation."""
        validator = AnalysisValidator()

        # Valid input
        assert validator.validate_input({"field1": "value1"}, ["field1"])

        # Invalid input - missing field
        assert not validator.validate_input({"field1": "value1"}, ["field1", "field2"])

        # Strict mode
        validator_strict = AnalysisValidator(strict_mode=True)
        with pytest.raises(ValueError):
            validator_strict.validate_input({"field1": "value1"}, ["field1", "field2"])

    def test_analysis_summary_dataclass(self):
        """Test AnalysisSummary dataclass."""
        summary = AnalysisSummary(
            name="test_analysis",
            description="Test analysis description",
            metrics={"accuracy": 0.95},
            warnings=["Warning 1"],
            errors=["Error 1"],
        )

        assert summary.name == "test_analysis"
        assert summary.metrics["accuracy"] == 0.95
        assert len(summary.warnings) == 1
        assert len(summary.errors) == 1

    def test_comparative_analysis_result_dataclass(self):
        """Test ComparativeAnalysisResult dataclass."""
        result = ComparativeAnalysisResult(
            baseline_name="baseline",
            comparison_name="comparison",
            metrics_comparison={"accuracy": {"baseline": 0.9, "comparison": 0.95}},
            summary="Comparison summary",
            recommendations=["Recommendation 1"],
        )

        assert result.baseline_name == "baseline"
        assert result.comparison_name == "comparison"
        assert result.metrics_comparison["accuracy"]["comparison"] == 0.95
        assert len(result.recommendations) == 1

    def test_comparative_analysis_result_default_recommendations(self):
        """Test ComparativeAnalysisResult with default recommendations."""
        result = ComparativeAnalysisResult(
            baseline_name="baseline",
            comparison_name="comparison",
            metrics_comparison={"accuracy": {"baseline": 0.9, "comparison": 0.95}},
            summary="Comparison summary",
        )

        # __post_init__ should initialize empty recommendations list
        assert result.recommendations == []


class TestAnalysisPathManager:
    """Tests for AnalysisPathManager."""

    def test_path_manager_initialization(self, tmp_path):
        """Test path manager initialization."""
        manager = AnalysisPathManager(tmp_path)

        assert manager.get_path("analysis") == tmp_path / "analysis"
        assert manager.get_path("results") == tmp_path / "results"

    def test_get_path_unknown_name(self, tmp_path):
        """Test getting unknown path name."""
        manager = AnalysisPathManager(tmp_path)

        with pytest.raises(PathManagerError):
            manager.get_path("unknown_path")

    def test_resolve_experiment_path(self, tmp_path):
        """Test experiment path resolution."""
        manager = AnalysisPathManager(tmp_path)
        experiment_path = manager.resolve_experiment_path(
            "test_experiment", create_if_missing=True
        )

        assert experiment_path == tmp_path / "backtest_experiments" / "test_experiment"
        assert experiment_path.exists()

    def test_find_latest_experiment_dir(self, tmp_path):
        """Test finding latest experiment directory."""
        import time

        manager = AnalysisPathManager(tmp_path)

        # Create test experiment directories
        exp_base = tmp_path / "backtest_experiments" / "test_exp"
        exp_base.mkdir(parents=True)

        exp1 = exp_base / "run_001"
        exp1.mkdir()
        time.sleep(0.01)  # Ensure different mtime
        exp2 = exp_base / "run_002"
        exp2.mkdir()

        latest = manager.find_latest_experiment_dir("test_exp")
        assert latest == exp2  # exp2 should be more recent

    def test_resolve_output_path(self, tmp_path):
        """Test output path resolution."""
        manager = AnalysisPathManager(tmp_path)
        output_path = manager.resolve_output_path("test_output.txt", "analysis_results")

        expected = tmp_path / "analysis_results" / "test_output.txt"
        assert output_path == expected
        assert expected.parent.exists()

    def test_list_files(self, tmp_path):
        """Test file listing functionality."""
        manager = AnalysisPathManager(tmp_path)

        # Create test files
        test_file1 = tmp_path / "analysis" / "test1.json"
        test_file1.parent.mkdir(parents=True, exist_ok=True)
        test_file1.write_text("{}")

        test_file2 = tmp_path / "analysis" / "subdir" / "test2.json"
        test_file2.parent.mkdir(parents=True, exist_ok=True)
        test_file2.write_text("{}")

        # Test non-recursive
        files = manager.list_files("analysis", "*.json")
        assert len(files) == 1
        assert test_file1 in files

        # Test recursive
        files_recursive = manager.list_files("analysis", "*.json", recursive=True)
        assert len(files_recursive) == 2
        assert test_file1 in files_recursive
        assert test_file2 in files_recursive

    def test_get_path_manager_singleton(self):
        """Test path manager singleton behavior."""
        manager1 = get_path_manager()
        manager2 = get_path_manager()

        assert manager1 is manager2


class TestAnalysisDisplayManager:
    """Tests for AnalysisDisplayManager integration."""

    def test_display_manager_initialization(self, tmp_path):
        """Test AnalysisDisplayManager initialization."""
        display_manager = AnalysisDisplayManager(output_dir=str(tmp_path / "analysis"))
        assert display_manager.output_dir == tmp_path / "analysis"

    def test_display_analysis_results(self, tmp_path, capsys):
        """Test displaying analysis results."""
        display_manager = AnalysisDisplayManager(output_dir=str(tmp_path / "analysis"))

        analysis_results = {
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "win_rate": 0.65,
            "max_drawdown": 0.08,
        }

        display_manager.display_backtest_results(analysis_results)

        # Check that output was generated
        captured = capsys.readouterr()
        assert "BACKTEST RESULTS" in captured.out

    def test_display_with_comparison_table(self, tmp_path, capsys):
        """Test displaying results with comparison table."""
        display_manager = AnalysisDisplayManager(output_dir=str(tmp_path / "analysis"))

        comparisons = [{"return": 0.10, "risk": 0.15}, {"return": 0.15, "risk": 0.12}]
        metric_names = ["return", "risk"]
        titles = ["Baseline", "Improved"]

        display_manager.display_comparison_results(
            comparisons, metric_names, titles, show_plots=False, save_plots=False
        )

        captured = capsys.readouterr()
        assert "Comparison Results" in captured.out or "Baseline" in captured.out


class TestRiskTypes:
    """Tests for Risk-related type definitions."""

    def test_risk_profile_enum(self):
        """Test RiskProfile enum values."""
        assert RiskProfile.CONSERVATIVE.value == "conservative"
        assert RiskProfile.BALANCED.value == "balanced"
        assert RiskProfile.AGGRESSIVE.value == "aggressive"

    def test_risk_profile_limits_creation(self):
        """Test RiskProfileLimits dataclass creation."""
        limits = RiskProfileLimits(
            max_position_notional=1000000.0,
            max_single_trade_pct=0.05,
            daily_loss_limit_pct=0.02,
            max_drawdown_pct=0.10,
            max_trades_per_hour=10,
            min_trade_interval_sec=60,
            max_volatility_pct=0.15,
            required_sharpe_ratio=1.5,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
        )

        assert limits.max_position_notional == 1000000.0
        assert limits.max_single_trade_pct == 0.05
        assert limits.daily_loss_limit_pct == 0.02
        assert limits.max_drawdown_pct == 0.10
        assert limits.max_trades_per_hour == 10
        assert limits.min_trade_interval_sec == 60
        assert limits.max_volatility_pct == 0.15
        assert limits.required_sharpe_ratio == 1.5
        assert limits.stop_loss_pct == 0.02
        assert limits.take_profit_pct == 0.05

    def test_extended_risk_status_creation(self):
        """Test ExtendedRiskStatus TypedDict creation."""
        status: ExtendedRiskStatus = {
            "daily_pnl": 15000.0,
            "daily_trades": 25,
            "daily_trade_limit": 50,
            "hourly_trades": 5,
            "hourly_trade_limit": 10,
            "emergency_stop_loss": 0.03,
            "position": 50000.0,
            "entry_price": 4500000.0,
            "statistics": {"sharpe_ratio": 1.8, "volatility": 0.12},
        }

        assert status["daily_pnl"] == 15000.0
        assert status["daily_trades"] == 25
        assert status["daily_trade_limit"] == 50
        assert status["hourly_trades"] == 5
        assert status["hourly_trade_limit"] == 10
        assert status["emergency_stop_loss"] == 0.03
        assert status["position"] == 50000.0
        assert status["entry_price"] == 4500000.0
        assert status["statistics"]["sharpe_ratio"] == 1.8

    def test_extended_risk_status_partial(self):
        """Test ExtendedRiskStatus with partial fields."""
        status: ExtendedRiskStatus = {"daily_pnl": -5000.0, "daily_trades": 10}

        assert status["daily_pnl"] == -5000.0
        assert status["daily_trades"] == 10
        # Optional fields should not be present
        assert "hourly_trades" not in status

    def test_trigger_status_creation(self):
        """Test TriggerStatus TypedDict creation."""
        trigger: TriggerStatus = {
            "triggered": True,
            "reason": "Stop loss triggered due to price drop",
        }

        assert trigger["triggered"] is True
        assert trigger["reason"] == "Stop loss triggered due to price drop"

    def test_position_monitor_result_creation(self):
        """Test PositionMonitorResult TypedDict creation."""
        result: PositionMonitorResult = {
            "trailing_stop": {
                "triggered": False,
                "reason": "Price above trailing stop level",
            },
            "take_profit": {"triggered": True, "reason": "Take profit target reached"},
        }

        assert result["trailing_stop"]["triggered"] is False
        assert result["trailing_stop"]["reason"] == "Price above trailing stop level"
        assert result["take_profit"]["triggered"] is True
        assert result["take_profit"]["reason"] == "Take profit target reached"

    def test_risk_status_report_creation(self):
        """Test RiskStatusReport TypedDict creation."""
        # Create mock profile and status
        profile = RiskProfileLimits(
            max_position_notional=2000000.0,
            max_single_trade_pct=0.03,
            daily_loss_limit_pct=0.015,
            max_drawdown_pct=0.08,
            max_trades_per_hour=15,
            min_trade_interval_sec=45,
            max_volatility_pct=0.10,
            required_sharpe_ratio=2.0,
            stop_loss_pct=0.015,
            take_profit_pct=0.04,
        )

        # Note: RiskStatus is not defined in the current types.py, so we'll use a dict
        current_status = {
            "portfolio_value": 1000000.0,
            "daily_pnl": 25000.0,
            "position_size": 30000.0,
        }

        report: RiskStatusReport = {
            "profile": profile,
            "current_status": current_status,
            "limits": profile,
        }

        assert report["profile"].max_position_notional == 2000000.0
        assert report["current_status"]["portfolio_value"] == 1000000.0
        assert report["limits"].required_sharpe_ratio == 2.0


class TestProtocols:
    """Tests for Protocol definitions."""

    def test_feature_calculator_protocol(self):
        """Test FeatureCalculator protocol implementation."""

        class MockFeatureCalculator:
            def calculate(self, data):
                return {"feature1": 1.0, "feature2": 2.0}

            @property
            def feature_names(self):
                return ["feature1", "feature2"]

        calculator = MockFeatureCalculator()
        # FeatureCalculator is not @runtime_checkable, so we check method existence
        assert hasattr(calculator, "calculate")
        assert hasattr(calculator, "feature_names")
        assert calculator.calculate({"input": "data"}) == {
            "feature1": 1.0,
            "feature2": 2.0,
        }
        assert calculator.feature_names == ["feature1", "feature2"]

    def test_trainer_protocol(self):
        """Test TrainerProtocol implementation."""

        class MockTrainer:
            def train(self):
                return {"loss": 0.5, "accuracy": 0.9}

            def evaluate(self):
                return {"test_loss": 0.6, "test_accuracy": 0.85}

        trainer = MockTrainer()
        # TrainerProtocol is not @runtime_checkable, so we check method existence
        assert hasattr(trainer, "train")
        assert hasattr(trainer, "evaluate")
        train_result = trainer.train()
        eval_result = trainer.evaluate()
        assert "loss" in train_result
        assert "accuracy" in train_result
        assert "test_loss" in eval_result
        assert "test_accuracy" in eval_result

    def test_callback_protocol(self):
        """Test CallbackProtocol implementation."""

        class MockCallback:
            def __init__(self):
                self.called_with = None

            def __call__(self, locals_, globals_):
                self.called_with = (locals_, globals_)

        callback = MockCallback()
        # CallbackProtocol is not @runtime_checkable, so we check method existence
        assert hasattr(callback, "__call__")

        test_locals = {"loss": 0.5}
        test_globals = {"epoch": 10}
        callback(test_locals, test_globals)
        assert callback.called_with == (test_locals, test_globals)

    def test_performance_monitor_protocol(self):
        """Test PerformanceMonitorProtocol implementation."""

        class MockPerformanceMonitor:
            def __init__(self):
                self.decisions = []

            def record_decision(self, decision):
                self.decisions.append(decision)

            def get_metrics(self):
                return {
                    "total_decisions": len(self.decisions),
                    "last_decision": self.decisions[-1] if self.decisions else None,
                }

        monitor = MockPerformanceMonitor()
        assert isinstance(monitor, PerformanceMonitorProtocol)

        monitor.record_decision("buy")
        monitor.record_decision("sell")

        metrics = monitor.get_metrics()
        assert metrics["total_decisions"] == 2
        assert metrics["last_decision"] == "sell"

    def test_threshold_manager_protocol(self):
        """Test ThresholdManagerProtocol implementation."""

        class MockThresholdManager:
            def __init__(self):
                self.gates = {"upper": 1.0, "lower": -1.0}

            def get_adaptive_gates(self):
                return self.gates

            def update_thresholds(self, evaluation_results):
                # Update gates based on evaluation results
                if "performance_score" in evaluation_results:
                    score = evaluation_results["performance_score"]
                    self.gates["upper"] = min(2.0, self.gates["upper"] + score * 0.1)
                    self.gates["lower"] = max(-2.0, self.gates["lower"] - score * 0.1)

        manager = MockThresholdManager()
        assert isinstance(manager, ThresholdManagerProtocol)

        gates = manager.get_adaptive_gates()
        assert gates["upper"] == 1.0
        assert gates["lower"] == -1.0

        manager.update_thresholds({"performance_score": 0.8})
        updated_gates = manager.get_adaptive_gates()
        assert updated_gates["upper"] == 1.08  # 1.0 + 0.8 * 0.1
        assert updated_gates["lower"] == -1.08  # -1.0 - 0.8 * 0.1


if __name__ == "__main__":
    pytest.main([__file__])
