#!/usr/bin/env python3
"""
Unified Analysis Suite - Integrated analysis tools for trading systems

A comprehensive toolkit for analyzing trading models, data, training processes,
and performance across all algorithms (SAC, PPO, etc.).

Usage:
    python unified_analyze.py <category> <tool> [options]

Categories:
    model       Model analysis and validation
    data        Data quality and feature analysis
    training    Training process analysis
    performance System and memory performance analysis
    comparative Version comparison and statistical tests
    diagnostic  System diagnosis and debugging
    specialized Feature, reward, and risk specific analysis
    session     Session-specific analysis

Examples:
    python unified_analyze.py model sac --model models/sac_model.zip
    python unified_analyze.py data quality --dataset data/train.csv
    python unified_analyze.py training tensorboard --logdir logs/
    python unified_analyze.py comparative versions --versions v378 v381 v384
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.utils.logging_utils import get_logger
from ztb.utils.path_utils import get_project_root

# Get project root using utility
project_root = get_project_root()

logger = get_logger(__name__)


class UnifiedAnalysisSuite:
    """Unified analysis toolkit interface."""

    def __init__(self):
        """Initialize analysis suite."""
        self.project_root = project_root
        self.categories = {
            "model": ModelAnalysis,
            "data": DataAnalysis,
            "training": TrainingAnalysis,
            "performance": PerformanceAnalysis,
            "comparative": ComparativeAnalysis,
            "diagnostic": DiagnosticAnalysis,
            "specialized": SpecializedAnalysis,
            "session": SessionAnalysis,
        }

    def run(self, args: argparse.Namespace) -> int:
        """Run analysis command."""
        try:
            category = args.category

            # If no tool specified, show available tools for the category
            if args.tool is None:
                if category not in self.categories:
                    logger.error(f"Unknown category: {category}")
                    logger.info(
                        f"Available categories: {', '.join(self.categories.keys())}"
                    )
                    return 1

                analyzer_class = self.categories[category]
                analyzer = analyzer_class()
                print(
                    f"Available tools in category '{category}': {', '.join(analyzer.get_available_tools())}"
                )
                return 0

            tool = args.tool

            if category not in self.categories:
                logger.error(f"Unknown category: {category}")
                logger.info(
                    f"Available categories: {', '.join(self.categories.keys())}"
                )
                return 1

            analyzer_class = self.categories[category]
            analyzer = analyzer_class()

            if not hasattr(analyzer, f"run_{tool}"):
                logger.error(f"Unknown tool '{tool}' in category '{category}'")
                logger.info(f"Available tools: {analyzer.get_available_tools()}")
                return 1

            method = getattr(analyzer, f"run_{tool}")
            return method(args)

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return 1


class BaseAnalyzer:
    """Base class for analysis tools."""

    def get_available_tools(self) -> List[str]:
        """Get list of available tools in this category."""
        methods = [m for m in dir(self) if m.startswith("run_")]
        return [m[4:] for m in methods]  # Remove 'run_' prefix


class ModelAnalysis(BaseAnalyzer):
    """Model analysis tools."""

    def run_sac(self, args: argparse.Namespace) -> int:
        """Run SAC model analysis."""
        try:
            from ztb.analysis.core.model.sac_analyzer import SACAnalyzer

            analyzer = SACAnalyzer(
                model_path=getattr(args, "model", None),
                config_path=getattr(args, "config", None),
                samples=getattr(args, "samples", 10000),
            )

            results = analyzer.run_full_analysis()
            analyzer.print_results(results)
            return 0
        except Exception as e:
            logger.error(f"SAC analysis failed: {e}")
            return 1

    def run_extract(self, args: argparse.Namespace) -> int:
        """Extract model information."""
        try:
            from ztb.analysis.core.model.extract_model_info import extract_model_info

            result = extract_model_info(
                model_path=args.model, output_path=getattr(args, "output", None)
            )
            logger.info("Model info extracted successfully")
            return 0
        except Exception as e:
            logger.error(f"Model extraction failed: {e}")
            return 1

    def run_validate(self, args: argparse.Namespace) -> int:
        """Validate model behavior."""
        try:
            from ztb.analysis.core.model.validate_model_behavior import (
                validate_model_behavior,
            )

            result = validate_model_behavior(
                model_path=args.model,
                test_data=getattr(args, "data", None),
                episodes=getattr(args, "episodes", 10),
            )
            logger.info("Model validation completed")
            return 0
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return 1

    def run_features(self, args: argparse.Namespace) -> int:
        """Show model features."""
        try:
            from ztb.analysis.core.model.show_model_features import show_model_features

            result = show_model_features(
                model_path=args.model, detailed=getattr(args, "detailed", False)
            )
            return 0
        except Exception as e:
            logger.error(f"Feature analysis failed: {e}")
            return 1

    def run_sac_v423(self, args: argparse.Namespace) -> int:
        """Analyze SAC v423 model."""
        try:
            from scripts.analyze_sac_v423 import SACv423Analyzer

            analyzer = SACv423Analyzer()
            analyzer.analyze_training_results()
            logger.info("SAC v423 analysis completed")
            return 0
        except Exception as e:
            logger.error(f"SAC v423 analysis failed: {e}")
            return 1

    def run_sac_v423_series(self, args: argparse.Namespace) -> int:
        """Analyze SAC v423 series."""
        try:
            from scripts.analyze_sac_v423_series import SACv423SeriesAnalyzer

            analyzer = SACv423SeriesAnalyzer()
            analyzer.analyze_series_results()
            logger.info("SAC v423 series analysis completed")
            return 0
        except Exception as e:
            logger.error(f"SAC v423 series analysis failed: {e}")
            return 1

    def run_detailed_sac_v423b(self, args: argparse.Namespace) -> int:
        """Run detailed SAC v423b analysis."""
        try:
            import glob

            from scripts.detailed_sac_v423b_analysis import DetailedSACv423bAnalyzer

            # Find the latest SAC v423b report
            report_files = glob.glob("reports/training_report_sac_sac_v423b*.json")
            if not report_files:
                logger.error("No SAC v423b training reports found")
                return 1

            # Use the most recent report
            latest_report = max(report_files, key=lambda x: Path(x).stat().st_mtime)

            analyzer = DetailedSACv423bAnalyzer(latest_report)
            analyzer.generate_comprehensive_report()
            logger.info("Detailed SAC v423b analysis completed")
            return 0
        except Exception as e:
            logger.error(f"Detailed SAC v423b analysis failed: {e}")
            return 1


class DataAnalysis(BaseAnalyzer):
    """Data analysis tools."""

    def run_quality(self, args: argparse.Namespace) -> int:
        """Check data quality."""
        try:
            from ztb.analysis.core.data.check_datasets import check_dataset_quality

            result = check_dataset_quality(
                data_path=args.dataset, detailed=getattr(args, "detailed", False)
            )
            logger.info("Data quality check completed")
            return 0
        except Exception as e:
            logger.error(f"Data quality check failed: {e}")
            return 1

    def run_schema(self, args: argparse.Namespace) -> int:
        """Check feature schema."""
        try:
            from ztb.analysis.core.data.check_feature_schema import check_feature_schema

            result = check_feature_schema(
                data_path=args.dataset, schema_path=getattr(args, "schema", None)
            )
            logger.info("Schema check completed")
            return 0
        except Exception as e:
            logger.error(f"Schema check failed: {e}")
            return 1

    def run_correlation(self, args: argparse.Namespace) -> int:
        """Analyze feature correlations."""
        try:
            from ztb.analysis.core.data.correlation import analyze_correlations

            result = analyze_correlations(
                data_path=args.dataset,
                threshold=getattr(args, "threshold", 0.8),
                output_path=getattr(args, "output", None),
            )
            logger.info("Correlation analysis completed")
            return 0
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return 1

    def run_timeseries(self, args: argparse.Namespace) -> int:
        """Analyze time series properties."""
        try:
            from ztb.analysis.core.data.timeseries import analyze_timeseries

            result = analyze_timeseries(
                data_path=args.dataset,
                analysis_type=getattr(args, "type", "stationarity"),
                output_path=getattr(args, "output", None),
            )
            logger.info("Time series analysis completed")
            return 0
        except Exception as e:
            logger.error(f"Time series analysis failed: {e}")
            return 1


class TrainingAnalysis(BaseAnalyzer):
    """Training analysis tools."""

    def run_tensorboard(self, args: argparse.Namespace) -> int:
        """Analyze TensorBoard events."""
        try:
            from ztb.analysis.core.training.analyze_tensorboard_events import (
                analyze_tensorboard_events,
            )

            result = analyze_tensorboard_events(
                logdir=args.logdir,
                metrics=getattr(args, "metrics", None),
                output_path=getattr(args, "output", None),
            )
            logger.info("TensorBoard analysis completed")
            return 0
        except Exception as e:
            logger.error(f"TensorBoard analysis failed: {e}")
            return 1

    def run_metrics(self, args: argparse.Namespace) -> int:
        """Compare training metrics."""
        try:
            from ztb.analysis.core.training.compare_training_metrics import (
                compare_training_metrics,
            )

            result = compare_training_metrics(
                logdirs=args.logdirs,
                metrics=getattr(args, "metrics", None),
                output_path=getattr(args, "output", None),
            )
            logger.info("Metrics comparison completed")
            return 0
        except Exception as e:
            logger.error(f"Metrics comparison failed: {e}")
            return 1

    def run_progress(self, args: argparse.Namespace) -> int:
        """Monitor training progress."""
        try:
            from ztb.analysis.core.training.monitor_v394_progress import (
                monitor_training_progress,
            )

            result = monitor_training_progress(
                session_id=getattr(args, "session", None),
                realtime=getattr(args, "realtime", False),
            )
            logger.info("Progress monitoring completed")
            return 0
        except Exception as e:
            logger.error(f"Progress monitoring failed: {e}")
            return 1

    def run_profile(self, args: argparse.Namespace) -> int:
        """Profile training performance."""
        try:
            from ztb.analysis.core.training.profile_training import profile_training

            result = profile_training(
                config_path=getattr(args, "config", None),
                output_path=getattr(args, "output", None),
            )
            logger.info("Training profiling completed")
            return 0
        except Exception as e:
            logger.error(f"Training profiling failed: {e}")
            return 1


class PerformanceAnalysis(BaseAnalyzer):
    """Performance analysis tools."""

    def run_memory(self, args: argparse.Namespace) -> int:
        """Monitor memory usage."""
        try:
            from ztb.analysis.core.performance.monitor_memory import (
                monitor_memory_usage,
            )

            result = monitor_memory_usage(
                pid=getattr(args, "pid", None),
                duration=getattr(args, "duration", 60),
                interval=getattr(args, "interval", 1.0),
            )
            logger.info("Memory monitoring completed")
            return 0
        except Exception as e:
            logger.error(f"Memory monitoring failed: {e}")
            return 1

    def run_profile_memory(self, args: argparse.Namespace) -> int:
        """Profile memory usage in detail."""
        try:
            from ztb.analysis.core.performance.profile_memory import (
                profile_memory_usage,
            )

            result = profile_memory_usage(
                code_path=getattr(args, "code", None),
                output_path=getattr(args, "output", None),
            )
            logger.info("Memory profiling completed")
            return 0
        except Exception as e:
            logger.error(f"Memory profiling failed: {e}")
            return 1

    def run_transaction_cost(self, args: argparse.Namespace) -> int:
        """Analyze transaction costs."""
        try:
            from ztb.analysis.core.performance.transaction_cost_analysis import (
                analyze_transaction_costs,
            )

            result = analyze_transaction_costs(
                backtest_results=getattr(args, "results", None),
                cost_structure=getattr(args, "costs", None),
                output_path=getattr(args, "output", None),
            )
            logger.info("Transaction cost analysis completed")
            return 0
        except Exception as e:
            logger.error(f"Transaction cost analysis failed: {e}")
            return 1

    def run_position_duration(self, args: argparse.Namespace) -> int:
        """Analyze position duration."""
        try:
            from scripts.position_duration_analyzer import PositionDurationAnalyzer

            analyzer = PositionDurationAnalyzer(
                data_path=getattr(args, "data", None),
                output_path=getattr(args, "output", None),
            )
            analyzer.run_analysis()
            logger.info("Position duration analysis completed")
            return 0
        except Exception as e:
            logger.error(f"Position duration analysis failed: {e}")
            return 1

    def run_regime_performance(self, args: argparse.Namespace) -> int:
        """Analyze regime performance."""
        try:
            from ztb.analysis.regime_performance_analyzer import (
                RegimePerformanceAnalyzer,
            )

            analyzer = RegimePerformanceAnalyzer(
                data_path=getattr(args, "data", None),
                regime_config=getattr(args, "config", None),
                output_path=getattr(args, "output", None),
            )
            analyzer.run_analysis()
            logger.info("Regime performance analysis completed")
            return 0
        except Exception as e:
            logger.error(f"Regime performance analysis failed: {e}")
            return 1


class ComparativeAnalysis(BaseAnalyzer):
    """Comparative analysis tools."""

    def run_versions(self, args: argparse.Namespace) -> int:
        """Compare multiple versions."""
        try:
            from ztb.analysis.comparative.compare_three_sac_versions import (
                compare_sac_versions,
            )

            result = compare_sac_versions(
                versions=args.versions,
                metrics=getattr(args, "metrics", None),
                output_path=getattr(args, "output", None),
            )
            logger.info("Version comparison completed")
            return 0
        except Exception as e:
            logger.error(f"Version comparison failed: {e}")
            return 1

    def run_backtest(self, args: argparse.Namespace) -> int:
        """Compare backtest results."""
        try:
            from ztb.analysis.comparative.compare_backtest_v378_v381 import (
                compare_backtest_results,
            )

            result = compare_backtest_results(
                results_a=getattr(args, "results_a", None),
                results_b=getattr(args, "results_b", None),
                output_path=getattr(args, "output", None),
            )
            logger.info("Backtest comparison completed")
            return 0
        except Exception as e:
            logger.error(f"Backtest comparison failed: {e}")
            return 1

    def run_statistical(self, args: argparse.Namespace) -> int:
        """Run statistical tests."""
        try:
            from ztb.analysis.comparative.statistical_test_v395g_v395i import (
                run_statistical_tests,
            )

            result = run_statistical_tests(
                data_a=getattr(args, "data_a", None),
                data_b=getattr(args, "data_b", None),
                test_type=getattr(args, "test", "ttest"),
                output_path=getattr(args, "output", None),
            )
            logger.info("Statistical tests completed")
            return 0
        except Exception as e:
            logger.error(f"Statistical tests failed: {e}")
            return 1

    def run_analyze_backtest(self, args: argparse.Namespace) -> int:
        """Run comprehensive backtest analysis."""
        try:
            from ztb.analysis.comparative.analyze_backtest import BacktestAnalyzer

            analyzer = BacktestAnalyzer(
                results_path=getattr(args, "results", None),
                training_report_path=getattr(args, "training_report", None),
            )

            report = analyzer.generate_comprehensive_report()

            # Save report to file if output path specified
            if hasattr(args, "output") and args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(report)
                logger.info(f"Report saved to {args.output}")
            else:
                # Print report to console
                print(report)

            logger.info("Backtest analysis completed")
            return 0
        except Exception as e:
            logger.error(f"Backtest analysis failed: {e}")
            return 1

    def run_benchmark(self, args: argparse.Namespace) -> int:
        """Run benchmark comparison analysis."""
        try:
            import pandas as pd

            from scripts.analysis.benchmark_comparison import (
                BenchmarkComparisonAnalyzer,
            )

            # Load strategy returns
            strategy_returns = pd.read_csv(
                getattr(args, "strategy", ""), index_col=0, parse_dates=True
            )
            benchmark_returns = pd.read_csv(
                getattr(args, "benchmark", ""), index_col=0, parse_dates=True
            )

            analyzer = BenchmarkComparisonAnalyzer()
            result = analyzer.compare_with_benchmark(
                strategy_returns=strategy_returns.squeeze(),
                benchmark_returns=benchmark_returns.squeeze(),
            )

            print("Benchmark Comparison Results:")
            print(f"Strategy Return: {result.strategy_performance['total_return']:.4f}")
            print(
                f"Benchmark Return: {result.benchmark_performance['total_return']:.4f}"
            )
            print(f"Excess Return: {result.excess_returns.mean():.4f}")
            print(f"Information Ratio: {result.information_ratio:.4f}")

            if getattr(args, "output", None):
                with open(args.output, "w") as f:
                    f.write(
                        f"Strategy Return: {result.strategy_performance['total_return']:.4f}\n"
                    )
                    f.write(
                        f"Benchmark Return: {result.benchmark_performance['total_return']:.4f}\n"
                    )
                    f.write(f"Excess Return: {result.excess_returns.mean():.4f}\n")
                    f.write(f"Information Ratio: {result.information_ratio:.4f}\n")

            logger.info("Benchmark comparison analysis completed")
            return 0
        except Exception as e:
            logger.error(f"Benchmark comparison analysis failed: {e}")
            return 1


class DiagnosticAnalysis(BaseAnalyzer):
    """Diagnostic analysis tools."""

    def run_environment(self, args: argparse.Namespace) -> int:
        """Diagnose SAC environment."""
        try:
            from ztb.analysis.diagnostic.diagnose_sac_environment import (
                diagnose_sac_environment,
            )

            result = diagnose_sac_environment(
                config_path=getattr(args, "config", None),
                verbose=getattr(args, "verbose", False),
            )
            logger.info("Environment diagnosis completed")
            return 0
        except Exception as e:
            logger.error(f"Environment diagnosis failed: {e}")
            return 1

    def run_simple(self, args: argparse.Namespace) -> int:
        """Run simple SAC diagnosis."""
        try:
            from ztb.analysis.diagnostic.diagnose_sac_simple import diagnose_sac_simple

            result = diagnose_sac_simple(
                model_path=getattr(args, "model", None),
                quick=getattr(args, "quick", True),
            )
            logger.info("Simple diagnosis completed")
            return 0
        except Exception as e:
            logger.error(f"Simple diagnosis failed: {e}")
            return 1

    def run_features_diag(self, args: argparse.Namespace) -> int:
        """Diagnose feature issues."""
        try:
            from ztb.analysis.diagnostic.diagnose_v381_features import (
                diagnose_feature_issues,
            )

            result = diagnose_feature_issues(
                data_path=getattr(args, "data", None),
                model_path=getattr(args, "model", None),
            )
            logger.info("Feature diagnosis completed")
            return 0
        except Exception as e:
            logger.error(f"Feature diagnosis failed: {e}")
            return 1

    def run_wave3(self, args: argparse.Namespace) -> int:
        """Run Wave 3 diagnosis."""
        try:
            from ztb.analysis.diagnostic.wave3_diag import run_wave3_diagnosis

            result = run_wave3_diagnosis(
                project_root=str(self.project_root),
                output_path=getattr(args, "output", None),
            )
            logger.info("Wave 3 diagnosis completed")
            return 0
        except Exception as e:
            logger.error(f"Wave 3 diagnosis failed: {e}")
            return 1

    def run_explainability(self, args: argparse.Namespace) -> int:
        """Run explainability analysis."""
        try:
            from ztb.adaptation.explainability.analyzer import ExplainabilityAnalyzer

            analyzer = ExplainabilityAnalyzer(
                model_path=getattr(args, "model", None),
                data_path=getattr(args, "data", None),
            )
            analyzer.run_analysis()
            logger.info("Explainability analysis completed")
            return 0
        except Exception as e:
            logger.error(f"Explainability analysis failed: {e}")
            return 1


class SpecializedAnalysis(BaseAnalyzer):
    """Specialized analysis tools."""

    def run_features_quality(self, args: argparse.Namespace) -> int:
        """Analyze feature quality."""
        try:
            from ztb.analysis.specialized.features.analyze_features_quality import (
                analyze_features_quality,
            )

            result = analyze_features_quality(
                data_path=args.data,
                feature_cols=getattr(args, "features", None),
                output_path=getattr(args, "output", None),
            )
            logger.info("Feature quality analysis completed")
            return 0
        except Exception as e:
            logger.error(f"Feature quality analysis failed: {e}")
            return 1

    def run_feature_selection(self, args: argparse.Namespace) -> int:
        """Analyze feature selection."""
        try:
            from ztb.analysis.specialized.features.analyze_feature_selection import (
                analyze_feature_selection,
            )

            result = analyze_feature_selection(
                data_path=args.data,
                target_col=getattr(args, "target", "close"),
                method=getattr(args, "method", "correlation"),
                output_path=getattr(args, "output", None),
            )
            logger.info("Feature selection analysis completed")
            return 0
        except Exception as e:
            logger.error(f"Feature selection analysis failed: {e}")
            return 1

    def run_reward_function(self, args: argparse.Namespace) -> int:
        """Analyze reward function."""
        try:
            from ztb.analysis.specialized.rewards.analyze_reward_function import (
                analyze_reward_function,
            )

            result = analyze_reward_function(
                config_path=getattr(args, "config", None),
                test_data=getattr(args, "data", None),
                output_path=getattr(args, "output", None),
            )
            logger.info("Reward function analysis completed")
            return 0
        except Exception as e:
            logger.error(f"Reward function analysis failed: {e}")
            return 1

    def run_reward_improvements(self, args: argparse.Namespace) -> int:
        """Analyze reward improvements."""
        try:
            from ztb.analysis.specialized.rewards.analyze_reward_improvements import (
                analyze_reward_improvements,
            )

            result = analyze_reward_improvements(
                baseline_config=getattr(args, "baseline", None),
                improved_config=getattr(args, "improved", None),
                test_data=getattr(args, "data", None),
                output_path=getattr(args, "output", None),
            )
            logger.info("Reward improvements analysis completed")
            return 0
        except Exception as e:
            logger.error(f"Reward improvements analysis failed: {e}")
            return 1

    def run_risk_metrics(self, args: argparse.Namespace) -> int:
        """Analyze risk metrics."""
        try:
            from ztb.analysis.specialized.risk.analyze_risk_metrics import (
                analyze_risk_metrics,
            )

            result = analyze_risk_metrics(
                backtest_results=getattr(args, "results", None),
                risk_measures=getattr(args, "measures", None),
                output_path=getattr(args, "output", None),
            )
            logger.info("Risk metrics analysis completed")
            return 0
        except Exception as e:
            logger.error(f"Risk metrics analysis failed: {e}")
            return 1

    def run_ab_test(self, args: argparse.Namespace) -> int:
        """Run A/B testing analysis."""
        try:
            import numpy as np

            from ztb.adaptation.ab_test.analyzer import ABTestAnalyzer

            # Load data for A/B testing
            data_a = np.loadtxt(getattr(args, "data_a", ""), delimiter=",")
            data_b = np.loadtxt(getattr(args, "data_b", ""), delimiter=",")

            analyzer = ABTestAnalyzer()
            result = analyzer.analyze_parallel(data_a, data_b)

            print("A/B Test Results:")
            print(f"P-value: {result.p_value}")
            print(f"Effect size: {result.effect_size}")
            print(f"Confidence interval: {result.confidence_interval}")

            if getattr(args, "output", None):
                with open(args.output, "w") as f:
                    f.write(f"P-value: {result.p_value}\n")
                    f.write(f"Effect size: {result.effect_size}\n")
                    f.write(f"Confidence interval: {result.confidence_interval}\n")

            logger.info("A/B testing analysis completed")
            return 0
        except Exception as e:
            logger.error(f"A/B testing analysis failed: {e}")
            return 1

    def run_enhanced_features(self, args: argparse.Namespace) -> int:
        """Run enhanced feature analysis."""
        try:
            from ztb.analysis.specialized.features.analyze_feature_selection import (
                EnhancedFeatureAnalyzer,
            )

            analyzer = EnhancedFeatureAnalyzer()
            result = analyzer.analyze(
                data_path=args.data,
                target_col=getattr(args, "target", "close"),
                output_path=getattr(args, "output", None),
            )
            logger.info("Enhanced feature analysis completed")
            return 0
        except Exception as e:
            logger.error(f"Enhanced feature analysis failed: {e}")
            return 1


class SessionAnalysis(BaseAnalyzer):
    """Session-specific analysis tools."""

    def run_sac_logs(self, args: argparse.Namespace) -> int:
        """Analyze SAC training logs."""
        try:
            from ztb.analysis.sessions.analyze_sac_logs import analyze_sac_logs

            result = analyze_sac_logs(
                log_path=getattr(args, "log", None),
                session_id=getattr(args, "session", None),
                output_path=getattr(args, "output", None),
            )
            logger.info("SAC logs analysis completed")
            return 0
        except Exception as e:
            logger.error(f"SAC logs analysis failed: {e}")
            return 1

    def run_action_distribution(self, args: argparse.Namespace) -> int:
        """Analyze action distribution."""
        try:
            from ztb.analysis.sessions.analyze_action_distribution import (
                analyze_action_distribution,
            )

            result = analyze_action_distribution(
                model_path=getattr(args, "model", None),
                test_data=getattr(args, "data", None),
                output_path=getattr(args, "output", None),
            )
            logger.info("Action distribution analysis completed")
            return 0
        except Exception as e:
            logger.error(f"Action distribution analysis failed: {e}")
            return 1

    def run_v394_training(self, args: argparse.Namespace) -> int:
        """Analyze V394 training."""
        try:
            from ztb.analysis.sessions.analyze_v394_training import (
                analyze_v394_training,
            )

            result = analyze_v394_training(
                session_id=getattr(args, "session", "v394"),
                detailed=getattr(args, "detailed", False),
                output_path=getattr(args, "output", None),
            )
            logger.info("V394 training analysis completed")
            return 0
        except Exception as e:
            logger.error(f"V394 training analysis failed: {e}")
            return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="unified_analyze",
        description="Unified Analysis Suite - Integrated analysis tools for trading systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("category", help="Analysis category")
    parser.add_argument(
        "tool",
        nargs="?",
        help="Specific analysis tool (optional - shows available tools if not specified)",
    )

    # Common arguments
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Model analysis arguments
    parser.add_argument("--model", "-m", help="Model file path")
    parser.add_argument(
        "--samples", "-s", type=int, default=1000, help="Number of samples"
    )

    # Data analysis arguments
    parser.add_argument("--dataset", "--data", "-d", help="Dataset file path")
    parser.add_argument("--detailed", action="store_true", help="Detailed analysis")
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.8, help="Correlation threshold"
    )
    parser.add_argument("--type", help="Analysis type")

    # Training analysis arguments
    parser.add_argument("--logdir", help="TensorBoard log directory")
    parser.add_argument("--metrics", nargs="+", help="Metrics to analyze")
    parser.add_argument("--logdirs", nargs="+", help="Multiple log directories")
    parser.add_argument("--session", help="Session ID")
    parser.add_argument("--realtime", action="store_true", help="Real-time monitoring")

    # Performance analysis arguments
    parser.add_argument("--pid", type=int, help="Process ID to monitor")
    parser.add_argument(
        "--duration", type=int, default=60, help="Monitoring duration (seconds)"
    )
    parser.add_argument(
        "--interval", type=float, default=1.0, help="Monitoring interval (seconds)"
    )
    parser.add_argument("--code", help="Code file to profile")

    # Comparative analysis arguments
    parser.add_argument("--versions", nargs="+", help="Versions to compare")
    parser.add_argument("--results-a", help="First backtest results")
    parser.add_argument("--results-b", help="Second backtest results")
    parser.add_argument("--data-a", help="First dataset")
    parser.add_argument("--data-b", help="Second dataset")
    parser.add_argument("--test", default="ttest", help="Statistical test type")

    # Specialized analysis arguments
    parser.add_argument("--features", nargs="+", help="Feature columns")
    parser.add_argument("--target", default="close", help="Target column")
    parser.add_argument("--method", default="correlation", help="Analysis method")
    parser.add_argument("--baseline", help="Baseline configuration")
    parser.add_argument("--improved", help="Improved configuration")
    parser.add_argument("--measures", nargs="+", help="Risk measures")
    parser.add_argument("--results", help="Backtest results")
    parser.add_argument("--costs", help="Cost structure file")

    # Session analysis arguments
    parser.add_argument("--log", help="Log file path")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")

    # Additional arguments for new integrated tools
    parser.add_argument("--data_a", help="First dataset for A/B testing")
    parser.add_argument("--data_b", help="Second dataset for A/B testing")
    parser.add_argument("--strategy", help="Strategy returns CSV file")
    parser.add_argument("--benchmark", help="Benchmark returns CSV file")
    parser.add_argument("--model_dir", help="Model directory path")
    parser.add_argument("--training_report", help="Training report path")
    parser.add_argument("--quick", action="store_true", help="Quick analysis mode")

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    suite = UnifiedAnalysisSuite()
    exit_code = suite.run(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
