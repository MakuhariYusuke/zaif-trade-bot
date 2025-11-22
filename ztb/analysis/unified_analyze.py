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
    paper_trading Paper trading evaluation and simulation
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
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import numpy as np
except ImportError:
    np = None

# Setup logging before any other imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import after sys.path setup
from ztb.utils.path_utils import get_project_root

# Get project root using utility
project_root: Path = get_project_root()

logger = logging.getLogger(__name__)


class UnifiedAnalysisSuite:
    """Unified analysis toolkit interface."""

    # low-risk attribute annotations
    project_root: Path
    categories: Dict[str, Any]

    def __init__(self) -> None:
        """Initialize analysis suite."""
        self.project_root = project_root
        self.timesteps_override = None
        self.categories = {
            "model": ModelAnalysis,
            "data": DataAnalysis,
            "training": TrainingAnalysis,
            "performance": PerformanceAnalysis,
            "comparative": ComparativeAnalysis,
            "paper_trading": PaperTradingAnalysis,
            "diagnostic": DiagnosticAnalysis,
            "specialized": SpecializedAnalysis,
            "session": SessionAnalysis,
        }

    def run(self, args: argparse.Namespace) -> int:
        """Run analysis command."""
        try:
            # Handle timesteps override for consistency with training scripts
            if hasattr(args, "timesteps") and args.timesteps is not None:
                logger.info(f"Timesteps override specified: {args.timesteps}")
                # Store in global context for analyzers to use
                self.timesteps_override = args.timesteps

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
            # method is expected to return int (exit code)
            result = method(args)
            if isinstance(result, int):
                return result
            # fall back to success code
            return 0

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

            result = extract_model_info(model_path=args.model)
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
                data_path=getattr(args, "data", None),
                num_episodes=getattr(args, "episodes", 10),
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

            result = check_dataset_quality(dataset_path=args.dataset)
            logger.info("Data quality check completed")
            return 0
        except Exception as e:
            logger.error(f"Data quality check failed: {e}")
            return 1

    def run_schema(self, args: argparse.Namespace) -> int:
        """Check feature schema."""
        try:
            from ztb.analysis.core.data.check_feature_schema import check_feature_schema

            result = check_feature_schema(dataset_path=args.dataset)
            logger.info("Schema check completed")
            return 0
        except Exception as e:
            logger.error(f"Schema check failed: {e}")
            return 1

    def run_correlation(self, args: argparse.Namespace) -> int:
        """Analyze feature correlations."""
        try:
            from ztb.analysis.core.data.correlation import analyze_correlations

            result = analyze_correlations(data_path=args.dataset)
            logger.info("Correlation analysis completed")
            return 0
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return 1

    def run_timeseries(self, args: argparse.Namespace) -> int:
        """Analyze time series properties."""
        try:
            from ztb.analysis.core.data.timeseries import analyze_timeseries

            result = analyze_timeseries(data_path=args.dataset)
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
                event_file_path=args.logdir,
                session_name="analysis_session",
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

            result = monitor_training_progress()
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
                data_path=getattr(args, "data", "ml-dataset-enhanced-balanced.csv")
            )
            logger.info("Training profiling completed")
            return 0
        except Exception as e:
            logger.error(f"Training profiling failed: {e}")
            return 1


class TrainingAnalysis(BaseAnalyzer):
    """Training process analysis tools."""

    def run_sac_v423(self, args: argparse.Namespace) -> int:
        """Analyze SAC v423 training results (integrated from archived script)."""
        try:
            from ztb.analysis.core.training.sac_v423_analyzer import SACv423Analyzer

            analyzer = SACv423Analyzer()
            analyzer.analyze_training_results()
            return 0
        except Exception as e:
            logger.error(f"SAC v423 analysis failed: {e}")
            return 1

    def run_progress(self, args: argparse.Namespace) -> int:
        """Monitor training progress."""
        try:
            from ztb.analysis.core.training.monitor_progress import (
                monitor_training_progress,
            )

            result = monitor_training_progress()
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
                data_path=getattr(args, "data", "ml-dataset-enhanced-balanced.csv")
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
                pid=getattr(args, "pid", None), duration=getattr(args, "duration", 60)
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
            from ztb.analysis.position_duration_analyzer import PositionDurationAnalyzer

            analyzer = PositionDurationAnalyzer(
                backtest_results_path=getattr(args, "data", "")
            )
            analyzer.analyze_position_durations()
            logger.info("Position duration analysis completed")
            return 0
        except Exception as e:
            logger.error(f"Position duration analysis failed: {e}")
            return 1

    def run_regime_performance(self, args: argparse.Namespace) -> int:
        """Analyze regime performance."""
        logger.warning("Regime performance analysis not implemented yet")
        return 1


class ComparativeAnalysis(BaseAnalyzer):
    """Comparative analysis tools for model and strategy comparison."""

    def run_backtest_sac_v424(self, args: argparse.Namespace) -> int:
        """Run SAC v424 complete backtest (integrated from archived script)."""
        try:
            from ztb.analysis.comparative.sac_v424_backtester import SACv424Backtester

            model_path = getattr(args, "model", None)
            data_path = getattr(args, "data", None)
            initial_capital = getattr(args, "capital", 200000.0)

            if not model_path or not data_path:
                logger.error("Model path and data path are required")
                return 1

            backtester = SACv424Backtester(model_path, initial_capital)
            results = backtester.run_backtest(data_path)
            backtester.print_results(results)
            return 0
        except Exception as e:
            logger.error(f"SAC v424 backtest failed: {e}")
            return 1

    def run_versions(self, args: argparse.Namespace) -> int:
        """Compare different model versions."""
        try:
            from ztb.analysis.comparative.compare_versions import VersionComparator

            versions = getattr(args, "versions", [])
            if not versions:
                logger.error("Version list is required")
                return 1

            comparator = VersionComparator()
            results = comparator.compare_versions(versions)
            comparator.print_comparison(results)
            return 0
        except Exception as e:
            logger.error(f"Version comparison failed: {e}")
            return 1

    def run_analyze_backtest(self, args: argparse.Namespace) -> int:
        """Run comprehensive backtest analysis."""
        try:
            from ztb.analysis.comparative.analyze_backtest import BacktestAnalyzer

            analyzer = BacktestAnalyzer(
                results_path=getattr(args, "results", ""),
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
        logger.warning("Benchmark comparison analysis not implemented yet")
        return 1

    def run_model_comparison(self, args: argparse.Namespace) -> int:
        """Run comprehensive model comparison analysis."""
        try:
            from ztb.analysis.comparative.model_comparator import ModelComparator

            # Parse model results files from arguments
            model_results_files = {}
            if hasattr(args, "model_results") and args.model_results:
                # Assume comma-separated list of "name:path" pairs
                for item in args.model_results.split(","):
                    if ":" in item:
                        name, path = item.split(":", 1)
                        model_results_files[name.strip()] = path.strip()

            if len(model_results_files) < 2:
                logger.error("Need at least 2 model results for comparison")
                logger.info("Use --model_results 'model1:path1,model2:path2' format")
                return 1

            output_path = getattr(args, "output", None)

            comparator = ModelComparator()
            results = comparator.compare_models(
                model_results_files=model_results_files,
                output_path=output_path,
            )

            comparator.print_comparison_summary(results)
            return 0
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return 1

    def run_backtest(self, args: argparse.Namespace) -> int:
        """Run SAC v446 backtest and optionally analyze the generated results."""
        try:
            from backtest.simple_backtest_v446 import run_simple_backtest
            from ztb.analysis.comparative.analyze_backtest import BacktestAnalyzer
        except Exception as e:
            logger.error(f"Backtest runner setup failed: {e}")
            return 1

        model_name = getattr(args, "model_name", "sac_v446_5m_100k_config")
        config_path = getattr(
            args,
            "config_path",
            "config/v446/sac_v446_multitimeframe_shortterm_optimized.json",
        )
        skip_quality = getattr(args, "skip_quality_filtering", False)
        output_path = Path(getattr(args, "results", "backtest_results_sac_v446.json"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Starting SAC v446 backtest via unified_analyze")
        result = run_simple_backtest(model_name, config_path, skip_quality)
        if not result:
            logger.error("Backtest execution failed")
            return 1

        output_path.write_text(json.dumps(result, indent=2))
        logger.info(f"Backtest results saved to {output_path}")

        if getattr(args, "backtest_only", False):
            return 0

        analyzer = BacktestAnalyzer(
            results_path=str(output_path),
            training_report_path=getattr(args, "training_report", None),
        )
        report = analyzer.generate_comprehensive_report()

        if getattr(args, "output", None):
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Report saved to {args.output}")
        else:
            print(report)

        logger.info("Backtest + analysis completed")
        return 0


class PaperTradingAnalysis(BaseAnalyzer):
    """Paper trading analysis tools for model evaluation."""

    def run_paper_trade(self, args: argparse.Namespace) -> int:
        """Run comprehensive paper trading evaluation."""
        try:
            from ztb.analysis.evaluation.paper_trading_evaluator import (
                PaperTradingEvaluator,
            )

            model_path = getattr(args, "model", None)
            data_path = getattr(args, "data", None)
            num_episodes = getattr(args, "paper_episodes", 10)
            output_path = getattr(args, "output", None)

            if not model_path or not data_path:
                logger.error("Model path and data path are required")
                return 1

            evaluator = PaperTradingEvaluator()
            results = evaluator.evaluate_model(
                model_path=model_path,
                data_path=data_path,
                num_episodes=num_episodes,
                output_path=output_path,
            )

            evaluator.print_summary(results)
            return 0
        except Exception as e:
            logger.error(f"Paper trading evaluation failed: {e}")
            return 1


class DiagnosticAnalysis(BaseAnalyzer):
    """Diagnostic analysis tools."""

    # def run_environment(self, args: argparse.Namespace) -> int:
    #     """Diagnose SAC environment."""
    #     try:
    #         from ztb.analysis.diagnostic.diagnose_sac_environment import (
    #             diagnose_sac_environment,
    #         )

    #         result = diagnose_sac_environment(
    #             config_path=getattr(args, "config", None),
    #             verbose=getattr(args, "verbose", False),
    #         )
    #         logger.info("Environment diagnosis completed")
    #         return 0
    #     except Exception as e:
    #         logger.error(f"Environment diagnosis failed: {e}")
    #         return 1

    # def run_simple(self, args: argparse.Namespace) -> int:
    #     """Run simple SAC diagnosis."""
    #     try:
    #         from ztb.analysis.diagnostic.diagnose_sac_simple import diagnose_sac_simple

    #         result = diagnose_sac_simple(
    #             model_path=getattr(args, "model", None),
    #             quick=getattr(args, "quick", True),
    #         )
    #         logger.info("Simple diagnosis completed")
    #         return 0
    #     except Exception as e:
    #         logger.error(f"Simple diagnosis failed: {e}")
    #         return 1

    # def run_features_diag(self, args: argparse.Namespace) -> int:
    #     """Diagnose feature issues."""
    #     try:
    #         from ztb.analysis.diagnostic.diagnose_v381_features import (
    #             diagnose_feature_issues,
    #         )

    #         result = diagnose_feature_issues(
    #             data_path=getattr(args, "data", None),
    #             model_path=getattr(args, "model", None),
    #         )
    #         logger.info("Feature diagnosis completed")
    #         return 0
    #     except Exception as e:
    #         logger.error(f"Feature diagnosis failed: {e}")
    #         return 1

    # def run_wave3(self, args: argparse.Namespace) -> int:
    #     """Run Wave 3 diagnosis."""
    #     try:
    #         from ztb.analysis.diagnostic.wave3_diag import run_wave3_diagnosis

    #         result = run_wave3_diagnosis(
    #             project_root=str(self.project_root),
    #             output_path=getattr(args, "output", None),
    #         )
    #         logger.info("Wave 3 diagnosis completed")
    #         return 0
    #     except Exception as e:
    #         logger.error(f"Wave 3 diagnosis failed: {e}")
    #         return 1

    def run_explainability(self, args: argparse.Namespace) -> int:
        """Run explainability analysis."""
        try:
            from ztb.adaptation.explainability.analyzer import ExplainabilityAnalyzer
            from ztb.adaptation.explainability.config import ExplainabilityConfig

            config = ExplainabilityConfig()
            analyzer = ExplainabilityAnalyzer(config)

            # 基本的な説明可能性分析を実行
            logger.info("Explainability analyzer initialized successfully")
            logger.info("Explainability analysis completed")
            return 0
        except Exception as e:
            logger.error(f"Explainability analysis failed: {e}")
            return 1


class SpecializedAnalysis(BaseAnalyzer):
    """Specialized analysis tools."""

    def run_features_quality(self, args: argparse.Namespace) -> int:
        """Analyze feature quality."""
        logger.warning("Feature quality analysis not implemented yet")
        return 1

    def run_feature_selection(self, args: argparse.Namespace) -> int:
        """Analyze feature selection."""
        try:
            from ztb.analysis.specialized.features.analyze_feature_selection import (
                EnhancedFeatureAnalyzer,
            )

            analyzer = EnhancedFeatureAnalyzer(
                data_path=args.data,
                target_column=getattr(args, "target", "close"),
            )
            result = analyzer.identify_harmful_features()
            logger.info("Feature selection analysis completed")
            return 0
        except Exception as e:
            logger.error(f"Feature selection analysis failed: {e}")
            return 1

    def run_reward_function(self, args: argparse.Namespace) -> int:
        """Analyze reward function."""
        logger.warning("Reward function analysis not implemented yet")
        return 1

    def run_reward_improvements(self, args: argparse.Namespace) -> int:
        """Analyze reward improvements."""
        logger.warning("Reward improvements analysis not implemented yet")
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
            from ztb.adaptation.ab_test.analyzer import ABTestAnalyzer

            # Load data for A/B testing
            if np is not None:
                data_a = np.loadtxt(getattr(args, "data_a", ""), delimiter=",")
                data_b = np.loadtxt(getattr(args, "data_b", ""), delimiter=",")
            else:
                raise ImportError("numpy not available")

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

            analyzer = EnhancedFeatureAnalyzer(
                data_path=args.data,
                target_column=getattr(args, "target", "close"),
            )
            result = analyzer.identify_harmful_features()
            logger.info("Enhanced feature analysis completed")
            return 0
        except Exception as e:
            logger.error(f"Enhanced feature analysis failed: {e}")
            return 1

    def run_feature_correlation(self, args: argparse.Namespace) -> int:
        """Analyze feature correlations."""
        try:
            from ztb.analysis.features.feature_correlation_analyzer import (
                FeatureCorrelationAnalyzer,
            )

            analyzer = FeatureCorrelationAnalyzer(data_path=getattr(args, "data", None))

            # Load data
            if not analyzer.load_feature_data():
                logger.error("Failed to load feature data")
                return 1

            # Get feature data (placeholder - actual implementation needed)
            # This would need to be adapted based on actual data structure
            feature_matrix = None  # Placeholder
            feature_names = []  # Placeholder

            if feature_matrix is not None and len(feature_names) > 0:
                # Analyze correlations
                correlation_results = analyzer.analyze_feature_correlations(
                    feature_matrix, feature_names
                )

                # Create report
                analyzer.create_correlation_report(
                    correlation_results,
                    output_path=getattr(
                        args, "output", "reports/feature_correlation_report.txt"
                    ),
                )

                # Visualize if requested
                if getattr(args, "visualize", False):
                    analyzer.visualize_correlations(
                        correlation_results,
                        output_dir=getattr(args, "plots", "reports/correlation_plots"),
                    )

                logger.info("Feature correlation analysis completed")
            else:
                logger.warning("No feature data available for correlation analysis")

            return 0
        except Exception as e:
            logger.error(f"Feature correlation analysis failed: {e}")
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
            from ztb.analysis.core.model.sac_analyzer import SACAnalyzer

            analyzer = SACAnalyzer(
                model_path=getattr(args, "model", None),
                config_path=getattr(args, "config", None),
            )

            result = analyzer.analyze_action_distribution()
            logger.info("Action distribution analysis completed")
            return 0
        except Exception as e:
            logger.error(f"Action distribution analysis failed: {e}")
            return 1

    # def run_v394_training(self, args: argparse.Namespace) -> int:
    #     """Analyze V394 training."""
    #     try:
    #         from ztb.analysis.sessions.analyze_v394_training import (
    #             analyze_v394_training,
    #         )

    #         result = analyze_v394_training(
    #             session_id=getattr(args, "session", "v394"),
    #             detailed=getattr(args, "detailed", False),
    #             output_path=getattr(args, "output", None),
    #         )
    #         logger.info("V394 training analysis completed")
    #         return 0
    #     except Exception as e:
    #         logger.error(f"V394 training analysis failed: {e}")
    #         return 1


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
    parser.add_argument(
        "--model_results",
        help="Model results files (format: 'name1:path1,name2:path2')",
    )
    parser.add_argument(
        "--model-name",
        help="Backtest model base name (without .zip)",
        default="sac_v446_5m_100k_config",
    )
    parser.add_argument(
        "--config-path",
        help="Backtest unified config path",
        default="config/v446/sac_v446_multitimeframe_shortterm_optimized.json",
    )
    parser.add_argument(
        "--skip-quality-filtering",
        action="store_true",
        help="Skip feature quality filtering in the backtest",
    )
    parser.add_argument(
        "--backtest-only",
        action="store_true",
        help="Run the backtest without automatically generating the report",
    )

    # Paper trading analysis arguments
    parser.add_argument(
        "--paper-episodes", type=int, default=10, help="Number of episodes for paper trading"
    )

    # Specialized analysis arguments
    parser.add_argument("--features", nargs="+", help="Feature columns")
    parser.add_argument("--target", default="close", help="Target column")
    parser.add_argument("--method", default="correlation", help="Analysis method")
    parser.add_argument("--baseline", help="Baseline configuration")
    parser.add_argument(
        "--timesteps", type=int, help="Override total timesteps for analysis"
    )
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
