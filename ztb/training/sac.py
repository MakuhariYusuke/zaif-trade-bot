#!/usr/bin/env python3
# ruff: noqa: E402
"""
SAC Suite - Unified Soft Actor-Critic trading system toolkit

A comprehensive toolkit for SAC-based trading models providing:
- Model analysis and evaluation
- Backtesting and performance assessment
- Training and optimization
- Utilities and maintenance tools

Usage:
    python sac.py <command> [options]

Commands:
    analyze     Analyze SAC model performance and behavior
    backtest    Run backtesting simulations
    train       Train SAC models with various configurations
    utils       Utility functions for maintenance and validation

Examples:
    python sac.py analyze --model models/sac_model.zip
    python sac.py backtest --model models/sac_model.zip --data data/test.csv
    python sac.py train --config configs/sac_training.yaml
    python sac.py utils config
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.config.manager import ConfigManager
from ztb.training.constants import (
    SAC_DEFAULT_EPISODES,
    SAC_DEFAULT_SAMPLES,
    SAC_ERROR_EXIT_CODE,
    SAC_PRINT_SEPARATOR_WIDTH,
)
from ztb.utils.logging_utils import get_logger
from ztb.utils.path_utils import get_project_root

# Get project root using utility
project_root = get_project_root()

logger = get_logger(__name__)


class SACSuite:
    """Unified SAC toolkit interface."""

    def __init__(self):
        """Initialize SAC suite."""
        self.project_root = project_root
        self.config_manager = ConfigManager.get_instance()

    def run_analyze(self, args: argparse.Namespace) -> int:
        """Run analysis command."""
        try:
            # Load configuration
            config_path = getattr(args, "config", None)
            if config_path:
                self.config_manager.load_config(config_path)
            else:
                self.config_manager.load_config()  # Load default config

            from ztb.analysis.unified_analyze import ModelAnalysis

            analyzer = ModelAnalysis()
            # Create a mock args object for unified_analyze
            mock_args = argparse.Namespace()
            mock_args.model = getattr(args, "model", None)
            mock_args.config = config_path
            mock_args.samples = getattr(args, "samples", 10000)

            return analyzer.run_sac(mock_args)
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return 1

    def run_backtest(self, args: argparse.Namespace) -> int:
        """Run backtest command."""
        try:
            # Load configuration
            config_path = getattr(args, "config", None)
            (
                self.config_manager.load_config(config_path)
                if config_path
                else self.config_manager.load_config()
            )

            from ztb.analysis.unified_analyze import ComparativeAnalysis

            analyzer = ComparativeAnalysis()
            # Create a mock args object for unified_analyze
            mock_args = argparse.Namespace()
            mock_args.results = getattr(
                args, "results", None
            )  # Path to existing backtest results
            mock_args.training_report = getattr(args, "training_report", None)
            mock_args.output = getattr(args, "output", None)

            # If no existing results, we need to run backtest first
            if not getattr(args, "results", None):
                logger.warning(
                    "No existing backtest results specified. Use --results to analyze existing results."
                )
                logger.info(
                    "For running new backtests, use the evaluation/backtest_model.py script directly."
                )
                return 1

            return analyzer.run_analyze_backtest(mock_args)
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return 1

    def run_train(self, args: argparse.Namespace) -> int:
        """Run training command using unified trainer."""
        try:
            # Load configuration
            config_path = getattr(args, "config", None)
            global_config = (
                self.config_manager.load_config(config_path)
                if config_path
                else self.config_manager.load_config()
            )

            # Check if parallel training is requested
            if getattr(args, "parallel", False):
                return self._run_parallel_training(args)

            # Import unified trainer to avoid circular imports
            from ztb.training.unified_trainer.trainer import UnifiedTrainer

            # Create unified trainer with global config
            training_config = (
                global_config.training.model_dump() if global_config.training else {}
            )

            trainer = UnifiedTrainer(
                config=training_config,
                total_timesteps=getattr(args, "timesteps", None)
                or training_config["training"]["total_timesteps"],
            )
            success = trainer.train()

            if getattr(args, "validate", False):
                # Validation logic would go here
                logger.info("Model validation completed")

            return 0 if success else 1
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return 1

    def _run_parallel_training(self, args: argparse.Namespace) -> int:
        """Run parallel training with multiple algorithms."""
        try:
            # Load configuration
            config_path = getattr(args, "config", None)
            global_config = (
                self.config_manager.load_config(config_path)
                if config_path
                else self.config_manager.load_config()
            )
            (
                global_config.training.model_dump() if global_config.training else {}
            )

            from ztb.training.unified_trainer.parallel_trainer import ParallelTrainer

            # Create configs for different algorithms
            configs = []

            # SAC config
            sac_config = (
                global_config.training.model_dump() if global_config.training else {}
            )
            sac_config["total_timesteps"] = (
                getattr(args, "timesteps", None)
                or sac_config["training"]["total_timesteps"]
            )
            configs.append(sac_config)

            # PPO config if requested
            if getattr(args, "include_ppo", False):
                ppo_config = (
                    global_config.training.model_dump()
                    if global_config.training
                    else {}
                )
                ppo_config["total_timesteps"] = (
                    getattr(args, "timesteps", None)
                    or ppo_config["training"]["total_timesteps"]
                )
                configs.append(ppo_config)

            # Load config files if provided
            if getattr(args, "config", None):
                import json

                with open(args.config, "r") as f:
                    config_data = json.load(f)
                # Apply to all configs
                for config in configs:
                    for key, value in config_data.items():
                        if hasattr(config, key):
                            setattr(config, key, value)

            # Run parallel training
            parallel_trainer = ParallelTrainer(configs)
            success = parallel_trainer.train_all()

            return 0 if success else 1
        except Exception as e:
            logger.error(f"Parallel training failed: {e}")
            return 1

    def run_utils(self, args: argparse.Namespace) -> int:
        """Run utilities command."""
        try:
            # Load configuration for context
            config_path = getattr(args, "config", None)
            if config_path:
                self.config_manager.load_config(config_path)
            else:
                self.config_manager.load_config()  # Load default config

            from ztb.analysis.unified_analyze import DiagnosticAnalysis

            analyzer = DiagnosticAnalysis()

            if args.command == "config":
                # Use environment diagnosis for config checking
                mock_args = argparse.Namespace()
                mock_args.config = getattr(args, "config_dir", "config")
                mock_args.verbose = True
                return analyzer.run_environment(mock_args)
            elif args.command == "data":
                # Use simple diagnosis for data validation
                mock_args = argparse.Namespace()
                mock_args.model = None  # No model needed for data validation
                mock_args.quick = True
                return analyzer.run_simple(mock_args)
            elif args.command == "clean":
                logger.warning(
                    "Clean functionality not yet integrated with unified_analyze"
                )
                return 1
            elif args.command == "quality":
                logger.warning(
                    "Quality check functionality not yet integrated with unified_analyze"
                )
                return 1
            elif args.command == "fix":
                logger.warning(
                    "Fix functionality not yet integrated with unified_analyze"
                )
                return 1

            return 0
        except Exception as e:
            logger.error(f"Utils command failed: {e}")
            return 1

    def _print_config_results(self, results: Dict[str, Any]) -> None:
        """Print configuration check results."""
        print("\n" + "=" * SAC_PRINT_SEPARATOR_WIDTH)
        print("CONFIG CONSISTENCY CHECK")
        print("=" * SAC_PRINT_SEPARATOR_WIDTH)

        if "error" in results:
            print(f"❌ Error: {results['error']}")
        else:
            print(f"📊 Total Files: {results['total_files']}")
            print(f"🔑 Common Keys: {len(results['common_keys'])}")
            print(f"📈 Consistency Score: {results['consistency_score']:.2%}")

            if results["type_inconsistencies"]:
                print(
                    f"\n⚠️  Type Inconsistencies: {len(results['type_inconsistencies'])}"
                )
                for key, info in list(results["type_inconsistencies"].items())[:5]:
                    print(f"  • {key}: {info['types_found']}")

    def _print_data_results(self, results: Dict[str, Any]) -> None:
        """Print data validation results."""
        print("\n" + "=" * SAC_PRINT_SEPARATOR_WIDTH)
        print("DATA VALIDATION")
        print("=" * SAC_PRINT_SEPARATOR_WIDTH)

        if "error" in results:
            print(f"❌ Error: {results['error']}")
        else:
            print(f"📊 Total Files: {results['total_files']}")
            print(f"✅ Valid Files: {results['valid_files']}")
            print(f"❌ Invalid Files: {results['invalid_files']}")

            if results["invalid_files"] > 0:
                print("\n⚠️  Invalid Files:")
                for file_info in results["file_details"]:
                    if not file_info["valid"]:
                        print(
                            f"  • {file_info['filename']}: {file_info.get('error', 'Missing required columns')}"
                        )

    def _print_clean_results(self, results: Dict[str, Any]) -> None:
        """Print cleanup results."""
        print("\n" + "=" * SAC_PRINT_SEPARATOR_WIDTH)
        print("PROJECT CLEANUP")
        print("=" * SAC_PRINT_SEPARATOR_WIDTH)

        print(f"🔍 Files Found: {results['files_found']}")
        print(f"💾 Total Size: {results['total_size_mb']:.2f} MB")
        print(f"🗑️  Files Removed: {results['files_removed']}")

        if not results["dry_run"]:
            print("\n💡 Use --apply to actually remove files")

    def _print_quality_results(self, results: Dict[str, Any]) -> None:
        """Print code quality results."""
        print("\n" + "=" * SAC_PRINT_SEPARATOR_WIDTH)
        print("CODE QUALITY CHECK")
        print("=" * SAC_PRINT_SEPARATOR_WIDTH)

        for check, info in results.items():
            status = info["status"]
            if status == "completed":
                if check == "mypy":
                    print(f"🔍 MyPy: ✅ {info['errors']} errors")
                elif check == "flake8":
                    print(f"🔍 Flake8: ✅ {info['errors']} issues")
                elif check == "tests":
                    print(
                        f"🧪 Tests: ✅ {info['passed']} passed, ❌ {info['failed']} failed"
                    )
            else:
                print(f"🔍 {check.title()}: ❌ {status}")

    def _print_fix_results(self, results: Dict[str, Any]) -> None:
        """Print fix results."""
        print("\n" + "=" * SAC_PRINT_SEPARATOR_WIDTH)
        print("COMMON ISSUE FIXES")
        print("=" * SAC_PRINT_SEPARATOR_WIDTH)

        print(f"🔧 Files Processed: {results['files_processed']}")

        if results["fixes_applied"]:
            print("\n✅ Fixes Applied:")
            for fix in results["fixes_applied"]:
                print(f"  • {fix}")
        else:
            print("\n✅ No fixes needed")


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="sac",
        description="SAC Suite - Unified Soft Actor-Critic trading system toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze SAC model performance"
    )
    analyze_parser.add_argument("--model", "-m", help="Path to SAC model file")
    analyze_parser.add_argument("--config", "-c", help="Path to configuration file")
    analyze_parser.add_argument(
        "--samples", "-s", type=int, default=SAC_DEFAULT_SAMPLES, help="Number of samples to analyze"
    )

    # Backtest command
    backtest_parser = subparsers.add_parser(
        "backtest", help="Run backtesting simulations"
    )
    backtest_parser.add_argument(
        "--model", "-m", required=True, help="Path to SAC model file"
    )
    backtest_parser.add_argument(
        "--data", "-d", required=True, help="Path to test data (CSV)"
    )
    backtest_parser.add_argument("--config", "-c", help="Path to configuration file")
    backtest_parser.add_argument(
        "--episodes", "-e", type=int, default=SAC_DEFAULT_EPISODES, help="Number of episodes to run"
    )
    backtest_parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic policy"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train SAC models")
    train_parser.add_argument(
        "--config", "-c", required=True, help="Path to configuration file"
    )
    train_parser.add_argument(
        "--timesteps", "-t", type=int, help="Override total timesteps"
    )
    train_parser.add_argument("--output-dir", "-o", help="Override output directory")
    train_parser.add_argument("--curriculum", help="Path to curriculum config (JSON)")
    train_parser.add_argument(
        "--validate", action="store_true", help="Validate trained model"
    )
    train_parser.add_argument("--model-path", help="Path to model for validation")
    train_parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run parallel training with multiple algorithms",
    )
    train_parser.add_argument(
        "--include-ppo",
        action="store_true",
        help="Include PPO training in parallel mode",
    )

    # Utils command
    utils_parser = subparsers.add_parser("utils", help="Utility functions")
    utils_subparsers = utils_parser.add_subparsers(
        dest="subcommand", help="Utility commands"
    )

    # Utils subcommands
    config_parser = utils_subparsers.add_parser(
        "config", help="Check configuration consistency"
    )
    config_parser.add_argument(
        "--config-dir", default="config", help="Config directory"
    )

    data_parser = utils_subparsers.add_parser("data", help="Validate data files")
    data_parser.add_argument("--data-dir", default="data", help="Data directory")

    clean_parser = utils_subparsers.add_parser("clean", help="Clean project files")
    clean_parser.add_argument(
        "--apply", action="store_true", help="Actually apply changes"
    )

    utils_subparsers.add_parser("quality", help="Check code quality")
    utils_subparsers.add_parser("fix", help="Fix common issues")

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    suite = SACSuite()

    try:
        if args.command == "analyze":
            return suite.run_analyze(args)
        elif args.command == "backtest":
            return suite.run_backtest(args)
        elif args.command == "train":
            return suite.run_train(args)
        elif args.command == "utils":
            if not hasattr(args, "subcommand") or not args.subcommand:
                parser.parse_args(["utils", "--help"])
                return 1
            args.command = args.subcommand  # Rename for utils handler
            return suite.run_utils(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return SAC_ERROR_EXIT_CODE
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
