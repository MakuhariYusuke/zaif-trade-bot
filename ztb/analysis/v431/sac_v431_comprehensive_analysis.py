#!/usr/bin/env python3
"""
SAC v431 Comprehensive Analysis Script
unified_analyzeを軸にした包括的分析ツール

Features:
- 自動トレーニングレポート生成
- バックテスト分析
- パフォーマンス比較
- リスク評価
- アンサンブル分析
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.analysis.unified_analyze import UnifiedAnalysisSuite
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SACv431Analyzer:
    """SAC v431 Comprehensive Analyzer"""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.analysis_suite = UnifiedAnalysisSuite()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_training_report(
        self, training_results_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive training report"""
        logger.info("📋 Generating SAC v431 Training Report")

        report = "# SAC v431 Training Report\n\n"
        report += "## Configuration\n"
        report += f"- Version: {self.config['version']}\n"
        report += f"- Description: {self.config['description']}\n"
        report += f"- Optimization Score: {self.config['optimization_score']:.4f}\n\n"

        # Advanced Learning Features
        if "advanced_learning" in self.config:
            report += "## Advanced Learning Features\n\n"

            if (
                self.config["advanced_learning"]
                .get("curriculum_learning", {})
                .get("enabled", False)
            ):
                report += "### 🎓 Curriculum Learning\n"
                stages = self.config["advanced_learning"]["curriculum_learning"][
                    "stages"
                ]
                for stage in stages:
                    report += f"- **{stage['name']}**: {stage['timesteps']} timesteps, LR={stage['learning_rate']}\n"
                report += "\n"

            if (
                self.config["advanced_learning"]
                .get("multi_stage_training", {})
                .get("enabled", False)
            ):
                report += "### 🔄 Multi-Stage Training\n"
                stages = self.config["advanced_learning"]["multi_stage_training"][
                    "stages"
                ]
                for stage in stages:
                    report += f"- **{stage['name']}**: {stage['timesteps']} timesteps ({stage['focus']})\n"
                report += "\n"

            if (
                self.config["advanced_learning"]
                .get("ensemble_training", {})
                .get("enabled", False)
            ):
                report += "### 👥 Ensemble Training\n"
                ensemble = self.config["advanced_learning"]["ensemble_training"]
                report += f"- Members: {ensemble['members']}\n"
                report += (
                    f"- Specializations: {', '.join(ensemble['specializations'])}\n"
                )
                report += f"- Voting: {ensemble['voting_mechanism']}\n\n"

        # Reward Function Changes
        report += "## Reward Function Changes (v430 → v431)\n\n"
        report += "| Parameter | v430 | v431 | Change |\n"
        report += "|-----------|------|------|--------|\n"

        # Compare with v430 config
        v430_config_path = Path("configs/v430/sac_v430_optimized.json")
        if v430_config_path.exists():
            with open(v430_config_path, "r") as f:
                v430_config = json.load(f)

            v430_reward = v430_config["reward_function"]
            v431_reward = self.config["reward_function"]

            changes = []
            for key in v431_reward:
                if key in v430_reward:
                    v430_val = v430_reward[key]
                    v431_val = v431_reward[key]
                    if v430_val != v431_val:
                        changes.append((key, v430_val, v431_val, "Modified"))
                else:
                    changes.append((key, "N/A", v431_reward[key], "Added"))

            for key in v430_reward:
                if key not in v431_reward:
                    changes.append((key, v430_reward[key], "N/A", "Removed"))

            for change in changes:
                report += f"| {change[0]} | {change[1]} | {change[2]} | {change[3]} |\n"

        report += "\n## Key Improvements\n\n"
        report += "- ✅ **Reward Function Redesign**: penalty → bonus (positive reinforcement)\n"
        report += "- ✅ **Symmetric Action Thresholds**: Prevents value sticking (from v428)\n"
        report += "- ✅ **Advanced Learning Integration**: Curriculum, Multi-stage, Ensemble\n"
        report += "- ✅ **Unified Analysis Integration**: Automated comprehensive reporting\n\n"

        return report

    def run_backtest_analysis(
        self, backtest_results_path: str, output_path: str
    ) -> bool:
        """Run comprehensive backtest analysis"""
        logger.info("📊 Running Backtest Analysis")

        try:
            # Use unified_analyze for backtest analysis
            result = self.analysis_suite.run(
                [
                    "comparative",
                    "analyze_backtest",
                    "--results",
                    backtest_results_path,
                    "--output",
                    output_path,
                ]
            )

            if result == 0:
                logger.info("✅ Backtest analysis completed")
                return True
            else:
                logger.error("❌ Backtest analysis failed")
                return False

        except Exception as e:
            logger.error(f"❌ Backtest analysis error: {e}")
            return False

    def run_performance_comparison(self, models: List[str], output_path: str) -> bool:
        """Run performance comparison across models"""
        logger.info("📈 Running Performance Comparison")

        try:
            # Create comparison command
            cmd_args = ["comparative", "versions", "--versions"] + models
            if output_path:
                cmd_args.extend(["--output", output_path])

            result = self.analysis_suite.run(cmd_args)

            if result == 0:
                logger.info("✅ Performance comparison completed")
                return True
            else:
                logger.error("❌ Performance comparison failed")
                return False

        except Exception as e:
            logger.error(f"❌ Performance comparison error: {e}")
            return False

    def run_risk_assessment(self, backtest_results_path: str, output_path: str) -> bool:
        """Run comprehensive risk assessment"""
        logger.info("⚠️ Running Risk Assessment")

        try:
            # Use specialized risk analysis
            result = self.analysis_suite.run(
                [
                    "specialized",
                    "risk",
                    "--backtest",
                    backtest_results_path,
                    "--output",
                    output_path,
                ]
            )

            if result == 0:
                logger.info("✅ Risk assessment completed")
                return True
            else:
                logger.error("❌ Risk assessment failed")
                return False

        except Exception as e:
            logger.error(f"❌ Risk assessment error: {e}")
            return False

    def run_ensemble_analysis(
        self, ensemble_models: List[str], output_path: str
    ) -> bool:
        """Run ensemble-specific analysis"""
        logger.info("👥 Running Ensemble Analysis")

        try:
            # Analyze ensemble performance
            ensemble_report = "# SAC v431 Ensemble Analysis\n\n"

            for model_path in ensemble_models:
                model_name = Path(model_path).stem
                specialization = model_name.replace("sac_v431_ensemble_", "")

                ensemble_report += f"## {specialization.upper()} Model\n\n"

                # Run individual model analysis
                analysis_result = self.analysis_suite.run(
                    ["model", "evaluate", "--model", model_path]
                )

                if analysis_result == 0:
                    ensemble_report += (
                        f"✅ {specialization} model analysis completed\n\n"
                    )
                else:
                    ensemble_report += f"❌ {specialization} model analysis failed\n\n"

            # Save ensemble report
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(ensemble_report)

            logger.info("✅ Ensemble analysis completed")
            return True

        except Exception as e:
            logger.error(f"❌ Ensemble analysis error: {e}")
            return False

    def run_comprehensive_analysis(self, args: argparse.Namespace) -> bool:
        """Run complete analysis suite"""
        logger.info("🎯 Running SAC v431 Comprehensive Analysis")

        success = True

        # Generate training report
        if getattr(args, "training_report", True):
            training_report = self.generate_training_report()
            report_path = (
                getattr(args, "output", "reports") + "/sac_v431_training_report.md"
            )
            os.makedirs(os.path.dirname(report_path), exist_ok=True)

            with open(report_path, "w", encoding="utf-8") as f:
                f.write(training_report)

            logger.info(f"📋 Training report saved to {report_path}")

        # Run backtest analysis
        if hasattr(args, "backtest_results") and args.backtest_results:
            backtest_output = (
                getattr(args, "output", "reports") + "/sac_v431_backtest_analysis.md"
            )
            if not self.run_backtest_analysis(args.backtest_results, backtest_output):
                success = False

        # Run performance comparison
        if hasattr(args, "compare_models") and args.compare_models:
            comparison_output = (
                getattr(args, "output", "reports")
                + "/sac_v431_performance_comparison.md"
            )
            if not self.run_performance_comparison(
                args.compare_models, comparison_output
            ):
                success = False

        # Run risk assessment
        if hasattr(args, "risk_assessment") and args.risk_assessment:
            risk_output = (
                getattr(args, "output", "reports") + "/sac_v431_risk_assessment.md"
            )
            if not self.run_risk_assessment(args.risk_assessment, risk_output):
                success = False

        # Run ensemble analysis
        if hasattr(args, "ensemble_models") and args.ensemble_models:
            ensemble_output = (
                getattr(args, "output", "reports") + "/sac_v431_ensemble_analysis.md"
            )
            if not self.run_ensemble_analysis(args.ensemble_models, ensemble_output):
                success = False

        return success


def main():
    parser = argparse.ArgumentParser(description="SAC v431 Comprehensive Analysis")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v431/sac_v431_advanced.json",
        help="Configuration file path",
    )
    parser.add_argument(
        "--output", type=str, default="reports/sac_v431", help="Output directory"
    )
    parser.add_argument(
        "--backtest_results", type=str, help="Backtest results JSON file path"
    )
    parser.add_argument(
        "--compare_models", nargs="+", help="Models to compare (space-separated list)"
    )
    parser.add_argument(
        "--risk_assessment", type=str, help="Backtest results for risk assessment"
    )
    parser.add_argument(
        "--ensemble_models", nargs="+", help="Ensemble model paths for analysis"
    )
    parser.add_argument(
        "--training_report",
        action="store_true",
        default=True,
        help="Generate training report",
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = SACv431Analyzer(args.config)

    # Run comprehensive analysis
    success = analyzer.run_comprehensive_analysis(args)

    if success:
        logger.info("🎉 SAC v431 comprehensive analysis completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 SAC v431 comprehensive analysis failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
