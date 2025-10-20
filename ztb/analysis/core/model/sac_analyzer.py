#!/usr/bin/env python3
"""
SAC Analysis Suite - Comprehensive analysis tools for SAC models

This script provides unified analysis capabilities for SAC trading models including:
- Action distribution analysis
- Performance evaluation
- Reward function analysis
- Model comparison
- Bias detection and correction
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.utils.logging_utils import get_logger
from ztb.utils.path_utils import get_project_root

# Import SAC
try:
    from stable_baselines3 import SAC
except ImportError:
    SAC = None  # type: ignore

# Get project root using utility
project_root = get_project_root()

logger = get_logger(__name__)


@dataclass
class AnalysisResult:
    """Container for analysis results."""

    action_distribution: Dict[str, float]
    performance_metrics: Dict[str, float]
    bias_analysis: Dict[str, Any]
    recommendations: List[str]


class SACAnalyzer:
    """Comprehensive SAC model analyzer."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        samples: int = 10000,
    ):
        """
        Initialize SAC analyzer.

        Args:
            model_path: Path to SAC model file
            config_path: Path to configuration file
            samples: Number of samples for analysis
        """
        self.model_path = Path(model_path) if model_path else None
        self.config_path = Path(config_path) if config_path else None
        self.samples = samples
        self.model: Optional[SAC] = None
        self.config = None

        if self.model_path and self.model_path.exists():
            self.load_model()

        if self.config_path and self.config_path.exists():
            self.load_config()

    def load_model(self) -> bool:
        """Load SAC model."""
        try:
            self.model = SAC.load(str(self.model_path))
            logger.info(f"Model loaded from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def load_config(self) -> bool:
        """Load configuration."""
        if self.config_path is None:
            logger.warning("No config path provided")
            return False
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            logger.info(f"Config loaded from {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False

    def analyze_action_distribution(
        self, num_samples: int = 10000, deterministic: bool = True
    ) -> Dict[str, float]:
        """
        Analyze action distribution of the SAC model.

        Args:
            num_samples: Number of samples to analyze
            deterministic: Whether to use deterministic policy

        Returns:
            Action distribution statistics
        """
        if not self.model:
            raise ValueError("Model not loaded")

        logger.info(f"Analyzing action distribution with {num_samples} samples")

        # Create dummy observations
        assert self.model is not None  # For mypy
        obs_shape = self.model.observation_space.shape
        if obs_shape is None:
            raise ValueError("Observation space shape is None")
        dummy_obs = np.random.rand(num_samples, obs_shape[0])

        # Normalize to realistic ranges
        dummy_obs[:, :4] = dummy_obs[:, :4] * 1000000 + 5000000  # Price data
        dummy_obs[:, 4:] = (dummy_obs[:, 4:] - 0.5) * 2  # Technical indicators

        actions_continuous = []

        for i in range(num_samples):
            action, _ = self.model.predict(dummy_obs[i], deterministic=deterministic)
            actions_continuous.append(action[0])

        actions = np.array(actions_continuous)

        # Analyze distribution
        buy_actions = np.sum(actions > 0.3333)
        sell_actions = np.sum(actions < -0.3333)
        hold_actions = np.sum((actions >= -0.3333) & (actions <= 0.3333))

        total_actions = len(actions)

        distribution = {
            "buy_ratio": float(buy_actions / total_actions),
            "sell_ratio": float(sell_actions / total_actions),
            "hold_ratio": float(hold_actions / total_actions),
            "total_samples": float(total_actions),
            "sell_bias_detected": float((sell_actions / total_actions) > 0.4),
        }

        logger.info(
            f"Action distribution: BUY={distribution['buy_ratio']:.3f}, "
            f"HOLD={distribution['hold_ratio']:.3f}, SELL={distribution['sell_ratio']:.3f}"
        )

        return distribution

    def analyze_bias_patterns(self, action_dist: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze bias patterns in action distribution.

        Args:
            action_dist: Action distribution from analyze_action_distribution

        Returns:
            Bias analysis results
        """
        sell_ratio = action_dist["sell_ratio"]
        buy_ratio = action_dist["buy_ratio"]
        hold_ratio = action_dist["hold_ratio"]

        bias_analysis = {
            "sell_bias": sell_ratio > 0.4,
            "buy_bias": buy_ratio > 0.4,
            "balanced_distribution": abs(sell_ratio - buy_ratio) < 0.1,
            "excessive_holding": hold_ratio > 0.6,
            "bias_severity": max(
                abs(sell_ratio - 0.3333),
                abs(buy_ratio - 0.3333),
                abs(hold_ratio - 0.3333),
            ),
        }

        return bias_analysis

    def generate_recommendations(
        self, action_dist: Dict[str, float], bias_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Generate recommendations based on analysis.

        Args:
            action_dist: Action distribution
            bias_analysis: Bias analysis results

        Returns:
            List of recommendations
        """
        recommendations = []

        if bias_analysis["sell_bias"]:
            recommendations.append("SELL bias detected. Consider:")
            recommendations.append(
                "  - Implement symmetric action thresholds (±0.3333)"
            )
            recommendations.append("  - Add sell penalty to reward function")
            recommendations.append(
                "  - Increase action balance weight in reward calculation"
            )

        if bias_analysis["buy_bias"]:
            recommendations.append("BUY bias detected. Consider:")
            recommendations.append("  - Add buy penalty to reward function")
            recommendations.append("  - Adjust action thresholds")

        if bias_analysis["excessive_holding"]:
            recommendations.append("Excessive holding detected. Consider:")
            recommendations.append("  - Reduce hold penalty in reward function")
            recommendations.append("  - Adjust hold range in action conversion")

        if not bias_analysis["balanced_distribution"]:
            recommendations.append("Unbalanced action distribution. Consider:")
            recommendations.append("  - Use SAC v429 with symmetric thresholds")
            recommendations.append("  - Run reward function optimization")

        if bias_analysis["bias_severity"] > 0.2:
            recommendations.append(
                "Severe bias detected. Immediate action recommended:"
            )
            recommendations.append("  - Switch to v429 training with --version v429")
            recommendations.append("  - Run comprehensive reward optimization")

        return recommendations

    def run_full_analysis(self) -> AnalysisResult:
        """
        Run complete analysis suite.

        Args:
            num_samples: Number of samples for analysis

        Returns:
            Complete analysis results
        """
        logger.info("Running full SAC analysis suite")

        # Action distribution analysis
        action_dist = self.analyze_action_distribution(self.samples)

        # Bias pattern analysis
        bias_analysis = self.analyze_bias_patterns(action_dist)

        # Generate recommendations
        recommendations = self.generate_recommendations(action_dist, bias_analysis)

        # Performance metrics (placeholder for now)
        performance_metrics = {
            "analysis_completed": 1.0,
            "samples_analyzed": float(self.samples),
            "model_loaded": 1.0 if self.model is not None else 0.0,
        }

        result = AnalysisResult(
            action_distribution=action_dist,
            performance_metrics=performance_metrics,
            bias_analysis=bias_analysis,
            recommendations=recommendations,
        )

        return result

    def print_results(self, result: AnalysisResult) -> None:
        """
        Print analysis results in a formatted way.

        Args:
            result: Analysis results to print
        """
        print("\n🔬 SAC Model Analysis Results")
        print("=" * 50)

        print("\n📊 Action Distribution:")
        for key, value in result.action_distribution.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        print("\n⚖️  Bias Analysis:")
        for key, value in result.bias_analysis.items():
            print(f"  {key}: {value}")

        print("\n💡 Recommendations:")
        for rec in result.recommendations:
            print(f"  {rec}")

        print("\n✅ Analysis complete!")

    def print_report(self, result: AnalysisResult) -> None:
        """Print formatted analysis report."""
        print("\n" + "=" * 60)
        print("SAC MODEL ANALYSIS REPORT")
        print("=" * 60)

        print("\n📊 ACTION DISTRIBUTION:")
        print(f"  BUY:  {result.action_distribution['buy_ratio']:.3f}")
        print(f"  HOLD: {result.action_distribution['hold_ratio']:.3f}")
        print(f"  SELL: {result.action_distribution['sell_ratio']:.3f}")
        print("  → BUY/HOLD/SELLの割合。理想的には33%前後でバランスが取れていること")

        print("\n🎯 BIAS ANALYSIS:")
        if result.bias_analysis["sell_bias"]:
            print("⚠️  SELL bias detected → 売りが40%以上。売られすぎの可能性")
        if result.bias_analysis["buy_bias"]:
            print("⚠️  BUY bias detected → 買いが40%以上。買われすぎの可能性")
        if result.bias_analysis["balanced_distribution"]:
            print("✅ Balanced distribution → 各アクションがバランスよく分布")
        if result.bias_analysis["excessive_holding"]:
            print("⚠️  Excessive holding → ホールドが60%以上。取引が少ない可能性")

        print(f"  Bias Severity: {result.bias_analysis['bias_severity']:.3f}")
        print("  → 理想分布からの乖離度。低いほどバランスが良い")

        print("\n💡 RECOMMENDATIONS:")
        for rec in result.recommendations:
            print(f"  • {rec}")

        print("\n" + "=" * 60)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SAC Analysis Suite")
    parser.add_argument("--model", type=str, help="Path to SAC model file")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--samples", type=int, default=10000, help="Number of samples to analyze"
    )
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = SACAnalyzer(args.model, args.config, args.samples)

    # Run analysis
    result = analyzer.run_full_analysis()

    # Print report
    analyzer.print_report(result)

    # Save results if requested
    if args.output:
        output_data = {
            "action_distribution": result.action_distribution,
            "performance_metrics": result.performance_metrics,
            "bias_analysis": result.bias_analysis,
            "recommendations": result.recommendations,
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n📄 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
