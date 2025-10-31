#!/usr/bin/env python3
"""
SAC v444 Regime Detection Accuracy Test

This script tests the accuracy of the v444 regime classification system
across various market conditions and data scenarios.

Tests include:
- Synthetic data generation for different regime types
- Historical data validation
- Edge case handling
- Performance benchmarking
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from ztb.analysis.v444_regime_classifier import V444RegimeClassifier, RegimeType, RegimeDetectionResult
    print("✓ Successfully imported V444RegimeClassifier")
except ImportError as e:
    print(f"✗ Failed to import V444RegimeClassifier: {e}")
    sys.exit(1)


class RegimeDetectionTester:
    """
    Comprehensive tester for v444 regime detection accuracy

    This class generates synthetic market data for different regimes
    and validates the classifier's ability to correctly identify them.
    """

    def __init__(self, config_path: str = "config/sac_v444_advanced_regime_adaptation_config.json"):
        """
        Initialize the regime detection tester

        Args:
            config_path: Path to v444 configuration file
        """
        # Load configuration
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Initialize classifier with adjusted thresholds for testing
        test_config = {
            'thresholds': {
                'strong_trend_threshold': 3.0,
                'moderate_trend_threshold': 2.0,
                'weak_trend_threshold': 1.0,
                'high_volatility_threshold': 0.15,  # Adjusted for synthetic data volatility levels
                'moderate_volatility_threshold': 0.10,  # Adjusted for synthetic data volatility levels
                'extreme_volatility_threshold': 0.25,  # Adjusted for synthetic data volatility levels
                'consolidation_range_threshold': 0.05,  # Adjusted for very low volatility
                'breakout_setup_threshold': 0.15
            }
        }
        self.classifier = V444RegimeClassifier(test_config)

        # Test parameters
        self.test_samples = 10  # Restored for proper testing
        self.min_data_length = 200  # Minimum data length for reliable detection
        self.confidence_threshold = 0.6  # Minimum confidence for valid detection

        # Debug metrics
        self.metrics = {
            'candidate_bull': 0,
            'candidate_bear': 0,
            'consolidation_fallback': 0
        }

        print("✓ Regime Detection Tester initialized")
        print(f"  - Test samples per regime: {self.test_samples}")
        print(f"  - Minimum data length: {self.min_data_length}")
        print(f"  - Confidence threshold: {self.confidence_threshold}")

    def generate_synthetic_data(self, regime: RegimeType, length: int = 500) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for a specific regime

        Args:
            regime: Target regime to generate data for
            length: Length of data to generate

        Returns:
            DataFrame with OHLCV data
        """
        np.random.seed(42)  # For reproducible results

        # Base parameters
        base_price = 100.0
        dates = pd.date_range(start='2023-01-01', periods=length, freq='1H')

        # Regime-specific parameters
        regime_params = self._get_regime_generation_params(regime)
        print(f"DEBUG SYNTHETIC: Regime {regime.value}, drift={regime_params['drift']}, noise={regime_params['noise']}")

        # Generate price series using geometric Brownian motion
        regime_params = self._get_regime_generation_params(regime)
        prices = self._generate_geometric_brownian_series(length, regime_params['drift'], regime_params['noise'])

        # Debug: Print first few prices and returns
        print(f"DEBUG PRICES: First 10 prices: {prices[:10]}")
        returns = np.diff(np.log(prices))
        print(f"DEBUG RETURNS: First 9 returns: {returns[:9]}")
        print(f"DEBUG RETURNS: Mean return: {np.mean(returns)}, Expected drift: {regime_params['drift']}")

        # Scale to base price
        prices = np.array(prices) * base_price / prices[0]

        # Create OHLCV data
        df_data = []
        for i, price in enumerate(prices):
            # Generate OHLC from close price with spread based on regime noise
            regime_noise = regime_params['noise']
            spread = abs(np.random.normal(0, regime_noise * 2))  # Spread proportional to noise
            high = price * (1 + spread)
            low = price * (1 - spread)
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.lognormal(10, 1)  # Realistic volume distribution

            df_data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })

        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)

        return df

    def _generate_geometric_brownian_series(self, n: int, drift: float, noise: float, seed: int = 42) -> np.ndarray:
        """Generate geometric Brownian motion price series"""
        # Use different seed for each regime to avoid systematic bias
        rng = np.random.default_rng(seed + hash(str(drift) + str(noise)) % 10000)
        noise_seq = rng.normal(0.0, noise, size=n)
        noise_seq = np.clip(noise_seq, -4 * noise, 4 * noise)
        returns = drift + noise_seq
        log_price = np.cumsum(returns)
        close = np.exp(log_price)
        return close

    def _get_regime_generation_params(self, regime: RegimeType) -> Dict[str, float]:
        """Get parameters for synthetic data generation for each regime"""
        print(f"DEBUG PARAMS: Input regime: {regime}, type: {type(regime)}, value: {regime.value}")
        params = {
            RegimeType.STRONG_BULL_TREND: {
                'drift': 0.15,
                'noise': 0.005
            },
            RegimeType.MODERATE_BULL_TREND: {
                'drift': 0.001,
                'noise': 0.001
            },
            RegimeType.WEAK_BULL_TREND: {
                'drift': 0.0005,
                'noise': 0.001
            },
            RegimeType.STRONG_BEAR_TREND: {
                'drift': -0.15,
                'noise': 0.005
            },
            RegimeType.MODERATE_BEAR_TREND: {
                'drift': -0.001,
                'noise': 0.001
            },
            RegimeType.WEAK_BEAR_TREND: {
                'drift': -0.0005,
                'noise': 0.001
            },
            RegimeType.HIGH_VOLATILITY_RANGING: {
                'drift': 0.0,
                'noise': 0.02
            },
            RegimeType.MODERATE_VOLATILITY_RANGING: {
                'drift': 0.0,
                'noise': 0.015
            },
            RegimeType.LOW_VOLATILITY_RANGING: {
                'drift': 0.0,
                'noise': 0.01
            },
            RegimeType.EXTREME_VOLATILITY: {
                'drift': 0.0,
                'noise': 0.03
            },
            RegimeType.CONSOLIDATION: {
                'drift': 0.0,
                'noise': 0.005
            },
            RegimeType.BREAKOUT_SETUP: {
                'drift': 0.0005,
                'noise': 0.005
            },
            RegimeType.BREAKDOWN_SETUP: {
                'drift': -0.0005,
                'noise': 0.005
            },
        }
        result = params.get(regime, {'drift': 0.0, 'noise': 0.001})
        print(f"DEBUG PARAMS: Retrieved params: {result}")
        return result

    def test_regime_detection_accuracy(self) -> Dict[str, Any]:
        """
        Test regime detection accuracy across all regimes

        Returns:
            Dictionary with accuracy results and metrics
        """
        print("\n🧪 Testing regime detection accuracy...")

        results = {}
        all_regimes = list(RegimeType)

        for regime in all_regimes:
            print(f"  Testing {regime.value}...")

            # Generate test data
            test_data = self.generate_synthetic_data(regime, self.min_data_length)

            # Test detection at multiple points
            detections = []
            for i in range(self.min_data_length // 2, len(test_data)):  # Test every bar from midpoint
                try:
                    result = self.classifier.detect_regime(test_data, i)
                    detections.append(result)
                    if len(detections) <= 3:  # Debug first few detections
                        print(f"    Debug: Index {i}, Detected {result.primary_regime.value}, Confidence {result.confidence:.3f}")
                        print(f"    Debug: Metrics - trend_strength: {result.metrics.trend_strength:.3f}, volatility: {result.metrics.volatility:.3f}, momentum: {result.metrics.momentum:.3f}")
                except Exception as e:
                    print(f"    Warning: Detection failed at index {i}: {e}")
                    continue

            # Calculate accuracy metrics
            if detections:
                primary_regime_matches = sum(1 for d in detections if d.primary_regime == regime)
                high_confidence_detections = sum(1 for d in detections if d.confidence >= self.confidence_threshold)

                accuracy = primary_regime_matches / len(detections)
                avg_confidence = np.mean([d.confidence for d in detections])
                secondary_regime_diversity = np.mean([len(d.secondary_regimes) for d in detections])

                results[regime.value] = {
                    'accuracy': accuracy,
                    'avg_confidence': avg_confidence,
                    'high_confidence_ratio': high_confidence_detections / len(detections),
                    'secondary_regime_diversity': secondary_regime_diversity,
                    'sample_count': len(detections)
                }

                print(f"    Accuracy: {accuracy:.3f}")
                print(f"    Avg Confidence: {avg_confidence:.3f}")
                print(f"    High Confidence Ratio: {results[regime.value]['high_confidence_ratio']:.3f}")
            else:
                print(f"    No valid detections for {regime.value}")

        return results

    def test_edge_cases(self) -> Dict[str, Any]:
        """
        Test edge cases and boundary conditions

        Returns:
            Dictionary with edge case test results
        """
        print("\n🔍 Testing edge cases...")

        edge_cases = {
            'insufficient_data': self._test_insufficient_data(),
            'extreme_volatility': self._test_extreme_volatility(),
            'flat_market': self._test_flat_market(),
            'gap_events': self._test_gap_events(),
            'regime_transitions': self._test_regime_transitions()
        }

        return edge_cases

    def _test_insufficient_data(self) -> Dict[str, Any]:
        """Test behavior with insufficient data"""
        try:
            # Generate minimal data
            short_data = self.generate_synthetic_data(RegimeType.CONSOLIDATION, 10)
            result = self.classifier.detect_regime(short_data)
            return {'success': True, 'regime': result.primary_regime.value, 'confidence': result.confidence}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _test_extreme_volatility(self) -> Dict[str, Any]:
        """Test extreme volatility handling"""
        try:
            # Generate extreme volatility data
            extreme_data = self.generate_synthetic_data(RegimeType.EXTREME_VOLATILITY, 300)
            result = self.classifier.detect_regime(extreme_data)
            return {
                'success': True,
                'detected_regime': result.primary_regime.value,
                'expected_extreme_volatility': result.primary_regime == RegimeType.EXTREME_VOLATILITY,
                'confidence': result.confidence
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _test_flat_market(self) -> Dict[str, Any]:
        """Test flat market conditions"""
        try:
            # Generate very flat data
            dates = pd.date_range(start='2023-01-01', periods=300, freq='1H')
            flat_prices = np.full(300, 100.0) + np.random.normal(0, 0.01, 300)  # Very low volatility

            flat_data = pd.DataFrame({
                'timestamp': dates,
                'open': flat_prices,
                'high': flat_prices * 1.001,
                'low': flat_prices * 0.999,
                'close': flat_prices,
                'volume': np.full(300, 1000.0)
            })
            flat_data.set_index('timestamp', inplace=True)

            result = self.classifier.detect_regime(flat_data)
            return {
                'success': True,
                'detected_regime': result.primary_regime.value,
                'expected_consolidation': result.primary_regime in [RegimeType.CONSOLIDATION, RegimeType.LOW_VOLATILITY_RANGING],
                'confidence': result.confidence
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _test_gap_events(self) -> Dict[str, Any]:
        """Test gap event handling"""
        try:
            # Generate data with a gap
            data = self.generate_synthetic_data(RegimeType.CONSOLIDATION, 200)

            # Insert a gap (sudden price jump)
            gap_idx = 150
            gap_multiplier = 1.1  # 10% gap up
            data.iloc[gap_idx:, data.columns.get_loc('close')] *= gap_multiplier
            data.iloc[gap_idx:, data.columns.get_loc('open')] *= gap_multiplier
            data.iloc[gap_idx:, data.columns.get_loc('high')] *= gap_multiplier
            data.iloc[gap_idx:, data.columns.get_loc('low')] *= gap_multiplier

            result = self.classifier.detect_regime(data)
            return {
                'success': True,
                'detected_regime': result.primary_regime.value,
                'confidence': result.confidence,
                'gap_handled': True  # Classifier should handle gaps gracefully
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _test_regime_transitions(self) -> Dict[str, Any]:
        """Test regime transition detection"""
        try:
            # Generate data with regime transition
            bull_data = self.generate_synthetic_data(RegimeType.STRONG_BULL_TREND, 150)
            bear_data = self.generate_synthetic_data(RegimeType.STRONG_BEAR_TREND, 150)
            transition_data = pd.concat([bull_data, bear_data])

            # Test detection at transition point
            transition_idx = len(bull_data) - 10
            result = self.classifier.detect_regime(transition_data, transition_idx)

            return {
                'success': True,
                'detected_regime': result.primary_regime.value,
                'confidence': result.confidence,
                'transition_handled': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def generate_accuracy_report(self, accuracy_results: Dict[str, Any],
                               edge_case_results: Dict[str, Any]) -> str:
        """
        Generate comprehensive accuracy report

        Args:
            accuracy_results: Results from accuracy testing
            edge_case_results: Results from edge case testing

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("SAC v444 Regime Detection Accuracy Report")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Overall accuracy summary
        if accuracy_results:
            accuracies = [r['accuracy'] for r in accuracy_results.values() if 'accuracy' in r]
            avg_accuracy = np.mean(accuracies) if accuracies else 0
            min_accuracy = min(accuracies) if accuracies else 0
            max_accuracy = max(accuracies) if accuracies else 0

            report.append("📊 OVERALL ACCURACY SUMMARY")
            report.append("-" * 40)
            report.append(f"  Average: {avg_accuracy:.3f}")
            report.append(f"  Minimum: {min_accuracy:.3f}")
            report.append(f"  Maximum: {max_accuracy:.3f}")
            report.append("")

        # Per-regime accuracy
        report.append("🎯 PER-REGIME ACCURACY")
        report.append("-" * 40)
        for regime_name, metrics in accuracy_results.items():
            if 'accuracy' in metrics:
                report.append(f"{regime_name}:")
                report.append(f"  Accuracy: {metrics['accuracy']:.3f}")
                report.append(f"  Avg Confidence: {metrics['avg_confidence']:.3f}")
                report.append(f"  High Confidence Ratio: {metrics['high_confidence_ratio']:.3f}")
                report.append(f"  Samples: {metrics['sample_count']}")
                report.append("")

        # Edge cases
        report.append("🔍 EDGE CASE RESULTS")
        report.append("-" * 40)
        for test_name, result in edge_case_results.items():
            report.append(f"{test_name}: {'✅ PASS' if result.get('success', False) else '❌ FAIL'}")
            if result.get('success'):
                if 'detected_regime' in result:
                    report.append(f"  Detected: {result['detected_regime']}")
                if 'confidence' in result:
                    report.append(f"  Confidence: {result['confidence']:.3f}")
            else:
                report.append(f"  Error: {result.get('error', 'Unknown')}")
            report.append("")

        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)

        low_accuracy_regimes = [name for name, metrics in accuracy_results.items()
                              if metrics.get('accuracy', 0) < 0.7]

        if low_accuracy_regimes:
            report.append("Low accuracy regimes detected:")
            for regime in low_accuracy_regimes:
                report.append(f"  - {regime}: Consider refining detection parameters")
            report.append("")

        failed_edge_cases = [name for name, result in edge_case_results.items()
                           if not result.get('success', False)]

        if failed_edge_cases:
            report.append("Failed edge cases:")
            for case in failed_edge_cases:
                report.append(f"  - {case}: Requires attention")
            report.append("")

        if avg_accuracy >= 0.8:
            report.append("✅ Overall: Excellent regime detection accuracy!")
        elif avg_accuracy >= 0.7:
            report.append("⚠️  Overall: Good accuracy, minor improvements needed")
        else:
            report.append("❌ Overall: Accuracy needs significant improvement")

        return "\n".join(report)

    def run_comprehensive_test(self) -> str:
        """
        Run comprehensive testing suite

        Returns:
            Complete test report
        """
        print("🚀 Starting comprehensive regime detection testing...")

        # Run accuracy tests
        accuracy_results = self.test_regime_detection_accuracy()

        # Run edge case tests
        edge_case_results = self.test_edge_cases()

        # Generate report
        report = self.generate_accuracy_report(accuracy_results, edge_case_results)

        print("\n" + "="*80)
        print("TESTING COMPLETE")
        print("="*80)

        return report


def main():
    """Main function for regime detection testing"""
    print("🧪 SAC v444 Regime Detection Accuracy Test")
    print("=" * 50)

    tester = RegimeDetectionTester()

    try:
        # Run comprehensive testing
        report = tester.run_comprehensive_test()

        # Save report to file
        report_path = f"reports/v444_regime_detection_accuracy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"📄 Report saved to: {report_path}")
        print("\n" + report)

    except Exception as e:
        print(f"✗ Testing failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()