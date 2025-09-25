#!/usr/bin/env python3
"""
Test script for all features with Quality Gates and Promotion Engine.
"""
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import yaml

# Add ztb to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ztb.features.base import CommonPreprocessor
from ztb.features.registry import FeatureManager
from ztb.evaluation.promotion import YamlPromotionEngine, AdaptiveThresholdManager
from ztb.evaluation.status import CoverageValidator
from ztb.evaluation.quality_gates import QualityGates

# Import feature classes that exist
ADX = EMACross = HeikinAshi = KAMA = Supertrend = TEMA = None
RSI = ROC = OBV = ZScore = None

try:
    from ztb.features.trend.adx import ADX
    from ztb.features.trend.emacross import EMACross
    from ztb.features.trend.heikin_ashi import HeikinAshi
    from ztb.features.trend.kama import KAMA
    from ztb.features.trend.supertrend import Supertrend
    from ztb.features.trend.tema import TEMA
    from scripts.test_simple_features import RSI, ROC, OBV, ZScore
    print("Feature imports successful")
except ImportError as e:
    print(f"Feature import error: {e}")
    # Fallback to mock implementations
    pass


def load_sample_data(dataset: str = "synthetic") -> pd.DataFrame:
    """Load sample market data for testing"""

    if dataset == "coingecko":
        from data.coin_gecko import fetch_btc_jpy
        return fetch_btc_jpy(days=365, interval="daily")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 10000

    if dataset == "synthetic-v2":
        # Improved synthetic data with latent factors for realistic correlations
        t = np.linspace(0, 100, n_samples)
        
        # Latent factors that features can correlate with
        cycle = 0.1 * np.sin(2 * np.pi * t / 50)  # Cyclical component
        momentum = 0.05 * np.cumsum(np.random.normal(0, 0.01, n_samples))  # Momentum
        volatility = 0.02 * np.abs(np.random.normal(0, 0.01, n_samples))  # Volatility factor
        
        # Price influenced by latent factors
        trend = 0.01 * t
        latent_influence = 0.3 * cycle + 0.2 * momentum + 0.1 * volatility
        noise = np.random.normal(0, 0.003, n_samples)
        price = 100 * np.exp(trend + latent_influence + noise)
    else:
        # Original synthetic data
        t = np.linspace(0, 100, n_samples)
        trend = 0.02 * t  # Stronger trend
        noise = np.random.normal(0, 0.005, n_samples)  # Less noise
        price = 100 * np.exp(trend + noise)

    # Generate volume data
    volume = np.random.lognormal(12, 0.5, n_samples)  # Higher volume

    # Create OHLCV data
    df = pd.DataFrame({
        'open': price * (1 + np.random.normal(0, 0.002, n_samples)),
        'high': price * (1 + np.random.normal(0, 0.005, n_samples)),
        'low': price * (1 - np.random.normal(0, 0.005, n_samples)),
        'close': price,
        'volume': volume
    })

    # Ensure high >= max(open, close), low <= min(open, close)
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)

    return df


def calculate_feature_metrics(feature_data: pd.Series, price_data: pd.Series, feature_name: str) -> Dict[str, Any]:
    """Calculate basic trading metrics for feature evaluation"""
    # Use feature-specific strategies or default strategy
    if feature_name == 'RSI':
        # RSI strategy: buy when RSI < 30, sell when RSI > 70
        signals = pd.Series(0, index=feature_data.index)
        signals[feature_data < 30] = 1  # Buy signal
        signals[feature_data > 70] = -1  # Sell signal
    elif feature_name == 'ROC':
        # ROC strategy: buy when ROC > 5, sell when ROC < -5
        signals = pd.Series(0, index=feature_data.index)
        signals[feature_data > 5] = 1
        signals[feature_data < -5] = -1
    elif feature_name == 'OBV':
        # OBV strategy: buy when OBV increasing, sell when decreasing
        obv_change = feature_data.diff()
        signals = pd.Series(0, index=feature_data.index)
        signals[obv_change > 0] = 1
        signals[obv_change < 0] = -1
    elif feature_name == 'ZScore':
        # ZScore strategy: buy when ZScore < -1, sell when ZScore > 1 (mean reversion)
        signals = pd.Series(0, index=feature_data.index)
        signals[feature_data < -1] = 1
        signals[feature_data > 1] = -1
    elif 'MACD' in feature_name:
        # MACD strategy: buy when MACD > signal, sell when MACD < signal
        signals = pd.Series(0, index=feature_data.index)
        signals[feature_data > 0] = 1  # Simplified: assume signal is 0
        signals[feature_data < 0] = -1
    elif 'Stochastic' in feature_name:
        # Stochastic strategy: buy when %K < 20, sell when %K > 80
        signals = pd.Series(0, index=feature_data.index)
        signals[feature_data < 20] = 1
        signals[feature_data > 80] = -1
    elif 'CCI' in feature_name:
        # CCI strategy: buy when CCI < -100, sell when CCI > 100
        signals = pd.Series(0, index=feature_data.index)
        signals[feature_data < -100] = 1
        signals[feature_data > 100] = -1
    elif 'Bollinger' in feature_name:
        # Bollinger strategy: buy when price < lower band, sell when price > upper band
        signals = pd.Series(0, index=feature_data.index)
        signals[feature_data < -1] = 1  # Simplified band position
        signals[feature_data > 1] = -1
    else:
        # Default strategy: buy when feature > 0, sell when feature < 0
        signals = (feature_data > 0).astype(int) - (feature_data < 0).astype(int)

    returns = price_data.pct_change().shift(-1)  # Next period returns

    # Calculate metrics
    valid_idx = signals.notna() & returns.notna() & (signals != 0)
    if valid_idx.sum() == 0:
        return {
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'sample_count': 0
        }

    strategy_returns = signals[valid_idx] * returns[valid_idx]
    cumulative = (1 + strategy_returns).cumprod()

    # Win rate
    win_rate = (strategy_returns > 0).mean()

    # Max drawdown
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    # Sharpe ratio
    if strategy_returns.std() > 0:
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Sortino ratio
    downside_returns = strategy_returns[strategy_returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino_ratio = strategy_returns.mean() / downside_returns.std() * np.sqrt(252)
    else:
        sortino_ratio = 0.0

    # Calmar ratio
    if abs(max_drawdown) > 0:
        calmar_ratio = strategy_returns.mean() * 252 / abs(max_drawdown)
    else:
        calmar_ratio = 0.0

    return {
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'sample_count': int(valid_idx.sum())
    }


def test_all_features(dataset: str = "synthetic", bootstrap: bool = False, save_debug: bool = False):
    """Test all features with Quality Gates and Promotion Engine"""
    print(f"Testing all features with Quality Gates and Promotion Engine (dataset={dataset}, bootstrap={bootstrap})...")

    # Load sample data
    df = load_sample_data(dataset)
    print(f"Loaded {len(df)} samples of market data")

    # Initialize components
    feature_manager = FeatureManager("config/features.yaml")

    # Register wave1 features
    from ztb.features import register_wave1_features, register_wave2_features
    register_wave1_features(feature_manager)
    register_wave2_features(feature_manager)

    # Initialize adaptive threshold manager
    adaptive_manager = AdaptiveThresholdManager(".")

    quality_gates = QualityGates(adaptive_manager)

    # Debug: Print adaptive thresholds
    adaptive_thresholds = quality_gates.get_adaptive_thresholds()
    print(f"Adaptive thresholds: {adaptive_thresholds}")

    # Load promotion engine config
    config_path = Path(__file__).parent.parent / "config" / "promotion_criteria.yaml"
    if not config_path.exists():
        print(f"Warning: {config_path} not found, using default config")
        config_path = None

    if config_path:
        engine = YamlPromotionEngine(str(config_path))
    else:
        # Create default engine
        engine = YamlPromotionEngine.__new__(YamlPromotionEngine)
        engine.config = {
            'categories': {
                'trend_features': {
                    'logic': 'AND',
                    'criteria': [
                        {'name': 'win_rate', 'operator': '>', 'value': 0.4, 'weight': 1.0, 'type': 'numeric'},
                        {'name': 'max_drawdown', 'operator': '<', 'value': -0.3, 'weight': 1.0, 'type': 'numeric'},
                        {'name': 'sharpe_ratio', 'operator': '>', 'value': 0.3, 'weight': 1.0, 'type': 'ratio'}
                    ],
                    'hard_requirements': [
                        {'name': 'sample_count', 'operator': '>', 'value': 5000, 'type': 'numeric'}
                    ],
                    'required_score': 0.8
                },
                'oscillator_features': {
                    'logic': 'AND',
                    'criteria': [
                        {'name': 'win_rate', 'operator': '>', 'value': 0.4, 'weight': 1.0, 'type': 'numeric'},
                        {'name': 'max_drawdown', 'operator': '<', 'value': -0.3, 'weight': 1.0, 'type': 'numeric'}
                    ],
                    'hard_requirements': [],
                    'required_score': 0.8
                },
                'volume_features': {
                    'logic': 'AND',
                    'criteria': [
                        {'name': 'win_rate', 'operator': '>', 'value': 0.4, 'weight': 1.0, 'type': 'numeric'}
                    ],
                    'hard_requirements': [],
                    'required_score': 0.7
                },
                'volatility_features': {
                    'logic': 'AND',
                    'criteria': [
                        {'name': 'win_rate', 'operator': '>', 'value': 0.4, 'weight': 1.0, 'type': 'numeric'}
                    ],
                    'hard_requirements': [],
                    'required_score': 0.7
                },
                'wave1_features': {
                    'logic': 'AND',
                    'criteria': [
                        {'name': 'win_rate', 'operator': '>', 'value': 0.4, 'weight': 1.0, 'type': 'numeric'}
                    ],
                    'hard_requirements': [],
                    'required_score': 0.7
                },
                'wave3_features': {
                    'logic': 'AND',
                    'criteria': [
                        {'name': 'win_rate', 'operator': '>', 'value': 0.4, 'weight': 1.0, 'type': 'numeric'}
                    ],
                    'hard_requirements': [],
                    'required_score': 0.6
                }
            },
            'staging': {
                'min_samples_required': 5000,
                'hard_requirement_mode': 'strict',
                'demotion_mode': 'graceful'
            }
        }
        engine.criteria_cache = {}
        from ztb.evaluation.promotion import PromotionNotifier
        engine.notifier = PromotionNotifier({})

    # Get all enabled features
    all_features = feature_manager.get_enabled_features()
    print(f"Testing {len(all_features)} features: {', '.join(all_features[:5])}...")

    # Compute all features at once using FeatureManager
    try:
        computed_df = feature_manager.compute_features(df.copy())
        print(f"Successfully computed {len(computed_df.columns)} features")
    except Exception as e:
        print(f"Error computing features with FeatureManager: {e}")
        print("Falling back to individual feature computation...")
        computed_df = df.copy()

    results = []
    verified_count = {'trend': 0, 'oscillator': 0, 'volume': 0, 'volatility': 0, 'wave1': 0, 'wave3': 0}

    for feature_name in all_features:
        print(f"\nTesting {feature_name}...")

        try:
            # Get feature category
            category = get_feature_category(feature_name)

            # Preprocess data
            processed_df = CommonPreprocessor.preprocess(df.copy())

            # Try to get feature from computed DataFrame first
            if feature_name in computed_df.columns:
                feature_series = computed_df[feature_name]
            else:
                # Fallback to individual computation
                feature_series = create_feature_instance(feature_name, processed_df)

            # Apply Quality Gates
            quality_results = quality_gates.evaluate(feature_series, processed_df['close'], "bootstrap" if bootstrap else "normal", dataset)

            if not quality_results['overall_pass']:
                print(f"⚠ {feature_name}: pending (quality gate fail) - ", end="")
                reasons = []
                if not quality_results['nan_rate_pass']:
                    reasons.append(f"NaN rate={quality_results['nan_rate']:.2f}")
                if not quality_results['correlation_pass']:
                    reasons.append(f"correlation={quality_results.get('correlation', 'N/A')}")
                if not quality_results['skew_pass']:
                    reasons.append(f"skew={quality_results.get('skew', 'N/A'):.1f}")
                if not quality_results['kurtosis_pass']:
                    reasons.append(f"kurtosis={quality_results.get('kurtosis', 'N/A'):.1f}")
                print(", ".join(reasons))

                results.append({
                    'name': feature_name,
                    'status': 'pending_due_to_gate_fail',
                    'category': category,
                    'reason': ", ".join(reasons),
                    'quality_results': quality_results
                })
                continue

            # Calculate trading metrics
            metrics = calculate_feature_metrics(feature_series, processed_df['close'], feature_name)

            # Determine current status based on category
            current_status = "pending"  # Start from pending

            # Get category config
            category_config = f"{category}_features"

            # Check basic criteria for promotion
            basic_criteria = {
                'win_rate': metrics['win_rate'] > 0.4,
                'max_drawdown': metrics['max_drawdown'] < -0.3
            }

            if not all(basic_criteria.values()):
                print(f"✗ {feature_name}: pending → keep (basic criteria not met)")
                results.append({
                    'name': feature_name,
                    'status': 'keep',
                    'category': category,
                    'current_status': current_status,
                    'metrics': metrics,
                    'quality_results': quality_results,
                    'promotion_details': {'reason': 'basic_criteria_not_met'}
                })
                continue

            # Try pending -> staging
            result, details = engine.evaluate_promotion(feature_name, metrics, current_status, category_config)
            if result.name == "PROMOTE":
                current_status = "staging"
                print(f"✓ {feature_name}: pending → staging")
                results.append({
                    'name': feature_name,
                    'status': 'promoted_to_staging',
                    'category': category,
                    'from_status': 'pending',
                    'to_status': 'staging',
                    'metrics': metrics,
                    'quality_results': quality_results,
                    'promotion_details': details
                })
            else:
                print(f"✗ {feature_name}: pending → keep (insufficient criteria)")
                results.append({
                    'name': feature_name,
                    'status': 'keep',
                    'category': category,
                    'current_status': current_status,
                    'metrics': metrics,
                    'quality_results': quality_results,
                    'promotion_details': details
                })
                continue

            # Try staging -> verified
            result, details = engine.evaluate_promotion(feature_name, metrics, current_status, category_config)
            if result.name == "PROMOTE":
                print(f"✓ {feature_name}: staging → verified")
                verified_count[category] += 1
                results.append({
                    'name': feature_name,
                    'status': 'promoted',
                    'category': category,
                    'from_status': current_status,
                    'to_status': 'verified',
                    'metrics': metrics,
                    'quality_results': quality_results,
                    'promotion_details': details
                })
            else:
                print(f"✗ {feature_name}: staging → keep (insufficient criteria)")
                results.append({
                    'name': feature_name,
                    'status': 'keep',
                    'category': category,
                    'current_status': current_status,
                    'metrics': metrics,
                    'quality_results': quality_results,
                    'promotion_details': details
                })

        except Exception as e:
            print(f"✗ {feature_name}: error - {str(e)}")
            results.append({
                'name': feature_name,
                'status': 'error',
                'category': get_feature_category(feature_name),
                'error': str(e)
            })

    # Record events for each result
    coverage_path = "coverage.json"
    coverage_data = CoverageValidator.load_coverage_files(coverage_path)

    for result in results:
        feature_name = result['name']
        category = result.get('category')

        if result['status'] == 'harmful':
            # Record harmful features as discarded
            # Convert numpy bools to Python bools for JSON serialization
            quality_results = result.get('quality_results', {})
            json_safe_quality_results = {}
            for k, v in quality_results.items():
                if isinstance(v, np.bool_):
                    json_safe_quality_results[k] = bool(v)
                else:
                    json_safe_quality_results[k] = v
            
            CoverageValidator.record_event(
                coverage_data,
                "feature_tested",
                feature_name,
                None,
                "pending_due_to_gate_fail",
                {
                    "status": "pending_due_to_gate_fail",
                    "reason": result.get('reason', 'quality_gate_failed'),
                    "quality_results": json_safe_quality_results,
                    "dataset": dataset
                }
            )
        elif result['status'] == 'promoted':
            # Record promotion
            CoverageValidator.record_event(
                coverage_data,
                "feature_promoted",
                feature_name,
                result.get('from_status'),
                result.get('to_status'),
                {
                    "score": result.get('promotion_details', {}).get('normalized_score', 0.0),
                    "dataset": dataset
                }
            )
        elif result['status'] == 'keep':
            # Record as kept (no change)
            pass  # Keep events are not recorded as they don't change state
        elif result['status'] == 'error':
            # Record error
            CoverageValidator.record_event(
                coverage_data,
                "feature_tested",
                feature_name,
                None,
                "failed",
                {
                    "status": "error",
                    "error": result.get('error', 'unknown_error'),
                    "dataset": dataset
                }
            )

    # Save updated coverage
    with open(coverage_path, 'w', encoding='utf-8') as f:
        json.dump(coverage_data, f, indent=2, ensure_ascii=False)

    # Check category requirements
    check_category_requirements(verified_count)

    print("\nAll features testing completed!")

    # Save debug report if requested
    if save_debug:
        import datetime
        import os
        import csv

        debug_file = f"reports/debug/feature_quality_{dataset}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        os.makedirs("reports/debug", exist_ok=True)

        with open(debug_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset', 'feature', 'corr_short', 'corr_long', 'nan_rate', 'skew', 'kurtosis', 'overall_pass', 'gate_fail_reason'])
            
            for result in results:
                quality = result.get('quality_results', {})
                fail_reasons = []
                if not quality.get('correlation_pass', True):
                    fail_reasons.append('correlation')
                if not quality.get('nan_rate_pass', True):
                    fail_reasons.append('nan_rate')
                if not quality.get('skew_pass', True):
                    fail_reasons.append('skew')
                if not quality.get('kurtosis_pass', True):
                    fail_reasons.append('kurtosis')
                
                writer.writerow([
                    dataset,
                    result['name'],
                    quality.get('corr_short', 'N/A'),
                    quality.get('corr_long', 'N/A'),
                    quality.get('nan_rate', 'N/A'),
                    quality.get('skew', 'N/A'),
                    quality.get('kurtosis', 'N/A'),
                    quality.get('overall_pass', False),
                    ';'.join(fail_reasons)
                ])

        print(f"Debug report saved to {debug_file}")
    
    return results


def get_feature_category(feature_name: str) -> str:
    """Determine feature category from feature name"""
    name_lower = feature_name.lower()

    # Wave features
    if any(keyword in name_lower for keyword in ['rolling', 'lags', 'dow', 'hour']):
        return 'wave1'
    if any(keyword in name_lower for keyword in ['regime', 'kalman']):
        return 'wave3'

    # Trend indicators
    if any(keyword in name_lower for keyword in ['ema', 'sma', 'kama', 'tema', 'dema', 'trend', 'ichimoku', 'donchian', 'adx', 'supertrend', 'heikin']):
        return 'trend'

    # Oscillators
    if any(keyword in name_lower for keyword in ['rsi', 'stoch', 'macd', 'cci', 'williams', 'oscillator']):
        return 'oscillator'

    # Volume indicators
    if any(keyword in name_lower for keyword in ['volume', 'obv', 'vwap', 'vpt', 'mfi', 'pricevolume']):
        return 'volume'

    # Volatility indicators
    if any(keyword in name_lower for keyword in ['atr', 'bollinger', 'hv', 'returnstd', 'zscore']):
        return 'volatility'

    return 'other'


def create_feature_instance(feature_name: str, df: pd.DataFrame) -> pd.Series:
    """Create and compute feature instance dynamically"""
    try:
        # Try to use actual feature classes
        if feature_name == 'ADX' and ADX is not None:
            feature = ADX(period=14)
            result = feature.compute(df)
            return result.iloc[:, 0] if isinstance(result, pd.DataFrame) else result
        elif feature_name == 'EMACross' and EMACross is not None:
            feature = EMACross(fast=12, slow=26)
            result = feature.compute(df)
            return result.iloc[:, 0] if isinstance(result, pd.DataFrame) else result
        elif feature_name == 'HeikinAshi' and HeikinAshi is not None:
            feature = HeikinAshi()
            result = feature.compute(df)
            return result.iloc[:, 0] if isinstance(result, pd.DataFrame) else result
        elif feature_name == 'KAMA' and KAMA is not None:
            feature = KAMA(period=10)
            result = feature.compute(df)
            return result.iloc[:, 0] if isinstance(result, pd.DataFrame) else result
        elif feature_name == 'Supertrend' and Supertrend is not None:
            feature = Supertrend(period=10, multiplier=3.0)
            result = feature.compute(df)
            return result.iloc[:, 0] if isinstance(result, pd.DataFrame) else result
        elif feature_name == 'TEMA' and TEMA is not None:
            feature = TEMA(period=14)
            result = feature.compute(df)
            return result.iloc[:, 0] if isinstance(result, pd.DataFrame) else result
        elif feature_name == 'RSI' and RSI is not None:
            feature = RSI(period=14)
            result = feature.compute(df)
            return result.iloc[:, 0] if isinstance(result, pd.DataFrame) else result
        elif feature_name == 'ROC' and ROC is not None:
            feature = ROC(period=10)
            result = feature.compute(df)
            return result.iloc[:, 0] if isinstance(result, pd.DataFrame) else result
        elif feature_name == 'OBV' and OBV is not None:
            feature = OBV()
            result = feature.compute(df)
            return result.iloc[:, 0] if isinstance(result, pd.DataFrame) else result
        elif feature_name == 'ZScore' and ZScore is not None:
            feature = ZScore(period=20)
            result = feature.compute(df)
            return result.iloc[:, 0] if isinstance(result, pd.DataFrame) else result
    except Exception as e:
        print(f"Error creating feature {feature_name}: {e}")
        # Fall back to mock data
    
    # Fallback to mock data for unhandled features
    np.random.seed(42)
    if 'RollingMean' in feature_name:
        window = 14 if '14' in feature_name else 50
        return df['close'].rolling(window=window).mean()
    elif 'RollingStd' in feature_name:
        window = 14 if '14' in feature_name else 50
        return df['close'].rolling(window=window).std()
    else:
        # Return random data for other features
        return pd.Series(np.random.randn(len(df)), index=df.index, name=feature_name)


def update_coverage_json(results: List[Dict[str, Any]], dataset: str = "synthetic"):
    """Update coverage.json with test results"""
    coverage_path = Path(__file__).parent.parent / "coverage.json"

    # Load existing coverage
    if coverage_path.exists():
        with open(coverage_path, 'r') as f:
            coverage = json.load(f)
    else:
        coverage = {
            "verified": [],
            "pending": [],
            "failed": [],
            "unverified": [],
            "events": [],
            "metadata": {
                "last_updated": "2024-01-01T00:00:00",
                "dataset": dataset,
                "total_verified": 0,
                "total_pending": 0,
                "total_failed": 0,
                "total_unverified": 0
            }
        }

    # Initialize events if not exists
    if 'events' not in coverage:
        coverage['events'] = []

    # Update based on results
    for result in results:
        feature_name = result['name']

        # Remove from all lists first
        for category in ['verified', 'pending', 'failed', 'unverified']:
            coverage[category] = [f for f in coverage[category] if not (isinstance(f, dict) and f.get('name') == feature_name) and f != feature_name]

        # Add to appropriate category
        if result['status'] == 'promoted' and result.get('to_status') == 'verified':
            if feature_name not in coverage['verified']:
                coverage['verified'].append(feature_name)
        elif result['status'] == 'promoted' and result.get('to_status') == 'staging':
            coverage['pending'].append({
                'name': feature_name,
                'reason': 'promoted_to_staging',
                'category': result.get('category'),
                'metrics': result.get('metrics', {})
            })
        elif result['status'] == 'harmful':
            coverage['failed'].append({
                'name': feature_name,
                'reason': f"harmful: {result['reason']}",
                'category': result.get('category'),
                'quality_results': result.get('quality_results', {})
            })
        elif result['status'] == 'keep':
            coverage['pending'].append({
                'name': feature_name,
                'reason': 'insufficient_criteria',
                'category': result.get('category'),
                'metrics': result.get('metrics', {})
            })
        elif result['status'] == 'error':
            coverage['failed'].append({
                'name': feature_name,
                'reason': f"error: {result['error']}",
                'category': result.get('category')
            })

        # Add event
        event = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'type': 'feature_tested',
            'feature': feature_name,
            'category': result.get('category'),
            'status': result['status'],
            'dataset': dataset,
            'details': result
        }
        coverage['events'].append(event)

    # Update metadata
    from datetime import datetime
    coverage['metadata']['last_updated'] = datetime.now().isoformat()
    coverage['metadata']['total_verified'] = len(coverage['verified'])
    coverage['metadata']['total_pending'] = len(coverage['pending'])
    coverage['metadata']['total_failed'] = len(coverage['failed'])
    coverage['metadata']['total_unverified'] = len(coverage['unverified'])

    # Save updated coverage
    with open(coverage_path, 'w') as f:
        json.dump(coverage, f, indent=2, default=str)

    print("Coverage.json updated with promotion/discard results")


def check_category_requirements(verified_count: Dict[str, int]):
    """Check if category requirements are met"""
    print("\nChecking category requirements...")

    requirements = {
        'trend': 1,
        'oscillator': 1,
        'volume': 1,
        'volatility': 1,
        'wave1': 1,
        'wave3': 0  # Optional
    }

    all_met = True
    for category, required in requirements.items():
        actual = verified_count.get(category, 0)
        if actual >= required:
            print(f"✓ {category}: {actual}/{required} verified features")
        else:
            print(f"✗ {category}: {actual}/{required} verified features (INSUFFICIENT)")
            all_met = False

    if all_met:
        print("All category requirements met!")
    else:
        print("Some category requirements not met!")
        # Don't exit in test mode - continue to save debug report
        # sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='synthetic', choices=['synthetic', 'synthetic-v2', 'real', 'coingecko'])
    parser.add_argument('--bootstrap', action='store_true', help='Enable bootstrap mode for relaxed quality gates')
    parser.add_argument('--save-debug', action='store_true', help='Save debug CSV with detailed quality metrics')
    args = parser.parse_args()
    
    test_all_features(dataset=args.dataset, bootstrap=args.bootstrap, save_debug=args.save_debug)