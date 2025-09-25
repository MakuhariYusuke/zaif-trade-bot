#!/usr/bin/env python3
"""
re_evaluate_features.py
Comprehensive re-evaluation of harmful features with extended analysis
"""

import sys
import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, TypedDict, List, Union
import re
import importlib
import inspect
import time
from datetime import datetime
import json


class EvaluationResult(TypedDict, total=False):
    """Type definition for feature evaluation results"""
    feature_name: str
    status: str
    computation_time_ms: float
    nan_rate: float
    total_columns: int
    aligned_periods: int
    baseline_sharpe: float
    cv_results: Dict[str, Any]
    feature_correlations: Dict[str, float]
    feature_performances: Dict[str, Any]
    best_delta_sharpe: float
    best_feature_name: str
    avg_correlation: float
    # Additional fields for error handling and classification
    error: str
    status_emoji: str
    reason_tag: str
    num_periods: int
    data_quality_score: float
    recommendation: str
    module_path: str
    # Pending status fields
    reason: str
    required_periods: int
    current_periods: int

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ztb.features.trend.ichimoku_ext import calculate_ichimoku_extended
from ztb.features.trend.donchian_ext import calculate_donchian_extended
from ztb.features.volatility.kalman_ext import calculate_kalman_extended
from ztb.metrics.metrics import calculate_all_metrics
from ztb.evaluation.preprocess import prepare_ohlc_data, SmartPreprocessor
from ztb.evaluation.cv import evaluate_with_cv
from ztb.evaluation.logging import EvaluationLogger


# Feature discovery system
def discover_feature_classes(module_path: str) -> Dict[str, Any]:
    """
    Dynamically discover feature classes from Python modules
    
    Args:
        module_path: Path to Python module (e.g., 'src.trading.features.experimental')
        
    Returns:
        Dictionary mapping feature names to feature classes
    """
    discovered_features = {}
    
    try:
        # Import the module
        module = importlib.import_module(module_path)
        
        # Find all classes in the module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Skip imported classes and base classes
            if obj.__module__ == module_path and name != 'BaseFeature':
                discovered_features[name] = obj
                
    except ImportError as e:
        print(f"Could not import {module_path}: {e}")
    except Exception as e:
        print(f"Error discovering features in {module_path}: {e}")
    
    return discovered_features


def evaluate_feature_class(feature_class: Any, ohlc_data: pd.DataFrame, 
                          feature_name: str) -> Union[EvaluationResult, Dict[str, Any]]:
    """
    Evaluate a single feature class with timing and performance metrics
    
    Args:
        feature_class: Feature class to evaluate
        ohlc_data: OHLC price data
        feature_name: Name of the feature for logging
        
    Returns:
        Evaluation results dictionary
    """
    try:
        # Prepare and align OHLC data
        ohlc_data = prepare_ohlc_data(ohlc_data)
        
        # Initialize feature instance
        feature_instance = feature_class()
        
        # Apply SmartPreprocessor for required calculations
        req = getattr(feature_instance, "required_calculations", set())
        if req:
            smart_prep = SmartPreprocessor(req)
            ohlc_data = smart_prep.preprocess(ohlc_data)
        
        # Measure computation time
        start_time = time.time()
        feature_df = feature_instance.compute(ohlc_data)
        computation_time = (time.time() - start_time) * 1000  # milliseconds
        
        if feature_df is None or feature_df.empty:
            return {
                'feature_name': feature_name,
                'status': 'empty_output',
                'computation_time_ms': computation_time,
                'error': 'Empty or None output'
            }
        
        # Calculate NaN rate
        total_values = feature_df.size
        nan_values = feature_df.isna().sum().sum()
        nan_rate = nan_values / total_values if total_values > 0 else 1.0
        
        # Calculate returns for evaluation
        returns = ohlc_data['close'].pct_change().dropna()
        
        # Align features with returns
        aligned_data = pd.concat([feature_df, returns.rename('returns')], axis=1).dropna()
        
        if len(aligned_data) < 20:
            return {
                'feature_name': feature_name,
                'status': 'pending',
                'reason': 'insufficient_data',
                'computation_time_ms': computation_time,
                'nan_rate': nan_rate,
                'total_columns': len(feature_df.columns),
                'aligned_periods': len(aligned_data),
                'required_periods': 20,
                'current_periods': len(aligned_data)
            }
        
        # Baseline metrics
        baseline_metrics = calculate_all_metrics(np.array(aligned_data['returns']))
        
        # Cross-validation evaluation
        cv_results = evaluate_with_cv(feature_df, aligned_data['returns'], n_splits=3, test_size=30)
        
        # Feature evaluation
        feature_correlations = {}
        feature_performances = {}
        
        for col in feature_df.columns:
            if col in aligned_data.columns and col != 'returns':
                feature_data = aligned_data[col]
                
                # Skip if too many NaN or constant values
                if feature_data.isna().sum() > len(feature_data) * 0.5:
                    continue
                if feature_data.nunique() <= 2:
                    continue
                
                # Correlation with returns
                correlation = feature_data.corr(aligned_data['returns'])
                feature_correlations[col] = correlation
                
                # Signal-based evaluation
                if feature_data.nunique() <= 10:  # Likely categorical/signal
                    # Use median as threshold for binary signal
                    signal_threshold = feature_data.median()
                    signal_periods = aligned_data[feature_data > signal_threshold]
                    
                    if len(signal_periods) > 10:
                        signal_returns = signal_periods['returns']
                        signal_metrics = calculate_all_metrics(np.array(signal_returns))
                        delta_sharpe = signal_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio']
                        
                        feature_performances[col] = {
                            'type': 'signal',
                            'delta_sharpe': delta_sharpe,
                            'signal_periods': len(signal_returns),
                            'signal_sharpe': signal_metrics['sharpe_ratio'],
                            'baseline_sharpe': baseline_metrics['sharpe_ratio']
                        }
                
                # Quantile-based evaluation for continuous features
                else:
                    try:
                        q75 = feature_data.quantile(0.75)
                        q25 = feature_data.quantile(0.25)
                        
                        top_periods = aligned_data[feature_data >= q75]
                        bottom_periods = aligned_data[feature_data <= q25]
                        
                        if len(top_periods) > 5 and len(bottom_periods) > 5:
                            top_metrics = calculate_all_metrics(np.array(top_periods['returns']))
                            bottom_metrics = calculate_all_metrics(np.array(bottom_periods['returns']))
                            
                            feature_performances[col] = {
                                'type': 'quantile',
                                'top_sharpe': top_metrics['sharpe_ratio'],
                                'bottom_sharpe': bottom_metrics['sharpe_ratio'],
                                'quantile_spread': top_metrics['sharpe_ratio'] - bottom_metrics['sharpe_ratio'],
                                'top_periods': len(top_periods),
                                'bottom_periods': len(bottom_periods)
                            }
                    except Exception as e:
                        print(f"Error in quantile evaluation for {col}: {e}")
        
        # Best performance metrics
        best_delta_sharpe = 0.0
        best_feature_name = None
        
        for fname, perf in feature_performances.items():
            if perf['type'] == 'signal':
                delta_sharpe_val = perf.get('delta_sharpe')
                delta_sharpe = abs(float(delta_sharpe_val if isinstance(delta_sharpe_val, (int, float)) else 0.0))
            else:
                quantile_spread_val = perf.get('quantile_spread')
                delta_sharpe = abs(float(quantile_spread_val if isinstance(quantile_spread_val, (int, float)) else 0.0))
            
            if delta_sharpe > best_delta_sharpe:
                best_delta_sharpe = delta_sharpe
                best_feature_name = fname
        
        result = {
            'feature_name': feature_name,
            'status': 'success',
            'computation_time_ms': computation_time,
            'nan_rate': nan_rate,
            'total_columns': len(feature_df.columns),
            'aligned_periods': len(aligned_data),
            'baseline_sharpe': baseline_metrics['sharpe_ratio'],
            'cv_results': cv_results,
            'feature_correlations': feature_correlations,
            'feature_performances': feature_performances,
            'best_delta_sharpe': best_delta_sharpe,
            'best_feature_name': best_feature_name,
            'avg_correlation': np.mean([abs(c) for c in feature_correlations.values()]) if feature_correlations else 0
        }
        
        # Log successful evaluation
        logger = EvaluationLogger()
        logger.log_evaluation(result)
        
        return result
        
    except Exception as e:
        error_result = {
            'feature_name': feature_name,
            'status': 'error',
            'computation_time_ms': 0,
            'nan_rate': 1.0,
            'total_columns': 0,
            'aligned_periods': 0,
            'error': str(e)
        }
        
        # Log error result
        logger = EvaluationLogger()
        logger.log_evaluation(error_result)
        
        return error_result


class ComprehensiveFeatureReEvaluator:
    """
    Comprehensive re-evaluation system for harmful and experimental features
    """
    
    def __init__(self, config_path: str = "config/features.yaml", 
                 harmful_path: str = "docs/features/harmful.md",
                 price_data_path: str = "price_cache.json"):
        self.config_path = config_path
        self.harmful_path = harmful_path
        self.price_data_path = price_data_path
        
        # Set cache to re-evaluation mode for shorter TTL
        try:
            from cache.sqlite_cache import SQLiteCache
            cache = SQLiteCache()
            cache.set_task_mode("re_evaluation")
            print("Cache optimized for re-evaluation tasks (TTL: 10 minutes)")
        except ImportError:
            print("SQLiteCache not available, using default caching")
        
        # I/O optimization: select only essential columns
        self.essential_columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'price', 'bid', 'ask', 'spread'
        ]
        
        self.config = self.load_config()
        self.harmful_info = self.load_harmful_info()
        self.price_data = self.load_price_data()
        
        # Extended evaluators - harmful features with enhanced analysis
        self.harmful_evaluators = {
            'Ichimoku': calculate_ichimoku_extended,
            'Donchian': calculate_donchian_extended,
            'KalmanFilter': calculate_kalman_extended
        }
        
        # Experimental/Wave feature modules to discover
        self.experimental_modules = [
            'src.trading.features.experimental',
            'src.trading.features.wave1', 
            'src.trading.features.wave2',
            'src.trading.features.wave3'
        ]
        
        self.results: Dict[str, Any] = {
            'harmful': {},
            'experimental': {},
            'wave': {}
        }
    
    def load_config(self) -> Dict[str, Any]:
        """Load features configuration"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            print(f"Config file not found: {self.config_path}")
            return {}
    
    def load_harmful_info(self) -> Dict[str, Dict]:
        """Load harmful features information from markdown"""
        harmful_info: Dict[str, Dict] = {}
        
        if not os.path.exists(self.harmful_path):
            print(f"Harmful features file not found: {self.harmful_path}")
            return harmful_info
        
        with open(self.harmful_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse harmful features from markdown
        feature_pattern = r'### (.+?)\n(.*?)(?=###|\Z)'
        matches = re.findall(feature_pattern, content, re.DOTALL)
        
        for feature_name, feature_content in matches:
            feature_name = feature_name.strip()
            
            # Extract status, reason, date if available
            status_match = re.search(r'status:\s*(\w+)', feature_content)
            reason_match = re.search(r'reason:\s*([^,\n]+)', feature_content)
            date_match = re.search(r'last_eval:\s*([^,\n]+)', feature_content)
            
            harmful_info[feature_name] = {
                'status': status_match.group(1) if status_match else 'unknown',
                'reason': reason_match.group(1).strip() if reason_match else 'unknown',
                'last_eval': date_match.group(1).strip() if date_match else 'unknown',
                'content': feature_content
            }
        
        return harmful_info
    
    def load_price_data(self) -> Optional[pd.DataFrame]:
        """Load price data with I/O optimization"""
        if not os.path.exists(self.price_data_path):
            print(f"Price data file not found: {self.price_data_path}")
            return None
        
        try:
            if self.price_data_path.endswith('.parquet'):
                # Optimized Parquet loading - only essential columns
                try:
                    import pyarrow.parquet as pq
                    parquet_file = pq.ParquetFile(self.price_data_path)
                    available_columns = set(parquet_file.schema.names)
                    
                    # Select only essential columns that exist
                    columns_to_load = [col for col in self.essential_columns 
                                     if col in available_columns]
                    
                    if columns_to_load:
                        df = pd.read_parquet(self.price_data_path, columns=columns_to_load)
                        print(f"Loaded {len(columns_to_load)}/{len(self.essential_columns)} essential columns from parquet")
                        return df
                    else:
                        print("No essential columns found in parquet, loading all columns")
                        return pd.read_parquet(self.price_data_path)
                except ImportError:
                    print("PyArrow not available, loading all columns")
                    return pd.read_parquet(self.price_data_path)
            else:
                # Default JSON loading
                return pd.read_json(self.price_data_path)
        except Exception as e:
            print(f"Error loading price data: {e}")
            return None
    
    def prepare_ohlc_data(self, symbol: str = "BTC/JPY") -> Optional[pd.DataFrame]:
        """Prepare OHLC data for feature calculation"""
        if self.price_data is None or symbol not in self.price_data.columns:
            print(f"Symbol {symbol} not found in price data")
            return None
        
        symbol_data = self.price_data[symbol]
        if not isinstance(symbol_data.iloc[0], dict):
            print(f"Invalid data structure for {symbol}")
            return None
        
        # Convert to DataFrame
        ohlc_data = []
        timestamps = []
        
        for timestamp, data in symbol_data.items():
            if isinstance(data, dict) and all(k in data for k in ['open', 'high', 'low', 'close', 'volume']):
                ohlc_data.append([
                    float(data['open']),
                    float(data['high']),
                    float(data['low']),
                    float(data['close']),
                    float(data['volume'])
                ])
                timestamps.append(pd.to_datetime(str(timestamp)))
        
        if len(ohlc_data) < 100:
            print(f"Insufficient OHLC data (got {len(ohlc_data)})")
            return None
        
        ohlc_df = pd.DataFrame(ohlc_data, 
                              columns=['open', 'high', 'low', 'close', 'volume'],
                              index=timestamps)
        ohlc_df.sort_index(inplace=True)
        
        return ohlc_df
    
    def evaluate_experimental_features(self, ohlc_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate experimental and wave features from Python modules
        
        Args:
            ohlc_data: OHLC price data
            
        Returns:
            Dictionary of evaluation results for experimental/wave features
        """
        all_results = {}
        
        for module_path in self.experimental_modules:
            print(f"\nEvaluating features from {module_path}...")
            
            # Discover feature classes in the module
            discovered_features = discover_feature_classes(module_path)
            
            if not discovered_features:
                print(f"No feature classes found in {module_path}")
                continue
            
            print(f"Found {len(discovered_features)} feature classes: {list(discovered_features.keys())}")
            
            # Evaluate each discovered feature
            for feature_name, feature_class in discovered_features.items():
                print(f"  Evaluating {feature_name}...")
                
                result = evaluate_feature_class(feature_class, ohlc_data, feature_name)
                
                # Classify the feature
                status_emoji, reason_tag, recommendation = self.classify_harmful_feature_status(result)
                
                result['status_emoji'] = status_emoji
                result['reason_tag'] = reason_tag  
                result['recommendation'] = recommendation
                result['module_path'] = module_path
                
                # Store result with module prefix for uniqueness
                module_name = module_path.split('.')[-1]  # experimental, wave1, etc.
                result_key = f"{module_name}.{feature_name}"
                all_results[result_key] = result
                
                print(f"    {status_emoji} {feature_name}: {reason_tag}")
        
        return all_results

    def classify_experimental_feature_status(self, evaluation_result: Dict[str, Any]) -> Tuple[str, str, str]:
        """
        Classify experimental feature status based on evaluation results
        
        Returns:
            Tuple of (status_emoji, reason_tag, recommendation)
        """
        if evaluation_result.get('status') != 'success':
            if evaluation_result.get('status') == 'error':
                return "🔴", "error", "Fix implementation errors"
            elif evaluation_result.get('status') == 'empty_output':
                return "🔴", "empty_output", "Check feature computation logic"
            else:
                return "⚪", "insufficient_data", "Need more data for evaluation"
        
        # Performance-based classification
        best_delta_sharpe = evaluation_result.get('best_delta_sharpe', 0)
        nan_rate = evaluation_result.get('nan_rate', 1.0)
        computation_time = evaluation_result.get('computation_time_ms', 0)
        avg_correlation = evaluation_result.get('avg_correlation', 0)
        
        # Check for various quality issues
        if nan_rate > 0.3:
            return "🔴", "high_nan_rate", f"Too many NaN values ({nan_rate:.1%})"
        
        if computation_time > 1000:  # > 1 second
            return "🟡", "slow_computation", f"Computation time too high ({computation_time:.0f}ms)"
        
        if avg_correlation > 0.8:
            return "🔴", "high_corr", f"High correlation with returns ({avg_correlation:.3f})"
        
        # Performance-based classification
        if best_delta_sharpe > 0.15:
            return "🟢", "strong_performance", f"Strong delta Sharpe: {best_delta_sharpe:.4f}"
        elif best_delta_sharpe > 0.05:
            return "🟡", "moderate_performance", f"Moderate delta Sharpe: {best_delta_sharpe:.4f}"  
        elif best_delta_sharpe < -0.05:
            return "🔴", "negative_performance", f"Negative delta Sharpe: {best_delta_sharpe:.4f}"
        else:
            return "🟡", "weak_performance", f"Weak delta Sharpe: {best_delta_sharpe:.4f}"

    def classify_harmful_feature_status(self, evaluation_result: Union[EvaluationResult, Dict[str, Any]]) -> Tuple[str, str, str]:
        """
        Classify harmful feature status based on evaluation results
        
        Returns:
            Tuple of (status_emoji, reason_tag, recommendation)
        """
        if not evaluation_result:
            return "⚪", "insufficient_data", "Insufficient data for evaluation"
        
        composite = evaluation_result.get('composite_result', {})
        best_individual = evaluation_result.get('best_individual_features', [])
        
        # Check composite performance first
        if composite:
            delta_sharpe = composite.get('delta_sharpe', 0)
            trade_periods = composite.get('trade_periods', 0)
            
            if delta_sharpe > 0.15 and trade_periods > 50:
                return "🟢", "strong_composite", "Remove from harmful list - Strong composite performance"
            elif delta_sharpe > 0.05 and trade_periods > 30:
                return "🟡", "moderate_composite", "Conditional use - Moderate composite improvement"
        
        # Check individual feature performance
        if best_individual:
            best_score = abs(best_individual[0][1]) if best_individual[0][1] is not None else 0
            
            if best_score > 0.2:
                return "🟡", "strong_individual", "Conditional use - Strong individual features"
            elif best_score > 0.1:
                return "🟡", "moderate_individual", "Monitor closely - Some promising features"
        
        return "🔴", "no_improvement", "Keep in harmful list - No significant improvement"

    def evaluate_harmful_feature(self, feature_name: str, ohlc_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate a single harmful feature with extended analysis
        
        Args:
            feature_name: Name of the feature to evaluate
            ohlc_data: OHLC price data
            
        Returns:
            Evaluation results dictionary
        """
        if feature_name not in self.harmful_evaluators:
            print(f"No evaluator found for {feature_name}")
            return {}
        
        evaluator = self.harmful_evaluators[feature_name]
        
        try:
            # Calculate extended features
            if feature_name == 'KalmanFilter':
                extended_features = evaluator(ohlc_data, 
                                            process_variance=0.001,
                                            measurement_variance=0.1)
            else:
                extended_features = evaluator(ohlc_data)
            
            print(f"Calculated {len(extended_features.columns)} features for {feature_name}")
            
            # Calculate returns for evaluation
            returns = ohlc_data['close'].pct_change().dropna()
            
            # Align features with returns
            aligned_data = pd.concat([extended_features, returns.rename('returns')], axis=1).dropna()
            
            if len(aligned_data) < 50:
                print(f"Insufficient aligned data for {feature_name}")
                return {}
            
            # Feature-by-feature evaluation
            feature_results = {}
            baseline_metrics = calculate_all_metrics(np.array(aligned_data['returns']))
            
            for feature_col in extended_features.columns:
                if feature_col not in aligned_data.columns:
                    continue
                
                feature_data = aligned_data[feature_col]
                
                # Skip if feature has too many NaN or constant values
                if feature_data.isna().sum() > len(feature_data) * 0.5:
                    continue
                if feature_data.nunique() <= 2:
                    continue
                
                # Correlation analysis
                correlation = feature_data.corr(aligned_data['returns'])
                
                # Signal-based evaluation for binary/categorical features
                if feature_data.nunique() <= 5:
                    # Binary signal evaluation
                    signal_periods = aligned_data[feature_data > feature_data.median()]
                    if len(signal_periods) > 10:
                        signal_returns = signal_periods['returns']
                        signal_metrics = calculate_all_metrics(np.array(signal_returns))
                        
                        delta_sharpe = signal_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio']
                        
                        feature_results[feature_col] = {
                            'type': 'signal',
                            'correlation': correlation,
                            'signal_sharpe': signal_metrics['sharpe_ratio'],
                            'baseline_sharpe': baseline_metrics['sharpe_ratio'],
                            'delta_sharpe': delta_sharpe,
                            'signal_periods': len(signal_returns),
                            'win_rate': signal_metrics['win_rate'],
                            'max_drawdown': signal_metrics['max_drawdown']
                        }
                
                # Quantile-based evaluation for continuous features
                else:
                    # Top/bottom quantile comparison
                    q75 = feature_data.quantile(0.75)
                    q25 = feature_data.quantile(0.25)
                    
                    top_quantile = aligned_data[feature_data >= q75]['returns']
                    bottom_quantile = aligned_data[feature_data <= q25]['returns']
                    
                    if len(top_quantile) > 10 and len(bottom_quantile) > 10:
                        top_metrics = calculate_all_metrics(np.array(top_quantile))
                        bottom_metrics = calculate_all_metrics(np.array(bottom_quantile))
                        
                        feature_results[feature_col] = {
                            'type': 'quantile',
                            'correlation': correlation,
                            'top_sharpe': top_metrics['sharpe_ratio'],
                            'bottom_sharpe': bottom_metrics['sharpe_ratio'],
                            'quantile_spread': top_metrics['sharpe_ratio'] - bottom_metrics['sharpe_ratio'],
                            'top_periods': len(top_quantile),
                            'bottom_periods': len(bottom_quantile)
                        }
            
            # Composite evaluation (best performing features combination)
            best_features = []
            for feature_col, result in feature_results.items():
                if result['type'] == 'signal' and result['delta_sharpe'] > 0.05:
                    best_features.append(feature_col)
                elif result['type'] == 'quantile' and abs(result['quantile_spread']) > 0.1:
                    best_features.append(feature_col)
            
            composite_result = {}
            if len(best_features) >= 2:
                # Create composite signal
                composite_signals = []
                for feature_col in best_features[:3]:  # Top 3 features
                    if feature_col in aligned_data.columns:
                        if aligned_data[feature_col].nunique() <= 5:
                            # Binary feature
                            composite_signals.append(aligned_data[feature_col] > aligned_data[feature_col].median())
                        else:
                            # Continuous feature - use top quantile
                            q75 = aligned_data[feature_col].quantile(0.75)
                            composite_signals.append(aligned_data[feature_col] >= q75)
                
                if len(composite_signals) >= 2:
                    # Require at least 2 out of N signals
                    composite_signal = sum(composite_signals) >= 2
                    composite_periods = aligned_data[composite_signal]
                    
                    if len(composite_periods) > 20:
                        composite_returns = composite_periods['returns']
                        composite_metrics = calculate_all_metrics(np.array(composite_returns))
                        
                        composite_result = {
                            'features_used': best_features,
                            'composite_sharpe': composite_metrics['sharpe_ratio'],
                            'baseline_sharpe': baseline_metrics['sharpe_ratio'],
                            'delta_sharpe': composite_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio'],
                            'trade_periods': len(composite_returns),
                            'win_rate': composite_metrics['win_rate'],
                            'max_drawdown': composite_metrics['max_drawdown'],
                            'calmar_ratio': composite_metrics['calmar_ratio']
                        }
            
            return {
                'feature_name': feature_name,
                'total_features': len(extended_features.columns),
                'evaluation_periods': len(aligned_data),
                'baseline_metrics': baseline_metrics,
                'feature_results': feature_results,
                'composite_result': composite_result,
                'best_individual_features': sorted(
                    [(k, v.get('delta_sharpe', v.get('quantile_spread', 0))) 
                     for k, v in feature_results.items()],
                    key=lambda x: abs(x[1]), reverse=True
                )[:5]
            }
            
        except Exception as e:
            print(f"Error evaluating {feature_name}: {e}")
            return {}
    
    def classify_feature_status(self, evaluation_result: Dict[str, Any]) -> Tuple[str, str]:
        """
        Classify feature status based on evaluation results
        
        Returns:
            Tuple of (status_emoji, recommendation)
        """
        if not evaluation_result:
            return "⚪", "Insufficient data for evaluation"
        
        composite = evaluation_result.get('composite_result', {})
        best_individual = evaluation_result.get('best_individual_features', [])
        
        # Check composite performance first
        if composite:
            delta_sharpe = composite.get('delta_sharpe', 0)
            trade_periods = composite.get('trade_periods', 0)
            
            if delta_sharpe > 0.15 and trade_periods > 50:
                return "🟢", "Remove from harmful list - Strong composite performance"
            elif delta_sharpe > 0.05 and trade_periods > 30:
                return "🟡", "Conditional use - Moderate composite improvement"
        
        # Check individual feature performance
        if best_individual:
            best_score = abs(best_individual[0][1]) if best_individual[0][1] is not None else 0
            
            if best_score > 0.2:
                return "🟡", "Conditional use - Strong individual features"
            elif best_score > 0.1:
                return "🟡", "Monitor closely - Some promising features"
        
        return "🔴", "Keep in harmful list - No significant improvement"
    
    def run_comprehensive_evaluation(self, symbol: str = "BTC/JPY") -> Dict[str, Any]:
        """
        Run comprehensive evaluation of harmful and experimental features
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Complete evaluation results
        """
        print(f"Starting comprehensive feature evaluation for {symbol}...")
        
        # Prepare OHLC data
        ohlc_data = self.prepare_ohlc_data(symbol)
        if ohlc_data is None:
            return {}
        
        print(f"Prepared {len(ohlc_data)} OHLC records")
        
        # Evaluate harmful features  
        print("\n=== Evaluating Harmful Features ===")
        harmful_results = {}
        
        for feature_name in self.harmful_evaluators.keys():
            if feature_name in self.harmful_info:
                print(f"\nEvaluating harmful feature: {feature_name}...")
                
                result = self.evaluate_harmful_feature(feature_name, ohlc_data)
                if result:
                    status_emoji, reason_tag, recommendation = self.classify_harmful_feature_status(result)
                    result['status_emoji'] = status_emoji
                    result['reason_tag'] = reason_tag
                    result['recommendation'] = recommendation
                    
                    harmful_results[feature_name] = result
                    
                    print(f"  {feature_name}: {status_emoji} - {reason_tag}")
        
        # Evaluate experimental/wave features
        print("\n=== Evaluating Experimental/Wave Features ===")
        experimental_results = self.evaluate_experimental_features(ohlc_data)
        
        # Update results storage
        self.results['harmful'] = harmful_results
        self.results['experimental'] = experimental_results
        
        return {
            'symbol': symbol,
            'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_periods': len(ohlc_data),
            'harmful_features_evaluated': list(harmful_results.keys()),
            'experimental_features_evaluated': list(experimental_results.keys()),
            'results': {
                'harmful': harmful_results,
                'experimental': experimental_results
            }
        }
    
    def generate_comprehensive_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate comprehensive markdown report for all feature evaluations"""
        if not evaluation_results:
            return "# Comprehensive Feature Evaluation Report\n\nNo evaluation results available."
        
        report = []
        report.append("# Comprehensive Feature Evaluation Report")
        report.append(f"Generated: {evaluation_results['evaluation_date']}")
        report.append(f"Symbol: {evaluation_results['symbol']}")
        report.append(f"Data Periods: {evaluation_results['data_periods']}")
        report.append("")
        
        # Executive Summary
        harmful_results = evaluation_results['results']['harmful']
        experimental_results = evaluation_results['results']['experimental']
        
        report.append("## Executive Summary")
        report.append(f"- **Harmful Features Evaluated**: {len(harmful_results)}")
        report.append(f"- **Experimental/Wave Features Evaluated**: {len(experimental_results)}")
        
        # Count by status
        all_results = {**harmful_results, **experimental_results}
        green_count = sum(1 for r in all_results.values() if r.get('status_emoji') == '🟢')
        yellow_count = sum(1 for r in all_results.values() if r.get('status_emoji') == '🟡') 
        red_count = sum(1 for r in all_results.values() if r.get('status_emoji') == '🔴')
        
        report.append(f"- **🟢 Ready for Production**: {green_count}")
        report.append(f"- **🟡 Conditional/Monitor**: {yellow_count}")
        report.append(f"- **🔴 Avoid/Fix Required**: {red_count}")
        report.append("")
        
        # Harmful Features Section
        if harmful_results:
            report.append("## Harmful Feature Re-evaluation")
            report.append("| Feature | Status | Delta Sharpe | Reason | Recommendation |")
            report.append("|---------|--------|--------------|--------|-----------------|")
            
            for feature_name, result in harmful_results.items():
                composite = result.get('composite_result', {})
                delta_sharpe = composite.get('delta_sharpe', 0.0)
                status_emoji = result.get('status_emoji', '⚪')
                reason_tag = result.get('reason_tag', 'unknown')
                recommendation = result.get('recommendation', 'No recommendation')
                
                report.append(f"| {feature_name} | {status_emoji} | {delta_sharpe:+.4f} | {reason_tag} | {recommendation} |")
            
            report.append("")
        
        # Experimental/Wave Features Section
        if experimental_results:
            report.append("## Experimental/Wave Feature Evaluation")
            report.append("| Feature | Module | Status | NaN Rate | Time (ms) | Delta Sharpe | Reason | Recommendation |")
            report.append("|---------|--------|--------|----------|-----------|--------------|--------|-----------------|")
            
            for feature_name, result in experimental_results.items():
                module = result.get('module_path', 'unknown').split('.')[-1]
                status_emoji = result.get('status_emoji', '⚪')
                nan_rate = result.get('nan_rate', 0.0)
                time_ms = result.get('computation_time_ms', 0)
                delta_sharpe = result.get('best_delta_sharpe', 0.0)
                reason_tag = result.get('reason_tag', 'unknown')
                recommendation = result.get('recommendation', 'No recommendation')
                
                report.append(f"| {feature_name} | {module} | {status_emoji} | {nan_rate:.1%} | {time_ms:.0f} | {delta_sharpe:+.4f} | {reason_tag} | {recommendation} |")
            
            report.append("")
        
        # Detailed Analysis Sections
        if harmful_results:
            report.append("## Detailed Harmful Feature Analysis")
            for feature_name, result in harmful_results.items():
                report.append(f"### {feature_name}")
                report.append(f"**Status**: {result.get('status_emoji', '⚪')} {result.get('recommendation', '')}")
                report.append(f"**Extended Features**: {result.get('total_features', 'N/A')}")
                report.append(f"**Evaluation Periods**: {result.get('evaluation_periods', 'N/A')}")
                report.append("")
                
                # Composite performance
                composite = result.get('composite_result', {})
                if composite:
                    report.append("#### Composite Strategy Performance")
                    report.append(f"- Features Used: {', '.join(composite.get('features_used', []))}")
                    report.append(f"- Composite Sharpe: {composite.get('composite_sharpe', 0):.4f}")
                    report.append(f"- Baseline Sharpe: {composite.get('baseline_sharpe', 0):.4f}")
                    report.append(f"- **Improvement**: {composite.get('delta_sharpe', 0):+.4f}")
                    report.append(f"- Trade Periods: {composite.get('trade_periods', 0)}")
                    report.append(f"- Win Rate: {composite.get('win_rate', 0):.1%}")
                    report.append("")
                
                # Best individual features
                best_features = result.get('best_individual_features', [])
                if best_features:
                    report.append("#### Top Individual Features")
                    for feature_col, score in best_features[:3]:
                        report.append(f"- {feature_col}: {score:+.4f}")
                    report.append("")
        
        # Recommendations by category
        report.append("## Recommendations by Category")
        
        green_harmful = [name for name, result in harmful_results.items() if result.get('status_emoji') == '🟢']
        green_experimental = [name for name, result in experimental_results.items() if result.get('status_emoji') == '🟢']
        
        if green_harmful:
            report.append("### 🟢 Remove from Harmful List")
            for feature in green_harmful:
                report.append(f"- **{feature}**: {harmful_results[feature].get('recommendation', '')}")
            report.append("")
        
        if green_experimental:
            report.append("### 🟢 Promote to Production")
            for feature in green_experimental:
                report.append(f"- **{feature}**: {experimental_results[feature].get('recommendation', '')}")
            report.append("")
        
        yellow_features = [name for name, result in all_results.items() if result.get('status_emoji') == '🟡']
        if yellow_features:
            report.append("### 🟡 Monitor/Conditional Use")
            for feature in yellow_features:
                result = all_results[feature]
                report.append(f"- **{feature}**: {result.get('recommendation', '')}")
            report.append("")
        
        red_features = [name for name, result in all_results.items() if result.get('status_emoji') == '🔴']
        if red_features:
            report.append("### 🔴 Requires Attention")
            for feature in red_features:
                result = all_results[feature]
                report.append(f"- **{feature}**: {result.get('recommendation', '')}")
            report.append("")
        
        report.append("---")
        report.append("*Report generated by re_evaluate_features.py (Comprehensive)*")
        
        # Add unverified features section
        report.append("")
        report.append("## Unverified Features Status")
        report.extend(self._generate_unverified_features_section())
        
        return "\n".join(report)
    
    def _generate_unverified_features_section(self) -> List[str]:
        """Generate unverified features section for the report"""
        section = []
        
        try:
            # Load coverage.json
            coverage_path = Path(__file__).parent.parent.parent / "coverage.json"
            if coverage_path.exists():
                with open(coverage_path, 'r', encoding='utf-8') as f:
                    coverage = json.load(f)
            else:
                section.append("Coverage data not found.")
                return section
            
            # Summary
            metadata = coverage.get('metadata', {})
            section.append(f"**Total Verified**: {metadata.get('total_verified', 0)} | "
                          f"**Pending**: {metadata.get('total_pending', 0)} | "
                          f"**Failed**: {metadata.get('total_failed', 0)} | "
                          f"**Unverified**: {metadata.get('total_unverified', 0)}")
            section.append("")
            
            # Pending features table
            pending = coverage.get('pending_features', {})
            if pending:
                section.append("### ⏳ Pending Features (Insufficient Data)")
                section.append("| Feature | Reason | Required Samples | Current Samples | Last Attempt |")
                section.append("|---------|--------|------------------|-----------------|--------------|")
                
                for feature_name, data in pending.items():
                    reason = data.get('reason', 'unknown')
                    required = data.get('required_samples', 'N/A')
                    current = data.get('current_samples', 'N/A')
                    last_attempt = data.get('last_attempt', 'N/A')
                    
                    section.append(f"| {feature_name} | {reason} | {required} | {current} | {last_attempt} |")
                
                section.append("")
            
            # Failed features table
            failed = coverage.get('failed_features', {})
            if failed:
                section.append("### ❌ Failed Features")
                section.append("| Feature | Reason | Sharpe | Win Rate | Max Drawdown | Last Attempt |")
                section.append("|---------|--------|--------|----------|--------------|--------------|")
                
                for feature_name, data in failed.items():
                    reason = data.get('reason', 'unknown')
                    perf = data.get('performance', {})
                    sharpe = perf.get('sharpe_ratio', 'N/A')
                    win_rate = perf.get('win_rate', 'N/A')
                    max_dd = perf.get('max_drawdown', 'N/A')
                    last_attempt = data.get('last_attempt', 'N/A')
                    
                    section.append(f"| {feature_name} | {reason} | {sharpe} | {win_rate} | {max_dd} | {last_attempt} |")
                
                section.append("")
            
            # Unverified features list
            unverified = coverage.get('unverified_features', [])
            if unverified:
                section.append("### ❓ Unverified Features")
                section.append("New features awaiting initial evaluation:")
                for feature in unverified:
                    section.append(f"- {feature}")
                section.append("")
            
            # Create JSON attachment
            json_attachment = json.dumps(coverage, indent=2, ensure_ascii=False)
            section.append("### 📎 Coverage Data (JSON)")
            section.append("```json")
            section.append(json_attachment)
            section.append("```")
            
        except Exception as e:
            section.append(f"Error loading coverage data: {e}")
        
        return section
    
    def generate_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate markdown report for harmful feature re-evaluation"""
        if not evaluation_results:
            return "# Harmful Feature Re-evaluation Report\n\nNo evaluation results available."
        
        report = []
        report.append("# Harmful Feature Re-evaluation Report")
        report.append(f"Generated: {evaluation_results['evaluation_date']}")
        report.append(f"Symbol: {evaluation_results['symbol']}")
        report.append(f"Data Periods: {evaluation_results['data_periods']}")
        report.append("")
        
        # Summary table
        report.append("## Executive Summary")
        report.append("| Feature | Status | Delta Sharpe | Recommendation |")
        report.append("|---------|--------|--------------|-----------------|")
        
        for feature_name, result in evaluation_results['results'].items():
            composite = result.get('composite_result', {})
            delta_sharpe = composite.get('delta_sharpe', 0.0)
            
            report.append(f"| {feature_name} | {result['status_emoji']} | {delta_sharpe:+.4f} | {result['recommendation']} |")
        
        report.append("")
        
        # Detailed results
        report.append("## Detailed Analysis")
        
        for feature_name, result in evaluation_results['results'].items():
            report.append(f"### {feature_name}")
            report.append(f"**Status**: {result['status_emoji']} {result['recommendation']}")
            report.append(f"**Extended Features**: {result['total_features']}")
            report.append(f"**Evaluation Periods**: {result['evaluation_periods']}")
            report.append("")
            
            # Composite performance
            composite = result.get('composite_result', {})
            if composite:
                report.append("#### Composite Strategy Performance")
                report.append(f"- Features Used: {', '.join(composite['features_used'])}")
                report.append(f"- Composite Sharpe: {composite['composite_sharpe']:.4f}")
                report.append(f"- Baseline Sharpe: {composite['baseline_sharpe']:.4f}")
                report.append(f"- **Improvement**: {composite['delta_sharpe']:+.4f}")
                report.append(f"- Trade Periods: {composite['trade_periods']}")
                report.append(f"- Win Rate: {composite['win_rate']:.1%}")
                report.append(f"- Max Drawdown: {composite['max_drawdown']:.1%}")
                report.append("")
            
            # Best individual features
            best_features = result.get('best_individual_features', [])
            if best_features:
                report.append("#### Top Individual Features")
                for feature_col, score in best_features:
                    report.append(f"- {feature_col}: {score:+.4f}")
                report.append("")
        
        return "\n".join(report)


def main():
    """Main execution function"""
    print("Starting comprehensive feature evaluation...")
    
    # Initialize evaluator
    evaluator = ComprehensiveFeatureReEvaluator()
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    if not results:
        print("Evaluation failed. No results to report.")
        return
    
    # Generate report
    report = evaluator.generate_comprehensive_report(results)
    
    # Save report
    report_path = "reports/comprehensive_feature_evaluation.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nComprehensive evaluation report saved to: {report_path}")
    
    # Display summary
    print(f"\nEvaluation Summary:")
    
    # Harmful features summary
    harmful_results = results['results']['harmful']
    print(f"\nHarmful Features ({len(harmful_results)}):")
    for feature_name, result in harmful_results.items():
        print(f"  {feature_name}: {result.get('status_emoji', '⚪')} ({result.get('reason_tag', 'unknown')})")
    
    # Experimental features summary
    experimental_results = results['results']['experimental']
    print(f"\nExperimental/Wave Features ({len(experimental_results)}):")
    for feature_name, result in experimental_results.items():
        print(f"  {feature_name}: {result.get('status_emoji', '⚪')} ({result.get('reason_tag', 'unknown')})")
    
    # Recommendations
    green_features = []
    yellow_features = []
    red_features = []
    
    all_results = {**harmful_results, **experimental_results}
    for feature_name, result in all_results.items():
        status = result.get('status_emoji', '⚪')
        if status == '🟢':
            green_features.append(feature_name)
        elif status == '🟡':
            yellow_features.append(feature_name)
        elif status == '🔴':
            red_features.append(feature_name)
    
    print(f"\nRecommendations:")
    if green_features:
        print(f"  🟢 Consider for production ({len(green_features)}): {', '.join(green_features)}")
    if yellow_features:
        print(f"  🟡 Monitor/conditional use ({len(yellow_features)}): {', '.join(yellow_features)}")
    if red_features:
        print(f"  🔴 Avoid or fix ({len(red_features)}): {', '.join(red_features)}")


if __name__ == "__main__":
    main()
