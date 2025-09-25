"""
Feature auto-generation utilities.

This module provides automatic generation of technical indicators
with standardized na        if fast_periods is None:
            fast_periods = self.params.get('kama', {}).get('fast_periods', [5, 10])
        if slow_periods is None:
            slow_periods = self.params.get('kama', {}).get('slow_periods', [20, 30])
        if efficiency_periods is None:
            efficiency_periods = self.params.get('kama', {}).get('efficiency_periods', [10, 20, 30, 50])

        # Ensure we have valid lists
        assert isinstance(fast_periods, list) and all(isinstance(x, int) for x in fast_periods)
        assert isinstance(slow_periods, list) and all(isinstance(x, int) for x in slow_periods)
        assert isinstance(efficiency_periods, list) and all(isinstance(x, int) for x in efficiency_periods)nd promotion rules.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple, Iterator
from pathlib import Path
import json
import yaml
from datetime import datetime
from abc import ABC, abstractmethod
from ztb.evaluation.logging import EvaluationLogger
from ztb.features.base import CommonPreprocessor


class ParameterCombinationGenerator:
    """Generic parameter combination generator for feature creation"""

    @staticmethod
    def generate_combinations(*param_lists: List[Any], max_combinations: Optional[int] = None) -> Iterator[Tuple[Any, ...]]:
        """Generate all combinations from parameter lists"""
        if not param_lists:
            yield ()
            return

        from itertools import product
        combinations = list(product(*param_lists))

        # Check max combinations limit
        if max_combinations is not None and len(combinations) > max_combinations:
            print(f"Warning: Generated {len(combinations)} combinations, limiting to {max_combinations}")
            combinations = combinations[:max_combinations]

        yield from combinations

    @staticmethod
    def validate_combination(combination: Tuple[Any, ...], 
                           feature_type: Optional[str] = None,
                           validators: Optional[List[Callable]] = None) -> bool:
        """Validate a parameter combination with built-in rules"""
        # Built-in validations by feature type
        if feature_type:
            if feature_type in ['kama', 'ema_cross']:
                if len(combination) >= 2:
                    fast, slow = combination[0], combination[1]
                    if not isinstance(fast, (int, float)) or not isinstance(slow, (int, float)):
                        return False
                    if fast >= slow:  # fast must be < slow
                        return False
        
        # Custom validators
        if validators:
            for validator in validators:
                if not validator(*combination):
                    return False
        
        return True


class FeatureCalculator:
    """Delegated calculation logic for different feature types"""

    @staticmethod
    def calculate_ema_cross(ohlc_data: pd.DataFrame, fast_period: int, slow_period: int) -> pd.DataFrame:
        """Calculate EMA cross features"""
        fast_ema = ohlc_data['close'].ewm(span=fast_period).mean()
        slow_ema = ohlc_data['close'].ewm(span=slow_period).mean()

        ema_cross = (fast_ema - slow_ema) / slow_ema
        ema_above = (fast_ema > slow_ema).astype(int)

        return pd.DataFrame({
            f'ema_{fast_period}_cross_{slow_period}': ema_cross,
            f'ema_{fast_period}_above_{slow_period}': ema_above
        })

    @staticmethod
    def calculate_kama(ohlc_data: pd.DataFrame, fast: int, slow: int, er_period: int) -> pd.DataFrame:
        """Calculate KAMA features"""
        price = np.asarray(ohlc_data['close'].values)
        kama = FeatureCalculator._calculate_simple_kama(price, slow, er_period)

        return pd.DataFrame({
            f'kama_{fast}_{slow}_{er_period}': kama
        }, index=ohlc_data.index)

    @staticmethod
    def _calculate_simple_kama(price: np.ndarray, slow: int, er_period: int) -> np.ndarray:
        """Simplified KAMA calculation"""
        n = len(price)
        kama = np.full(n, np.nan)

        if n < slow + er_period:
            return kama

        # Simple exponential smoothing as approximation
        alpha = 2.0 / (slow + 1)
        kama[slow] = price[slow]

        for i in range(slow + 1, n):
            kama[i] = kama[i-1] + alpha * (price[i] - kama[i-1])

        return kama


class FeatureGeneratorTemplate(ABC):
    """Template for feature generation with common workflow"""

    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params

    def generate_features(self, ohlc_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Template method for feature generation"""
        # 1. Parameter validation
        self._validate_parameters()

        # 2. Generate parameter combinations
        combinations = self._generate_parameter_combinations()

        # 3. Calculate features
        features = {}
        for combo in combinations:
            try:
                feature_name = self._create_feature_name(combo)
                feature_data = self._calculate_feature(ohlc_data, combo)
                features[feature_name] = feature_data
            except Exception as e:
                print(f"Error generating {self.name} feature with params {combo}: {e}")

        # 4. Apply naming convention
        features = self._apply_naming_convention(features)

        # 5. Register to coverage.json
        self._register_to_coverage(features)

        return features

    def _validate_parameters(self) -> None:
        """Validate input parameters"""
        pass  # Override in subclasses

    @abstractmethod
    def _generate_parameter_combinations(self) -> List[Tuple[Any, ...]]:
        """Generate parameter combinations"""
        pass

    @abstractmethod
    def _create_feature_name(self, combination: Tuple[Any, ...]) -> str:
        """Create feature name from parameter combination"""
        pass

    @abstractmethod
    def _calculate_feature(self, ohlc_data: pd.DataFrame, combination: Tuple[Any, ...]) -> pd.DataFrame:
        """Calculate feature data"""
        pass

    def _apply_naming_convention(self, features: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply auto_ naming convention with category prefixes"""
        renamed = {}
        for name, data in features.items():
            if not name.startswith("auto_"):
                # Add category prefix based on feature type
                if "ema" in name.lower() or "cross" in name.lower():
                    category = "ma"  # moving average
                elif "kama" in name.lower():
                    category = "ma"  # moving average
                elif "rsi" in name.lower() or "stoch" in name.lower():
                    category = "osc"  # oscillator
                elif "bb" in name.lower() or "band" in name.lower():
                    category = "vol"  # volatility
                elif "ichimoku" in name.lower() or "cloud" in name.lower():
                    category = "ch"  # channel
                else:
                    category = "misc"  # miscellaneous

                name = f"auto_{category}_{name}"
            renamed[name] = data
        return renamed

    def _register_to_coverage(self, features: Dict[str, pd.DataFrame]) -> None:
        """Register generated features to coverage.json"""
        # Implementation would update coverage.json
        pass


class AutoFeatureGenerator:
    """Automatic feature generation with promotion rules"""

    def __init__(self, logger: Optional[EvaluationLogger] = None, config_path: Optional[Path] = None):
        self.logger = logger or EvaluationLogger()

        # Load parameter configurations
        self.config_path = config_path or Path("config/feature_params.yaml")
        self.params = self._load_params()

        self.promotion_criteria = {
            'min_sharpe_ratio': 0.3,
            'min_win_rate': 0.55,
            'max_drawdown_limit': -0.15,
            'min_sample_size': 100
        }

    def _load_params(self) -> Dict[str, Any]:
        """Load parameter configurations from YAML file"""
        if not self.config_path.exists():
            # Fallback to default parameters
            return {
                'max_combinations': 100,
                'ema': {'periods': [5, 8, 12, 20, 25, 30, 40, 50]},
                'kama': {
                    'fast_periods': [5, 10],
                    'slow_periods': [20, 30],
                    'efficiency_periods': [10, 20, 30, 50]
                },
                'ichimoku': {'periods': [9, 26, 52]}
            }

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                params = yaml.safe_load(f)
                # Ensure max_combinations exists
                if 'max_combinations' not in params:
                    params['max_combinations'] = 100
                return params
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
            # Fallback to default parameters
            return {
                'max_combinations': 100,
                'ema': {'periods': [5, 8, 12, 20, 25, 30, 40, 50]},
                'kama': {
                    'fast_periods': [5, 10],
                    'slow_periods': [20, 30],
                    'efficiency_periods': [10, 20, 30, 50]
                },
                'ichimoku': {'periods': [9, 26, 52]}
            }

    def generate_ema_cross_features(self, ohlc_data: pd.DataFrame,
                                  fast_periods: Optional[List[int]] = None,
                                  slow_periods: Optional[List[int]] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate EMA cross features with multiple parameter combinations

        Args:
            ohlc_data: OHLC data
            fast_periods: Fast EMA periods (uses config if None)
            slow_periods: Slow EMA periods (uses config if None)

        Returns:
            Dictionary of generated features
        """
        # Preprocess data to ensure common calculations are available
        ohlc_data = CommonPreprocessor.preprocess(ohlc_data.copy())

        if fast_periods is None:
            fast_periods = self.params.get('ema', {}).get('periods', [5, 8, 12, 20])
        if slow_periods is None:
            slow_periods = self.params.get('ema', {}).get('periods', [25, 30, 40, 50])

        # Ensure we have valid lists
        assert isinstance(fast_periods, list) and all(isinstance(x, int) for x in fast_periods)
        assert isinstance(slow_periods, list) and all(isinstance(x, int) for x in slow_periods)

        features = {}

        # Generate combinations with validation
        max_combinations = self.params.get('max_combinations', 100)
        all_combinations = list(ParameterCombinationGenerator.generate_combinations(
            fast_periods, slow_periods, max_combinations=max_combinations
        ))
        
        # Filter valid combinations
        combinations = []
        for combo in all_combinations:
            if ParameterCombinationGenerator.validate_combination(combo, feature_type='ema_cross'):
                combinations.append(combo)

        for fast, slow in combinations:

            name = self._validate_feature_name(f"ema_cross_{fast}_{slow}")
            try:
                # Use pre-calculated EMAs if available, otherwise calculate
                fast_col = f'ema_{fast}'
                slow_col = f'ema_{slow}'

                if fast_col in ohlc_data.columns and slow_col in ohlc_data.columns:
                    fast_ema = ohlc_data[fast_col]
                    slow_ema = ohlc_data[slow_col]
                else:
                    # Fallback to calculation
                    fast_ema = ohlc_data['close'].ewm(span=fast).mean()
                    slow_ema = ohlc_data['close'].ewm(span=slow).mean()

                # Create features
                ema_cross = (fast_ema - slow_ema) / slow_ema
                ema_above = (fast_ema > slow_ema).astype(int)

                features[name] = pd.DataFrame({
                    f'ema_{fast}_cross_{slow}': ema_cross,
                    f'ema_{fast}_above_{slow}': ema_above
                })

            except Exception as e:
                print(f"Error generating {name}: {e}")

        # Apply quality gates
        features = self._apply_quality_gates(features, ohlc_data)
        
        return features

    def _apply_quality_gates(self, features: Dict[str, pd.DataFrame], ohlc_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Apply quality gates to filter out poor quality features"""
        quality_gates = self.params.get('quality_gates', {})
        max_nan_rate = quality_gates.get('max_nan_rate_threshold', 0.8)
        min_abs_correlation = quality_gates.get('min_abs_correlation_threshold', 0.05)
        
        filtered_features = {}
        
        for name, feature_df in features.items():
            # Check NaN rate
            nan_rate = feature_df.isna().mean().mean()  # Average NaN rate across all columns
            if nan_rate > max_nan_rate:
                print(f"Quality gate: {name} rejected (NaN rate: {nan_rate:.3f} > {max_nan_rate})")
                continue
            
            # Check correlation with base signal (close or return)
            correlations = []
            for col in feature_df.columns:
                if col in feature_df.columns:
                    # Try correlation with close, fallback to return
                    if 'close' in ohlc_data.columns:
                        corr = feature_df[col].corr(ohlc_data['close'])
                    elif 'return' in ohlc_data.columns:
                        corr = feature_df[col].corr(ohlc_data['return'])
                    else:
                        corr = 0.0
                    
                    if not pd.isna(corr):
                        correlations.append(abs(corr))
            
            avg_correlation = sum(correlations) / len(correlations) if correlations else 0.0
            
            if avg_correlation < min_abs_correlation:
                print(f"Quality gate: {name} rejected (correlation: {avg_correlation:.3f} < {min_abs_correlation})")
                continue
            
            filtered_features[name] = feature_df
        
        return filtered_features

    def generate_kama_features(self, ohlc_data: pd.DataFrame,
                             fast_periods: Optional[List[int]] = None,
                             slow_periods: Optional[List[int]] = None,
                             efficiency_periods: Optional[List[int]] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate KAMA features with multiple parameter combinations

        Args:
            ohlc_data: OHLC data
            fast_periods: Fast KAMA periods
            slow_periods: Slow KAMA periods
            efficiency_periods: Efficiency ratio periods

        Returns:
            Dictionary of generated features
        """
        if fast_periods is None:
            fast_periods = [2, 3, 5]
        if slow_periods is None:
            slow_periods = [20, 30, 40]
        if efficiency_periods is None:
            efficiency_periods = [5, 10, 15]

        features = {}

        for fast in fast_periods:
            for slow in slow_periods:
                for er in efficiency_periods:
                    name = self._validate_feature_name(f"kama_{fast}_{slow}_{er}")
                    try:
                        # Calculate KAMA (simplified version)
                        price = np.asarray(ohlc_data['close'].values)
                        kama = self._calculate_simple_kama(price, slow, er)

                        features[name] = pd.DataFrame({
                            f'kama_{fast}_{slow}_{er}': kama
                        }, index=ohlc_data.index)

                    except Exception as e:
                        print(f"Error generating {name}: {e}")

        return features

    def _validate_feature_name(self, name: str) -> str:
        """Validate and ensure feature name follows auto_ naming convention with category"""
        if not name.startswith("auto_"):
            # Add category prefix based on feature type
            if "ema" in name.lower() or "cross" in name.lower():
                category = "ma"  # moving average
            elif "kama" in name.lower():
                category = "ma"  # moving average
            elif "rsi" in name.lower() or "stoch" in name.lower():
                category = "osc"  # oscillator
            elif "bb" in name.lower() or "band" in name.lower():
                category = "vol"  # volatility
            elif "ichimoku" in name.lower() or "cloud" in name.lower():
                category = "ch"  # channel
            else:
                category = "misc"  # miscellaneous

            validated_name = f"auto_{category}_{name}"
        else:
            validated_name = name
        return validated_name

    def _generate_unique_name(self, base_name: str, existing_names: set) -> str:
        """Generate unique feature name to avoid conflicts"""
        name = base_name
        counter = 1
        while name in existing_names:
            name = f"{base_name}_{counter}"
            counter += 1
        return name

    def _calculate_simple_kama(self, price: np.ndarray, slow: int, er_period: int) -> np.ndarray:
        """Simplified KAMA calculation for auto-generation"""
        n = len(price)
        kama = np.full(n, np.nan)

        if n < slow + er_period:
            return kama

        # Simple exponential smoothing as approximation
        alpha = 2.0 / (slow + 1)
        kama[slow] = price[slow]

        for i in range(slow + 1, n):
            kama[i] = kama[i-1] + alpha * (price[i] - kama[i-1])

        return kama

    def evaluate_and_promote_features(self, features: Dict[str, pd.DataFrame],
                                    ohlc_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate generated features and promote successful ones

        Args:
            features: Generated features dictionary
            ohlc_data: OHLC data for evaluation

        Returns:
            Evaluation results and promotion decisions
        """
        results: Dict[str, Any] = {}
        promoted_features = {}
        temporary_features = {}

        # Calculate returns for evaluation
        returns = ohlc_data['close'].pct_change().dropna()

        for feature_name, feature_df in features.items():
            try:
                # Align data
                aligned_data = pd.concat([feature_df, returns.rename('returns')], axis=1).dropna()

                if len(aligned_data) < self.promotion_criteria['min_sample_size']:
                    temporary_features[feature_name] = {
                        'status': 'insufficient_data',
                        'samples': len(aligned_data),
                        'required': self.promotion_criteria['min_sample_size']
                    }
                    continue

                # Evaluate each column in the feature
                feature_results = {}
                for col in feature_df.columns:
                    if col in aligned_data.columns:
                        # Simple correlation-based evaluation
                        correlation = aligned_data[col].corr(aligned_data['returns'])

                        # Signal evaluation for binary features
                        if aligned_data[col].nunique() <= 5:
                            signal_returns = aligned_data[aligned_data[col] > aligned_data[col].median()]['returns']
                            if len(signal_returns) > 10:
                                signal_metrics = self._calculate_basic_metrics(np.asarray(signal_returns.values))
                                feature_results[col] = {
                                    'correlation': correlation,
                                    'sharpe_ratio': signal_metrics['sharpe'],
                                    'win_rate': signal_metrics['win_rate'],
                                    'max_drawdown': signal_metrics['max_drawdown']
                                }
                        else:
                            feature_results[col] = {'correlation': correlation}

                # Check promotion criteria
                should_promote = self._check_promotion_criteria(feature_results)

                if should_promote:
                    promoted_features[feature_name] = {
                        'status': 'promoted',
                        'feature_data': feature_df,
                        'evaluation_results': feature_results,
                        'promoted_at': datetime.now().isoformat()
                    }
                else:
                    temporary_features[feature_name] = {
                        'status': 'temporary',
                        'evaluation_results': feature_results
                    }

                # Log evaluation
                eval_result = {
                    'feature_name': feature_name,
                    'status': 'promoted' if should_promote else 'temporary',
                    'computation_time_ms': 0.0,  # Placeholder
                    'nan_rate': feature_df.isna().mean().mean(),
                    'total_columns': len(feature_df.columns),
                    'aligned_periods': len(aligned_data)
                }
                self.logger.log_evaluation(eval_result)

            except Exception as e:
                temporary_features[feature_name] = {
                    'status': 'error',
                    'error': str(e)
                }

        results['promoted'] = promoted_features
        results['temporary'] = temporary_features
        results['summary'] = {
            'total_generated': len(features),
            'promoted_count': len(promoted_features),
            'temporary_count': len(temporary_features),
            'promotion_rate': len(promoted_features) / len(features) if features else 0
        }

        return results

    def _check_promotion_criteria(self, feature_results: Dict[str, Any]) -> bool:
        """Check if feature meets promotion criteria"""
        for col, results in feature_results.items():
            sharpe = results.get('sharpe_ratio', 0)
            win_rate = results.get('win_rate', 0)
            max_dd = results.get('max_drawdown', 0)

            if (sharpe >= self.promotion_criteria['min_sharpe_ratio'] and
                win_rate >= self.promotion_criteria['min_win_rate'] and
                max_dd >= self.promotion_criteria['max_drawdown_limit']):
                return True

        return False

    def _calculate_basic_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        returns = np.asarray(returns)  # Ensure returns is np.ndarray
        if len(returns) == 0:
            return {'sharpe': 0.0, 'win_rate': 0.0, 'max_drawdown': 0.0}

        # Sharpe ratio
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0.0

        # Win rate
        win_rate = np.mean(returns > 0)

        # Max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        return {
            'sharpe': float(sharpe),
            'win_rate': float(win_rate),
            'max_drawdown': float(max_drawdown)
        }

    def save_promoted_features(self, promoted_features: Dict[str, Any],
                             output_dir: str = "ztb/features/auto_generated") -> None:
        """
        Save promoted features to permanent location

        Args:
            promoted_features: Features to save
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for feature_name, data in promoted_features.items():
            try:
                # Save feature data as pickle
                feature_file = output_path / f"{feature_name}.pkl"
                data['feature_data'].to_pickle(feature_file)

                # Save metadata
                metadata_file = output_path / f"{feature_name}_metadata.json"
                metadata = {
                    'feature_name': feature_name,
                    'generated_at': data.get('promoted_at'),
                    'evaluation_results': data.get('evaluation_results'),
                    'promotion_criteria': self.promotion_criteria
                }

                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)

            except Exception as e:
                print(f"Error saving {feature_name}: {e}")

    def generate_comprehensive_feature_set(self, ohlc_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive set of features with evaluation and promotion

        Args:
            ohlc_data: OHLC data

        Returns:
            Complete generation results
        """
        print("Starting comprehensive feature auto-generation...")

        # Generate different types of features
        ema_features = self.generate_ema_cross_features(ohlc_data)
        kama_features = self.generate_kama_features(ohlc_data)

        all_features = {**ema_features, **kama_features}

        print(f"Generated {len(all_features)} feature combinations")

        # Evaluate and promote
        evaluation_results = self.evaluate_and_promote_features(all_features, ohlc_data)

        # Save promoted features
        if evaluation_results['promoted']:
            self.save_promoted_features(evaluation_results['promoted'])
            print(f"Promoted {len(evaluation_results['promoted'])} features to permanent catalog")

        print(f"Completed auto-generation: {evaluation_results['summary']}")

        return evaluation_results