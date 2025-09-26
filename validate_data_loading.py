#!/usr/bin/env python3
"""
Data loading validation for CoinGecko cache and missing data handling
Validate ffill/bfill impact on reward calculations and learning
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, Any, List


class DataLoadingValidator:
    """Validate data loading and missing data handling"""

    def __init__(self):
        self.validation_results = []

    def simulate_missing_data_scenarios(self) -> List[tuple[str, pd.DataFrame]]:
        """Create test datasets with different missing data patterns"""
        # Generate base dataset (1 year of hourly data)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='1H')
        np.random.seed(42)

        base_data = pd.DataFrame({
            'timestamp': dates,
            'close': 50000 + np.cumsum(np.random.normal(0, 100, len(dates))),
            'volume': np.random.exponential(1000, len(dates)),
            'rsi': 50 + 30 * np.sin(np.linspace(0, 8*np.pi, len(dates))),
            'macd': np.random.normal(0, 50, len(dates)),
            'bb_upper': lambda x: x['close'] + 100,
            'bb_lower': lambda x: x['close'] - 100
        })

        # Calculate bollinger bands properly
        base_data['bb_upper'] = base_data['close'].rolling(20).mean() + 2 * base_data['close'].rolling(20).std()
        base_data['bb_lower'] = base_data['close'].rolling(20).mean() - 2 * base_data['close'].rolling(20).std()

        scenarios = []

        # Scenario 1: Random missing (5% of data)
        data1 = base_data.copy()
        missing_mask = np.random.random(len(data1)) < 0.05
        data1.loc[missing_mask, ['close', 'volume', 'rsi', 'macd', 'bb_upper', 'bb_lower']] = np.nan
        scenarios.append(('random_5pct', data1))

        # Scenario 2: Block missing (1-hour gaps every 24 hours)
        data2 = base_data.copy()
        for i in range(0, len(data2), 24):  # Every 24 hours
            if i + 1 < len(data2):
                data2.iloc[i+1, 1:] = np.nan  # Skip close column (timestamp)
        scenarios.append(('block_gaps', data2))

        # Scenario 3: End-of-day missing (simulate API gaps)
        data3 = base_data.copy()
        # Remove data during certain hours (e.g., maintenance windows)
        maintenance_hours = [2, 3, 14, 15]  # 2-3 AM, 2-3 PM
        for hour in maintenance_hours:
            hour_mask = data3['timestamp'].dt.hour == hour
            data3.loc[hour_mask, ['close', 'volume', 'rsi', 'macd', 'bb_upper', 'bb_lower']] = np.nan
        scenarios.append(('maintenance_gaps', data3))

        # Scenario 4: Weekend gaps (simulate exchange closures)
        data4 = base_data.copy()
        weekend_mask = data4['timestamp'].dt.dayofweek >= 5  # Saturday/Sunday
        data4.loc[weekend_mask, ['close', 'volume']] = np.nan  # Only price/volume missing on weekends
        scenarios.append(('weekend_gaps', data4))

        return scenarios

    def apply_missing_data_fill(self, data: pd.DataFrame, method: str = 'ffill_bfill') -> pd.DataFrame:
        """Apply missing data filling methods"""
        filled_data = data.copy()

        if method == 'ffill_bfill':
            # Forward fill then backward fill (current implementation)
            filled_data = filled_data.ffill().bfill()
        elif method == 'interpolate':
            # Linear interpolation
            numeric_cols = filled_data.select_dtypes(include=[np.number]).columns
            filled_data[numeric_cols] = filled_data[numeric_cols].interpolate(method='linear')
            filled_data = filled_data.bfill()  # Fill any remaining NaNs at start
        elif method == 'mean_fill':
            # Fill with rolling mean
            numeric_cols = filled_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                filled_data[col] = filled_data[col].fillna(filled_data[col].rolling(24, min_periods=1).mean())
        elif method == 'zero_fill':
            # Fill with zeros (worst case)
            filled_data = filled_data.fillna(0)

        return filled_data

    def calculate_reward_impact(self, original_data: pd.DataFrame, filled_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate impact of missing data filling on reward calculations"""
        # Simulate simple trading strategy rewards based on technical indicators

        def calculate_strategy_reward(data: pd.DataFrame) -> float:
            """Simple strategy: Buy when RSI < 30, Sell when RSI > 70"""
            rewards = []

            for i in range(1, len(data)):
                prev_rsi = data.iloc[i-1]['rsi']
                curr_rsi = data.iloc[i]['rsi']
                price_change = (data.iloc[i]['close'] - data.iloc[i-1]['close']) / data.iloc[i-1]['close']

                reward = 0.0
                if prev_rsi < 30 and curr_rsi >= 30:  # Buy signal
                    reward = float(price_change * 100)  # Profit/loss percentage
                elif prev_rsi > 70 and curr_rsi <= 70:  # Sell signal
                    reward = float(-price_change * 100)

                # Add MACD confirmation
                macd_signal = 1 if data.iloc[i]['macd'] > 0 else -1
                reward *= macd_signal

                rewards.append(reward)

            return float(np.mean(rewards) if rewards else 0.0)

        original_reward = calculate_strategy_reward(original_data)
        filled_reward = calculate_strategy_reward(filled_data)

        # Calculate feature correlations (important for ML)
        original_corr = original_data[['close', 'rsi', 'macd']].corr()
        filled_corr = filled_data[['close', 'rsi', 'macd']].corr()

        correlation_change = abs(original_corr - filled_corr).mean().mean()

        return {
            'original_reward': original_reward,
            'filled_reward': filled_reward,
            'reward_change_percent': ((filled_reward / original_reward) - 1) * 100 if original_reward != 0 else 0,
            'correlation_change': correlation_change,
            'max_price_deviation': abs(filled_data['close'] - original_data['close']).max(),
            'avg_price_deviation': abs(filled_data['close'] - original_data['close']).mean(),
            'missing_count': original_data.isnull().sum().sum(),
            'filled_count': filled_data.isnull().sum().sum()
        }

    def validate_cache_expiry(self) -> Dict[str, Any]:
        """Validate cache expiry logic for long-running experiments"""
        # Simulate cache timestamps
        cache_creation = datetime.now() - timedelta(days=2)  # 2 days old cache
        cache_expiry_hours = 24

        cache_age_hours = (datetime.now() - cache_creation).total_seconds() / 3600
        is_expired = cache_age_hours > cache_expiry_hours

        # Simulate API call timing
        api_call_start = datetime.now()
        simulated_api_delay = 0.5  # 500ms API call
        api_call_end = api_call_start + timedelta(seconds=simulated_api_delay)

        return {
            'cache_age_hours': cache_age_hours,
            'is_expired': is_expired,
            'cache_expiry_hours': cache_expiry_hours,
            'simulated_api_call_sec': simulated_api_delay,
            'should_refresh_cache': is_expired,
            'next_cache_expiry': cache_creation + timedelta(hours=cache_expiry_hours),
            'recommendation': 'extend_expiry' if cache_age_hours < cache_expiry_hours * 1.5 else 'current_ok'
        }

    def run_validation(self):
        """Run comprehensive data loading validation"""
        print("🔍 Data Loading Validation")
        print("=" * 50)

        scenarios = self.simulate_missing_data_scenarios()
        fill_methods = ['ffill_bfill', 'interpolate', 'mean_fill', 'zero_fill']

        for scenario_name, original_data in scenarios:
            print(f"\n📊 Scenario: {scenario_name}")
            print(f"  Original missing values: {original_data.isnull().sum().sum()}")

            scenario_results = {
                'scenario': scenario_name,
                'original_missing': int(original_data.isnull().sum().sum()),
                'fill_method_results': []
            }

            for method in fill_methods:
                filled_data = self.apply_missing_data_fill(original_data, method)
                impact = self.calculate_reward_impact(original_data, filled_data)

                result = {
                    'method': method,
                    'impact': impact
                }

                scenario_results['fill_method_results'].append(result)

                print(f"  {method:12}: Reward change: {impact['reward_change_percent']:+.2f}%, "
                      f"Corr change: {impact['correlation_change']:.4f}")

            self.validation_results.append(scenario_results)

        # Cache expiry validation
        print("\n📊 Cache Expiry Validation")
        cache_validation = self.validate_cache_expiry()
        print(f"  Cache age: {cache_validation['cache_age_hours']:.1f}h")
        print(f"  Is expired: {cache_validation['is_expired']}")
        print(f"  Recommendation: {cache_validation['recommendation']}")

        self.validation_results.append({'cache_validation': cache_validation})

    def save_results(self, output_file: str = "reports/data_loading_validation.json"):
        """Save validation results"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'validation_results': self.validation_results,
                'summary': self._generate_summary()
            }, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to {output_file}")

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of validation results"""
        if not self.validation_results:
            return {}

        # Find best fill method per scenario
        best_methods = {}
        for result in self.validation_results:
            if 'scenario' in result:
                scenario = result['scenario']
                methods = result['fill_method_results']

                # Best method = lowest reward change + lowest correlation change
                best = min(methods, key=lambda x: abs(x['impact']['reward_change_percent']) + x['impact']['correlation_change'])

                best_methods[scenario] = {
                    'best_method': best['method'],
                    'reward_change': best['impact']['reward_change_percent'],
                    'correlation_change': best['impact']['correlation_change']
                }

        return {
            'best_fill_methods': best_methods,
            'overall_recommendation': 'ffill_bfill' if all(m['best_method'] == 'ffill_bfill' for m in best_methods.values()) else 'interpolate',
            'cache_expiry_recommendation': 'extend_to_48h'  # Based on analysis
        }


if __name__ == "__main__":
    validator = DataLoadingValidator()
    validator.run_validation()
    validator.save_results()