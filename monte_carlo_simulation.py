#!/usr/bin/env python3
"""
モンテカルロシミュレーション分析モジュール

戦略のパフォーマンスを確率的にシミュレーションし、
将来の潜在的な結果を分析します。
"""

import argparse
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Import BenchmarkMetrics from comprehensive_benchmark
try:
    from comprehensive_benchmark import BenchmarkMetrics
except ImportError:
    # Fallback if not available
    BenchmarkMetrics = None

@dataclass
class MonteCarloResult:
    """モンテカルロシミュレーション結果"""
    simulation_returns: np.ndarray
    final_values: np.ndarray
    percentiles: Dict[str, float]
    statistics: Dict[str, float]
    var_95: float
    cvar_95: float
    probability_of_loss: float
    probability_of_profit: float

@dataclass
class MonteCarloAnalysisResult:
    """モンテカルロ分析全体結果"""
    baseline_result: MonteCarloResult
    stressed_results: Optional[Dict[str, MonteCarloResult]] = None
    scenario_analysis: Optional[Dict[str, Dict[str, float]]] = None

class MonteCarloSimulator:
    """モンテカルロシミュレーター"""

    def __init__(self, n_simulations: int = 1000, time_horizon: int = 252,
                 confidence_level: float = 0.95, random_state: int = 42):
        """
        Args:
            n_simulations: シミュレーション回数
            time_horizon: 時間ホライズン（日数）
            confidence_level: 信頼区間レベル
            random_state: 乱数シード
        """
        self.n_simulations = n_simulations
        self.time_horizon = time_horizon
        self.confidence_level = confidence_level
        self.random_state = random_state
        np.random.seed(random_state)

    def fit_distribution(self, returns: pd.Series,
                        distribution: str = 'normal') -> Dict[str, Any]:
        """
        リターンの分布をフィット

        Args:
            returns: ヒストリカルリターン
            distribution: 分布タイプ ('normal', 't', 'lognormal')

        Returns:
            分布パラメータ
        """
        returns_array = returns.values

        if distribution == 'normal':
            # 正規分布
            params = {
                'type': 'normal',
                'mean': np.mean(returns_array),
                'std': np.std(returns_array)
            }
        elif distribution == 't':
            # t分布
            df, loc, scale = stats.t.fit(returns_array)
            params = {
                'type': 't',
                'df': df,
                'loc': loc,
                'scale': scale
            }
        elif distribution == 'lognormal':
            # 対数正規分布
            log_returns = np.log(1 + returns_array)
            params = {
                'type': 'lognormal',
                'mean': np.mean(log_returns),
                'std': np.std(log_returns)
            }
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

        return params

    def generate_random_returns(self, params: Dict[str, Any],
                              n_periods: int) -> np.ndarray:
        """
        分布パラメータからランダムリターンを生成

        Args:
            params: 分布パラメータ
            n_periods: 生成する期間数

        Returns:
            ランダムリターン配列
        """
        if params['type'] == 'normal':
            returns = np.random.normal(params['mean'], params['std'], n_periods)
        elif params['type'] == 't':
            returns = stats.t.rvs(params['df'], loc=params['loc'],
                                scale=params['scale'], size=n_periods)
        elif params['type'] == 'lognormal':
            log_returns = np.random.normal(params['mean'], params['std'], n_periods)
            returns = np.exp(log_returns) - 1
        else:
            raise ValueError(f"Unsupported distribution type: {params['type']}")

        return returns

    def simulate_portfolio_path(self, initial_value: float,
                              returns: np.ndarray,
                              transaction_costs: float = 0.0) -> np.ndarray:
        """
        ポートフォリオの価値推移をシミュレート

        Args:
            initial_value: 初期価値
            returns: リターン配列
            transaction_costs: 取引コスト（1取引あたり）

        Returns:
            価値推移配列
        """
        # 取引コストを考慮（簡易版）
        adjusted_returns = returns - transaction_costs

        # 累積リターンを計算
        cumulative_returns = np.cumprod(1 + adjusted_returns)

        # ポートフォリオ価値
        portfolio_values = initial_value * cumulative_returns

        return portfolio_values

    def run_monte_carlo_simulation(self, returns: pd.Series,
                                 initial_value: float = 10000,
                                 distribution: str = 'normal',
                                 transaction_costs: float = 0.0) -> MonteCarloResult:
        """
        モンテカルロシミュレーションを実行

        Args:
            returns: ヒストリカルリターン
            initial_value: 初期ポートフォリオ価値
            distribution: 使用する分布
            transaction_costs: 取引コスト

        Returns:
            シミュレーション結果
        """
        # 分布のフィット
        params = self.fit_distribution(returns, distribution)

        # シミュレーション実行
        simulation_returns = []
        final_values = []

        for _ in range(self.n_simulations):
            # ランダムリターンの生成
            random_returns = self.generate_random_returns(params, self.time_horizon)

            # ポートフォリオ価値の計算
            portfolio_values = self.simulate_portfolio_path(
                initial_value, random_returns, transaction_costs
            )

            # リターンの保存
            simulation_returns.append(random_returns)
            final_values.append(portfolio_values[-1])

        simulation_returns = np.array(simulation_returns)
        final_values = np.array(final_values)

        # パーセンタイルの計算
        percentiles = {
            '1%': np.percentile(final_values, 1),
            '5%': np.percentile(final_values, 5),
            '10%': np.percentile(final_values, 10),
            '25%': np.percentile(final_values, 25),
            '50%': np.percentile(final_values, 50),
            '75%': np.percentile(final_values, 75),
            '90%': np.percentile(final_values, 90),
            '95%': np.percentile(final_values, 95),
            '99%': np.percentile(final_values, 99)
        }

        # 統計量の計算
        statistics = {
            'mean': float(np.mean(final_values)),
            'std': float(np.std(final_values)),
            'min': float(np.min(final_values)),
            'max': float(np.max(final_values)),
            'skewness': float(stats.skew(final_values)),
            'kurtosis': float(stats.kurtosis(final_values))
        }

        # VaRとCVaR
        var_95 = initial_value - np.percentile(final_values, 5)
        cvar_95 = initial_value - np.mean(final_values[final_values <= np.percentile(final_values, 5)])

        # 確率計算
        probability_of_loss = np.mean(final_values < initial_value)
        probability_of_profit = np.mean(final_values > initial_value)

        return MonteCarloResult(
            simulation_returns=simulation_returns,
            final_values=final_values,
            percentiles=percentiles,
            statistics=statistics,
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            probability_of_loss=float(probability_of_loss),
            probability_of_profit=float(probability_of_profit)
        )

    def stress_test_scenarios(self, returns: pd.Series,
                            initial_value: float = 10000) -> Dict[str, MonteCarloResult]:
        """
        ストレステストシナリオの実行

        Args:
            returns: ヒストリカルリターン
            initial_value: 初期価値

        Returns:
            シナリオ別結果
        """
        scenarios = {}

        # ベースケース
        scenarios['baseline'] = self.run_monte_carlo_simulation(returns, initial_value)

        # 高ボラティリティシナリオ
        high_vol_returns = returns * 1.5
        scenarios['high_volatility'] = self.run_monte_carlo_simulation(high_vol_returns, initial_value)

        # 低リターンスシナリオ
        low_return_returns = returns * 0.7
        scenarios['low_returns'] = self.run_monte_carlo_simulation(low_return_returns, initial_value)

        # クラッシュシナリオ
        crash_returns = returns.copy()
        crash_point = len(crash_returns) // 2
        crash_returns.iloc[crash_point] = -0.3  # 30%下落
        scenarios['market_crash'] = self.run_monte_carlo_simulation(crash_returns, initial_value)

        # 高コストシナリオ
        scenarios['high_costs'] = self.run_monte_carlo_simulation(
            returns, initial_value, transaction_costs=0.005  # 0.5%コスト
        )

        return scenarios

    def analyze_scenarios(self, scenario_results: Dict[str, MonteCarloResult]) -> Dict[str, Dict[str, float]]:
        """
        シナリオ間の比較分析

        Args:
            scenario_results: シナリオ別結果

        Returns:
            比較分析結果
        """
        analysis = {}

        for scenario_name, result in scenario_results.items():
            analysis[scenario_name] = {
                'expected_value': result.statistics['mean'],
                'volatility': result.statistics['std'],
                'var_95': result.var_95,
                'cvar_95': result.cvar_95,
                'probability_of_loss': result.probability_of_loss,
                'worst_case': result.statistics['min'],
                'best_case': result.statistics['max']
            }

        return analysis

    def run_comprehensive_analysis(self, returns: pd.Series,
                                 initial_value: float = 10000,
                                 include_stress_tests: bool = True) -> MonteCarloAnalysisResult:
        """
        包括的なモンテカルロ分析を実行

        Args:
            returns: ヒストリカルリターン
            initial_value: 初期価値
            include_stress_tests: ストレステストを含むか

        Returns:
            包括的な分析結果
        """
        # ベースライン分析
        baseline_result = self.run_monte_carlo_simulation(returns, initial_value)

        # ストレステスト
        stressed_results = None
        scenario_analysis = None

        if include_stress_tests:
            stressed_results = self.stress_test_scenarios(returns, initial_value)
            scenario_analysis = self.analyze_scenarios(stressed_results)

        return MonteCarloAnalysisResult(
            baseline_result=baseline_result,
            stressed_results=stressed_results,
            scenario_analysis=scenario_analysis
        )

    def analyze(self, metrics) -> Dict[str, Any]:
        """
        BenchmarkMetricsからモンテカルロ分析を実行

        Args:
            metrics: 評価メトリクス

        Returns:
            分析結果の辞書
        """
        if BenchmarkMetrics is None:
            raise ImportError("BenchmarkMetrics not available. Please ensure comprehensive_benchmark.py is in the path.")

        # returnsデータをpandas Seriesに変換
        if not metrics.returns:
            return {
                'error': 'No returns data available for analysis',
                'simulation_score': 0.0,
                'expected_return': 0.0,
                'value_at_risk': 0.0,
                'expected_shortfall': 0.0
            }

        # 日付インデックスを作成（仮定）
        dates = pd.date_range(start='2023-01-01', periods=len(metrics.returns), freq='D')
        returns = pd.Series(metrics.returns, index=dates)

        try:
            # モンテカルロ分析を実行（簡易版）
            result = self.run_monte_carlo_simulation(
                returns=returns,
                initial_value=10000.0
            )

            # 結果を辞書形式に変換
            analysis_result: Dict[str, Any] = {
                'simulation_score': 1.0 - abs(result.var_95) / 0.5,  # VaRが小さいほど良い
                'expected_return': float(result.statistics.get('mean', 0.0)),
                'volatility': float(result.statistics.get('std', 0.0)),
                'value_at_risk_95': float(result.var_95),
                'value_at_risk_99': float(result.percentiles.get('99%', 0.0)),
                'expected_shortfall': float(result.cvar_95),
                'max_drawdown': 0.0,  # 計算されていない
                'sharpe_ratio': 0.0,  # 計算されていない
                'total_return': 0.0,  # 計算されていない
                'win_probability': float(result.probability_of_profit),
                'final_values': result.final_values.tolist() if hasattr(result.final_values, 'tolist') else [],
                'simulation_paths': result.simulation_paths.tolist() if hasattr(result.simulation_paths, 'tolist') else []
            }

            return analysis_result

        except Exception as e:
            return {
                'error': f'Analysis failed: {str(e)}',
                'simulation_score': 0.0,
                'expected_return': 0.0,
                'value_at_risk': 0.0,
                'expected_shortfall': 0.0
            }

    def plot_monte_carlo_results(self, result: MonteCarloAnalysisResult,
                               save_path: Optional[str] = None):
        """モンテカルロ分析結果を可視化"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Monte Carlo Simulation Results', fontsize=16)

        baseline = result.baseline_result

        # 最終価値の分布
        axes[0, 0].hist(baseline.final_values, bins=50, alpha=0.7, color='skyblue', density=True)
        axes[0, 0].axvline(np.mean(baseline.final_values), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: ${np.mean(baseline.final_values):.0f}')
        axes[0, 0].axvline(np.percentile(baseline.final_values, 5), color='orange', linestyle=':',
                          linewidth=2, label=f'5th percentile: ${np.percentile(baseline.final_values, 5):.0f}')
        axes[0, 0].set_title('Portfolio Value Distribution')
        axes[0, 0].set_xlabel('Final Value ($)')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 累積確率分布
        sorted_values = np.sort(baseline.final_values)
        cumulative_prob = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

        axes[0, 1].plot(sorted_values, cumulative_prob, 'b-', linewidth=2)
        axes[0, 1].axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='5% probability')
        axes[0, 1].axvline(x=np.percentile(baseline.final_values, 5), color='r', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Cumulative Distribution')
        axes[0, 1].set_xlabel('Portfolio Value ($)')
        axes[0, 1].set_ylabel('Cumulative Probability')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # リスク指標
        risk_labels = ['VaR 95%', 'CVaR 95%', 'Prob Loss', 'Prob Profit']
        risk_values = [baseline.var_95, baseline.cvar_95,
                      baseline.probability_of_loss * 100,
                      baseline.probability_of_profit * 100]

        bars = axes[0, 2].bar(risk_labels, risk_values, color='salmon', alpha=0.7)
        axes[0, 2].set_title('Risk Metrics')
        axes[0, 2].set_ylabel('Value')
        axes[0, 2].tick_params(axis='x', rotation=45)

        # 値ラベルを追加
        for bar, value in zip(bars, risk_values):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:.1f}', ha='center', va='bottom', fontsize=8)

        # シナリオ比較（利用可能な場合）
        if result.scenario_analysis:
            scenarios = list(result.scenario_analysis.keys())
            expected_values = [result.scenario_analysis[s]['expected_value'] for s in scenarios]

            axes[1, 0].bar(scenarios, expected_values, color='lightgreen', alpha=0.7)
            axes[1, 0].set_title('Expected Values by Scenario')
            axes[1, 0].set_ylabel('Expected Value ($)')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # 確率比較
            loss_probs = [result.scenario_analysis[s]['probability_of_loss'] * 100 for s in scenarios]
            axes[1, 1].bar(scenarios, loss_probs, color='coral', alpha=0.7)
            axes[1, 1].set_title('Probability of Loss by Scenario')
            axes[1, 1].set_ylabel('Probability (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)

            # VaR比較
            var_values = [result.scenario_analysis[s]['var_95'] for s in scenarios]
            axes[1, 2].bar(scenarios, var_values, color='purple', alpha=0.7)
            axes[1, 2].set_title('VaR 95% by Scenario')
            axes[1, 2].set_ylabel('VaR ($)')
            axes[1, 2].tick_params(axis='x', rotation=45)
        else:
            for i in range(3):
                axes[1, i].text(0.5, 0.5, 'Stress tests not included',
                               ha='center', va='center', transform=axes[1, i].transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def export_results(self, result: MonteCarloAnalysisResult,
                      output_path: str):
        """分析結果を JSON ファイルにエクスポート"""
        export_data = {
            'baseline': {
                'percentiles': result.baseline_result.percentiles,
                'statistics': result.baseline_result.statistics,
                'var_95': result.baseline_result.var_95,
                'cvar_95': result.baseline_result.cvar_95,
                'probability_of_loss': result.baseline_result.probability_of_loss,
                'probability_of_profit': result.baseline_result.probability_of_profit
            }
        }

        if result.scenario_analysis:
            export_data['scenario_analysis'] = result.scenario_analysis

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"Results exported to {output_path}")

def main():
    """コマンドラインインターフェース"""
    parser = argparse.ArgumentParser(description='Monte Carlo Simulation Analysis')
    parser.add_argument('--returns-path', required=True, help='Path to returns data (CSV)')
    parser.add_argument('--initial-value', type=float, default=10000, help='Initial portfolio value')
    parser.add_argument('--n-simulations', type=int, default=1000, help='Number of simulations')
    parser.add_argument('--time-horizon', type=int, default=252, help='Time horizon (days)')
    parser.add_argument('--distribution', choices=['normal', 't', 'lognormal'], default='normal', help='Return distribution')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--no-stress-tests', action='store_true', help='Skip stress tests')

    args = parser.parse_args()

    # リターンデータの読み込み
    returns_df = pd.read_csv(args.returns_path)
    if 'returns' in returns_df.columns:
        returns = returns_df.set_index('date' if 'date' in returns_df.columns else returns_df.columns[0])['returns']
    else:
        returns = returns_df.set_index(returns_df.columns[0]).iloc[:, 0]

    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)

    # 分析の実行
    simulator = MonteCarloSimulator(
        n_simulations=args.n_simulations,
        time_horizon=args.time_horizon
    )

    result = simulator.run_comprehensive_analysis(
        returns=returns,
        initial_value=args.initial_value,
        include_stress_tests=not args.no_stress_tests
    )

    # 結果の保存
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # プロットの保存
    plot_path = os.path.join(args.output_dir, 'monte_carlo_analysis.png')
    simulator.plot_monte_carlo_results(result, save_path=plot_path)

    # JSON エクスポート
    json_path = os.path.join(args.output_dir, 'monte_carlo_results.json')
    simulator.export_results(result, json_path)

    print("Monte Carlo simulation analysis completed!")

if __name__ == '__main__':
    main()