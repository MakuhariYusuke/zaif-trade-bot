#!/usr/bin/env python3
"""
戦略堅牢性分析モジュール

戦略パラメータの感度分析を行い、戦略の安定性と最適パラメータ範囲を評価します。
"""

import argparse
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.model_selection import ParameterGrid
from scipy import stats
from scipy.optimize import minimize_scalar

# Import BenchmarkMetrics from comprehensive_benchmark
try:
    from comprehensive_benchmark import BenchmarkMetrics
except ImportError:
    # Fallback if not available
    BenchmarkMetrics = None

@dataclass
class ParameterSensitivity:
    """パラメータ感度"""
    parameter_name: str
    parameter_range: Tuple[float, float]
    sensitivity_score: float
    optimal_value: float
    confidence_interval: Tuple[float, float]

@dataclass
class RobustnessResult:
    """堅牢性分析結果"""
    parameter_sensitivities: List[ParameterSensitivity]
    stability_score: float
    robustness_score: float
    optimal_parameters: Dict[str, float]
    sensitivity_matrix: pd.DataFrame

@dataclass
class StrategyRobustnessResult:
    """戦略堅牢性分析全体結果"""
    baseline_performance: Dict[str, float]
    robustness_analysis: RobustnessResult
    parameter_ranges: Dict[str, Tuple[float, float]]
    monte_carlo_stability: Optional[Dict[str, float]] = None

class StrategyRobustnessAnalyzer:
    """戦略堅牢性分析クラス"""

    def __init__(self, n_simulations: int = 100, confidence_level: float = 0.95):
        """
        Args:
            n_simulations: 感度分析のシミュレーション回数
            confidence_level: 信頼区間レベル
        """
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level

    def evaluate_strategy_performance(self, strategy_func: Callable[..., Dict[str, float]],
                                    parameters: Dict[str, float],
                                    data: pd.DataFrame) -> Dict[str, float]:
        """
        戦略のパフォーマンスを評価

        Args:
            strategy_func: 戦略評価関数
            parameters: 戦略パラメータ
            data: 市場データ

        Returns:
            パフォーマンス指標
        """
        try:
            # 戦略の実行
            result = strategy_func(data, **parameters)

            # パフォーマンス指標の計算
            if isinstance(result, dict) and 'returns' in result:
                returns = result['returns']
            elif isinstance(result, pd.Series):
                returns = result
            else:
                # デフォルトの指標計算
                returns = pd.Series(result)

            # 基本指標
            total_return = float((1 + returns).prod() - 1)  # type: ignore
            volatility = float(returns.std() * np.sqrt(252))  # type: ignore
            sharpe_ratio = float(total_return / volatility if volatility > 0 else 0)
            max_drawdown = float((returns.cumsum() - returns.cumsum().cummax()).min())  # type: ignore

            return {
                'total_return': float(total_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'win_rate': float((returns > 0).mean())  # type: ignore
            }

        except Exception as e:
            warnings.warn(f"Strategy evaluation failed: {e}")
            return {
                'total_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }

    def calculate_parameter_sensitivity(self, strategy_func: Callable[..., Dict[str, float]],
                                      base_parameters: Dict[str, float],
                                      parameter_ranges: Dict[str, Tuple[float, float]],
                                      data: pd.DataFrame,
                                      metric: str = 'sharpe_ratio') -> ParameterSensitivity:
        """
        単一パラメータの感度を計算

        Args:
            strategy_func: 戦略関数
            base_parameters: 基準パラメータ
            parameter_ranges: パラメータ範囲
            data: 市場データ
            metric: 評価指標

        Returns:
            パラメータ感度
        """
        param_name = list(parameter_ranges.keys())[0]
        param_range = parameter_ranges[param_name]

        # パラメータ値を生成
        param_values = np.linspace(param_range[0], param_range[1], self.n_simulations)
        performances = []

        for param_value in param_values:
            test_params = base_parameters.copy()
            test_params[param_name] = param_value

            perf = self.evaluate_strategy_performance(strategy_func, test_params, data)
            performances.append(perf[metric])

        performances = np.array(performances)

        # 感度スコアの計算（標準偏差）
        sensitivity_score = np.std(performances)

        # 最適値の探索
        best_idx = np.argmax(performances)
        optimal_value = param_values[best_idx]

        # 信頼区間の計算
        ci_lower = float(np.percentile(performances, (1 - self.confidence_level) * 100 / 2))
        ci_upper = float(np.percentile(performances, (1 + self.confidence_level) * 100 / 2))
        confidence_interval = (ci_lower, ci_upper)

        return ParameterSensitivity(
            parameter_name=param_name,
            parameter_range=param_range,
            sensitivity_score=float(sensitivity_score),
            optimal_value=float(optimal_value),
            confidence_interval=confidence_interval
        )

    def analyze_parameter_interactions(self, strategy_func: Callable[..., Dict[str, float]],
                                     parameters: Dict[str, float],
                                     parameter_ranges: Dict[str, Tuple[float, float]],
                                     data: pd.DataFrame,
                                     metric: str = 'sharpe_ratio') -> pd.DataFrame:
        """
        パラメータ間の相互作用を分析

        Args:
            strategy_func: 戦略関数
            parameters: 基準パラメータ
            parameter_ranges: パラメータ範囲
            data: 市場データ
            metric: 評価指標

        Returns:
            感度行列
        """
        param_names = list(parameter_ranges.keys())
        n_params = len(param_names)

        # パラメータグリッドの生成
        param_grid = {}
        for param_name, param_range in parameter_ranges.items():
            param_grid[param_name] = np.linspace(param_range[0], param_range[1], 10)

        grid = ParameterGrid(param_grid)
        results = []

        for param_set in grid:
            test_params = parameters.copy()
            test_params.update(param_set)

            perf = self.evaluate_strategy_performance(strategy_func, test_params, data)
            row = {param_name: param_set[param_name] for param_name in param_names}
            row[metric] = perf[metric]
            results.append(row)

        return pd.DataFrame(results)

    def calculate_stability_score(self, sensitivity_scores: List[float]) -> float:
        """
        安定性スコアの計算

        Args:
            sensitivity_scores: 各パラメータの感度スコア

        Returns:
            安定性スコア（0-1、1が安定）
        """
        if not sensitivity_scores:
            return 1.0

        # 感度スコアの逆数を使用（感度が低いほど安定）
        avg_sensitivity = np.mean(sensitivity_scores)
        if avg_sensitivity == 0:
            return 1.0

        stability_score = 1 / (1 + avg_sensitivity)
        return float(min(stability_score, 1.0))

    def calculate_robustness_score(self, performances: List[float],
                                 baseline_performance: float) -> float:
        """
        堅牢性スコアの計算

        Args:
            performances: パラメータ変化時のパフォーマンス
            baseline_performance: 基準パフォーマンス

        Returns:
            堅牢性スコア（0-1、1が堅牢）
        """
        if not performances or baseline_performance == 0:
            return 0.0

        # パフォーマンスの変動係数
        cv = np.std(performances) / abs(baseline_performance)

        # 堅牢性スコア（変動係数の逆数）
        robustness_score = 1 / (1 + cv)
        return float(min(robustness_score, 1.0))

    def find_optimal_parameters(self, strategy_func: Callable,
                              parameter_ranges: Dict[str, Tuple[float, float]],
                              data: pd.DataFrame,
                              metric: str = 'sharpe_ratio') -> Dict[str, float]:
        """
        最適パラメータの探索

        Args:
            strategy_func: 戦略関数
            parameter_ranges: パラメータ範囲
            data: 市場データ
            metric: 最適化指標

        Returns:
            最適パラメータ
        """
        def objective(params_list: List[float]) -> float:
            param_dict = dict(zip(parameter_ranges.keys(), params_list))
            perf = self.evaluate_strategy_performance(strategy_func, param_dict, data)
            return -perf[metric]  # 最大化なので符号反転

        # 初期値
        initial_guess = [np.mean(param_range) for param_range in parameter_ranges.values()]

        # 境界
        bounds = list(parameter_ranges.values())

        try:
            # 最適化実行（簡易版：各パラメータを個別に最適化）
            optimal_params = {}
            for param_name, param_range in parameter_ranges.items():
                result = minimize_scalar(
                    lambda x: -self.evaluate_strategy_performance(
                        strategy_func,
                        {param_name: x, **{k: v for k, v in optimal_params.items()}},
                        data
                    )[metric],
                    bounds=param_range,
                    method='bounded'
                )
                optimal_params[param_name] = float(result.x)

            return optimal_params

        except Exception as e:
            warnings.warn(f"Parameter optimization failed: {e}")
            # デフォルト値を使用
            return {name: float(np.mean(param_range)) for name, param_range in parameter_ranges.items()}

    def run_robustness_analysis(self, strategy_func: Callable[..., Dict[str, float]],
                              base_parameters: Dict[str, float],
                              parameter_ranges: Dict[str, Tuple[float, float]],
                              data: pd.DataFrame,
                              metric: str = 'sharpe_ratio') -> RobustnessResult:
        """
        堅牢性分析を実行

        Args:
            strategy_func: 戦略関数
            base_parameters: 基準パラメータ
            parameter_ranges: パラメータ範囲
            data: 市場データ
            metric: 評価指標

        Returns:
            堅牢性分析結果
        """
        # 基準パフォーマンス
        baseline_perf = self.evaluate_strategy_performance(strategy_func, base_parameters, data)
        baseline_value = baseline_perf[metric]

        # パラメータ感度の計算
        parameter_sensitivities = []
        all_performances = []

        for param_name, param_range in parameter_ranges.items():
            sensitivity = self.calculate_parameter_sensitivity(
                strategy_func, base_parameters, {param_name: param_range}, data, metric
            )
            parameter_sensitivities.append(sensitivity)
            all_performances.extend([
                self.evaluate_strategy_performance(
                    strategy_func,
                    {**base_parameters, param_name: val}, data
                )[metric]
                for val in np.linspace(param_range[0], param_range[1], 20)
            ])

        # 感度行列の計算
        sensitivity_matrix = self.analyze_parameter_interactions(
            strategy_func, base_parameters, parameter_ranges, data, metric
        )

        # 安定性と堅牢性の計算
        sensitivity_scores = [s.sensitivity_score for s in parameter_sensitivities]
        stability_score = self.calculate_stability_score(sensitivity_scores)
        robustness_score = self.calculate_robustness_score(all_performances, baseline_value)

        # 最適パラメータの探索
        optimal_parameters = self.find_optimal_parameters(strategy_func, parameter_ranges, data, metric)

        return RobustnessResult(
            parameter_sensitivities=parameter_sensitivities,
            stability_score=stability_score,
            robustness_score=robustness_score,
            optimal_parameters=optimal_parameters,
            sensitivity_matrix=sensitivity_matrix
        )

    def run_monte_carlo_stability_test(self, strategy_func: Callable[..., Dict[str, float]],
                                     parameters: Dict[str, float],
                                     parameter_ranges: Dict[str, Tuple[float, float]],
                                     data: pd.DataFrame,
                                     n_monte_carlo: int = 100,
                                     metric: str = 'sharpe_ratio') -> Dict[str, float]:
        """
        モンテカルロ安定性テスト

        Args:
            strategy_func: 戦略関数
            parameters: 基準パラメータ
            parameter_ranges: パラメータ範囲
            data: 市場データ
            n_monte_carlo: モンテカルロ試行回数
            metric: 評価指標

        Returns:
            安定性指標
        """
        performances = []

        for _ in range(n_monte_carlo):
            # パラメータのランダムサンプリング
            random_params = parameters.copy()
            for param_name, param_range in parameter_ranges.items():
                random_params[param_name] = np.random.uniform(param_range[0], param_range[1])

            perf = self.evaluate_strategy_performance(strategy_func, random_params, data)
            performances.append(perf[metric])

        performances = np.array(performances)

        return {
            'mean_performance': float(np.mean(performances)),
            'std_performance': float(np.std(performances)),
            'cv_performance': float(np.std(performances) / abs(np.mean(performances))),
            'best_performance': float(np.max(performances)),
            'worst_performance': float(np.min(performances)),
            'performance_range': float(np.max(performances) - np.min(performances))
        }

    def run_comprehensive_robustness_analysis(self, strategy_func: Callable[..., Dict[str, float]],
                                            base_parameters: Dict[str, float],
                                            parameter_ranges: Dict[str, Tuple[float, float]],
                                            data: pd.DataFrame,
                                            metric: str = 'sharpe_ratio',
                                            include_monte_carlo: bool = True) -> StrategyRobustnessResult:
        """
        包括的な堅牢性分析を実行

        Args:
            strategy_func: 戦略関数
            base_parameters: 基準パラメータ
            parameter_ranges: パラメータ範囲
            data: 市場データ
            metric: 評価指標
            include_monte_carlo: モンテカルロテストを含むか

        Returns:
            包括的な堅牢性分析結果
        """
        # 基準パフォーマンス
        baseline_performance = self.evaluate_strategy_performance(strategy_func, base_parameters, data)

        # 堅牢性分析
        robustness_analysis = self.run_robustness_analysis(
            strategy_func, base_parameters, parameter_ranges, data, metric
        )

        # モンテカルロ安定性テスト
        monte_carlo_stability = None
        if include_monte_carlo:
            monte_carlo_stability = self.run_monte_carlo_stability_test(
                strategy_func, base_parameters, parameter_ranges, data, metric=metric
            )

        return StrategyRobustnessResult(
            baseline_performance=baseline_performance,
            robustness_analysis=robustness_analysis,
            parameter_ranges=parameter_ranges,
            monte_carlo_stability=monte_carlo_stability
        )

    def analyze(self, metrics) -> Dict[str, Any]:
        """
        BenchmarkMetricsから戦略堅牢性分析を実行

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
                'robustness_score': 0.0,
                'stability_score': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0
            }

        # 日付インデックスを作成（仮定）
        dates = pd.date_range(start='2023-01-01', periods=len(metrics.returns), freq='D')
        returns = pd.Series(metrics.returns, index=dates)

        try:
            # 基本的な統計分析
            returns_mean = float(returns.mean())
            returns_std = float(returns.std())
            returns_array = returns.values
            skewness = float(stats.skew(returns_array))
            kurtosis = float(stats.kurtosis(returns_array))

            # ロバストネス指標の計算
            # 1. 収益の安定性（標準偏差の逆数）
            stability_score = 1.0 / (1.0 + returns_std) if returns_std > 0 else 1.0

            # 2. 収益のロバストネス（シャープレシオに基づく）
            sharpe_ratio = metrics.sharpe_ratio if hasattr(metrics, 'sharpe_ratio') and metrics.sharpe_ratio != 0 else returns_mean / returns_std if returns_std > 0 else 0.0

            # 3. ドローダウン分析
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = float(drawdowns.min())
            avg_drawdown = float(drawdowns.mean())

            # 4. 収益分布の分析
            returns_quantiles = returns.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()

            # 5. 連続収益の分析
            positive_streaks = []
            negative_streaks = []
            current_streak = 0
            current_sign = 0

            for ret in returns:
                if ret > 0:
                    if current_sign <= 0:
                        if current_streak < 0:
                            negative_streaks.append(current_streak)
                        current_streak = 1
                        current_sign = 1
                    else:
                        current_streak += 1
                elif ret < 0:
                    if current_sign >= 0:
                        if current_streak > 0:
                            positive_streaks.append(current_streak)
                        current_streak = -1
                        current_sign = -1
                    else:
                        current_streak -= 1
                else:
                    if current_streak != 0:
                        if current_streak > 0:
                            positive_streaks.append(current_streak)
                        elif current_streak < 0:
                            negative_streaks.append(current_streak)
                        current_streak = 0
                        current_sign = 0

            # 残りのストリークを追加
            if current_streak > 0:
                positive_streaks.append(current_streak)
            elif current_streak < 0:
                negative_streaks.append(current_streak)

            max_positive_streak = max(positive_streaks) if positive_streaks else 0
            max_negative_streak = abs(min(negative_streaks)) if negative_streaks else 0
            avg_positive_streak = float(np.mean(positive_streaks)) if positive_streaks else 0.0
            avg_negative_streak = abs(float(np.mean(negative_streaks))) if negative_streaks else 0.0

            # 総合ロバストネススコア
            robustness_score = (
                0.3 * stability_score +
                0.3 * max(0, sharpe_ratio / 2.0) +  # シャープレシオを正規化
                0.2 * (1.0 / (1.0 + abs(max_drawdown))) +  # 最大ドローダウンを正規化
                0.2 * max(0, min(1, (skewness / 2.0 + 0.5)))  # 歪度を0-1に正規化
            )

            return {
                'robustness_score': float(robustness_score),
                'stability_score': float(stability_score),
                'volatility': returns_std,
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': max_drawdown,
                'avg_drawdown': avg_drawdown,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'returns_quantiles': returns_quantiles,
                'max_positive_streak': int(max_positive_streak),
                'max_negative_streak': int(max_negative_streak),
                'avg_positive_streak': avg_positive_streak,
                'avg_negative_streak': avg_negative_streak,
                'total_positive_days': int((returns > 0).sum()),
                'total_negative_days': int((returns < 0).sum()),
                'win_rate': float((returns > 0).mean())
            }

        except Exception as e:
            return {
                'error': f'Analysis failed: {str(e)}',
                'robustness_score': 0.0,
                'stability_score': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0
            }

    def plot_robustness_analysis(self, result: StrategyRobustnessResult,
                               save_path: Optional[str] = None):
        """堅牢性分析結果を可視化"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Strategy Robustness Analysis', fontsize=16)

        # パラメータ感度
        param_names = [s.parameter_name for s in result.robustness_analysis.parameter_sensitivities]
        sensitivity_scores = [s.sensitivity_score for s in result.robustness_analysis.parameter_sensitivities]

        bars = axes[0, 0].bar(param_names, sensitivity_scores, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Parameter Sensitivity')
        axes[0, 0].set_ylabel('Sensitivity Score')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 値ラベルを追加
        for bar, value in zip(bars, sensitivity_scores):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        # 安定性スコア
        stability_labels = ['Stability Score', 'Robustness Score']
        stability_values = [result.robustness_analysis.stability_score,
                           result.robustness_analysis.robustness_score]

        bars2 = axes[0, 1].bar(stability_labels, stability_values, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Stability Metrics')
        axes[0, 1].set_ylabel('Score (0-1)')
        axes[0, 1].set_ylim(0, 1)

        for bar, value in zip(bars2, stability_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        # パラメータ vs パフォーマンス（最初の2パラメータ）
        if len(result.robustness_analysis.parameter_sensitivities) >= 2:
            param1 = result.robustness_analysis.parameter_sensitivities[0]
            param2 = result.robustness_analysis.parameter_sensitivities[1]

            # 感度行列からデータを抽出
            matrix = result.robustness_analysis.sensitivity_matrix
            if len(matrix.columns) >= 3:  # パラメータ + 指標
                x_vals = matrix[param1.parameter_name].values
                y_vals = matrix[param2.parameter_name].values
                z_vals = matrix.iloc[:, -1].values  # 最後の列が指標

                scatter = axes[0, 2].scatter(x_vals, y_vals, c=z_vals, cmap='viridis', alpha=0.7)
                axes[0, 2].set_xlabel(param1.parameter_name)
                axes[0, 2].set_ylabel(param2.parameter_name)
                axes[0, 2].set_title('Parameter Interaction')
                plt.colorbar(scatter, ax=axes[0, 2])
        else:
            axes[0, 2].text(0.5, 0.5, 'Need at least 2 parameters\nfor interaction plot',
                           ha='center', va='center', transform=axes[0, 2].transAxes)

        # モンテカルロ安定性（利用可能な場合）
        if result.monte_carlo_stability:
            mc_labels = ['Mean', 'Std Dev', 'CV', 'Best', 'Worst']
            mc_values = [
                result.monte_carlo_stability['mean_performance'],
                result.monte_carlo_stability['std_performance'],
                result.monte_carlo_stability['cv_performance'],
                result.monte_carlo_stability['best_performance'],
                result.monte_carlo_stability['worst_performance']
            ]

            bars3 = axes[1, 0].bar(mc_labels, mc_values, color='salmon', alpha=0.7)
            axes[1, 0].set_title('Monte Carlo Stability')
            axes[1, 0].set_ylabel('Performance')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'Monte Carlo test\nnot included',
                           ha='center', va='center', transform=axes[1, 0].transAxes)

        # 最適パラメータ
        opt_param_names = list(result.robustness_analysis.optimal_parameters.keys())
        opt_param_values = list(result.robustness_analysis.optimal_parameters.values())

        bars4 = axes[1, 1].bar(opt_param_names, opt_param_values, color='purple', alpha=0.7)
        axes[1, 1].set_title('Optimal Parameters')
        axes[1, 1].set_ylabel('Parameter Value')
        axes[1, 1].tick_params(axis='x', rotation=45)

        # 基準パフォーマンス
        baseline_labels = list(result.baseline_performance.keys())
        baseline_values = list(result.baseline_performance.values())

        bars5 = axes[1, 2].bar(baseline_labels, baseline_values, color='orange', alpha=0.7)
        axes[1, 2].set_title('Baseline Performance')
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def export_results(self, result: StrategyRobustnessResult,
                      output_path: str):
        """分析結果を JSON ファイルにエクスポート"""
        export_data = {
            'baseline_performance': result.baseline_performance,
            'parameter_ranges': {k: list(v) for k, v in result.parameter_ranges.items()},
            'robustness_analysis': {
                'stability_score': result.robustness_analysis.stability_score,
                'robustness_score': result.robustness_analysis.robustness_score,
                'optimal_parameters': result.robustness_analysis.optimal_parameters,
                'parameter_sensitivities': [
                    {
                        'parameter_name': s.parameter_name,
                        'parameter_range': list(s.parameter_range),
                        'sensitivity_score': s.sensitivity_score,
                        'optimal_value': s.optimal_value,
                        'confidence_interval': list(s.confidence_interval)
                    }
                    for s in result.robustness_analysis.parameter_sensitivities
                ]
            }
        }

        if result.monte_carlo_stability:
            export_data['monte_carlo_stability'] = result.monte_carlo_stability

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"Results exported to {output_path}")

def main():
    """コマンドラインインターフェース"""
    parser = argparse.ArgumentParser(description='Strategy Robustness Analysis')
    parser.add_argument('--data-path', required=True, help='Path to market data (CSV)')
    parser.add_argument('--strategy-module', required=True, help='Strategy module name')
    parser.add_argument('--strategy-function', required=True, help='Strategy function name')
    parser.add_argument('--parameters', required=True, help='Base parameters (JSON string)')
    parser.add_argument('--parameter-ranges', required=True, help='Parameter ranges (JSON string)')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--metric', default='sharpe_ratio', help='Evaluation metric')
    parser.add_argument('--no-monte-carlo', action='store_true', help='Skip Monte Carlo test')

    args = parser.parse_args()

    # データの読み込み
    data = pd.read_csv(args.data_path)
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')

    # 戦略関数の読み込み
    import importlib
    strategy_module = importlib.import_module(args.strategy_module)
    strategy_func = getattr(strategy_module, args.strategy_function)

    # パラメータの読み込み
    import json
    base_parameters = json.loads(args.parameters)
    parameter_ranges = {k: tuple(v) for k, v in json.loads(args.parameter_ranges).items()}

    # 分析の実行
    analyzer = StrategyRobustnessAnalyzer()
    result = analyzer.run_comprehensive_robustness_analysis(
        strategy_func=strategy_func,
        base_parameters=base_parameters,
        parameter_ranges=parameter_ranges,
        data=data,
        metric=args.metric,
        include_monte_carlo=not args.no_monte_carlo
    )

    # 結果の保存
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # プロットの保存
    plot_path = os.path.join(args.output_dir, 'strategy_robustness_analysis.png')
    analyzer.plot_robustness_analysis(result, save_path=plot_path)

    # JSON エクスポート
    json_path = os.path.join(args.output_dir, 'strategy_robustness_results.json')
    analyzer.export_results(result, json_path)

    print("Strategy robustness analysis completed!")

if __name__ == '__main__':
    main()