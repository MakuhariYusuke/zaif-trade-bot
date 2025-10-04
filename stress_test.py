#!/usr/bin/env python3
"""
ストレステスト分析モジュール

極端な市場条件での戦略パフォーマンスを評価します。
ブラックスワンイベント、市場クラッシュ、急激なボラティリティ上昇などの
ストレスシナリオに対する耐性を分析します。
"""

import argparse
import json
from typing import Dict, List, Optional, Callable
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy import stats

# Type aliases for better type safety
SeriesFloat = pd.Series[float]

@dataclass
class StressTestScenario:
    """ストレステストシナリオ"""
    name: str
    description: str
    shock_function: Callable[[pd.Series], pd.Series]
    severity_levels: List[float]

@dataclass
class StressTestResult:
    """ストレステスト結果"""
    scenario_name: str
    severity: float
    original_returns: pd.Series
    stressed_returns: pd.Series
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    survival_probability: float

@dataclass
class StressTestAnalysisResult:
    """ストレステスト分析全体結果"""
    results: List[StressTestResult]
    scenarios: List[StressTestScenario]
    baseline_metrics: Dict[str, float]

class StressTestAnalyzer:
    """ストレステスト分析クラス"""

    def __init__(self, confidence_level: float = 0.95):
        """
        Args:
            confidence_level: 信頼区間レベル
        """
        self.confidence_level = confidence_level

    def _create_market_crash_scenario(self) -> StressTestScenario:
        """市場クラッシュシナリオ"""
        def crash_shock(returns: SeriesFloat, severity: float = 1.0) -> SeriesFloat:
            # 突然の大きな下落
            crash_point = len(returns) // 2
            shock = np.zeros(len(returns))
            shock[crash_point] = -severity * 0.3  # 30%下落
            # 回復期
            recovery_period = min(20, len(returns) - crash_point)
            for i in range(recovery_period):
                shock[crash_point + i] += severity * 0.01 * (i / recovery_period)
            return returns + shock

        return StressTestScenario(
            name="Market Crash",
            description="Sudden market crash with recovery period",
            shock_function=crash_shock,
            severity_levels=[0.5, 1.0, 1.5, 2.0]
        )

    def _create_volatility_spike_scenario(self) -> StressTestScenario:
        """ボラティリティ急上昇シナリオ"""
        def volatility_shock(returns: SeriesFloat, severity: float = 1.0) -> SeriesFloat:
            # ボラティリティを severity 倍に増加
            stressed_returns: SeriesFloat = returns.copy()
            # 中央部でボラティリティを増加
            mid_point = len(returns) // 2
            window_size = min(50, len(returns) // 4)
            start_idx = max(0, mid_point - window_size // 2)
            end_idx = min(len(returns), mid_point + window_size // 2)

            # 窓内のリターンをスケーリング
            window_returns = returns.iloc[start_idx:end_idx]
            mean_return = window_returns.mean()
            scaled_returns = mean_return + severity * (window_returns - mean_return)

            stressed_returns.iloc[start_idx:end_idx] = scaled_returns
            return stressed_returns

        return StressTestScenario(
            name="Volatility Spike",
            description="Sudden increase in market volatility",
            shock_function=volatility_shock,
            severity_levels=[1.5, 2.0, 3.0, 5.0]
        )

    def _create_liquidity_crisis_scenario(self) -> StressTestScenario:
        """流動性危機シナリオ"""
        def liquidity_shock(returns: SeriesFloat, severity: float = 1.0) -> SeriesFloat:
            # 取引コストの急上昇（スプレッド拡大）
            stressed_returns: SeriesFloat = returns.copy()
            # コストを severity 倍に
            transaction_costs = severity * 0.002  # 0.2% の基本コスト

            # トレードをシミュレート（ランダム）
            np.random.seed(42)
            trade_signals = np.random.choice([-1, 0, 1], size=len(returns), p=[0.3, 0.4, 0.3])

            # コストを適用
            cost_impact = transaction_costs * np.abs(np.diff(trade_signals, prepend=0))
            stressed_returns = stressed_returns - cost_impact

            return stressed_returns

        return StressTestScenario(
            name="Liquidity Crisis",
            description="Deterioration of market liquidity with higher transaction costs",
            shock_function=liquidity_shock,
            severity_levels=[1.0, 2.0, 5.0, 10.0]
        )

    def _create_black_swan_scenario(self) -> StressTestScenario:
        """ブラックスワンイベントシナリオ"""
        def black_swan_shock(returns: SeriesFloat, severity: float = 1.0) -> SeriesFloat:
            # 極端なイベント（正規分布の5シグマイベント）
            stressed_returns: SeriesFloat = returns.copy()

            # ランダムに数日間の極端な動きを挿入
            n_events = max(1, int(severity * 2))
            event_indices = np.random.choice(len(returns), size=n_events, replace=False)

            for idx in event_indices:
                # 5-10シグマのイベント
                shock_magnitude = severity * np.random.choice([-1, 1]) * 5 * returns.std()
                stressed_returns.iloc[idx] += shock_magnitude

            return stressed_returns

        return StressTestScenario(
            name="Black Swan",
            description="Extreme rare events (5+ sigma movements)",
            shock_function=black_swan_shock,
            severity_levels=[1.0, 2.0, 3.0, 5.0]
        )

    def _create_correlation_breakdown_scenario(self) -> StressTestScenario:
        """相関崩壊シナリオ"""
        def correlation_shock(returns: SeriesFloat, severity: float = 1.0) -> SeriesFloat:
            # 相関性の変化（通常の市場相関が崩れる）
            stressed_returns: SeriesFloat = returns.copy()

            # 中央部で相関を逆転
            mid_point = len(returns) // 2
            window_size = min(30, len(returns) // 3)
            start_idx = max(0, mid_point - window_size // 2)
            end_idx = min(len(returns), mid_point + window_size // 2)

            # 窓内のリターンを逆相関に
            window_returns = returns.iloc[start_idx:end_idx]
            stressed_returns.iloc[start_idx:end_idx] = -severity * window_returns

            return stressed_returns

        return StressTestScenario(
            name="Correlation Breakdown",
            description="Breakdown of normal market correlations",
            shock_function=correlation_shock,
            severity_levels=[0.5, 1.0, 1.5, 2.0]
        )

    def get_default_scenarios(self) -> List[StressTestScenario]:
        """デフォルトのストレステストシナリオを取得"""
        return [
            self._create_market_crash_scenario(),
            self._create_volatility_spike_scenario(),
            self._create_liquidity_crisis_scenario(),
            self._create_black_swan_scenario(),
            self._create_correlation_breakdown_scenario()
        ]

    def calculate_performance_metrics(self, returns: SeriesFloat) -> Dict[str, float]:
        """パフォーマンス指標を計算"""
        if len(returns) == 0:
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }

        # 基本指標
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)

        # Sharpe Ratio
        if volatility > 0:
            sharpe_ratio = annual_return / volatility
        else:
            sharpe_ratio = 0.0

        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win Rate (正のリターンの割合)
        win_rate = (returns > 0).mean()

        return {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate)
        }

    def calculate_risk_metrics(self, returns: SeriesFloat) -> Dict[str, float]:
        """リスク指標を計算"""
        if len(returns) == 0:
            return {
                'var_95': 0.0,
                'cvar_95': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'tail_ratio': 0.0
            }

        # VaR (95%)
        var_95 = np.percentile(returns, 5)

        # CVaR (95%)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

        # 歪度と尖度
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)

        # Tail Ratio (95%分位点 / 5%分位点の絶対値)
        tail_ratio = abs(np.percentile(returns, 95) / np.percentile(returns, 5))

        return {
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'tail_ratio': float(tail_ratio)
        }

    def calculate_survival_probability(self, returns: SeriesFloat,
                                     capital: float = 10000,
                                     max_drawdown_limit: float = 0.5) -> float:
        """戦略の生存確率を計算"""
        if len(returns) == 0:
            return 1.0

        # 資本推移をシミュレート
        capital_series = capital * (1 + returns).cumprod()

        # 最大ドローダウン制限を超えたかチェック
        peak = capital
        max_drawdown = 0.0

        for cap in capital_series:
            if cap > peak:
                peak = cap
            current_dd = (peak - cap) / peak
            max_drawdown = max(max_drawdown, current_dd)

            if max_drawdown > max_drawdown_limit:
                return 0.0  # 破綻

        return 1.0  # 生存

    def run_stress_test(self, returns: SeriesFloat,
                       scenarios: Optional[List[StressTestScenario]] = None) -> StressTestAnalysisResult:
        """
        ストレステストを実行

        Args:
            returns: 元のリターン系列
            scenarios: テストシナリオ（Noneの場合はデフォルトを使用）

        Returns:
            ストレステスト分析結果
        """
        if scenarios is None:
            scenarios = self.get_default_scenarios()

        # ベースライン指標の計算
        baseline_metrics = self.calculate_performance_metrics(returns)
        baseline_metrics.update(self.calculate_risk_metrics(returns))

        results = []

        for scenario in scenarios:
            for severity in scenario.severity_levels:
                try:
                    # ストレス適用
                    stressed_returns = scenario.shock_function(returns, severity)

                    # 指標計算
                    performance_metrics = self.calculate_performance_metrics(stressed_returns)
                    risk_metrics = self.calculate_risk_metrics(stressed_returns)

                    # 生存確率
                    survival_prob = self.calculate_survival_probability(stressed_returns)

                    result = StressTestResult(
                        scenario_name=scenario.name,
                        severity=severity,
                        original_returns=returns,
                        stressed_returns=stressed_returns,
                        performance_metrics=performance_metrics,
                        risk_metrics=risk_metrics,
                        survival_probability=survival_prob
                    )

                    results.append(result)

                except Exception as e:
                    warnings.warn(f"Failed to run scenario {scenario.name} at severity {severity}: {e}")
                    continue

        return StressTestAnalysisResult(
            results=results,
            scenarios=scenarios,
            baseline_metrics=baseline_metrics
        )

    def plot_stress_test_results(self, result: StressTestAnalysisResult,
                                save_path: Optional[str] = None):
        """ストレステスト結果を可視化"""
        n_scenarios = len(set(r.scenario_name for r in result.results))
        n_cols = min(3, n_scenarios)
        n_rows = int(np.ceil(n_scenarios / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        fig.suptitle('Stress Test Results', fontsize=16)

        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # シナリオごとにグループ化
        scenario_groups = {}
        for r in result.results:
            if r.scenario_name not in scenario_groups:
                scenario_groups[r.scenario_name] = []
            scenario_groups[r.scenario_name].append(r)

        for i, (scenario_name, scenario_results) in enumerate(scenario_groups.items()):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            # Sharpe Ratioの変化をプロット
            severities = [r.severity for r in scenario_results]
            sharpe_ratios = [r.performance_metrics['sharpe_ratio'] for r in scenario_results]
            baseline_sharpe = result.baseline_metrics['sharpe_ratio']

            ax.plot(severities, sharpe_ratios, 'ro-', linewidth=2, markersize=8, label='Stressed')
            ax.axhline(y=baseline_sharpe, color='blue', linestyle='--', linewidth=2, label='Baseline')

            ax.set_xlabel('Severity Level')
            ax.set_ylabel('Sharpe Ratio')
            ax.set_title(f'{scenario_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 余ったサブプロットを削除
        for i in range(n_scenarios, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def export_results(self, result: StressTestAnalysisResult,
                      output_path: str):
        """分析結果を JSON ファイルにエクスポート"""
        export_data = {
            'baseline_metrics': result.baseline_metrics,
            'scenarios': []
        }

        # シナリオごとにグループ化
        scenario_groups = {}
        for r in result.results:
            if r.scenario_name not in scenario_groups:
                scenario_groups[r.scenario_name] = []
            scenario_groups[r.scenario_name].append(r)

        for scenario_name, scenario_results in scenario_groups.items():
            scenario_data = {
                'name': scenario_name,
                'results': []
            }

            for r in scenario_results:
                scenario_data['results'].append({
                    'severity': r.severity,
                    'performance_metrics': r.performance_metrics,
                    'risk_metrics': r.risk_metrics,
                    'survival_probability': r.survival_probability
                })

            export_data['scenarios'].append(scenario_data)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"Results exported to {output_path}")

def main():
    """コマンドラインインターフェース"""
    parser = argparse.ArgumentParser(description='Stress Test Analysis')
    parser.add_argument('--returns-path', required=True, help='Path to returns data (CSV)')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--confidence-level', type=float, default=0.95, help='Confidence level')

    args = parser.parse_args()

    # リターンデータの読み込み
    returns_df = pd.read_csv(args.returns_path)
    if 'returns' in returns_df.columns:
        returns = returns_df['returns']
    elif len(returns_df.columns) == 1:
        returns = returns_df.iloc[:, 0]
    else:
        raise ValueError("Could not identify returns column")

    # 分析の実行
    analyzer = StressTestAnalyzer(confidence_level=args.confidence_level)
    result = analyzer.run_stress_test(returns)

    # 結果の保存
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # プロットの保存
    plot_path = os.path.join(args.output_dir, 'stress_test_analysis.png')
    analyzer.plot_stress_test_results(result, save_path=plot_path)

    # JSON エクスポート
    json_path = os.path.join(args.output_dir, 'stress_test_results.json')
    analyzer.export_results(result, json_path)

    print("Stress test analysis completed!")

if __name__ == '__main__':
    main()