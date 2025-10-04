#!/usr/bin/env python3
"""
ブートストラップ信頼区間分析モジュール

ブートストラップ法を使用して、戦略のパフォーマンス指標に対する
統計的な信頼区間を計算します。
"""

import argparse
import json
from typing import Dict, List, Optional, Tuple, Callable
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy import stats
from sklearn.utils import resample
from numpy.typing import NDArray

@dataclass
class BootstrapConfidenceResult:
    """ブートストラップ信頼区間分析結果"""
    metric_name: str
    original_value: float
    bootstrap_samples: NDArray[np.floating]
    confidence_interval: Tuple[float, float]
    confidence_level: float
    standard_error: float
    bias: float
    percentile_method: bool = True

@dataclass
class BootstrapAnalysisResult:
    """ブートストラップ分析全体結果"""
    results: Dict[str, BootstrapConfidenceResult]
    n_bootstrap: int
    confidence_level: float
    random_state: int

class BootstrapConfidenceAnalyzer:
    """ブートストラップ信頼区間分析クラス"""

    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.95,
                 random_state: int = 42):
        """
        Args:
            n_bootstrap: ブートストラップサンプル数
            confidence_level: 信頼区間レベル (0-1)
            random_state: 乱数シード
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state
        np.random.seed(random_state)

    def calculate_bootstrap_confidence(self, data: NDArray[np.floating],
                                      metric_func: Callable[[NDArray[np.floating]], float],
                                      metric_name: str = "custom_metric",
                                      method: str = "percentile") -> BootstrapConfidenceResult:
        """
        単一指標に対するブートストラップ信頼区間を計算

        Args:
            data: 元データ
            metric_func: 指標計算関数
            metric_name: 指標名
            method: 信頼区間計算方法 ('percentile' or 'bca')

        Returns:
            信頼区間分析結果
        """
        # 元データの指標計算
        original_value = metric_func(data)

        # ブートストラップサンプリング
        bootstrap_samples: List[float] = []
        n_samples = len(data)

        for _ in range(self.n_bootstrap):
            # ブートストラップサンプル生成（復元抽出）
            bootstrap_data = resample(data, n_samples=n_samples, random_state=self.random_state)
            self.random_state += 1  # 各サンプルで異なるシードを使用

            # 指標計算
            try:
                sample_metric = metric_func(bootstrap_data)
                bootstrap_samples.append(sample_metric)
            except Exception as e:
                warnings.warn(f"Failed to calculate metric for bootstrap sample: {e}")
                continue

        bootstrap_samples = np.array(bootstrap_samples)

        # 信頼区間の計算
        if method == "percentile":
            # パーセンタイル法
            alpha = (1 - self.confidence_level) / 2
            lower_bound = np.percentile(bootstrap_samples, alpha * 100)
            upper_bound = np.percentile(bootstrap_samples, (1 - alpha) * 100)
        elif method == "bca":
            # BCa (Bias-Corrected and Accelerated) 法
            lower_bound, upper_bound = self._calculate_bca_interval(
                bootstrap_samples, original_value, data, metric_func
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # 標準誤差とバイアスの計算
        standard_error = np.std(bootstrap_samples)
        bias = np.mean(bootstrap_samples) - original_value

        return BootstrapConfidenceResult(
            metric_name=metric_name,
            original_value=float(original_value),
            bootstrap_samples=bootstrap_samples,
            confidence_interval=(float(lower_bound), float(upper_bound)),
            confidence_level=self.confidence_level,
            standard_error=float(standard_error),
            bias=float(bias),
            percentile_method=(method == "percentile")
        )

    def _calculate_bca_interval(self, bootstrap_samples: NDArray[np.floating],
                               original_value: float,
                               data: NDArray[np.floating],
                               metric_func: Callable[[NDArray[np.floating]], float]) -> Tuple[float, float]:
        """
        BCa (Bias-Corrected and Accelerated) 信頼区間を計算

        Args:
            bootstrap_samples: ブートストラップサンプル
            original_value: 元データの指標値
            data: 元データ
            metric_func: 指標計算関数

        Returns:
            信頼区間の下限と上限
        """
        # バイアス補正値 z0 の計算
        n_bootstrap = len(bootstrap_samples)
        n_less = np.sum(bootstrap_samples < original_value)
        p0 = n_less / n_bootstrap

        if p0 == 0:
            z0 = -np.inf
        elif p0 == 1:
            z0 = np.inf
        else:
            z0 = stats.norm.ppf(p0)

        # 加速値 a の計算（ジャックナイフ法）
        n_data = len(data)
        jackknife_samples = []

        for i in range(n_data):
            jackknife_data = np.delete(data, i)
            try:
                jackknife_metric = metric_func(jackknife_data)
                jackknife_samples.append(jackknife_metric)
            except:
                continue

        jackknife_samples = np.array(jackknife_samples)

        if len(jackknife_samples) < 2:
            # ジャックナイフが失敗した場合、パーセンタイル法にフォールバック
            alpha = (1 - self.confidence_level) / 2
            lower_bound = np.percentile(bootstrap_samples, alpha * 100)
            upper_bound = np.percentile(bootstrap_samples, (1 - alpha) * 100)
            return lower_bound, upper_bound

        jackknife_mean = np.mean(jackknife_samples)
        jackknife_var = np.sum((jackknife_samples - jackknife_mean) ** 2) / (n_data - 1)

        if jackknife_var == 0:
            a = 0
        else:
            a = np.sum((jackknife_mean - jackknife_samples) ** 3) / \
                (6 * (jackknife_var ** 1.5) * (n_data - 1))

        # BCa 信頼区間
        alpha = (1 - self.confidence_level) / 2
        z_alpha = stats.norm.ppf(alpha)
        z_1_minus_alpha = stats.norm.ppf(1 - alpha)

        z0_term = z0 + z_alpha
        z1_term = z0 + z_1_minus_alpha

        # 加速補正
        alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
        alpha2 = stats.norm.cdf(z0 + (z0 + z_1_minus_alpha) / (1 - a * (z0 + z_1_minus_alpha)))

        lower_bound = np.percentile(bootstrap_samples, alpha1 * 100)
        upper_bound = np.percentile(bootstrap_samples, alpha2 * 100)

        return lower_bound, upper_bound

    def analyze_trading_metrics(self, returns: pd.Series,
                               trades: Optional[pd.DataFrame] = None) -> BootstrapAnalysisResult:
        """
        取引パフォーマンス指標のブートストラップ分析を実行

        Args:
            returns: リターン系列
            trades: 取引データ（オプション）

        Returns:
            ブートストラップ分析結果
        """
        results = {}

        # 基本リターン指標
        returns_array = returns.values

        # Sharpe Ratio
        def sharpe_ratio(data):
            if len(data) == 0 or np.std(data) == 0:
                return 0.0
            return np.mean(data) / np.std(data) * np.sqrt(252)  # 年率化

        results['sharpe_ratio'] = self.calculate_bootstrap_confidence(
            returns_array, sharpe_ratio, "Sharpe Ratio"
        )

        # Maximum Drawdown
        def max_drawdown(data):
            cumulative = np.cumprod(1 + data)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return np.min(drawdown)

        results['max_drawdown'] = self.calculate_bootstrap_confidence(
            returns_array, max_drawdown, "Maximum Drawdown"
        )

        # Annual Return
        def annual_return(data):
            if len(data) == 0:
                return 0.0
            total_return = np.prod(1 + data)
            return total_return ** (252 / len(data)) - 1

        results['annual_return'] = self.calculate_bootstrap_confidence(
            returns_array, annual_return, "Annual Return"
        )

        # Volatility (Annualized)
        def volatility(data):
            if len(data) == 0:
                return 0.0
            return np.std(data) * np.sqrt(252)

        results['volatility'] = self.calculate_bootstrap_confidence(
            returns_array, volatility, "Annual Volatility"
        )

        # Win Rate (取引データがある場合)
        if trades is not None and not trades.empty:
            if 'pnl' in trades.columns:
                pnl_array = trades['pnl'].values

                def win_rate(data):
                    if len(data) == 0:
                        return 0.0
                    return np.mean(data > 0)

                results['win_rate'] = self.calculate_bootstrap_confidence(
                    pnl_array, win_rate, "Win Rate"
                )

                # Profit Factor
                def profit_factor(data):
                    wins = data[data > 0]
                    losses = data[data < 0]
                    if len(losses) == 0:
                        return np.inf
                    return np.sum(wins) / abs(np.sum(losses))

                results['profit_factor'] = self.calculate_bootstrap_confidence(
                    pnl_array, profit_factor, "Profit Factor"
                )

        return BootstrapAnalysisResult(
            results=results,
            n_bootstrap=self.n_bootstrap,
            confidence_level=self.confidence_level,
            random_state=self.random_state
        )

    def plot_bootstrap_analysis(self, result: BootstrapAnalysisResult,
                               save_path: Optional[str] = None):
        """ブートストラップ分析結果を可視化"""
        n_metrics = len(result.results)
        n_cols = min(3, n_metrics)
        n_rows = int(np.ceil(n_metrics / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        fig.suptitle(f'Bootstrap Confidence Intervals ({result.confidence_level:.0%})',
                    fontsize=16)

        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i, (metric_name, metric_result) in enumerate(result.results.items()):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            # ヒストグラム
            ax.hist(metric_result.bootstrap_samples, bins=50, alpha=0.7,
                   color='skyblue', edgecolor='black', density=True)

            # 元の値
            ax.axvline(metric_result.original_value, color='red', linestyle='--',
                      linewidth=2, label=f'Original: {metric_result.original_value:.4f}')

            # 信頼区間
            ax.axvline(metric_result.confidence_interval[0], color='green', linestyle=':',
                      linewidth=2, label=f'Lower: {metric_result.confidence_interval[0]:.4f}')
            ax.axvline(metric_result.confidence_interval[1], color='green', linestyle=':',
                      linewidth=2, label=f'Upper: {metric_result.confidence_interval[1]:.4f}')

            # 信頼区間の範囲を塗りつぶし
            ax.axvspan(metric_result.confidence_interval[0], metric_result.confidence_interval[1],
                      alpha=0.2, color='green', label='Confidence Interval')

            ax.set_title(f'{metric_result.metric_name}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 余ったサブプロットを削除
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def export_results(self, result: BootstrapAnalysisResult,
                      output_path: str):
        """分析結果を JSON ファイルにエクスポート"""
        export_data = {
            'n_bootstrap': result.n_bootstrap,
            'confidence_level': result.confidence_level,
            'random_state': result.random_state,
            'metrics': {}
        }

        for metric_name, metric_result in result.results.items():
            export_data['metrics'][metric_name] = {
                'original_value': metric_result.original_value,
                'confidence_interval': list(metric_result.confidence_interval),
                'standard_error': metric_result.standard_error,
                'bias': metric_result.bias,
                'percentile_method': metric_result.percentile_method,
                'bootstrap_summary': {
                    'mean': float(np.mean(metric_result.bootstrap_samples)),
                    'std': float(np.std(metric_result.bootstrap_samples)),
                    'min': float(np.min(metric_result.bootstrap_samples)),
                    'max': float(np.max(metric_result.bootstrap_samples))
                }
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"Results exported to {output_path}")

def main():
    """コマンドラインインターフェース"""
    parser = argparse.ArgumentParser(description='Bootstrap Confidence Interval Analysis')
    parser.add_argument('--returns-path', required=True, help='Path to returns data (CSV)')
    parser.add_argument('--trades-path', help='Path to trades data (CSV, optional)')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--n-bootstrap', type=int, default=1000, help='Number of bootstrap samples')
    parser.add_argument('--confidence-level', type=float, default=0.95, help='Confidence level (0-1)')
    parser.add_argument('--random-state', type=int, default=42, help='Random state')

    args = parser.parse_args()

    # リターンデータの読み込み
    returns_df = pd.read_csv(args.returns_path)
    if 'returns' in returns_df.columns:
        returns = returns_df['returns']
    elif len(returns_df.columns) == 1:
        returns = returns_df.iloc[:, 0]
    else:
        raise ValueError("Could not identify returns column")

    # 取引データの読み込み（オプション）
    trades = None
    if args.trades_path:
        trades = pd.read_csv(args.trades_path)

    # 分析の実行
    analyzer = BootstrapConfidenceAnalyzer(
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence_level,
        random_state=args.random_state
    )

    result = analyzer.analyze_trading_metrics(returns, trades)

    # 結果の保存
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # プロットの保存
    plot_path = os.path.join(args.output_dir, 'bootstrap_confidence_analysis.png')
    analyzer.plot_bootstrap_analysis(result, save_path=plot_path)

    # JSON エクスポート
    json_path = os.path.join(args.output_dir, 'bootstrap_confidence_results.json')
    analyzer.export_results(result, json_path)

    print("Bootstrap confidence interval analysis completed!")

if __name__ == '__main__':
    main()