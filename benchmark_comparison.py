"""
ベンチマーク比較分析モジュール

戦略のパフォーマンスを市場ベンチマークと比較し、相対的な優位性を評価します。
"""

import argparse
import json
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy import stats

@dataclass
class BenchmarkComparison:
    """ベンチマーク比較結果"""
    benchmark_name: str
    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    excess_returns: pd.Series
    tracking_error: float
    information_ratio: float
    beta: float
    alpha: float
    r_squared: float
    max_drawdown_diff: float
    win_rate_vs_benchmark: float

@dataclass
class RollingComparison:
    """ローリング比較結果"""
    window_size: int
    rolling_alpha: pd.Series
    rolling_beta: pd.Series
    rolling_tracking_error: pd.Series
    rolling_excess_returns: pd.Series

@dataclass
class BenchmarkComparisonResult:
    """包括的なベンチマーク比較結果"""
    strategy_performance: Dict[str, float]
    benchmark_performance: Dict[str, float]
    comparisons: List[BenchmarkComparison]
    rolling_comparisons: Optional[List[RollingComparison]] = None
    multi_benchmark_summary: Optional[Dict[str, Any]] = None

class BenchmarkComparisonAnalyzer:
    """ベンチマーク比較分析クラス"""

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Args:
            risk_free_rate: 無リスク金利（年率）
        """
        self.risk_free_rate = risk_free_rate

    def calculate_performance_metrics(self, returns: pd.Series,
                                    annualize: bool = True) -> Dict[str, float]:
        """
        パフォーマンス指標の計算

        Args:
            returns: リターン系列
            annualize: 年率化するか

        Returns:
            パフォーマンス指標
        """
        if len(returns) == 0:
            return {
                'total_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }

        # 基本指標
        cumprod_result = (1 + returns).prod()  # type: ignore
        total_return = float(cumprod_result - 1)  # type: ignore
        volatility = float(returns.std())

        if annualize:
            # 年率化
            periods_per_year = 252  # 取引日数
            volatility = volatility * np.sqrt(periods_per_year)
            total_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1

        sharpe_ratio = float((total_return - self.risk_free_rate) / volatility) if volatility > 0 else 0.0
        max_drawdown = float((returns.cumsum() - returns.cumsum().cummax()).min())
        win_rate = float((returns > 0).mean())

        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }

    def calculate_regression_metrics(self, strategy_returns: pd.Series,
                                   benchmark_returns: pd.Series) -> Tuple[float, float, float, float]:
        """
        回帰分析による指標計算

        Args:
            strategy_returns: 戦略リターン
            benchmark_returns: ベンチマークリターン

        Returns:
            (alpha, beta, r_squared, tracking_error)
        """
        # データの整合性チェック
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_index) < 2:
            return 0.0, 1.0, 0.0, float(strategy_returns.std())

        strat_ret = strategy_returns.loc[common_index]
        bench_ret = benchmark_returns.loc[common_index]

        # OLS回帰
        try:
            linreg_result = stats.linregress(bench_ret, strat_ret)
            slope = linreg_result.slope  # type: ignore
            intercept = linreg_result.intercept  # type: ignore
            r_value = linreg_result.rvalue  # type: ignore
            beta = float(slope)
            alpha = float(intercept)
            r_squared = float(r_value ** 2)

            # トラッキングエラー
            predicted_returns = intercept + slope * bench_ret
            tracking_error = float(np.std(strat_ret - predicted_returns))

        except Exception as e:
            warnings.warn(f"Regression analysis failed: {e}")
            beta = 1.0
            alpha = 0.0
            r_squared = 0.0
            tracking_error = float(strat_ret.std())

        return alpha, beta, r_squared, tracking_error

    def compare_with_benchmark(self, strategy_returns: pd.Series,
                             benchmark_returns: pd.Series,
                             benchmark_name: str = "Benchmark") -> BenchmarkComparison:
        """
        戦略とベンチマークの比較

        Args:
            strategy_returns: 戦略のリターン系列
            benchmark_returns: ベンチマークのリターン系列
            benchmark_name: ベンチマーク名

        Returns:
            比較結果
        """
        # データの整合性チェック
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_index) == 0:
            raise ValueError("No overlapping data between strategy and benchmark")

        strat_ret = strategy_returns.loc[common_index]
        bench_ret = benchmark_returns.loc[common_index]

        # 超過リターン
        excess_returns = strat_ret - bench_ret

        # 回帰指標
        alpha, beta, r_squared, tracking_error = self.calculate_regression_metrics(strat_ret, bench_ret)

        # 情報比率
        excess_return_mean = excess_returns.mean()
        information_ratio = float(excess_return_mean / tracking_error) if tracking_error > 0 else 0.0

        # 最大ドローダウンの差
        strat_dd = (strat_ret.cumsum() - strat_ret.cumsum().cummax()).min()
        bench_dd = (bench_ret.cumsum() - bench_ret.cumsum().cummax()).min()
        max_drawdown_diff = float(strat_dd - bench_dd)

        # ベンチマークに対する勝率
        win_rate_vs_benchmark = float((strat_ret > bench_ret).mean())

        return BenchmarkComparison(
            benchmark_name=benchmark_name,
            strategy_returns=strat_ret,
            benchmark_returns=bench_ret,
            excess_returns=excess_returns,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            beta=beta,
            alpha=alpha,
            r_squared=r_squared,
            max_drawdown_diff=max_drawdown_diff,
            win_rate_vs_benchmark=win_rate_vs_benchmark
        )

    def rolling_comparison_analysis(self, strategy_returns: pd.Series,
                                  benchmark_returns: pd.Series,
                                  window_sizes: Optional[List[int]] = None) -> List[RollingComparison]:
        """
        ローリング比較分析

        Args:
            strategy_returns: 戦略のリターン系列
            benchmark_returns: ベンチマークのリターン系列
            window_sizes: ローリングウィンドウサイズ（日数）

        Returns:
            ローリング比較結果リスト
        """
        rolling_comparisons = []

        for window_size in window_sizes:
            if len(strategy_returns) < window_size or len(benchmark_returns) < window_size:
                warnings.warn(f"Insufficient data for window size {window_size}")
                continue

            # ローリング回帰
            rolling_alpha = []
            rolling_beta = []
            rolling_tracking_error = []
            rolling_excess_returns = []

            for i in range(window_size, len(strategy_returns) + 1):
                strat_window = strategy_returns.iloc[i-window_size:i]
                bench_window = benchmark_returns.iloc[i-window_size:i]

                # 共通インデックスを取得
                common_idx = strat_window.index.intersection(bench_window.index)
                if len(common_idx) < 10:  # 最小サンプルサイズ
                    rolling_alpha.append(np.nan)
                    rolling_beta.append(np.nan)
                    rolling_tracking_error.append(np.nan)
                    rolling_excess_returns.append(np.nan)
                    continue

                strat_win = strat_window.loc[common_idx]
                bench_win = bench_window.loc[common_idx]

                try:
                    alpha, beta, _, tracking_error = self.calculate_regression_metrics(strat_win, bench_win)
                    excess_return = (strat_win - bench_win).mean()

                    rolling_alpha.append(alpha)
                    rolling_beta.append(beta)
                    rolling_tracking_error.append(tracking_error)
                    rolling_excess_returns.append(excess_return)

                except Exception:
                    rolling_alpha.append(np.nan)
                    rolling_beta.append(np.nan)
                    rolling_tracking_error.append(np.nan)
                    rolling_excess_returns.append(np.nan)

            # Seriesに変換
            dates = strategy_returns.index[window_size-1:]
            rolling_comp = RollingComparison(
                window_size=window_size,
                rolling_alpha=pd.Series(rolling_alpha, index=dates),
                rolling_beta=pd.Series(rolling_beta, index=dates),
                rolling_tracking_error=pd.Series(rolling_tracking_error, index=dates),
                rolling_excess_returns=pd.Series(rolling_excess_returns, index=dates)
            )
            rolling_comparisons.append(rolling_comp)

        return rolling_comparisons

    def multi_benchmark_comparison(self, strategy_returns: pd.Series,
                                 benchmark_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        複数ベンチマークとの比較

        Args:
            strategy_returns: 戦略のリターン系列
            benchmark_data: ベンチマーク名とリターン系列の辞書

        Returns:
            複数ベンチマーク比較サマリー
        """
        comparisons = []
        summary_stats = {
            'best_benchmark': None,
            'worst_benchmark': None,
            'avg_information_ratio': 0.0,
            'avg_alpha': 0.0,
            'benchmark_correlations': {}
        }

        information_ratios = []
        alphas = []

        for name, bench_returns in benchmark_data.items():
            try:
                comparison = self.compare_with_benchmark(strategy_returns, bench_returns, name)
                comparisons.append(comparison)

                information_ratios.append(comparison.information_ratio)
                alphas.append(comparison.alpha)

                # ベンチマーク間の相関
                for other_name, other_bench in benchmark_data.items():
                    if name != other_name:
                        corr_key = f"{name}_vs_{other_name}"
                        if corr_key not in summary_stats['benchmark_correlations']:
                            try:
                                corr = bench_returns.corr(other_bench)
                                summary_stats['benchmark_correlations'][corr_key] = float(corr)
                            except:
                                pass

            except Exception as e:
                warnings.warn(f"Comparison with {name} failed: {e}")
                continue

        if information_ratios:
            summary_stats['avg_information_ratio'] = float(np.mean(information_ratios))
            summary_stats['avg_alpha'] = float(np.mean(alphas))

            # 最適/最悪ベンチマーク
            best_idx = np.argmax(information_ratios)
            worst_idx = np.argmin(information_ratios)

            summary_stats['best_benchmark'] = comparisons[best_idx].benchmark_name
            summary_stats['worst_benchmark'] = comparisons[worst_idx].benchmark_name

        return {
            'comparisons': comparisons,
            'summary_stats': summary_stats
        }

    def run_comprehensive_benchmark_analysis(self, strategy_returns: pd.Series,
                                           benchmark_data: Dict[str, pd.Series],
                                           include_rolling: bool = True) -> BenchmarkComparisonResult:
        """
        包括的なベンチマーク比較分析を実行

        Args:
            strategy_returns: 戦略のリターン系列
            benchmark_data: ベンチマーク名とリターン系列の辞書
            include_rolling: ローリング分析を含むか

        Returns:
            包括的な比較結果
        """
        # 戦略のパフォーマンス
        strategy_performance = self.calculate_performance_metrics(strategy_returns)

        # ベンチマークのパフォーマンス
        benchmark_performance = {}
        for name, returns in benchmark_data.items():
            benchmark_performance[name] = self.calculate_performance_metrics(returns)

        # 個別比較
        comparisons = []
        for name, returns in benchmark_data.items():
            try:
                comparison = self.compare_with_benchmark(strategy_returns, returns, name)
                comparisons.append(comparison)
            except Exception as e:
                warnings.warn(f"Benchmark comparison with {name} failed: {e}")

        # ローリング比較
        rolling_comparisons = None
        if include_rolling and comparisons:
            # 最初のベンチマークでローリング分析
            first_comparison = comparisons[0]
            rolling_comparisons = self.rolling_comparison_analysis(
                first_comparison.strategy_returns,
                first_comparison.benchmark_returns
            )

        # 複数ベンチマーク比較
        multi_benchmark_summary = self.multi_benchmark_comparison(strategy_returns, benchmark_data)

        return BenchmarkComparisonResult(
            strategy_performance=strategy_performance,
            benchmark_performance=benchmark_performance,
            comparisons=comparisons,
            rolling_comparisons=rolling_comparisons,
            multi_benchmark_summary=multi_benchmark_summary
        )

    def analyze(self, metrics: Any) -> Dict[str, Any]:
        """
        BenchmarkMetricsからベンチマーク比較分析を実行

        Args:
            metrics: 評価メトリクス

        Returns:
            分析結果の辞書
        """
        # returnsデータをpandas Seriesに変換
        if not hasattr(metrics, 'returns') or not metrics.returns:
            return {
                'error': 'No returns data available for analysis',
                'benchmark_score': 0.0,
                'alpha': 0.0,
                'beta': 0.0,
                'information_ratio': 0.0
            }

        # 日付インデックスを作成（仮定）
        dates = pd.date_range(start='2023-01-01', periods=len(metrics.returns), freq='D')
        strategy_returns = pd.Series(metrics.returns, index=dates)

        # ベンチマークリターンを生成（市場平均として）
        benchmark_returns = strategy_returns.rolling(20).mean().fillna(0)

        try:
            # ベンチマーク比較分析を実行
            result = self.run_comprehensive_benchmark_analysis(
                strategy_returns=strategy_returns,
                benchmark_data={'market_average': benchmark_returns}
            )

            # 結果を辞書形式に変換
            analysis_result: Dict[str, Any] = {
                'benchmark_score': result.comparisons[0].information_ratio if result.comparisons else 0.0,
                'alpha': result.comparisons[0].alpha if result.comparisons else 0.0,
                'beta': result.comparisons[0].beta if result.comparisons else 0.0,
                'information_ratio': result.comparisons[0].information_ratio if result.comparisons else 0.0,
                'tracking_error': result.comparisons[0].tracking_error if result.comparisons else 0.0,
                'r_squared': result.comparisons[0].r_squared if result.comparisons else 0.0,
                'strategy_performance': result.strategy_performance,
                'benchmark_performance': result.benchmark_performance,
                'rolling_alpha': [],
                'rolling_beta': [],
                'rolling_tracking_error': []
            }

            # ローリング比較結果を追加
            if result.rolling_comparisons:
                for rolling_comp in result.rolling_comparisons:
                    analysis_result['rolling_alpha'].extend(rolling_comp.rolling_alpha.tolist())
                    analysis_result['rolling_beta'].extend(rolling_comp.rolling_beta.tolist())
                    analysis_result['rolling_tracking_error'].extend(rolling_comp.rolling_tracking_error.tolist())

            # 複数ベンチマークサマリーを追加
            if result.multi_benchmark_summary:
                analysis_result['multi_benchmark_summary'] = result.multi_benchmark_summary

            return analysis_result

        except Exception as e:
            return {
                'error': f'Analysis failed: {str(e)}',
                'benchmark_score': 0.0,
                'alpha': 0.0,
                'beta': 0.0,
                'information_ratio': 0.0
            }

    def plot_benchmark_comparison(self, result: BenchmarkComparisonResult,
                                save_path: Optional[str] = None):
        """ベンチマーク比較結果を可視化"""
        if not result.comparisons:
            print("No benchmark comparisons available for plotting")
            return

        n_comparisons = len(result.comparisons)
        n_cols = min(3, n_comparisons)
        n_rows = (n_comparisons + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        fig.suptitle('Benchmark Comparison Analysis', fontsize=16)

        for i, comparison in enumerate(result.comparisons):
            if i >= len(axes):
                break

            ax = axes[i]

            # 累積リターンの比較
            cum_strat = (1 + comparison.strategy_returns).cumprod()
            cum_bench = (1 + comparison.benchmark_returns).cumprod()

            ax.plot(cum_strat.index, cum_strat, label='Strategy', linewidth=2)
            ax.plot(cum_bench.index, cum_bench, label=comparison.benchmark_name, linewidth=2, alpha=0.7)

            ax.set_title(f'Cumulative Returns vs {comparison.benchmark_name}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Return')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 主要指標をテキストで表示
            info_text = f"""Alpha: {comparison.alpha:.4f}
Beta: {comparison.beta:.4f}
IR: {comparison.information_ratio:.4f}
R²: {comparison.r_squared:.4f}"""
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 余ったサブプロットを非表示
        for i in range(len(result.comparisons), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_rolling_comparison(self, rolling_comparisons: List[RollingComparison],
                              save_path: Optional[str] = None):
        """ローリング比較結果を可視化"""
        if not rolling_comparisons:
            print("No rolling comparisons available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Rolling Benchmark Comparison Analysis', fontsize=16)

        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(rolling_comparisons)))  # type: ignore

        # Alpha
        ax = axes[0, 0]
        for i, rolling in enumerate(rolling_comparisons):
            valid_data = rolling.rolling_alpha.dropna()
            if len(valid_data) > 0:
                ax.plot(valid_data.index, valid_data,
                       label=f'Window: {rolling.window_size}d',
                       color=colors[i], linewidth=1.5)
        ax.set_title('Rolling Alpha')
        ax.set_ylabel('Alpha')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Beta
        ax = axes[0, 1]
        for i, rolling in enumerate(rolling_comparisons):
            valid_data = rolling.rolling_beta.dropna()
            if len(valid_data) > 0:
                ax.plot(valid_data.index, valid_data,
                       label=f'Window: {rolling.window_size}d',
                       color=colors[i], linewidth=1.5)
        ax.set_title('Rolling Beta')
        ax.set_ylabel('Beta')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Tracking Error
        ax = axes[1, 0]
        for i, rolling in enumerate(rolling_comparisons):
            valid_data = rolling.rolling_tracking_error.dropna()
            if len(valid_data) > 0:
                ax.plot(valid_data.index, valid_data,
                       label=f'Window: {rolling.window_size}d',
                       color=colors[i], linewidth=1.5)
        ax.set_title('Rolling Tracking Error')
        ax.set_ylabel('Tracking Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Excess Returns
        ax = axes[1, 1]
        for i, rolling in enumerate(rolling_comparisons):
            valid_data = rolling.rolling_excess_returns.dropna()
            if len(valid_data) > 0:
                ax.plot(valid_data.index, valid_data,
                       label=f'Window: {rolling.window_size}d',
                       color=colors[i], linewidth=1.5)
        ax.set_title('Rolling Excess Returns')
        ax.set_ylabel('Excess Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Rolling plot saved to {save_path}")

        plt.show()

    def export_results(self, result: BenchmarkComparisonResult,
                      output_path: str):
        """分析結果を JSON ファイルにエクスポート"""
        export_data = {
            'strategy_performance': result.strategy_performance,
            'benchmark_performance': result.benchmark_performance,
            'comparisons': [
                {
                    'benchmark_name': comp.benchmark_name,
                    'tracking_error': comp.tracking_error,
                    'information_ratio': comp.information_ratio,
                    'beta': comp.beta,
                    'alpha': comp.alpha,
                    'r_squared': comp.r_squared,
                    'max_drawdown_diff': comp.max_drawdown_diff,
                    'win_rate_vs_benchmark': comp.win_rate_vs_benchmark
                }
                for comp in result.comparisons
            ]
        }

        if result.multi_benchmark_summary:
            export_data['multi_benchmark_summary'] = result.multi_benchmark_summary['summary_stats']

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"Results exported to {output_path}")

def main():
    """コマンドラインインターフェース"""
    parser = argparse.ArgumentParser(description='Benchmark Comparison Analysis')
    parser.add_argument('--strategy-data', required=True, help='Path to strategy returns (CSV)')
    parser.add_argument('--benchmark-data', required=True, help='Path to benchmark data (JSON with benchmark names and CSV paths)')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--risk-free-rate', type=float, default=0.02, help='Risk-free rate')
    parser.add_argument('--no-rolling', action='store_true', help='Skip rolling analysis')

    args = parser.parse_args()

    # 戦略データの読み込み
    strategy_data = pd.read_csv(args.strategy_data)
    if 'date' in strategy_data.columns:
        strategy_data['date'] = pd.to_datetime(strategy_data['date'])
        strategy_data = strategy_data.set_index('date')

    # 戦略リターンを取得（最初の数値列を使用）
    numeric_cols = strategy_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in strategy data")
    strategy_returns = strategy_data[numeric_cols[0]]

    # ベンチマークデータの読み込み
    with open(args.benchmark_data, 'r') as f:
        benchmark_config = json.load(f)

    benchmark_data = {}
    for name, path in benchmark_config.items():
        bench_df = pd.read_csv(path)
        if 'date' in bench_df.columns:
            bench_df['date'] = pd.to_datetime(bench_df['date'])
            bench_df = bench_df.set_index('date')

        numeric_cols = bench_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            benchmark_data[name] = bench_df[numeric_cols[0]]

    # 分析の実行
    analyzer = BenchmarkComparisonAnalyzer(risk_free_rate=args.risk_free_rate)
    result = analyzer.run_comprehensive_benchmark_analysis(
        strategy_returns=strategy_returns,
        benchmark_data=benchmark_data,
        include_rolling=not args.no_rolling
    )

    # 結果の保存
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # 比較プロットの保存
    comparison_plot_path = os.path.join(args.output_dir, 'benchmark_comparison.png')
    analyzer.plot_benchmark_comparison(result, save_path=comparison_plot_path)

    # ローリングプロットの保存
    if result.rolling_comparisons:
        rolling_plot_path = os.path.join(args.output_dir, 'rolling_benchmark_comparison.png')
        analyzer.plot_rolling_comparison(result.rolling_comparisons, save_path=rolling_plot_path)

    # JSON エクスポート
    json_path = os.path.join(args.output_dir, 'benchmark_comparison_results.json')
    analyzer.export_results(result, json_path)

    print("Benchmark comparison analysis completed!")

if __name__ == '__main__':
    main()