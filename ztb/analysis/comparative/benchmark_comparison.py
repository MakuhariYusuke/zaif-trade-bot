"""
ベンチマーク比較分析モジュール

戦略のパフォーマンスを市場ベンチマークと比較し、相対的な優位性を評価します。
"""

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ztb.analysis.common.plot_utils import save_plot
# 年間取引日数
from ztb.trading.constants import TRADING_DAYS_PER_YEAR  # = 252
from ztb.types.evaluation_types import (
    BenchmarkComparison,
    BenchmarkComparisonResult,
    BenchmarkSummaryStats,
    MultiBenchmarkSummary,
    RollingComparison,
)
from ztb.metrics.metrics import calculate_performance_metrics, sharpe_ratio
from ztb.io.data_loader import DataLoader
from ztb.io.json_io import write_json

@dataclass
class BenchmarkComparisonAnalyzer:
    """ベンチマーク比較分析クラス"""

    risk_free_rate: float = 0.02

    def calculate_regression_metrics(
        self, strategy_returns: pd.Series, benchmark_returns: pd.Series
    ) -> tuple[float, float, float, float]:
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
            slope = linreg_result.slope  # type: ignore[attr-defined]
            intercept = linreg_result.intercept  # type: ignore[attr-defined]
            r_value = linreg_result.rvalue  # type: ignore[attr-defined]
            beta = float(slope)
            alpha = float(intercept)
            r_squared = float(r_value ** 2)

            # Tracking Error
            excess_returns = strat_ret - bench_ret
            tracking_error = float(excess_returns.std())
        except Exception:
            alpha = 0.0
            beta = 1.0
            r_squared = 0.0
            tracking_error = float(strat_ret.std())

        return alpha, beta, r_squared, tracking_error

    def compare_with_benchmark(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        benchmark_name: str = "Benchmark",
    ) -> BenchmarkComparison:
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
        alpha, beta, r_squared, tracking_error = self.calculate_regression_metrics(
            strat_ret, bench_ret
        )

        # 情報比率
        excess_return_mean = excess_returns.mean()
        information_ratio = (
            float(excess_return_mean / tracking_error) if tracking_error > 0 else 0.0
        )

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
            win_rate_vs_benchmark=win_rate_vs_benchmark,
        )

    def rolling_comparison_analysis(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        window_sizes: list[int] | None = None,
    ) -> list[RollingComparison]:
        """
        ローリング比較分析

        Args:
            strategy_returns: 戦略リターン
            benchmark_returns: ベンチマークリターン
            window_sizes: 分析ウィンドウサイズ（日数）のリスト

        Returns:
            ローリング比較結果のリスト
        """
        if window_sizes is None:
            window_sizes = [30, 60, 90, 180]  # デフォルトウィンドウ

        rolling_comparisons = []

        for window in window_sizes:
            # ローリング回帰分析
            rolling_alpha = []
            rolling_beta = []
            rolling_tracking_error = []
            rolling_excess_returns = []
            dates = []

            for i in range(window, len(strategy_returns)):
                window_strat = strategy_returns.iloc[i-window:i]
                window_bench = benchmark_returns.iloc[i-window:i]

                try:
                    alpha, beta, r_squared, tracking_error = self.calculate_regression_metrics(window_strat, window_bench)
                    excess_returns = window_strat - window_bench
                    rolling_alpha.append(alpha)
                    rolling_beta.append(beta)
                    rolling_tracking_error.append(tracking_error)
                    rolling_excess_returns.append(excess_returns.mean())
                    dates.append(strategy_returns.index[i])
                except Exception:
                    continue

            if dates:
                rolling_comp = RollingComparison(
                    window_size=window,
                    rolling_alpha=pd.Series(rolling_alpha, index=dates),
                    rolling_beta=pd.Series(rolling_beta, index=dates),
                    rolling_tracking_error=pd.Series(rolling_tracking_error, index=dates),
                    rolling_excess_returns=pd.Series(rolling_excess_returns, index=dates),
                )
                rolling_comparisons.append(rolling_comp)

        return rolling_comparisons

    def multi_benchmark_comparison(
        self, strategy_returns: pd.Series, benchmark_data: dict[str, pd.Series]
    ) -> MultiBenchmarkSummary:
        """
        複数ベンチマークとの比較

        Args:
            strategy_returns: 戦略のリターン系列
            benchmark_data: ベンチマーク名とリターン系列の辞書

        Returns:
            複数ベンチマーク比較サマリー
        """
        comparisons = []
        summary_stats: dict[str, Any] = {
            "best_benchmark": None,
            "worst_benchmark": None,
            "avg_information_ratio": 0.0,
            "avg_alpha": 0.0,
            "benchmark_correlations": {},
        }

        information_ratios = []
        alphas = []

        for name, bench_returns in benchmark_data.items():
            try:
                comparison = self.compare_with_benchmark(
                    strategy_returns, bench_returns, name
                )
                comparisons.append(comparison)

                information_ratios.append(comparison.information_ratio)
                alphas.append(comparison.alpha)

                # ベンチマーク間の相関
                for other_name, other_bench in benchmark_data.items():
                    if name != other_name:
                        corr_key = f"{name}_vs_{other_name}"
                        if corr_key not in summary_stats["benchmark_correlations"]:
                            try:
                                corr = bench_returns.corr(other_bench)
                                summary_stats["benchmark_correlations"][
                                    corr_key
                                ] = float(corr)
                            except Exception:
                                pass

            except Exception as e:
                warnings.warn(f"Comparison with {name} failed: {e}")
                continue

        if information_ratios:
            summary_stats["avg_information_ratio"] = float(np.mean(information_ratios))
            summary_stats["avg_alpha"] = float(np.mean(alphas))

            # 最適/最悪ベンチマーク
            best_idx = np.argmax(information_ratios)
            worst_idx = np.argmin(information_ratios)

            summary_stats["best_benchmark"] = comparisons[best_idx].benchmark_name
            summary_stats["worst_benchmark"] = comparisons[worst_idx].benchmark_name

        return {"comparisons": comparisons, "summary_stats": summary_stats}

    def run_comprehensive_benchmark_analysis(
        self,
        strategy_returns: pd.Series,
        benchmark_data: dict[str, pd.Series],
        include_rolling: bool = True,
    ) -> BenchmarkComparisonResult:
        """
        包括的なベンチマーク比較分析を実行

        Args:
            strategy_returns: 戦略のリターン系列
            benchmark_data: ベンチマーク名とリターン系列の辞書
            include_rolling: ローリング分析を含むか

        Returns:
            包括的な比較結果
        """
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
        strategy_performance = calculate_performance_metrics(strategy_returns, risk_free_rate=self.risk_free_rate)

        # ベンチマークのパフォーマンス
        benchmark_performance = {}
        for name, returns in benchmark_data.items():
            benchmark_performance[name] = calculate_performance_metrics(returns, risk_free_rate=self.risk_free_rate)

        # 個別比較
        comparisons = []
        for name, returns in benchmark_data.items():
            try:
                comparison = self.compare_with_benchmark(
                    strategy_returns, returns, name
                )
                comparisons.append(comparison)
            except Exception as e:
                warnings.warn(f"Benchmark comparison with {name} failed: {e}")

        # ローリング比較
        rolling_comparisons = None
        if include_rolling and comparisons:
            # 最初のベンチマークでローリング分析
            first_comparison = comparisons[0]
            rolling_comparisons = self.rolling_comparison_analysis(
                first_comparison.strategy_returns, first_comparison.benchmark_returns
            )

        # 複数ベンチマーク比較
        multi_benchmark_summary = self.multi_benchmark_comparison(
            strategy_returns, benchmark_data
        )

        return BenchmarkComparisonResult(
            strategy_performance=strategy_performance,
            benchmark_performance=benchmark_performance,
            comparisons=comparisons,
            rolling_comparisons=rolling_comparisons,
            multi_benchmark_summary=multi_benchmark_summary,
        )

    def plot_benchmark_comparison(
        self, result: BenchmarkComparisonResult, save_path: str | None = None
    ) -> None:
        """ベンチマーク比較結果を可視化"""
        if not result.comparisons:
            print("No benchmark comparisons available for plotting")
            return

        n_comparisons = len(result.comparisons)
        n_cols = min(3, n_comparisons)
        n_rows = (n_comparisons + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        fig.suptitle("Benchmark Comparison Analysis", fontsize=16)

        for i, comparison in enumerate(result.comparisons):
            if i >= len(axes):
                break

            ax = axes[i]

            # 累積リターンの比較
            cum_strat = (1 + comparison.strategy_returns).cumprod()
            cum_bench = (1 + comparison.benchmark_returns).cumprod()

            ax.plot(cum_strat.index, cum_strat, label="Strategy", linewidth=2)
            ax.plot(
                cum_bench.index,
                cum_bench,
                label=comparison.benchmark_name,
                linewidth=2,
                alpha=0.7,
            )

            ax.set_title(f"Cumulative Returns vs {comparison.benchmark_name}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Cumulative Return")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 主要指標をテキストで表示
            info_text = """Alpha: {comparison.alpha}
Beta: {comparison.beta}
IR: {comparison.information_ratio}
R2: {comparison.r_squared}""".format(comparison=comparison)
            ax.text(
                0.02,
                0.98,
                info_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        ax.set_ylabel("Beta")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Rolling comparisons plot
        if result.rolling_comparisons:
            rolling_comparisons = result.rolling_comparisons
            colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(rolling_comparisons)))

            # Tracking Error
            ax = axes[1, 0]  # type: ignore[index]
            for i, rolling in enumerate(rolling_comparisons):
                valid_data = rolling.rolling_tracking_error.dropna()
                if len(valid_data) > 0:
                    ax.plot(
                        valid_data.index,
                        valid_data,
                        label=f"Window: {rolling.window_size}d",
                        color=colors[i],
                        linewidth=1.5,
                    )
            ax.set_title("Rolling Tracking Error")
            ax.set_ylabel("Tracking Error")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Excess Returns
            ax = axes[1, 1]  # type: ignore[index]
            for i, rolling in enumerate(rolling_comparisons):
                valid_data = rolling.rolling_excess_returns.dropna()
                if len(valid_data) > 0:
                    ax.plot(
                        valid_data.index,
                        valid_data,
                        label=f"Window: {rolling.window_size}d",
                        color=colors[i],
                        linewidth=1.5,
                    )
            ax.set_title("Rolling Excess Returns")
            ax.set_ylabel("Excess Returns")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_plot(save_path)
            print(f"Rolling plot saved to {save_path}")

        plt.show()

    def export_results(self, result: BenchmarkComparisonResult, output_path: str) -> None:
        """分析結果を JSON ファイルにエクスポート"""
        export_data = {
            "strategy_performance": result.strategy_performance,
            "benchmark_performance": result.benchmark_performance,
            "comparisons": [
                {
                    "benchmark_name": comp.benchmark_name,
                    "tracking_error": comp.tracking_error,
                    "information_ratio": comp.information_ratio,
                    "beta": comp.beta,
                    "alpha": comp.alpha,
                }
                for comp in result.comparisons
            ]
        }

        write_json(output_path, export_data, indent=2, ensure_ascii=False)

    def analyze_benchmark_comparison(
        self,
        strategy_file: str,
        benchmark_files: list[str]
    ) -> BenchmarkComparisonResult:
        """
        ファイルからベンチマーク比較分析を実行

        Args:
            strategy_file: 戦略リターンファイルパス
            benchmark_files: ベンチマークリターンファイルパスのリスト

        Returns:
            包括的な比較結果
        """
        # 戦略データを読み込み
        strategy_returns = DataLoader.load_csv_strict(
            strategy_file, index_col=0, parse_dates=True
        ).iloc[:, 0]

        # ベンチマークデータを読み込み
        benchmark_data = {}
        for benchmark_file in benchmark_files:
            benchmark_name = os.path.splitext(os.path.basename(benchmark_file))[0]
            benchmark_returns = DataLoader.load_csv_strict(
                benchmark_file, index_col=0, parse_dates=True
            ).iloc[:, 0]
            benchmark_data[benchmark_name] = benchmark_returns

        # 包括的な分析実行
        return self.run_comprehensive_benchmark_analysis(strategy_returns, benchmark_data)

    def plot_rolling_comparison(self, rolling_comparisons: list[RollingComparison], save_path: str | None = None) -> None:
        """ローリング比較結果を可視化"""
        if not rolling_comparisons:
            print("No rolling comparisons available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Rolling Benchmark Comparison Analysis", fontsize=16)

        colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(rolling_comparisons)))

        # Alpha
        ax = axes[0, 0]
        for i, rolling in enumerate(rolling_comparisons):
            valid_data = rolling.rolling_alpha.dropna()
            if len(valid_data) > 0:
                ax.plot(
                    valid_data.index,
                    valid_data,
                    label=f"Window: {rolling.window_size}d",
                    color=colors[i],
                    linewidth=1.5,
                )
        ax.set_title("Rolling Alpha")
        ax.set_ylabel("Alpha")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Beta
        ax = axes[0, 1]
        for i, rolling in enumerate(rolling_comparisons):
            valid_data = rolling.rolling_beta.dropna()
            if len(valid_data) > 0:
                ax.plot(
                    valid_data.index,
                    valid_data,
                    label=f"Window: {rolling.window_size}d",
                    color=colors[i],
                    linewidth=1.5,
                )
        ax.set_title("Rolling Beta")
        ax.set_ylabel("Beta")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Tracking Error
        ax = axes[1, 0]
        for i, rolling in enumerate(rolling_comparisons):
            valid_data = rolling.rolling_tracking_error.dropna()
            if len(valid_data) > 0:
                ax.plot(
                    valid_data.index,
                    valid_data,
                    label=f"Window: {rolling.window_size}d",
                    color=colors[i],
                    linewidth=1.5,
                )
        ax.set_title("Rolling Tracking Error")
        ax.set_ylabel("Tracking Error")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Excess Returns
        ax = axes[1, 1]
        for i, rolling in enumerate(rolling_comparisons):
            valid_data = rolling.rolling_excess_returns.dropna()
            if len(valid_data) > 0:
                ax.plot(
                    valid_data.index,
                    valid_data,
                    label=f"Window: {rolling.window_size}d",
                    color=colors[i],
                    linewidth=1.5,
                )
        ax.set_title("Rolling Excess Returns")
        ax.set_ylabel("Excess Returns")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_plot(save_path)
            print(f"Rolling plot saved to {save_path}")

        plt.show()

def main() -> None:
    """メイン関数"""
    parser = argparse.ArgumentParser(description="ベンチマーク比較分析")
    parser.add_argument("--strategy-file", required=True, help="戦略リターンファイル")
    parser.add_argument("--benchmark-files", nargs="+", required=True, help="ベンチマークリターンファイル")
    parser.add_argument("--output-dir", default="benchmark_comparison_results", help="出力ディレクトリ")
    parser.add_argument("--risk-free-rate", type=float, default=0.02, help="無リスク金利")

    args = parser.parse_args()

    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)

    # アナライザー初期化
    analyzer = BenchmarkComparisonAnalyzer(risk_free_rate=args.risk_free_rate)

    # 分析実行
    result = analyzer.analyze_benchmark_comparison(args.strategy_file, args.benchmark_files)

    # プロット保存パス
    rolling_plot_path = os.path.join(args.output_dir, "rolling_comparison.png")

    # ローリング比較プロット
    if result.rolling_comparisons:
        analyzer.plot_rolling_comparison(result.rolling_comparisons, save_path=rolling_plot_path)

    # JSON エクスポート
    json_path = os.path.join(args.output_dir, "benchmark_comparison_results.json")
    analyzer.export_results(result, json_path)

    print("Benchmark comparison analysis completed!")

if __name__ == "__main__":
    main()
