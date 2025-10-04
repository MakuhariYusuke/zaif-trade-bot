#!/usr/bin/env python3
"""
手数料感度分析モジュール

手数料パラメータを変化させたときの性能感度を可視化します。
取引コストが戦略の性能に与える影響を分析します。
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ztb.utils.data_utils import load_csv_data

@dataclass
class CostSensitivityResult:
    """手数料感度分析結果"""
    fee_levels: List[float]
    metrics_by_fee: Dict[float, Dict[str, float]]
    sensitivity_scores: Dict[str, float]
    breakeven_analysis: Dict[str, float]

class CostSensitivityAnalyzer:
    """手数料感度分析クラス"""

    def __init__(self, base_fee: float = 0.001,  # 0.1%
                 fee_range: Tuple[float, float] = (0.0001, 0.01),  # 0.01% to 1%
                 fee_steps: int = 20):
        """
        Args:
            base_fee: 基準手数料（比較用）
            fee_range: 分析対象の手数料範囲（最小, 最大）
            fee_steps: 手数料のステップ数
        """
        self.base_fee = base_fee
        self.fee_range = fee_range
        self.fee_steps = fee_steps

    def generate_fee_levels(self) -> List[float]:
        """分析対象の手数料レベルを生成"""
        return list(np.linspace(self.fee_range[0], self.fee_range[1], self.fee_steps))

    def apply_trading_costs(self, returns: pd.Series, fee_per_trade: float,
                           trades_per_day: float = 1.0) -> pd.Series:
        """
        リターン系列に取引コストを適用

        Args:
            returns: コスト適用前のリターン系列
            fee_per_trade: 1取引あたりの手数料
            trades_per_day: 1日あたりの取引回数

        Returns:
            コスト適用後のリターン系列
        """
        # 取引コストの計算（片道手数料 × 取引回数）
        daily_cost = fee_per_trade * trades_per_day

        # コストをリターンから差し引く
        cost_adjusted_returns = returns - daily_cost

        return cost_adjusted_returns

    def calculate_cost_sensitivity_metrics(self, returns: pd.Series, fee_levels: List[float],
                                         trades_per_day: float = 1.0) -> Dict[float, Dict[str, float]]:
        """
        各手数料レベルでの性能メトリクスを計算

        Args:
            returns: 基準リターン系列
            fee_levels: 分析対象の手数料レベル
            trades_per_day: 1日あたりの取引回数

        Returns:
            手数料レベル -> メトリクス の辞書
        """
        metrics_by_fee = {}

        for fee in fee_levels:
            # コスト適用後のリターン
            cost_adjusted_returns = self.apply_trading_costs(returns, fee, trades_per_day)

            # メトリクス計算
            metrics = self._calculate_performance_metrics(cost_adjusted_returns)
            metrics_by_fee[fee] = metrics

        return metrics_by_fee

    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """性能メトリクスを計算"""
        if len(returns) == 0:
            return {
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'volatility': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }

        returns_array = returns.values

        # 基本メトリクス
        total_return = float(np.prod(1 + returns_array) - 1)  # type: ignore
        volatility = float(np.std(returns_array) * np.sqrt(252))  # type: ignore

        # Sharpe比率
        risk_free_rate = 0.02  # 仮定の無リスク金利
        sharpe_ratio = (returns.mean() - risk_free_rate/252) / volatility if volatility > 0 else 0

        # 勝率
        win_rate = float((returns > 0).mean())

        # 最大ドローダウン
        cumulative = np.cumprod(1 + returns_array)  # type: ignore
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(np.min(drawdown))

        # プロフィットファクター
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]

        gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return {
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }

    def calculate_sensitivity_scores(self, metrics_by_fee: Dict[float, Dict[str, float]]) -> Dict[str, float]:
        """
        感度スコアを計算（手数料変化に対する性能変化の大きさ）

        Args:
            metrics_by_fee: 手数料レベルごとのメトリクス

        Returns:
            メトリクス名 -> 感度スコア の辞書
        """
        if not metrics_by_fee:
            return {}

        fees = sorted(metrics_by_fee.keys())
        sensitivity_scores = {}

        for metric_name in metrics_by_fee[fees[0]].keys():
            metric_values = [metrics_by_fee[fee][metric_name] for fee in fees]

            # 感度スコア：手数料1%変化に対するメトリクスの変化率
            fee_changes = np.diff(fees)
            metric_changes = np.diff(metric_values)

            # 平均感度（絶対変化の平均）
            if len(fee_changes) > 0:
                sensitivities = metric_changes / fee_changes
                avg_sensitivity = np.mean(np.abs(sensitivities))
                sensitivity_scores[metric_name] = float(avg_sensitivity)
            else:
                sensitivity_scores[metric_name] = 0.0

        return sensitivity_scores

    def analyze_breakeven_points(self, metrics_by_fee: Dict[float, Dict[str, float]],
                               benchmark_return: float = 0.0) -> Dict[str, float]:
        """
        損益分岐点分析（ベンチマーク以上のリターンを達成する最大手数料）

        Args:
            metrics_by_fee: 手数料レベルごとのメトリクス
            benchmark_return: ベンチマークリターン（デフォルト: 0%）

        Returns:
            メトリクス名 -> 損益分岐点手数料 の辞書
        """
        breakeven_points = {}

        for metric_name in ['sharpe_ratio', 'total_return', 'profit_factor']:
            # ベンチマークを上回る手数料レベルを探す
            valid_fees = []
            for fee, metrics in metrics_by_fee.items():
                if metric_name == 'sharpe_ratio' and metrics[metric_name] > 0:
                    valid_fees.append(fee)
                elif metric_name == 'total_return' and metrics[metric_name] > benchmark_return:
                    valid_fees.append(fee)
                elif metric_name == 'profit_factor' and metrics[metric_name] > 1.0:
                    valid_fees.append(fee)

            if valid_fees:
                breakeven_points[metric_name] = max(valid_fees)
            else:
                breakeven_points[metric_name] = 0.0

        return breakeven_points

    def run_cost_sensitivity_analysis(self, returns: pd.Series,
                                    trades_per_day: float = 1.0) -> CostSensitivityResult:
        """
        手数料感度分析を実行

        Args:
            returns: リターン系列
            trades_per_day: 1日あたりの取引回数

        Returns:
            分析結果
        """
        # 手数料レベルの生成
        fee_levels = self.generate_fee_levels()

        # 各手数料レベルでの性能計算
        metrics_by_fee = self.calculate_cost_sensitivity_metrics(
            returns, fee_levels, trades_per_day
        )

        # 感度スコアの計算
        sensitivity_scores = self.calculate_sensitivity_scores(metrics_by_fee)

        # 損益分岐点分析
        breakeven_analysis = self.analyze_breakeven_points(metrics_by_fee)

        return CostSensitivityResult(
            fee_levels=fee_levels,
            metrics_by_fee=metrics_by_fee,
            sensitivity_scores=sensitivity_scores,
            breakeven_analysis=breakeven_analysis
        )

    def analyze(self, metrics: Any) -> Dict[str, Any]:
        """
        BenchmarkMetricsからコスト感度分析を実行

        Args:
            metrics: 評価メトリクス

        Returns:
            分析結果の辞書
        """
        # returnsデータをpandas Seriesに変換
        if not hasattr(metrics, 'returns') or not metrics.returns:
            return {
                'error': 'No returns data available for analysis',
                'cost_sensitivity_score': 0.0,
                'breakeven_fee': 0.0,
                'profit_erosion_rate': 0.0
            }

        # 日付インデックスを作成（仮定）
        dates = pd.date_range(start='2023-01-01', periods=len(metrics.returns), freq='D')
        returns = pd.Series(metrics.returns, index=dates)

        try:
            # コスト感度分析を実行
            result = self.run_cost_sensitivity_analysis(returns)

            # 結果を辞書形式に変換
            analysis_result: Dict[str, Any] = {
                'cost_sensitivity_score': result.sensitivity_scores.get('overall_sensitivity', 0.0),
                'breakeven_fee': result.breakeven_analysis.get('breakeven_fee', 0.0),
                'profit_erosion_rate': result.breakeven_analysis.get('profit_erosion_rate', 0.0),
                'fee_range': result.fee_levels,
                'sensitivity_scores': result.sensitivity_scores,
                'breakeven_analysis': result.breakeven_analysis
            }

            # 各手数料レベルのメトリクスを追加
            for fee, metrics_dict in result.metrics_by_fee.items():
                analysis_result[f'fee_{fee}'] = metrics_dict

            return analysis_result

        except Exception as e:
            return {
                'error': f'Analysis failed: {str(e)}',
                'cost_sensitivity_score': 0.0,
                'breakeven_fee': 0.0,
                'profit_erosion_rate': 0.0
            }

    def plot_cost_sensitivity_analysis(self, result: CostSensitivityResult,
                                     save_path: Optional[str] = None):
        """手数料感度分析結果を可視化"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Trading Cost Sensitivity Analysis', fontsize=16)

        fee_levels = result.fee_levels
        fee_percentages = [f * 100 for f in fee_levels]  # パーセント表示

        # Sharpe比率の感度
        sharpe_ratios = [result.metrics_by_fee[fee]['sharpe_ratio'] for fee in fee_levels]
        axes[0, 0].plot(fee_percentages, sharpe_ratios, 'b-o', linewidth=2, markersize=4)
        axes[0, 0].axvline(x=self.base_fee * 100, color='r', linestyle='--',
                          label=f'Base Fee ({self.base_fee*100:.1f}%)')
        axes[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[0, 0].set_title('Sharpe Ratio vs Trading Cost')
        axes[0, 0].set_xlabel('Trading Cost (%)')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 総リターンの感度
        total_returns = [result.metrics_by_fee[fee]['total_return'] * 100 for fee in fee_levels]
        axes[0, 1].plot(fee_percentages, total_returns, 'g-o', linewidth=2, markersize=4)
        axes[0, 1].axvline(x=self.base_fee * 100, color='r', linestyle='--',
                          label=f'Base Fee ({self.base_fee*100:.1f}%)')
        axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[0, 1].set_title('Total Return vs Trading Cost')
        axes[0, 1].set_xlabel('Trading Cost (%)')
        axes[0, 1].set_ylabel('Total Return (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 勝率の感度
        win_rates = [result.metrics_by_fee[fee]['win_rate'] * 100 for fee in fee_levels]
        axes[0, 2].plot(fee_percentages, win_rates, 'purple', marker='o', linewidth=2, markersize=4)
        axes[0, 2].axvline(x=self.base_fee * 100, color='r', linestyle='--',
                          label=f'Base Fee ({self.base_fee*100:.1f}%)')
        axes[0, 2].set_title('Win Rate vs Trading Cost')
        axes[0, 2].set_xlabel('Trading Cost (%)')
        axes[0, 2].set_ylabel('Win Rate (%)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # プロフィットファクターの感度
        profit_factors = [result.metrics_by_fee[fee]['profit_factor'] for fee in fee_levels]
        axes[1, 0].plot(fee_percentages, profit_factors, 'orange', marker='o', linewidth=2, markersize=4)
        axes[1, 0].axvline(x=self.base_fee * 100, color='r', linestyle='--',
                          label=f'Base Fee ({self.base_fee*100:.1f}%)')
        axes[1, 0].axhline(y=1.0, color='k', linestyle='-', alpha=0.3, label='Break-even')
        axes[1, 0].set_title('Profit Factor vs Trading Cost')
        axes[1, 0].set_xlabel('Trading Cost (%)')
        axes[1, 0].set_ylabel('Profit Factor')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 最大ドローダウンの感度
        max_drawdowns = [result.metrics_by_fee[fee]['max_drawdown'] * 100 for fee in fee_levels]
        axes[1, 1].plot(fee_percentages, max_drawdowns, 'r-o', linewidth=2, markersize=4)
        axes[1, 1].axvline(x=self.base_fee * 100, color='r', linestyle='--',
                          label=f'Base Fee ({self.base_fee*100:.1f}%)')
        axes[1, 1].set_title('Max Drawdown vs Trading Cost')
        axes[1, 1].set_xlabel('Trading Cost (%)')
        axes[1, 1].set_ylabel('Max Drawdown (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 感度スコアの棒グラフ
        metric_names = list(result.sensitivity_scores.keys())
        sensitivity_values = list(result.sensitivity_scores.values())

        bars = axes[1, 2].bar(metric_names, sensitivity_values, color='skyblue', alpha=0.7)
        axes[1, 2].set_title('Sensitivity Scores')
        axes[1, 2].set_ylabel('Sensitivity (|Δmetric/Δfee|)')
        axes[1, 2].tick_params(axis='x', rotation=45)

        # 値ラベルを追加
        for bar, sensitivity_value in zip(bars, sensitivity_values):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{sensitivity_value:.2f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cost sensitivity analysis plot saved to {save_path}")

        plt.show()

    def print_cost_sensitivity_summary(self, result: CostSensitivityResult):
        """手数料感度分析のサマリーを表示"""
        print("\n" + "="*60)
        print("COST SENSITIVITY ANALYSIS SUMMARY")
        print("="*60)

        print(f"\nAnalyzed {len(result.fee_levels)} fee levels: "
              f"{result.fee_levels[0]*100:.2f}% to {result.fee_levels[-1]*100:.2f}%")

        print("\nSENSITIVITY SCORES (higher = more sensitive to fee changes):")
        print("-" * 50)

        for metric_name, sensitivity_score in result.sensitivity_scores.items():
            print(f"{metric_name:8s}: {sensitivity_score:.4f}")

        print("\nBREAKEVEN ANALYSIS (maximum fee for positive metrics):")
        print("-" * 50)

        for metric_name, breakeven_fee in result.breakeven_analysis.items():
            print(f"{metric_name:12s}: {breakeven_fee*100:.2f}%")

        # 推奨手数料範囲
        print("\nRECOMMENDED FEE RANGE:")
        print("-" * 30)

        # Sharpe比率が正の範囲
        positive_sharpe_fees = [
            fee for fee, metrics in result.metrics_by_fee.items()
            if metrics['sharpe_ratio'] > 0
        ]

        if positive_sharpe_fees:
            min_fee_val = min(positive_sharpe_fees)
            max_fee_val = max(positive_sharpe_fees)
            print(f"Positive Sharpe ratio range: {min_fee_val*100:.2f}% to {max_fee_val*100:.2f}%")
        else:
            print("No fee level produces positive Sharpe ratio")

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='Cost Sensitivity Analysis')
    parser.add_argument('--returns-csv', required=True,
                       help='Path to returns CSV file (with date index)')
    parser.add_argument('--output-dir', default='results/cost_sensitivity',
                       help='Output directory for results')
    parser.add_argument('--trades-per-day', type=float, default=1.0,
                       help='Average trades per day')
    parser.add_argument('--base-fee', type=float, default=0.001,
                       help='Base fee for comparison (0.001 = 0.1%)')
    parser.add_argument('--fee-min', type=float, default=0.0001,
                       help='Minimum fee to analyze (0.0001 = 0.01%)')
    parser.add_argument('--fee-max', type=float, default=0.01,
                       help='Maximum fee to analyze (0.01 = 1%)')
    parser.add_argument('--fee-steps', type=int, default=20,
                       help='Number of fee levels to analyze')

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # リターンデータの読み込み
    try:
        returns = load_csv_data(args.returns_csv, index_col=0, parse_dates=True)
        if returns.shape[1] == 1:
            returns = returns.iloc[:, 0]  # Seriesに変換
        else:
            # 複数列ある場合は最初の列を使用
            returns = returns.iloc[:, 0]
        print(f"Loaded returns data with {len(returns)} observations")
    except Exception as e:
        print(f"Error loading returns data: {e}")
        return

    # 手数料感度分析器の初期化
    analyzer = CostSensitivityAnalyzer(
        base_fee=args.base_fee,
        fee_range=(args.fee_min, args.fee_max),
        fee_steps=args.fee_steps
    )

    try:
        # 分析実行
        print("Running cost sensitivity analysis...")
        result = analyzer.run_cost_sensitivity_analysis(returns, args.trades_per_day)

        # サマリー表示
        analyzer.print_cost_sensitivity_summary(result)

        # 結果保存
        result_file = output_dir / 'cost_sensitivity_results.json'

        # 結果を辞書に変換
        result_dict = {
            'fee_levels': result.fee_levels,
            'metrics_by_fee': result.metrics_by_fee,
            'sensitivity_scores': result.sensitivity_scores,
            'breakeven_analysis': result.breakeven_analysis,
            'analysis_parameters': {
                'trades_per_day': args.trades_per_day,
                'base_fee': args.base_fee,
                'fee_range': (args.fee_min, args.fee_max),
                'fee_steps': args.fee_steps
            }
        }

        # JSONとして保存
        import json
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)

        print(f"\nResults saved to {result_file}")

        # 可視化
        plot_file = output_dir / 'cost_sensitivity_analysis.png'
        analyzer.plot_cost_sensitivity_analysis(result, str(plot_file))

    except Exception as e:
        print(f"Error during analysis: {e}")
        return

if __name__ == '__main__':
    main()