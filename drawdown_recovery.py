#!/usr/bin/env python3
"""
ドローダウン回復分析モジュール

最大ドローダウンから資産が回復するまでの期間を計測し、
回復パターンとリスク指標を分析します。
"""

import argparse
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DrawdownPeriod:
    """ドローダウン期間"""
    start_date: datetime
    bottom_date: datetime
    recovery_date: Optional[datetime]
    drawdown_depth: float
    recovery_time: Optional[int]  # 日数
    peak_value: float
    bottom_value: float
    recovery_value: Optional[float]

@dataclass
class DrawdownRecoveryResult:
    """ドローダウン回復分析結果"""
    drawdown_periods: List[DrawdownPeriod]
    recovery_stats: Dict[str, float]
    risk_metrics: Dict[str, float]
    recovery_patterns: Dict[str, List[float]]

class DrawdownRecoveryAnalyzer:
    """ドローダウン回復分析クラス"""

    def __init__(self, min_drawdown_threshold: float = 0.05,  # 5%以上のドローダウン
                 recovery_threshold: float = 0.99):  # ピークの99%まで回復
        """
        Args:
            min_drawdown_threshold: 分析対象とする最小ドローダウン深度
            recovery_threshold: 回復とみなすピークからの割合
        """
        self.min_drawdown_threshold = min_drawdown_threshold
        self.recovery_threshold = recovery_threshold  # type: ignore

    def calculate_drawdown_series(self, returns: pd.Series) -> pd.Series:
        """
        ドローダウン系列を計算

        Args:
            returns: リターン系列

        Returns:
            ドローダウン系列（負の値）
        """
        # 累積リターンの計算
        cumulative = (1 + returns).cumprod()

        # 実行中の最大値
        running_max = cumulative.expanding().max()

        # ドローダウンの計算
        drawdown = (cumulative - running_max) / running_max

        return drawdown

    def identify_drawdown_periods(self, returns: pd.Series) -> List[DrawdownPeriod]:
        """
        ドローダウン期間を特定

        Args:
            returns: リターン系列

        Returns:
            ドローダウン期間のリスト
        """
        drawdown_series = self.calculate_drawdown_series(returns)
        cumulative = (1 + returns).cumprod()

        drawdown_periods = []
        in_drawdown = False
        current_period: Optional[DrawdownPeriod] = None

        for i in range(len(drawdown_series)):
            current_drawdown = drawdown_series.iloc[i]
            current_cumulative = cumulative.iloc[i]
            current_date = drawdown_series.index[i]

            if not in_drawdown and current_drawdown <= -self.min_drawdown_threshold:
                # ドローダウン開始
                in_drawdown = True
                # 直前のピークを見つける
                peak_idx = drawdown_series.iloc[:i][drawdown_series.iloc[:i] == 0].index
                peak_date = peak_idx[-1] if len(peak_idx) > 0 else drawdown_series.index[0]
                peak_value = cumulative.loc[peak_date]

                current_period = DrawdownPeriod(
                    start_date=peak_date,
                    bottom_date=current_date,
                    recovery_date=None,
                    drawdown_depth=abs(current_drawdown),
                    recovery_time=None,
                    peak_value=peak_value,
                    bottom_value=current_cumulative,
                    recovery_value=None
                )

            elif in_drawdown:
                # ドローダウン継続中
                assert current_period is not None
                if current_drawdown < current_period.drawdown_depth:
                    # より深いドローダウン
                    current_period.bottom_date = current_date
                    current_period.drawdown_depth = abs(current_drawdown)
                    current_period.bottom_value = current_cumulative

                # 回復チェック
                recovery_target = current_period.peak_value * self.recovery_threshold
                if current_cumulative >= recovery_target:
                    # 回復完了
                    current_period.recovery_date = current_date
                    current_period.recovery_time = (current_date - current_period.start_date).days
                    current_period.recovery_value = current_cumulative
                    drawdown_periods.append(current_period)
                    in_drawdown = False
                    current_period = None

        # 未回復のドローダウンも追加
        if in_drawdown and current_period:
            drawdown_periods.append(current_period)

        return drawdown_periods

    def calculate_recovery_statistics(self, drawdown_periods: List[DrawdownPeriod]) -> Dict[str, Union[int, float]]:
        """
        回復統計を計算

        Args:
            drawdown_periods: ドローダウン期間リスト

        Returns:
            回復統計
        """
        if not drawdown_periods:
            return {
                'total_periods': 0,
                'recovered_periods': 0,
                'recovery_rate': 0.0,
                'avg_recovery_time': 0.0,
                'max_recovery_time': 0.0,
                'avg_drawdown_depth': 0.0,
                'max_drawdown_depth': 0.0
            }

        recovered_periods = [p for p in drawdown_periods if p.recovery_date is not None]
        recovery_times = [p.recovery_time for p in recovered_periods if p.recovery_time is not None]
        drawdown_depths = [p.drawdown_depth for p in drawdown_periods]

        stats = {
            'total_periods': len(drawdown_periods),
            'recovered_periods': len(recovered_periods),
            'recovery_rate': len(recovered_periods) / len(drawdown_periods),
            'avg_recovery_time': np.mean(recovery_times) if recovery_times else 0.0,  # type: ignore
            'max_recovery_time': np.max(recovery_times) if recovery_times else 0.0,  # type: ignore
            'avg_drawdown_depth': np.mean(drawdown_depths),  # type: ignore
            'max_drawdown_depth': np.max(drawdown_depths)  # type: ignore
        }

        return stats  # type: ignore

    def calculate_risk_metrics(self, drawdown_periods: List[DrawdownPeriod],
                             returns: pd.Series) -> Dict[str, float]:
        """
        リスク指標を計算

        Args:
            drawdown_periods: ドローダウン期間リスト
            returns: リターン系列

        Returns:
            リスク指標
        """
        if not drawdown_periods:
            return {
                'max_drawdown': 0.0,
                'avg_drawdown': 0.0,
                'drawdown_frequency': 0.0,
                'recovery_efficiency': 0.0,
                'pain_index': 0.0
            }

        # 基本指標
        max_dd = max(p.drawdown_depth for p in drawdown_periods)
        avg_dd = np.mean([p.drawdown_depth for p in drawdown_periods])

        # ドローダウン頻度（年間）
        total_days = (returns.index[-1] - returns.index[0]).days
        annual_frequency = len(drawdown_periods) * 365 / total_days

        # 回復効率（回復したドローダウンの平均回復時間 / ドローダウン深度）
        recovered = [p for p in drawdown_periods if p.recovery_date is not None]
        if recovered:
            recovery_efficiency = float(np.mean([
                p.recovery_time / p.drawdown_depth
                for p in recovered if p.recovery_time is not None
            ]))
        else:
            recovery_efficiency = 0.0

        # Pain Index（ドローダウンの時間重み平均）
        drawdown_series = self.calculate_drawdown_series(returns)
        pain_index = np.mean(np.abs(drawdown_series[drawdown_series < 0]))

        return {
            'max_drawdown': max_dd,
            'avg_drawdown': avg_dd,
            'drawdown_frequency': annual_frequency,
            'recovery_efficiency': recovery_efficiency,
            'pain_index': pain_index
        }  # type: ignore

    def analyze_recovery_patterns(self, drawdown_periods: List[DrawdownPeriod]) -> Dict[str, List[float]]:
        """
        回復パターンを分析

        Args:
            drawdown_periods: ドローダウン期間リスト

        Returns:
            回復パターン分析結果
        """
        patterns = {
            'recovery_times': [],
            'drawdown_depths': [],
            'recovery_ratios': []  # 回復後の超過リターン
        }

        for period in drawdown_periods:
            patterns['drawdown_depths'].append(period.drawdown_depth)

            if period.recovery_date is not None and period.recovery_time is not None:
                patterns['recovery_times'].append(period.recovery_time)

                # 回復後の超過リターン（ピークからの追加リターン）
                if period.recovery_value is not None:
                    excess_return = (period.recovery_value - period.peak_value) / period.peak_value
                    patterns['recovery_ratios'].append(excess_return)

        return patterns

    def run_drawdown_recovery_analysis(self, returns: pd.Series) -> DrawdownRecoveryResult:
        """
        ドローダウン回復分析を実行

        Args:
            returns: リターン系列

        Returns:
            分析結果
        """
        # ドローダウン期間の特定
        drawdown_periods = self.identify_drawdown_periods(returns)

        # 回復統計の計算
        recovery_stats = self.calculate_recovery_statistics(drawdown_periods)

        # リスク指標の計算
        risk_metrics = self.calculate_risk_metrics(drawdown_periods, returns)

        # 回復パターンの分析
        recovery_patterns = self.analyze_recovery_patterns(drawdown_periods)

        return DrawdownRecoveryResult(
            drawdown_periods=drawdown_periods,
            recovery_stats=recovery_stats,
            risk_metrics=risk_metrics,
            recovery_patterns=recovery_patterns
        )

    def plot_drawdown_recovery_analysis(self, result: DrawdownRecoveryResult,
                                      returns: pd.Series, save_path: Optional[str] = None):
        """ドローダウン回復分析結果を可視化"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Drawdown Recovery Analysis', fontsize=16)

        # ドローダウン系列のプロット
        drawdown_series = self.calculate_drawdown_series(returns)
        axes[0, 0].fill_between(drawdown_series.index, drawdown_series.values, 0,
                               where=(drawdown_series < 0), color='red', alpha=0.3)
        axes[0, 0].plot(drawdown_series.index, drawdown_series.values, 'r-', linewidth=1)
        axes[0, 0].set_title('Drawdown Series')
        axes[0, 0].set_ylabel('Drawdown (%)')
        axes[0, 0].grid(True, alpha=0.3)

        # ドローダウン期間のハイライト
        for period in result.drawdown_periods:
            axes[0, 0].axvspan(period.start_date, period.bottom_date,
                              color='red', alpha=0.2, label='Drawdown Period' if period == result.drawdown_periods[0] else "")

        # 回復時間の分布
        if result.recovery_patterns['recovery_times']:
            axes[0, 1].hist(result.recovery_patterns['recovery_times'], bins=20,
                           color='skyblue', alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(np.mean(result.recovery_patterns['recovery_times']),
                              color='red', linestyle='--', linewidth=2,
                              label='Mean')
            axes[0, 1].set_title('Recovery Time Distribution')
            axes[0, 1].set_xlabel('Recovery Time (days)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # ドローダウン深度 vs 回復時間
        if result.recovery_patterns['recovery_times'] and result.recovery_patterns['drawdown_depths']:
            recovered_depths = []
            recovered_times = []

            for period in result.drawdown_periods:
                if period.recovery_date is not None and period.recovery_time is not None:
                    recovered_depths.append(period.drawdown_depth)
                    recovered_times.append(period.recovery_time)

            if recovered_depths and recovered_times:
                axes[0, 2].scatter(recovered_depths, recovered_times,
                                  color='blue', alpha=0.6, s=50)
                axes[0, 2].set_title('Drawdown Depth vs Recovery Time')
                axes[0, 2].set_xlabel('Drawdown Depth')
                axes[0, 2].set_ylabel('Recovery Time (days)')
                axes[0, 2].grid(True, alpha=0.3)

        # リスク指標のサマリー
        risk_labels = list(result.risk_metrics.keys())
        risk_values = list(result.risk_metrics.values())

        bars = axes[1, 0].bar(risk_labels, risk_values, color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('Risk Metrics Summary')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 値ラベルを追加
        for bar, value in zip(bars, risk_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        # 回復統計
        recovery_labels = ['Recovery Rate', 'Avg Recovery Time', 'Max Recovery Time']
        recovery_values = [
            result.recovery_stats['recovery_rate'] * 100,
            result.recovery_stats['avg_recovery_time'],
            result.recovery_stats['max_recovery_time']
        ]

        bars2 = axes[1, 1].bar(recovery_labels, recovery_values, color='lightgreen', alpha=0.7)
        axes[1, 1].set_title('Recovery Statistics')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].tick_params(axis='x', rotation=45)

        # 値ラベルを追加
        for bar, value in zip(bars2, recovery_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:.1f}', ha='center', va='bottom', fontsize=8)

        # ドローダウン深度の分布
        if result.recovery_patterns['drawdown_depths']:
            axes[1, 2].hist(result.recovery_patterns['drawdown_depths'], bins=15,
                           color='orange', alpha=0.7, edgecolor='black')
            axes[1, 2].axvline(np.mean(result.recovery_patterns['drawdown_depths']),
                              color='red', linestyle='--', linewidth=2,
                              label='Mean')
            axes[1, 2].set_title('Drawdown Depth Distribution')
            axes[1, 2].set_xlabel('Drawdown Depth')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Drawdown recovery analysis plot saved to {save_path}")

        plt.show()

    def print_drawdown_recovery_summary(self, result: DrawdownRecoveryResult):
        """ドローダウン回復分析のサマリーを表示"""
        print("\n" + "="*60)
        print("DRAWDOWN RECOVERY ANALYSIS SUMMARY")
        print("="*60)

        print(f"\nTotal Drawdown Periods: {result.recovery_stats['total_periods']}")
        print(f"Recovered Periods: {result.recovery_stats['recovered_periods']}")
        print(".1f")

        if result.recovery_stats['recovered_periods'] > 0:
            print(".1f")
            print(".1f")

        print(".1f")
        print(".1f")

        print("\nRISK METRICS:")
        print("-" * 20)
        print(".1f")
        print(".1f")
        print(".2f")
        print(".2f")
        print(".4f")

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='Drawdown Recovery Analysis')
    parser.add_argument('--returns-csv', required=True,
                       help='Path to returns CSV file (with date index)')
    parser.add_argument('--output-dir', default='results/drawdown_recovery',
                       help='Output directory for results')
    parser.add_argument('--min-drawdown', type=float, default=0.05,
                       help='Minimum drawdown threshold (0.05 = 5%)')
    parser.add_argument('--recovery-threshold', type=float, default=0.99,
                       help='Recovery threshold (0.99 = 99% of peak)')

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # リターンデータの読み込み
    try:
        returns = pd.read_csv(args.returns_csv, index_col=0, parse_dates=True)
        if returns.shape[1] == 1:
            returns = returns.iloc[:, 0]  # Seriesに変換
        else:
            # 複数列ある場合は最初の列を使用
            returns = returns.iloc[:, 0]
        print(f"Loaded returns data with {len(returns)} observations")
    except Exception as e:
        print(f"Error loading returns data: {e}")
        return

    # ドローダウン回復分析器の初期化
    analyzer = DrawdownRecoveryAnalyzer(
        min_drawdown_threshold=args.min_drawdown,
        recovery_threshold=args.recovery_threshold
    )

    try:
        # 分析実行
        print("Running drawdown recovery analysis...")
        result = analyzer.run_drawdown_recovery_analysis(returns)

        # サマリー表示
        analyzer.print_drawdown_recovery_summary(result)

        # 結果保存
        result_file = output_dir / 'drawdown_recovery_results.json'

        # 結果を辞書に変換
        result_dict = {
            'recovery_stats': result.recovery_stats,
            'risk_metrics': result.risk_metrics,
            'recovery_patterns': result.recovery_patterns,
            'drawdown_periods': [
                {
                    'start_date': p.start_date.isoformat(),
                    'bottom_date': p.bottom_date.isoformat(),
                    'recovery_date': p.recovery_date.isoformat() if p.recovery_date else None,
                    'drawdown_depth': p.drawdown_depth,
                    'recovery_time': p.recovery_time,
                    'peak_value': p.peak_value,
                    'bottom_value': p.bottom_value,
                    'recovery_value': p.recovery_value
                } for p in result.drawdown_periods
            ],
            'analysis_parameters': {
                'min_drawdown_threshold': args.min_drawdown,
                'recovery_threshold': args.recovery_threshold
            }
        }

        # JSONとして保存
        import json
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)

        print(f"\nResults saved to {result_file}")

        # 可視化
        plot_file = output_dir / 'drawdown_recovery_analysis.png'
        analyzer.plot_drawdown_recovery_analysis(result, returns, str(plot_file))

    except Exception as e:
        print(f"Error during analysis: {e}")
        return

if __name__ == '__main__':
    main()