#!/usr/bin/env python3

"""
ウォークフォワード分析モジュール

学習→検証→テストを逐次シフトしながら各期間の性能を記録し、
平均と分散を算出します。
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


@dataclass
class WalkforwardWindow:
    """ウォークフォワードの1ウィンドウ"""

    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: datetime
    test_end: datetime
    window_id: int


@dataclass
class WalkforwardResult:
    """ウォークフォワード分析結果"""

    windows: List[WalkforwardWindow]
    metrics_history: List[Dict[str, float]]
    cumulative_metrics: Dict[str, Dict[str, float]]
    rolling_sharpe: List[float]
    rolling_volatility: List[float]


class WalkforwardAnalyzer:
    """ウォークフォワード分析クラス"""

    def __init__(self,
                 initial_train_days: int = 252,  # 約1年
                 validation_days: int = 21,      # 約1ヶ月
                 test_days: int = 21,            # 約1ヶ月
                 step_days: int = 21):           # 1ヶ月ごとにシフト

        """
        Args:
            initial_train_days: 初期訓練期間の日数
            validation_days: 検証期間の日数
            test_days: テスト期間の日数
            step_days: ウィンドウシフトの日数
        """

        self.initial_train_days = initial_train_days
        self.validation_days = validation_days
        self.test_days = test_days
        self.step_days = step_days

    def create_walkforward_windows(self, data: pd.DataFrame) -> List[WalkforwardWindow]:
        """
        ウォークフォワードウィンドウを生成

        Args:
            data: 時系列データ（日付インデックス）

        Returns:
            ウォークフォワードウィンドウのリスト
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("データは日付インデックスである必要があります")

        data = data.sort_index()
        start_date = data.index[0]
        end_date = data.index[-1]

        windows: List[WalkforwardWindow] = []
        window_id = 0
        current_train_end = start_date + timedelta(days=self.initial_train_days)

        while current_train_end + timedelta(days=self.validation_days + self.test_days) <= end_date:
            # 訓練期間
            train_start = start_date if window_id == 0 else windows[-1].test_end + timedelta(days=1)
            train_end = current_train_end

            # 検証期間
            val_start = train_end + timedelta(days=1)
            val_end = val_start + timedelta(days=self.validation_days - 1)

            # テスト期間
            test_start = val_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.test_days - 1)

            # データが存在するか確認
            if test_end > end_date:
                break

            window = WalkforwardWindow(
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
                window_id=window_id
            )

            windows.append(window)
            window_id += 1

            # 次のウィンドウへシフト
            current_train_end += timedelta(days=self.step_days)

        return windows

    def evaluate_window_performance(self, returns: pd.Series, window: WalkforwardWindow) -> Dict[str, float]:
        """
        指定ウィンドウの性能を評価

        Args:
            returns: リターン系列
            window: 評価対象ウィンドウ

        Returns:
            性能指標の辞書
        """
        # テスト期間のデータを抽出
        test_returns = returns.loc[window.test_start:window.test_end]  # type: ignore

        if len(test_returns) == 0:
            return {
                'total_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0
            }

        # pandas Seriesをnumpy arrayに変換
        test_returns_array = np.asarray(test_returns.values, dtype=float)

        # 総リターン
        total_return = float(np.prod(1 + test_returns_array) - 1)

        # ボラティリティ（年率化）
        volatility = float(np.std(test_returns_array) * np.sqrt(252))

        # Sharpe比率
        risk_free_rate = 0.02  # 仮定の無リスク金利
        sharpe_ratio = (test_returns.mean() - risk_free_rate/252) / volatility if volatility > 0 else 0

        # 勝率
        win_rate = float((test_returns > 0).mean())

        # 最大ドローダウン
        cumulative = np.cumprod(1 + test_returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(np.min(drawdown))

        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown
        }

    def run_walkforward_analysis(self, returns: pd.Series) -> WalkforwardResult:
        """
        ウォークフォワード分析を実行

        Args:
            returns: リターン系列

        Returns:
            分析結果
        """
        # データフレームに変換
        data = returns.to_frame()

        # ウィンドウ生成
        windows = self.create_walkforward_windows(data)

        # 各ウィンドウの性能評価
        metrics_history = []
        rolling_sharpe = []
        rolling_volatility = []

        for window in windows:
            metrics = self.evaluate_window_performance(returns, window)
            metrics_history.append(metrics)
            rolling_sharpe.append(metrics['sharpe_ratio'])
            rolling_volatility.append(metrics['volatility'])

        # 累積指標の計算
        cumulative_metrics = {}
        if metrics_history:
            for key in metrics_history[0].keys():
                values = [m[key] for m in metrics_history]
                cumulative_metrics[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }

        return WalkforwardResult(
            windows=windows,
            metrics_history=metrics_history,
            cumulative_metrics=cumulative_metrics,
            rolling_sharpe=rolling_sharpe,
            rolling_volatility=rolling_volatility
        )


def main() -> None:
    """メイン関数"""
    parser = argparse.ArgumentParser(description='ウォークフォワード分析')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='入力データファイル（CSV）')
    parser.add_argument('--output', '-o', type=str, default='output',
                       help='出力ディレクトリ')
    parser.add_argument('--initial-train-days', type=int, default=252,
                       help='初期訓練期間の日数')
    parser.add_argument('--validation-days', type=int, default=21,
                       help='検証期間の日数')
    parser.add_argument('--test-days', type=int, default=21,
                       help='テスト期間の日数')
    parser.add_argument('--step-days', type=int, default=21,
                       help='ウィンドウシフトの日数')

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    try:
        # データ読み込み
        data = pd.read_csv(args.input, index_col=0, parse_dates=True)
        if len(data.columns) == 1:
            returns = data.iloc[:, 0]
        else:
            # 最初の列を使用
            returns = data.iloc[:, 0]
        print(f"Loaded returns data with {len(returns)} observations")

    except Exception as e:
        print(f"Error loading returns data: {e}")
        return

    # 分析実行
    analyzer = WalkforwardAnalyzer(
        initial_train_days=args.initial_train_days,
        validation_days=args.validation_days,
        test_days=args.test_days,
        step_days=args.step_days
    )

    try:
        print("Running walkforward analysis...")
        result = analyzer.run_walkforward_analysis(returns)

        # 結果保存
        result_file = output_dir / 'walkforward_results.json'
        result_dict = {
            'windows': [
                {
                    'window_id': w.window_id,
                    'train_start': w.train_start.isoformat(),
                    'train_end': w.train_end.isoformat(),
                    'val_start': w.val_start.isoformat(),
                    'val_end': w.val_end.isoformat(),
                    'test_start': w.test_start.isoformat(),
                    'test_end': w.test_end.isoformat()
                } for w in result.windows
            ],
            'metrics_history': result.metrics_history,
            'cumulative_metrics': result.cumulative_metrics,
            'rolling_sharpe': result.rolling_sharpe,
            'rolling_volatility': result.rolling_volatility
        }

        import json
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {result_file}")

        # プロット生成
        if result.rolling_sharpe:
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 1, 1)
            plt.plot(result.rolling_sharpe, marker='o', linestyle='-')
            plt.title('Rolling Sharpe Ratio')
            plt.xlabel('Window')
            plt.ylabel('Sharpe Ratio')
            plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot(result.rolling_volatility, marker='s', linestyle='-', color='orange')
            plt.title('Rolling Volatility')
            plt.xlabel('Window')
            plt.ylabel('Volatility')
            plt.grid(True)

            plt.tight_layout()
            plot_file = output_dir / 'walkforward_analysis.png'
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {plot_file}")

    except Exception as e:
        print(f"Error during analysis: {e}")


if __name__ == '__main__':
    main()
