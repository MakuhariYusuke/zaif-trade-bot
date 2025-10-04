#!/usr/bin/env python3
"""
市場レジーム評価モジュール

市場をトレンド/レンジ/高ボラ/低ボラに分類し、
各レジームにおけるモデルの性能を比較します。
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
from numpy.typing import NDArray
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RegimeMetrics:
    """レジームごとのメトリクス"""
    sharpe_ratio: float
    win_rate: float
    total_return: float
    max_drawdown: float
    volatility: float
    trade_count: int

@dataclass
class RegimeAnalysisResult:
    """レジーム分析結果"""
    regime_labels: NDArray[np.int_]  # np.ndarray
    regime_counts: Dict[str, int]
    regime_metrics: Dict[str, Dict[str, RegimeMetrics]]
    regime_transitions: pd.DataFrame

class RegimeEvaluator:
    """市場レジーム評価クラス"""

    def __init__(self, volatility_window: int = 20, trend_window: int = 50):
        """
        Args:
            volatility_window: ボラティリティ計算の窓幅
            trend_window: トレンド検出の窓幅
        """
        self.volatility_window = volatility_window
        self.trend_window = trend_window

    def classify_market_regime(self, price_data: pd.DataFrame) -> Tuple[NDArray[np.int_], Dict[str, int]]:
        """
        市場レジームを分類

        Args:
            price_data: OHLCVデータ

        Returns:
            レジームラベル配列とカウント辞書
        """
        # 価格データの検証
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in price_data.columns for col in required_cols):
            raise ValueError(f"価格データに必要な列がありません: {required_cols}")

        # リターンの計算
        returns = price_data['close'].pct_change().fillna(0)

        # ボラティリティの計算（標準偏差）
        volatility = returns.rolling(window=self.volatility_window).std() * np.sqrt(252)  # 年率化

        # トレンド方向の計算（移動平均の傾き）
        ma_short = price_data['close'].rolling(window=self.trend_window//2).mean()
        ma_long = price_data['close'].rolling(window=self.trend_window).mean()
        trend_strength = (ma_short - ma_long) / ma_long

        # ボラティリティの閾値
        vol_median = volatility.median()
        vol_high_threshold = vol_median * 1.5
        vol_low_threshold = vol_median * 0.5

        # トレンド強度の閾値
        trend_threshold = 0.02  # 2%のトレンド

        # レジーム分類
        regime_labels = []

        for i in range(len(price_data)):
            vol = volatility.iloc[i] if i >= self.volatility_window else vol_median
            trend = abs(trend_strength.iloc[i]) if i >= self.trend_window else 0

            if trend > trend_threshold:
                regime = 'trend'
            elif vol > vol_high_threshold:
                regime = 'high_vol'
            elif vol < vol_low_threshold:
                regime = 'low_vol'
            else:
                regime = 'range'

            regime_labels.append(regime)

        regime_array = np.array(regime_labels)

        # レジームカウント
        unique, counts = np.unique(regime_array, return_counts=True)
        regime_counts = dict(zip(unique, counts))

        return regime_array, regime_counts

    def calculate_regime_metrics(self, returns: pd.Series, regime_labels: NDArray[np.int_],
                               regime: str) -> RegimeMetrics:
        """
        指定レジームのメトリクスを計算

        Args:
            returns: リターン系列
            regime_labels: レジームラベル配列
            regime: 対象レジーム

        Returns:
            レジームメトリクス
        """
        # 指定レジームのデータを抽出
        regime_mask = regime_labels == regime
        regime_returns = returns[regime_mask]

        if len(regime_returns) == 0:
            return RegimeMetrics(0, 0, 0, 0, 0, 0)

        # 基本メトリクス計算
        regime_returns_array = regime_returns.values
        cumulative_return = float(np.prod(1 + regime_returns_array) - 1)  # type: ignore
        volatility = float(np.std(regime_returns_array) * np.sqrt(252))  # type: ignore

        # Sharpe比率
        risk_free_rate = 0.02  # 仮定の無リスク金利
        sharpe_ratio = (regime_returns.mean() - risk_free_rate/252) / volatility if volatility > 0 else 0

        # 勝率（正のリターンの割合）
        win_rate = (regime_returns > 0).mean()

        # 最大ドローダウン
        cumulative = (1 + regime_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return RegimeMetrics(
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            total_return=cumulative_return,
            max_drawdown=max_drawdown,
            volatility=volatility,
            trade_count=len(regime_returns)
        )

    def analyze_regime_performance(self, models: Dict[str, str], price_data: pd.DataFrame,
                                 regime_labels: NDArray[np.int_]) -> RegimeAnalysisResult:
        """
        全モデルのレジーム別性能を分析

        Args:
            models: モデル名 -> パス の辞書
            price_data: 価格データ
            regime_labels: レジームラベル

        Returns:
            レジーム分析結果
        """
        # リターンの計算（簡易版：価格変化をリターンとする）
        returns = price_data['close'].pct_change().fillna(0)

        # レジームカウント
        unique, counts = np.unique(regime_labels, return_counts=True)
        regime_counts = dict(zip(unique, counts))

        # 各モデルのレジーム別メトリクス
        regime_metrics = {}

        for model_name in models.keys():
            model_regime_metrics = {}

            for regime in ['trend', 'range', 'high_vol', 'low_vol']:
                metrics = self.calculate_regime_metrics(returns, regime_labels, regime)
                model_regime_metrics[regime] = metrics

            regime_metrics[model_name] = model_regime_metrics

        # レジーム遷移マトリックス
        regime_transitions = self._calculate_regime_transitions(regime_labels)

        return RegimeAnalysisResult(
            regime_labels=regime_labels,
            regime_counts=regime_counts,
            regime_metrics=regime_metrics,
            regime_transitions=regime_transitions
        )

    def _calculate_regime_transitions(self, regime_labels: NDArray[np.int_]) -> pd.DataFrame:
        """レジーム遷移マトリックスを計算"""
        regimes = ['trend', 'range', 'high_vol', 'low_vol']
        transition_matrix = pd.DataFrame(0, index=regimes, columns=regimes)

        for i in range(1, len(regime_labels)):
            from_regime = regime_labels[i-1]
            to_regime = regime_labels[i]
            if from_regime in regimes and to_regime in regimes:
                transition_matrix.loc[from_regime, to_regime] += 1

        # 確率に変換
        transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)

        return transition_matrix

    def plot_regime_analysis(self, result: RegimeAnalysisResult, save_path: Optional[str] = None):
        """レジーム分析結果を可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Market Regime Analysis', fontsize=16)

        # レジーム分布
        regimes = list(result.regime_counts.keys())
        counts = list(result.regime_counts.values())

        axes[0, 0].bar(regimes, counts, color='skyblue')
        axes[0, 0].set_title('Regime Distribution')
        axes[0, 0].set_ylabel('Count')

        # Sharpe比率比較
        model_names = list(result.regime_metrics.keys())
        regimes = ['trend', 'range', 'high_vol', 'low_vol']

        sharpe_data = []
        for model in model_names:
            sharpe_row = [result.regime_metrics[model][regime].sharpe_ratio for regime in regimes]
            sharpe_data.append(sharpe_row)

        sharpe_df = pd.DataFrame(sharpe_data, columns=regimes, index=model_names)
        sharpe_df.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Sharpe Ratio by Regime')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 勝率比較
        winrate_data = []
        for model in model_names:
            winrate_row = [result.regime_metrics[model][regime].win_rate for regime in regimes]
            winrate_data.append(winrate_row)

        winrate_df = pd.DataFrame(winrate_data, columns=regimes, index=model_names)
        winrate_df.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Win Rate by Regime')
        axes[1, 0].set_ylabel('Win Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # レジーム遷移ヒートマップ
        sns.heatmap(result.regime_transitions, annot=True, fmt='.2f', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title('Regime Transition Matrix')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Regime analysis plot saved to {save_path}")

        plt.show()

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='Market Regime Evaluation')
    parser.add_argument('--models', nargs='+', required=True,
                       help='Model paths (name:path format, e.g., single:model.zip ensemble:ensemble.zip)')
    parser.add_argument('--price-data', required=True,
                       help='Path to price data CSV file')
    parser.add_argument('--output-dir', default='results/regime_analysis',
                       help='Output directory for results')
    parser.add_argument('--volatility-window', type=int, default=20,
                       help='Window size for volatility calculation')
    parser.add_argument('--trend-window', type=int, default=50,
                       help='Window size for trend detection')

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # モデル辞書の作成
    models = {}
    for model_spec in args.models:
        if ':' in model_spec:
            name, path = model_spec.split(':', 1)
            models[name] = path
        else:
            name = Path(model_spec).stem
            models[name] = model_spec

    # 価格データの読み込み
    try:
        price_data = pd.read_csv(args.price_data, index_col=0, parse_dates=True)
        print(f"Loaded price data with {len(price_data)} rows")
    except Exception as e:
        print(f"Error loading price data: {e}")
        return

    # レジーム評価器の初期化
    evaluator = RegimeEvaluator(
        volatility_window=args.volatility_window,
        trend_window=args.trend_window
    )

    # レジーム分類
    print("Classifying market regimes...")
    regime_labels, regime_counts = evaluator.classify_market_regime(price_data)
    print(f"Regime distribution: {regime_counts}")

    # レジーム別性能分析
    print("Analyzing regime-specific performance...")
    result = evaluator.analyze_regime_performance(models, price_data, regime_labels)

    # 結果の保存
    result_file = output_dir / 'regime_analysis_results.json'

    # 結果を辞書に変換して保存
    result_dict = {
        'regime_counts': result.regime_counts,
        'regime_metrics': {}
    }

    for model_name, metrics in result.regime_metrics.items():
        result_dict['regime_metrics'][model_name] = {}  # type: ignore
        for regime, metric in metrics.items():
            result_dict['regime_metrics'][model_name][regime] = {  # type: ignore
                'sharpe_ratio': metric.sharpe_ratio,
                'win_rate': metric.win_rate,
                'total_return': metric.total_return,
                'max_drawdown': metric.max_drawdown,
                'volatility': metric.volatility,
                'trade_count': metric.trade_count
            }

    # JSONとして保存
    import json
    with open(result_file, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)

    print(f"Results saved to {result_file}")

    # 可視化
    plot_file = output_dir / 'regime_analysis.png'
    evaluator.plot_regime_analysis(result, str(plot_file))

if __name__ == '__main__':
    main()