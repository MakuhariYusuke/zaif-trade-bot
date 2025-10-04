#!/usr/bin/env python3
"""
パフォーマンス帰属分析モジュール

戦略のリターンを様々な要因（市場、セクター、スタイルなど）に分解し、
各要因の貢献度を分析します。
"""

import argparse
import json
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Import BenchmarkMetrics from comprehensive_benchmark
try:
    from comprehensive_benchmark import BenchmarkMetrics
except ImportError:
    # Fallback if not available
    BenchmarkMetrics = None

@dataclass
class AttributionFactor:
    """帰属要因"""
    name: str
    returns: pd.Series
    weight: float
    contribution: float

@dataclass
class PerformanceAttributionResult:
    """パフォーマンス帰属分析結果"""
    total_return: float
    explained_return: float
    unexplained_return: float
    r_squared: float
    factors: List[AttributionFactor]
    factor_contributions: Dict[str, float]
    intercept: float
    residuals: pd.Series

@dataclass
class AttributionAnalysisResult:
    """包括的な帰属分析結果"""
    single_period: PerformanceAttributionResult
    multi_period: Optional[Dict[str, PerformanceAttributionResult]] = None
    rolling_attribution: Optional[pd.DataFrame] = None

class PerformanceAttributionAnalyzer:
    """パフォーマンス帰属分析クラス"""

    def __init__(self, annualization_factor: int = 252):
        """
        Args:
            annualization_factor: 年率化係数（日次データの場合252）
        """
        self.annualization_factor = annualization_factor

    def create_market_factors(self, returns: pd.Series,
                            market_returns: Optional[pd.Series] = None,
                            risk_free_rate: Optional[pd.Series] = None) -> Dict[str, pd.Series]:
        """
        標準的な市場要因を作成

        Args:
            returns: 戦略リターン
            market_returns: 市場リターン（オプション）
            risk_free_rate: 無リスク金利（オプション）

        Returns:
            要因辞書
        """
        factors = {}

        # 市場要因
        if market_returns is not None:
            factors['market'] = market_returns
        else:
            # 市場リターンを戦略リターンから推定（簡易版）
            factors['market'] = returns.rolling(20).mean()

        # サイズ要因（小 capitalization）
        factors['size'] = -returns.rolling(60).std()  # ボラティリティの逆数で近似

        # バリュー要因
        factors['value'] = returns.rolling(120).apply(lambda x: (x < 0).sum() / len(x))

        # モメンタム要因
        factors['momentum'] = returns.rolling(20).sum()

        # クオリティ要因（安定性）
        factors['quality'] = -returns.rolling(60).std()

        # ボラティリティ要因
        factors['volatility'] = returns.rolling(20).std()

        # 金利要因
        if risk_free_rate is not None:
            factors['interest_rate'] = risk_free_rate
        else:
            factors['interest_rate'] = pd.Series(0.02/252, index=returns.index)  # 2%年率

        return factors

    def single_period_attribution(self, strategy_returns: pd.Series,
                                factor_returns: Dict[str, pd.Series],
                                weights: Optional[Dict[str, float]] = None) -> PerformanceAttributionResult:
        """
        単一期間のパフォーマンス帰属分析

        Args:
            strategy_returns: 戦略リターン
            factor_returns: 要因リターン辞書
            weights: 要因の重み（オプション）

        Returns:
            帰属分析結果
        """
        # データの整合性チェック
        common_index = strategy_returns.index
        for factor_name, factor_ret in factor_returns.items():
            common_index = common_index.intersection(factor_ret.index)

        strategy_returns = strategy_returns.loc[common_index]
        factor_data = {}
        for factor_name, factor_ret in factor_returns.items():
            factor_data[factor_name] = factor_ret.loc[common_index]

        # 重みの設定
        if weights is None:
            weights = {name: 1.0 / len(factor_data) for name in factor_data.keys()}

        # 線形回帰による帰属分析
        X = pd.DataFrame(factor_data)
        y = strategy_returns.values

        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)

        # 予測値と残差
        y_pred = model.predict(X)
        residuals = y - y_pred

        # R²と総リターン
        r_squared = r2_score(y, y_pred)
        total_return = (1 + strategy_returns).prod() - 1
        explained_return = (1 + pd.Series(y_pred, index=strategy_returns.index)).prod() - 1
        unexplained_return = total_return - explained_return

        # 要因寄与度の計算
        factors = []
        factor_contributions = {}

        for i, factor_name in enumerate(X.columns):
            coefficient = model.coef_[i]
            avg_factor_return = X[factor_name].mean()
            weight = weights.get(factor_name, 1.0)

            # 寄与度 = 係数 × 平均要因リターン × 重み
            contribution = coefficient * avg_factor_return * weight

            factor_contributions[factor_name] = contribution

            factors.append(AttributionFactor(
                name=factor_name,
                returns=X[factor_name],
                weight=weight,
                contribution=contribution
            ))

        return PerformanceAttributionResult(
            total_return=float(total_return),
            explained_return=float(explained_return),
            unexplained_return=float(unexplained_return),
            r_squared=float(r_squared),
            factors=factors,
            factor_contributions=factor_contributions,
            intercept=float(model.intercept_),
            residuals=pd.Series(residuals, index=strategy_returns.index)
        )

    def multi_period_attribution(self, strategy_returns: pd.Series,
                               factor_returns: Dict[str, pd.Series],
                               window_size: int = 60) -> Dict[str, PerformanceAttributionResult]:
        """
        複数期間のパフォーマンス帰属分析

        Args:
            strategy_returns: 戦略リターン
            factor_returns: 要因リターン辞書
            window_size: 分析ウィンドウサイズ

        Returns:
            期間別帰属分析結果
        """
        results = {}

        for i in range(window_size, len(strategy_returns), window_size // 2):
            start_idx = i - window_size
            end_idx = i

            window_strategy = strategy_returns.iloc[start_idx:end_idx]
            window_factors = {}
            for name, returns in factor_returns.items():
                window_factors[name] = returns.iloc[start_idx:end_idx]

            period_name = f"{strategy_returns.index[start_idx].strftime('%Y-%m-%d')}_to_{strategy_returns.index[end_idx-1].strftime('%Y-%m-%d')}"

            try:
                result = self.single_period_attribution(window_strategy, window_factors)
                results[period_name] = result
            except Exception as e:
                warnings.warn(f"Failed to analyze period {period_name}: {e}")
                continue

        return results

    def rolling_attribution(self, strategy_returns: pd.Series,
                          factor_returns: Dict[str, pd.Series],
                          window_size: int = 60) -> pd.DataFrame:
        """
        ローリング帰属分析

        Args:
            strategy_returns: 戦略リターン
            factor_returns: 要因リターン辞書
            window_size: ローリングウィンドウサイズ

        Returns:
            ローリング帰属データフレーム
        """
        rolling_data = []

        for i in range(window_size, len(strategy_returns) + 1):
            window_strategy = strategy_returns.iloc[i-window_size:i]
            window_factors = {}
            for name, returns in factor_returns.items():
                window_factors[name] = returns.iloc[i-window_size:i]

            try:
                result = self.single_period_attribution(window_strategy, window_factors)

                row_data = {
                    'date': strategy_returns.index[i-1],
                    'total_return': result.total_return,
                    'explained_return': result.explained_return,
                    'unexplained_return': result.unexplained_return,
                    'r_squared': result.r_squared,
                    'intercept': result.intercept
                }

                # 要因寄与度を追加
                for factor_name, contribution in result.factor_contributions.items():
                    row_data[f'{factor_name}_contribution'] = contribution

                rolling_data.append(row_data)

            except Exception as e:
                warnings.warn(f"Failed rolling analysis at {strategy_returns.index[i-1]}: {e}")
                continue

        return pd.DataFrame(rolling_data).set_index('date')

    def run_performance_attribution_analysis(self, strategy_returns: pd.Series,
                                           factor_returns: Optional[Dict[str, pd.Series]] = None,
                                           market_returns: Optional[pd.Series] = None,
                                           risk_free_rate: Optional[pd.Series] = None,
                                           include_multi_period: bool = True,
                                           include_rolling: bool = True) -> AttributionAnalysisResult:
        """
        パフォーマンス帰属分析を実行

        Args:
            strategy_returns: 戦略リターン
            factor_returns: カスタム要因リターン（オプション）
            market_returns: 市場リターン（オプション）
            risk_free_rate: 無リスク金利（オプション）
            include_multi_period: 複数期間分析を含むか
            include_rolling: ローリング分析を含むか

        Returns:
            包括的な帰属分析結果
        """
        # 要因の作成
        if factor_returns is None:
            factor_returns = self.create_market_factors(
                strategy_returns, market_returns, risk_free_rate
            )

        # 単一期間分析
        single_period = self.single_period_attribution(strategy_returns, factor_returns)

        # 複数期間分析
        multi_period = None
        if include_multi_period:
            multi_period = self.multi_period_attribution(strategy_returns, factor_returns)

        # ローリング分析
        rolling_attribution = None
        if include_rolling:
            rolling_attribution = self.rolling_attribution(strategy_returns, factor_returns)

        return AttributionAnalysisResult(
            single_period=single_period,
            multi_period=multi_period,
            rolling_attribution=rolling_attribution
        )

    def analyze(self, metrics) -> Dict[str, Any]:
        """
        BenchmarkMetricsからパフォーマンス帰属分析を実行

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
                'attribution_score': 0.0,
                'factor_contributions': {},
                'r_squared': 0.0
            }

        # 日付インデックスを作成（仮定）
        dates = pd.date_range(start='2023-01-01', periods=len(metrics.returns), freq='D')
        strategy_returns = pd.Series(metrics.returns, index=dates)
        
        # NaN値を処理（LinearRegressionがNaNを処理できないため）
        if strategy_returns.isnull().any():
            strategy_returns = strategy_returns.fillna(strategy_returns.mean())

        try:
            # パフォーマンス帰属分析を実行
            result = self.run_performance_attribution_analysis(
                strategy_returns=strategy_returns,
                market_returns=None,  # 市場データがない場合はNone
                include_multi_period=True,
                include_rolling=True
            )

            # 結果を辞書形式に変換
            analysis_result: Dict[str, Any] = {
                'attribution_score': result.single_period.r_squared if result.single_period else 0.0,
                'factor_contributions': {},
                'r_squared': result.single_period.r_squared if result.single_period else 0.0,
                'total_return': result.single_period.total_return if result.single_period else 0.0,
                'explained_return': result.single_period.explained_return if result.single_period else 0.0,
                'unexplained_return': result.single_period.unexplained_return if result.single_period else 0.0,
                'intercept': result.single_period.intercept if result.single_period else 0.0,
                'factor_exposures': {},
                'rolling_r_squared': [],
                'multi_period_factors': {}
            }

            # 要因寄与度を追加
            if result.single_period and result.single_period.factors:
                for factor in result.single_period.factors:
                    analysis_result['factor_contributions'][factor.name] = factor.contribution

            # 要因の重みを追加
            if result.single_period and result.single_period.factor_contributions:
                for factor_name, contribution in result.single_period.factor_contributions.items():
                    analysis_result['factor_exposures'][factor_name] = contribution

            # ローリング分析結果を追加
            if result.rolling_attribution is not None:
                # DataFrameの場合
                if hasattr(result.rolling_attribution, 'r_squared'):
                    analysis_result['rolling_r_squared'] = result.rolling_attribution.r_squared.tolist() if hasattr(result.rolling_attribution.r_squared, 'tolist') else []

            # マルチ期間分析結果を追加
            if result.multi_period is not None:
                for period_name, period_result in result.multi_period.items():
                    if hasattr(period_result, 'factor_contributions'):
                        analysis_result['multi_period_factors'][period_name] = dict(period_result.factor_contributions)

            return analysis_result

        except Exception as e:
            return {
                'error': f'Analysis failed: {str(e)}',
                'attribution_score': 0.0,
                'factor_contributions': {},
                'r_squared': 0.0
            }

    def plot_attribution_analysis(self, result: AttributionAnalysisResult,
                                save_path: Optional[str] = None):
        """パフォーマンス帰属分析結果を可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Attribution Analysis', fontsize=16)

        # 要因寄与度の棒グラフ
        factor_names = list(result.single_period.factor_contributions.keys())
        contributions = list(result.single_period.factor_contributions.values())

        bars = axes[0, 0].barh(factor_names, contributions, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Factor Contributions')
        axes[0, 0].set_xlabel('Contribution')
        axes[0, 0].grid(True, alpha=0.3)

        # 値ラベルを追加
        for bar, value in zip(bars, contributions):
            axes[0, 0].text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                           f'{value:.4f}', ha='left', va='center', fontsize=8)

        # リターン分解
        labels = ['Explained Return', 'Unexplained Return']
        values = [result.single_period.explained_return, result.single_period.unexplained_return]
        colors = ['lightgreen', 'lightcoral']

        axes[0, 1].pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Return Decomposition')
        axes[0, 1].axis('equal')

        # 残差分析
        axes[1, 0].plot(result.single_period.residuals.index, result.single_period.residuals.values,
                       'b-', alpha=0.7)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Residual Analysis')
        axes[1, 0].set_ylabel('Residual Return')
        axes[1, 0].grid(True, alpha=0.3)

        # ローリングR²（利用可能な場合）
        if result.rolling_attribution is not None and not result.rolling_attribution.empty:
            axes[1, 1].plot(result.rolling_attribution.index, result.rolling_attribution['r_squared'],
                           'g-', linewidth=2, label='R²')
            axes[1, 1].set_title('Rolling R²')
            axes[1, 1].set_ylabel('R²')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'Rolling analysis not available',
                           ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def export_results(self, result: AttributionAnalysisResult,
                      output_path: str):
        """分析結果を JSON ファイルにエクスポート"""
        export_data: Dict[str, Any] = {
            'single_period': {
                'total_return': result.single_period.total_return,
                'explained_return': result.single_period.explained_return,
                'unexplained_return': result.single_period.unexplained_return,
                'r_squared': result.single_period.r_squared,
                'intercept': result.single_period.intercept,
                'factor_contributions': result.single_period.factor_contributions
            }
        }

        if result.multi_period:
            export_data['multi_period'] = {}
            for period_name, period_result in result.multi_period.items():
                export_data['multi_period'][period_name] = {
                    'total_return': period_result.total_return,
                    'explained_return': period_result.explained_return,
                    'unexplained_return': period_result.unexplained_return,
                    'r_squared': period_result.r_squared,
                    'intercept': period_result.intercept,
                    'factor_contributions': dict(period_result.factor_contributions)
                }

        if result.rolling_attribution is not None:
            export_data['rolling_attribution'] = result.rolling_attribution.to_dict('index')  # type: ignore

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"Results exported to {output_path}")

def main():
    """コマンドラインインターフェース"""
    parser = argparse.ArgumentParser(description='Performance Attribution Analysis')
    parser.add_argument('--returns-path', required=True, help='Path to strategy returns (CSV)')
    parser.add_argument('--market-returns-path', help='Path to market returns (CSV, optional)')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--no-multi-period', action='store_true', help='Skip multi-period analysis')
    parser.add_argument('--no-rolling', action='store_true', help='Skip rolling analysis')

    args = parser.parse_args()

    # 戦略リターンの読み込み
    strategy_df = pd.read_csv(args.returns_path)
    if 'returns' in strategy_df.columns:
        strategy_returns = strategy_df.set_index('date' if 'date' in strategy_df.columns else strategy_df.columns[0])['returns']
    else:
        strategy_returns = strategy_df.set_index(strategy_df.columns[0]).iloc[:, 0]

    if not isinstance(strategy_returns.index, pd.DatetimeIndex):
        strategy_returns.index = pd.to_datetime(strategy_returns.index)

    # 市場リターンの読み込み（オプション）
    market_returns = None
    if args.market_returns_path:
        market_df = pd.read_csv(args.market_returns_path)
        if 'returns' in market_df.columns:
            market_returns = market_df.set_index('date' if 'date' in market_df.columns else market_df.columns[0])['returns']
        else:
            market_returns = market_df.set_index(market_df.columns[0]).iloc[:, 0]

        if not isinstance(market_returns.index, pd.DatetimeIndex):
            market_returns.index = pd.to_datetime(market_returns.index)

    # 分析の実行
    analyzer = PerformanceAttributionAnalyzer()
    result = analyzer.run_performance_attribution_analysis(
        strategy_returns=strategy_returns,
        market_returns=market_returns,
        include_multi_period=not args.no_multi_period,
        include_rolling=not args.no_rolling
    )

    # 結果の保存
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # プロットの保存
    plot_path = os.path.join(args.output_dir, 'performance_attribution_analysis.png')
    analyzer.plot_attribution_analysis(result, save_path=plot_path)

    # JSON エクスポート
    json_path = os.path.join(args.output_dir, 'performance_attribution_results.json')
    analyzer.export_results(result, json_path)

    print("Performance attribution analysis completed!")

if __name__ == '__main__':
    main()