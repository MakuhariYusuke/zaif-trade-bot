#!/usr/bin/env python3
"""
リスクパリティ分析モジュール

ポートフォリオのリスクを均等に分配し、最適な資産配分を求める分析を行います。
"""

import argparse
import json
from typing import Dict, Optional, Any
from numpy.typing import NDArray
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import minimize

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import minimize

# Import BenchmarkMetrics from comprehensive_benchmark
try:
    from comprehensive_benchmark import BenchmarkMetrics
except ImportError:
    # Fallback if not available
    BenchmarkMetrics = None

@dataclass
class RiskParityResult:
    """リスクパリティ分析結果"""
    optimal_weights: pd.Series
    risk_contributions: pd.Series
    portfolio_volatility: float
    diversification_ratio: float
    risk_parity_score: float
    concentration_metrics: Dict[str, float]

@dataclass
class RiskParityOptimization:
    """リスクパリティ最適化結果"""
    equal_risk_weights: pd.Series
    minimum_variance_weights: pd.Series
    maximum_diversification_weights: pd.Series
    risk_parity_weights: pd.Series

@dataclass
class RiskParityAnalysisResult:
    """包括的なリスクパリティ分析結果"""
    current_allocation: Optional[Dict[str, float]] = None
    risk_parity_result: Optional[RiskParityResult] = None
    optimization_results: Optional[RiskParityOptimization] = None
    rebalancing_suggestions: Optional[Dict[str, float]] = None

class RiskParityAnalyzer:
    """リスクパリティ分析クラス"""

    def __init__(self, risk_measure: str = 'volatility', target_risk_level: Optional[float] = None):
        """
        Args:
            risk_measure: リスク尺度 ('volatility', 'var', 'cvar')
            target_risk_level: 目標リスクレベル（オプション）
        """
        super().__init__()
        self.risk_measure = risk_measure
        self.target_risk_level = target_risk_level

    def calculate_covariance_matrix(self, returns: pd.DataFrame,
                                  method: str = 'sample') -> pd.DataFrame:
        """
        共分散行列の計算

        Args:
            returns: リターン系列
            method: 計算方法 ('sample', 'ewma', 'shrinkage')

        Returns:
            共分散行列
        """
        if method == 'sample':
            cov_matrix = returns.cov()
        elif method == 'ewma':
            # EWMA (指数加重移動平均)
            lambda_param = 0.94
            weights = np.array([(1 - lambda_param) * (lambda_param ** i)
                              for i in range(len(returns))])
            weights = weights[::-1]  # 最新のデータに重み付け

            # 加重共分散
            weighted_returns = returns * np.sqrt(weights[:, np.newaxis])
            cov_matrix = weighted_returns.T @ weighted_returns / weights.sum()
        else:
            # デフォルトはサンプル共分散
            cov_matrix = returns.cov()

        return cov_matrix

    def calculate_risk_contribution(self, weights: NDArray[np.floating],
                                    cov_matrix: pd.DataFrame) -> NDArray[np.floating]:
        """
        各資産のリスク寄与を計算

        Args:
            weights: 資産配分ウェイト
            cov_matrix: 共分散行列

        Returns:
            リスク寄与ベクトル
        """
        portfolio_volatility = np.sqrt(weights.T @ cov_matrix.values @ weights)

        if portfolio_volatility == 0:
            return np.zeros(len(weights))

        # 各資産のリスク寄与（パーシャル分散）
        marginal_contributions = cov_matrix.values @ weights
        risk_contributions = weights * marginal_contributions / portfolio_volatility

        return risk_contributions

    def calculate_portfolio_volatility(self, weights: NDArray[np.floating],
                                       cov_matrix: pd.DataFrame) -> float:
        """
        ポートフォリオのボラティリティを計算

        Args:
            weights: 資産配分ウェイト
            cov_matrix: 共分散行列

        Returns:
            ポートフォリオボラティリティ
        """
        return float(np.sqrt(weights.T @ cov_matrix.values @ weights))

    def calculate_diversification_ratio(self, weights: NDArray[np.floating],
                                        cov_matrix: pd.DataFrame) -> float:
        """
        分散化比率の計算

        Args:
            weights: 資産配分ウェイト
            cov_matrix: 共分散行列

        Returns:
            分散化比率
        """
        portfolio_vol = self.calculate_portfolio_volatility(weights, cov_matrix)

        if portfolio_vol == 0:
            return 1.0

        # 加重平均ボラティリティ
        individual_vols = np.sqrt(np.diag(cov_matrix.values))
        weighted_avg_vol = weights @ individual_vols

        return float(weighted_avg_vol / portfolio_vol)

    def calculate_risk_parity_score(self, risk_contributions: NDArray[np.floating]) -> float:
        """
        リスクパリティスコアの計算

        Args:
            risk_contributions: リスク寄与ベクトル

        Returns:
            リスクパリティスコア（0-1、1が完全パリティ）
        """
        if len(risk_contributions) <= 1:
            return 1.0

        # 理想的なリスク寄与（等分）
        target_contribution = 1.0 / len(risk_contributions)

        # 実際の寄与との差のRMS
        deviations = risk_contributions - target_contribution
        rms_deviation = np.sqrt(np.mean(deviations ** 2))

        # スコアの計算（1 - 正規化された偏差）
        max_possible_deviation = np.sqrt(target_contribution * (1 - target_contribution))
        score = 1.0 - (rms_deviation / max_possible_deviation)

        return max(0.0, min(1.0, score))

    def calculate_concentration_metrics(self, weights: NDArray[np.floating]) -> Dict[str, float]:
        """
        集中度指標の計算

        Args:
            weights: 資産配分ウェイト

        Returns:
            集中度指標
        """
        weights = np.abs(weights)  # 絶対値

        # Herfindahl-Hirschman Index (HHI)
        hhi = float(np.sum(weights ** 2))

        # 最大ウェイト
        max_weight = float(np.max(weights))

        # 上位3資産の集中度
        sorted_weights = np.sort(weights)[::-1]
        top3_concentration = float(np.sum(sorted_weights[:3]))

        # エントロピー（分散度の逆指標）
        non_zero_weights = weights[weights > 1e-8]
        if len(non_zero_weights) > 0:
            entropy = -np.sum(non_zero_weights * np.log(non_zero_weights))
            normalized_entropy = entropy / np.log(len(non_zero_weights))
        else:
            normalized_entropy = 0.0

        return {
            'hhi': hhi,
            'max_weight': max_weight,
            'top3_concentration': top3_concentration,
            'normalized_entropy': float(normalized_entropy)
        }

    def optimize_risk_parity(self, cov_matrix: pd.DataFrame,
                           initial_weights: Optional[NDArray[np.floating]] = None) -> RiskParityResult:
        """
        リスクパリティ最適化

        Args:
            cov_matrix: 共分散行列
            initial_weights: 初期ウェイト（オプション）

        Returns:
            最適化結果
        """
        n_assets = len(cov_matrix)

        if initial_weights is None:
            initial_weights = np.ones(n_assets) / n_assets

        # 制約条件：ウェイトの合計 = 1, 非負制約
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 合計 = 1
        ]

        bounds = [(0, 1) for _ in range(n_assets)]  # 非負制約

        def risk_parity_objective(weights: NDArray[np.floating]) -> float:
            """リスクパリティ目的関数"""
            risk_contribs = self.calculate_risk_contribution(weights, cov_matrix)

            if len(risk_contribs) <= 1:
                return 0.0

            # 目的関数：リスク寄与の分散を最小化
            return float(np.var(risk_contribs))

        # 最適化実行
        try:
            result = minimize(
                risk_parity_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )

            optimal_weights = result.x

            # 結果の計算
            risk_contributions = self.calculate_risk_contribution(optimal_weights, cov_matrix)
            portfolio_volatility = self.calculate_portfolio_volatility(optimal_weights, cov_matrix)
            diversification_ratio = self.calculate_diversification_ratio(optimal_weights, cov_matrix)
            risk_parity_score = self.calculate_risk_parity_score(risk_contributions)
            concentration_metrics = self.calculate_concentration_metrics(optimal_weights)

            return RiskParityResult(
                optimal_weights=pd.Series(optimal_weights, index=cov_matrix.index),
                risk_contributions=pd.Series(risk_contributions, index=cov_matrix.index),
                portfolio_volatility=portfolio_volatility,
                diversification_ratio=diversification_ratio,
                risk_parity_score=risk_parity_score,
                concentration_metrics=concentration_metrics
            )

        except Exception as e:
            warnings.warn(f"Risk parity optimization failed: {e}")
            # フォールバック：等ウェイト
            equal_weights = np.ones(n_assets) / n_assets
            risk_contributions = self.calculate_risk_contribution(equal_weights, cov_matrix)
            portfolio_volatility = self.calculate_portfolio_volatility(equal_weights, cov_matrix)
            diversification_ratio = self.calculate_diversification_ratio(equal_weights, cov_matrix)
            risk_parity_score = self.calculate_risk_parity_score(risk_contributions)
            concentration_metrics = self.calculate_concentration_metrics(equal_weights)

            return RiskParityResult(
                optimal_weights=pd.Series(equal_weights, index=cov_matrix.index),
                risk_contributions=pd.Series(risk_contributions, index=cov_matrix.index),
                portfolio_volatility=portfolio_volatility,
                diversification_ratio=diversification_ratio,
                risk_parity_score=risk_parity_score,
                concentration_metrics=concentration_metrics
            )

    def optimize_minimum_variance(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """
        最小分散ポートフォリオの最適化

        Args:
            cov_matrix: 共分散行列

        Returns:
            最適ウェイト
        """
        n_assets = len(cov_matrix)

        def objective(weights: NDArray[np.floating]) -> float:
            return float(weights.T @ cov_matrix.values @ weights)

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]

        bounds = [(0, 1) for _ in range(n_assets)]

        try:
            result = minimize(
                objective,
                np.ones(n_assets) / n_assets,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            return pd.Series(result.x, index=cov_matrix.index)
        except Exception:
            return pd.Series(np.ones(n_assets) / n_assets, index=cov_matrix.index)

    def optimize_maximum_diversification(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """
        最大分散化ポートフォリオの最適化

        Args:
            cov_matrix: 共分散行列

        Returns:
            最適ウェイト
        """
        individual_vols = np.sqrt(np.diag(cov_matrix.values))

        def objective(weights: NDArray[np.floating]) -> float:
            portfolio_vol = np.sqrt(weights.T @ cov_matrix.values @ weights)
            weighted_avg_vol = weights @ individual_vols
            return float(-weighted_avg_vol / portfolio_vol)  # 最大化なので符号反転

        n_assets = len(cov_matrix)
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]

        bounds = [(0, 1) for _ in range(n_assets)]

        try:
            result = minimize(
                objective,
                np.ones(n_assets) / n_assets,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            return pd.Series(result.x, index=cov_matrix.index)
        except Exception:
            return pd.Series(np.ones(n_assets) / n_assets, index=cov_matrix.index)

    def run_portfolio_optimization_comparison(self, cov_matrix: pd.DataFrame) -> RiskParityOptimization:
        """
        様々なポートフォリオ最適化手法の比較

        Args:
            cov_matrix: 共分散行列

        Returns:
            最適化手法比較結果
        """
        n_assets = len(cov_matrix)

        # 等リスク寄与ウェイト（リスクパリティ）
        risk_parity_result = self.optimize_risk_parity(cov_matrix)
        risk_parity_weights = risk_parity_result.optimal_weights

        # 等ウェイト
        equal_risk_weights = pd.Series(np.ones(n_assets) / n_assets, index=cov_matrix.index)

        # 最小分散
        minimum_variance_weights = self.optimize_minimum_variance(cov_matrix)

        # 最大分散化
        maximum_diversification_weights = self.optimize_maximum_diversification(cov_matrix)

        return RiskParityOptimization(
            equal_risk_weights=equal_risk_weights,
            minimum_variance_weights=minimum_variance_weights,
            maximum_diversification_weights=maximum_diversification_weights,
            risk_parity_weights=risk_parity_weights
        )

    def generate_rebalancing_suggestions(self, current_weights: pd.Series,
                                       optimal_weights: pd.Series,
                                       threshold: float = 0.05) -> Dict[str, Any]:
        """
        リバランス提案の生成

        Args:
            current_weights: 現在のウェイト
            optimal_weights: 最適ウェイト
            threshold: リバランス閾値

        Returns:
            リバランス提案
        """
        weight_differences = optimal_weights - current_weights
        significant_changes = weight_differences.abs() > threshold

        suggestions = {
            'assets_to_increase': weight_differences[weight_differences > threshold].index.tolist(),
            'assets_to_decrease': weight_differences[weight_differences < -threshold].index.tolist(),
            'no_change_assets': weight_differences[~significant_changes].index.tolist(),
            'max_change': float(weight_differences.abs().max()),
            'total_trades_needed': int(significant_changes.sum()),
            'trade_sizes': weight_differences[significant_changes].to_dict()
        }

        return suggestions

    def run_comprehensive_risk_parity_analysis(self, returns: pd.DataFrame,
                                             current_allocation: Optional[Dict[str, float]] = None,
                                             cov_method: str = 'sample') -> RiskParityAnalysisResult:
        """
        包括的なリスクパリティ分析を実行

        Args:
            returns: 資産リターン系列
            current_allocation: 現在の資産配分（オプション）
            cov_method: 共分散計算方法

        Returns:
            包括的な分析結果
        """
        # 共分散行列の計算
        cov_matrix = self.calculate_covariance_matrix(returns, method=cov_method)

        # リスクパリティ最適化
        risk_parity_result = self.optimize_risk_parity(cov_matrix)

        # 最適化手法比較
        optimization_results = self.run_portfolio_optimization_comparison(cov_matrix)

        # リバランス提案（現在の配分がある場合）
        rebalancing_suggestions = None
        if current_allocation is not None:
            current_weights = pd.Series(current_allocation).reindex(cov_matrix.index).fillna(0)
            rebalancing_suggestions = self.generate_rebalancing_suggestions(
                current_weights, risk_parity_result.optimal_weights
            )

        return RiskParityAnalysisResult(
            current_allocation=current_allocation,
            risk_parity_result=risk_parity_result,
            optimization_results=optimization_results,
            rebalancing_suggestions=rebalancing_suggestions
        )

    def analyze(self, metrics) -> Dict[str, Any]:
        """
        BenchmarkMetricsからリスクパリティ分析を実行

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
                'risk_parity_score': 0.0,
                'diversification_ratio': 0.0,
                'concentration_index': 0.0
            }

        # 日付インデックスを作成（仮定）
        dates = pd.date_range(start='2023-01-01', periods=len(metrics.returns), freq='D')
        returns = pd.Series(metrics.returns, index=dates)

        try:
            # 簡易的なリスクパリティ分析
            # 単一資産の場合のリスクパリティ指標を計算
            volatility = returns.std()
            total_return = (1 + returns).prod() - 1

            # リスクパリティスコア（安定性とリターンのバランス）
            if volatility > 0:
                risk_adjusted_return = total_return / volatility
                risk_parity_score = min(1.0, max(0.0, risk_adjusted_return / 2.0 + 0.5))  # 0-1に正規化
            else:
                risk_parity_score = 0.5

            # 分散化比率（単一資産なので1.0）
            diversification_ratio = 1.0

            # 集中度指標（低いほど分散されている）
            concentration_index = 1.0  # 単一資産なので最大集中

            # シャープレシオ
            sharpe_ratio = metrics.sharpe_ratio if hasattr(metrics, 'sharpe_ratio') and metrics.sharpe_ratio != 0 else total_return / volatility if volatility > 0 else 0.0

            # ソルティノレシオ（下落リスクのみ考慮）
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() if len(downside_returns) > 0 else 0.0
            sortino_ratio = total_return / downside_volatility if downside_volatility > 0 else 0.0

            return {
                'risk_parity_score': float(risk_parity_score),
                'diversification_ratio': float(diversification_ratio),
                'concentration_index': float(concentration_index),
                'volatility': float(volatility),
                'total_return': float(total_return),
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'downside_volatility': float(downside_volatility),
                'win_rate': float((returns > 0).mean()),
                'loss_rate': float((returns < 0).mean()),
                'avg_win': float(returns[returns > 0].mean()) if len(returns[returns > 0]) > 0 else 0.0,
                'avg_loss': float(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0.0
            }

        except Exception as e:
            return {
                'error': f'Analysis failed: {str(e)}',
                'risk_parity_score': 0.0,
                'diversification_ratio': 0.0,
                'concentration_index': 0.0
            }

    def plot_risk_parity_analysis(self, result: RiskParityAnalysisResult,
                                save_path: Optional[str] = None):
        """リスクパリティ分析結果を可視化"""
        if not result.risk_parity_result or not result.optimization_results:
            print("No risk parity results available for plotting")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Risk Parity Analysis Results', fontsize=16)

        # リスク寄与
        ax = axes[0, 0]
        risk_contribs = result.risk_parity_result.risk_contributions
        bars = ax.bar(range(len(risk_contribs)), risk_contribs, color='skyblue', alpha=0.7)
        ax.set_title('Risk Contributions (Risk Parity)')
        ax.set_xlabel('Assets')
        ax.set_ylabel('Risk Contribution')
        ax.set_xticks(range(len(risk_contribs)))
        ax.set_xticklabels(risk_contribs.index, rotation=45)

        # 値ラベルを追加
        for bar, contrib_value in zip(bars, risk_contribs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{contrib_value:.3f}', ha='center', va='bottom', fontsize=8)

        # 最適ウェイト
        ax = axes[0, 1]
        weights = result.risk_parity_result.optimal_weights
        bars = ax.bar(range(len(weights)), weights, color='lightgreen', alpha=0.7)
        ax.set_title('Optimal Weights (Risk Parity)')
        ax.set_xlabel('Assets')
        ax.set_ylabel('Weight')
        ax.set_xticks(range(len(weights)))
        ax.set_xticklabels(weights.index, rotation=45)

        # 最適化手法比較
        ax = axes[0, 2]
        methods = ['Equal Risk', 'Min Variance', 'Max Divers', 'Risk Parity']
        portfolios = [
            result.optimization_results.equal_risk_weights,
            result.optimization_results.minimum_variance_weights,
            result.optimization_results.maximum_diversification_weights,
            result.optimization_results.risk_parity_weights
        ]

        for i, (method, weights) in enumerate(zip(methods, portfolios)):
            ax.scatter(weights, [i] * len(weights), label=method, alpha=0.7, s=50)

        ax.set_title('Portfolio Optimization Comparison')
        ax.set_xlabel('Weight')
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.legend()

        # 集中度指標
        ax = axes[1, 0]
        metrics = result.risk_parity_result.concentration_metrics
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        bars = ax.bar(metric_names, metric_values, color='salmon', alpha=0.7)
        ax.set_title('Concentration Metrics')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)

        # 主要指標
        ax = axes[1, 1]
        key_metrics = {
            'Portfolio Vol': result.risk_parity_result.portfolio_volatility,
            'Diversification Ratio': result.risk_parity_result.diversification_ratio,
            'Risk Parity Score': result.risk_parity_result.risk_parity_score
        }

        bars = ax.bar(key_metrics.keys(), key_metrics.values(), color='purple', alpha=0.7)
        ax.set_title('Key Portfolio Metrics')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)

        # リバランス提案（利用可能な場合）
        ax = axes[1, 2]
        if result.rebalancing_suggestions:
            suggestions = result.rebalancing_suggestions
            categories = ['To Increase', 'To Decrease', 'No Change']
            counts = [
                len(suggestions['assets_to_increase']),
                len(suggestions['assets_to_decrease']),
                len(suggestions['no_change_assets'])
            ]

            bars = ax.bar(categories, counts, color=['green', 'red', 'blue'], alpha=0.7)
            ax.set_title('Rebalancing Suggestions')
            ax.set_ylabel('Number of Assets')

            # 値ラベルを追加
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       str(count), ha='center', va='bottom', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No current\nallocation\nprovided',
                   ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def export_results(self, result: RiskParityAnalysisResult,
                      output_path: str):
        """分析結果を JSON ファイルにエクスポート"""
        export_data = {}

        if result.current_allocation:
            export_data['current_allocation'] = result.current_allocation

        if result.risk_parity_result:
            export_data['risk_parity_result'] = {
                'optimal_weights': result.risk_parity_result.optimal_weights.to_dict(),
                'risk_contributions': result.risk_parity_result.risk_contributions.to_dict(),
                'portfolio_volatility': result.risk_parity_result.portfolio_volatility,
                'diversification_ratio': result.risk_parity_result.diversification_ratio,
                'risk_parity_score': result.risk_parity_result.risk_parity_score,
                'concentration_metrics': result.risk_parity_result.concentration_metrics
            }

        if result.optimization_results:
            export_data['optimization_results'] = {
                'equal_risk_weights': result.optimization_results.equal_risk_weights.to_dict(),
                'minimum_variance_weights': result.optimization_results.minimum_variance_weights.to_dict(),
                'maximum_diversification_weights': result.optimization_results.maximum_diversification_weights.to_dict(),
                'risk_parity_weights': result.optimization_results.risk_parity_weights.to_dict()
            }

        if result.rebalancing_suggestions:
            export_data['rebalancing_suggestions'] = result.rebalancing_suggestions

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"Results exported to {output_path}")

def main():
    """コマンドラインインターフェース"""
    parser = argparse.ArgumentParser(description='Risk Parity Analysis')
    parser.add_argument('--returns-data', required=True, help='Path to asset returns (CSV)')
    parser.add_argument('--current-allocation', help='Path to current allocation (JSON)')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--cov-method', default='sample', choices=['sample', 'ewma'], help='Covariance calculation method')

    args = parser.parse_args()

    # リターンデータの読み込み
    returns = pd.read_csv(args.returns_data)
    if 'date' in returns.columns:
        returns['date'] = pd.to_datetime(returns['date'])
        returns = returns.set_index('date')

    # 現在の配分の読み込み
    current_allocation = None
    if args.current_allocation:
        with open(args.current_allocation, 'r') as f:
            current_allocation = json.load(f)

    # 分析の実行
    analyzer = RiskParityAnalyzer()
    result = analyzer.run_comprehensive_risk_parity_analysis(
        returns=returns,
        current_allocation=current_allocation,
        cov_method=args.cov_method
    )

    # 結果の保存
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # プロットの保存
    plot_path = os.path.join(args.output_dir, 'risk_parity_analysis.png')
    analyzer.plot_risk_parity_analysis(result, save_path=plot_path)

    # JSON エクスポート
    json_path = os.path.join(args.output_dir, 'risk_parity_results.json')
    analyzer.export_results(result, json_path)

    print("Risk parity analysis completed!")

if __name__ == '__main__':
    main()