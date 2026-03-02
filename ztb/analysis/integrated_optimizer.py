"""
Phase 3-2: パラメータ最適化 - 統合最適化システム

ウォークフォワード分析、Kelly基準、ATRリスク管理、動的信頼度調整を統合した
完全なパラメータ最適化システムを実装します。
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

from ztb.analysis.adaptive_confidence_adjuster import AdaptiveConfidenceAdjuster
from ztb.analysis.atr_risk_manager import ATRRiskManager, RiskManagementMode
from ztb.analysis.kelly_position_sizer import KellyParameters, KellyPositionSizer
from ztb.analysis.walk_forward_analyzer import (
    OptimizationResult,
    ParameterSet,
    WalkForwardAnalyzer,
)
from ztb.io.json_io import read_json, write_json
from ztb.utils.performance_profiler import PerformanceProfiler

@dataclass
class IntegratedOptimizationConfig:
    """統合最適化設定"""

    # ウォークフォワード設定
    train_days: int = 90
    test_days: int = 30
    step_days: int = 15
    min_samples: int = 30

    # Kelly基準設定
    min_trades_for_kelly: int = 10
    kelly_risk_tolerance: str = "half"  # "full", "half", "quarter"
    max_position_size: float = 0.05  # 5%

    # ATRリスク管理設定
    atr_period: int = 14
    risk_management_mode: RiskManagementMode = RiskManagementMode.DYNAMIC

    # 信頼度調整設定
    base_confidence_threshold: float = 0.7
    adaptive_thresholds_enabled: bool = True

    # 最適化設定
    optimization_target: str = (
        "sharpe_ratio"  # "sharpe_ratio", "total_return", "win_rate"
    )
    min_optimization_trades: int = 100

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "train_days": self.train_days,
            "test_days": self.test_days,
            "step_days": self.step_days,
            "min_trades_for_kelly": self.min_trades_for_kelly,
            "kelly_risk_tolerance": self.kelly_risk_tolerance,
            "max_position_size": self.max_position_size,
            "atr_period": self.atr_period,
            "risk_management_mode": self.risk_management_mode.value,
            "base_confidence_threshold": self.base_confidence_threshold,
            "adaptive_thresholds_enabled": self.adaptive_thresholds_enabled,
            "optimization_target": self.optimization_target,
            "min_optimization_trades": self.min_optimization_trades,
        }

@dataclass
class IntegratedOptimizationResult:
    """統合最適化結果"""

    walk_forward_results: list[OptimizationResult]
    optimal_parameters: ParameterSet
    kelly_parameters: KellyParameters
    performance_summary: dict[str, Any]
    regime_analysis: dict[str, Any]
    optimization_timestamp: datetime
    config_used: IntegratedOptimizationConfig

    @property
    def average_sharpe_ratio(self) -> float:
        """平均Sharpe Ratio"""
        sharpes = [
            r.out_of_sample_performance.get("sharpe_ratio", 0)
            for r in self.walk_forward_results
        ]
        return np.mean(sharpes) if sharpes else 0.0

    @property
    def average_win_rate(self) -> float:
        """平均勝率"""
        win_rates = [
            r.out_of_sample_performance.get("win_rate", 0)
            for r in self.walk_forward_results
        ]
        return np.mean(win_rates) if win_rates else 0.0

    @property
    def total_return(self) -> float:
        """総リターン"""
        returns = [
            r.out_of_sample_performance.get("total_return", 0)
            for r in self.walk_forward_results
        ]
        # 複利計算
        if returns:
            cumulative_return = 1.0
            for r in returns:
                cumulative_return *= 1 + r
            return cumulative_return - 1
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "optimal_parameters": self.optimal_parameters.to_dict(),
            "kelly_parameters": {
                "win_rate": self.kelly_parameters.win_rate,
                "win_loss_ratio": self.kelly_parameters.win_loss_ratio,
                "kelly_fraction": self.kelly_parameters.kelly_fraction,
                "total_trades": self.kelly_parameters.total_trades,
            },
            "performance_summary": self.performance_summary,
            "regime_analysis": self.regime_analysis,
            "average_sharpe_ratio": self.average_sharpe_ratio,
            "average_win_rate": self.average_win_rate,
            "total_return": self.total_return,
            "optimization_timestamp": self.optimization_timestamp.isoformat(),
            "config_used": self.config_used.to_dict(),
        }

class IntegratedParameterOptimizer:
    """統合パラメータ最適化システム"""

    def __init__(self, config: IntegratedOptimizationConfig | None = None):
        self.config = config or IntegratedOptimizationConfig()
        self.profiler = PerformanceProfiler()
        self.logger = logging.getLogger(__name__)

        # サブシステムの初期化
        self.walk_forward_analyzer = WalkForwardAnalyzer()
        self.kelly_sizer = KellyPositionSizer(
            min_trades=self.config.min_trades_for_kelly
        )
        self.atr_risk_manager = ATRRiskManager()
        self.confidence_adjuster = AdaptiveConfidenceAdjuster()

        # 最適化履歴
        self.optimization_history: list[IntegratedOptimizationResult] = []

    def create_integrated_strategy_evaluator(
        self,
        base_strategy_func: Callable[[pd.DataFrame, ParameterSet], dict[str, float]],
    ) -> Callable[[pd.DataFrame, ParameterSet], dict[str, float]]:
        """
        統合された戦略評価関数を作成

        Args:
            base_strategy_func: 基本戦略評価関数

        Returns:
            統合戦略評価関数
        """

        def integrated_evaluator(
            data: pd.DataFrame, params: ParameterSet
        ) -> dict[str, float]:
            # 基本戦略評価
            base_performance = base_strategy_func(data, params)

            # Kelly基準適用
            trades = base_performance.get("trades", [])
            if len(trades) >= self.config.min_trades_for_kelly:
                kelly_decision = self.kelly_sizer.calculate_dynamic_position_size(
                    trades, 10000, self.config.kelly_risk_tolerance
                )
                # ポジションサイズをKelly基準で調整
                adjusted_position_size = min(
                    kelly_decision.position_size_fraction, self.config.max_position_size
                )
            else:
                adjusted_position_size = self.config.max_position_size

            # ATRリスク管理適用
            if len(data) >= self.config.atr_period:
                atr_series = self.atr_risk_manager.calculate_atr(
                    data, self.config.atr_period
                )
                current_atr = atr_series.iloc[-1]

                # リスクレベル評価
                risk_level = self.atr_risk_manager.assess_risk_level(current_atr, data)

                # ポジション制限計算
                limits = self.atr_risk_manager.calculate_position_limits(
                    entry_price=data["close"].iloc[-1],
                    position_size=adjusted_position_size,
                    current_atr=current_atr,
                    risk_level=risk_level,
                    mode=self.config.risk_management_mode,
                )

                # リスク調整リターンを計算
                risk_adjusted_return = (
                    base_performance.get("total_return", 0) / limits.risk_amount
                    if limits.risk_amount > 0
                    else 0
                )
            else:
                risk_adjusted_return = base_performance.get("total_return", 0)

            # 信頼度調整適用
            if self.config.adaptive_thresholds_enabled and len(data) >= 20:
                threshold_decision = (
                    self.confidence_adjuster.calculate_adaptive_threshold(data, trades)
                )
                # 信頼度フィルタ適用
                filtered_trades = [
                    trade
                    for trade in trades
                    if trade.get("confidence", 0) >= threshold_decision.final_threshold
                ]
                if filtered_trades:
                    win_rate = sum(
                        1 for t in filtered_trades if t.get("pnl", 0) > 0
                    ) / len(filtered_trades)
                else:
                    win_rate = base_performance.get("win_rate", 0)
            else:
                win_rate = base_performance.get("win_rate", 0)

            # 統合パフォーマンス計算
            integrated_performance = base_performance.copy()
            integrated_performance.update(
                {
                    "risk_adjusted_return": risk_adjusted_return,
                    "kelly_adjusted_position_size": adjusted_position_size,
                    "filtered_win_rate": win_rate,
                    "integrated_score": self._calculate_integrated_score(
                        base_performance, risk_adjusted_return, win_rate
                    ),
                }
            )

            return integrated_performance

        return integrated_evaluator

    def _calculate_integrated_score(
        self,
        base_performance: dict[str, float],
        risk_adjusted_return: float,
        filtered_win_rate: float,
    ) -> float:
        """統合スコアを計算"""
        sharpe = base_performance.get("sharpe_ratio", 0)
        max_drawdown = base_performance.get("max_drawdown", 0)

        # スコア計算（Sharpe Ratio + リスク調整リターン + 勝率）
        score = sharpe * 0.5 + risk_adjusted_return * 0.3 + filtered_win_rate * 0.2

        # 最大ドローダウン penalty
        if max_drawdown > 0.2:  # 20%以上のドローダウン
            score *= 0.8

        return score

    @PerformanceProfiler.profile
    def run_integrated_optimization(
        self,
        market_data: pd.DataFrame,
        base_strategy_func: Callable[[pd.DataFrame, ParameterSet], dict[str, float]],
        parameter_sets: list[ParameterSet] | None = None,
    ) -> IntegratedOptimizationResult:
        """
        統合最適化を実行

        Args:
            market_data: 市場データ
            base_strategy_func: 基本戦略評価関数
            parameter_sets: 評価するパラメータセット

        Returns:
            統合最適化結果
        """
        self.logger.info("統合パラメータ最適化を開始します")

        # 統合戦略評価関数作成
        integrated_evaluator = self.create_integrated_strategy_evaluator(
            base_strategy_func
        )

        # ウォークフォワード最適化実行
        walk_forward_results = self.walk_forward_analyzer.walk_forward_optimization(
            data=market_data,
            strategy_func=integrated_evaluator,
            train_days=self.config.train_days,
            test_days=self.config.test_days,
            step_days=self.config.step_days,
            parameter_sets=parameter_sets,
            min_samples=self.config.min_samples,
        )

        # 最適パラメータの選択
        optimal_result = self._select_optimal_parameters(walk_forward_results)
        optimal_params = optimal_result.best_parameters

        # Kellyパラメータ計算
        all_trades = []
        for result in walk_forward_results:
            trades = result.out_of_sample_performance.get("trades", [])
            all_trades.extend(trades)

        kelly_params = self.kelly_sizer.calculate_kelly_parameters(all_trades, 10000)
        if kelly_params is None:
            kelly_params = KellyParameters(0.5, 1.0, len(all_trades))

        # パフォーマンス要約
        performance_summary = self.walk_forward_analyzer.summarize_results(
            walk_forward_results
        )

        # レジーム分析
        regime_analysis = self._analyze_market_regimes(
            market_data, walk_forward_results
        )

        # 結果作成
        result = IntegratedOptimizationResult(
            walk_forward_results=walk_forward_results,
            optimal_parameters=optimal_params,
            kelly_parameters=kelly_params,
            performance_summary=performance_summary,
            regime_analysis=regime_analysis,
            optimization_timestamp=datetime.now(),
            config_used=self.config,
        )

        # 履歴保存
        self.optimization_history.append(result)

        self.logger.info(
            f"統合最適化完了: 平均Sharpe Ratio = {result.average_sharpe_ratio:.3f}"
        )
        return result

    def _select_optimal_parameters(
        self, results: list[OptimizationResult]
    ) -> OptimizationResult:
        """最適パラメータを選択"""
        if not results:
            raise ValueError("最適化結果がありません")

        # 設定された最適化目標に基づいて選択
        if self.config.optimization_target == "sharpe_ratio":
            key_func = lambda r: r.out_of_sample_performance.get(
                "sharpe_ratio", float("-inf")
            )
        elif self.config.optimization_target == "total_return":
            key_func = lambda r: r.out_of_sample_performance.get(
                "total_return", float("-inf")
            )
        elif self.config.optimization_target == "win_rate":
            key_func = lambda r: r.out_of_sample_performance.get(
                "win_rate", float("-inf")
            )
        else:
            key_func = lambda r: r.out_of_sample_performance.get(
                "integrated_score", float("-inf")
            )

        return max(results, key=key_func)

    def _analyze_market_regimes(
        self, market_data: pd.DataFrame, results: list[OptimizationResult]
    ) -> dict[str, Any]:
        """市場レジームを分析"""
        regime_performance = {}

        for result in results:
            # テスト期間のデータ取得
            test_data = market_data.loc[
                result.window.test_start : result.window.test_end
            ]

            if len(test_data) >= 20:
                # レジーム検出
                regime = self.confidence_adjuster.regime_detector.detect_regime(
                    test_data
                )

                if regime.value not in regime_performance:
                    regime_performance[regime.value] = []

                regime_performance[regime.value].append(
                    result.out_of_sample_performance.get("sharpe_ratio", 0)
                )

        # レジーム別平均パフォーマンス
        regime_summary = {}
        for regime, performances in regime_performance.items():
            regime_summary[regime] = {
                "average_sharpe": np.mean(performances),
                "best_sharpe": np.max(performances),
                "worst_sharpe": np.min(performances),
                "sample_count": len(performances),
            }

        return regime_summary

    def save_optimization_results(
        self, result: IntegratedOptimizationResult, filepath: str
    ):
        """最適化結果を保存"""
        write_json(filepath, result.to_dict(), indent=2, ensure_ascii=False)

        self.logger.info(f"最適化結果を保存しました: {filepath}")

    def load_optimization_results(self, filepath: str) -> IntegratedOptimizationResult:
        """最適化結果を読み込み"""
        data = read_json(filepath)

        # データからオブジェクト再構築（簡易版）
        optimal_params = ParameterSet(**data["optimal_parameters"])
        kelly_params = KellyParameters(**data["kelly_parameters"])
        config = IntegratedOptimizationConfig(**data["config_used"])

        # ウォークフォワード結果の再構築は複雑なので、主要な結果のみ
        result = IntegratedOptimizationResult(
            walk_forward_results=[],  # 再構築しない
            optimal_parameters=optimal_params,
            kelly_parameters=kelly_params,
            performance_summary=data["performance_summary"],
            regime_analysis=data["regime_analysis"],
            optimization_timestamp=pd.Timestamp(data["optimization_timestamp"]),
            config_used=config,
        )

        return result

    def get_optimization_recommendations(
        self, result: IntegratedOptimizationResult
    ) -> list[str]:
        """最適化結果に基づく推奨事項を生成"""
        recommendations = []

        # Sharpe Ratio改善
        avg_sharpe = result.average_sharpe_ratio
        if avg_sharpe > 0.5:
            recommendations.append("良好なリスク調整リターンを達成")
        elif avg_sharpe < 0:
            recommendations.append("リスク調整リターンが負 - パラメータ見直し検討")

        # Kelly基準の妥当性
        kelly_fraction = result.kelly_parameters.kelly_fraction
        if kelly_fraction > 0.1:
            recommendations.append(
                f"Kelly分数 {kelly_fraction:.1%} - 積極的なポジションサイズ可能"
            )
        elif kelly_fraction < 0.02:
            recommendations.append(
                f"Kelly分数 {kelly_fraction:.1%} - 保守的なポジションサイズ推奨"
            )

        # レジーム分析
        regime_analysis = result.regime_analysis
        best_regime = (
            max(regime_analysis.items(), key=lambda x: x[1]["average_sharpe"])
            if regime_analysis
            else None
        )

        if best_regime:
            recommendations.append(
                f"最適レジーム: {best_regime[0]} "
                f"(Sharpe: {best_regime[1]['average_sharpe']:.2f})"
            )

        return recommendations

# ===== 使用例とテスト =====

if __name__ == "__main__":
    """使用例"""

    # 統合最適化システムの初期化
    config = IntegratedOptimizationConfig(
        train_days=60,  # 短めに設定
        test_days=20,
        kelly_risk_tolerance="half",
        risk_management_mode=RiskManagementMode.DYNAMIC,
        adaptive_thresholds_enabled=True,
    )

    optimizer = IntegratedParameterOptimizer(config)

    # サンプル市場データ生成
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    np.random.seed(42)

    # トレンド + ノイズのデータ生成
    trend = np.linspace(0, 50, 200)
    noise = np.random.randn(200) * 3
    prices = 100 + trend + noise

    market_data = pd.DataFrame(
        {
            "open": prices,
            "high": prices + np.abs(np.random.randn(200)),
            "low": prices - np.abs(np.random.randn(200)),
            "close": prices + np.random.randn(200) * 0.5,
        },
        index=dates,
    )

    # 基本戦略評価関数（モック）
    def mock_strategy_evaluator(
        data: pd.DataFrame, params: ParameterSet
    ) -> dict[str, float]:
        """モック戦略評価関数"""
        # 単純なリターンモデル
        returns = data["close"].pct_change().dropna()
        total_return = (1 + returns).prod() - 1

        # 勝率計算（ランダム）
        win_rate = 0.5 + np.random.randn() * 0.1
        win_rate = np.clip(win_rate, 0.3, 0.7)

        # Sharpe Ratio
        from ztb.metrics.metrics import sharpe_ratio as calc_sharpe_ratio

        sharpe_ratio = calc_sharpe_ratio(returns)

        # モックトレード生成
        num_trades = np.random.randint(10, 30)
        trades = []
        for _ in range(num_trades):
            pnl = np.random.randn() * 100
            confidence = np.random.uniform(0.5, 0.9)
            trades.append({"pnl": pnl, "confidence": confidence})

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "max_drawdown": 0.15,  # 固定値
            "total_trades": num_trades,
            "trades": trades,
        }

    # 統合最適化実行
    print("統合パラメータ最適化を実行中...")
    result = optimizer.run_integrated_optimization(
        market_data=market_data, base_strategy_func=mock_strategy_evaluator
    )

    # 結果表示
    print("\n=== 統合最適化結果 ===")
    print(f"平均Sharpe Ratio: {result.average_sharpe_ratio:.3f}")
    print(f"平均勝率: {result.average_win_rate:.1%}")
    print(f"総リターン: {result.total_return:.1%}")
    print(f"最適Kelly分数: {result.kelly_parameters.kelly_fraction:.1%}")

    print("\n最適パラメータ:")
    opt_params = result.optimal_parameters
    print(f"  ストップロスATR乗数: {opt_params.stop_loss_atr_multiplier}")
    print(f"  テイクプロフィット乗数: {opt_params.take_profit_risk_multiplier}")
    print(f"  Kellyポジションサイズ: {opt_params.position_size_kelly_fraction:.1%}")
    print(f"  信頼度閾値: {opt_params.confidence_threshold:.2f}")

    # 推奨事項
    recommendations = optimizer.get_optimization_recommendations(result)
    print("\n推奨事項:")
    for rec in recommendations:
        print(f"  - {rec}")

    # 結果保存
    optimizer.save_optimization_results(result, "integrated_optimization_result.json")
    print("\n結果を integrated_optimization_result.json に保存しました")
