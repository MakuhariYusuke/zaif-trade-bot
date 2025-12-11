#!/usr/bin/env python3
"""
Integrated Backtest Runner for SAC v445
Phase 3: Risk Management & Statistical Validation

既存のバックテスト機能を統合し、リスク管理と統計検証を組み合わせた
包括的なバックテストシステム。
"""

# Workaround for [WinError 1114] DLL initialization failed
# Torch must be imported before pandas/scipy/numpy in some environments
try:
    import torch
except ImportError:
    pass

from datetime import datetime
from typing import Any, Callable, Dict, List, cast

import numpy as np
import pandas as pd

from ztb.metrics.statistical_validator import StatisticalValidator
from ztb.metrics.statistics import calculate_atr
from ztb.risk.enhanced_risk_manager import EnhancedRiskManager
from ztb.trading.backtest.runner import BacktestEngine
from ztb.trading.risk.compat import ensure_risk_manager_protocol
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class IntegratedBacktestRunner:
    """
    統合バックテストランナー

    既存のバックテスト機能を拡張し、リスク管理と統計検証を統合。
    Phase 2のマルチタイムフレーム分析も組み込み。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 統合バックテスト設定
        """
        self.config = config

        # コンポーネント初期化
        backtest_config = config.get("backtest_config", {})
        self.backtest_runner = BacktestEngine(
            initial_capital=backtest_config.get("initial_capital", 10000.0),
            slippage_bps=backtest_config.get("slippage_bps", 5.0),
            commission_bps=backtest_config.get("commission_bps", 0.0),
            enable_risk=backtest_config.get("enable_risk", False),
            risk_profile=backtest_config.get("risk_profile", "balanced"),
            target_vol=backtest_config.get("target_vol"),
            enable_adaptation=backtest_config.get("enable_adaptation", False),
            max_position_size=backtest_config.get("max_position_size", 1.0),
        )
        self.risk_manager = ensure_risk_manager_protocol(
            EnhancedRiskManager(config.get("risk_config", {}))
        )
        self.statistical_validator = StatisticalValidator(
            config.get("validation_config", {})
        )

        # 統合設定
        self.enable_risk_management = config.get("enable_risk_management", True)
        self.enable_statistical_validation = config.get(
            "enable_statistical_validation", True
        )
        self.multi_timeframe_enabled = config.get("multi_timeframe_enabled", True)

        # テスト設定
        self.n_iterations = config.get("n_iterations", 100)
        self.confidence_level = config.get("confidence_level", 0.95)

        logger.info(
            f"IntegratedBacktestRunner initialized with "
            f"risk_management={self.enable_risk_management}, "
            f"statistical_validation={self.enable_statistical_validation}"
        )

    def run_integrated_backtest(
        self,
        strategy_func: Callable,
        market_data: pd.DataFrame,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
    ) -> Dict[str, Any]:
        """
        統合バックテスト実行

        Args:
            strategy_func: 戦略関数
            market_data: 市場データ
            initial_capital: 初期資本
            commission: 手数料

        Returns:
            統合バックテスト結果
        """
        results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "iterations": [],
        }

        try:
            # 複数イテレーション実行
            for i in range(self.n_iterations):
                logger.info(
                    f"Running integrated backtest iteration {i+1}/{self.n_iterations}"
                )

                iteration_result = self._run_single_integrated_backtest(
                    strategy_func, market_data, initial_capital, commission, iteration=i
                )

                results["iterations"].append(iteration_result)

                if iteration_result.get("error"):
                    logger.warning(
                        f"Iteration {i+1} failed: {iteration_result['error']}"
                    )

            # 統合結果集計
            results["summary"] = self._aggregate_results(results["iterations"])
            results["risk_analysis"] = self._analyze_risk_across_iterations(
                results["iterations"]
            )
            results["statistical_validation"] = self._validate_statistically(
                results["iterations"]
            )

            results["success"] = True

        except Exception as e:
            logger.error(f"Integrated backtest failed: {e}")
            results["error"] = str(e)
            results["success"] = False

        return results

    def _run_single_integrated_backtest(
        self,
        strategy_func: Callable,
        market_data: pd.DataFrame,
        initial_capital: float,
        commission: float,
        iteration: int = 0,
    ) -> Dict[str, Any]:
        """
        単一統合バックテスト実行

        Args:
            strategy_func: 戦略関数
            market_data: 市場データ
            initial_capital: 初期資本
            commission: 手数料
            iteration: イテレーション番号

        Returns:
            単一イテレーション結果
        """
        result = {
            "iteration": iteration,
            "trades": [],
            "portfolio_values": [],
            "risk_adjustments": [],
        }

        try:
            # バックテスト実行（拡張版）
            backtest_result = self._run_enhanced_backtest(
                strategy_func, market_data, initial_capital, commission
            )

            result.update(backtest_result)

            # equity_curveをportfolio_valuesに変換
            if "equity_curve" in backtest_result:
                result["portfolio_values"] = backtest_result["equity_curve"].tolist()

            # tradesを更新
            if "trades" in backtest_result:
                result["trades"] = backtest_result["trades"]

            # リスク分析
            if self.enable_risk_management:
                result["risk_analysis"] = self._analyze_iteration_risk(result)

            # 統計的検証
            if self.enable_statistical_validation and result["portfolio_values"]:
                returns = self._calculate_returns_from_portfolio_values(
                    cast(List[float], result["portfolio_values"])
                )
                result[
                    "statistical_metrics"
                ] = self.statistical_validator.validate_performance_metrics(returns)

            result["success"] = True

        except Exception as e:
            logger.error(
                f"Single integrated backtest iteration {iteration} failed: {e}"
            )
            result["error"] = str(e)
            result["success"] = False

        return result

    def _run_enhanced_backtest(
        self,
        strategy_func: Callable,
        market_data: pd.DataFrame,
        initial_capital: float,
        commission: float,
    ) -> Dict[str, Any]:
        """
        リスク管理統合拡張バックテスト

        Args:
            strategy_func: 戦略関数
            market_data: 市場データ
            initial_capital: 初期資本
            commission: 手数料

        Returns:
            拡張バックテスト結果
        """

        # 基本バックテスト実行
        # strategy_funcをStrategyAdapterでラップ
        class FunctionStrategyAdapter:
            def __init__(self, strategy_func):
                self.strategy_func = strategy_func

            def generate_signal(
                self, data: pd.DataFrame, current_position: int
            ) -> Dict[str, Any]:
                # データの最新行を取得
                latest_data = data.iloc[-1].to_dict()
                # strategy_funcを呼び出し
                result = self.strategy_func(
                    latest_data, 1000.0
                )  # portfolio_valueは仮定

                # signalをactionに変換
                signal_value = result.get("signal", 0)
                if signal_value == 1:
                    action = "buy"
                elif signal_value == -1:
                    action = "sell"
                else:
                    action = None  # 何もしない

                return {
                    "action": action,
                    "position_size": result.get("position_size", 0.0),
                    "price": result.get("price", latest_data.get("close", 0)),
                }

            def update_hyperparameters(self, hyperparameters: Dict[str, float]) -> None:
                pass  # ハイパーパラメータ更新は未実装

        strategy_adapter = FunctionStrategyAdapter(strategy_func)
        (
            equity_curve,
            trades_df,
            metadata,
            signal_performance_summary,
        ) = self.backtest_runner.run_backtest(strategy_adapter, market_data)

        # 結果を辞書形式に変換
        basic_result = {
            "equity_curve": equity_curve,
            "trades": trades_df,
            "metadata": metadata or {},
            "signal_performance_summary": signal_performance_summary or {},
        }

        # リスク管理統合
        if self.enable_risk_management:
            enhanced_trades = []
            risk_adjustments = []

            trades_list: List[Dict[str, Any]] = []
            if isinstance(trades_df, pd.DataFrame):
                trades_list = cast(List[Dict[str, Any]], trades_df.to_dict("records"))
            elif isinstance(trades_df, list):
                trades_list = trades_df

            for trade in trades_list:
                # リスク調整適用
                risk_adjusted_trade = self._apply_risk_management_to_trade(
                    trade, market_data
                )
                enhanced_trades.append(risk_adjusted_trade)
                risk_adjustments.append(risk_adjusted_trade.get("risk_info", {}))

            basic_result["trades"] = enhanced_trades
            basic_result["risk_adjustments"] = risk_adjustments

        return basic_result

    def _apply_risk_management_to_trade(
        self, trade: Dict[str, Any], market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        取引にリスク管理を適用

        Args:
            trade: 基本取引情報
            market_data: 市場データ

        Returns:
            リスク調整済み取引情報
        """
        enhanced_trade = trade.copy()

        try:
            # リスク調整ポジション計算
            base_position = trade.get("position_size", 1.0)
            current_price = trade.get("price", market_data.iloc[-1]["close"])
            portfolio_value = trade.get("portfolio_value", 10000.0)

            # ATR計算（簡易版）
            atr = self._calculate_atr(market_data)

            # リスク調整
            risk_info = self.risk_manager.calculate_enhanced_risk_adjusted_position(
                base_position=base_position,
                current_price=current_price,
                portfolio_value=portfolio_value,
                atr=atr,
                df=market_data,
            )

            # ポジションサイズ調整
            adjusted_position = risk_info.get("adjusted_position", base_position)
            enhanced_trade["original_position_size"] = base_position
            enhanced_trade["adjusted_position_size"] = adjusted_position
            enhanced_trade["risk_info"] = risk_info

        except Exception as e:
            logger.warning(f"Risk management application failed for trade: {e}")
            enhanced_trade["risk_info"] = {"error": str(e)}

        return enhanced_trade

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """ATR計算"""
        atr_series = calculate_atr(data, period)
        if atr_series.empty or pd.isna(atr_series.iloc[-1]):
            return 0.01
        return float(atr_series.iloc[-1])

    def _analyze_iteration_risk(
        self, iteration_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """イテレーション別リスク分析"""
        risk_adjustments = iteration_result.get("risk_adjustments", [])

        if not risk_adjustments:
            return {"error": "No risk adjustments available"}

        # リスク調整の統計
        risk_levels = [
            adj.get("risk_level", 0.0)
            for adj in risk_adjustments
            if "risk_level" in adj
        ]
        position_adjustments = [
            adj.get("adjusted_position", 1.0)
            / max(adj.get("original_position", 1.0), 0.001)
            for adj in risk_adjustments
            if "adjusted_position" in adj and "original_position" in adj
        ]

        return {
            "avg_risk_level": float(np.mean(risk_levels)) if risk_levels else 0.0,
            "max_risk_level": float(np.max(risk_levels)) if risk_levels else 0.0,
            "avg_position_adjustment": float(np.mean(position_adjustments))
            if position_adjustments
            else 1.0,
            "risk_adjustment_count": len(risk_adjustments),
        }

    def _aggregate_results(self, iterations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """結果集計"""
        successful_iterations = [it for it in iterations if it.get("success")]

        if not successful_iterations:
            return {"error": "No successful iterations"}

        # 基本指標集計
        final_portfolio_values = [
            it.get("portfolio_values", [10000.0])[-1] for it in successful_iterations
        ]

        total_returns = [(pv - 10000.0) / 10000.0 for pv in final_portfolio_values]

        # 統計量計算
        summary: Dict[str, Any] = {
            "total_iterations": len(iterations),
            "successful_iterations": len(successful_iterations),
            "success_rate": len(successful_iterations) / len(iterations),
            "avg_final_portfolio": float(np.mean(final_portfolio_values)),
            "std_final_portfolio": float(np.std(final_portfolio_values)),
            "avg_total_return": float(np.mean(total_returns)),
            "std_total_return": float(np.std(total_returns)),
            "max_portfolio": float(np.max(final_portfolio_values)),
            "min_portfolio": float(np.min(final_portfolio_values)),
            "sharpe_ratio": float(np.mean(total_returns) / np.std(total_returns))
            if np.std(total_returns) > 0
            else 0.0,
        }

        # 信頼区間
        if len(total_returns) >= 30:  # 十分なサンプル数
            ci_low, ci_high = np.percentile(total_returns, [2.5, 97.5])
            summary["return_confidence_interval"] = [float(ci_low), float(ci_high)]

        return summary

    def _analyze_risk_across_iterations(
        self, iterations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """イテレーション間リスク分析"""
        risk_analyses = [
            it.get("risk_analysis", {})
            for it in iterations
            if it.get("success") and "risk_analysis" in it
        ]

        if not risk_analyses:
            return {"error": "No risk analysis data available"}

        # リスク指標集計
        avg_risk_levels = [ra.get("avg_risk_level", 0.0) for ra in risk_analyses]
        max_risk_levels = [ra.get("max_risk_level", 0.0) for ra in risk_analyses]
        position_adjustments = [
            ra.get("avg_position_adjustment", 1.0) for ra in risk_analyses
        ]

        return {
            "avg_risk_level_across_iterations": float(np.mean(avg_risk_levels)),
            "std_risk_level_across_iterations": float(np.std(avg_risk_levels)),
            "max_risk_level_across_iterations": float(np.max(max_risk_levels)),
            "avg_position_adjustment_across_iterations": float(
                np.mean(position_adjustments)
            ),
            "risk_stability_score": 1.0
            / (1.0 + np.std(avg_risk_levels)),  # リスクの安定性スコア
        }

    def _validate_statistically(
        self, iterations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """統計的検証"""
        # 有効なイテレーションの収集
        valid_iterations = [
            it for it in iterations if it.get("success") and "statistical_metrics" in it
        ]

        if not valid_iterations:
            return {"error": "No valid statistical data"}

        # リターン系列の収集
        all_returns = []
        for it in valid_iterations:
            portfolio_values = it.get("portfolio_values", [])
            if len(portfolio_values) > 1:
                returns = self._calculate_returns_from_portfolio_values(
                    portfolio_values
                )
                all_returns.extend(returns)

        if not all_returns:
            return {"error": "No return data available"}

        # 統計的検証実行
        validation_result = self.statistical_validator.validate_performance_metrics(
            all_returns, multiple_metrics=True
        )

        # イテレーション間の一貫性検証
        iteration_returns = []
        for it in valid_iterations:
            portfolio_values = it.get("portfolio_values", [])
            if len(portfolio_values) > 1:
                returns = self._calculate_returns_from_portfolio_values(
                    portfolio_values
                )
                if returns:
                    iteration_returns.append(float(np.mean(returns)))

        if len(iteration_returns) >= 3:
            validation_result[
                "iteration_consistency"
            ] = self.statistical_validator.validate_multiple_strategies(
                {"iteration_" + str(i): [r] for i, r in enumerate(iteration_returns)}
            )

        return validation_result

    def _calculate_returns_from_portfolio_values(
        self, portfolio_values: List[float]
    ) -> List[float]:
        """ポートフォリオ価値からリターンを計算"""
        if len(portfolio_values) < 2:
            return []

        returns = []
        for i in range(1, len(portfolio_values)):
            ret = (portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[
                i - 1
            ]
            returns.append(ret)

        return returns

    def generate_integrated_report(self, results: Dict[str, Any]) -> str:
        """
        統合レポート生成

        Args:
            results: 統合バックテスト結果

        Returns:
            レポート文字列
        """
        if not results.get("success"):
            return (
                f"Integrated Backtest Failed: {results.get('error', 'Unknown error')}"
            )

        summary = results.get("summary", {})
        risk_analysis = results.get("risk_analysis", {})
        validation = results.get("statistical_validation", {})

        report = f"""
# Integrated Backtest Report
Generated: {results['timestamp']}

## Summary
- Total Iterations: {summary.get('total_iterations', 0)}
- Successful Iterations: {summary.get('successful_iterations', 0)}
- Success Rate: {summary.get('success_rate', 0):.1%}

## Performance Metrics
- Average Final Portfolio: ${summary.get('avg_final_portfolio', 0):,.2f}
- Average Total Return: {summary.get('avg_total_return', 0):.2%}
- Sharpe Ratio: {summary.get('sharpe_ratio', 0):.3f}
"""

        if "return_confidence_interval" in summary:
            ci = summary["return_confidence_interval"]
            report += f"- Return Confidence Interval: {ci[0]:.2%} to {ci[1]:.2%}\n"

        report += f"""
## Risk Analysis
- Average Risk Level: {risk_analysis.get('avg_risk_level_across_iterations', 0):.3f}
- Risk Stability Score: {risk_analysis.get('risk_stability_score', 0):.3f}
- Average Position Adjustment: {risk_analysis.get('avg_position_adjustment_across_iterations', 1):.3f}
"""

        if validation.get("valid"):
            report += f"""
## Statistical Validation
- Validation Status: {'✓ Valid' if validation.get('valid') else '✗ Invalid'}
- Sharpe Ratio: {validation.get('sharpe_ratio', {}).get('value', 0):.3f}
"""

            if "stability_analysis" in validation:
                stability = validation["stability_analysis"]
                report += f"- Max Drawdown: {stability.get('max_drawdown', 0):.2%}\n"

        return report
