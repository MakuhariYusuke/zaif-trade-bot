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
from collections.abc import Callable, Mapping
from typing import cast

import numpy as np
import pandas as pd

from ztb.metrics.statistical_validator import StatisticalValidator
from ztb.metrics.metrics import calculate_atr
from ztb.risk.enhanced_risk_manager import EnhancedRiskManager
from ztb.trading.backtest.runner import BacktestEngine
from ztb.trading.risk.compat import ensure_risk_manager_protocol
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

ObjectMap = dict[str, object]
TradeList = list[ObjectMap]
IterationList = list[ObjectMap]
StrategyFunction = Callable[[object, float], object]


def _as_object_map(value: object) -> ObjectMap:
    """Return dict-like payload or empty dict."""
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items() if isinstance(k, str)}
    return {}


def _as_float(value: object, default: float = 0.0) -> float:
    """Coerce value to float with safe fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    """Coerce value to int with safe fallback."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: object, default: bool = False) -> bool:
    """Coerce value to bool while preserving explicit bool values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _as_float_list(value: object) -> list[float]:
    """Normalize list-like values to a float list."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        return value.astype(np.float64, copy=False).tolist()
    if isinstance(value, pd.Series):
        if value.empty:
            return []
        return value.astype(float).tolist()
    if isinstance(value, list):
        return [_as_float(v, 0.0) for v in value]
    return []


def _as_trade_list(value: object) -> TradeList:
    """Normalize trade records to list[dict[str, object]]."""
    records: object
    if isinstance(value, pd.DataFrame):
        records = value.to_dict("records")
    else:
        records = value

    if not isinstance(records, list):
        return []

    trades: TradeList = []
    for record in records:
        if isinstance(record, Mapping):
            trades.append(_as_object_map(record))
    return trades


def _last_portfolio_value(iteration: ObjectMap, default: float = 10000.0) -> float:
    """Extract last portfolio value from iteration payload safely."""
    portfolio_values = _as_float_list(iteration.get("portfolio_values", []))
    if not portfolio_values:
        return default
    return portfolio_values[-1]


def _commission_to_bps(commission: float) -> float:
    """Accept decimal-rate or bps commission and normalize to bps."""
    if commission <= 1.0:
        return commission * 10000.0
    return commission


class _FunctionStrategyAdapter:
    """Adapter that wraps plain strategy function for BacktestEngine."""

    def __init__(self, strategy_func: StrategyFunction):
        self.strategy_func = strategy_func

    def _invoke_strategy(
        self, data: pd.DataFrame, latest_data: ObjectMap
    ) -> object:
        """Call strategy while preserving compatibility with legacy signatures."""
        try:
            return self.strategy_func(latest_data, 1000.0)
        except Exception as dict_input_error:
            try:
                return self.strategy_func(data, 1000.0)
            except Exception as dataframe_input_error:
                raise dataframe_input_error from dict_input_error

    @staticmethod
    def _resolve_action(result: ObjectMap) -> str | None:
        """Resolve action from action/signal style strategy outputs."""
        raw_action = result.get("action")
        if isinstance(raw_action, str):
            normalized = raw_action.strip().lower()
            if normalized in {"buy", "long"}:
                return "buy"
            if normalized in {"sell", "short"}:
                return "sell"
            if normalized in {"hold", "none", "flat", ""}:
                return None

        signal_value = _as_float(result.get("signal", 0.0), 0.0)
        if signal_value > 0:
            return "buy"
        if signal_value < 0:
            return "sell"
        return None

    def generate_signal(
        self, data: pd.DataFrame, current_position: int
    ) -> dict[str, object]:
        # Keep compatibility with existing strategy function contract:
        # strategy(latest_row_dict, portfolio_value) -> signal payload.
        latest_data = cast(ObjectMap, data.iloc[-1].to_dict())
        strategy_output = self._invoke_strategy(data, latest_data)

        if isinstance(strategy_output, str):
            result: ObjectMap = {"action": strategy_output}
        else:
            result = _as_object_map(strategy_output)

        action = self._resolve_action(result)
        default_position_size = 1.0 if action else 0.0

        return {
            "action": action,
            "position_size": _as_float(
                result.get("position_size", default_position_size), default_position_size
            ),
            "price": _as_float(result.get("price", latest_data.get("close", 0.0)), 0.0),
        }

    def update_hyperparameters(self, hyperparameters: dict[str, float]) -> None:
        # Hyperparameter updates are currently not supported by this adapter.
        return None


class IntegratedBacktestRunner:
    """
    統合バックテストランナー

    既存のバックテスト機能を拡張し、リスク管理と統計検証を統合。
    Phase 2のマルチタイムフレーム分析も組み込み。
    """

    def __init__(self, config: Mapping[str, object]):
        """
        Args:
            config: 統合バックテスト設定
        """
        self.config = _as_object_map(config)

        # コンポーネント初期化
        backtest_config = _as_object_map(self.config.get("backtest_config", {}))
        target_vol_raw = backtest_config.get("target_vol")
        target_vol = (
            _as_float(target_vol_raw, 0.0) if target_vol_raw is not None else None
        )
        self.backtest_runner = BacktestEngine(
            initial_capital=_as_float(backtest_config.get("initial_capital", 10000.0), 10000.0),
            slippage_bps=_as_float(backtest_config.get("slippage_bps", 5.0), 5.0),
            commission_bps=_as_float(backtest_config.get("commission_bps", 0.0), 0.0),
            enable_risk=_as_bool(backtest_config.get("enable_risk", False), False),
            risk_profile=str(backtest_config.get("risk_profile", "balanced")),
            target_vol=target_vol,
            enable_adaptation=_as_bool(
                backtest_config.get("enable_adaptation", False), False
            ),
            max_position_size=_as_float(
                backtest_config.get("max_position_size", 1.0), 1.0
            ),
        )
        self.risk_manager = ensure_risk_manager_protocol(
            EnhancedRiskManager(_as_object_map(self.config.get("risk_config", {})))
        )
        self.statistical_validator = StatisticalValidator(
            _as_object_map(self.config.get("validation_config", {}))
        )

        # 統合設定
        self.enable_risk_management = _as_bool(
            self.config.get("enable_risk_management", True), True
        )
        self.enable_statistical_validation = _as_bool(
            self.config.get("enable_statistical_validation", True), True
        )
        self.multi_timeframe_enabled = _as_bool(
            self.config.get("multi_timeframe_enabled", True), True
        )

        # テスト設定
        self.n_iterations = max(0, _as_int(self.config.get("n_iterations", 100), 100))
        self.confidence_level = _as_float(
            self.config.get("confidence_level", 0.95), 0.95
        )

        logger.info(
            f"IntegratedBacktestRunner initialized with "
            f"risk_management={self.enable_risk_management}, "
            f"statistical_validation={self.enable_statistical_validation}"
        )

    def run_integrated_backtest(
        self,
        strategy_func: StrategyFunction,
        market_data: pd.DataFrame,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
    ) -> ObjectMap:
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
        iterations: IterationList = []
        results: ObjectMap = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "iterations": iterations,
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

                iterations.append(iteration_result)

                if iteration_result.get("error"):
                    logger.warning(
                        f"Iteration {i+1} failed: {iteration_result['error']}"
                    )

            # 統合結果集計
            results["summary"] = self._aggregate_results(iterations)
            if self.enable_risk_management:
                results["risk_analysis"] = self._analyze_risk_across_iterations(
                    iterations
                )
            else:
                results["risk_analysis"] = {"disabled": True}
            if self.enable_statistical_validation:
                results["statistical_validation"] = self._validate_statistically(
                    iterations
                )
            else:
                results["statistical_validation"] = {"disabled": True}

            results["success"] = True

        except Exception as e:
            logger.error(f"Integrated backtest failed: {e}")
            results["error"] = str(e)
            results["success"] = False

        return results

    def _run_single_integrated_backtest(
        self,
        strategy_func: StrategyFunction,
        market_data: pd.DataFrame,
        initial_capital: float,
        commission: float,
        iteration: int = 0,
    ) -> ObjectMap:
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
        result: ObjectMap = {
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
                result["portfolio_values"] = _as_float_list(backtest_result["equity_curve"])

            # tradesを更新
            if "trades" in backtest_result:
                result["trades"] = _as_trade_list(backtest_result["trades"])

            # リスク分析
            if self.enable_risk_management:
                result["risk_analysis"] = self._analyze_iteration_risk(result)

            # 統計的検証
            portfolio_values = _as_float_list(result.get("portfolio_values", []))
            if self.enable_statistical_validation and len(portfolio_values) > 1:
                returns = self._calculate_returns_from_portfolio_values(portfolio_values)
                if returns:
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
        strategy_func: StrategyFunction,
        market_data: pd.DataFrame,
        initial_capital: float,
        commission: float,
    ) -> ObjectMap:
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

        # 呼び出し時のパラメータを反映。
        # commissionは小数(0.001) / bps(10)のどちらでも受け取れるようにする。
        self.backtest_runner.initial_capital = initial_capital
        self.backtest_runner.commission_bps = _commission_to_bps(commission)

        strategy_adapter = _FunctionStrategyAdapter(strategy_func)
        (
            equity_curve,
            trades_payload,
            metadata,
            signal_performance_summary,
        ) = self.backtest_runner.run_backtest(strategy_adapter, market_data)

        # 結果を辞書形式に変換
        basic_result: ObjectMap = {
            "equity_curve": equity_curve,
            "trades": trades_payload,
            "metadata": _as_object_map(metadata),
            "signal_performance_summary": _as_object_map(signal_performance_summary),
        }

        # リスク管理統合
        if self.enable_risk_management:
            enhanced_trades: TradeList = []
            risk_adjustments: list[ObjectMap] = []
            trades_list = _as_trade_list(trades_payload)
            atr = self._calculate_atr(market_data)
            for trade in trades_list:
                # リスク調整適用
                risk_adjusted_trade = self._apply_risk_management_to_trade(
                    trade, market_data, precomputed_atr=atr
                )
                enhanced_trades.append(risk_adjusted_trade)
                risk_adjustments.append(
                    _as_object_map(risk_adjusted_trade.get("risk_info", {}))
                )

            basic_result["trades"] = enhanced_trades
            basic_result["risk_adjustments"] = risk_adjustments

        return basic_result

    def _apply_risk_management_to_trade(
        self,
        trade: ObjectMap,
        market_data: pd.DataFrame,
        precomputed_atr: float | None = None,
    ) -> ObjectMap:
        """
        取引にリスク管理を適用

        Args:
            trade: 基本取引情報
            market_data: 市場データ

        Returns:
            リスク調整済み取引情報
        """
        enhanced_trade: ObjectMap = dict(trade)

        try:
            # リスク調整ポジション計算
            fallback_price = _as_float(
                market_data.iloc[-1].get("close", 0.0) if not market_data.empty else 0.0,
                0.0,
            )
            base_position = _as_float(trade.get("position_size", 1.0), 1.0)
            current_price = _as_float(trade.get("price", fallback_price), fallback_price)
            portfolio_value = _as_float(trade.get("portfolio_value", 10000.0), 10000.0)

            # ATR計算（簡易版）
            atr = precomputed_atr if precomputed_atr is not None else self._calculate_atr(market_data)

            # リスク調整
            risk_info = _as_object_map(
                self.risk_manager.calculate_enhanced_risk_adjusted_position(
                    base_position=base_position,
                    current_price=current_price,
                    portfolio_value=portfolio_value,
                    atr=atr,
                    df=market_data,
                )
            )
            adjusted_position = _as_float(
                risk_info.get("adjusted_position", base_position), base_position
            )
            risk_info.setdefault("original_position", base_position)
            risk_info.setdefault("adjusted_position", adjusted_position)

            # ポジションサイズ調整
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
        self, iteration_result: ObjectMap
    ) -> ObjectMap:
        """イテレーション別リスク分析"""
        raw_risk_adjustments = iteration_result.get("risk_adjustments", [])
        if not isinstance(raw_risk_adjustments, list) or not raw_risk_adjustments:
            return {"error": "No risk adjustments available"}

        risk_adjustments = [
            _as_object_map(adj)
            for adj in raw_risk_adjustments
            if isinstance(adj, Mapping)
        ]
        if not risk_adjustments:
            return {"error": "No risk adjustments available"}

        # リスク調整の統計
        risk_levels = [
            _as_float(adj.get("risk_level", 0.0), 0.0)
            for adj in risk_adjustments
            if "risk_level" in adj
        ]
        position_adjustments: list[float] = []
        for adj in risk_adjustments:
            if "adjusted_position" not in adj or "original_position" not in adj:
                continue
            adjusted_position = _as_float(adj.get("adjusted_position", 1.0), 1.0)
            original_position = max(
                _as_float(adj.get("original_position", 1.0), 1.0), 0.001
            )
            position_adjustments.append(adjusted_position / original_position)

        risk_levels_arr = np.asarray(risk_levels, dtype=np.float64)
        position_adjustments_arr = np.asarray(position_adjustments, dtype=np.float64)

        return {
            "avg_risk_level": float(risk_levels_arr.mean()) if risk_levels else 0.0,
            "max_risk_level": float(risk_levels_arr.max()) if risk_levels else 0.0,
            "avg_position_adjustment": float(position_adjustments_arr.mean())
            if position_adjustments
            else 1.0,
            "risk_adjustment_count": len(risk_adjustments),
        }

    def _aggregate_results(self, iterations: IterationList) -> ObjectMap:
        """結果集計"""
        if not iterations:
            return {"error": "No iterations available"}

        successful_iterations = [
            it for it in iterations if _as_bool(it.get("success", False), False)
        ]
        if not successful_iterations:
            return {"error": "No successful iterations"}

        baseline_capital = max(_as_float(self.backtest_runner.initial_capital, 10000.0), 1e-9)
        final_portfolio_values = np.asarray(
            [_last_portfolio_value(it, baseline_capital) for it in successful_iterations],
            dtype=np.float64,
        )
        total_returns = (final_portfolio_values - baseline_capital) / baseline_capital

        avg_final_portfolio = float(final_portfolio_values.mean())
        std_final_portfolio = float(final_portfolio_values.std())
        avg_total_return = float(total_returns.mean())
        std_total_return = float(total_returns.std())

        total_iterations = len(iterations)
        successful_count = len(successful_iterations)
        success_rate = successful_count / total_iterations if total_iterations > 0 else 0.0
        sharpe_ratio = avg_total_return / std_total_return if std_total_return > 0 else 0.0

        summary: ObjectMap = {
            "total_iterations": total_iterations,
            "successful_iterations": successful_count,
            "success_rate": success_rate,
            "avg_final_portfolio": avg_final_portfolio,
            "std_final_portfolio": std_final_portfolio,
            "avg_total_return": avg_total_return,
            "std_total_return": std_total_return,
            "max_portfolio": float(final_portfolio_values.max()),
            "min_portfolio": float(final_portfolio_values.min()),
            "sharpe_ratio": float(sharpe_ratio),
        }

        # 信頼区間
        if total_returns.size >= 30:  # 十分なサンプル数
            ci_low, ci_high = np.percentile(total_returns, [2.5, 97.5])
            summary["return_confidence_interval"] = [float(ci_low), float(ci_high)]

        return summary

    def _analyze_risk_across_iterations(
        self, iterations: IterationList
    ) -> ObjectMap:
        """イテレーション間リスク分析"""
        risk_analyses = [
            _as_object_map(it.get("risk_analysis", {}))
            for it in iterations
            if _as_bool(it.get("success", False), False)
            and isinstance(it.get("risk_analysis", {}), Mapping)
        ]

        if not risk_analyses:
            return {"error": "No risk analysis data available"}

        avg_risk_levels = np.asarray(
            [_as_float(ra.get("avg_risk_level", 0.0), 0.0) for ra in risk_analyses],
            dtype=np.float64,
        )
        max_risk_levels = np.asarray(
            [_as_float(ra.get("max_risk_level", 0.0), 0.0) for ra in risk_analyses],
            dtype=np.float64,
        )
        position_adjustments = np.asarray(
            [_as_float(ra.get("avg_position_adjustment", 1.0), 1.0) for ra in risk_analyses],
            dtype=np.float64,
        )

        risk_level_std = float(avg_risk_levels.std())
        return {
            "avg_risk_level_across_iterations": float(avg_risk_levels.mean()),
            "std_risk_level_across_iterations": risk_level_std,
            "max_risk_level_across_iterations": float(max_risk_levels.max()),
            "avg_position_adjustment_across_iterations": float(
                position_adjustments.mean()
            ),
            "risk_stability_score": 1.0 / (1.0 + risk_level_std),  # リスクの安定性スコア
        }

    def _validate_statistically(
        self, iterations: IterationList
    ) -> ObjectMap:
        """統計的検証"""
        valid_iterations = [
            it for it in iterations if _as_bool(it.get("success", False), False)
        ]

        if not valid_iterations:
            return {"error": "No valid statistical data"}

        # リターン系列の収集（1回の走査で一貫性データも収集）
        all_returns: list[float] = []
        iteration_returns: list[float] = []
        for it in valid_iterations:
            portfolio_values = _as_float_list(it.get("portfolio_values", []))
            if len(portfolio_values) < 2:
                continue
            returns = self._calculate_returns_from_portfolio_values(portfolio_values)
            if not returns:
                continue
            all_returns.extend(returns)
            iteration_returns.append(float(np.mean(np.asarray(returns, dtype=np.float64))))

        if not all_returns:
            return {"error": "No return data available"}

        validation_result = _as_object_map(
            self.statistical_validator.validate_performance_metrics(
                all_returns, multiple_metrics=True
            )
        )

        # イテレーション間の一貫性検証
        if len(iteration_returns) >= 3:
            validation_result[
                "iteration_consistency"
            ] = self.statistical_validator.validate_multiple_strategies(
                {
                    f"iteration_{i}": [ret]
                    for i, ret in enumerate(iteration_returns)
                }
            )

        return validation_result

    def _calculate_returns_from_portfolio_values(
        self, portfolio_values: list[float]
    ) -> list[float]:
        """ポートフォリオ価値からリターンを計算"""
        if len(portfolio_values) < 2:
            return []

        values = np.asarray(portfolio_values, dtype=np.float64)
        previous_values = values[:-1]
        returns = np.divide(
            np.diff(values),
            previous_values,
            out=np.zeros(values.size - 1, dtype=np.float64),
            where=np.abs(previous_values) > 1e-12,
        )
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        return returns.tolist()

    def generate_integrated_report(self, results: ObjectMap) -> str:
        """
        統合レポート生成

        Args:
            results: 統合バックテスト結果

        Returns:
            レポート文字列
        """
        if not _as_bool(results.get("success", False), False):
            return (
                f"Integrated Backtest Failed: {results.get('error', 'Unknown error')}"
            )

        summary = _as_object_map(results.get("summary", {}))
        risk_analysis = _as_object_map(results.get("risk_analysis", {}))
        validation = _as_object_map(results.get("statistical_validation", {}))
        timestamp = str(results.get("timestamp", "N/A"))

        report = f"""
# Integrated Backtest Report
Generated: {timestamp}

## Summary
- Total Iterations: {int(_as_float(summary.get('total_iterations', 0), 0))}
- Successful Iterations: {int(_as_float(summary.get('successful_iterations', 0), 0))}
- Success Rate: {_as_float(summary.get('success_rate', 0.0), 0.0):.1%}

## Performance Metrics
- Average Final Portfolio: ${_as_float(summary.get('avg_final_portfolio', 0.0), 0.0):,.2f}
- Average Total Return: {_as_float(summary.get('avg_total_return', 0.0), 0.0):.2%}
- Sharpe Ratio: {_as_float(summary.get('sharpe_ratio', 0.0), 0.0):.3f}
"""

        ci = _as_float_list(summary.get("return_confidence_interval", []))
        if len(ci) == 2:
            report += f"- Return Confidence Interval: {ci[0]:.2%} to {ci[1]:.2%}\n"

        report += f"""
## Risk Analysis
- Average Risk Level: {_as_float(risk_analysis.get('avg_risk_level_across_iterations', 0.0), 0.0):.3f}
- Risk Stability Score: {_as_float(risk_analysis.get('risk_stability_score', 0.0), 0.0):.3f}
- Average Position Adjustment: {_as_float(risk_analysis.get('avg_position_adjustment_across_iterations', 1.0), 1.0):.3f}
"""

        if _as_bool(validation.get("valid", False), False):
            sharpe_ratio_info = _as_object_map(validation.get("sharpe_ratio", {}))
            report += f"""
## Statistical Validation
- Validation Status: {'✓ Valid' if _as_bool(validation.get('valid', False), False) else '✗ Invalid'}
- Sharpe Ratio: {_as_float(sharpe_ratio_info.get('value', 0.0), 0.0):.3f}
"""

            stability = _as_object_map(validation.get("stability_analysis", {}))
            if stability:
                report += f"- Max Drawdown: {_as_float(stability.get('max_drawdown', 0.0), 0.0):.2%}\n"

        return report
