#!/usr/bin/env python3
"""
V433 Phase 3: リスクオーバーレイシステム
VaR計算、緊急停止機能、リアルタイムリスク監視
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from ztb.trading.position_manager import PortfolioState, PositionManager
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RiskOverlayConfig:
    """リスクオーバーレイ設定"""

    # VaR設定
    var_confidence_level: float = 0.95  # 95%信頼区間
    var_time_horizon_days: int = 1  # 1日VaR
    var_calculation_window_days: int = 30  # 計算ウィンドウ (30日)
    var_update_interval_seconds: int = 300  # 5分ごと更新

    # 緊急停止設定
    emergency_stop_enabled: bool = True
    emergency_stop_var_threshold: float = 0.15  # VaR 15%で緊急停止
    emergency_stop_drawdown_threshold: float = 0.10  # ドローダウン 10%で緊急停止
    emergency_stop_volatility_threshold: float = 0.05  # ボラティリティ 5%で緊急停止

    # リスク制限
    max_portfolio_var_pct: float = 0.10  # ポートフォリオVaR最大 10%
    max_single_position_var_pct: float = 0.05  # 単一ポジションVaR最大 5%
    max_correlation_risk_pct: float = 0.08  # 相関リスク最大 8%

    # ストレステスト設定
    stress_test_enabled: bool = True
    stress_test_scenarios: List[str] = field(
        default_factory=lambda: [
            "market_crash",
            "flash_crash",
            "high_volatility",
            "liquidity_crisis",
        ]
    )
    stress_test_frequency_hours: int = 24  # 24時間ごと

    # アラート設定
    risk_alert_enabled: bool = True
    risk_alert_levels: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.15])
    alert_cooldown_minutes: int = 30

    # リスク調整
    dynamic_risk_adjustment: bool = True
    risk_adjustment_factor: float = 0.8  # リスク超過時の調整係数


@dataclass
class VaRCalculation:
    """VaR計算結果"""

    portfolio_var: float
    portfolio_var_pct: float
    expected_shortfall: float
    expected_shortfall_pct: float
    calculation_method: str
    confidence_level: float
    time_horizon: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def var_breach(self) -> bool:
        """VaR制限違反"""
        return self.portfolio_var_pct > 0.10  # 10%制限


@dataclass
class RiskMetrics:
    """リスク指標"""

    # VaR指標
    value_at_risk: float
    expected_shortfall: float

    # ボラティリティ指標
    portfolio_volatility: float
    max_drawdown: float
    current_drawdown: float

    # 相関指標
    average_correlation: float
    max_correlation: float

    # 集中度指標
    herfindahl_index: float
    largest_position_weight: float

    # 流動性指標
    average_spread: float
    volume_risk: float

    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskAlert:
    """リスクアラート"""

    alert_type: str
    severity: str  # "low", "medium", "high", "critical"
    message: str
    threshold: float
    current_value: float
    timestamp: datetime = field(default_factory=datetime.now)


class VaRCalculator:
    """VaR計算器"""

    def __init__(self, config: RiskOverlayConfig):
        self.config = config
        self.logger = get_logger(__name__)

        # 価格履歴データ
        self.price_history: Dict[str, deque] = {}
        self.return_history: Dict[str, deque] = {}

        # VaR計算結果キャッシュ
        self.var_cache: Optional[VaRCalculation] = None
        self.cache_timestamp: Optional[datetime] = None

    def update_price_data(self, symbol: str, price: float):
        """価格データを更新"""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(
                maxlen=self.config.var_calculation_window_days * 24
            )  # 1時間足
            self.return_history[symbol] = deque(
                maxlen=self.config.var_calculation_window_days * 24 - 1
            )

        # 価格を追加
        self.price_history[symbol].append(price)

        # リターンを計算
        if len(self.price_history[symbol]) >= 2:
            prices = list(self.price_history[symbol])
            returns = np.diff(prices) / prices[:-1]
            self.return_history[symbol] = deque(returns, maxlen=len(returns))

    def calculate_portfolio_var(
        self,
        positions: Dict[str, Any],
        volatilities: Dict[str, float],
        correlations: Dict[Tuple[str, str], float],
    ) -> VaRCalculation:
        """ポートフォリオVaRを計算"""
        try:
            if not positions:
                return VaRCalculation(
                    portfolio_var=0.0,
                    portfolio_var_pct=0.0,
                    expected_shortfall=0.0,
                    expected_shortfall_pct=0.0,
                    calculation_method="empty_portfolio",
                    confidence_level=self.config.var_confidence_level,
                    time_horizon=self.config.var_time_horizon_days,
                )

            # ヒストリカルVaR計算
            portfolio_returns = self._calculate_portfolio_returns(
                positions, correlations
            )

            if len(portfolio_returns) < 10:
                # データ不足時はパラメトリックVaR
                return self._calculate_parametric_var(
                    positions, volatilities, correlations
                )

            # ヒストリカルVaR
            return self._calculate_historical_var(portfolio_returns, positions)

        except Exception as e:
            self.logger.error(f"VaR calculation failed: {e}")
            return VaRCalculation(
                portfolio_var=0.0,
                portfolio_var_pct=0.0,
                expected_shortfall=0.0,
                expected_shortfall_pct=0.0,
                calculation_method="error",
                confidence_level=self.config.var_confidence_level,
                time_horizon=self.config.var_time_horizon_days,
            )

    def _calculate_portfolio_returns(
        self, positions: Dict[str, Any], correlations: Dict[Tuple[str, str], float]
    ) -> np.ndarray:
        """ポートフォリオリターンを計算"""
        symbols = list(positions.keys())
        if not symbols:
            return np.array([])

        # 各シンボルのリターンを取得
        symbol_returns = []
        for symbol in symbols:
            if symbol in self.return_history and len(self.return_history[symbol]) > 0:
                symbol_returns.append(np.array(self.return_history[symbol]))
            else:
                # デフォルトのリターン（ゼロ）
                symbol_returns.append(np.zeros(100))

        # 相関行列を作成
        n_symbols = len(symbols)
        corr_matrix = np.eye(n_symbols)

        for i in range(n_symbols):
            for j in range(i + 1, n_symbols):
                symbol_i, symbol_j = symbols[i], symbols[j]
                corr = correlations.get((symbol_i, symbol_j), 0.5)  # デフォルト相関0.5
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        # ポートフォリオウェイトを計算
        total_value = sum(pos.market_value for pos in positions.values())
        weights = np.array(
            [pos.market_value / total_value for pos in positions.values()]
        )

        # 共分散行列を計算
        cov_matrix = np.zeros((n_symbols, n_symbols))
        for i in range(n_symbols):
            vol_i = np.std(symbol_returns[i]) if len(symbol_returns[i]) > 0 else 0.02
            cov_matrix[i, i] = vol_i**2
            for j in range(i + 1, n_symbols):
                vol_j = (
                    np.std(symbol_returns[j]) if len(symbol_returns[j]) > 0 else 0.02
                )
                cov = corr_matrix[i, j] * vol_i * vol_j
                cov_matrix[i, j] = cov
                cov_matrix[j, i] = cov

        # ポートフォリオボラティリティを計算
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)

        # ポートフォリオリターンをシミュレーション
        n_simulations = 1000
        portfolio_returns = np.random.normal(0, portfolio_vol, n_simulations)

        return portfolio_returns

    def _calculate_parametric_var(
        self,
        positions: Dict[str, Any],
        volatilities: Dict[str, float],
        correlations: Dict[Tuple[str, str], float],
    ) -> VaRCalculation:
        """パラメトリックVaRを計算"""
        total_value = sum(pos.market_value for pos in positions.values())

        if total_value == 0:
            return VaRCalculation(
                portfolio_var=0.0,
                portfolio_var_pct=0.0,
                expected_shortfall=0.0,
                expected_shortfall_pct=0.0,
                calculation_method="parametric",
                confidence_level=self.config.var_confidence_level,
                time_horizon=self.config.var_time_horizon_days,
            )

        # ポートフォリオボラティリティを計算
        symbols = list(positions.keys())
        weights = np.array(
            [pos.market_value / total_value for pos in positions.values()]
        )

        # 共分散行列
        n_symbols = len(symbols)
        cov_matrix = np.zeros((n_symbols, n_symbols))

        for i in range(n_symbols):
            vol_i = volatilities.get(symbols[i], 0.02)
            cov_matrix[i, i] = vol_i**2
            for j in range(i + 1, n_symbols):
                vol_j = volatilities.get(symbols[j], 0.02)
                corr = correlations.get((symbols[i], symbols[j]), 0.5)
                cov = corr * vol_i * vol_j
                cov_matrix[i, j] = cov
                cov_matrix[j, i] = cov

        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)

        # 正規分布VaR
        z_score = stats.norm.ppf(1 - self.config.var_confidence_level)
        portfolio_var = (
            abs(z_score)
            * portfolio_vol
            * total_value
            * np.sqrt(self.config.var_time_horizon_days)
        )

        # Expected Shortfall (CVaR)
        expected_shortfall = (
            portfolio_vol
            * total_value
            * np.sqrt(self.config.var_time_horizon_days)
            * (stats.norm.pdf(z_score) / (1 - self.config.var_confidence_level))
        )

        return VaRCalculation(
            portfolio_var=portfolio_var,
            portfolio_var_pct=portfolio_var / total_value,
            expected_shortfall=expected_shortfall,
            expected_shortfall_pct=expected_shortfall / total_value,
            calculation_method="parametric",
            confidence_level=self.config.var_confidence_level,
            time_horizon=self.config.var_time_horizon_days,
        )

    def _calculate_historical_var(
        self, portfolio_returns: np.ndarray, positions: Dict[str, Any]
    ) -> VaRCalculation:
        """ヒストリカルVaRを計算"""
        total_value = sum(pos.market_value for pos in positions.values())

        if len(portfolio_returns) == 0 or total_value == 0:
            return VaRCalculation(
                portfolio_var=0.0,
                portfolio_var_pct=0.0,
                expected_shortfall=0.0,
                expected_shortfall_pct=0.0,
                calculation_method="historical",
                confidence_level=self.config.var_confidence_level,
                time_horizon=self.config.var_time_horizon_days,
            )

        # VaRを計算
        var_percentile = (1 - self.config.var_confidence_level) * 100
        portfolio_var_pct = np.percentile(portfolio_returns, var_percentile)

        # スケーリング（1日→複数日）
        portfolio_var_pct *= np.sqrt(self.config.var_time_horizon_days)

        portfolio_var = abs(portfolio_var_pct) * total_value

        # Expected Shortfall
        tail_returns = portfolio_returns[portfolio_returns <= portfolio_var_pct]
        expected_shortfall_pct = (
            np.mean(tail_returns) if len(tail_returns) > 0 else portfolio_var_pct
        )
        expected_shortfall_pct *= np.sqrt(self.config.var_time_horizon_days)
        expected_shortfall = abs(expected_shortfall_pct) * total_value

        return VaRCalculation(
            portfolio_var=portfolio_var,
            portfolio_var_pct=abs(portfolio_var_pct),
            expected_shortfall=expected_shortfall,
            expected_shortfall_pct=abs(expected_shortfall_pct),
            calculation_method="historical",
            confidence_level=self.config.var_confidence_level,
            time_horizon=self.config.var_time_horizon_days,
        )


class StressTester:
    """ストレステスト実行器"""

    def __init__(self, config: RiskOverlayConfig):
        self.config = config
        self.logger = get_logger(__name__)

        # ストレスシナリオ定義
        self.scenarios = {
            "market_crash": {
                "description": "市場暴落シナリオ",
                "shock_returns": {"btc_jpy": -0.15, "eth_jpy": -0.20, "xrp_jpy": -0.10},
                "volatility_multiplier": 2.0,
            },
            "flash_crash": {
                "description": "瞬間暴落シナリオ",
                "shock_returns": {"btc_jpy": -0.30, "eth_jpy": -0.35, "xrp_jpy": -0.25},
                "volatility_multiplier": 3.0,
            },
            "high_volatility": {
                "description": "高ボラティリティシナリオ",
                "shock_returns": {},
                "volatility_multiplier": 2.5,
            },
            "liquidity_crisis": {
                "description": "流動性危機シナリオ",
                "shock_returns": {"btc_jpy": -0.05, "eth_jpy": -0.08, "xrp_jpy": -0.03},
                "volatility_multiplier": 1.8,
                "spread_multiplier": 5.0,
            },
        }

    def run_stress_test(
        self, portfolio_state: PortfolioState, current_prices: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """ストレステストを実行"""
        results = {}

        for scenario_name in self.config.stress_test_scenarios:
            if scenario_name not in self.scenarios:
                continue

            scenario = self.scenarios[scenario_name]
            scenario_results = self._run_single_scenario(
                scenario, portfolio_state, current_prices
            )
            results[scenario_name] = scenario_results

        return results

    def _run_single_scenario(
        self,
        scenario: Dict[str, Any],
        portfolio_state: PortfolioState,
        current_prices: Dict[str, float],
    ) -> Dict[str, float]:
        """単一シナリオを実行"""
        # ショック後の価格を計算
        shocked_prices = {}
        for symbol, current_price in current_prices.items():
            shock_return = scenario["shock_returns"].get(symbol, 0.0)
            shocked_prices[symbol] = current_price * (1 + shock_return)

        # ポートフォリオ価値の変化を計算
        total_loss = 0.0
        for symbol, position in portfolio_state.positions.items():
            if symbol not in shocked_prices:
                continue

            current_value = position.market_value
            shocked_value = position.quantity * shocked_prices[symbol]
            loss = current_value - shocked_value
            total_loss += loss

        # 損失率を計算
        loss_pct = (
            total_loss / portfolio_state.total_value
            if portfolio_state.total_value > 0
            else 0
        )

        # ボラティリティ調整
        vol_multiplier = scenario.get("volatility_multiplier", 1.0)
        adjusted_volatility = portfolio_state.total_risk * vol_multiplier

        return {
            "loss_amount": total_loss,
            "loss_percentage": loss_pct,
            "adjusted_volatility": adjusted_volatility,
            "breach_threshold": loss_pct > 0.20,  # 20%損失で重大
        }


class EmergencyStopSystem:
    """緊急停止システム"""

    def __init__(self, config: RiskOverlayConfig):
        self.config = config
        self.logger = get_logger(__name__)

        # 緊急停止状態
        self.emergency_stop_triggered = False
        self.emergency_stop_reason = ""
        self.emergency_stop_timestamp = None

        # ドローダウン追跡
        self.peak_value = 0.0
        self.current_drawdown = 0.0

    def check_emergency_conditions(
        self,
        portfolio_state: PortfolioState,
        var_calculation: VaRCalculation,
        volatility: float,
    ) -> bool:
        """緊急停止条件をチェック"""
        if not self.config.emergency_stop_enabled:
            return False

        # VaR閾値チェック
        if var_calculation.portfolio_var_pct > self.config.emergency_stop_var_threshold:
            self._trigger_emergency_stop(
                f"VaR threshold exceeded: {var_calculation.portfolio_var_pct:.2%}"
            )
            return True

        # ドローダウンチェック
        self._update_drawdown(portfolio_state.total_value)
        if self.current_drawdown > self.config.emergency_stop_drawdown_threshold:
            self._trigger_emergency_stop(
                f"Drawdown threshold exceeded: {self.current_drawdown:.2%}"
            )
            return True

        # ボラティリティチェック
        if volatility > self.config.emergency_stop_volatility_threshold:
            self._trigger_emergency_stop(
                f"Volatility threshold exceeded: {volatility:.2%}"
            )
            return True

        return False

    def _update_drawdown(self, current_value: float):
        """ドローダウンを更新"""
        if current_value > self.peak_value:
            self.peak_value = current_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value

    def _trigger_emergency_stop(self, reason: str):
        """緊急停止を発動"""
        if not self.emergency_stop_triggered:
            self.emergency_stop_triggered = True
            self.emergency_stop_reason = reason
            self.emergency_stop_timestamp = datetime.now()

            self.logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")

    def reset_emergency_stop(self):
        """緊急停止をリセット"""
        self.emergency_stop_triggered = False
        self.emergency_stop_reason = ""
        self.emergency_stop_timestamp = None
        self.logger.info("Emergency stop reset")

    def get_emergency_status(self) -> Dict[str, Any]:
        """緊急停止状態を取得"""
        return {
            "triggered": self.emergency_stop_triggered,
            "reason": self.emergency_stop_reason,
            "timestamp": self.emergency_stop_timestamp,
            "current_drawdown": self.current_drawdown,
        }


class RiskOverlay:
    """
    V433 Phase 3: リスクオーバーレイシステム
    VaR計算、緊急停止機能、リアルタイムリスク監視
    """

    def __init__(self, position_manager: PositionManager):
        self.position_manager = position_manager
        self.logger = get_logger(__name__)

        # 設定の初期化
        self.config = RiskOverlayConfig()

        # コンポーネントの初期化
        self.var_calculator = VaRCalculator(self.config)
        self.stress_tester = StressTester(self.config)
        self.emergency_stop = EmergencyStopSystem(self.config)

        # 状態管理
        self.risk_metrics: Optional[RiskMetrics] = None
        self.last_var_calculation: Optional[VaRCalculation] = None
        self.alerts: List[RiskAlert] = []

        # モニタリング
        self.monitoring_thread = None
        self.is_running = False

        # 価格データ
        self.current_prices: Dict[str, float] = {}
        self.price_update_times: Dict[str, datetime] = {}

    def start_overlay(self):
        """リスクオーバーレイを開始"""
        if self.is_running:
            return

        self.is_running = True

        # モニタリングスレッド開始
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

        self.logger.info("Risk overlay started")

    def stop_overlay(self):
        """リスクオーバーレイを停止"""
        self.is_running = False

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)

        self.logger.info("Risk overlay stopped")

    def update_price(self, symbol: str, price: float):
        """価格を更新"""
        self.current_prices[symbol] = price
        self.price_update_times[symbol] = datetime.now()

        # VaR計算器に価格データを更新
        self.var_calculator.update_price_data(symbol, price)

    def calculate_risk_metrics(self) -> RiskMetrics:
        """リスク指標を計算"""
        try:
            portfolio_state = self.position_manager.portfolio_state

            # VaR計算
            var_calc = self.var_calculator.calculate_portfolio_var(
                portfolio_state.positions,
                self.position_manager.volatilities,
                self.position_manager.correlations,
            )
            self.last_var_calculation = var_calc

            # ボラティリティ指標
            portfolio_volatility = (
                np.sqrt(
                    sum(
                        vol**2 * (pos.market_value / portfolio_state.total_value) ** 2
                        for pos in portfolio_state.positions.values()
                        for vol in [
                            self.position_manager.volatilities.get(pos.symbol, 0.02)
                        ]
                    )
                )
                if portfolio_state.positions
                else 0.0
            )

            # ドローダウン計算
            max_drawdown = self._calculate_max_drawdown()
            current_drawdown = self.emergency_stop.current_drawdown

            # 相関指標
            avg_correlation, max_correlation = self._calculate_correlation_metrics()

            # 集中度指標
            herfindahl_index, largest_weight = self._calculate_concentration_metrics(
                portfolio_state
            )

            # 流動性指標（簡易版）
            avg_spread = 0.001  # 仮定値
            volume_risk = 0.01  # 仮定値

            self.risk_metrics = RiskMetrics(
                value_at_risk=var_calc.portfolio_var,
                expected_shortfall=var_calc.expected_shortfall,
                portfolio_volatility=portfolio_volatility,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                average_correlation=avg_correlation,
                max_correlation=max_correlation,
                herfindahl_index=herfindahl_index,
                largest_position_weight=largest_weight,
                average_spread=avg_spread,
                volume_risk=volume_risk,
            )

            return self.risk_metrics

        except Exception as e:
            self.logger.error(f"Risk metrics calculation failed: {e}")
            return self._get_default_risk_metrics()

    def _calculate_max_drawdown(self) -> float:
        """最大ドローダウンを計算"""
        # 簡易版：過去30日の価格データから計算
        # 実際の実装ではより詳細な履歴が必要
        # TODO: Use ztb.metrics.metrics.max_drawdown when historical data is available
        return 0.05  # 仮定値

    def _calculate_correlation_metrics(self) -> Tuple[float, float]:
        """相関指標を計算"""
        correlations = list(self.position_manager.correlations.values())
        if not correlations:
            return 0.0, 0.0

        avg_correlation = np.mean(correlations)
        max_correlation = max(correlations) if correlations else 0.0

        return avg_correlation, max_correlation

    def _calculate_concentration_metrics(
        self, portfolio_state: PortfolioState
    ) -> Tuple[float, float]:
        """集中度指標を計算"""
        if not portfolio_state.positions:
            return 0.0, 0.0

        # Herfindahl-Hirschman Index
        weights = [
            pos.market_value / portfolio_state.total_value
            for pos in portfolio_state.positions.values()
        ]
        herfindahl = sum(w**2 for w in weights)

        largest_weight = max(weights) if weights else 0.0

        return herfindahl, largest_weight

    def _get_default_risk_metrics(self) -> RiskMetrics:
        """デフォルトのリスク指標を返す"""
        return RiskMetrics(
            value_at_risk=0.0,
            expected_shortfall=0.0,
            portfolio_volatility=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            average_correlation=0.0,
            max_correlation=0.0,
            herfindahl_index=0.0,
            largest_position_weight=0.0,
            average_spread=0.0,
            volume_risk=0.0,
        )

    def run_stress_tests(self) -> Dict[str, Dict[str, float]]:
        """ストレステストを実行"""
        portfolio_state = self.position_manager.portfolio_state
        return self.stress_tester.run_stress_test(portfolio_state, self.current_prices)

    def check_emergency_conditions(self) -> bool:
        """緊急停止条件をチェック"""
        if not self.last_var_calculation:
            return False

        portfolio_volatility = (
            self.risk_metrics.portfolio_volatility if self.risk_metrics else 0.0
        )

        return self.emergency_stop.check_emergency_conditions(
            self.position_manager.portfolio_state,
            self.last_var_calculation,
            portfolio_volatility,
        )

    def generate_risk_alerts(self) -> List[RiskAlert]:
        """リスクアラートを生成"""
        alerts = []

        if not self.risk_metrics or not self.last_var_calculation:
            return alerts

        # VaRアラート
        var_pct = self.last_var_calculation.portfolio_var_pct
        for threshold in self.config.risk_alert_levels:
            if var_pct > threshold:
                alerts.append(
                    RiskAlert(
                        alert_type="var_breach",
                        severity=self._get_severity(threshold),
                        message=f"Portfolio VaR exceeded {threshold:.0%}: {var_pct:.2%}",
                        threshold=threshold,
                        current_value=var_pct,
                    )
                )
                break

        # ドローダウンアラート
        drawdown = self.risk_metrics.current_drawdown
        if drawdown > 0.05:  # 5%ドローダウン
            severity = "high" if drawdown > 0.10 else "medium"
            alerts.append(
                RiskAlert(
                    alert_type="drawdown",
                    severity=severity,
                    message=f"Portfolio drawdown: {drawdown:.2%}",
                    threshold=0.05,
                    current_value=drawdown,
                )
            )

        # 集中度アラート
        if self.risk_metrics.largest_position_weight > 0.5:  # 50%集中
            alerts.append(
                RiskAlert(
                    alert_type="concentration",
                    severity="medium",
                    message=f"High position concentration: {self.risk_metrics.largest_position_weight:.2%}",
                    threshold=0.5,
                    current_value=self.risk_metrics.largest_position_weight,
                )
            )

        self.alerts.extend(alerts)
        return alerts

    def _get_severity(self, threshold: float) -> str:
        """アラート重要度を取得"""
        if threshold >= 0.15:
            return "critical"
        elif threshold >= 0.10:
            return "high"
        elif threshold >= 0.05:
            return "medium"
        else:
            return "low"

    def _monitoring_loop(self):
        """モニタリングループ"""
        last_update = datetime.now()

        while self.is_running:
            try:
                current_time = datetime.now()

                # VaR更新チェック
                if (
                    current_time - last_update
                ).seconds >= self.config.var_update_interval_seconds:
                    # リスク指標計算
                    self.calculate_risk_metrics()

                    # 緊急停止チェック
                    emergency_triggered = self.check_emergency_conditions()

                    # アラート生成
                    if self.config.risk_alert_enabled:
                        alerts = self.generate_risk_alerts()
                        for alert in alerts:
                            self.logger.warning(
                                f"RISK ALERT: {alert.alert_type} - {alert.message}"
                            )

                    # ストレステスト（定期実行）
                    if self.config.stress_test_enabled:
                        stress_results = self.run_stress_tests()
                        critical_scenarios = [
                            s
                            for s, r in stress_results.items()
                            if r.get("breach_threshold", False)
                        ]
                        if critical_scenarios:
                            self.logger.warning(
                                f"Critical stress test scenarios: {critical_scenarios}"
                            )

                    last_update = current_time

                time.sleep(10)  # 10秒間隔

            except Exception as e:
                self.logger.error(f"Risk overlay monitoring error: {e}")
                time.sleep(30)

    def get_risk_report(self) -> Dict[str, Any]:
        """リスクレポートを取得"""
        return {
            "risk_metrics": self.risk_metrics.__dict__ if self.risk_metrics else None,
            "var_calculation": self.last_var_calculation.__dict__
            if self.last_var_calculation
            else None,
            "emergency_status": self.emergency_stop.get_emergency_status(),
            "active_alerts": [
                alert.__dict__ for alert in self.alerts[-10:]
            ],  # 最新10件
            "stress_test_results": self.run_stress_tests(),
            "portfolio_exposure": self.position_manager.portfolio_state.used_capital,
            "risk_limits": {
                "max_var_pct": self.config.max_portfolio_var_pct,
                "max_single_var_pct": self.config.max_single_position_var_pct,
                "emergency_var_threshold": self.config.emergency_stop_var_threshold,
            },
        }


def create_risk_overlay(position_manager: PositionManager) -> RiskOverlay:
    """RiskOverlayのファクトリ関数"""
    return RiskOverlay(position_manager)


# 使用例
if __name__ == "__main__":
    from ztb.trading.trade_execution_engine import TradeExecutionEngine

    # 取引実行エンジンの作成
    execution_engine = TradeExecutionEngine("zaif")

    # ポジション管理システムの作成
    position_manager = PositionManager(execution_engine, "zaif")

    # リスクオーバーレイの作成
    risk_overlay = create_risk_overlay(position_manager)

    # システム開始
    execution_engine.start_execution()
    position_manager.start_management()
    risk_overlay.start_overlay()

    try:
        # サンプル価格更新
        risk_overlay.update_price("btc_jpy", 5000000)
        risk_overlay.update_price("eth_jpy", 300000)

        # リスク指標計算
        metrics = risk_overlay.calculate_risk_metrics()
        print(
            f"Risk metrics: VaR={metrics.value_at_risk:.2f}, Volatility={metrics.portfolio_volatility:.2%}"
        )

        # ストレステスト実行
        stress_results = risk_overlay.run_stress_tests()
        print(f"Stress test results: {stress_results}")

        # リスクレポート取得
        report = risk_overlay.get_risk_report()
        print(f"Risk report generated with {len(report['active_alerts'])} alerts")

        time.sleep(10)

    finally:
        # システム停止
        risk_overlay.stop_overlay()
        position_manager.stop_management()
        execution_engine.stop_execution()
