"""
Risk Manager for Position Management.

This module provides comprehensive risk management functionality including:
- Portfolio risk calculation
- Risk limit checking
- Stop loss and take profit calculations
"""

from typing import Any

import pandas as pd

from ztb.trading.risk.interfaces import RiskManagerProtocol
from ztb.trading.trade_execution_engine import Position
from ztb.trading.types import PositionManagementConfig, PortfolioState
from ztb.utils.logging_utils import get_logger
from ztb.utils.errors import validate_price

class RiskManager(RiskManagerProtocol):
    """リスクマネージャー"""

    def __init__(self, config: PositionManagementConfig):
        self.config = config
        self.logger = get_logger(__name__)

        # RiskManagerProtocol attributes
        self.test_mode = False
        self.portfolio_value = 1.0

        # リスク追跡
        self.portfolio_risk = 0.0
        self.position_risks: dict[str, float] = {}

    def calculate_portfolio_risk(
        self, positions: dict[str, Position], volatilities: dict[str, float]
    ) -> float:
        """ポートフォリオリスクを計算"""
        if not positions:
            return 0.0

        total_risk = 0.0
        position_weights = {}

        # 各ポジションのリスクを計算
        for symbol, position in positions.items():
            vol = volatilities.get(symbol, 0.02)  # デフォルト2%
            position_value = position.market_value

            # 簡易VaR計算 (1日, 95%信頼区間)
            position_var = position_value * vol * 1.645  # 正規分布の95%分位点

            self.position_risks[symbol] = position_var
            total_risk += position_var

            position_weights[symbol] = position_value

        # 相関リスクの考慮 (簡易版)
        # 実際には相関係数行列を使って計算
        correlation_factor = 1.2  # 相関調整係数
        total_risk *= correlation_factor

        self.portfolio_risk = total_risk
        return total_risk

    def check_risk_limits(self, portfolio_state: PortfolioState) -> dict[str, bool]:
        """リスク制限をチェック"""
        checks = {
            "portfolio_risk_ok": True,
            "single_position_risk_ok": True,
            "capital_buffer_ok": True,
        }

        # ポートフォリオリスクチェック
        if (
            portfolio_state.total_risk
            > portfolio_state.total_capital * self.config.max_portfolio_risk_pct
        ):
            checks["portfolio_risk_ok"] = False
            self.logger.warning(
                f"Portfolio risk limit exceeded: "
                f"{portfolio_state.total_risk:.2f} > "
                f"{portfolio_state.total_capital * self.config.max_portfolio_risk_pct:.2f}"
            )

        # 単一ポジションリスクチェック
        for symbol, risk in self.position_risks.items():
            if (
                risk
                > portfolio_state.total_capital
                * self.config.max_single_position_risk_pct
            ):
                checks["single_position_risk_ok"] = False
                self.logger.warning(
                    f"Single position risk limit exceeded for {symbol}: "
                        f"{risk:.2f} > "
                        f"{portfolio_state.total_capital * self.config.max_single_position_risk_pct:.2f}"
                )

        # 資本バッファチェック
        capital_buffer = portfolio_state.total_capital * self.config.capital_buffer_pct
        if portfolio_state.available_capital < capital_buffer:
            checks["capital_buffer_ok"] = False
            self.logger.warning(
                f"Capital buffer insufficient: "
                f"{portfolio_state.available_capital:.2f} < {capital_buffer:.2f}"
            )

        return checks

    def calculate_stop_loss_price(
        self, position: Position, current_price: float
    ) -> float:
        """ストップロス価格を計算"""
        validate_price(current_price, "current_price")

        if position.quantity > 0:  # ロングポジション
            stop_price = current_price * (1 - self.config.stop_loss_pct)
        else:  # ショートポジション
            stop_price = current_price * (1 + self.config.stop_loss_pct)

        return stop_price

    def calculate_take_profit_price(
        self, position: Position, current_price: float
    ) -> float:
        """テイクプロフィット価格を計算"""
        validate_price(current_price, "current_price")

        if position.quantity > 0:  # ロングポジション
            take_price = current_price * (1 + self.config.take_profit_pct)
        else:  # ショートポジション
            take_price = current_price * (1 - self.config.take_profit_pct)

        return take_price

    def should_stop_loss(self, position: Position, current_price: float) -> bool:
        """ストップロスを実行すべきか"""
        stop_price = self.calculate_stop_loss_price(position, position.average_price)

        if position.quantity > 0:  # ロング
            return current_price <= stop_price
        else:  # ショート
            return current_price >= stop_price

    def should_take_profit(self, position: Position, current_price: float) -> bool:
        """テイクプロフィットを実行すべきか"""
        take_price = self.calculate_take_profit_price(position, position.average_price)

        if position.quantity > 0:  # ロング
            return current_price >= take_price
        else:  # ショート
            return current_price <= take_price

    # RiskManagerProtocol implementation
    def should_open_position(
        self,
        signal_strength: float,
        market_volatility: float,
        current_portfolio_value: float,
    ) -> bool:
        """ポジションを開くべきか判断"""
        # ポートフォリオリスクチェック
        if self.portfolio_risk > current_portfolio_value * self.config.max_portfolio_risk_pct:
            return False

        # 信号強度チェック
        if signal_strength < self.config.min_signal_strength:
            return False

        # 市場ボラティリティチェック
        if market_volatility > self.config.max_volatility_threshold:
            return False

        return True

    def should_close_position(
        self,
        position_data: dict[str, Any],
        current_price: float,
        current_portfolio_value: float,
    ) -> tuple[bool, str]:
        """ポジションを閉じるべきか判断"""
        # position_data から Position オブジェクトを作成
        position = Position(
            symbol=position_data.get("symbol", ""),
            quantity=position_data.get("quantity", 0),
            average_price=position_data.get("average_price", current_price),
            market_value=position_data.get("market_value", 0),
        )

        # ストップロスチェック
        if self.should_stop_loss(position, current_price):
            return True, "stop_loss"

        # テイクプロフィットチェック
        if self.should_take_profit(position, current_price):
            return True, "take_profit"

        return False, ""

    def get_risk_adjusted_position_size(
        self, signal_strength: float, market_volatility: float
    ) -> float:
        """リスク調整済みポジションサイズを計算"""
        # 基本ポジションサイズ
        base_size = self.config.default_position_size_pct

        # 信号強度による調整
        signal_multiplier = min(signal_strength / self.config.min_signal_strength, 2.0)

        # ボラティリティによる調整
        vol_multiplier = 1.0 / max(market_volatility, 0.01)

        # リスク制限を適用
        adjusted_size = base_size * signal_multiplier * vol_multiplier
        adjusted_size = min(adjusted_size, self.config.max_single_position_risk_pct)

        return adjusted_size

    def calculate_atr_stop_levels(
        self, data: pd.DataFrame, entry_price: float, position_type: str
    ) -> tuple[float, float]:
        """ATRベースのストップレベルを計算"""
        if "atr" not in data.columns:
            # ATRがない場合は固定パーセントを使用
            stop_loss_pct = self.config.stop_loss_pct
            take_profit_pct = self.config.take_profit_pct

            if position_type == "long":
                stop_price = entry_price * (1 - stop_loss_pct)
                take_price = entry_price * (1 + take_profit_pct)
            else:
                stop_price = entry_price * (1 + stop_loss_pct)
                take_price = entry_price * (1 - take_profit_pct)

            return stop_price, take_price

        # ATRを使用した計算
        current_atr = data["atr"].iloc[-1]
        atr_multiplier_sl = getattr(self.config, "atr_stop_loss_multiplier", 2.0)
        atr_multiplier_tp = getattr(self.config, "atr_take_profit_multiplier", 4.0)

        if position_type == "long":
            stop_price = entry_price - (current_atr * atr_multiplier_sl)
            take_price = entry_price + (current_atr * atr_multiplier_tp)
        else:
            stop_price = entry_price + (current_atr * atr_multiplier_sl)
            take_price = entry_price - (current_atr * atr_multiplier_tp)

        return stop_price, take_price

    def update_risk_metrics(
        self, trade_result: dict[str, Any] | None = None
    ) -> None:
        """リスクメトリクスを更新"""
        if trade_result:
            # 取引結果に基づいてリスクを更新
            pnl = trade_result.get("pnl", 0)
            if pnl < 0:
                self.portfolio_risk += abs(pnl)
            else:
                # 利益の場合はリスクを減少
                self.portfolio_risk = max(0, self.portfolio_risk - pnl * 0.1)

    def reset(self) -> None:
        """リスクマネージャーをリセット"""
        self.portfolio_risk = 0.0
        self.position_risks.clear()
        self.portfolio_value = 1.0