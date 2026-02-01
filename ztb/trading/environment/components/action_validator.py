# Action validation utilities for trading environment
# 取引環境のアクション検証ユーティリティ

import logging
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy.typing import NDArray

from ztb.trading.environment.constants import (
    BTC_MIN_UNIT,
)  # 最小取引単位 (0.01 mBTC, 約1,800円相当)
from ztb.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from ztb.trading.environment.utils.config import EnvironmentConfig

logger = get_logger(__name__)


class ActionValidator:
    """Handles action validation and masking for trading actions."""

    def __init__(
        self,
        config: "EnvironmentConfig",
        initial_portfolio_value: float,
    ):
        self.config = config
        self.initial_portfolio_value = initial_portfolio_value

    def get_legal_actions(
        self,
        current_step: int,
        position: float,
        total_pnl: float,
        trades_count: int,
        last_trade_step: Optional[int],
        consecutive_trade_steps: int,
        close_array: Optional[NDArray[np.float32]] = None,
        price_array: Optional[NDArray[np.float32]] = None,
        df: Optional[Any] = None,
        market_regime: Optional[str] = None,
        hybrid_filters: Optional[dict] = None,
    ) -> NDArray[np.int_]:
        """現在の状態で合法なアクションを返す（1=合法, 0=非法）"""
        legal = np.zeros(3, dtype=np.int_)  # [HOLD, BUY, SELL] - デフォルト非法

        current_price = self._resolve_price(current_step, price_array, df)
        if current_price == 0.0:
            logger.warning(
                f"Price information could not be resolved at step {current_step}. Returning only HOLD as legal action."
            )
            legal = np.zeros(3, dtype=np.int_)
            legal[0] = 1
            return legal

        portfolio_value = self.initial_portfolio_value + total_pnl
        position_size = self.config.max_position_size
        transaction_cost = self.config.transaction_cost

        # HOLDは常に合法
        legal[0] = 1

        # --- Hybrid Regime Filter (for action masking observability) ---
        if hybrid_filters and hybrid_filters.get("enabled", False) and market_regime:
            regime_filter = hybrid_filters.get("regime_filter", {})
            if regime_filter.get("enabled", False):
                mode = str(regime_filter.get("mode", "hard")).lower()
                permission_raw: Any = None

                if mode == "soft":
                    constraints = regime_filter.get("regime_constraints", {})
                    if isinstance(constraints, dict):
                        constraint = constraints.get(str(market_regime))
                        if isinstance(constraint, dict):
                            permission_raw = constraint.get("action_permission")

                if permission_raw is None:
                    excluded_regimes = regime_filter.get("excluded_regimes", [])
                    if market_regime in excluded_regimes:
                        permission_raw = "deny"

                if str(permission_raw or "allow").lower() == "deny":
                    # Allow HOLD always; allow closing existing position for risk management.
                    if position > 0:
                        legal[2] = 1  # SELL to close long
                    elif position < 0:
                        legal[1] = 1  # BUY to close short
                    return legal

        # 取引所別取引頻度制限（Coincheckは手数料無料なので制限緩和）
        exchange = getattr(self.config, "exchange", "coincheck").lower()
        if exchange != "coincheck":
            # Coincheck以外は取引頻度制限を適用
            # 最小ホールド期間チェック
            min_holding_period = getattr(self.config, "min_holding_period", 3)
            if last_trade_step is not None:
                steps_since_last_trade = current_step - last_trade_step
                if steps_since_last_trade < min_holding_period:
                    # 最小ホールド期間中でも、ポジションクローズは許可（リスク管理上重要）
                    if position > 0:
                        # ロングポジション保有中: SELLでクローズ可能
                        legal[2] = 1
                    elif position < 0:
                        # ショートポジション保有中: BUYでクローズ可能
                        legal[1] = 1
                    # その他の新規建ては制限
                    return legal

            # 連続取引制限チェック
            max_consecutive_trades = getattr(self.config, "max_consecutive_trades", 5)
            if consecutive_trade_steps >= max_consecutive_trades:
                # 連続取引上限に達した場合もポジションクローズは許可
                if position > 0:
                    legal[2] = 1  # SELL to close long
                elif position < 0:
                    legal[1] = 1  # BUY to close short
                return legal

        # 市場ボラティリティチェック（高ボラティリティ時は取引制限）
        volatility_threshold = getattr(
            self.config, "volatility_trade_threshold", 0.02
        )  # 2%ボラティリティ閾値
        if current_step > 20 and self._check_volatility(
            current_step, volatility_threshold, close_array, price_array, df
        ):
            return legal

        # Initialize variables to avoid UnboundLocalError
        ideal_buy_cost = 0.0
        ideal_sell_value = 0.0
        affordable_size = 0.0

        # BUY: 常に許可（資金があれば）
        # 🔧 CRITICAL FIX: アクションバイアスを排除するため常に許可
        # - 実際の取引ではポジションに関係なくBUY可能
        # - 資金があればいつでも買える
        ideal_buy_cost = position_size * current_price * (1 + transaction_cost)
        affordable_size = (
            portfolio_value * 0.9 / (current_price * (1 + transaction_cost))
        )
        min_purchase_amount = 10000.0
        min_affordable_value = affordable_size * current_price
        if portfolio_value >= ideal_buy_cost or (
            affordable_size >= BTC_MIN_UNIT
            and min_affordable_value >= min_purchase_amount
        ):
            legal[1] = 1

        # SELL: 常に許可（資金があれば）
        # 🔧 CRITICAL FIX: アクションバイアスを排除するため常に許可
        # - 実際の取引ではポジションに関係なくSELL可能
        # - 資金があればいつでも売れる
        ideal_sell_value = position_size * current_price
        affordable_size = portfolio_value * 0.9 / current_price
        min_sell_amount = 10000.0
        min_affordable_value = affordable_size * current_price
        if portfolio_value >= ideal_sell_value or (
            affordable_size >= BTC_MIN_UNIT and min_affordable_value >= min_sell_amount
        ):
            legal[2] = 1

        # HOLDは常に合法なので、全て0になることはない
        # Emit debug info unconditionally (logger.debug will be a no-op if level > DEBUG)
        logger.debug(
            "ActionValidator: legal_actions=%s, affordable_size=%.6f, "
            "ideal_buy_cost=%.2f, ideal_sell_value=%.2f, portfolio_value=%.2f, "
            "position=%.6f",
            [bool(x) for x in legal],
            affordable_size,
            ideal_buy_cost,
            ideal_sell_value,
            portfolio_value,
            position,
        )
        return legal

    def _resolve_price(
        self,
        current_step: int,
        price_array: Optional[NDArray[np.float32]],
        df: Optional[Any],
    ) -> float:
        """Resolve current price for action validation."""
        if price_array is not None and price_array.size > current_step:
            return float(price_array[current_step])

        if df is not None and hasattr(df, "iloc"):
            try:
                row = df.iloc[current_step]
                for column in ("price", "close", "adj_close", "open"):
                    if column in row.index:
                        value = row[column]
                        if np.isfinite(value):
                            return float(value)
            except (IndexError, KeyError):
                pass

        return 0.0

    def _check_volatility(
        self,
        current_step: int,
        threshold: float,
        close_array: Optional[NDArray[np.float32]],
        price_array: Optional[NDArray[np.float32]],
        df: Optional[Any],
    ) -> bool:
        """Check if current market volatility exceeds threshold."""
        price_slice: Optional[NDArray[np.float32]] = None
        start_idx = max(0, current_step - 20)
        end_idx = current_step

        if (
            close_array is not None
            and close_array.size >= end_idx
            and end_idx > start_idx
        ):
            price_slice = close_array[start_idx:end_idx]
        elif (
            price_array is not None
            and price_array.size >= end_idx
            and end_idx > start_idx
        ):
            price_slice = price_array[start_idx:end_idx]
        elif df is not None:
            try:
                recent_prices = df.iloc[start_idx:end_idx]["close"]
                with np.errstate(divide="ignore", invalid="ignore"):
                    denominators = recent_prices[:-1]
                    returns = np.diff(recent_prices) / np.where(
                        denominators == 0.0, np.nan, denominators
                    )
                returns = returns[np.isfinite(returns)]
                if returns.size:
                    current_volatility = float(np.std(returns))
                    return current_volatility > threshold
            except (IndexError, KeyError):
                pass
        return False

    def action_mask(self, legal_actions: NDArray[np.int_]) -> NDArray[np.bool_]:
        """Return action mask for gymnasium ActionMasker wrapper."""
        return legal_actions.astype(np.bool_)

    def get_action_masks(self, legal_actions: NDArray[np.int_]) -> NDArray[np.bool_]:
        """Return action masks for SB3 MaskablePPO."""
        return self.action_mask(legal_actions)
