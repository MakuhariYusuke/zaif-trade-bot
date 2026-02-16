#!/usr/bin/env python3
"""
V433 Phase 3: 取引実行エンジン
現実的取引コストと動的ポジションサイジングを考慮した実行システム
"""

import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from ztb.trading.environment.constants import (
    BASIS_POINTS,
    DEFAULT_FEE_RATE,
    DEFAULT_MAX_TRADE_SIZE_JPY,
    DEFAULT_TOTAL_CAPITAL,
    MAXIMUM_FEE_RATE,
)
from ztb.types.common import BaseComponent
from ztb.utils.errors import (
    validate_non_negative,
    validate_positive,
    validate_price_range,
    validate_range,
)
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TransactionCostConfig:
    """取引コスト設定"""

    # 取引所別手数料設定
    exchange_fees: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "zaif": {
                "maker_fee": 0.0,  # メイカー手数料
                "taker_fee": DEFAULT_FEE_RATE,  # テイカー手数料 (0.1%)
                "minimum_fee": 0.0,  # 最小手数料
                "maximum_fee": MAXIMUM_FEE_RATE,  # 最大手数料
            },
            "bitflyer": {
                "maker_fee": -0.0002,  # メイカー手数料 (マイナス = 報酬)
                "taker_fee": 0.0012,  # テイカー手数料 (0.12%)
                "minimum_fee": 0.0,
                "maximum_fee": MAXIMUM_FEE_RATE,
            },
            "coincheck": {
                "maker_fee": 0.0,
                "taker_fee": 0.0008,  # テイカー手数料 (0.08%)
                "minimum_fee": 0.0,
                "maximum_fee": MAXIMUM_FEE_RATE,
            },
        }
    )

    # スプレッド設定
    spread_model: str = "realistic"  # "fixed", "realistic", "adaptive"
    fixed_spread_bps: float = 2.0  # 固定スプレッド (bps)
    adaptive_spread_factor: float = 1.5  # 適応スプレッド係数

    # スリッページ設定
    enable_slippage: bool = True
    slippage_model: str = "volume_based"  # "fixed", "volume_based", "time_based"
    max_slippage_bps: float = 5.0  # 最大スリッページ (bps)

    # 市場インパクト考慮
    enable_market_impact: bool = True
    market_impact_factor: float = 0.1  # 市場インパクト係数


@dataclass
class PositionSizingConfig:
    """ポジションサイジング設定"""

    # 基本設定
    capital_utilization: float = 1.0  # 資本利用率 (100%)
    max_position_size_pct: float = 0.5  # 最大ポジションサイズ (% of capital)

    # リスクベースサイジング
    enable_risk_based_sizing: bool = True
    risk_per_trade_pct: float = 0.02  # 1トレードあたりのリスク (2%)
    max_risk_per_trade_pct: float = 0.05  # 最大リスク (5%)

    # ケリー基準適応
    enable_kelly_sizing: bool = True
    kelly_fraction: float = 0.5  # ケリー基準の使用割合

    # ボラティリティ調整
    enable_volatility_adjustment: bool = True
    volatility_lookback_days: int = 30
    volatility_target: float = 0.02  # 目標ボラティリティ

    # 最小/最大取引サイズ
    min_trade_size_jpy: float = 100.0  # 最小取引サイズ (円)
    max_trade_size_jpy: float = DEFAULT_MAX_TRADE_SIZE_JPY  # 最大取引サイズ (円)

    # 通貨ペア別最小単位
    min_trade_units: Dict[str, float] = field(
        default_factory=lambda: {
            "btc_jpy": 0.0001,  # BTC最小単位
            "eth_jpy": DEFAULT_FEE_RATE,  # ETH最小単位
            "xrp_jpy": 1.0,  # XRP最小単位
            "mona_jpy": 1.0,  # MONA最小単位
        }
    )


@dataclass
class TradeOrder:
    """取引注文"""

    order_id: str
    symbol: str
    side: str  # "buy", "sell"
    order_type: str  # "market", "limit"
    quantity: float
    price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    # 実行結果
    executed_quantity: float = 0.0
    executed_price: float = 0.0
    fee: float = 0.0
    slippage: float = 0.0
    status: str = "pending"  # "pending", "executed", "cancelled", "rejected"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "executed_quantity": self.executed_quantity,
            "executed_price": self.executed_price,
            "fee": self.fee,
            "slippage": self.slippage,
            "status": self.status,
        }


@dataclass
class Position:
    """ポジション情報"""

    symbol: str
    quantity: float
    average_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def market_value(self) -> float:
        """時価評価額"""
        return self.quantity * self.current_price


class TransactionCostCalculator(BaseComponent):
    """取引コスト計算器"""

    def __init__(
        self,
        config: TransactionCostConfig,
        component_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name="TransactionCostCalculator", config=component_config)
        self.config = config

    def calculate_fee(
        self,
        exchange: str,
        side: str,
        quantity: float,
        price: float,
        is_maker: bool = False,
    ) -> float:
        """取引手数料を計算"""
        if exchange not in self.config.exchange_fees:
            self.logger.warning(f"Unknown exchange: {exchange}, using default fees")
            exchange = "zaif"  # デフォルト

        fees = self.config.exchange_fees[exchange]

        # メイカー/テイカー手数料の選択
        if is_maker and fees["maker_fee"] != 0.0:
            fee_rate = fees["maker_fee"]
        else:
            fee_rate = fees["taker_fee"]

        # 取引金額の計算
        trade_value = quantity * price

        # 手数料計算
        fee = trade_value * abs(fee_rate)

        # 最小/最大手数料の適用
        fee = max(fee, fees["minimum_fee"])
        fee = min(fee, fees["maximum_fee"])

        return fee

    def calculate_spread(
        self, symbol: str, base_price: float, volatility: float = 0.0
    ) -> float:
        """スプレッドを計算"""
        if self.config.spread_model == "fixed":
            spread = base_price * (self.config.fixed_spread_bps / BASIS_POINTS)
        elif self.config.spread_model == "realistic":
            # ボラティリティに基づく現実的スプレッド
            base_spread = base_price * (self.config.fixed_spread_bps / BASIS_POINTS)
            vol_adjustment = base_price * (
                volatility * self.config.adaptive_spread_factor / 100
            )
            spread = base_spread + vol_adjustment
        else:
            spread = 0.0

        return spread

    def calculate_slippage(
        self,
        symbol: str,
        side: str,
        quantity: float,
        base_price: float,
        market_volume: float = 0.0,
    ) -> float:
        """スリッページを計算"""
        if not self.config.enable_slippage:
            return 0.0

        if self.config.slippage_model == "fixed":
            slippage = base_price * (self.config.max_slippage_bps / BASIS_POINTS)
        elif self.config.slippage_model == "volume_based":
            # 市場ボリュームに基づくスリッページ
            if market_volume > 0:
                impact_ratio = min(quantity / market_volume, 1.0)
                slippage = base_price * (
                    impact_ratio * self.config.max_slippage_bps / BASIS_POINTS
                )
            else:
                slippage = base_price * (self.config.max_slippage_bps / BASIS_POINTS)
        else:
            slippage = 0.0

        # 買い注文は価格が上がり、売り注文は価格が下がる
        if side == "buy":
            return slippage
        else:
            return -slippage

    def calculate_market_impact(
        self, symbol: str, quantity: float, market_volume: float = 0.0
    ) -> float:
        """市場インパクトを計算"""
        if not self.config.enable_market_impact or market_volume == 0:
            return 0.0

        # 取引量が市場ボリュームの何%かを計算
        impact_ratio = min(quantity / market_volume, 1.0)
        impact = impact_ratio * self.config.market_impact_factor

        return impact


class PositionSizer(BaseComponent):
    """ポジションサイザー"""

    def __init__(
        self,
        config: PositionSizingConfig,
        cost_calculator: TransactionCostCalculator,
        component_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name="PositionSizer", config=component_config)
        self.config = config
        self.cost_calculator = cost_calculator

        # ボラティリティ履歴
        self.volatility_history: Dict[str, List[float]] = {}

    def calculate_position_size(
        self,
        symbol: str,
        signal_strength: float,
        current_price: float,
        capital: float,
        volatility: float = 0.0,
        win_rate: float = 0.5,
    ) -> float:
        """ポジションサイズを計算"""
        try:
            # 入力バリデーション
            validate_price_range(current_price, name="current_price")
            validate_positive(capital, name="capital")
            validate_range(signal_strength, 0.0, 1.0, name="signal_strength")
            validate_non_negative(volatility, name="volatility")
            validate_range(win_rate, 0.0, 1.0, name="win_rate")

            # 基本サイズ計算
            base_size = self._calculate_base_size(capital, current_price)

            # リスクベース調整
            if self.config.enable_risk_based_sizing:
                risk_size = self._calculate_risk_based_size(
                    capital, current_price, volatility
                )
                base_size = min(base_size, risk_size)

            # ケリー基準適応
            if self.config.enable_kelly_sizing:
                kelly_size = self._calculate_kelly_size(
                    capital, current_price, win_rate, signal_strength
                )
                base_size = min(base_size, kelly_size)

            # ボラティリティ調整
            if self.config.enable_volatility_adjustment:
                vol_adjusted_size = self._adjust_for_volatility(base_size, volatility)
                base_size = vol_adjusted_size

            # シグナル強度による調整
            signal_adjusted_size = base_size * signal_strength

            # 最小/最大サイズ制約
            final_size = self._apply_size_constraints(
                symbol, signal_adjusted_size, current_price
            )

            # 最小取引単位に合わせる
            final_size = self._round_to_min_unit(symbol, final_size)

            self.logger.debug(
                f"Position size for {symbol}: {final_size:.6f} "
                f"(base: {base_size:.6f}, signal: {signal_strength:.2f})"
            )

            return final_size

        except Exception as e:
            self.logger.error(f"Position size calculation failed for {symbol}: {e}")
            return 0.0

    def _calculate_base_size(self, capital: float, price: float) -> float:
        """基本ポジションサイズを計算"""
        # 資本利用率に基づく最大サイズ
        max_size_value = (
            capital
            * self.config.capital_utilization
            * self.config.max_position_size_pct
        )

        # 価格に基づく数量
        base_size = max_size_value / price

        return base_size

    def _calculate_risk_based_size(
        self, capital: float, price: float, volatility: float
    ) -> float:
        """リスクベースのポジションサイズを計算"""
        # 1トレードあたりの最大リスク金額
        max_risk_amount = capital * self.config.max_risk_per_trade_pct

        # ボラティリティに基づくリスク調整
        if volatility > 0:
            adjusted_risk_pct = self.config.risk_per_trade_pct * (
                self.config.volatility_target / volatility
            )
            adjusted_risk_pct = min(
                adjusted_risk_pct, self.config.max_risk_per_trade_pct
            )
        else:
            adjusted_risk_pct = self.config.risk_per_trade_pct

        risk_amount = capital * adjusted_risk_pct

        # リスク金額に基づくポジションサイズ
        # (リスク金額) / (価格 × ストップロス距離) の近似
        # ここでは簡易的にリスク金額 ÷ 価格で計算
        risk_based_size = risk_amount / price

        return risk_based_size

    def _calculate_kelly_size(
        self, capital: float, price: float, win_rate: float, signal_strength: float
    ) -> float:
        """ケリー基準のポジションサイズを計算"""
        # ケリー公式: K = (勝率 × 利幅) - 負け率
        # ここでは簡易版を使用
        if win_rate <= 0.5:
            kelly_pct = 0.0
        else:
            # 勝率に基づくケリー比率
            kelly_ratio = (win_rate - 0.5) / 0.5
            kelly_pct = kelly_ratio * self.config.kelly_fraction

        kelly_amount = capital * kelly_pct
        kelly_size = kelly_amount / price

        return kelly_size

    def _adjust_for_volatility(self, base_size: float, volatility: float) -> float:
        """ボラティリティによる調整"""
        if volatility <= 0:
            return base_size

        # 目標ボラティリティに対する調整
        vol_ratio = self.config.volatility_target / volatility
        vol_ratio = min(vol_ratio, 2.0)  # 最大2倍まで

        adjusted_size = base_size * vol_ratio

        return adjusted_size

    def _apply_size_constraints(self, symbol: str, size: float, price: float) -> float:
        """サイズ制約を適用"""
        # 金額ベースの制約
        size_value = size * price

        if size_value < self.config.min_trade_size_jpy:
            return 0.0  # 最小取引サイズ未満は取引しない

        if size_value > self.config.max_trade_size_jpy:
            size_value = self.config.max_trade_size_jpy
            size = size_value / price

        return size

    def _round_to_min_unit(self, symbol: str, size: float) -> float:
        """最小取引単位に丸める"""
        if symbol not in self.config.min_trade_units:
            return size

        min_unit = self.config.min_trade_units[symbol]

        # 小数点以下を最小単位に合わせて丸める
        rounded_size = (size // min_unit) * min_unit

        return rounded_size


class TradeExecutionEngine(BaseComponent):
    """
    V433 Phase 3: 取引実行エンジン
    現実的コストと動的サイジングを考慮した実行システム
    """

    def __init__(self, exchange: str = "zaif", config: Optional[Dict[str, Any]] = None):
        super().__init__(name="TradeExecutionEngine", config=config)
        self.exchange = exchange

        # 設定の初期化
        self.cost_config = TransactionCostConfig()
        self.sizing_config = PositionSizingConfig()

        # コンポーネントの初期化
        self.cost_calculator = TransactionCostCalculator(self.cost_config)
        self.position_sizer = PositionSizer(self.sizing_config, self.cost_calculator)

        # 状態管理
        self.positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, TradeOrder] = {}
        self.completed_orders: Deque[TradeOrder] = deque(maxlen=10000)

        # パフォーマンス追跡
        self.total_capital = DEFAULT_TOTAL_CAPITAL  # 初期資本 (仮定)
        self.available_capital = self.total_capital
        self.realized_pnl = 0.0

        # 実行制御
        self.execution_queue = queue.Queue()
        self.is_running = False
        self.execution_thread = None

    def start_execution(self):
        """実行エンジンを開始"""
        if self.is_running:
            return

        self.is_running = True
        self.execution_thread = threading.Thread(
            target=self._execution_loop, daemon=True
        )
        self.execution_thread.start()

        self.logger.info("Trade execution engine started")

    def stop_execution(self):
        """実行エンジンを停止"""
        self.is_running = False
        if self.execution_thread and self.execution_thread.is_alive():
            self.execution_thread.join(timeout=5)

        self.logger.info("Trade execution engine stopped")

    def submit_order(
        self,
        symbol: str,
        side: str,
        signal_strength: float,
        current_price: float,
        volatility: float = 0.0,
        win_rate: float = 0.5,
    ) -> Optional[str]:
        """注文を送信"""
        try:
            # 入力バリデーション
            validate_price_range(current_price, name="current_price")
            validate_range(signal_strength, 0.0, 1.0, name="signal_strength")
            validate_non_negative(volatility, name="volatility")
            validate_range(win_rate, 0.0, 1.0, name="win_rate")
            if side not in ["buy", "sell"]:
                raise ValueError(f"Invalid side: {side}, must be 'buy' or 'sell'")

            # ポジションサイズを計算
            position_size = self.position_sizer.calculate_position_size(
                symbol,
                signal_strength,
                current_price,
                self.available_capital,
                volatility,
                win_rate,
            )

            if position_size <= 0:
                self.logger.info(
                    f"Order rejected: position size too small for {symbol}"
                )
                return None

            # 注文の作成
            order_id = f"{symbol}_{side}_{int(time.time()*1000)}"
            order = TradeOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type="market",
                quantity=position_size,
            )

            # キューに追加
            self.execution_queue.put(order)
            self.pending_orders[order_id] = order

            self.logger.info(
                f"Order submitted: {order_id} ({side} {position_size:.6f} {symbol})"
            )

            return order_id

        except Exception as e:
            self.logger.error(f"Order submission failed: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """注文をキャンセル"""
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order.status = "cancelled"
            del self.pending_orders[order_id]
            self.logger.info(f"Order cancelled: {order_id}")
            return True

        return False

    def get_position(self, symbol: str) -> Optional[Position]:
        """ポジションを取得"""
        return self.positions.get(symbol)

    def get_portfolio_status(self) -> Dict[str, Any]:
        """ポートフォリオ状態を取得"""
        total_value = self.available_capital
        unrealized_pnl = 0.0

        for position in self.positions.values():
            total_value += position.market_value
            unrealized_pnl += position.unrealized_pnl

        return {
            "total_capital": self.total_capital,
            "available_capital": self.available_capital,
            "total_value": total_value,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_pnl": unrealized_pnl + self.realized_pnl,
            "positions": {
                symbol: pos.__dict__ for symbol, pos in self.positions.items()
            },
            "pending_orders": len(self.pending_orders),
        }

    def _execution_loop(self):
        """実行ループ"""
        while self.is_running:
            try:
                # キューから注文を取得
                if not self.execution_queue.empty():
                    order = self.execution_queue.get_nowait()
                    self._execute_order(order)

                # ポジションの更新
                self._update_positions()

                time.sleep(0.1)  # 100ms間隔

            except Exception as e:
                self.logger.error(f"Execution loop error: {e}")
                time.sleep(1)

    def _execute_order(self, order: TradeOrder):
        """注文を実行"""
        try:
            # 現在の市場価格を取得（シミュレーション）
            current_price = self._get_current_price(order.symbol)

            # コスト計算
            fee = self.cost_calculator.calculate_fee(
                self.exchange, order.side, order.quantity, current_price
            )

            spread = self.cost_calculator.calculate_spread(order.symbol, current_price)
            slippage = self.cost_calculator.calculate_slippage(
                order.symbol, order.side, order.quantity, current_price
            )

            # 実行価格の計算
            if order.side == "buy":
                executed_price = current_price + spread + slippage
            else:
                executed_price = current_price - spread + slippage

            # 資本チェック
            order_value = order.quantity * executed_price
            total_cost = order_value + fee

            if total_cost > self.available_capital and order.side == "buy":
                self.logger.warning(f"Insufficient capital for order {order.order_id}")
                order.status = "rejected"
                return

            # 注文実行
            order.executed_quantity = order.quantity
            order.executed_price = executed_price
            order.fee = fee
            order.slippage = slippage
            order.status = "executed"

            # ポジション更新
            self._update_position_from_order(order)

            # 資本更新
            if order.side == "buy":
                self.available_capital -= total_cost
            else:
                self.available_capital += order_value - fee

            # 注文完了リストに追加
            self.completed_orders.append(order)
            if order.order_id in self.pending_orders:
                del self.pending_orders[order.order_id]

            self.logger.info(
                f"Order executed: {order.order_id} "
                f"({order.side} {order.executed_quantity:.6f} @ {executed_price:.2f}) "
                f"fee: {fee:.2f}"
            )

        except Exception as e:
            self.logger.error(f"Order execution failed: {e}")
            order.status = "rejected"

    def _update_position_from_order(self, order: TradeOrder):
        """注文からポジションを更新"""
        symbol = order.symbol

        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0.0,
                average_price=0.0,
                current_price=order.executed_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
            )

        position = self.positions[symbol]

        if order.side == "buy":
            # 買い注文: ポジション増加
            total_quantity = position.quantity + order.executed_quantity
            total_cost = (position.quantity * position.average_price) + (
                order.executed_quantity * order.executed_price
            )
            new_avg_price = total_cost / total_quantity if total_quantity > 0 else 0.0

            position.quantity = total_quantity
            position.average_price = new_avg_price

        else:
            # 売り注文: ポジション減少または利益確定
            sell_quantity = min(order.executed_quantity, position.quantity)
            sell_value = sell_quantity * order.executed_price
            cost_basis = sell_quantity * position.average_price

            # 実現損益計算
            realized_pnl = sell_value - cost_basis - order.fee
            self.realized_pnl += realized_pnl

            # ポジション更新
            position.quantity -= sell_quantity
            position.realized_pnl += realized_pnl

            # ポジションが0になった場合
            if position.quantity <= 0:
                position.quantity = 0.0
                position.average_price = 0.0

        position.timestamp = datetime.now()

    def _update_positions(self):
        """ポジションを更新"""
        try:
            for symbol, position in list(self.positions.items()):
                # 現在の価格を取得
                current_price = self._get_current_price(symbol)

                if current_price > 0:
                    position.current_price = current_price

                    # 未実現損益の計算
                    if position.quantity > 0:
                        position.unrealized_pnl = (
                            current_price - position.average_price
                        ) * position.quantity
                    else:
                        position.unrealized_pnl = 0.0

                    position.timestamp = datetime.now()

                # ポジションが0になった場合削除
                if position.quantity <= 0.00001:  # 最小単位以下
                    del self.positions[symbol]

        except Exception as e:
            self.logger.error(f"Position update failed: {e}")

    def _get_current_price(self, symbol: str) -> float:
        """現在の価格を取得（シミュレーション）"""
        # 実際の実装では取引所APIから取得
        # ここでは簡易的なシミュレーション
        base_prices = {
            "btc_jpy": 5000000.0,
            "eth_jpy": 300000.0,
            "xrp_jpy": 100.0,
            "mona_jpy": 200.0,
        }

        base_price = base_prices.get(symbol, 1000.0)

        # ランダムな変動を加える（±1%）
        import random

        variation = random.uniform(-0.01, 0.01)
        current_price = base_price * (1 + variation)

        return current_price


def create_trade_execution_engine(exchange: str = "zaif") -> TradeExecutionEngine:
    """TradeExecutionEngineのファクトリ関数"""
    return TradeExecutionEngine(exchange)


# 使用例
if __name__ == "__main__":
    # 取引実行エンジンの作成
    engine = create_trade_execution_engine("zaif")

    # エンジン開始
    engine.start_execution()

    try:
        # サンプル注文
        order_id = engine.submit_order(
            symbol="btc_jpy",
            side="buy",
            signal_strength=0.8,
            current_price=5000000.0,
            volatility=0.02,
            win_rate=0.55,
        )

        if order_id:
            print(f"Order submitted: {order_id}")

        # 少し待機
        time.sleep(2)

        # ポートフォリオ状態確認
        status = engine.get_portfolio_status()
        print(f"Portfolio status: {status}")

    finally:
        # エンジン停止
        engine.stop_execution()
