# ztb/live_trading/live_trader.py

"""
ライブトレーダー実装

このモジュールは、リアルタイムトレーディングを実行し、
モデル予測、ポジション管理、リスク制御を統合します。
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import time
import threading

from .trading_api import TradingAPI, OrderInfo, BalanceInfo, TickerInfo

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """トレーディングシグナル"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    amount: float
    price: Optional[float]
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class Position:
    """ポジション情報"""
    symbol: str
    amount: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: datetime


class LiveTrader:
    """
    ライブトレーダー

    リアルタイムトレーディングを実行し、以下の機能を統合：
    - モデル予測とシグナル生成
    - ポジション管理とリスク制御
    - 取引実行と監視
    - 異常検知と自動停止
    """

    def __init__(self,
                 trading_api: TradingAPI,
                 signal_generator: Any,  # シグナル生成オブジェクト
                 risk_manager: Optional[Any] = None,
                 max_positions: int = 5,
                 max_position_size: float = 0.01):
        """
        初期化

        Args:
            trading_api: 取引APIインスタンス
            signal_generator: シグナル生成関数
            risk_manager: リスクマネージャー（オプション）
            max_positions: 最大ポジション数
            max_position_size: 最大ポジションサイズ（BTC単位）
        """
        self.trading_api = trading_api
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.max_positions = max_positions
        self.max_position_size = max_position_size

        self.positions: List[Position] = []
        self.active_orders: Dict[str, OrderInfo] = {}
        self.is_running = False
        self.trading_thread: Optional[threading.Thread] = None

        # パフォーマンス追跡
        self.total_trades = 0
        self.successful_trades = 0
        self.total_pnl = 0.0

        logger.info("LiveTrader initialized")

    def start_trading(self, interval_seconds: int = 60):
        """
        トレーディング開始

        Args:
            interval_seconds: トレーディング間隔（秒）
        """
        if self.is_running:
            logger.warning("Trading is already running")
            return

        self.is_running = True
        self.trading_thread = threading.Thread(
            target=self._trading_loop,
            args=(interval_seconds,)
        )
        self.trading_thread.daemon = True
        self.trading_thread.start()

        logger.info(f"Live trading started with {interval_seconds}s interval")

    def stop_trading(self):
        """トレーディング停止"""
        if not self.is_running:
            logger.info("Trading is not running")
            return

        self.is_running = False
        if self.trading_thread:
            self.trading_thread.join(timeout=10)

        # 未決済注文の取消
        self._cancel_all_orders()

        logger.info("Live trading stopped")

    def _trading_loop(self, interval_seconds: int):
        """トレーディングメインループ"""
        logger.info("Trading loop started")

        while self.is_running:
            try:
                self._execute_trading_cycle()
                time.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(interval_seconds)  # エラー時も待機

        logger.info("Trading loop ended")

    def _execute_trading_cycle(self):
        """1サイクルのトレーディング実行"""
        try:
            # 1. シグナル生成
            signals = self.signal_generator()

            # 2. ポジション更新
            self._update_positions()

            # 3. シグナル処理
            for signal in signals:
                self._process_signal(signal)

            # 4. リスクチェック
            self._check_risk_limits()

            # 5. パフォーマンスログ
            self._log_performance()

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            raise

    def _process_signal(self, signal: TradingSignal):
        """
        シグナル処理

        Args:
            signal: トレーディングシグナル
        """
        try:
            # リスクチェック
            if not self._validate_signal(signal):
                return

            # アクション実行
            if signal.action == 'buy':
                self._execute_buy(signal)
            elif signal.action == 'sell':
                self._execute_sell(signal)
            elif signal.action == 'hold':
                self._execute_hold(signal)

        except Exception as e:
            logger.error(f"Error processing signal {signal}: {e}")

    def _validate_signal(self, signal: TradingSignal) -> bool:
        """
        シグナルバリデーション

        Args:
            signal: トレーディングシグナル

        Returns:
            bool: 有効性
        """
        # 基本バリデーション
        if signal.confidence < 0.5:
            logger.debug(f"Signal confidence too low: {signal.confidence}")
            return False

        if signal.amount <= 0:
            logger.warning(f"Invalid signal amount: {signal.amount}")
            return False

        # ポジション制限チェック
        if len(self.positions) >= self.max_positions and signal.action == 'buy':
            logger.debug("Maximum positions reached")
            return False

        # サイズ制限チェック
        if signal.amount > self.max_position_size:
            logger.debug(f"Signal amount exceeds max position size: {signal.amount}")
            return False

        # リスクマネージャーチェック
        if self.risk_manager:
            if not self.risk_manager.validate_signal(signal):
                logger.debug("Signal rejected by risk manager")
                return False

        return True

    def _execute_buy(self, signal: TradingSignal):
        """買い注文実行"""
        try:
            # 現在の価格取得
            ticker = self.trading_api.get_ticker(signal.symbol)

            # 注文作成
            order = self.trading_api.create_order(
                symbol=signal.symbol,
                side='buy',
                amount=signal.amount,
                price=ticker.ask if signal.price is None else signal.price,
                order_type='limit'
            )

            # 注文追跡
            self.active_orders[order.order_id] = order

            logger.info(f"Buy order placed: {order}")

        except Exception as e:
            logger.error(f"Failed to execute buy order: {e}")

    def _execute_sell(self, signal: TradingSignal):
        """売り注文実行"""
        try:
            # ポジション確認
            if signal.symbol not in self.positions:
                logger.debug(f"No position to sell for {signal.symbol}")
                return

            position = self.positions[signal.symbol]

            # 現在の価格取得
            ticker = self.trading_api.get_ticker(signal.symbol)

            # 注文作成
            order = self.trading_api.create_order(
                symbol=signal.symbol,
                side='sell',
                amount=min(signal.amount, position.amount),
                price=ticker.bid if signal.price is None else signal.price,
                order_type='limit'
            )

            # 注文追跡
            self.active_orders[order.order_id] = order

            logger.info(f"Sell order placed: {order}")

        except Exception as e:
            logger.error(f"Failed to execute sell order: {e}")

    def _execute_hold(self, signal: TradingSignal):
        """ホールド（何もしない）"""
        logger.debug(f"Hold signal for {signal.symbol}")

    def _update_positions(self):
        """ポジション更新"""
        try:
            # 残高取得
            balance = self.trading_api.get_balance()

            # アクティブ注文のステータス確認
            orders_to_remove = []
            for order_id, order in self.active_orders.items():
                try:
                    updated_order = self.trading_api.get_order_status(order_id)
                    if updated_order.status in ['filled', 'canceled', 'rejected']:
                        orders_to_remove.append(order_id)

                        if updated_order.status == 'filled':
                            self._update_position_from_order(updated_order)

                except Exception as e:
                    logger.error(f"Failed to update order {order_id}: {e}")

            # 完了した注文を削除
            for order_id in orders_to_remove:
                del self.active_orders[order_id]

        except Exception as e:
            logger.error(f"Failed to update positions: {e}")

    def _update_position_from_order(self, order: OrderInfo):
        """注文からのポジション更新"""
        if order.side == 'buy':
            # 買い注文成立
            existing_position = None
            for pos in self.positions:
                if pos.symbol == order.symbol:
                    existing_position = pos
                    break

            if existing_position is None:
                position = Position(
                    symbol=order.symbol,
                    amount=order.amount,
                    entry_price=order.price,
                    current_price=order.price,
                    unrealized_pnl=0.0,
                    timestamp=order.timestamp
                )
                self.positions.append(position)
            else:
                # 既存ポジション更新
                total_amount = existing_position.amount + order.amount
                total_cost = (existing_position.amount * existing_position.entry_price) + (order.amount * order.price)
                new_entry_price = total_cost / total_amount

                existing_position.amount = total_amount
                existing_position.entry_price = new_entry_price
                existing_position.timestamp = order.timestamp

        elif order.side == 'sell':
            # 売り注文成立
            if order.symbol in self.positions:
                position = self.positions[order.symbol]
                position.amount -= order.amount

                # ポジションクローズ
                if position.amount <= 0:
                    # PnL計算
                    realized_pnl = (order.price - position.entry_price) * abs(position.amount)
                    self.total_pnl += realized_pnl
                    self.total_trades += 1

                    if realized_pnl > 0:
                        self.successful_trades += 1

                    del self.positions[order.symbol]
                    logger.info(f"Position closed for {order.symbol}, PnL: {realized_pnl}")

    def _check_risk_limits(self):
        """リスク制限チェック"""
        # ポジション数のチェック
        if len(self.positions) > self.max_positions:
            logger.warning(f"Too many positions: {len(self.positions)} > {self.max_positions}")

        # リスクマネージャーチェック
        if self.risk_manager:
            alerts = self.risk_manager.check_risk_limits(self.positions)
            for alert in alerts:
                logger.warning(f"Risk alert: {alert}")

    def _cancel_all_orders(self):
        """全注文取消"""
        for order_id in list(self.active_orders.keys()):
            try:
                self.trading_api.cancel_order(order_id)
                del self.active_orders[order_id]
                logger.info(f"Order canceled: {order_id}")
            except Exception as e:
                logger.error(f"Failed to cancel order {order_id}: {e}")

    def _log_performance(self):
        """パフォーマンスログ"""
        total_positions = len(self.positions)
        win_rate = self.successful_trades / max(self.total_trades, 1)

        logger.info(f"Performance: trades={self.total_trades}, "
                   f"win_rate={win_rate:.2%}, "
                   f"total_pnl={self.total_pnl:.2f}, "
                   f"positions={total_positions}")

    def get_status(self) -> Dict[str, Any]:
        """
        ステータス取得

        Returns:
            Dict[str, Any]: 現在のステータス
        """
        return {
            'is_running': self.is_running,
            'positions': self.positions,
            'active_orders': list(self.active_orders.values()),
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'total_pnl': self.total_pnl,
            'win_rate': self.successful_trades / max(self.total_trades, 1)
        }

    def _add_position(self, symbol: str, amount: float, price: float, side: str):
        """
        ポジション追加（テスト用）

        Args:
            symbol: 通貨ペア
            amount: 数量
            price: 価格
            side: 売買方向
        """
        position = Position(
            symbol=symbol,
            amount=amount,
            entry_price=price,
            current_price=price,
            unrealized_pnl=0.0,
            timestamp=datetime.now()
        )
        self.positions.append(position)

    def _remove_position(self, symbol: str):
        """
        ポジション削除（テスト用）

        Args:
            symbol: 通貨ペア
        """
        self.positions = [p for p in self.positions if p.symbol != symbol]

    def _trading_loop_iteration(self):
        """
        トレーディングループ1回実行（テスト用）
        """
        try:
            # シグナル生成
            signals = self.signal_generator.generate_signal()

            # 各シグナル処理
            for signal in signals:
                if signal.action in ['buy', 'sell']:
                    self._process_signal(signal)

        except Exception as e:
            logger.error(f"Error in trading loop iteration: {e}")

    def _check_risk_management(self, trade_signal: Dict[str, Any]) -> bool:
        """
        リスク管理チェック（テスト用）

        Args:
            trade_signal: 取引シグナル

        Returns:
            bool: リスクチェック通過
        """
        if self.risk_manager:
            return self.risk_manager.validate_trade(trade_signal)
        return True