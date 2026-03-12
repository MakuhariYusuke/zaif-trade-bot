#!/usr/bin/env python3
"""
V433 Phase 3: ポジション管理システム
100%資本利用と最小取引単位を考慮した高度なポジション管理
"""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Any

from ztb.trading.environment.constants import DEFAULT_FEE_RATE, DEFAULT_TOTAL_CAPITAL
from ztb.trading.risk.risk_manager import RiskManager
from ztb.trading.trade_execution_engine import Position, TradeExecutionEngine
from ztb.trading.types import PositionManagementConfig, PortfolioState, PositionSignal
from ztb.types.common import BaseComponent, ConfigDict
from ztb.utils.errors import validate_price
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class MinimumUnitManager:
    """最小取引単位管理"""

    def __init__(self):
        # 取引所別最小取引単位
        self.min_units = {
            "zaif": {
                "btc_jpy": Decimal("0.0001"),
                "eth_jpy": Decimal("0.001"),
                "xrp_jpy": Decimal("1"),
                "mona_jpy": Decimal("1"),
                "bcc_jpy": Decimal("0.0001"),
            },
            "bitflyer": {
                "btc_jpy": Decimal("0.001"),
                "eth_jpy": Decimal("0.01"),
                "xrp_jpy": Decimal("1"),
            },
            "coincheck": {
                "btc_jpy": Decimal("0.001"),
                "eth_jpy": Decimal("0.01"),
                "xrp_jpy": Decimal("1"),
            },
        }

    def get_min_unit(self, exchange: str, symbol: str) -> Decimal:
        """最小取引単位を取得"""
        return self.min_units.get(exchange, {}).get(symbol, Decimal("0.0001"))

    def validate_quantity(
        self, exchange: str, symbol: str, quantity: float
    ) -> tuple[bool, float]:
        """数量を最小単位で検証・調整"""
        min_unit = self.get_min_unit(exchange, symbol)
        quantity_dec = Decimal(str(quantity))

        # 最小単位で割って整数部分を取得
        units = (quantity_dec / min_unit).to_integral_value(ROUND_DOWN)

        # 最小単位に合わせた数量を計算
        adjusted_quantity = float(units * min_unit)

        # バッファを考慮した最小数量チェック
        min_required = float(min_unit)

        is_valid = adjusted_quantity >= min_required

        return is_valid, adjusted_quantity

    def round_to_min_unit(
        self, exchange: str, symbol: str, quantity: float, round_mode: str = "down"
    ) -> float:
        """最小単位に丸める"""
        min_unit = self.get_min_unit(exchange, symbol)
        quantity_dec = Decimal(str(quantity))

        if round_mode == "down":
            rounding = ROUND_DOWN
        elif round_mode == "up":
            rounding = ROUND_UP
        else:
            rounding = ROUND_DOWN

        # 最小単位で丸める
        units = (quantity_dec / min_unit).to_integral_value(rounding)
        rounded_quantity = float(units * min_unit)

        return rounded_quantity

    def calculate_min_trade_value(
        self, exchange: str, symbol: str, price: float
    ) -> float:
        """最小取引金額を計算"""
        min_unit = self.get_min_unit(exchange, symbol)
        return float(min_unit * Decimal(str(price)))

from ztb.trading.risk.risk_manager import RiskManager

class PositionManager(BaseComponent):
    """
    V433 Phase 3: ポジション管理システム
    100%資本利用と最小取引単位を考慮した高度なポジション管理
    """

    def __init__(
        self,
        execution_engine: TradeExecutionEngine,
        exchange: str = "coincheck",
        config: ConfigDict | None = None,
    ) -> None:
        super().__init__(name="PositionManager", config=config)
        self.execution_engine = execution_engine
        self.exchange = exchange

        # 設定の初期化
        self.config_obj = PositionManagementConfig()
        self.min_unit_manager = MinimumUnitManager()
        self.risk_manager = RiskManager(self.config_obj)

        # 状態管理
        self.portfolio_state = PortfolioState(
            total_capital=DEFAULT_TOTAL_CAPITAL,  # 初期値
            available_capital=DEFAULT_TOTAL_CAPITAL,
            used_capital=0.0,
            total_value=DEFAULT_TOTAL_CAPITAL,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            total_risk=0.0,
        )

        # シグナルキュー
        self.signal_queue: asyncio.Queue[PositionSignal] = asyncio.Queue()

        # モニタリング
        self.monitoring_thread = None
        self.rebalancing_thread = None
        self.is_running = False

        # 市場データ
        self.volatilities: dict[str, float] = {}
        self.correlations: dict[tuple[str, str], float] = {}

    def start_management(self):
        """ポジション管理を開始"""
        if self.is_running:
            return

        self.is_running = True

        # モニタリングスレッド開始
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

        # 再バランススレッド開始
        if self.config_obj.enable_rebalancing:
            self.rebalancing_thread = threading.Thread(
                target=self._rebalancing_loop, daemon=True
            )
            self.rebalancing_thread.start()

        self.logger.info("Position management started")

    def stop_management(self):
        """ポジション管理を停止"""
        self.is_running = False

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)

        if self.rebalancing_thread and self.rebalancing_thread.is_alive():
            self.rebalancing_thread.join(timeout=5)

        self.logger.info("Position management stopped")

    async def submit_signal(self, signal: PositionSignal):
        """ポジションシグナルを送信"""
        await self.signal_queue.put(signal)
        self.logger.info(
            f"Signal submitted: {signal.symbol} {signal.action} "
            f"(strength: {signal.strength:.2f})"
        )

    def process_signals_sync(self):
        """シグナルを同期的に処理（メインループ用）"""
        try:
            while not self.signal_queue.empty():
                signal = self.signal_queue.get_nowait()
                self._process_signal(signal)
        except asyncio.QueueEmpty:
            pass

    def _process_signal(self, signal: PositionSignal):
        """シグナルを処理"""
        try:
            # 最小取引単位の検証
            if not self._validate_signal_quantity(signal):
                self.logger.warning(
                    f"Signal rejected: invalid quantity for {signal.symbol}"
                )
                return

            # リスクチェック
            if not self._check_signal_risk(signal):
                self.logger.warning(
                    f"Signal rejected: risk limits exceeded for {signal.symbol}"
                )
                return

            # ポジション数のチェック
            if not self._check_position_limits(signal):
                self.logger.warning("Signal rejected: position limits exceeded")
                return

            # シグナル実行
            self._execute_signal(signal)

        except Exception as e:
            self.logger.error(f"Signal processing failed: {e}")

    def _validate_signal_quantity(self, signal: PositionSignal) -> bool:
        """シグナル数量を検証"""
        if signal.target_quantity <= 0:
            return False

        # 最小取引単位の検証
        is_valid, adjusted_quantity = self.min_unit_manager.validate_quantity(
            self.exchange, signal.symbol, signal.target_quantity
        )

        if not is_valid:
            self.logger.warning(
                f"Quantity {signal.target_quantity} too small for {signal.symbol}, "
                f"minimum required: {adjusted_quantity}"
            )
            return False

        # 数量を調整
        signal.target_quantity = adjusted_quantity
        return True

    def _check_signal_risk(self, signal: PositionSignal) -> bool:
        """シグナルリスクをチェック"""
        # 現在のポートフォリオ状態を取得
        self._update_portfolio_state()

        # リスク制限のチェック
        risk_checks = self.risk_manager.check_risk_limits(self.portfolio_state)

        # 全てのチェックが通っている場合のみ許可
        return all(risk_checks.values())

    def _check_position_limits(self, signal: PositionSignal) -> bool:
        """ポジション制限をチェック"""
        current_positions = len(self.portfolio_state.positions)

        # 総ポジション数チェック
        if current_positions >= self.config_obj.max_total_positions:
            return False

        # シンボル別ポジション数チェック
        symbol_positions = sum(
            1
            for pos in self.portfolio_state.positions.values()
            if pos.symbol == signal.symbol
        )
        if symbol_positions >= self.config_obj.max_positions_per_symbol:
            return False

        # 相関ポジション数チェック（簡易版）
        # 実際には相関係数行列を使って計算
        correlated_positions = current_positions  # 簡易的に全て相関ありと仮定
        if correlated_positions >= self.config_obj.max_correlation_positions:
            return False

        return True

    def _execute_signal(self, signal: PositionSignal):
        """シグナルを実行"""
        try:
            if signal.action in ["open_long", "open_short"]:
                self._open_position(signal)
            elif signal.action in ["close_long", "close_short"]:
                self._close_position(signal)
            elif signal.action == "adjust":
                self._adjust_position(signal)

        except Exception as e:
            self.logger.error(f"Signal execution failed: {e}")

    def _open_position(self, signal: PositionSignal):
        """ポジションを開く"""
        side = "buy" if signal.action == "open_long" else "sell"

        # 実行エンジンに注文
        order_id = self.execution_engine.submit_order(
            symbol=signal.symbol,
            side=side,
            signal_strength=signal.strength,
            current_price=self._get_current_price(signal.symbol),
            volatility=self.volatilities.get(signal.symbol, 0.02),
            win_rate=0.5,  # 仮定値
        )

        if order_id:
            self.logger.info(
                f"Position opened: {signal.symbol} {signal.action} "
                f"quantity: {signal.target_quantity:.6f}"
            )
        else:
            self.logger.warning(f"Position open failed: {signal.symbol}")

    def _close_position(self, signal: PositionSignal):
        """ポジションを閉じる"""
        position = self.portfolio_state.positions.get(signal.symbol)
        if not position or position.quantity == 0:
            return

        side = "sell" if position.quantity > 0 else "buy"
        quantity = abs(position.quantity)

        # 数量を最小単位に合わせる
        quantity = self.min_unit_manager.round_to_min_unit(
            self.exchange, signal.symbol, quantity
        )

        if quantity > 0:
            # 市場注文でクローズ
            order_id = self.execution_engine.submit_order(
                symbol=signal.symbol,
                side=side,
                signal_strength=1.0,  # クローズ時は最大強度
                current_price=self._get_current_price(signal.symbol),
                volatility=self.volatilities.get(signal.symbol, 0.02),
                win_rate=0.5,
            )

            if order_id:
                self.logger.info(
                    f"Position closed: {signal.symbol} " f"quantity: {quantity:.6f}"
                )
            else:
                self.logger.warning(f"Position close failed: {signal.symbol}")

    def _adjust_position(self, signal: PositionSignal):
        """ポジションを調整"""
        current_position = self.portfolio_state.positions.get(signal.symbol)
        if not current_position:
            # 新規ポジション
            self._open_position(signal)
            return

        current_quantity = current_position.quantity
        target_quantity = signal.target_quantity

        if current_quantity > target_quantity:
            # ポジション削減
            reduce_quantity = current_quantity - target_quantity
            signal.target_quantity = reduce_quantity
            signal.action = "close_long" if current_quantity > 0 else "close_short"
            self._close_position(signal)
        elif current_quantity < target_quantity:
            # ポジション増加
            add_quantity = target_quantity - current_quantity
            signal.target_quantity = add_quantity
            signal.action = "open_long" if target_quantity > 0 else "open_short"
            self._open_position(signal)

    def _monitoring_loop(self):
        """モニタリングループ"""
        while self.is_running:
            try:
                # ポートフォリオ状態更新
                self._update_portfolio_state()

                # リスクチェック
                self._check_risk_limits()

                # ストップロス/テイクプロフィットのチェック
                self._check_stop_conditions()

                # シグナル処理
                self.process_signals_sync()

                time.sleep(1)  # 1秒間隔

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)

    def _rebalancing_loop(self):
        """再バランスループ"""
        while self.is_running:
            try:
                time.sleep(self.config_obj.rebalance_interval_hours * 3600)

                # 再バランス実行
                self._perform_rebalancing()

            except Exception as e:
                self.logger.error(f"Rebalancing loop error: {e}")

    def _update_portfolio_state(self):
        """ポートフォリオ状態を更新"""
        try:
            # 実行エンジンから状態を取得
            engine_status = self.execution_engine.get_portfolio_status()

            # ポートフォリオ状態を更新
            self.portfolio_state.total_capital = engine_status["total_capital"]
            self.portfolio_state.available_capital = engine_status["available_capital"]
            self.portfolio_state.total_value = engine_status["total_value"]
            self.portfolio_state.unrealized_pnl = engine_status["unrealized_pnl"]
            self.portfolio_state.realized_pnl = engine_status["realized_pnl"]
            self.portfolio_state.used_capital = (
                self.portfolio_state.total_value
                - self.portfolio_state.available_capital
            )

            # ポジション情報を更新
            self.portfolio_state.positions = {}
            for symbol, pos_data in engine_status["positions"].items():
                position = Position(**pos_data)
                self.portfolio_state.positions[symbol] = position

            # リスク計算
            self.portfolio_state.total_risk = (
                self.risk_manager.calculate_portfolio_risk(
                    self.portfolio_state.positions, self.volatilities
                )
            )

            self.portfolio_state.timestamp = datetime.now()

        except Exception as e:
            self.logger.error(f"Portfolio state update failed: {e}")

    def _check_risk_limits(self):
        """リスク制限をチェック"""
        risk_checks = self.risk_manager.check_risk_limits(self.portfolio_state)

        # リスク違反時の処理
        if not all(risk_checks.values()):
            self.logger.warning("Risk limits violated, considering emergency actions")

            # 緊急時のポジション削減など
            # 実際の実装ではより詳細なロジックが必要

    def _check_stop_conditions(self) -> None:
        """ストップ条件をチェック"""
        for symbol, position in self.portfolio_state.positions.items():
            current_price = self._get_current_price(symbol)

            # ストップロスチェック
            if self.risk_manager.should_stop_loss(position, current_price):
                self.logger.warning(f"Stop loss triggered for {symbol}")
                signal = PositionSignal(
                    symbol=symbol,
                    action="close_long" if position.quantity > 0 else "close_short",
                    strength=1.0,
                    target_quantity=abs(position.quantity),
                    confidence=1.0,
                    reason="stop_loss",
                )
                asyncio.create_task(self.submit_signal(signal))

            # テイクプロフィットチェック
            elif self.risk_manager.should_take_profit(position, current_price):
                self.logger.info(f"Take profit triggered for {symbol}")
                signal = PositionSignal(
                    symbol=symbol,
                    action="close_long" if position.quantity > 0 else "close_short",
                    strength=1.0,
                    target_quantity=abs(position.quantity),
                    confidence=1.0,
                    reason="take_profit",
                )
                asyncio.create_task(self.submit_signal(signal))

    def _perform_rebalancing(self) -> None:
        """再バランスを実行"""
        self.logger.info("Performing portfolio rebalancing")

        # 簡易的な再バランスロジック
        # 実際の実装ではより詳細なアルゴリズムが必要

        total_value = self.portfolio_state.total_value
        target_allocation = (
            1.0 / len(self.portfolio_state.positions)
            if self.portfolio_state.positions
            else 0
        )

        for symbol, position in self.portfolio_state.positions.items():
            current_allocation = position.market_value / total_value
            deviation = abs(current_allocation - target_allocation)

            if deviation > self.config_obj.rebalance_threshold_pct:
                self.logger.info(
                    f"Rebalancing {symbol}: {current_allocation:.2%} -> {target_allocation:.2%}"
                )

                # 再バランスシグナルの生成
                target_value = total_value * target_allocation
                target_quantity = target_value / position.current_price

                signal = PositionSignal(
                    symbol=symbol,
                    action="adjust",
                    strength=0.8,
                    target_quantity=target_quantity,
                    confidence=0.7,
                    reason="rebalancing",
                )

                asyncio.create_task(self.submit_signal(signal))

    def _get_current_price(self, symbol: str) -> float:
        """現在の価格を取得"""
        # 実際には市場データフィードから取得
        return 1000.0  # 仮定値

    def get_portfolio_summary(self) -> dict[str, Any]:
        """ポートフォリオサマリーを取得"""
        return {
            "portfolio_state": self.portfolio_state.__dict__,
            "risk_checks": self.risk_manager.check_risk_limits(self.portfolio_state),
            "position_count": len(self.portfolio_state.positions),
            "total_exposure": self.portfolio_state.used_capital,
            "risk_metrics": {
                "portfolio_risk": self.portfolio_state.total_risk,
                "risk_adjusted_return": self.portfolio_state.risk_adjusted_return,
            },
        }

def create_position_manager(
    execution_engine: TradeExecutionEngine, exchange: str = "zaif"
) -> PositionManager:
    """PositionManagerのファクトリ関数"""
    return PositionManager(execution_engine, exchange)

# 使用例
if __name__ == "__main__":
    # 取引実行エンジンの作成
    execution_engine = TradeExecutionEngine("zaif")

    # ポジション管理システムの作成
    position_manager = create_position_manager(execution_engine, "zaif")

    # システム開始
    execution_engine.start_execution()
    position_manager.start_management()

    try:
        # サンプルシグナル
        signal = PositionSignal(
            symbol="btc_jpy",
            action="open_long",
            strength=0.8,
            target_quantity=0.001,  # 最小単位以上
            confidence=0.7,
            reason="test_signal",
        )

        # 非同期でシグナルを送信
        async def send_signal():
            await position_manager.submit_signal(signal)

        asyncio.run(send_signal())

        # 少し待機
        time.sleep(5)

        # ポートフォリオ状態確認
        summary = position_manager.get_portfolio_summary()
        print(f"Portfolio summary: {summary}")

    finally:
        # システム停止
        position_manager.stop_management()
        execution_engine.stop_execution()
