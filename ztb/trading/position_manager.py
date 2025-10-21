#!/usr/bin/env python3
"""
V433 Phase 3: ポジション管理システム
100%資本利用と最小取引単位を考慮した高度なポジション管理
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import threading
import numpy as np

from ztb.utils.logging_utils import get_logger
from ztb.trading.trade_execution_engine import TradeExecutionEngine, Position

logger = get_logger(__name__)

@dataclass
class PositionManagementConfig:
    """ポジション管理設定"""
    # 資本利用設定
    full_capital_utilization: bool = True  # 100%資本利用
    capital_buffer_pct: float = 0.05  # 資本バッファ (5%)

    # ポジション制限
    max_positions_per_symbol: int = 1  # シンボルあたり最大ポジション数
    max_total_positions: int = 5  # 総最大ポジション数
    max_correlation_positions: int = 2  # 相関ポジションの最大数

    # リスク管理
    max_portfolio_risk_pct: float = 0.10  # ポートフォリオ最大リスク (10%)
    max_single_position_risk_pct: float = 0.05  # 単一ポジション最大リスク (5%)
    stop_loss_pct: float = 0.02  # ストップロス (2%)
    take_profit_pct: float = 0.05  # テイクプロフィット (5%)

    # 再バランス設定
    enable_rebalancing: bool = True
    rebalance_interval_hours: int = 24
    rebalance_threshold_pct: float = 0.10  # 再バランス閾値 (10%)

    # 最小取引単位管理
    strict_min_unit_enforcement: bool = True
    round_to_min_unit: bool = True
    min_unit_buffer_pct: float = 0.001  # 最小単位バッファ (0.1%)

    # ダイナミックサイジング
    enable_dynamic_sizing: bool = True
    sizing_update_interval: int = 300  # 5分ごと
    market_condition_adjustment: bool = True


@dataclass
class PortfolioState:
    """ポートフォリオ状態"""
    total_capital: float
    available_capital: float
    used_capital: float
    total_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_risk: float
    positions: Dict[str, Position] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_pnl(self) -> float:
        """総損益"""
        return self.unrealized_pnl + self.realized_pnl

    @property
    def portfolio_return(self) -> float:
        """ポートフォリオリターン"""
        if self.total_capital > 0:
            return self.total_pnl / self.total_capital
        return 0.0

    @property
    def risk_adjusted_return(self) -> float:
        """リスク調整リターン"""
        if self.total_risk > 0:
            return self.portfolio_return / self.total_risk
        return 0.0


@dataclass
class PositionSignal:
    """ポジションシグナル"""
    symbol: str
    action: str  # "open_long", "open_short", "close_long", "close_short", "adjust"
    strength: float  # シグナル強度 (0-1)
    target_quantity: float
    confidence: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


class MinimumUnitManager:
    """最小取引単位管理"""

    def __init__(self):
        # 取引所別最小取引単位
        self.min_units = {
            "zaif": {
                "btc_jpy": Decimal('0.0001'),
                "eth_jpy": Decimal('0.001'),
                "xrp_jpy": Decimal('1'),
                "mona_jpy": Decimal('1'),
                "bcc_jpy": Decimal('0.0001'),
            },
            "bitflyer": {
                "btc_jpy": Decimal('0.001'),
                "eth_jpy": Decimal('0.01'),
                "xrp_jpy": Decimal('1'),
            },
            "coincheck": {
                "btc_jpy": Decimal('0.001'),
                "eth_jpy": Decimal('0.01'),
                "xrp_jpy": Decimal('1'),
            }
        }

    def get_min_unit(self, exchange: str, symbol: str) -> Decimal:
        """最小取引単位を取得"""
        return self.min_units.get(exchange, {}).get(symbol, Decimal('0.0001'))

    def validate_quantity(self, exchange: str, symbol: str, quantity: float) -> Tuple[bool, float]:
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

    def round_to_min_unit(self, exchange: str, symbol: str, quantity: float,
                         round_mode: str = "down") -> float:
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

    def calculate_min_trade_value(self, exchange: str, symbol: str, price: float) -> float:
        """最小取引金額を計算"""
        min_unit = self.get_min_unit(exchange, symbol)
        return float(min_unit * Decimal(str(price)))


class RiskManager:
    """リスクマネージャー"""

    def __init__(self, config: PositionManagementConfig):
        self.config = config
        self.logger = get_logger(__name__)

        # リスク追跡
        self.portfolio_risk = 0.0
        self.position_risks: Dict[str, float] = {}

    def calculate_portfolio_risk(self, positions: Dict[str, Position],
                               volatilities: Dict[str, float]) -> float:
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

    def check_risk_limits(self, portfolio_state: PortfolioState) -> Dict[str, bool]:
        """リスク制限をチェック"""
        checks = {
            "portfolio_risk_ok": True,
            "single_position_risk_ok": True,
            "capital_buffer_ok": True
        }

        # ポートフォリオリスクチェック
        if portfolio_state.total_risk > portfolio_state.total_capital * self.config.max_portfolio_risk_pct:
            checks["portfolio_risk_ok"] = False
            self.logger.warning(f"Portfolio risk limit exceeded: "
                              f"{portfolio_state.total_risk:.2f} > "
                              f"{portfolio_state.total_capital * self.config.max_portfolio_risk_pct:.2f}")

        # 単一ポジションリスクチェック
        for symbol, risk in self.position_risks.items():
            if risk > portfolio_state.total_capital * self.config.max_single_position_risk_pct:
                checks["single_position_risk_ok"] = False
                self.logger.warning(f"Single position risk limit exceeded for {symbol}: "
                                  f"{risk:.2f} > "
                                  f"{portfolio_state.total_capital * self.config.max_single_position_risk_pct:.2f}")

        # 資本バッファチェック
        capital_buffer = portfolio_state.total_capital * self.config.capital_buffer_pct
        if portfolio_state.available_capital < capital_buffer:
            checks["capital_buffer_ok"] = False
            self.logger.warning(f"Capital buffer insufficient: "
                              f"{portfolio_state.available_capital:.2f} < {capital_buffer:.2f}")

        return checks

    def calculate_stop_loss_price(self, position: Position, current_price: float) -> float:
        """ストップロス価格を計算"""
        if position.quantity > 0:  # ロングポジション
            stop_price = current_price * (1 - self.config.stop_loss_pct)
        else:  # ショートポジション
            stop_price = current_price * (1 + self.config.stop_loss_pct)

        return stop_price

    def calculate_take_profit_price(self, position: Position, current_price: float) -> float:
        """テイクプロフィット価格を計算"""
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


class PositionManager:
    """
    V433 Phase 3: ポジション管理システム
    100%資本利用と最小取引単位を考慮した高度なポジション管理
    """

    def __init__(self, execution_engine: TradeExecutionEngine, exchange: str = "zaif"):
        self.execution_engine = execution_engine
        self.exchange = exchange
        self.logger = get_logger(__name__)

        # 設定の初期化
        self.config = PositionManagementConfig()
        self.min_unit_manager = MinimumUnitManager()
        self.risk_manager = RiskManager(self.config)

        # 状態管理
        self.portfolio_state = PortfolioState(
            total_capital=100000.0,  # 初期値
            available_capital=100000.0,
            used_capital=0.0,
            total_value=100000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            total_risk=0.0
        )

        # シグナルキュー
        self.signal_queue: asyncio.Queue[PositionSignal] = asyncio.Queue()

        # モニタリング
        self.monitoring_thread = None
        self.rebalancing_thread = None
        self.is_running = False

        # 市場データ
        self.volatilities: Dict[str, float] = {}
        self.correlations: Dict[Tuple[str, str], float] = {}

    def start_management(self):
        """ポジション管理を開始"""
        if self.is_running:
            return

        self.is_running = True

        # モニタリングスレッド開始
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        # 再バランススレッド開始
        if self.config.enable_rebalancing:
            self.rebalancing_thread = threading.Thread(target=self._rebalancing_loop, daemon=True)
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
        self.logger.info(f"Signal submitted: {signal.symbol} {signal.action} "
                        f"(strength: {signal.strength:.2f})")

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
                self.logger.warning(f"Signal rejected: invalid quantity for {signal.symbol}")
                return

            # リスクチェック
            if not self._check_signal_risk(signal):
                self.logger.warning(f"Signal rejected: risk limits exceeded for {signal.symbol}")
                return

            # ポジション数のチェック
            if not self._check_position_limits(signal):
                self.logger.warning(f"Signal rejected: position limits exceeded")
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
            self.logger.warning(f"Quantity {signal.target_quantity} too small for {signal.symbol}, "
                              f"minimum required: {adjusted_quantity}")
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
        if current_positions >= self.config.max_total_positions:
            return False

        # シンボル別ポジション数チェック
        symbol_positions = sum(1 for pos in self.portfolio_state.positions.values()
                             if pos.symbol == signal.symbol)
        if symbol_positions >= self.config.max_positions_per_symbol:
            return False

        # 相関ポジション数チェック（簡易版）
        # 実際には相関係数行列を使って計算
        correlated_positions = current_positions  # 簡易的に全て相関ありと仮定
        if correlated_positions >= self.config.max_correlation_positions:
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
            win_rate=0.5  # 仮定値
        )

        if order_id:
            self.logger.info(f"Position opened: {signal.symbol} {signal.action} "
                           f"quantity: {signal.target_quantity:.6f}")
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
        quantity = self.min_unit_manager.round_to_min_unit(self.exchange, signal.symbol, quantity)

        if quantity > 0:
            # 市場注文でクローズ
            order_id = self.execution_engine.submit_order(
                symbol=signal.symbol,
                side=side,
                signal_strength=1.0,  # クローズ時は最大強度
                current_price=self._get_current_price(signal.symbol),
                volatility=self.volatilities.get(signal.symbol, 0.02),
                win_rate=0.5
            )

            if order_id:
                self.logger.info(f"Position closed: {signal.symbol} "
                               f"quantity: {quantity:.6f}")
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
                time.sleep(self.config.rebalance_interval_hours * 3600)

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
            self.portfolio_state.used_capital = self.portfolio_state.total_value - self.portfolio_state.available_capital

            # ポジション情報を更新
            self.portfolio_state.positions = {}
            for symbol, pos_data in engine_status["positions"].items():
                position = Position(**pos_data)
                self.portfolio_state.positions[symbol] = position

            # リスク計算
            self.portfolio_state.total_risk = self.risk_manager.calculate_portfolio_risk(
                self.portfolio_state.positions, self.volatilities
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

    def _check_stop_conditions(self):
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
                    reason="stop_loss"
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
                    reason="take_profit"
                )
                asyncio.create_task(self.submit_signal(signal))

    def _perform_rebalancing(self):
        """再バランスを実行"""
        self.logger.info("Performing portfolio rebalancing")

        # 簡易的な再バランスロジック
        # 実際の実装ではより詳細なアルゴリズムが必要

        total_value = self.portfolio_state.total_value
        target_allocation = 1.0 / len(self.portfolio_state.positions) if self.portfolio_state.positions else 0

        for symbol, position in self.portfolio_state.positions.items():
            current_allocation = position.market_value / total_value
            deviation = abs(current_allocation - target_allocation)

            if deviation > self.config.rebalance_threshold_pct:
                self.logger.info(f"Rebalancing {symbol}: {current_allocation:.2%} -> {target_allocation:.2%}")

                # 再バランスシグナルの生成
                target_value = total_value * target_allocation
                target_quantity = target_value / position.current_price

                signal = PositionSignal(
                    symbol=symbol,
                    action="adjust",
                    strength=0.8,
                    target_quantity=target_quantity,
                    confidence=0.7,
                    reason="rebalancing"
                )

                asyncio.create_task(self.submit_signal(signal))

    def _get_current_price(self, symbol: str) -> float:
        """現在の価格を取得"""
        # 実行エンジンから価格を取得
        # 実際の実装ではより効率的な方法が必要
        return 1000.0  # 仮定値

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """ポートフォリオサマリーを取得"""
        return {
            "portfolio_state": self.portfolio_state.__dict__,
            "risk_checks": self.risk_manager.check_risk_limits(self.portfolio_state),
            "position_count": len(self.portfolio_state.positions),
            "total_exposure": self.portfolio_state.used_capital,
            "risk_metrics": {
                "portfolio_risk": self.portfolio_state.total_risk,
                "risk_adjusted_return": self.portfolio_state.risk_adjusted_return
            }
        }


def create_position_manager(execution_engine: TradeExecutionEngine,
                          exchange: str = "zaif") -> PositionManager:
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
            reason="test_signal"
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