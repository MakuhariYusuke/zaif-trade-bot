"""
V433 Phase 5: Paper Trading Layer - Market Data Simulator

実市場データとの同期と取引遅延のシミュレーションを行う。
"""

import logging
import queue
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple


# Mock classes for testing
@dataclass
class Symbol:
    name: str
    base_currency: str
    quote_currency: str


@dataclass
class TickData:
    symbol: str
    bid: Decimal
    ask: Decimal
    timestamp: datetime
    volume: Optional[Decimal] = None


@dataclass
class BarData:
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    timestamp: datetime


@dataclass
class OrderBook:
    symbol: str
    bids: List[tuple[Decimal, Decimal]]  # (price, quantity)
    asks: List[tuple[Decimal, Decimal]]  # (price, quantity)
    timestamp: datetime


class MarketData:
    pass


class MarketDataProvider:
    pass


class SimulationMode(Enum):
    """シミュレーションモード"""

    REALTIME = "realtime"  # リアルタイム同期
    HISTORICAL = "historical"  # 過去データ再生
    ACCELERATED = "accelerated"  # 高速再生


@dataclass
class LatencyConfig:
    """遅延設定"""

    base_latency_ms: int = 50  # 基本遅延（ミリ秒）
    jitter_range_ms: int = 20  # ジッター範囲
    network_latency_ms: int = 10  # ネットワーク遅延
    processing_latency_ms: int = 5  # 処理遅延
    queue_latency_ms: int = 5  # キュー遅延


@dataclass
class SlippageConfig:
    """スリッページ設定"""

    base_slippage_bps: int = 5  # 基本スリッページ（bps）
    volatility_multiplier: float = 2.0  # ボラティリティ乗数
    volume_impact_bps: int = 2  # 出来高インパクト
    market_impact_threshold: Decimal = Decimal("0.01")  # 市場影響閾値


@dataclass
class SimulatedTick:
    """シミュレートティック"""

    symbol: str
    price: Decimal
    volume: Decimal
    timestamp: datetime
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    bid_volume: Optional[Decimal] = None
    ask_volume: Optional[Decimal] = None
    latency_ms: int = 0
    slippage_bps: int = 0


class MarketDataSimulator:
    """
    市場データシミュレーター

    実市場データとの同期を維持しつつ、取引遅延やスリッページを
    現実的にシミュレートする。
    """

    def __init__(
        self,
        market_data_provider: MarketDataProvider,
        latency_config: Optional[LatencyConfig] = None,
        slippage_config: Optional[SlippageConfig] = None,
        simulation_mode: SimulationMode = SimulationMode.REALTIME,
    ):
        """
        初期化

        Args:
            market_data_provider: 市場データプロバイダー
            latency_config: 遅延設定
            slippage_config: スリッページ設定
            simulation_mode: シミュレーションモード
        """
        self.market_data_provider = market_data_provider
        self.latency_config = latency_config or LatencyConfig()
        self.slippage_config = slippage_config or SlippageConfig()
        self.simulation_mode = simulation_mode

        # データ管理
        self.subscribed_symbols: set[str] = set()
        self.latest_prices: Dict[str, SimulatedTick] = {}
        self.price_history: Dict[str, List[SimulatedTick]] = {}
        self.order_books: Dict[str, OrderBook] = {}

        # シミュレーション制御
        self.is_running = False
        self.simulation_thread: Optional[threading.Thread] = None
        self.data_queue: queue.Queue = queue.Queue()

        # コールバック
        self.tick_callbacks: List[Callable[[SimulatedTick], Awaitable[None]]] = []
        self.bar_callbacks: List[Callable[[BarData], Awaitable[None]]] = []

        # パフォーマンス追跡
        self.total_latency_ms = 0
        self.latency_samples = 0
        self.total_slippage_bps = 0
        self.slippage_samples = 0

        # ロギング
        self.logger = logging.getLogger(__name__)

        self.logger.info("Market Data Simulator initialized")

    def start(self) -> None:
        """シミュレーター開始"""
        if self.is_running:
            return

        self.is_running = True
        self.simulation_thread = threading.Thread(
            target=self._simulation_loop, daemon=True
        )
        self.simulation_thread.start()

        self.logger.info("Market Data Simulator started")

    def stop(self) -> None:
        """シミュレーター停止"""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=5.0)

        self.logger.info("Market Data Simulator stopped")

    def subscribe_symbol(self, symbol: str) -> None:
        """
        シンボル購読

        Args:
            symbol: 購読するシンボル
        """
        if symbol not in self.subscribed_symbols:
            self.subscribed_symbols.add(symbol)
            self.price_history[symbol] = []
            self.logger.info(f"Subscribed to symbol: {symbol}")

    def unsubscribe_symbol(self, symbol: str) -> None:
        """
        シンボル購読解除

        Args:
            symbol: 購読解除するシンボル
        """
        if symbol in self.subscribed_symbols:
            self.subscribed_symbols.remove(symbol)
            self.logger.info(f"Unsubscribed from symbol: {symbol}")

    def get_latest_price(self, symbol: str) -> Optional[SimulatedTick]:
        """
        最新価格取得

        Args:
            symbol: シンボル

        Returns:
            Optional[SimulatedTick]: 最新ティックデータ
        """
        return self.latest_prices.get(symbol)

    def get_price_history(
        self, symbol: str, limit: Optional[int] = None
    ) -> List[SimulatedTick]:
        """
        価格履歴取得

        Args:
            symbol: シンボル
            limit: 取得件数制限

        Returns:
            List[SimulatedTick]: 価格履歴
        """
        history = self.price_history.get(symbol, [])
        if limit:
            history = history[-limit:]
        return history.copy()

    def simulate_order_execution(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        limit_price: Optional[Decimal] = None,
    ) -> Tuple[Decimal, int, int]:
        """
        注文実行シミュレーション

        Args:
            symbol: シンボル
            side: 売買方向 ('buy' or 'sell')
            quantity: 数量
            limit_price: 指値価格

        Returns:
            Tuple[Decimal, int, int]: (実行価格, 遅延ms, スリッページbps)
        """
        # 最新価格取得
        latest_tick = self.get_latest_price(symbol)
        if not latest_tick:
            raise ValueError(f"No price data available for symbol: {symbol}")

        # 遅延シミュレーション
        latency_ms = self._simulate_latency()

        # スリッページ計算
        slippage_bps = self._calculate_slippage(symbol, side, quantity, latest_tick)

        # 実行価格計算
        base_price = latest_tick.price
        if limit_price:
            # 指値注文の場合
            if side == "buy" and limit_price >= base_price:
                execution_price = base_price
            elif side == "sell" and limit_price <= base_price:
                execution_price = base_price
            else:
                # 指値未達
                return Decimal("0"), latency_ms, 0
        else:
            # 成行注文の場合
            execution_price = self._apply_slippage(base_price, slippage_bps, side)

        # パフォーマンス追跡
        self.total_latency_ms += latency_ms
        self.latency_samples += 1
        self.total_slippage_bps += slippage_bps
        self.slippage_samples += 1

        return execution_price, latency_ms, slippage_bps

    def _simulation_loop(self) -> None:
        """シミュレーションメインループ"""
        while self.is_running:
            try:
                # 実市場データ取得
                for symbol in self.subscribed_symbols:
                    self._process_symbol_data(symbol)

                # イベント処理
                self._process_data_queue()

                # 次のイテレーションまで待機
                time.sleep(0.1)  # 100ms間隔

            except Exception as e:
                self.logger.error(f"Simulation loop error: {e}")
                time.sleep(1.0)  # エラー時は1秒待機

    def _process_symbol_data(self, symbol: str) -> None:
        """
        シンボルデータ処理

        Args:
            symbol: シンボル
        """
        try:
            # 実市場データ取得
            tick_data = self.market_data_provider.get_latest_tick(symbol)
            if not tick_data:
                return

            # 遅延適用
            latency_ms = self._simulate_latency()
            delayed_timestamp = tick_data.timestamp + timedelta(milliseconds=latency_ms)

            # シミュレートティック作成
            simulated_tick = SimulatedTick(
                symbol=symbol,
                price=tick_data.price,
                volume=tick_data.volume,
                timestamp=delayed_timestamp,
                bid=tick_data.bid,
                ask=tick_data.ask,
                bid_volume=tick_data.bid_volume,
                ask_volume=tick_data.ask_volume,
                latency_ms=latency_ms,
            )

            # データ更新
            self.latest_prices[symbol] = simulated_tick
            self.price_history[symbol].append(simulated_tick)

            # 履歴サイズ制限
            if len(self.price_history[symbol]) > 10000:
                self.price_history[symbol] = self.price_history[symbol][-5000:]

            # コールバック実行
            self._notify_tick_callbacks(simulated_tick)

        except Exception as e:
            self.logger.error(f"Error processing symbol {symbol}: {e}")

    def _process_data_queue(self) -> None:
        """データキュー処理"""
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                # キューからのデータ処理（将来拡張用）
                pass
            except queue.Empty:
                break

    def _simulate_latency(self) -> int:
        """
        遅延シミュレーション

        Returns:
            int: 遅延時間（ミリ秒）
        """
        config = self.latency_config

        # 基本遅延
        latency = config.base_latency_ms

        # ジッター追加
        jitter = random.randint(-config.jitter_range_ms, config.jitter_range_ms)
        latency += jitter

        # ネットワーク遅延
        latency += config.network_latency_ms

        # 処理遅延
        latency += config.processing_latency_ms

        # キュー遅延
        latency += config.queue_latency_ms

        # 負の遅延を防ぐ
        return max(0, latency)

    def _calculate_slippage(
        self, symbol: str, side: str, quantity: Decimal, latest_tick: SimulatedTick
    ) -> int:
        """
        スリッページ計算

        Args:
            symbol: シンボル
            side: 売買方向
            quantity: 数量
            latest_tick: 最新ティック

        Returns:
            int: スリッページ（bps）
        """
        config = self.slippage_config

        # 基本スリッページ
        slippage = config.base_slippage_bps

        # ボラティリティ調整
        volatility = self._calculate_volatility(symbol)
        slippage += int(volatility * config.volatility_multiplier)

        # 出来高インパクト
        volume_impact = self._calculate_volume_impact(symbol, quantity, latest_tick)
        slippage += volume_impact

        return slippage

    def _calculate_volatility(self, symbol: str) -> float:
        """
        ボラティリティ計算

        Args:
            symbol: シンボル

        Returns:
            float: ボラティリティ（0-1の範囲）
        """
        history = self.get_price_history(symbol, limit=100)
        if len(history) < 2:
            return 0.0

        # 価格変化の標準偏差を計算
        prices = [float(tick.price) for tick in history]
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i - 1]) / prices[i - 1]
            returns.append(ret)

        if not returns:
            return 0.0

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        volatility = variance**0.5

        # 0-1の範囲に正規化
        return min(1.0, volatility * 10)  # スケーリング調整

    def _calculate_volume_impact(
        self, symbol: str, quantity: Decimal, latest_tick: SimulatedTick
    ) -> int:
        """
        出来高インパクト計算

        Args:
            symbol: シンボル
            quantity: 数量
            latest_tick: 最新ティック

        Returns:
            int: 出来高インパクト（bps）
        """
        config = self.slippage_config

        # 市場影響度の計算
        if latest_tick.volume > 0:
            impact_ratio = float(quantity) / float(latest_tick.volume)
            if impact_ratio > float(config.market_impact_threshold):
                # 大きな注文はより大きなインパクト
                return int(impact_ratio * config.volume_impact_bps * 10)
            else:
                return config.volume_impact_bps
        else:
            return config.volume_impact_bps

    def _apply_slippage(
        self, base_price: Decimal, slippage_bps: int, side: str
    ) -> Decimal:
        """
        スリッページ適用

        Args:
            base_price: 基準価格
            slippage_bps: スリッページ（bps）
            side: 売買方向

        Returns:
            Decimal: 適用後価格
        """
        slippage_ratio = Decimal(slippage_bps) / Decimal("10000")  # bps to decimal

        if side == "buy":
            # 買いの場合は価格が上がる
            return base_price * (Decimal("1") + slippage_ratio)
        else:
            # 売りの場合は価格が下がる
            return base_price * (Decimal("1") - slippage_ratio)

    async def _notify_tick_callbacks(self, tick: SimulatedTick) -> None:
        """
        ティックコールバック通知

        Args:
            tick: ティックデータ
        """
        for callback in self.tick_callbacks:
            try:
                await callback(tick)
            except Exception as e:
                self.logger.error(f"Tick callback error: {e}")

    def add_tick_callback(
        self, callback: Callable[[SimulatedTick], Awaitable[None]]
    ) -> None:
        """
        ティックコールバック追加

        Args:
            callback: コールバック関数
        """
        self.tick_callbacks.append(callback)

    def add_bar_callback(self, callback: Callable[[BarData], Awaitable[None]]) -> None:
        """
        バーコールバック追加

        Args:
            callback: コールバック関数
        """
        self.bar_callbacks.append(callback)

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        パフォーマンス統計取得

        Returns:
            Dict[str, Any]: パフォーマンス統計
        """
        avg_latency = (
            self.total_latency_ms / self.latency_samples
            if self.latency_samples > 0
            else 0
        )
        avg_slippage = (
            self.total_slippage_bps / self.slippage_samples
            if self.slippage_samples > 0
            else 0
        )

        return {
            "average_latency_ms": avg_latency,
            "average_slippage_bps": avg_slippage,
            "total_samples": self.latency_samples,
            "subscribed_symbols": len(self.subscribed_symbols),
            "is_running": self.is_running,
        }

    def reset_performance_stats(self) -> None:
        """パフォーマンス統計リセット"""
        self.total_latency_ms = 0
        self.latency_samples = 0
        self.total_slippage_bps = 0
        self.slippage_samples = 0
        self.logger.info("Performance stats reset")
