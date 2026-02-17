"""
V433 Phase 5: Paper Trading Layer - Paper Trading Manager

Paper Trading Layerの統合マネージャー。仮想ポートフォリオ、市場データシミュレーター、
パフォーマンスバリデーターを統合管理する。
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ztb.trading.production.state_persistence import (
    read_state_payload,
    write_state_payload,
)
from market_data_simulator import MarketDataSimulator, SimulatedTick
from performance_validator import PerformanceValidator, ValidationReport
from virtual_portfolio_manager import VirtualPortfolioManager


# Mock classes for testing


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Optional[Decimal] = None
    order_type: OrderType = OrderType.MARKET
    timestamp: Optional[datetime] = None


class MarketDataProvider:
    pass


class PaperTradingState(Enum):
    """Paper Trading状態"""

    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    VALIDATING = "validating"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class PaperTradingConfig:
    """Paper Trading設定"""

    initial_balance: Decimal = Decimal("100000")
    commission_rate: Decimal = Decimal("0.001")
    max_position_size: Decimal = Decimal("0.1")
    max_drawdown_limit: Decimal = Decimal("0.2")
    evaluation_period_days: int = 30
    min_trades_required: int = 30
    validation_interval_hours: int = 24
    auto_validation: bool = True


@dataclass
class PaperTradingSession:
    """Paper Tradingセッション"""

    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    config: PaperTradingConfig = field(default_factory=PaperTradingConfig)
    state: PaperTradingState = PaperTradingState.INITIALIZING

    total_trades: int = 0
    portfolio_value: Decimal = Decimal("0")
    validation_reports: List[ValidationReport] = field(default_factory=list)

    last_validation: Optional[datetime] = None
    next_validation: Optional[datetime] = None


class PaperTradingManager:
    """
    Paper Tradingマネージャー

    Paper Trading Layerの全コンポーネントを統合管理し、
    仮想取引環境の実行と評価を行う。
    """

    def __init__(
        self,
        market_data_provider: MarketDataProvider,
        config: Optional[PaperTradingConfig] = None,
    ):
        """
        初期化

        Args:
            market_data_provider: 市場データプロバイダー
            config: Paper Trading設定
        """
        self.config = config or PaperTradingConfig()
        self.market_data_provider = market_data_provider

        # コンポーネント初期化
        self.portfolio_manager = VirtualPortfolioManager(
            initial_balance=self.config.initial_balance,
            commission_rate=self.config.commission_rate,
            max_position_size=self.config.max_position_size,
            max_drawdown_limit=self.config.max_drawdown_limit,
        )

        self.market_simulator = MarketDataSimulator(
            market_data_provider=market_data_provider
        )

        self.performance_validator = PerformanceValidator(
            min_trades_required=self.config.min_trades_required
        )

        # セッション管理
        self.current_session: Optional[PaperTradingSession] = None
        self.session_history: List[PaperTradingSession] = []

        # コールバック
        self.order_callbacks: List[Callable[[Order, bool], Awaitable[None]]] = []
        self.validation_callbacks: List[
            Callable[[ValidationReport], Awaitable[None]]
        ] = []

        # 状態管理
        self.is_running = False
        self.monitoring_task: Optional[asyncio.Task] = None

        # ロギング
        self.logger = logging.getLogger(__name__)

        # 市場データ更新コールバック設定
        self.market_simulator.add_tick_callback(self._on_market_data_update)

        self.logger.info("Paper Trading Manager initialized")

    async def start_session(self, session_id: Optional[str] = None) -> str:
        """
        セッション開始

        Args:
            session_id: セッションID（指定なしの場合は自動生成）

        Returns:
            str: セッションID
        """
        if (
            self.current_session
            and self.current_session.state == PaperTradingState.RUNNING
        ):
            raise RuntimeError("A session is already running")

        # セッションID生成
        if not session_id:
            session_id = f"PT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 新規セッション作成
        self.current_session = PaperTradingSession(
            session_id=session_id,
            start_time=datetime.now(),
            config=self.config,
            state=PaperTradingState.INITIALIZING,
        )

        try:
            # コンポーネント開始
            self.market_simulator.start()
            self.current_session.state = PaperTradingState.RUNNING

            # モニタリング開始
            if self.config.auto_validation:
                self.monitoring_task = asyncio.create_task(self._monitoring_loop())

            self.logger.info(f"Paper Trading session started: {session_id}")

            return session_id

        except Exception as e:
            self.current_session.state = PaperTradingState.ERROR
            self.logger.error(f"Failed to start session: {e}")
            raise

    async def stop_session(self) -> None:
        """セッション停止"""
        if not self.current_session:
            return

        try:
            # モニタリング停止
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass

            # コンポーネント停止
            self.market_simulator.stop()

            # 最終検証実行
            if self.current_session.state == PaperTradingState.RUNNING:
                await self._perform_validation()

            # セッション完了
            self.current_session.end_time = datetime.now()
            self.current_session.state = PaperTradingState.COMPLETED

            # 履歴保存
            self.session_history.append(self.current_session)

            self.logger.info(
                f"Paper Trading session completed: {self.current_session.session_id}"
            )

        except Exception as e:
            self.current_session.state = PaperTradingState.ERROR
            self.logger.error(f"Error stopping session: {e}")
        finally:
            self.current_session = None

    async def submit_order(self, order: Order) -> bool:
        """
        注文送信

        Args:
            order: 注文オブジェクト

        Returns:
            bool: 注文成功フラグ
        """
        if (
            not self.current_session
            or self.current_session.state != PaperTradingState.RUNNING
        ):
            self.logger.warning("No active session or session not running")
            return False

        # シンボル購読確認
        if order.symbol not in self.market_simulator.subscribed_symbols:
            self.market_simulator.subscribe_symbol(order.symbol)

        # 注文実行
        success = self.portfolio_manager.place_order(order)

        if success:
            self.current_session.total_trades += 1

            # ポートフォリオ価値更新
            self.current_session.portfolio_value = (
                self.portfolio_manager.get_portfolio_value()
            )

        # コールバック実行
        for callback in self.order_callbacks:
            try:
                await callback(order, success)
            except Exception as e:
                self.logger.error(f"Order callback error: {e}")

        return success

    async def _on_market_data_update(self, tick: SimulatedTick) -> None:
        """
        市場データ更新コールバック

        Args:
            tick: シミュレートティック
        """
        try:
            # ポートフォリオ価格更新
            price_updates = {tick.symbol: tick.price}
            self.portfolio_manager.update_prices(price_updates)

            # セッションデータ更新
            if self.current_session:
                self.current_session.portfolio_value = (
                    self.portfolio_manager.get_portfolio_value()
                )

        except Exception as e:
            self.logger.error(f"Market data update error: {e}")

    async def _monitoring_loop(self) -> None:
        """モニタリングループ"""
        while (
            self.current_session
            and self.current_session.state == PaperTradingState.RUNNING
        ):
            try:
                now = datetime.now()

                # 定期検証チェック
                if (
                    self.current_session.last_validation is None
                    or now - self.current_session.last_validation
                    >= timedelta(hours=self.config.validation_interval_hours)
                ):
                    await self._perform_validation()

                # 次回検証時刻更新
                if self.current_session.last_validation:
                    self.current_session.next_validation = (
                        self.current_session.last_validation
                        + timedelta(hours=self.config.validation_interval_hours)
                    )

                # 1時間待機
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # エラー時は1分待機

    async def _perform_validation(self) -> None:
        """検証実行"""
        if not self.current_session:
            return

        try:
            self.current_session.state = PaperTradingState.VALIDATING

            # ポートフォリオ指標取得
            portfolio_metrics = self.portfolio_manager.portfolio_history[
                -1000:
            ]  # 最新1000件

            # 取引履歴取得
            trades = self.portfolio_manager.get_trades(limit=10000)  # 最新10000件

            # 評価期間計算
            evaluation_days = (datetime.now() - self.current_session.start_time).days
            evaluation_days = max(evaluation_days, 1)

            # 検証実行
            report = self.performance_validator.validate_performance(
                portfolio_metrics=portfolio_metrics,
                trades=trades,
                evaluation_period_days=evaluation_days,
            )

            # レポート保存
            self.current_session.validation_reports.append(report)
            self.current_session.last_validation = datetime.now()

            # コールバック実行
            for callback in self.validation_callbacks:
                try:
                    await callback(report)
                except Exception as e:
                    self.logger.error(f"Validation callback error: {e}")

            self.logger.info(
                f"Validation completed. Rating: {report.overall_rating.value}"
            )

        except Exception as e:
            self.logger.error(f"Validation error: {e}")
        finally:
            if self.current_session:
                self.current_session.state = PaperTradingState.RUNNING

    def get_session_status(self) -> Optional[Dict[str, Any]]:
        """
        セッション状態取得

        Returns:
            Optional[Dict[str, Any]]: セッション状態
        """
        if not self.current_session:
            return None

        portfolio_metrics = self.portfolio_manager.get_portfolio_metrics()

        return {
            "session_id": self.current_session.session_id,
            "state": self.current_session.state.value,
            "start_time": self.current_session.start_time.isoformat(),
            "duration_hours": (
                datetime.now() - self.current_session.start_time
            ).total_seconds()
            / 3600,
            "total_trades": self.current_session.total_trades,
            "portfolio_value": str(self.current_session.portfolio_value),
            "total_pnl": str(portfolio_metrics.total_pnl),
            "win_rate": portfolio_metrics.win_rate,
            "max_drawdown": str(portfolio_metrics.max_drawdown),
            "sharpe_ratio": portfolio_metrics.sharpe_ratio,
            "last_validation": self.current_session.last_validation.isoformat()
            if self.current_session.last_validation
            else None,
            "next_validation": self.current_session.next_validation.isoformat()
            if self.current_session.next_validation
            else None,
            "validation_count": len(self.current_session.validation_reports),
        }

    def get_latest_validation_report(self) -> Optional[ValidationReport]:
        """
        最新検証レポート取得

        Returns:
            Optional[ValidationReport]: 最新検証レポート
        """
        if not self.current_session or not self.current_session.validation_reports:
            return None

        return self.current_session.validation_reports[-1]

    def get_portfolio_snapshot(self) -> Dict[str, Any]:
        """
        ポートフォリオスナップショット取得

        Returns:
            Dict[str, Any]: ポートフォリオスナップショット
        """
        positions = self.portfolio_manager.get_positions()
        metrics = self.portfolio_manager.get_portfolio_metrics()

        return {
            "cash_balance": str(metrics.cash_balance),
            "total_value": str(metrics.total_value),
            "total_pnl": str(metrics.total_pnl),
            "realized_pnl": str(metrics.realized_pnl),
            "unrealized_pnl": str(metrics.unrealized_pnl),
            "positions": [
                {
                    "symbol": pos.symbol,
                    "side": pos.side.value,
                    "quantity": str(pos.quantity),
                    "entry_price": str(pos.entry_price),
                    "current_price": str(pos.current_price),
                    "unrealized_pnl": str(pos.unrealized_pnl),
                }
                for pos in positions.values()
            ],
        }

    def pause_trading(self) -> None:
        """取引一時停止"""
        if self.current_session:
            self.current_session.state = PaperTradingState.PAUSED
            self.portfolio_manager.pause_trading()
            self.logger.info("Paper trading paused")

    def resume_trading(self) -> None:
        """取引再開"""
        if self.current_session:
            self.current_session.state = PaperTradingState.RUNNING
            self.portfolio_manager.resume_trading()
            self.logger.info("Paper trading resumed")

    def save_session(self, filepath: str) -> None:
        """
        セッション保存

        Args:
            filepath: 保存ファイルパス
        """
        if not self.current_session:
            return

        # セッションデータ作成
        session_data = {
            "session_id": self.current_session.session_id,
            "start_time": self.current_session.start_time.isoformat(),
            "end_time": self.current_session.end_time.isoformat()
            if self.current_session.end_time
            else None,
            "state": self.current_session.state.value,
            "config": {
                "initial_balance": str(self.config.initial_balance),
                "commission_rate": str(self.config.commission_rate),
                "max_position_size": str(self.config.max_position_size),
                "max_drawdown_limit": str(self.config.max_drawdown_limit),
                "evaluation_period_days": self.config.evaluation_period_days,
                "min_trades_required": self.config.min_trades_required,
                "validation_interval_hours": self.config.validation_interval_hours,
                "auto_validation": self.config.auto_validation,
            },
            "total_trades": self.current_session.total_trades,
            "portfolio_value": str(self.current_session.portfolio_value),
            "last_validation": self.current_session.last_validation.isoformat()
            if self.current_session.last_validation
            else None,
            "validation_reports": [
                {
                    "timestamp": report.validation_timestamp.isoformat(),
                    "rating": report.overall_rating.value,
                    "total_trades": report.total_trades,
                    "evaluation_period_days": report.evaluation_period_days,
                    "recommendations": report.recommendations,
                    "warnings": report.warnings,
                    "critical_issues": report.critical_issues,
                }
                for report in self.current_session.validation_reports
            ],
        }

        # ポートフォリオ状態保存
        portfolio_file = f"{filepath}.portfolio"
        self.portfolio_manager.save_state(portfolio_file)

        # セッション情報保存
        write_state_payload(filepath, session_data)

        self.logger.info(f"Session saved to {filepath}")

    def load_session(self, filepath: str) -> bool:
        """
        セッション読み込み

        Args:
            filepath: 読み込みファイルパス

        Returns:
            bool: 読み込み成功フラグ
        """
        try:
            session_data = read_state_payload(filepath)

            # 設定復元
            config_data = session_data["config"]
            self.config = PaperTradingConfig(
                initial_balance=Decimal(config_data["initial_balance"]),
                commission_rate=Decimal(config_data["commission_rate"]),
                max_position_size=Decimal(config_data["max_position_size"]),
                max_drawdown_limit=Decimal(config_data["max_drawdown_limit"]),
                evaluation_period_days=config_data["evaluation_period_days"],
                min_trades_required=config_data["min_trades_required"],
                validation_interval_hours=config_data["validation_interval_hours"],
                auto_validation=config_data["auto_validation"],
            )

            # セッション復元
            self.current_session = PaperTradingSession(
                session_id=session_data["session_id"],
                start_time=datetime.fromisoformat(session_data["start_time"]),
                end_time=datetime.fromisoformat(session_data["end_time"])
                if session_data["end_time"]
                else None,
                config=self.config,
                state=PaperTradingState(session_data["state"]),
                total_trades=session_data["total_trades"],
                portfolio_value=Decimal(session_data["portfolio_value"]),
                last_validation=datetime.fromisoformat(session_data["last_validation"])
                if session_data["last_validation"]
                else None,
            )

            # 検証レポート復元
            for report_data in session_data.get("validation_reports", []):
                # 簡易復元（完全なレポートオブジェクトは再生成が必要）
                pass

            # ポートフォリオ状態読み込み
            portfolio_file = f"{filepath}.portfolio"
            if os.path.exists(portfolio_file):
                self.portfolio_manager.load_state(portfolio_file)

            self.logger.info(f"Session loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load session: {e}")
            return False

    def add_order_callback(
        self, callback: Callable[[Order, bool], Awaitable[None]]
    ) -> None:
        """
        注文コールバック追加

        Args:
            callback: コールバック関数
        """
        self.order_callbacks.append(callback)

    def add_validation_callback(
        self, callback: Callable[[ValidationReport], Awaitable[None]]
    ) -> None:
        """
        検証コールバック追加

        Args:
            callback: コールバック関数
        """
        self.validation_callbacks.append(callback)

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        パフォーマンス統計取得

        Returns:
            Dict[str, Any]: パフォーマンス統計
        """
        portfolio_stats = self.portfolio_manager.get_portfolio_metrics()
        market_stats = self.market_simulator.get_performance_stats()

        return {
            "portfolio": {
                "total_value": str(portfolio_stats.total_value),
                "total_pnl": str(portfolio_stats.total_pnl),
                "win_rate": portfolio_stats.win_rate,
                "max_drawdown": str(portfolio_stats.max_drawdown),
                "sharpe_ratio": portfolio_stats.sharpe_ratio,
                "total_trades": portfolio_stats.total_trades,
            },
            "market_simulation": market_stats,
            "session": self.get_session_status() if self.current_session else None,
        }
