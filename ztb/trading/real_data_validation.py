#!/usr/bin/env python3
"""
V433 Phase 4: リアルデータ検証システム
ライブ取引環境でのパフォーマンス検証と安定性テスト
"""

import threading
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

from ztb.trading.v433_integration_manager import V433IntegrationManager
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class LiveValidationConfig:
    """ライブ検証設定"""

    symbol: str = "btc_jpy"
    validation_period_days: int = 30  # 検証期間（日）
    paper_trading_balance: float = 100000.0  # ペーパートレーディング残高
    max_position_size_pct: float = 0.05  # 最大ポジションサイズ（残高の5%）
    risk_per_trade_pct: float = 0.01  # 1トレードあたりのリスク（1%）
    min_trade_interval_minutes: int = 5  # 最小取引間隔（分）
    max_daily_trades: int = 10  # 1日あたりの最大取引数
    validation_mode: str = "paper_trading"  # 検証モード


@dataclass
class LiveTradeRecord:
    """ライブ取引記録"""

    timestamp: datetime
    symbol: str
    action: str
    quantity: float
    price: float
    reason: str
    pnl: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    confidence: float = 0.0
    market_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveValidationMetrics:
    """ライブ検証指標"""

    start_time: datetime
    end_time: Optional[datetime] = None
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    max_drawdown: float = 0.0
    peak_balance: float = 0.0
    current_balance: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    trade_frequency: float = 0.0  # trades per day
    execution_quality: float = 0.0  # slippage as % of spread
    market_adaptation: float = 0.0  # performance vs market
    stability_score: float = 0.0  # consistency measure


@dataclass
class MarketCondition:
    """市場状況"""

    timestamp: datetime
    volatility: float
    trend_strength: float
    volume_profile: str
    liquidity_score: float
    market_regime: str  # trending, ranging, volatile
    sentiment_score: float


class PaperTradingEngine:
    """ペーパートレーディングエンジン"""

    def __init__(
        self, integration_manager: V433IntegrationManager, config: LiveValidationConfig
    ):
        self.integration_manager = integration_manager
        self.config = config
        self.logger = get_logger(__name__)

        # 取引状態
        self.balance = config.paper_trading_balance
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[LiveTradeRecord] = []
        self.equity_curve: List[Tuple[datetime, float]] = [
            (datetime.now(), config.paper_trading_balance)
        ]

        # リスク管理
        self.daily_trade_count = 0
        self.last_trade_time = None
        self.daily_reset_time = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        # パフォーマンス追跡
        self.metrics = LiveValidationMetrics(
            start_time=datetime.now(), current_balance=config.paper_trading_balance
        )

    def execute_paper_trade(
        self, signal: Dict[str, Any], market_data: Dict[str, Any]
    ) -> Optional[LiveTradeRecord]:
        """ペーパートレード実行"""
        try:
            # 取引間隔チェック
            if (
                self.last_trade_time
                and (datetime.now() - self.last_trade_time).seconds
                < self.config.min_trade_interval_minutes * 60
            ):
                return None

            # 日次取引数制限チェック
            if datetime.now().date() > self.daily_reset_time.date():
                self.daily_trade_count = 0
                self.daily_reset_time = datetime.now().replace(
                    hour=0, minute=0, second=0, microsecond=0
                )

            if self.daily_trade_count >= self.config.max_daily_trades:
                return None

            # ポジションサイズ計算
            position_size = self._calculate_position_size(signal, market_data)

            if position_size <= 0:
                return None

            # 取引実行
            if signal["action"] in ["open_long", "open_short"]:
                return self._open_position(signal, market_data, position_size)
            elif signal["action"] == "close_position":
                return self._close_position(signal, market_data)

            return None

        except Exception as e:
            self.logger.error(f"Paper trade execution failed: {e}")
            return None

    def _open_position(
        self, signal: Dict[str, Any], market_data: Dict[str, Any], position_size: float
    ) -> LiveTradeRecord:
        """ポジションオープン"""
        symbol = signal["symbol"]
        action = signal["action"]
        price = market_data.get("price", market_data.get("close", 0))

        if price <= 0:
            return None

        # 取引コスト計算
        commission = price * position_size * 0.001  # 0.1% commission
        slippage = price * position_size * 0.0005  # 0.05% slippage

        # 残高チェック
        total_cost = price * position_size + commission
        if self.balance < total_cost:
            return None

        # ポジション作成
        position = {
            "symbol": symbol,
            "side": "long" if action == "open_long" else "short",
            "quantity": position_size,
            "entry_price": price,
            "entry_time": datetime.now(),
            "commission": commission,
            "slippage": slippage,
        }

        self.positions[symbol] = position
        self.balance -= total_cost

        # 取引記録
        trade_record = LiveTradeRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            action=action,
            quantity=position_size,
            price=price,
            reason=signal.get("reason", "paper_trade"),
            commission=commission,
            slippage=slippage,
            confidence=signal.get("confidence", 0.0),
            market_conditions=self._capture_market_conditions(market_data),
        )

        self.trade_history.append(trade_record)
        self.daily_trade_count += 1
        self.last_trade_time = datetime.now()

        # エクイティ更新
        self._update_equity_curve()

        # 指標更新
        self._update_metrics()

        return trade_record

    def _close_position(
        self, signal: Dict[str, Any], market_data: Dict[str, Any]
    ) -> LiveTradeRecord:
        """ポジションクローズ"""
        symbol = signal["symbol"]
        price = market_data.get("price", market_data.get("close", 0))

        if symbol not in self.positions or price <= 0:
            return None

        position = self.positions[symbol]

        # 取引コスト計算
        commission = price * position["quantity"] * 0.001
        slippage = price * position["quantity"] * 0.0005

        # P&L計算
        if position["side"] == "long":
            gross_pnl = (price - position["entry_price"]) * position["quantity"]
        else:
            gross_pnl = (position["entry_price"] - price) * position["quantity"]

        net_pnl = (
            gross_pnl
            - commission
            - slippage
            - position["commission"]
            - position["slippage"]
        )

        # 残高更新
        self.balance += price * position["quantity"] - commission

        # 取引記録
        trade_record = LiveTradeRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            action="close_position",
            quantity=position["quantity"],
            price=price,
            reason=signal.get("reason", "paper_trade"),
            pnl=net_pnl,
            commission=commission,
            slippage=slippage,
            confidence=signal.get("confidence", 0.0),
            market_conditions=self._capture_market_conditions(market_data),
        )

        self.trade_history.append(trade_record)
        self.daily_trade_count += 1
        self.last_trade_time = datetime.now()

        # ポジション削除
        del self.positions[symbol]

        # エクイティ更新
        self._update_equity_curve()

        # 指標更新
        self._update_metrics()

        return trade_record

    def _calculate_position_size(
        self, signal: Dict[str, Any], market_data: Dict[str, Any]
    ) -> float:
        """ポジションサイズ計算"""
        # リスクベースのポジションサイジング
        risk_amount = self.config.paper_trading_balance * self.config.risk_per_trade_pct
        stop_loss_pct = 0.02  # 2%ストップロス

        # 最大ポジションサイズ制限
        max_position_value = (
            self.config.paper_trading_balance * self.config.max_position_size_pct
        )

        # 価格取得
        price = market_data.get("price", market_data.get("close", 0))
        if price <= 0:
            return 0

        # リスクベースの計算
        position_value = risk_amount / stop_loss_pct
        quantity = min(position_value / price, max_position_value / price)

        # 残高チェック
        max_quantity_by_balance = (self.balance * 0.95) / price  # 95%まで使用
        quantity = min(quantity, max_quantity_by_balance)

        return max(0, quantity)

    def _capture_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """市場状況のキャプチャ"""
        return {
            "price": market_data.get("price", 0),
            "volume": market_data.get("volume", 0),
            "volatility": market_data.get("volatility", 0),
            "trend": market_data.get("trend", "neutral"),
            "liquidity": market_data.get("liquidity", "normal"),
        }

    def _update_equity_curve(self):
        """エクイティ曲線の更新"""
        current_equity = self.balance

        # ポジション価値の追加
        for position in self.positions.values():
            # 簡易的な現在価格（実際の実装ではリアルタイム価格を使用）
            current_price = position["entry_price"]  # 簡易実装
            if position["side"] == "long":
                current_equity += current_price * position["quantity"]
            else:
                current_equity += (
                    2 * position["entry_price"] - current_price
                ) * position["quantity"]

        self.equity_curve.append((datetime.now(), current_equity))

    def _update_metrics(self):
        """指標の更新"""
        if not self.trade_history:
            return

        # 取引数
        self.metrics.total_trades = len(
            [t for t in self.trade_history if t.pnl is not None]
        )

        # 勝敗数
        winning_trades = [
            t for t in self.trade_history if t.pnl is not None and t.pnl > 0
        ]
        losing_trades = [
            t for t in self.trade_history if t.pnl is not None and t.pnl < 0
        ]

        self.metrics.winning_trades = len(winning_trades)
        self.metrics.losing_trades = len(losing_trades)

        # P&L
        self.metrics.total_pnl = sum(
            t.pnl for t in self.trade_history if t.pnl is not None
        )
        self.metrics.total_commission = sum(t.commission for t in self.trade_history)
        self.metrics.total_slippage = sum(t.slippage for t in self.trade_history)

        # 勝率
        if self.metrics.total_trades > 0:
            self.metrics.win_rate = (
                self.metrics.winning_trades / self.metrics.total_trades
            )

        # プロフィットファクター
        total_win = sum(t.pnl for t in winning_trades)
        total_loss = abs(sum(t.pnl for t in losing_trades))
        if total_loss > 0:
            self.metrics.profit_factor = total_win / total_loss

        # 平均取引
        if self.metrics.total_trades > 0:
            self.metrics.avg_trade_pnl = (
                self.metrics.total_pnl / self.metrics.total_trades
            )

        if winning_trades:
            self.metrics.avg_win = np.mean([t.pnl for t in winning_trades])
            self.metrics.largest_win = max(t.pnl for t in winning_trades)

        if losing_trades:
            self.metrics.avg_loss = abs(np.mean([t.pnl for t in losing_trades]))
            self.metrics.largest_loss = abs(min(t.pnl for t in losing_trades))

        # ドローダウン
        equity_values = [e[1] for e in self.equity_curve]
        if equity_values:
            peak = max(equity_values)
            current = equity_values[-1]
            self.metrics.max_drawdown = max(
                self.metrics.max_drawdown, (peak - current) / peak
            )
            self.metrics.peak_balance = max(self.metrics.peak_balance, peak)
            self.metrics.current_balance = current

        # 取引頻度
        if len(self.equity_curve) > 1:
            days_elapsed = (
                self.equity_curve[-1][0] - self.equity_curve[0][0]
            ).total_seconds() / (24 * 3600)
            if days_elapsed > 0:
                self.metrics.trade_frequency = self.metrics.total_trades / days_elapsed


class MarketConditionAnalyzer:
    """市場状況分析器"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.market_history: List[MarketCondition] = []

    def analyze_market_conditions(self, market_data: Dict[str, Any]) -> MarketCondition:
        """市場状況の分析"""
        timestamp = datetime.now()

        # ボラティリティ計算
        volatility = self._calculate_volatility(market_data)

        # トレンド強度計算
        trend_strength = self._calculate_trend_strength(market_data)

        # 出来高プロファイル分析
        volume_profile = self._analyze_volume_profile(market_data)

        # 流動性スコア計算
        liquidity_score = self._calculate_liquidity_score(market_data)

        # 市場レジーム判定
        market_regime = self._determine_market_regime(volatility, trend_strength)

        # センチメントスコア
        sentiment_score = self._calculate_sentiment_score(market_data)

        condition = MarketCondition(
            timestamp=timestamp,
            volatility=volatility,
            trend_strength=trend_strength,
            volume_profile=volume_profile,
            liquidity_score=liquidity_score,
            market_regime=market_regime,
            sentiment_score=sentiment_score,
        )

        self.market_history.append(condition)

        # 古いデータを削除（最近1000件のみ保持）
        if len(self.market_history) > 1000:
            self.market_history = self.market_history[-1000:]

        return condition

    def _calculate_volatility(self, market_data: Dict[str, Any]) -> float:
        """ボラティリティ計算"""
        # 価格変動の標準偏差
        prices = market_data.get("price_history", [])
        if len(prices) < 2:
            return 0.0

        returns = np.diff(prices) / prices[:-1]
        return np.std(returns) if len(returns) > 0 else 0.0

    def _calculate_trend_strength(self, market_data: Dict[str, Any]) -> float:
        """トレンド強度計算"""
        # ADXのようなトレンド強度指標
        prices = market_data.get("price_history", [])
        if len(prices) < 14:
            return 0.0

        # 簡易的なトレンド強度（価格の線形回帰勾配）
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        trend_strength = abs(slope) / np.mean(prices)

        return min(trend_strength, 1.0)  # 0-1に正規化

    def _analyze_volume_profile(self, market_data: Dict[str, Any]) -> str:
        """出来高プロファイル分析"""
        volume = market_data.get("volume", 0)
        avg_volume = market_data.get("avg_volume", 1)

        if volume > avg_volume * 1.5:
            return "high"
        elif volume < avg_volume * 0.5:
            return "low"
        else:
            return "normal"

    def _calculate_liquidity_score(self, market_data: Dict[str, Any]) -> float:
        """流動性スコア計算"""
        # スプレッドと出来高に基づく流動性
        spread = market_data.get("spread", 0.001)  # デフォルト0.1%
        volume = market_data.get("volume", 1)
        avg_volume = market_data.get("avg_volume", 1)

        # 流動性スコア（0-1、高いほど良い）
        spread_score = max(0, 1 - spread / 0.01)  # 1%スプレッドで0
        volume_score = min(1, volume / avg_volume)

        return (spread_score + volume_score) / 2

    def _determine_market_regime(self, volatility: float, trend_strength: float) -> str:
        """市場レジーム判定"""
        if trend_strength > 0.7 and volatility < 0.02:
            return "strong_trend"
        elif trend_strength > 0.5:
            return "trending"
        elif volatility > 0.05:
            return "volatile"
        else:
            return "ranging"

    def _calculate_sentiment_score(self, market_data: Dict[str, Any]) -> float:
        """センチメントスコア計算"""
        # 簡易的なセンチメント（価格モメンタム）
        momentum = market_data.get("momentum", 0)
        return np.tanh(momentum / 0.01)  # -1 to 1


class StabilityTester:
    """安定性テスター"""

    def __init__(self, paper_trading_engine: PaperTradingEngine):
        self.paper_trading_engine = paper_trading_engine
        self.logger = get_logger(__name__)

        # 安定性指標
        self.performance_stability: List[float] = []
        self.adaptation_metrics: Dict[str, Any] = {}

    def run_stability_tests(self) -> Dict[str, Any]:
        """安定性テスト実行"""
        self.logger.info("Running stability tests...")

        stability_results = {}

        # 1. パフォーマンス安定性テスト
        stability_results["performance_stability"] = self._test_performance_stability()

        # 2. 市場適応性テスト
        stability_results["market_adaptation"] = self._test_market_adaptation()

        # 3. ストレス耐性テスト
        stability_results["stress_resilience"] = self._test_stress_resilience()

        # 4. 回復力テスト
        stability_results["recovery_capability"] = self._test_recovery_capability()

        # 5. 一貫性テスト
        stability_results["consistency_analysis"] = self._analyze_consistency()

        # 総合安定性スコア
        stability_results[
            "overall_stability_score"
        ] = self._calculate_overall_stability_score(stability_results)

        return stability_results

    def _test_performance_stability(self) -> Dict[str, Any]:
        """パフォーマンス安定性テスト"""
        if len(self.paper_trading_engine.equity_curve) < 10:
            return {"stability_score": 0.0, "volatility": 0.0}

        # エクイティ曲線の安定性分析
        equity_values = [e[1] for e in self.paper_trading_engine.equity_curve]
        returns = np.diff(equity_values) / equity_values[:-1]

        # 安定性スコア（低いボラティリティが良い）
        volatility = np.std(returns) if len(returns) > 0 else 0
        stability_score = max(0, 1 - volatility * 10)  # 10%ボラティリティで0

        return {
            "stability_score": stability_score,
            "return_volatility": volatility,
            "sharpe_ratio": self.paper_trading_engine.metrics.sharpe_ratio,
            "max_drawdown": self.paper_trading_engine.metrics.max_drawdown,
        }

    def _test_market_adaptation(self) -> Dict[str, Any]:
        """市場適応性テスト"""
        # 市場状況ごとのパフォーマンス分析
        market_performance = {}

        for trade in self.paper_trading_engine.trade_history:
            if trade.pnl is not None:
                regime = trade.market_conditions.get("regime", "unknown")
                if regime not in market_performance:
                    market_performance[regime] = []
                market_performance[regime].append(trade.pnl)

        # 各市場状況での平均パフォーマンス
        adaptation_scores = {}
        for regime, pnls in market_performance.items():
            if pnls:
                avg_pnl = np.mean(pnls)
                win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
                adaptation_scores[regime] = {
                    "avg_pnl": avg_pnl,
                    "win_rate": win_rate,
                    "consistency": 1 - np.std(pnls) / abs(avg_pnl)
                    if avg_pnl != 0
                    else 0,
                }

        # 全体適応性スコア
        if adaptation_scores:
            consistency_scores = [
                score["consistency"] for score in adaptation_scores.values()
            ]
            overall_adaptation = np.mean(consistency_scores)
        else:
            overall_adaptation = 0.0

        return {
            "market_performance": adaptation_scores,
            "overall_adaptation_score": overall_adaptation,
        }

    def _test_stress_resilience(self) -> Dict[str, Any]:
        """ストレス耐性テスト"""
        # ドローダウン中のパフォーマンス分析
        equity_values = [e[1] for e in self.paper_trading_engine.equity_curve]

        if len(equity_values) < 20:
            return {"stress_resilience": 0.0}

        # ピークからのドローダウン期間を特定
        peak = max(equity_values)
        peak_idx = equity_values.index(peak)

        # ドローダウン期間中の取引
        drawdown_trades = []
        for trade in self.paper_trading_engine.trade_history:
            if trade.timestamp > self.paper_trading_engine.equity_curve[peak_idx][0]:
                drawdown_trades.append(trade)

        # ストレス耐性スコア
        if drawdown_trades:
            winning_stress_trades = sum(
                1 for t in drawdown_trades if t.pnl and t.pnl > 0
            )
            stress_win_rate = winning_stress_trades / len(drawdown_trades)
            stress_resilience = stress_win_rate  # ストレス時の勝率
        else:
            stress_resilience = 0.5  # デフォルト中間値

        return {
            "stress_resilience": stress_resilience,
            "drawdown_trades": len(drawdown_trades),
            "stress_win_rate": stress_win_rate if "stress_win_rate" in locals() else 0,
        }

    def _test_recovery_capability(self) -> Dict[str, Any]:
        """回復力テスト"""
        equity_values = [e[1] for e in self.paper_trading_engine.equity_curve]

        if len(equity_values) < 10:
            return {"recovery_speed": 0.0}

        # ドローダウンからの回復速度
        peak = max(equity_values)
        trough = min(equity_values[equity_values.index(peak) :])

        if peak > 0 and trough < peak:
            drawdown_pct = (peak - trough) / peak

            # 回復にかかる時間
            trough_idx = equity_values.index(trough)
            recovery_idx = None

            for i in range(trough_idx, len(equity_values)):
                if equity_values[i] >= peak * 0.95:  # 95%回復
                    recovery_idx = i
                    break

            if recovery_idx:
                recovery_periods = recovery_idx - trough_idx
                recovery_speed = 1 / (recovery_periods + 1)  # 速い回復ほど高いスコア
            else:
                recovery_speed = 0.0
        else:
            recovery_speed = 1.0  # ドローダウンなし

        return {
            "recovery_speed": recovery_speed,
            "max_drawdown": self.paper_trading_engine.metrics.max_drawdown,
        }

    def _analyze_consistency(self) -> Dict[str, Any]:
        """一貫性分析"""
        if len(self.paper_trading_engine.trade_history) < 10:
            return {"consistency_score": 0.0}

        # 取引結果の一貫性分析
        pnls = [
            t.pnl for t in self.paper_trading_engine.trade_history if t.pnl is not None
        ]

        if not pnls:
            return {"consistency_score": 0.0}

        # 一貫性指標
        win_streak_max = 0
        loss_streak_max = 0
        current_win_streak = 0
        current_loss_streak = 0

        for pnl in pnls:
            if pnl > 0:
                current_win_streak += 1
                current_loss_streak = 0
                win_streak_max = max(win_streak_max, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                loss_streak_max = max(loss_streak_max, current_loss_streak)

        # 一貫性スコア（勝率の安定性とストリークのバランス）
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
        pnl_volatility = (
            np.std(pnls) / abs(np.mean(pnls)) if np.mean(pnls) != 0 else float("inf")
        )

        consistency_score = win_rate * (
            1 - min(pnl_volatility, 1)
        )  # 低いボラティリティほど良い

        return {
            "consistency_score": consistency_score,
            "win_rate": win_rate,
            "pnl_volatility": pnl_volatility,
            "max_win_streak": win_streak_max,
            "max_loss_streak": loss_streak_max,
        }

    def _calculate_overall_stability_score(
        self, stability_results: Dict[str, Any]
    ) -> float:
        """総合安定性スコア計算"""
        weights = {
            "performance_stability": 0.3,
            "market_adaptation": 0.25,
            "stress_resilience": 0.2,
            "recovery_capability": 0.15,
            "consistency_analysis": 0.1,
        }

        overall_score = 0.0

        for test_name, weight in weights.items():
            if test_name in stability_results:
                test_result = stability_results[test_name]

                # 各テストのスコア抽出
                if test_name == "performance_stability":
                    score = test_result.get("stability_score", 0)
                elif test_name == "market_adaptation":
                    score = test_result.get("overall_adaptation_score", 0)
                elif test_name == "stress_resilience":
                    score = test_result.get("stress_resilience", 0)
                elif test_name == "recovery_capability":
                    score = test_result.get("recovery_speed", 0)
                elif test_name == "consistency_analysis":
                    score = test_result.get("consistency_score", 0)
                else:
                    score = 0

                overall_score += score * weight

        return overall_score


class RealDataValidationSystem:
    """
    V433 Phase 4: リアルデータ検証システム
    ライブ取引環境でのパフォーマンス検証と安定性テスト
    """

    def __init__(self, integration_manager: V433IntegrationManager):
        self.integration_manager = integration_manager
        self.logger = get_logger(__name__)

        # 検証コンポーネント
        self.paper_trading_engine = None
        self.market_analyzer = MarketConditionAnalyzer()
        self.stability_tester = None

        # 検証状態
        self.is_validating = False
        self.validation_start_time = None
        self.validation_config = None

        # 結果保存
        self.validation_results: Dict[str, Any] = {}
        self.performance_history: List[Dict[str, Any]] = []

    def start_live_validation(self, config: LiveValidationConfig) -> bool:
        """ライブ検証開始"""
        if self.is_validating:
            self.logger.warning("Live validation already running")
            return False

        try:
            self.logger.info(f"Starting live validation for {config.symbol}")

            # 設定保存
            self.validation_config = config

            # ペーパートレーディングエンジン初期化
            self.paper_trading_engine = PaperTradingEngine(
                self.integration_manager, config
            )

            # 安定性テスター初期化
            self.stability_tester = StabilityTester(self.paper_trading_engine)

            # 検証状態設定
            self.is_validating = True
            self.validation_start_time = datetime.now()

            # 継続的な検証ループ開始
            validation_thread = threading.Thread(
                target=self._validation_loop, daemon=True
            )
            validation_thread.start()

            self.logger.info("Live validation started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start live validation: {e}")
            self.is_validating = False
            return False

    def stop_live_validation(self) -> Dict[str, Any]:
        """ライブ検証停止"""
        if not self.is_validating:
            return {"error": "No active validation"}

        self.logger.info("Stopping live validation...")

        self.is_validating = False

        # 最終結果生成
        final_results = self.generate_validation_report()

        # 全ポジション決済
        if self.paper_trading_engine:
            self._close_all_positions()

        self.logger.info("Live validation stopped")
        return final_results

    def _validation_loop(self):
        """検証ループ"""
        last_signal_check = datetime.now()
        signal_check_interval = timedelta(minutes=1)  # 1分間隔でシグナルチェック

        while self.is_validating:
            try:
                current_time = datetime.now()

                # シグナルチェック
                if current_time - last_signal_check >= signal_check_interval:
                    self._check_and_execute_signals()
                    last_signal_check = current_time

                # パフォーマンス監視
                self._monitor_performance()

                # 1分待機
                time.sleep(60)

            except Exception as e:
                self.logger.error(f"Validation loop error: {e}")
                time.sleep(300)  # エラー時は5分待機

    def _check_and_execute_signals(self):
        """シグナルチェックと実行"""
        try:
            # V433システムからシグナル取得
            # 実際の実装ではV433のシグナル生成を使用
            signal = self._generate_validation_signal()

            if signal:
                # 市場データ取得
                market_data = self._get_current_market_data()

                if market_data:
                    # 市場状況分析
                    market_condition = self.market_analyzer.analyze_market_conditions(
                        market_data
                    )

                    # ペーパートレード実行
                    trade_record = self.paper_trading_engine.execute_paper_trade(
                        signal, market_data
                    )

                    if trade_record:
                        self.logger.info(
                            f"Executed paper trade: {trade_record.action} {trade_record.quantity} "
                            f"{trade_record.symbol} at {trade_record.price}"
                        )

        except Exception as e:
            self.logger.error(f"Signal check/execution error: {e}")

    def _generate_validation_signal(self) -> Optional[Dict[str, Any]]:
        """検証用シグナル生成"""
        # 実際の実装ではV433システムのシグナルを使用
        # ここでは簡易的なランダムシグナルを生成（検証目的）

        if np.random.random() < 0.05:  # 5%の確率でシグナル
            action = np.random.choice(["open_long", "close_position"])
            return {
                "action": action,
                "symbol": self.validation_config.symbol,
                "quantity": 0.001,
                "confidence": np.random.uniform(0.5, 0.9),
                "reason": "validation_test",
            }

        return None

    def _get_current_market_data(self) -> Optional[Dict[str, Any]]:
        """現在の市場データ取得"""
        try:
            # V433システムから市場データ取得
            current_price = self.integration_manager.component_manager.v433_system.current_prices.get(
                self.validation_config.symbol, 5000000.0
            )

            return {
                "price": current_price,
                "close": current_price,
                "volume": np.random.lognormal(10, 1),  # シミュレーション
                "volatility": 0.02,  # シミュレーション
                "trend": "neutral",
                "liquidity": "normal",
                "price_history": [current_price] * 20,  # 簡易履歴
            }

        except Exception as e:
            self.logger.error(f"Failed to get market data: {e}")
            return None

    def _monitor_performance(self):
        """パフォーマンス監視"""
        if not self.paper_trading_engine:
            return

        # 現在の指標取得
        metrics = self.paper_trading_engine.metrics

        # パフォーマンス履歴保存
        performance_snapshot = {
            "timestamp": datetime.now(),
            "balance": metrics.current_balance,
            "total_pnl": metrics.total_pnl,
            "win_rate": metrics.win_rate,
            "total_trades": metrics.total_trades,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
        }

        self.performance_history.append(performance_snapshot)

        # 最近の履歴のみ保持
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

    def _close_all_positions(self):
        """全ポジション決済"""
        if not self.paper_trading_engine:
            return

        for symbol in list(self.paper_trading_engine.positions.keys()):
            signal = {
                "action": "close_position",
                "symbol": symbol,
                "reason": "validation_end",
            }

            market_data = self._get_current_market_data()
            if market_data:
                self.paper_trading_engine._close_position(signal, market_data)

    def generate_validation_report(self) -> Dict[str, Any]:
        """検証レポート生成"""
        if not self.paper_trading_engine:
            return {"error": "No validation data available"}

        self.logger.info("Generating validation report...")

        # 基本指標
        metrics = self.paper_trading_engine.metrics
        metrics.end_time = datetime.now()

        # 安定性テスト実行
        stability_results = self.stability_tester.run_stability_tests()

        # パフォーマンス分析
        performance_analysis = self._analyze_validation_performance()

        # 市場適応分析
        market_adaptation = self._analyze_market_adaptation()

        # リスク分析
        risk_analysis = self._analyze_validation_risks()

        # 推奨事項
        recommendations = self._generate_validation_recommendations(
            performance_analysis, stability_results, risk_analysis
        )

        report = {
            "validation_period": {
                "start_time": self.validation_start_time,
                "end_time": metrics.end_time,
                "duration_days": (metrics.end_time - self.validation_start_time).days,
            },
            "performance_metrics": {
                "total_return": metrics.total_pnl
                / self.validation_config.paper_trading_balance,
                "total_pnl": metrics.total_pnl,
                "win_rate": metrics.win_rate,
                "total_trades": metrics.total_trades,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "profit_factor": metrics.profit_factor,
                "avg_trade_pnl": metrics.avg_trade_pnl,
            },
            "stability_analysis": stability_results,
            "performance_analysis": performance_analysis,
            "market_adaptation": market_adaptation,
            "risk_analysis": risk_analysis,
            "recommendations": recommendations,
            "trade_history": [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "action": t.action,
                    "symbol": t.symbol,
                    "quantity": t.quantity,
                    "price": t.price,
                    "pnl": t.pnl,
                    "reason": t.reason,
                }
                for t in self.paper_trading_engine.trade_history
            ],
            "equity_curve": [
                {"timestamp": t[0].isoformat(), "equity": t[1]}
                for t in self.paper_trading_engine.equity_curve
            ],
        }

        # 結果保存
        self.validation_results = report

        return report

    def _analyze_validation_performance(self) -> Dict[str, Any]:
        """検証パフォーマンス分析"""
        if not self.performance_history:
            return {}

        # パフォーマンスの時系列分析
        balances = [p["balance"] for p in self.performance_history]
        returns = np.diff(balances) / balances[:-1]

        return {
            "performance_trend": "improving"
            if len(returns) > 10 and returns[-1] > returns[0]
            else "declining",
            "best_period": max(balances) / self.validation_config.paper_trading_balance
            - 1,
            "worst_period": min(balances) / self.validation_config.paper_trading_balance
            - 1,
            "performance_volatility": np.std(returns) if len(returns) > 0 else 0,
            "consistency_score": 1 - (np.std(returns) / abs(np.mean(returns)))
            if len(returns) > 0 and np.mean(returns) != 0
            else 0,
        }

    def _analyze_market_adaptation(self) -> Dict[str, Any]:
        """市場適応分析"""
        # 市場状況ごとのパフォーマンス
        market_performance = {}

        for trade in self.paper_trading_engine.trade_history:
            if trade.pnl is not None:
                regime = trade.market_conditions.get("regime", "unknown")
                if regime not in market_performance:
                    market_performance[regime] = []
                market_performance[regime].append(trade.pnl)

        adaptation_analysis = {}
        for regime, pnls in market_performance.items():
            if pnls:
                adaptation_analysis[regime] = {
                    "avg_pnl": np.mean(pnls),
                    "win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
                    "total_trades": len(pnls),
                }

        return adaptation_analysis

    def _analyze_validation_risks(self) -> Dict[str, Any]:
        """検証リスク分析"""
        metrics = self.paper_trading_engine.metrics

        return {
            "max_drawdown_risk": "high"
            if metrics.max_drawdown > 0.1
            else "moderate"
            if metrics.max_drawdown > 0.05
            else "low",
            "volatility_risk": "high"
            if metrics.sharpe_ratio < 0.5
            else "moderate"
            if metrics.sharpe_ratio < 1.0
            else "low",
            "concentration_risk": "high"
            if len(self.paper_trading_engine.positions) > 3
            else "low",
            "liquidity_risk": "low",  # ペーパートレーディングなので低い
            "operational_risk": "low",  # 自動化されているため
        }

    def _generate_validation_recommendations(
        self,
        performance: Dict[str, Any],
        stability: Dict[str, Any],
        risk: Dict[str, Any],
    ) -> List[str]:
        """検証推奨事項生成"""
        recommendations = []

        # パフォーマンスベースの推奨
        if performance.get("consistency_score", 0) < 0.5:
            recommendations.append(
                "Improve consistency - current performance is volatile"
            )

        # 安定性ベースの推奨
        stability_score = stability.get("overall_stability_score", 0)
        if stability_score < 0.6:
            recommendations.append(
                "Enhance stability - consider risk management improvements"
            )

        # リスクベースの推奨
        if risk.get("max_drawdown_risk") == "high":
            recommendations.append(
                "Reduce drawdown risk - implement stricter stop losses"
            )

        if risk.get("volatility_risk") == "high":
            recommendations.append("Reduce volatility - optimize position sizing")

        # デフォルト推奨
        if not recommendations:
            recommendations.append(
                "Validation successful - ready for live trading consideration"
            )

        return recommendations

    def get_live_status(self) -> Dict[str, Any]:
        """ライブ検証ステータス取得"""
        if not self.is_validating or not self.paper_trading_engine:
            return {"status": "inactive"}

        metrics = self.paper_trading_engine.metrics

        return {
            "status": "active",
            "start_time": self.validation_start_time.isoformat(),
            "current_balance": metrics.current_balance,
            "total_pnl": metrics.total_pnl,
            "total_trades": metrics.total_trades,
            "win_rate": metrics.win_rate,
            "open_positions": len(self.paper_trading_engine.positions),
            "max_drawdown": metrics.max_drawdown,
            "sharpe_ratio": metrics.sharpe_ratio,
        }


def create_real_data_validation_system(
    integration_manager: V433IntegrationManager,
) -> RealDataValidationSystem:
    """リアルデータ検証システムのファクトリ関数"""
    return RealDataValidationSystem(integration_manager)


# 使用例
if __name__ == "__main__":
    from ztb.trading.v433_integration_manager import create_v433_integration_manager

    # V433統合マネージャーの作成
    integration_manager = create_v433_integration_manager("zaif")

    # システム初期化と開始
    if integration_manager.initialize_system() and integration_manager.start_system():
        try:
            # リアルデータ検証システムの作成
            validation_system = create_real_data_validation_system(integration_manager)

            # 検証設定
            config = LiveValidationConfig(
                symbol="btc_jpy",
                validation_period_days=7,
                paper_trading_balance=100000.0,  # 1週間検証
            )

            # ライブ検証開始
            print("Starting live validation...")
            if validation_system.start_live_validation(config):
                print("Live validation started successfully")

                # 検証実行（例: 1時間）
                time.sleep(3600)

                # 検証停止とレポート生成
                print("Stopping validation and generating report...")
                report = validation_system.stop_live_validation()

                print(
                    f"Validation completed: {report['performance_metrics']['total_trades']} trades, "
                    f"Return: {report['performance_metrics']['total_return']:.2%}"
                )

        finally:
            # システム停止
            integration_manager.stop_system()
    else:
        print("Failed to initialize/start V433 system")
