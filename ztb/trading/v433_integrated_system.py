#!/usr/bin/env python3
"""
V433 Phase 3: 統合トレーディングシステム
Phase 2適応学習システムとPhase 3取引実行・リスク管理の統合
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
import threading
import numpy as np
import pandas as pd

from ztb.utils.logging_utils import get_logger
from ztb.trading.trade_execution_engine import TradeExecutionEngine
from ztb.trading.position_manager import PositionManager, PositionSignal
from ztb.trading.risk_overlay import RiskOverlay

# Phase 2適応学習システムのインポート（仮定）
try:
    from ztb.learning.adaptive_sac import AdaptiveSAC
    from ztb.learning.online_optimizer import UnifiedOptimizer
except ImportError:
    # モッククラス（実際の実装では適切なインポート）
    class AdaptiveSAC:
        def get_action(self, state): return {"action": 0.5, "confidence": 0.7}
        def update(self, state, action, reward): pass

    class UnifiedOptimizer:
        def optimize(self, params): return params

logger = get_logger(__name__)

@dataclass
class V433Config:
    """V433統合設定"""
    # Phase 2設定
    phase2_enabled: bool = True
    adaptive_learning_interval: int = 300  # 5分ごと
    state_update_interval: int = 60  # 1分ごと

    # Phase 3設定
    phase3_enabled: bool = True
    signal_generation_interval: int = 60  # 1分ごと
    position_update_interval: int = 30  # 30秒ごと

    # 統合設定
    decision_fusion_method: str = "weighted"  # "weighted", "hierarchical", "ensemble"
    phase2_weight: float = 0.6  # Phase 2の重み
    phase3_weight: float = 0.4  # Phase 3の重み

    # 市場条件適応
    market_regime_detection: bool = True
    regime_adaptation_enabled: bool = True
    volatility_threshold_high: float = 0.05
    volatility_threshold_low: float = 0.02

    # パフォーマンス監視
    performance_monitoring_enabled: bool = True
    performance_report_interval: int = 3600  # 1時間ごと

    # 緊急停止統合
    emergency_stop_propagation: bool = True
    system_health_check_interval: int = 60  # 1分ごと


@dataclass
class SystemState:
    """システム状態"""
    # 市場状態
    market_regime: str = "normal"  # "normal", "high_volatility", "low_volatility", "trending"
    volatility_level: float = 0.0
    trend_strength: float = 0.0

    # Phase 2状態
    phase2_active: bool = False
    phase2_confidence: float = 0.0
    phase2_last_action: Any = None

    # Phase 3状態
    phase3_active: bool = False
    phase3_signals: List[PositionSignal] = field(default_factory=list)
    phase3_risk_status: str = "normal"

    # 統合状態
    system_health: str = "healthy"  # "healthy", "warning", "critical"
    last_decision_timestamp: Optional[datetime] = None
    decision_count: int = 0

    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradingDecision:
    """トレーディング決定"""
    symbol: str
    action: str
    strength: float
    confidence: float
    source: str  # "phase2", "phase3", "integrated"
    reasoning: str
    risk_adjusted: bool = True
    timestamp: datetime = field(default_factory=datetime.now)


class MarketRegimeDetector:
    """市場レジーム検知器"""

    def __init__(self):
        self.price_history: Dict[str, deque] = {}
        self.return_history: Dict[str, deque] = {}

    def update_price(self, symbol: str, price: float):
        """価格を更新"""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=100)  # 100期間
            self.return_history[symbol] = deque(maxlen=99)

        self.price_history[symbol].append(price)

        # リターンを計算
        if len(self.price_history[symbol]) >= 2:
            prices = list(self.price_history[symbol])
            returns = np.diff(prices) / prices[:-1]
            self.return_history[symbol] = deque(returns, maxlen=len(returns))

    def detect_regime(self, symbols: List[str]) -> Tuple[str, float, float]:
        """市場レジームを検知"""
        if not symbols or not all(s in self.return_history for s in symbols):
            return "unknown", 0.0, 0.0

        # 全シンボルのリターンを統合
        all_returns = []
        for symbol in symbols:
            all_returns.extend(list(self.return_history[symbol]))

        if len(all_returns) < 20:
            return "insufficient_data", 0.0, 0.0

        returns = np.array(all_returns)

        # ボラティリティ計算
        volatility = np.std(returns)

        # トレンド強度計算（簡易版）
        trend_strength = abs(np.mean(returns)) / volatility if volatility > 0 else 0.0

        # レジーム判定
        if volatility > 0.05:  # 高ボラティリティ
            regime = "high_volatility"
        elif volatility < 0.02:  # 低ボラティリティ
            regime = "low_volatility"
        elif trend_strength > 0.3:  # 強いトレンド
            regime = "trending"
        else:
            regime = "normal"

        return regime, volatility, trend_strength


class DecisionFusionEngine:
    """決定融合エンジン"""

    def __init__(self, config: V433Config):
        self.config = config
        self.logger = get_logger(__name__)

    def fuse_decisions(self, phase2_decision: Optional[Dict[str, Any]],
                      phase3_signals: List[PositionSignal],
                      market_regime: str,
                      system_state: SystemState) -> List[TradingDecision]:
        """決定を融合"""
        decisions = []

        if self.config.decision_fusion_method == "weighted":
            decisions = self._weighted_fusion(phase2_decision, phase3_signals,
                                            market_regime, system_state)
        elif self.config.decision_fusion_method == "hierarchical":
            decisions = self._hierarchical_fusion(phase2_decision, phase3_signals,
                                                market_regime, system_state)
        elif self.config.decision_fusion_method == "ensemble":
            decisions = self._ensemble_fusion(phase2_decision, phase3_signals,
                                            market_regime, system_state)

        return decisions

    def _weighted_fusion(self, phase2_decision: Optional[Dict[str, Any]],
                        phase3_signals: List[PositionSignal],
                        market_regime: str,
                        system_state: SystemState) -> List[TradingDecision]:
        """重み付き融合"""
        decisions = []

        # Phase 2決定の処理
        if phase2_decision and system_state.phase2_active:
            phase2_weight = self._adjust_weight_by_regime(self.config.phase2_weight, market_regime, "phase2")
            phase2_decision["confidence"] *= phase2_weight

            decision = TradingDecision(
                symbol=phase2_decision.get("symbol", "unknown"),
                action=phase2_decision.get("action", "hold"),
                strength=phase2_decision.get("strength", 0.5),
                confidence=phase2_decision["confidence"],
                source="phase2",
                reasoning=f"Phase 2 decision with {phase2_weight:.2f} weight in {market_regime} regime"
            )
            decisions.append(decision)

        # Phase 3シグナルの処理
        for signal in phase3_signals:
            phase3_weight = self._adjust_weight_by_regime(self.config.phase3_weight, market_regime, "phase3")
            signal.confidence *= phase3_weight

            decision = TradingDecision(
                symbol=signal.symbol,
                action=signal.action,
                strength=signal.strength,
                confidence=signal.confidence,
                source="phase3",
                reasoning=f"Phase 3 signal with {phase3_weight:.2f} weight in {market_regime} regime"
            )
            decisions.append(decision)

        return decisions

    def _hierarchical_fusion(self, phase2_decision: Optional[Dict[str, Any]],
                           phase3_signals: List[PositionSignal],
                           market_regime: str,
                           system_state: SystemState) -> List[TradingDecision]:
        """階層的融合（Phase 2が優先）"""
        decisions = []

        # 高ボラティリティ時はPhase 2を優先
        if market_regime == "high_volatility" and phase2_decision and system_state.phase2_active:
            decision = TradingDecision(
                symbol=phase2_decision.get("symbol", "unknown"),
                action=phase2_decision.get("action", "hold"),
                strength=phase2_decision.get("strength", 0.5),
                confidence=phase2_decision.get("confidence", 0.8),
                source="integrated",
                reasoning="Phase 2 prioritized in high volatility regime"
            )
            decisions.append(decision)
        else:
            # 通常時はPhase 3シグナルを使用
            for signal in phase3_signals:
                decision = TradingDecision(
                    symbol=signal.symbol,
                    action=signal.action,
                    strength=signal.strength,
                    confidence=signal.confidence,
                    source="integrated",
                    reasoning="Phase 3 signals used in normal regime"
                )
                decisions.append(decision)

        return decisions

    def _ensemble_fusion(self, phase2_decision: Optional[Dict[str, Any]],
                        phase3_signals: List[PositionSignal],
                        market_regime: str,
                        system_state: SystemState) -> List[TradingDecision]:
        """アンサンブル融合"""
        decisions = []

        # Phase 2とPhase 3の決定を統合
        if phase2_decision and phase3_signals:
            # 最も一致するPhase 3シグナルを探す
            best_match = None
            best_score = 0.0

            for signal in phase3_signals:
                if signal.symbol == phase2_decision.get("symbol"):
                    # アクションの一致度を計算
                    action_match = self._calculate_action_similarity(
                        phase2_decision.get("action"), signal.action
                    )
                    confidence_avg = (phase2_decision.get("confidence", 0.5) + signal.confidence) / 2
                    score = action_match * confidence_avg

                    if score > best_score:
                        best_score = score
                        best_match = signal

            if best_match:
                decision = TradingDecision(
                    symbol=best_match.symbol,
                    action=best_match.action,
                    strength=(phase2_decision.get("strength", 0.5) + best_match.strength) / 2,
                    confidence=best_score,
                    source="integrated",
                    reasoning=f"Ensemble decision with score {best_score:.2f}"
                )
                decisions.append(decision)

        return decisions

    def _adjust_weight_by_regime(self, base_weight: float, regime: str, source: str) -> float:
        """レジームによる重み調整"""
        if source == "phase2":
            # 高ボラティリティ時はPhase 2を重視
            if regime == "high_volatility":
                return min(base_weight * 1.5, 1.0)
            elif regime == "low_volatility":
                return base_weight * 0.8
        elif source == "phase3":
            # 安定した市場ではPhase 3を重視
            if regime == "normal":
                return min(base_weight * 1.3, 1.0)
            elif regime == "trending":
                return base_weight * 0.9

        return base_weight

    def _calculate_action_similarity(self, action1: str, action2: str) -> float:
        """アクションの類似度を計算"""
        # 簡易版：同じアクションなら1.0、逆なら0.0、中間なら0.5
        if action1 == action2:
            return 1.0
        elif self._are_opposite_actions(action1, action2):
            return 0.0
        else:
            return 0.5

    def _are_opposite_actions(self, action1: str, action2: str) -> bool:
        """アクションが相反するかチェック"""
        opposites = {
            "open_long": "open_short",
            "open_short": "open_long",
            "close_long": "open_long",
            "close_short": "open_short"
        }
        return opposites.get(action1) == action2


class PerformanceMonitor:
    """パフォーマンス監視器"""

    def __init__(self, config: V433Config):
        self.config = config
        self.logger = get_logger(__name__)

        # パフォーマンス履歴
        self.decision_history: List[TradingDecision] = []
        self.outcome_history: List[Dict[str, Any]] = []
        self.metrics_history: List[Dict[str, Any]] = []

    def record_decision(self, decision: TradingDecision):
        """決定を記録"""
        self.decision_history.append(decision)

        # 履歴を制限
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]

    def record_outcome(self, decision: TradingDecision, pnl: float, execution_time: float):
        """結果を記録"""
        outcome = {
            "decision": decision.__dict__,
            "pnl": pnl,
            "execution_time": execution_time,
            "timestamp": datetime.now()
        }
        self.outcome_history.append(outcome)

        # 履歴を制限
        if len(self.outcome_history) > 1000:
            self.outcome_history = self.outcome_history[-1000:]

    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """パフォーマンス指標を計算"""
        if not self.outcome_history:
            return {}

        outcomes = self.outcome_history[-100:]  # 最新100件

        pnls = [o["pnl"] for o in outcomes]
        execution_times = [o["execution_time"] for o in outcomes]

        # 基本指標
        total_pnl = sum(pnls)
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
        avg_pnl = np.mean(pnls)
        pnl_std = np.std(pnls)
        sharpe_ratio = avg_pnl / pnl_std if pnl_std > 0 else 0.0

        # 実行時間指標
        avg_execution_time = np.mean(execution_times)
        max_execution_time = max(execution_times)

        # ソース別パフォーマンス
        source_performance = {}
        for outcome in outcomes:
            source = outcome["decision"]["source"]
            if source not in source_performance:
                source_performance[source] = []
            source_performance[source].append(outcome["pnl"])

        for source, source_pnls in source_performance.items():
            source_performance[source] = {
                "total_pnl": sum(source_pnls),
                "win_rate": sum(1 for p in source_pnls if p > 0) / len(source_pnls),
                "avg_pnl": np.mean(source_pnls)
            }

        return {
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "sharpe_ratio": sharpe_ratio,
            "avg_execution_time": avg_execution_time,
            "max_execution_time": max_execution_time,
            "source_performance": source_performance,
            "sample_size": len(outcomes)
        }


class V433IntegratedSystem:
    """
    V433 Phase 3: 統合トレーディングシステム
    Phase 2適応学習システムとPhase 3取引実行・リスク管理の統合
    """

    def __init__(self, exchange: str = "zaif"):
        self.exchange = exchange
        self.logger = get_logger(__name__)

        # 設定の初期化
        self.config = V433Config()

        # Phase 3コンポーネントの初期化
        self.execution_engine = TradeExecutionEngine(exchange)
        self.position_manager = PositionManager(self.execution_engine, exchange)
        self.risk_overlay = RiskOverlay(self.position_manager)

        # Phase 2コンポーネントの初期化
        self.adaptive_sac = AdaptiveSAC() if self.config.phase2_enabled else None
        self.optimizer = UnifiedOptimizer() if self.config.phase2_enabled else None

        # 統合コンポーネントの初期化
        self.regime_detector = MarketRegimeDetector()
        self.decision_fusion = DecisionFusionEngine(self.config)
        self.performance_monitor = PerformanceMonitor(self.config)

        # システム状態
        self.system_state = SystemState()
        self.last_phase2_update = datetime.now()
        self.last_signal_generation = datetime.now()

        # モニタリング
        self.monitoring_thread = None
        self.decision_thread = None
        self.is_running = False

        # 価格データ
        self.current_prices: Dict[str, float] = {}
        self.active_symbols = ["btc_jpy", "eth_jpy", "xrp_jpy"]  # デフォルトシンボル

    def start_system(self):
        """統合システムを開始"""
        if self.is_running:
            return

        self.is_running = True

        # Phase 3システム開始
        self.execution_engine.start_execution()
        self.position_manager.start_management()
        self.risk_overlay.start_overlay()

        # モニタリングスレッド開始
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        # 決定生成スレッド開始
        self.decision_thread = threading.Thread(target=self._decision_loop, daemon=True)
        self.decision_thread.start()

        self.logger.info("V433 Integrated System started")

    def stop_system(self):
        """統合システムを停止"""
        self.is_running = False

        # スレッド停止
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)

        if self.decision_thread and self.decision_thread.is_alive():
            self.decision_thread.join(timeout=5)

        # Phase 3システム停止
        self.risk_overlay.stop_overlay()
        self.position_manager.stop_management()
        self.execution_engine.stop_execution()

        self.logger.info("V433 Integrated System stopped")

    def update_market_data(self, symbol: str, price: float, volume: float = 0.0):
        """市場データを更新"""
        self.current_prices[symbol] = price

        # レジーム検知器に価格を更新
        self.regime_detector.update_price(symbol, price)

        # リスクオーバーレイに価格を更新
        self.risk_overlay.update_price(symbol, price)

        # Phase 2に状態を更新（適宜）
        if self.config.phase2_enabled and self.adaptive_sac:
            # 市場状態の構築
            state = self._construct_market_state()
            # Phase 2の更新（実際の実装による）

    def _construct_market_state(self) -> Dict[str, Any]:
        """市場状態を構築"""
        return {
            "prices": self.current_prices.copy(),
            "regime": self.system_state.market_regime,
            "volatility": self.system_state.volatility_level,
            "positions": self.position_manager.portfolio_state.positions.copy(),
            "risk_metrics": self.risk_overlay.risk_metrics.__dict__ if self.risk_overlay.risk_metrics else {}
        }

    def _monitoring_loop(self):
        """モニタリングループ"""
        while self.is_running:
            try:
                # システム状態更新
                self._update_system_state()

                # ヘルスチェック
                self._perform_health_check()

                # パフォーマンスレポート
                if self.config.performance_monitoring_enabled:
                    self._generate_performance_report()

                time.sleep(self.config.system_health_check_interval)

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(30)

    def _decision_loop(self):
        """決定生成ループ"""
        while self.is_running:
            try:
                current_time = datetime.now()

                # Phase 2決定生成
                phase2_decision = None
                if (self.config.phase2_enabled and
                    (current_time - self.last_phase2_update).seconds >= self.config.adaptive_learning_interval):
                    phase2_decision = self._generate_phase2_decision()
                    self.last_phase2_update = current_time

                # Phase 3シグナル生成
                phase3_signals = []
                if (self.config.phase3_enabled and
                    (current_time - self.last_signal_generation).seconds >= self.config.signal_generation_interval):
                    phase3_signals = self._generate_phase3_signals()
                    self.last_signal_generation = current_time

                # 決定融合
                if phase2_decision or phase3_signals:
                    decisions = self.decision_fusion.fuse_decisions(
                        phase2_decision, phase3_signals,
                        self.system_state.market_regime, self.system_state
                    )

                    # 決定実行
                    for decision in decisions:
                        self._execute_decision(decision)

                time.sleep(10)  # 10秒間隔

            except Exception as e:
                self.logger.error(f"Decision loop error: {e}")
                time.sleep(30)

    def _generate_phase2_decision(self) -> Optional[Dict[str, Any]]:
        """Phase 2決定を生成"""
        if not self.adaptive_sac:
            return None

        try:
            # 市場状態の構築
            state = self._construct_market_state()

            # Phase 2からアクションを取得
            action = self.adaptive_sac.get_action(state)

            decision = {
                "symbol": "btc_jpy",  # デフォルトシンボル（実際は動的）
                "action": self._convert_phase2_action(action["action"]),
                "strength": abs(action["action"]),
                "confidence": action["confidence"]
            }

            self.system_state.phase2_active = True
            self.system_state.phase2_confidence = action["confidence"]
            self.system_state.phase2_last_action = action

            return decision

        except Exception as e:
            self.logger.error(f"Phase 2 decision generation failed: {e}")
            self.system_state.phase2_active = False
            return None

    def _generate_phase3_signals(self) -> List[PositionSignal]:
        """Phase 3シグナルを生成"""
        signals = []

        try:
            # 簡易シグナル生成（実際の実装ではより複雑なロジック）
            for symbol in self.active_symbols:
                if symbol not in self.current_prices:
                    continue

                # ランダムシグナル（実際はテクニカル分析やMLベース）
                if np.random.random() > 0.95:  # 5%の確率でシグナル
                    action = np.random.choice(["open_long", "open_short", "close_long", "close_short"])
                    strength = np.random.uniform(0.3, 0.8)
                    confidence = np.random.uniform(0.4, 0.9)

                    signal = PositionSignal(
                        symbol=symbol,
                        action=action,
                        strength=strength,
                        target_quantity=0.001,  # 最小数量
                        confidence=confidence,
                        reason="technical_analysis"
                    )
                    signals.append(signal)

            self.system_state.phase3_active = len(signals) > 0
            self.system_state.phase3_signals = signals

            return signals

        except Exception as e:
            self.logger.error(f"Phase 3 signal generation failed: {e}")
            return []

    def _execute_decision(self, decision: TradingDecision):
        """決定を実行"""
        try:
            start_time = time.time()

            # パフォーマンス監視に決定を記録
            self.performance_monitor.record_decision(decision)

            # Phase 3ポジション管理にシグナルを送信
            if decision.source in ["phase3", "integrated"]:
                signal = PositionSignal(
                    symbol=decision.symbol,
                    action=decision.action,
                    strength=decision.strength,
                    target_quantity=0.001,  # 実際は動的計算
                    confidence=decision.confidence,
                    reason=f"Integrated decision: {decision.reasoning}"
                )

                asyncio.create_task(self.position_manager.submit_signal(signal))

            # 実行時間を記録
            execution_time = time.time() - start_time

            # 仮定のPnLで結果を記録（実際は実際の取引結果）
            pnl = np.random.normal(0, 100)  # ランダムPnL
            self.performance_monitor.record_outcome(decision, pnl, execution_time)

            # システム状態更新
            self.system_state.last_decision_timestamp = datetime.now()
            self.system_state.decision_count += 1

            self.logger.info(f"Executed decision: {decision.symbol} {decision.action} "
                           f"(confidence: {decision.confidence:.2f})")

        except Exception as e:
            self.logger.error(f"Decision execution failed: {e}")

    def _update_system_state(self):
        """システム状態を更新"""
        try:
            # 市場レジーム検知
            regime, volatility, trend = self.regime_detector.detect_regime(self.active_symbols)
            self.system_state.market_regime = regime
            self.system_state.volatility_level = volatility
            self.system_state.trend_strength = trend

            # Phase 3リスク状態
            emergency_status = self.risk_overlay.emergency_stop.get_emergency_status()
            self.system_state.phase3_risk_status = "emergency" if emergency_status["triggered"] else "normal"

            # システムヘルス判定
            health_issues = []

            if emergency_status["triggered"]:
                health_issues.append("emergency_stop_triggered")

            if self.system_state.phase2_active and self.system_state.phase2_confidence < 0.3:
                health_issues.append("low_phase2_confidence")

            if len(self.position_manager.portfolio_state.positions) > 10:
                health_issues.append("too_many_positions")

            self.system_state.system_health = "critical" if len(health_issues) > 1 else "warning" if health_issues else "healthy"

            self.system_state.timestamp = datetime.now()

        except Exception as e:
            self.logger.error(f"System state update failed: {e}")

    def _perform_health_check(self):
        """ヘルスチェックを実行"""
        health_status = {
            "system_health": self.system_state.system_health,
            "phase2_active": self.system_state.phase2_active,
            "phase3_active": self.system_state.phase3_active,
            "emergency_stop": self.risk_overlay.emergency_stop.get_emergency_status(),
            "position_count": len(self.position_manager.portfolio_state.positions),
            "total_pnl": self.position_manager.portfolio_state.total_pnl,
            "timestamp": datetime.now()
        }

        # ヘルスレポート
        if self.system_state.system_health != "healthy":
            self.logger.warning(f"System health check: {health_status}")
        else:
            self.logger.debug(f"System health: OK")

    def _generate_performance_report(self):
        """パフォーマンスレポートを生成"""
        try:
            metrics = self.performance_monitor.calculate_performance_metrics()

            if metrics:
                self.logger.info(f"Performance Report: PnL={metrics['total_pnl']:.2f}, "
                               f"Win Rate={metrics['win_rate']:.2%}, "
                               f"Sharpe={metrics['sharpe_ratio']:.2f}")

                # ソース別パフォーマンス
                for source, perf in metrics.get("source_performance", {}).items():
                    self.logger.info(f"  {source}: PnL={perf['total_pnl']:.2f}, "
                                   f"Win Rate={perf['win_rate']:.2%}")

        except Exception as e:
            self.logger.error(f"Performance report generation failed: {e}")

    def _convert_phase2_action(self, action_value: float) -> str:
        """Phase 2アクションを変換"""
        # Phase 2のアクション値を取引アクションに変換
        if action_value > 0.3:
            return "open_long"
        elif action_value < -0.3:
            return "open_short"
        else:
            return "hold"

    def get_system_status(self) -> Dict[str, Any]:
        """システム状態を取得"""
        return {
            "system_state": self.system_state.__dict__,
            "portfolio_status": self.position_manager.get_portfolio_summary(),
            "risk_report": self.risk_overlay.get_risk_report(),
            "performance_metrics": self.performance_monitor.calculate_performance_metrics(),
            "market_data": {
                "current_prices": self.current_prices.copy(),
                "active_symbols": self.active_symbols.copy()
            },
            "component_status": {
                "phase2_enabled": self.config.phase2_enabled,
                "phase3_enabled": self.config.phase3_enabled,
                "execution_engine_running": self.execution_engine.is_running,
                "position_manager_running": self.position_manager.is_running,
                "risk_overlay_running": self.is_running
            }
        }


def create_v433_system(exchange: str = "zaif") -> V433IntegratedSystem:
    """V433統合システムのファクトリ関数"""
    return V433IntegratedSystem(exchange)


# 使用例
if __name__ == "__main__":
    # V433統合システムの作成
    v433_system = create_v433_system("zaif")

    # システム開始
    v433_system.start_system()

    try:
        # サンプル市場データ更新
        v433_system.update_market_data("btc_jpy", 5000000, 100)
        v433_system.update_market_data("eth_jpy", 300000, 50)
        v433_system.update_market_data("xrp_jpy", 100, 10000)

        # システム状態確認
        status = v433_system.get_system_status()
        print(f"System status: {status['system_state']['system_health']}")
        print(f"Portfolio PnL: {status['portfolio_status']['portfolio_state']['total_pnl']:.2f}")

        # しばらく実行
        time.sleep(60)

        # 最終状態確認
        final_status = v433_system.get_system_status()
        print(f"Final performance: {final_status['performance_metrics']}")

    finally:
        # システム停止
        v433_system.stop_system()