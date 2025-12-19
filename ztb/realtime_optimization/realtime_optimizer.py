# ztb/realtime_optimization/realtime_optimizer.py

"""
リアルタイム最適化エンジン

このモジュールは、市場条件変化への適応と継続的な
パラメータ再最適化を実現します。
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """最適化結果"""
    timestamp: datetime
    parameters: Dict[str, Any]
    performance_score: float
    market_regime: str
    confidence: float


@dataclass
class MarketCondition:
    """市場条件"""
    volatility: float
    trend_strength: float
    volume: float
    regime: str
    timestamp: datetime


class RealtimeOptimizer:
    """
    リアルタイム最適化エンジン

    継続的なパラメータ再最適化と市場適応を実現：
    - 市場レジーム変化検知
    - 動的リスク調整
    - パフォーマンスベースの戦略切り替え
    """

    def __init__(self,
                 base_optimizer: Any,
                 market_analyzer: Callable[[], MarketCondition],
                 optimization_interval: int = 3600,  # 1時間
                 performance_window: int = 24):  # 24時間
        """
        初期化

        Args:
            base_optimizer: ベースとなる最適化システム
            market_analyzer: 市場条件分析関数
            optimization_interval: 最適化間隔（秒）
            performance_window: パフォーマンス評価期間（時間）
        """
        self.base_optimizer = base_optimizer
        self.market_analyzer = market_analyzer
        self.optimization_interval = optimization_interval
        self.performance_window = performance_window

        self.current_parameters: Dict[str, Any] = {}
        self.optimization_history: List[OptimizationResult] = []
        self.is_running = False
        self.optimization_thread: Optional[threading.Thread] = None

        # パフォーマンス追跡
        self.performance_scores: List[float] = []
        self.market_conditions: List[MarketCondition] = []

        logger.info("RealtimeOptimizer initialized")

    def start_optimization(self):
        """最適化開始"""
        if self.is_running:
            logger.warning("Optimization is already running")
            return

        self.is_running = True
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop
        )
        self.optimization_thread.daemon = True
        self.optimization_thread.start()

        logger.info("Realtime optimization started")

    def stop_optimization(self):
        """最適化停止"""
        if not self.is_running:
            logger.info("Optimization is not running")
            return

        self.is_running = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=10)

        logger.info("Realtime optimization stopped")

    def _optimization_loop(self):
        """最適化メインループ"""
        logger.info("Optimization loop started")

        while self.is_running:
            try:
                self._execute_optimization_cycle()
                time.sleep(self.optimization_interval)

            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(self.optimization_interval)

        logger.info("Optimization loop ended")

    def _execute_optimization_cycle(self):
        """1サイクルの最適化実行"""
        try:
            # 1. 市場条件分析
            market_condition = self.market_analyzer()

            # 2. パフォーマンス評価
            performance_score = self._evaluate_performance()

            # 3. 最適化が必要か判定
            if self._should_optimize(market_condition, performance_score):
                # 4. パラメータ最適化実行
                new_parameters = self._run_optimization(market_condition)

                # 5. パラメータ適用
                self._apply_parameters(new_parameters, market_condition, performance_score)

                logger.info(f"Parameters optimized: {new_parameters}")

            # 6. 履歴更新
            self.market_conditions.append(market_condition)
            self._cleanup_old_data()

        except Exception as e:
            logger.error(f"Error in optimization cycle: {e}")

    def _evaluate_performance(self) -> float:
        """
        パフォーマンス評価

        Returns:
            float: パフォーマンススコア
        """
        # TODO: 実際のパフォーマンス指標計算
        # ここではモック実装
        if not self.performance_scores:
            return 0.5  # デフォルトスコア

        # 最近のパフォーマンス平均
        recent_scores = self.performance_scores[-10:]  # 直近10件
        return sum(recent_scores) / len(recent_scores)

    def _should_optimize(self,
                        market_condition: MarketCondition,
                        performance_score: float) -> bool:
        """
        最適化必要性判定

        Args:
            market_condition: 市場条件
            performance_score: パフォーマンススコア

        Returns:
            bool: 最適化が必要か
        """
        # パフォーマンスが閾値以下の場合
        if performance_score < 0.4:
            return True

        # 市場レジームが変化した場合
        if self.optimization_history:
            last_regime = self.optimization_history[-1].market_regime
            if last_regime != market_condition.regime:
                return True

        # 定期最適化（24時間ごと）
        if not self.optimization_history:
            return False  # 初期状態では最適化しない

        last_optimization = self.optimization_history[-1].timestamp
        if datetime.now() - last_optimization > timedelta(hours=self.performance_window):
            return True

        return False

    def _run_optimization(self, market_condition: MarketCondition) -> Dict[str, Any]:
        """
        最適化実行

        Args:
            market_condition: 市場条件

        Returns:
            Dict[str, Any]: 新しいパラメータ
        """
        try:
            # 市場条件に応じた最適化設定
            optimization_config = self._create_optimization_config(market_condition)

            # ベース最適化システムを使用
            result = self.base_optimizer.run_integrated_optimization(
                market_data=None,  # TODO: リアルタイムデータ
                base_strategy_func=None,  # TODO: 戦略関数
                config=optimization_config
            )

            # 最適パラメータ抽出
            new_parameters = self._extract_optimal_parameters(result)

            return new_parameters

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return self.current_parameters  # フォールバック

    def _create_optimization_config(self, market_condition: MarketCondition) -> Any:
        """
        最適化設定作成

        Args:
            market_condition: 市場条件

        Returns:
            Any: 最適化設定
        """
        # TODO: 市場条件に応じた設定作成
        # ここではモック実装
        from ztb.analysis.integrated_optimizer import IntegratedOptimizationConfig

        return IntegratedOptimizationConfig(
            train_days=30,  # 短期間で最適化
            test_days=7,
            step_days=7,
            kelly_risk_tolerance=self._get_kelly_tolerance(market_condition),
            risk_management_mode=self._get_risk_mode(market_condition),
            adaptive_thresholds_enabled=True
        )

    def _get_kelly_tolerance(self, market_condition: MarketCondition) -> str:
        """Kelly許容度決定"""
        if market_condition.volatility > 0.8:
            return "quarter"  # 高ボラティリティ時は保守的に
        elif market_condition.volatility > 0.5:
            return "half"
        else:
            return "full"

    def _get_risk_mode(self, market_condition: MarketCondition) -> str:
        """リスク管理モード決定"""
        if market_condition.regime == "high_volatility":
            return "conservative"
        elif market_condition.regime == "trending":
            return "moderate"
        else:
            return "dynamic"

    def _extract_optimal_parameters(self, optimization_result: Any) -> Dict[str, Any]:
        """
        最適パラメータ抽出

        Args:
            optimization_result: 最適化結果

        Returns:
            Dict[str, Any]: 最適パラメータ
        """
        # TODO: 最適化結果からのパラメータ抽出
        # ここではモック実装
        return {
            'kelly_fraction': 0.5,
            'atr_multiplier': 2.0,
            'confidence_threshold': 0.7,
            'max_positions': 3
        }

    def _apply_parameters(self,
                         parameters: Dict[str, Any],
                         market_condition: MarketCondition,
                         performance_score: float):
        """
        パラメータ適用

        Args:
            parameters: 新しいパラメータ
            market_condition: 市場条件
            performance_score: パフォーマンススコア
        """
        self.current_parameters = parameters

        # 最適化結果記録
        result = OptimizationResult(
            timestamp=datetime.now(),
            parameters=parameters,
            performance_score=performance_score,
            market_regime=market_condition.regime,
            confidence=0.8  # TODO: 計算
        )

        self.optimization_history.append(result)

        logger.info(f"Applied new parameters: {parameters}")

    def _cleanup_old_data(self):
        """古いデータクリーンアップ"""
        # 30日以上前のデータを削除
        cutoff_date = datetime.now() - timedelta(days=30)

        self.optimization_history = [
            h for h in self.optimization_history
            if h.timestamp > cutoff_date
        ]

        self.market_conditions = [
            c for c in self.market_conditions
            if c.timestamp > cutoff_date
        ]

    def get_current_parameters(self) -> Dict[str, Any]:
        """
        現在のパラメータ取得

        Returns:
            Dict[str, Any]: 現在のパラメータ
        """
        return self.current_parameters.copy()

    def get_optimization_history(self) -> List[OptimizationResult]:
        """
        最適化履歴取得

        Returns:
            List[OptimizationResult]: 最適化履歴
        """
        return self.optimization_history.copy()

    def get_status(self) -> Dict[str, Any]:
        """
        ステータス取得

        Returns:
            Dict[str, Any]: 現在のステータス
        """
        return {
            'is_running': self.is_running,
            'current_parameters': self.current_parameters,
            'last_optimization': self.optimization_history[-1] if self.optimization_history else None,
            'total_optimizations': len(self.optimization_history),
            'current_market_condition': self.market_conditions[-1] if self.market_conditions else None
        }
