# ztb/realtime_optimization/adaptive_learning_system.py

"""
適応型学習システム

このモジュールは、継続的な学習と適応を実現します。
市場変化への動的対応と戦略の進化を担います。
"""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LearningExperience:
    """学習経験"""

    timestamp: datetime
    market_condition: Dict[str, Any]
    action_taken: str
    reward: float
    next_state: Dict[str, Any]
    strategy_used: str


@dataclass
class StrategyPerformance:
    """戦略パフォーマンス"""

    strategy_name: str
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    last_updated: datetime


class AdaptiveLearningSystem:
    """
    適応型学習システム

    継続的な学習と適応を実現：
    - 経験ベースの学習
    - 戦略パフォーマンス評価
    - 動的戦略選択
    - 市場適応
    """

    def __init__(
        self,
        learning_interval: int = 1800,  # 30分
        experience_window: int = 1000,  # 1000件の経験
        strategy_evaluation_period: int = 24,
    ):  # 24時間
        """
        初期化

        Args:
            learning_interval: 学習間隔（秒）
            experience_window: 経験保持数
            strategy_evaluation_period: 戦略評価期間（時間）
        """
        self.learning_interval = learning_interval
        self.experience_window = experience_window
        self.strategy_evaluation_period = strategy_evaluation_period

        self.learning_experiences: deque[LearningExperience] = deque(maxlen=experience_window)
        self.strategy_performances: Dict[str, StrategyPerformance] = {}
        self.current_best_strategy: Optional[str] = None

        self.is_learning = False
        self.learning_thread: Optional[threading.Thread] = None

        # 学習パラメータ
        self.learning_rate = 0.01
        self.discount_factor = 0.95
        self.exploration_rate = 0.1

        logger.info("AdaptiveLearningSystem initialized")

    def start_learning(self):
        """学習開始"""
        if self.is_learning:
            logger.warning("Learning is already running")
            return

        self.is_learning = True
        self.learning_thread = threading.Thread(target=self._learning_loop)
        self.learning_thread.daemon = True
        self.learning_thread.start()

        logger.info("Adaptive learning started")

    def stop_learning(self):
        """学習停止"""
        if not self.is_learning:
            logger.info("Learning is not running")
            return

        self.is_learning = False
        if self.learning_thread:
            self.learning_thread.join(timeout=10)

        logger.info("Adaptive learning stopped")

    def _learning_loop(self):
        """学習メインループ"""
        logger.info("Learning loop started")

        while self.is_learning:
            try:
                self._execute_learning_cycle()
                time.sleep(self.learning_interval)

            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                time.sleep(self.learning_interval)

        logger.info("Learning loop ended")

    def _execute_learning_cycle(self):
        """1サイクルの学習実行"""
        try:
            # 1. 戦略パフォーマンス評価
            self._evaluate_strategy_performances()

            # 2. 最適戦略選択
            self._select_best_strategy()

            # 3. 学習モデル更新
            self._update_learning_model()

            # 4. 経験データクリーンアップ
            self._cleanup_experiences()

        except Exception as e:
            logger.error(f"Error in learning cycle: {e}")

    def add_experience(
        self,
        market_condition: Dict[str, Any],
        action_taken: str,
        reward: float,
        next_state: Dict[str, Any],
        strategy_used: str,
    ):
        """
        学習経験追加

        Args:
            market_condition: 市場条件
            action_taken: 実行アクション
            reward: 報酬
            next_state: 次の状態
            strategy_used: 使用戦略
        """
        experience = LearningExperience(
            timestamp=datetime.now(),
            market_condition=market_condition,
            action_taken=action_taken,
            reward=reward,
            next_state=next_state,
            strategy_used=strategy_used,
        )

        self.learning_experiences.append(experience)

    def _evaluate_strategy_performances(self):
        """戦略パフォーマンス評価"""
        # 各戦略の経験を集計
        strategy_experiences = {}
        for exp in self.learning_experiences:
            if exp.strategy_used not in strategy_experiences:
                strategy_experiences[exp.strategy_used] = []
            strategy_experiences[exp.strategy_used].append(exp)

        # パフォーマンス計算
        for strategy_name, experiences in strategy_experiences.items():
            if not experiences:
                continue

            performance = self._calculate_strategy_performance(
                strategy_name, experiences
            )
            self.strategy_performances[strategy_name] = performance

    def _calculate_strategy_performance(
        self, strategy_name: str, experiences: List[LearningExperience]
    ) -> StrategyPerformance:
        """
        戦略パフォーマンス計算

        Args:
            strategy_name: 戦略名
            experiences: 経験リスト

        Returns:
            StrategyPerformance: パフォーマンス指標
        """
        if not experiences:
            return StrategyPerformance(
                strategy_name=strategy_name,
                total_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                last_updated=datetime.now(),
            )

        # 勝率計算
        wins = sum(1 for exp in experiences if exp.reward > 0)
        total_trades = len(experiences)
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        # プロフィットファクター
        profits = [exp.reward for exp in experiences if exp.reward > 0]
        losses = [abs(exp.reward) for exp in experiences if exp.reward < 0]

        total_profit = sum(profits)
        total_loss = sum(losses)
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        # 最大ドローダウン（簡易計算）
        cumulative = np.cumsum([exp.reward for exp in experiences])
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

        # シャープレシオ（簡易計算）
        returns = np.array([exp.reward for exp in experiences])
        from ztb.metrics.metrics import sharpe_ratio as calc_sharpe_ratio

        sharpe_ratio = calc_sharpe_ratio(returns)

        return StrategyPerformance(
            strategy_name=strategy_name,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            last_updated=datetime.now(),
        )

    def _select_best_strategy(self):
        """最適戦略選択"""
        if not self.strategy_performances:
            return

        # 複合スコアで評価
        best_strategy = None
        best_score = -float("inf")

        for strategy_name, performance in self.strategy_performances.items():
            # 重み付きスコア計算
            score = (
                performance.win_rate * 0.3
                + min(performance.profit_factor / 3.0, 1.0) * 0.3
                + (1.0 - min(performance.max_drawdown / 0.2, 1.0))
                * 0.2  # 3.0以上を1.0に正規化
                + max(performance.sharpe_ratio / 2.0, 0.0)
                * 0.2  # 20%ドローダウンを1.0に正規化  # 2.0以上を1.0に正規化
            )

            if score > best_score:
                best_score = score
                best_strategy = strategy_name

        if best_strategy != self.current_best_strategy:
            logger.info(
                f"Best strategy changed from {self.current_best_strategy} to {best_strategy}"
            )
            self.current_best_strategy = best_strategy

    def _update_learning_model(self):
        """学習モデル更新"""
        # Q学習ベースの更新（簡易実装）
        if len(self.learning_experiences) < 2:
            return

        # 最新の経験を取得
        recent_experiences = self.learning_experiences[-10:]  # 直近10件

        for i in range(len(recent_experiences) - 1):
            current_exp = recent_experiences[i]
            next_exp = recent_experiences[i + 1]

            # Q値更新（簡易版）
            # 実際の実装ではより洗練された学習アルゴリズムを使用
            reward = current_exp.reward
            next_max_q = self._estimate_q_value(next_exp)

            # 学習率に基づく更新
            current_q = self._estimate_q_value(current_exp)
            new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * next_max_q - current_q
            )

            # ここではログに記録するだけ（実際のQテーブル更新は別途実装）
            logger.debug(
                f"Updated Q-value for {current_exp.action_taken}: {current_q} -> {new_q}"
            )

    def _estimate_q_value(self, experience: LearningExperience) -> float:
        """
        Q値推定（簡易版）

        Args:
            experience: 学習経験

        Returns:
            float: 推定Q値
        """
        # 報酬ベースの簡易推定
        base_value = experience.reward

        # 市場条件による調整
        market_multiplier = 1.0
        if experience.market_condition.get("volatility", 0.5) > 0.7:
            market_multiplier = 0.8  # 高ボラティリティ時は保守的に

        return base_value * market_multiplier

    def _cleanup_experiences(self):
        """経験データクリーンアップ"""
        # 7日以上前のデータを削除
        cutoff_date = datetime.now() - timedelta(days=7)

        self.learning_experiences = deque(
            (exp for exp in self.learning_experiences if exp.timestamp > cutoff_date),
            maxlen=self.experience_window,
        )

    def get_best_strategy(self) -> Optional[str]:
        """
        最適戦略取得

        Returns:
            Optional[str]: 最適戦略名
        """
        return self.current_best_strategy

    def get_strategy_performances(self) -> Dict[str, StrategyPerformance]:
        """
        戦略パフォーマンス取得

        Returns:
            Dict[str, StrategyPerformance]: 全戦略のパフォーマンス
        """
        return self.strategy_performances.copy()

    def get_learning_status(self) -> Dict[str, Any]:
        """
        学習ステータス取得

        Returns:
            Dict[str, Any]: 学習ステータス
        """
        return {
            "is_learning": self.is_learning,
            "total_experiences": len(self.learning_experiences),
            "current_best_strategy": self.current_best_strategy,
            "strategy_count": len(self.strategy_performances),
            "last_learning_cycle": datetime.now(),  # 簡易的に現在時刻
        }

    def recommend_action(
        self, market_condition: Dict[str, Any], available_actions: List[str]
    ) -> str:
        """
        アクション推奨

        Args:
            market_condition: 市場条件
            available_actions: 利用可能アクション

        Returns:
            str: 推奨アクション
        """
        if not available_actions:
            return "hold"

        # 探索 vs 活用
        if np.random.random() < self.exploration_rate:
            # 探索：ランダム選択
            return np.random.choice(available_actions)

        # 活用：最適戦略に基づく選択
        if self.current_best_strategy:
            # 戦略固有の推奨ロジック（ここでは簡易実装）
            return self._get_strategy_specific_action(
                self.current_best_strategy, market_condition, available_actions
            )

        # デフォルト：リスクベースの選択
        return self._get_risk_based_action(market_condition, available_actions)

    def _get_strategy_specific_action(
        self,
        strategy: str,
        market_condition: Dict[str, Any],
        available_actions: List[str],
    ) -> str:
        """
        戦略固有アクション取得

        Args:
            strategy: 戦略名
            market_condition: 市場条件
            available_actions: 利用可能アクション

        Returns:
            str: アクション
        """
        # TODO: 各戦略固有のロジック実装
        # ここでは簡易実装
        volatility = market_condition.get("volatility", 0.5)

        if strategy == "conservative":
            return "hold" if volatility > 0.7 else "buy"
        elif strategy == "aggressive":
            return "buy" if volatility < 0.8 else "hold"
        else:
            return available_actions[0] if available_actions else "hold"

    def _get_risk_based_action(
        self, market_condition: Dict[str, Any], available_actions: List[str]
    ) -> str:
        """
        リスクベースアクション取得

        Args:
            market_condition: 市場条件
            available_actions: 利用可能アクション

        Returns:
            str: アクション
        """
        volatility = market_condition.get("volatility", 0.5)
        trend = market_condition.get("trend", 0.0)

        if volatility > 0.8:
            return "hold"
        elif trend > 0.2 and "buy" in available_actions:
            return "buy"
        elif trend < -0.2 and "sell" in available_actions:
            return "sell"
        else:
            return "hold"
