"""
Phase 3-2: パラメータ最適化 - Kelly基準ポジションサイズ最適化

Kelly基準に基づく動的ポジションサイズ決定システムを実装します。
過去のトレードパフォーマンスから最適なリスク割合を計算します。
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ztb.utils.performance_profiler import PerformanceProfiler


@dataclass
class KellyParameters:
    """Kelly基準計算パラメータ"""

    win_rate: float  # 勝率
    win_loss_ratio: float  # 平均勝ち額/平均負け額
    total_trades: int  # 総トレード数
    confidence_interval: float = 0.95  # 信頼区間

    @property
    def kelly_fraction(self) -> float:
        """Kelly分数を計算"""
        if self.win_rate <= 0 or self.win_rate >= 1:
            return 0.0

        # Kelly公式: K = (bp - q) / b
        # ここで b = win_loss_ratio, p = win_rate, q = 1-p
        b = self.win_loss_ratio
        p = self.win_rate
        q = 1 - p

        if b <= 0:
            return 0.0

        kelly = (b * p - q) / b

        # 負のKelly分数は0にクリップ
        return max(0.0, kelly)

    @property
    def half_kelly_fraction(self) -> float:
        """ハーフKelly分数（保守的）"""
        return self.kelly_fraction * 0.5

    @property
    def quarter_kelly_fraction(self) -> float:
        """クォーターKelly分数（超保守的）"""
        return self.kelly_fraction * 0.25

    def get_position_size_fraction(
        self, risk_tolerance: str = "half", max_fraction: float = 0.1
    ) -> float:
        """
        リスク許容度に基づくポジションサイズ割合を取得

        Args:
            risk_tolerance: リスク許容度 ("full", "half", "quarter")
            max_fraction: 最大ポジションサイズ割合

        Returns:
            ポジションサイズ割合
        """
        if risk_tolerance == "full":
            fraction = self.kelly_fraction
        elif risk_tolerance == "half":
            fraction = self.half_kelly_fraction
        elif risk_tolerance == "quarter":
            fraction = self.quarter_kelly_fraction
        else:
            fraction = self.half_kelly_fraction

        # 最大割合でクリップ
        return min(fraction, max_fraction)


@dataclass
class PositionSizeDecision:
    """ポジションサイズ決定結果"""

    kelly_params: KellyParameters
    position_size_fraction: float
    risk_amount: float
    confidence_score: float
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "kelly_fraction": self.kelly_params.kelly_fraction,
            "position_size_fraction": self.position_size_fraction,
            "risk_amount": self.risk_amount,
            "confidence_score": self.confidence_score,
            "reasoning": self.reasoning,
            "win_rate": self.kelly_params.win_rate,
            "win_loss_ratio": self.kelly_params.win_loss_ratio,
            "total_trades": self.kelly_params.total_trades,
        }


class KellyPositionSizer:
    """Kelly基準ポジションサイザー"""

    def __init__(self, min_trades: int = 10, max_lookback_trades: int = 100):
        """
        Args:
            min_trades: Kelly計算の最小トレード数
            max_lookback_trades: 最大遡及トレード数
        """
        self.min_trades = min_trades
        self.max_lookback_trades = max_lookback_trades
        self.profiler = PerformanceProfiler()
        self.logger = logging.getLogger(__name__)

    def calculate_kelly_parameters(
        self, trades: List[Dict[str, Any]], current_balance: float
    ) -> Optional[KellyParameters]:
        """
        トレード履歴からKellyパラメータを計算

        Args:
            trades: トレード履歴 [{'pnl': float, 'entry_price': float, ...}]
            current_balance: 現在の残高

        Returns:
            Kellyパラメータ（計算できない場合はNone）
        """
        if len(trades) < self.min_trades:
            self.logger.warning(
                f"Kelly計算に十分なトレードがありません: {len(trades)} < {self.min_trades}"
            )
            return None

        # 最新のトレードを使用
        recent_trades = trades[-self.max_lookback_trades :]

        # P&Lの計算
        pnls = [trade.get("pnl", 0) for trade in recent_trades]

        # 勝ちトレードと負けトレードを分離
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]

        if not winning_trades or not losing_trades:
            self.logger.warning("勝ちトレードまたは負けトレードがありません")
            return None

        # 勝率
        win_rate = len(winning_trades) / len(pnls)

        # 平均勝ち額と平均負け額
        avg_win = np.mean(winning_trades)
        avg_loss = abs(np.mean(losing_trades))  # 負け額は絶対値

        if avg_loss == 0:
            self.logger.warning("平均負け額が0です")
            return None

        # 勝ち負け比率
        win_loss_ratio = avg_win / avg_loss

        return KellyParameters(
            win_rate=win_rate, win_loss_ratio=win_loss_ratio, total_trades=len(pnls)
        )

    def calculate_dynamic_position_size(
        self,
        trades: List[Dict[str, Any]],
        current_balance: float,
        risk_tolerance: str = "half",
        max_risk_fraction: float = 0.02,  # 1トレードあたり最大2%リスク
        volatility_adjustment: bool = True,
    ) -> PositionSizeDecision:
        """
        動的ポジションサイズを計算

        Args:
            trades: トレード履歴
            current_balance: 現在の残高
            risk_tolerance: リスク許容度
            max_risk_fraction: 最大リスク割合
            volatility_adjustment: ボラティリティ調整を行うか

        Returns:
            ポジションサイズ決定結果
        """
        # Kellyパラメータ計算
        kelly_params = self.calculate_kelly_parameters(trades, current_balance)

        if kelly_params is None:
            # デフォルト値を使用
            position_size_fraction = 0.01  # 1%
            risk_amount = current_balance * position_size_fraction
            confidence_score = 0.5

            reasoning = f"Kelly計算不可: トレード数={len(trades)}。デフォルト1%を使用"

            return PositionSizeDecision(
                kelly_params=KellyParameters(0.5, 1.0, len(trades)),
                position_size_fraction=position_size_fraction,
                risk_amount=risk_amount,
                confidence_score=confidence_score,
                reasoning=reasoning,
            )

        # ポジションサイズ割合を決定
        position_size_fraction = kelly_params.get_position_size_fraction(
            risk_tolerance=risk_tolerance,
            max_fraction=max_risk_fraction * 5,  # Kellyの5倍まで
        )

        # 最大リスク割合でクリップ
        position_size_fraction = min(position_size_fraction, max_risk_fraction)

        # ボラティリティ調整
        if volatility_adjustment:
            position_size_fraction = self._adjust_for_volatility(
                position_size_fraction, trades
            )

        risk_amount = current_balance * position_size_fraction

        # 信頼度スコア計算
        confidence_score = self._calculate_confidence_score(kelly_params)

        reasoning = self._generate_reasoning(
            kelly_params, position_size_fraction, risk_tolerance
        )

        return PositionSizeDecision(
            kelly_params=kelly_params,
            position_size_fraction=position_size_fraction,
            risk_amount=risk_amount,
            confidence_score=confidence_score,
            reasoning=reasoning,
        )

    def _adjust_for_volatility(
        self, base_fraction: float, trades: List[Dict[str, Any]]
    ) -> float:
        """ボラティリティに基づく調整"""
        if len(trades) < 5:
            return base_fraction

        # 最近のP&Lボラティリティを計算
        recent_pnls = [trade.get("pnl", 0) for trade in trades[-20:]]
        volatility = np.std(recent_pnls) / np.mean(np.abs(recent_pnls))

        # 高ボラティリティ時はポジションサイズを小さく
        if volatility > 0.5:  # 50%以上のボラティリティ
            adjustment_factor = 0.7
        elif volatility > 0.3:  # 30%以上のボラティリティ
            adjustment_factor = 0.85
        else:
            adjustment_factor = 1.0

        return base_fraction * adjustment_factor

    def _calculate_confidence_score(self, kelly_params: KellyParameters) -> float:
        """信頼度スコアを計算"""
        # トレード数の影響
        trade_confidence = min(
            1.0, kelly_params.total_trades / 50
        )  # 50トレードで最大信頼度

        # 勝率の安定性
        win_rate_confidence = (
            1.0 - abs(kelly_params.win_rate - 0.5) * 2
        )  # 50%に近いほど信頼度高

        # 勝ち負け比率の安定性
        ratio_confidence = min(
            1.0, kelly_params.win_loss_ratio / 2.0
        )  # 2.0以上で最大信頼度

        return (trade_confidence + win_rate_confidence + ratio_confidence) / 3.0

    def _generate_reasoning(
        self, kelly_params: KellyParameters, position_size: float, risk_tolerance: str
    ) -> str:
        """決定理由を生成"""
        kelly_pct = kelly_params.kelly_fraction * 100
        half_kelly_pct = kelly_params.half_kelly_fraction * 100
        position_pct = position_size * 100

        reasoning_parts = [
            f"Kelly分数: {kelly_pct:.1f}%, 勝率: {kelly_params.win_rate:.1%}, "
            f"勝ち負け比率: {kelly_params.win_loss_ratio:.2f}, 総トレード: {kelly_params.total_trades}",
            f"リスク許容度 '{risk_tolerance}' を適用: {position_pct:.2f}% のポジションサイズ",
        ]

        if position_size < kelly_params.half_kelly_fraction:
            reasoning_parts.append("保守的なポジションサイズを選択")
        elif position_size > kelly_params.kelly_fraction:
            reasoning_parts.append("リスク制限によりKelly分数を上回るポジションサイズ")

        return " | ".join(reasoning_parts)

    @PerformanceProfiler.profile
    def optimize_portfolio_allocation(
        self,
        trades_by_symbol: Dict[str, List[Dict[str, Any]]],
        total_balance: float,
        max_allocation_per_symbol: float = 0.3,
    ) -> Dict[str, PositionSizeDecision]:
        """
        ポートフォリオ全体での最適配分を計算

        Args:
            trades_by_symbol: シンボル別トレード履歴
            total_balance: 総残高
            max_allocation_per_symbol: 1シンボルあたりの最大配分割合

        Returns:
            シンボル別ポジションサイズ決定
        """
        allocations = {}

        for symbol, symbol_trades in trades_by_symbol.items():
            if not symbol_trades:
                continue

            # シンボル別ポジションサイズ計算
            decision = self.calculate_dynamic_position_size(
                symbol_trades, total_balance
            )

            # ポートフォリオ全体の制約を適用
            max_allocation = total_balance * max_allocation_per_symbol
            decision.risk_amount = min(decision.risk_amount, max_allocation)

            # ポジションサイズ割合を再計算
            decision.position_size_fraction = decision.risk_amount / total_balance

            allocations[symbol] = decision

        # 総配分の正規化（100%を超えないように）
        total_allocation = sum(d.position_size_fraction for d in allocations.values())
        if total_allocation > 1.0:
            for symbol in allocations:
                allocations[symbol].position_size_fraction /= total_allocation
                allocations[symbol].risk_amount = (
                    total_balance * allocations[symbol].position_size_fraction
                )

        return allocations


class KellyCriterionValidator:
    """Kelly基準の妥当性検証"""

    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.logger = logging.getLogger(__name__)

    def validate_kelly_effectiveness(
        self,
        trades: List[Dict[str, Any]],
        balance_history: List[float],
        kelly_fractions: List[float] = [0.1, 0.25, 0.5, 0.75, 1.0],
    ) -> Dict[str, Any]:
        """
        Kelly基準の有効性を検証

        Args:
            trades: トレード履歴
            balance_history: 残高履歴
            kelly_fractions: 検証するKelly分数

        Returns:
            検証結果
        """
        results = {}

        for fraction in kelly_fractions:
            # 指定されたKelly分数でのバックテスト
            simulated_balance = self._simulate_kelly_strategy(
                trades, balance_history[0], fraction
            )

            # 性能指標計算
            returns = pd.Series(simulated_balance).pct_change().dropna()
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(simulated_balance)
            total_return = (
                simulated_balance[-1] - simulated_balance[0]
            ) / simulated_balance[0]

            results[fraction] = {
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "total_return": total_return,
                "final_balance": simulated_balance[-1],
            }

        # 最適Kelly分数の特定
        best_fraction = max(results.keys(), key=lambda k: results[k]["sharpe_ratio"])

        return {
            "fraction_results": results,
            "optimal_fraction": best_fraction,
            "optimal_performance": results[best_fraction],
        }

    def _simulate_kelly_strategy(
        self,
        trades: List[Dict[str, Any]],
        initial_balance: float,
        kelly_fraction: float,
    ) -> List[float]:
        """Kelly戦略のシミュレーション"""
        balance = initial_balance
        balance_history = [balance]

        for trade in trades:
            pnl = trade.get("pnl", 0)
            # Kelly分数に基づくポジションサイズ調整
            position_size = balance * kelly_fraction
            adjusted_pnl = pnl * (position_size / balance)

            balance += adjusted_pnl
            balance_history.append(balance)

        return balance_history

    def _calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """Sharpe Ratio計算"""
        from ztb.metrics.metrics import sharpe_ratio

        return sharpe_ratio(returns, rf=risk_free_rate)

    def _calculate_max_drawdown(self, balance_history: List[float]) -> float:
        """最大ドローダウン計算"""
        balance_series = pd.Series(balance_history)
        peak = balance_series.expanding().max()
        drawdown = (balance_series - peak) / peak
        return drawdown.min()


# ===== 使用例 =====

if __name__ == "__main__":
    """使用例"""

    # Kellyポジションサイザーの初期化
    sizer = KellyPositionSizer()

    # サンプルトレードデータ
    sample_trades = [
        {"pnl": 100, "entry_price": 1000},
        {"pnl": -50, "entry_price": 1050},
        {"pnl": 150, "entry_price": 1020},
        {"pnl": -30, "entry_price": 1080},
        {"pnl": 200, "entry_price": 1100},
        {"pnl": 80, "entry_price": 1120},
        {"pnl": -70, "entry_price": 1150},
        {"pnl": 120, "entry_price": 1180},
        {"pnl": -40, "entry_price": 1200},
        {"pnl": 180, "entry_price": 1220},
    ]

    current_balance = 10000

    # Kellyパラメータ計算
    kelly_params = sizer.calculate_kelly_parameters(sample_trades, current_balance)
    if kelly_params:
        print(
            f"Kellyパラメータ: 勝率={kelly_params.win_rate:.1%}, "
            f"勝ち負け比率={kelly_params.win_loss_ratio:.2f}, "
            f"Kelly分数={kelly_params.kelly_fraction:.1%}"
        )

    # 動的ポジションサイズ計算
    decision = sizer.calculate_dynamic_position_size(
        sample_trades, current_balance, risk_tolerance="half"
    )

    print(
        f"ポジションサイズ決定: {decision.position_size_fraction:.2%} "
        f"({decision.risk_amount:.0f}円), 信頼度: {decision.confidence_score:.2f}"
    )
    print(f"理由: {decision.reasoning}")

    # Kelly基準検証
    validator = KellyCriterionValidator()
    validation_result = validator.validate_kelly_effectiveness(
        sample_trades, [current_balance] * len(sample_trades)
    )

    print(f"最適Kelly分数: {validation_result['optimal_fraction']}")
    print(
        f"最適性能: Sharpe={validation_result['optimal_performance']['sharpe_ratio']:.2f}, "
        f"総リターン={validation_result['optimal_performance']['total_return']:.1%}"
    )
