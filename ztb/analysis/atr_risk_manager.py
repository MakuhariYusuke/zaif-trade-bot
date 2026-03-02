"""
Phase 3-2: パラメータ最適化 - ATRベース動的リスク管理システム

ATR (Average True Range) を使用した動的リスク管理を実装します。
市場ボラティリティに応じてストップロスとテイクプロフィットを調整します。
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any
from ztb.analysis.common.types import RiskProfile, RiskProfileLimits

import numpy as np
import pandas as pd

from ztb.utils.performance_profiler import PerformanceProfiler

# RiskManagementMode is an alias for RiskProfile
RiskManagementMode = RiskProfile

@dataclass
class ATRParameters(RiskProfileLimits):
    """ATRパラメータ"""

    period: int = 14  # ATR計算期間
    stop_loss_multiplier: float = 2.0  # ストップロス乗数
    take_profit_risk_multiplier: float = 2.0  # テイクプロフィットリスク乗数
    trailing_stop_activation: float = 1.5  # トレーリングストップ起動乗数
    max_stop_distance: float = 0.05  # 最大ストップ距離（5%）
    min_stop_distance: float = 0.005  # 最小ストップ距離（0.5%）

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        base_dict = super().to_dict() if hasattr(super(), 'to_dict') else {}
        return {
            **base_dict,
            "period": self.period,
            "stop_loss_multiplier": self.stop_loss_multiplier,
            "take_profit_risk_multiplier": self.take_profit_risk_multiplier,
            "trailing_stop_activation": self.trailing_stop_activation,
            "max_stop_distance": self.max_stop_distance,
            "min_stop_distance": self.min_stop_distance,
        }

@dataclass
class RiskLevel:
    """リスクレベル"""

    atr_value: float
    volatility_percentile: float  # ボラティリティのパーセンタイル
    market_regime: str  # "low_vol", "normal_vol", "high_vol", "extreme_vol"

    @property
    def is_high_volatility(self) -> bool:
        """高ボラティリティか"""
        return self.volatility_percentile > 0.7

    @property
    def is_low_volatility(self) -> bool:
        """低ボラティリティか"""
        return self.volatility_percentile < 0.3

@dataclass
class PositionRiskLimits:
    """ポジションリスク制限"""

    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    trailing_stop_price: float | None
    risk_amount: float
    risk_percentage: float
    atr_value: float
    timestamp: pd.Timestamp

    @property
    def stop_distance(self) -> float:
        """ストップロス距離"""
        return abs(self.entry_price - self.stop_loss_price) / self.entry_price

    @property
    def profit_target_distance(self) -> float:
        """利益目標距離"""
        return abs(self.take_profit_price - self.entry_price) / self.entry_price

    @property
    def risk_reward_ratio(self) -> float:
        """リスク報酬比率"""
        risk = self.stop_distance
        reward = self.profit_target_distance
        return reward / risk if risk > 0 else 0.0

    @property
    def max_position_size(self) -> float:
        """Maximum position size derived from risk percentage (simple proxy)."""
        return self.risk_percentage

class ATRRiskManager:
    """ATRベースリスクマネージャー"""

    def __init__(self, atr_params: ATRParameters | None = None):
        self.atr_params = atr_params or ATRParameters()
        self.profiler = PerformanceProfiler()
        self.logger = logging.getLogger(__name__)

        # ATR履歴（動的調整用）
        self.atr_history: list[float] = []
        self.volatility_history: list[float] = []

    def calculate_atr(
        self, data: pd.DataFrame, period: int | None = None
    ) -> pd.Series:
        """
        ATRを計算

        Args:
            data: OHLCデータ (columns: ['open', 'high', 'low', 'close'])
            period: ATR期間（Noneの場合はデフォルト使用）

        Returns:
            ATR系列
        """
        period = period or self.atr_params.period

        if not all(col in data.columns for col in ["high", "low", "close"]):
            raise ValueError("データに 'high', 'low', 'close' カラムが必要です")

        # True Range計算
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR計算（ロール平均を使用して、期間不足時はNaNを保持）
        atr = true_range.rolling(window=period, min_periods=period).mean()

        return atr

    def assess_risk_level(self, current_atr: float, data: pd.DataFrame) -> RiskLevel:
        """
        リスクレベルを評価

        Args:
            current_atr: 現在のATR値
            data: 市場データ

        Returns:
            リスクレベル
        """
        # ATR履歴の更新
        self.atr_history.append(current_atr)
        if len(self.atr_history) > 100:  # 最大100期間保持
            self.atr_history.pop(0)

        if len(self.atr_history) < 10:
            # 十分な履歴がない場合は中間値を使用
            volatility_percentile = 0.5
        else:
            # ボラティリティのパーセンタイルを計算
            atr_series = pd.Series(self.atr_history)
            volatility_percentile = atr_series.rank(pct=True).iloc[-1]

        # 市場レジームの判定
        if volatility_percentile > 0.9:
            regime = "extreme_vol"
        elif volatility_percentile > 0.7:
            regime = "high_vol"
        elif volatility_percentile < 0.3:
            regime = "low_vol"
        else:
            regime = "normal_vol"

        return RiskLevel(
            atr_value=current_atr,
            volatility_percentile=volatility_percentile,
            market_regime=regime,
        )

    def calculate_position_limits(
        self,
        entry_price: float,
        position_size: float,
        current_atr: float,
        risk_level: RiskLevel,
        is_long: bool = True,
        mode: RiskManagementMode = RiskManagementMode.DYNAMIC,
    ) -> PositionRiskLimits:
        """
        ポジションのリスク制限を計算

        Args:
            entry_price: エントリー価格
            position_size: ポジションサイズ
            current_atr: 現在のATR値
            risk_level: リスクレベル
            is_long: ロングポジションか
            mode: リスク管理モード

        Returns:
            ポジションリスク制限
        """
        # モードに応じたパラメータ調整
        params = self._adjust_parameters_for_mode(mode, risk_level)

        # リスク金額計算
        risk_percentage = position_size  # ポジションサイズをリスク%として使用
        risk_amount = entry_price * risk_percentage

        # ストップロス計算
        stop_distance = current_atr * params.stop_loss_multiplier

        # 最小・最大ストップ距離の制約
        min_distance = entry_price * params.min_stop_distance
        max_distance = entry_price * params.max_stop_distance

        stop_distance = np.clip(stop_distance, min_distance, max_distance)

        if is_long:
            stop_loss_price = entry_price - stop_distance
            trailing_stop_price = entry_price + (
                current_atr * params.trailing_stop_activation
            )
        else:
            stop_loss_price = entry_price + stop_distance
            trailing_stop_price = entry_price - (
                current_atr * params.trailing_stop_activation
            )

        # テイクプロフィット計算
        risk_distance = abs(entry_price - stop_loss_price)
        profit_distance = risk_distance * params.take_profit_risk_multiplier

        if is_long:
            take_profit_price = entry_price + profit_distance
        else:
            take_profit_price = entry_price - profit_distance

        return PositionRiskLimits(
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            trailing_stop_price=trailing_stop_price,
            risk_amount=risk_amount,
            risk_percentage=risk_percentage,
            atr_value=current_atr,
            timestamp=pd.Timestamp.now(),
        )

    def _adjust_parameters_for_mode(
        self, mode: RiskManagementMode, risk_level: RiskLevel
    ) -> ATRParameters:
        """モードに応じてパラメータを調整"""
        base_params = self.atr_params

        if mode == RiskManagementMode.CONSERVATIVE:
            # 保守的: 広いストップ、狭い利益目標
            return ATRParameters(
                period=base_params.period,
                stop_loss_multiplier=base_params.stop_loss_multiplier * 1.5,
                take_profit_risk_multiplier=base_params.take_profit_risk_multiplier
                * 0.8,
                trailing_stop_activation=base_params.trailing_stop_activation,
                max_stop_distance=base_params.max_stop_distance,
                min_stop_distance=base_params.min_stop_distance,
            )

        elif mode == RiskManagementMode.AGGRESSIVE:
            # 積極的: 狭いストップ、広い利益目標
            return ATRParameters(
                period=base_params.period,
                stop_loss_multiplier=base_params.stop_loss_multiplier * 0.7,
                take_profit_risk_multiplier=base_params.take_profit_risk_multiplier
                * 1.5,
                trailing_stop_activation=base_params.trailing_stop_activation,
                max_stop_distance=base_params.max_stop_distance * 1.2,
                min_stop_distance=base_params.min_stop_distance,
            )

        elif mode == RiskManagementMode.DYNAMIC:
            # 動的: ボラティリティに応じて調整
            if risk_level.is_high_volatility:
                # 高ボラティリティ: 広いストップ
                multiplier = 1.3
            elif risk_level.is_low_volatility:
                # 低ボラティリティ: 狭いストップ
                multiplier = 0.8
            else:
                multiplier = 1.0

            return ATRParameters(
                period=base_params.period,
                stop_loss_multiplier=base_params.stop_loss_multiplier * multiplier,
                take_profit_risk_multiplier=base_params.take_profit_risk_multiplier,
                trailing_stop_activation=base_params.trailing_stop_activation,
                max_stop_distance=base_params.max_stop_distance,
                min_stop_distance=base_params.min_stop_distance,
            )

        else:  # MODERATE
            return base_params

    def update_trailing_stop(
        self, current_price: float, limits: PositionRiskLimits, is_long: bool = True
    ) -> float | None:
        """
        トレーリングストップを更新

        Args:
            current_price: 現在の価格
            limits: 現在のリスク制限
            is_long: ロングポジションか

        Returns:
            更新されたトレーリングストップ価格（Noneの場合は変更なし）
        """
        if limits.trailing_stop_price is None:
            return None

        if is_long:
            # ロング: 価格が上昇したらトレーリングストップを引き上げる
            if current_price > limits.trailing_stop_price:
                new_stop = current_price - (
                    limits.atr_value * self.atr_params.stop_loss_multiplier
                )
                # 既存のストップロスより有利な場合のみ更新
                if new_stop > limits.stop_loss_price:
                    return new_stop
        else:
            # ショート: 価格が下降したらトレーリングストップを引き下げる
            if current_price < limits.trailing_stop_price:
                new_stop = current_price + (
                    limits.atr_value * self.atr_params.stop_loss_multiplier
                )
                # 既存のストップロスより有利な場合のみ更新
                if new_stop < limits.stop_loss_price:
                    return new_stop

        return None

    def should_exit_position(
        self, current_price: float, limits: PositionRiskLimits, is_long: bool = True
    ) -> tuple[bool, str]:
        """
        ポジションを退出すべきかを判定

        Args:
            current_price: 現在の価格
            limits: リスク制限
            is_long: ロングポジションか

        Returns:
            (退出フラグ, 理由)
        """
        # ストップロス判定
        if is_long:
            if current_price <= limits.stop_loss_price:
                return (
                    True,
                    f"ストップロス発動: {current_price:.2f} <= {limits.stop_loss_price:.2f}",
                )
        else:
            if current_price >= limits.stop_loss_price:
                return (
                    True,
                    f"ストップロス発動: {current_price:.2f} >= {limits.stop_loss_price:.2f}",
                )

        # テイクプロフィット判定
        if is_long:
            if current_price >= limits.take_profit_price:
                return (
                    True,
                    f"テイクプロフィット発動: {current_price:.2f} >= {limits.take_profit_price:.2f}",
                )
        else:
            if current_price <= limits.take_profit_price:
                return (
                    True,
                    f"テイクプロフィット発動: {current_price:.2f} <= {limits.take_profit_price:.2f}",
                )

        return False, ""

    @PerformanceProfiler.profile
    def optimize_atr_parameters(
        self,
        historical_data: pd.DataFrame,
        trades: list[dict[str, Any]],
        parameter_ranges: dict[str, list[float]] | None = None,
    ) -> ATRParameters:
        """
        ATRパラメータを最適化

        Args:
            historical_data: 過去データ
            trades: トレード履歴
            parameter_ranges: パラメータ範囲（Noneの場合はデフォルト）

        Returns:
            最適化されたパラメータ
        """
        if parameter_ranges is None:
            parameter_ranges = {
                "stop_loss_multiplier": [1.0, 1.5, 2.0, 2.5, 3.0],
                "take_profit_risk_multiplier": [1.5, 2.0, 2.5, 3.0, 4.0],
                "period": [10, 14, 20, 28],
            }

        best_params = None
        best_score = float("-inf")

        # グリッドサーチ
        for sl_mult in parameter_ranges["stop_loss_multiplier"]:
            for tp_mult in parameter_ranges["take_profit_risk_multiplier"]:
                for period in parameter_ranges["period"]:
                    test_params = ATRParameters(
                        period=period,
                        stop_loss_multiplier=sl_mult,
                        take_profit_risk_multiplier=tp_mult,
                    )

                    # パフォーマンス評価
                    score = self._evaluate_parameters(
                        test_params, historical_data, trades
                    )

                    if score > best_score:
                        best_score = score
                        best_params = test_params

        self.logger.info(f"ATRパラメータ最適化完了: 最高スコア = {best_score:.3f}")
        return best_params or self.atr_params

    def _evaluate_parameters(
        self, params: ATRParameters, data: pd.DataFrame, trades: list[dict[str, Any]]
    ) -> float:
        """パラメータを評価"""
        # ATR計算
        atr_series = self.calculate_atr(data, params.period)

        total_return = 0
        winning_trades = 0

        for trade in trades:
            entry_time = pd.Timestamp(trade.get("entry_time"))
            exit_time = pd.Timestamp(trade.get("exit_time", entry_time))

            # エントリー時のATRを取得
            entry_atr = (
                atr_series.loc[:entry_time].iloc[-1]
                if entry_time in atr_series.index
                else atr_series.mean()
            )

            # リスク制限計算
            limits = self.calculate_position_limits(
                entry_price=trade.get("entry_price", 0),
                position_size=0.01,  # 1%固定で評価
                current_atr=entry_atr,
                risk_level=RiskLevel(
                    atr_value=entry_atr,
                    volatility_percentile=0.5,
                    market_regime="normal_vol",
                ),
            )

            # 実際の出口価格で評価
            actual_exit = trade.get("exit_price", trade.get("entry_price", 0))
            pnl = trade.get("pnl", 0)

            # リスク調整リターン
            risk_adjusted_return = (
                pnl / limits.risk_amount if limits.risk_amount > 0 else 0
            )

            total_return += risk_adjusted_return
            if pnl > 0:
                winning_trades += 1

        # スコア計算（リスク調整リターン + 勝率ボーナス）
        win_rate = winning_trades / len(trades) if trades else 0
        score = total_return + (win_rate * 0.1)  # 勝率に10%の重み

        return score

# ===== 使用例 =====

if __name__ == "__main__":
    """使用例"""

    # ATRリスクマネージャーの初期化
    risk_manager = ATRRiskManager()

    # サンプルOHLCデータ
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)

    sample_data = pd.DataFrame(
        {
            "open": 100 + np.random.randn(100).cumsum(),
            "high": 105 + np.random.randn(100).cumsum(),
            "low": 95 + np.random.randn(100).cumsum(),
            "close": 100 + np.random.randn(100).cumsum(),
        },
        index=dates,
    )

    # ATR計算
    atr_series = risk_manager.calculate_atr(sample_data)
    current_atr = atr_series.iloc[-1]

    print(f"現在のATR: {current_atr:.4f}")

    # リスクレベル評価
    risk_level = risk_manager.assess_risk_level(current_atr, sample_data)
    print(
        f"リスクレベル: {risk_level.market_regime} (パーセンタイル: {risk_level.volatility_percentile:.1%})"
    )

    # ポジション制限計算
    limits = risk_manager.calculate_position_limits(
        entry_price=100.0,
        position_size=0.02,  # 2%
        current_atr=current_atr,
        risk_level=risk_level,
        is_long=True,
        mode=RiskManagementMode.DYNAMIC,
    )

    print("ポジション制限:")
    print(f"  エントリー: {limits.entry_price:.2f}")
    print(
        f"  ストップロス: {limits.stop_loss_price:.2f} (距離: {limits.stop_distance:.1%})"
    )
    print(
        f"  テイクプロフィット: {limits.take_profit_price:.2f} (距離: {limits.profit_target_distance:.1%})"
    )
    print(f"  リスク報酬比率: {limits.risk_reward_ratio:.2f}")

    # 退出判定テスト
    test_prices = [98.0, 102.0, 105.0]  # ストップロス、通常、テイクプロフィット

    for price in test_prices:
        should_exit, reason = risk_manager.should_exit_position(
            price, limits, is_long=True
        )
        print(f"価格 {price:.2f}: {'退出' if should_exit else '継続'} - {reason}")
