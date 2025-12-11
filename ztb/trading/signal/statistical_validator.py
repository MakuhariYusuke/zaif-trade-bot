"""
Statistical Validator for Signal Quality Assessment
Phase 3統合: 統計的有意性評価
実装完了: 2025年11月12日
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TradeResult:
    """取引結果のデータクラス"""

    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    position_size: float = 0.0
    signal_score: float = 0.0
    type: str = "BUY"


class StatisticalValidator:
    """統計的シグナルバリデーション"""

    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level

    def validate_signal_quality(
        self,
        trades: List[Union[TradeResult, Dict[str, Any]]],
        market_returns: np.ndarray,
    ) -> Dict[str, float]:
        """
        シグナルの統計的有意性を評価
        """
        # シグナルベースのリターンを計算
        signal_returns = self._calculate_signal_returns(trades, market_returns)

        if len(signal_returns) == 0:
            return {
                "t_statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "mean_return": 0.0,
                "volatility": 0.0,
            }

        # t検定で有意性を確認
        t_stat, p_value = stats.ttest_1samp(signal_returns, 0)

        # シャープレシオ計算
        sharpe_ratio = self._calculate_sharpe_ratio(signal_returns)

        # 最大ドローダウン計算
        max_drawdown = self._calculate_max_drawdown(signal_returns)

        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "mean_return": np.mean(signal_returns),
            "volatility": np.std(signal_returns),
        }

    def _calculate_signal_returns(
        self,
        trades: List[Union[TradeResult, Dict[str, Any]]],
        market_returns: np.ndarray,
    ) -> np.ndarray:
        """シグナルベースのリターン計算"""
        if len(trades) == 0:
            return np.array([])

        signal_returns = []

        for trade in trades:
            if isinstance(trade, TradeResult):
                pnl = trade.pnl
                position_size = trade.position_size
            else:
                # Dictの場合
                pnl = trade.get("pnl", 0.0)
                position_size = trade.get("position_size", 1.0)

            if pnl is not None and position_size > 0:
                # パーセンテージリターン = PnL / ポジションサイズ
                return_pct = pnl / position_size
                signal_returns.append(return_pct)
            else:
                signal_returns.append(0.0)

        return np.array(signal_returns)

    def _calculate_sharpe_ratio(
        self, returns: np.ndarray, risk_free_rate: float = 0.03
    ) -> float:
        """シャープレシオ計算"""
        from ztb.metrics.metrics import sharpe_ratio

        return sharpe_ratio(returns, rf=risk_free_rate)

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """最大ドローダウン計算"""
        if len(returns) == 0:
            return 0.0

        from ztb.metrics.metrics import max_drawdown

        cumulative = np.cumprod(1 + returns)
        # ztb.metrics.max_drawdown returns negative value, convert to positive
        return -max_drawdown(cumulative)
