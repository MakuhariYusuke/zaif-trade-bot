"""
戦略評価関数

ウォークフォワード分析で使用する戦略評価関数を提供します。
"""

from typing import Callable

import pandas as pd

from ztb.analysis.walk_forward_analyzer import ParameterSet
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

def create_simple_strategy_evaluator(
    trades_data: list = None,
) -> Callable[[pd.DataFrame, ParameterSet], dict[str, float]]:
    """
    シンプルな戦略評価関数を作成

    Args:
        trades_data: トレードデータ（オプション）

    Returns:
        戦略評価関数
    """

    def strategy_evaluator(
        data: pd.DataFrame, params: ParameterSet
    ) -> dict[str, float]:
        """
        戦略を評価して性能指標を返す

        Args:
            data: 市場データ
            params: パラメータセット

        Returns:
            性能指標
        """
        try:
            # 価格変化の計算
            returns = data["close"].pct_change().dropna()

            if len(returns) == 0:
                return {
                    "sharpe_ratio": 0.0,
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "total_trades": 0,
                }

            # 基本指標の計算
            total_return = (1 + returns).prod() - 1
            from ztb.metrics.technical import calculate_volatility_from_returns

            volatility = calculate_volatility_from_returns(
                returns, window=len(returns), annualize=False
            )

            from ztb.metrics.metrics import sharpe_ratio as calc_sharpe_ratio

            sharpe_ratio = calc_sharpe_ratio(returns)

            # ドローダウンの計算
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # 勝率の計算（簡易版）
            positive_returns = (returns > 0).sum()
            total_trades = len(returns)
            win_rate = positive_returns / total_trades if total_trades > 0 else 0

            # パラメータによる調整
            # ストップロスが厳しいほどリスクが減る
            risk_adjustment = params.stop_loss_atr_multiplier / 3.0  # 基準値3.0
            adjusted_sharpe = sharpe_ratio * (1 - risk_adjustment * 0.1)

            # 信頼度閾値による調整
            confidence_adjustment = params.confidence_threshold / 0.7  # 基準値0.7
            final_sharpe = adjusted_sharpe * confidence_adjustment

            return {
                "sharpe_ratio": final_sharpe,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "total_trades": total_trades,
                "volatility": volatility,
                "risk_adjusted_return": final_sharpe / abs(max_drawdown)
                if max_drawdown < 0
                else 0,
            }

        except Exception as e:
            # エラー時はデフォルト値を返す
            logger.warning("Strategy evaluation failed: %s", e)
            return {
                "sharpe_ratio": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
            }

    return strategy_evaluator

def create_trend_following_strategy_evaluator() -> (
    Callable[[pd.DataFrame, ParameterSet], dict[str, float]]
):
    """
    トレンドフォロー戦略の評価関数を作成

    Returns:
        トレンドフォロー戦略評価関数
    """

    def strategy_evaluator(
        data: pd.DataFrame, params: ParameterSet
    ) -> dict[str, float]:
        """
        トレンドフォロー戦略を評価

        Args:
            data: 市場データ
            params: パラメータセット

        Returns:
            性能指標
        """
        try:
            # 移動平均の計算
            short_ma = data["close"].rolling(window=10).mean()
            long_ma = data["close"].rolling(window=30).mean()

            # トレンドシグナル
            trend_signal = (short_ma > long_ma).astype(int)

            # リターンの計算
            returns = data["close"].pct_change().dropna()

            # トレンド方向の取引のみを考慮
            trend_aligned_returns = returns * trend_signal.shift(1).dropna()

            if len(trend_aligned_returns) == 0:
                return {
                    "sharpe_ratio": 0.0,
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "total_trades": 0,
                }

            # 性能指標の計算
            total_return = (1 + trend_aligned_returns).prod() - 1
            from ztb.metrics.technical import calculate_volatility_from_returns

            volatility = calculate_volatility_from_returns(
                trend_aligned_returns,
                window=len(trend_aligned_returns),
                annualize=False,
            )

            from ztb.metrics.metrics import sharpe_ratio as calc_sharpe_ratio

            sharpe_ratio = calc_sharpe_ratio(trend_aligned_returns)

            # ドローダウン
            cumulative = (1 + trend_aligned_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # 勝率
            positive_trades = (trend_aligned_returns > 0).sum()
            total_trades = (trend_aligned_returns != 0).sum()
            win_rate = positive_trades / total_trades if total_trades > 0 else 0

            return {
                "sharpe_ratio": sharpe_ratio,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "total_trades": int(total_trades),
            }

        except Exception as e:
            logger.warning("Trend-following evaluation failed: %s", e)
            return {
                "sharpe_ratio": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
            }

    return strategy_evaluator
