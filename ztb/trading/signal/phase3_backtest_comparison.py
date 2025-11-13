"""
Phase 3 Integrated Backtest System
リスク管理統合済みバックテスト実行
実装完了: 2025年11月12日
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .enhanced_risk_manager import EnhancedRiskManager
from .statistical_validator import StatisticalValidator

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class IntegratedBacktestRunner:
    """統合バックテスト実行システム"""

    def __init__(self):
        self.risk_manager = EnhancedRiskManager()
        self.validator = StatisticalValidator()

    def run_enhanced_backtest_aggressive(self,
                                       market_data: pd.DataFrame,
                                       initial_balance: float = 10000) -> Dict:
        """
        Phase 3 Aggressiveバージョン実行
        リスク調整済み + 緩和条件
        """
        balance = initial_balance
        trades = []
        peak_balance = initial_balance
        position = 0.0  # BTC position
        entry_price = 0.0

        # 緩和されたパラメータ
        position_size_pct = 0.10  # 10%
        stop_loss_pct = 0.05      # 5%
        take_profit_pct = 0.12    # 12%

        for i in range(len(market_data)):
            current_data = market_data.iloc[i]
            current_price = current_data['close']

            # Phase 2スコア計算（既存）
            phase2_score = self._calculate_phase2_score(current_data)

            # Phase 3リスク調整
            risk_signal = self.risk_manager.calculate_risk_adjusted_score(
                phase2_score=phase2_score,
                market_data=self._get_multi_timeframe_data(market_data, i),
                volatility=current_data.get('volatility', 0.02)
            )

            # 取引実行判定
            if risk_signal.position_size > 0 and risk_signal.confidence > 0.7:
                if risk_signal.action == "BUY" and position == 0:
                    # BUYシグナル - ポジション取得
                    position_value = balance * risk_signal.position_size
                    position = position_value / current_price
                    entry_price = current_price
                    balance -= position_value

                    # 取引記録
                    trade = {
                        'entry_time': current_data.name,
                        'entry_price': entry_price,
                        'position_size': position_value,
                        'signal_score': risk_signal.score,
                        'risk_multiplier': risk_signal.risk_multiplier,
                        'confidence': risk_signal.confidence,
                        'type': 'BUY'
                    }
                    trades.append(trade)

                elif risk_signal.action == "SELL" and position > 0:
                    # SELLシグナル - ポジション決済
                    exit_value = position * current_price
                    pnl = exit_value - (position * entry_price)
                    balance += exit_value

                    # 取引記録更新
                    if trades and trades[-1]['type'] == 'BUY':
                        trades[-1].update({
                            'exit_time': current_data.name,
                            'exit_price': current_price,
                            'pnl': pnl,
                            'type': 'COMPLETED'
                        })

                    position = 0.0
                    entry_price = 0.0

            # ストップロス/テイクプロフィットチェック（ポジション保有中）
            if position > 0:
                if current_price <= entry_price * (1 - stop_loss_pct):
                    # ストップロス
                    exit_value = position * current_price
                    pnl = exit_value - (position * entry_price)
                    balance += exit_value

                    if trades and trades[-1]['type'] == 'BUY':
                        trades[-1].update({
                            'exit_time': current_data.name,
                            'exit_price': current_price,
                            'pnl': pnl,
                            'type': 'STOP_LOSS'
                        })

                    position = 0.0
                    entry_price = 0.0

                elif current_price >= entry_price * (1 + take_profit_pct):
                    # テイクプロフィット
                    exit_value = position * current_price
                    pnl = exit_value - (position * entry_price)
                    balance += exit_value

                    if trades and trades[-1]['type'] == 'BUY':
                        trades[-1].update({
                            'exit_time': current_data.name,
                            'exit_price': current_price,
                            'pnl': pnl,
                            'type': 'TAKE_PROFIT'
                        })

                    position = 0.0
                    entry_price = 0.0

        # 統計的バリデーション
        validation_results = self.validator.validate_signal_quality(
            trades, market_data['returns'].values
        )

        # 勝率計算
        completed_trades = [t for t in trades if t.get('pnl', 0) != 0]
        win_rate = len([t for t in completed_trades if t['pnl'] > 0]) / len(completed_trades) if completed_trades else 0

        return {
            'trades': trades,
            'final_balance': balance,
            'total_return': (balance - initial_balance) / initial_balance,
            'max_drawdown': validation_results['max_drawdown'],
            'sharpe_ratio': validation_results['sharpe_ratio'],
            'win_rate': win_rate,
            'validation': validation_results
        }

    def _calculate_phase2_score(self, row: pd.Series) -> float:
        """Phase 2スコア計算（簡易版）"""
        # 実際の実装ではSignalQualityScorerを使用
        # ここでは簡易的な計算
        close = row.get('close', 0)
        if close == 0:
            return 50.0

        # RSIベースの簡易スコア
        if 'rsi' in row:
            rsi = row['rsi']
            if rsi < 30:
                return 20.0  # SELL
            elif rsi > 70:
                return 80.0  # BUY
            else:
                return 50.0  # HOLD
        else:
            return 50.0

    def _get_multi_timeframe_data(self, market_data: pd.DataFrame, current_idx: int) -> Dict[str, np.ndarray]:
        """マルチタイムフレームデータを取得"""
        multi_tf_data = {}

        # 簡易的に同じデータを複数の時間軸として使用
        # 実際の実装では適切な時間軸データを取得
        if current_idx >= 50:
            prices = market_data['close'].iloc[current_idx-50:current_idx+1].values
            multi_tf_data['1m'] = prices
            multi_tf_data['5m'] = prices[::5]  # 5分間隔サンプリング
            multi_tf_data['15m'] = prices[::15]  # 15分間隔サンプリング

        return multi_tf_data