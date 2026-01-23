"""
v459 Phase 0.2a: BacktestReporter拡張の単体テスト
Doc04仕様準拠の検証
"""

import numpy as np
import pandas as pd
import pytest

from ztb.evaluation.walk_forward.reporter import (
    BacktestReporter,
    classify_trade_type,
    decompose_reverse_trade,
)


class TestClassifyTradeType:
    """Trade Type分類のテスト"""
    
    def test_long_open(self):
        """ロングオープン: 0 → 正"""
        assert classify_trade_type(0.0, 1.0) == "long_open"
        assert classify_trade_type(0.0, 0.5) == "long_open"
    
    def test_long_close(self):
        """ロングクローズ: 正 → 0"""
        assert classify_trade_type(1.0, 0.0) == "long_close"
        assert classify_trade_type(0.5, 0.0) == "long_close"
    
    def test_long_add(self):
        """ロング追加: 正 → より大きい正"""
        assert classify_trade_type(0.5, 1.0) == "long_add"
        assert classify_trade_type(1.0, 1.5) == "long_add"
    
    def test_long_reduce(self):
        """ロング縮小: 正 → より小さい正"""
        assert classify_trade_type(1.0, 0.5) == "long_reduce"
        assert classify_trade_type(1.5, 1.0) == "long_reduce"
    
    def test_short_open(self):
        """ショートオープン: 0 → 負"""
        assert classify_trade_type(0.0, -1.0) == "short_open"
        assert classify_trade_type(0.0, -0.5) == "short_open"
    
    def test_short_close(self):
        """ショートクローズ: 負 → 0"""
        assert classify_trade_type(-1.0, 0.0) == "short_close"
        assert classify_trade_type(-0.5, 0.0) == "short_close"
    
    def test_short_add(self):
        """ショート追加: 負 → より小さい負"""
        assert classify_trade_type(-0.5, -1.0) == "short_add"
        assert classify_trade_type(-1.0, -1.5) == "short_add"
    
    def test_short_reduce(self):
        """ショート縮小: 負 → より大きい負（0に近い）"""
        assert classify_trade_type(-1.0, -0.5) == "short_reduce"
        assert classify_trade_type(-1.5, -1.0) == "short_reduce"
    
    def test_reverse_long_to_short(self):
        """反転: ロング → ショート"""
        assert classify_trade_type(1.0, -1.0) == "reverse"
        assert classify_trade_type(0.5, -0.5) == "reverse"
    
    def test_reverse_short_to_long(self):
        """反転: ショート → ロング"""
        assert classify_trade_type(-1.0, 1.0) == "reverse"
        assert classify_trade_type(-0.5, 0.5) == "reverse"
    
    def test_hold(self):
        """ホールド: 変化なし"""
        assert classify_trade_type(0.0, 0.0) == "hold"
        assert classify_trade_type(1.0, 1.0) == "hold"
        assert classify_trade_type(-1.0, -1.0) == "hold"
    
    def test_near_zero_tolerance(self):
        """ゼロ近傍の許容誤差"""
        assert classify_trade_type(1e-9, 1e-9) == "hold"
        assert classify_trade_type(1e-9, 1.0) == "long_open"
        assert classify_trade_type(1.0, 1e-9) == "long_close"


class TestDecomposeReverseTrade:
    """反転取引分解のテスト"""
    
    def test_long_to_short_decomposition(self):
        """ロング → ショートの分解"""
        timestamp = pd.Timestamp("2026-01-22 12:00:00")
        trades = decompose_reverse_trade(1.0, -1.0, 50000.0, timestamp)
        
        assert len(trades) == 2
        
        # 1. ロングクローズ
        assert trades[0]["type"] == "long_close"
        assert trades[0]["position_before"] == 1.0
        assert trades[0]["position_after"] == 0.0
        assert trades[0]["size"] == 1.0
        assert trades[0]["price"] == 50000.0
        
        # 2. ショートオープン
        assert trades[1]["type"] == "short_open"
        assert trades[1]["position_before"] == 0.0
        assert trades[1]["position_after"] == -1.0
        assert trades[1]["size"] == 1.0
        assert trades[1]["price"] == 50000.0
    
    def test_short_to_long_decomposition(self):
        """ショート → ロングの分解"""
        timestamp = pd.Timestamp("2026-01-22 12:00:00")
        trades = decompose_reverse_trade(-0.5, 0.5, 50000.0, timestamp)
        
        assert len(trades) == 2
        
        # 1. ショートクローズ
        assert trades[0]["type"] == "short_close"
        assert trades[0]["position_before"] == -0.5
        assert trades[0]["position_after"] == 0.0
        assert trades[0]["size"] == 0.5
        
        # 2. ロングオープン
        assert trades[1]["type"] == "long_open"
        assert trades[1]["position_before"] == 0.0
        assert trades[1]["position_after"] == 0.5
        assert trades[1]["size"] == 0.5


class TestBacktestReporterV459:
    """BacktestReporter Doc04仕様拡張のテスト"""
    
    def test_record_trade_long_open(self):
        """ロングオープンの記録"""
        reporter = BacktestReporter()
        
        reporter.record_trade(
            position_before=0.0,
            position_after=1.0,
            pnl=100.0,
            entry_price=50000.0,
            exit_price=50100.0,
            size=1.0,
            fee=10.0,
            slippage=5.0,
            timestamp=pd.Timestamp("2026-01-22 12:00:00")
        )
        
        assert reporter.stats["total_trades"] == 1
        assert reporter.stats["long_trades"] == 1
        assert reporter.stats["short_trades"] == 0
        assert len(reporter.trade_history) == 1
        assert reporter.trade_history[0]["type"] == "long_open"
    
    def test_record_trade_reverse_decomposition(self):
        """反転取引の分解記録"""
        reporter = BacktestReporter()
        
        reporter.record_trade(
            position_before=1.0,
            position_after=-1.0,
            pnl=200.0,
            entry_price=50000.0,
            exit_price=50200.0,
            size=2.0,
            fee=20.0,
            slippage=10.0,
            timestamp=pd.Timestamp("2026-01-22 12:00:00")
        )
        
        # 分解により2つの取引が記録される
        assert len(reporter.trade_history) == 2
        assert reporter.trade_history[0]["type"] == "long_close"
        assert reporter.trade_history[1]["type"] == "short_open"
        
        # PnL/fee/slippageは均等分割
        assert reporter.trade_history[0]["net_pnl"] == 100.0
        assert reporter.trade_history[1]["net_pnl"] == 100.0
        assert reporter.trade_history[0]["fee"] == 10.0
        assert reporter.trade_history[1]["fee"] == 10.0
    
    def test_finalize_stats_profit_factor_zero_loss(self):
        """Profit Factor: 損失ゼロの場合はinf"""
        reporter = BacktestReporter()
        
        # 勝ちトレードのみ
        reporter.record_trade(0.0, 1.0, 100.0, 50000.0, 50100.0, 1.0, 10.0, 5.0)
        reporter.record_trade(1.0, 0.0, 150.0, 50100.0, 50250.0, 1.0, 10.0, 5.0)
        
        reporter.finalize_stats()
        
        assert reporter.stats["profit_factor"] == float('inf')
    
    def test_finalize_stats_profit_factor_zero_profit(self):
        """Profit Factor: 利益ゼロの場合は0"""
        reporter = BacktestReporter()
        
        # 負けトレードのみ
        reporter.record_trade(0.0, 1.0, -100.0, 50000.0, 49900.0, 1.0, 10.0, 5.0)
        reporter.record_trade(1.0, 0.0, -150.0, 49900.0, 49750.0, 1.0, 10.0, 5.0)
        
        reporter.finalize_stats()
        
        # 負けのみの場合、profit=0, loss>0 → PF=0
        assert reporter.stats["profit_factor"] == 0.0
    
    def test_finalize_stats_profit_factor_normal(self):
        """Profit Factor: 通常計算"""
        reporter = BacktestReporter()
        
        # 勝ち: 300
        reporter.record_trade(0.0, 1.0, 100.0, 50000.0, 50100.0, 1.0, 10.0, 5.0)
        reporter.record_trade(1.0, 0.0, 200.0, 50100.0, 50300.0, 1.0, 10.0, 5.0)
        
        # 負け: -150
        reporter.record_trade(0.0, -1.0, -100.0, 50000.0, 49900.0, 1.0, 10.0, 5.0)
        reporter.record_trade(-1.0, 0.0, -50.0, 49900.0, 49850.0, 1.0, 10.0, 5.0)
        
        reporter.finalize_stats()
        
        expected_pf = 300.0 / 150.0
        assert abs(reporter.stats["profit_factor"] - expected_pf) < 1e-6
    
    def test_calculate_sharpe_ratio_daily_aggregation(self):
        """Sharpe Ratio: 日次集約化"""
        reporter = BacktestReporter()
        
        # 3日分のデータ（1440分 × 3）
        initial_balance = 1000000.0
        daily_growth = 1.01  # 1日あたり1%増加
        
        for day in range(3):
            daily_balance = initial_balance * (daily_growth ** (day + 1))
            for minute in range(1440):
                reporter.portfolio_history.append(daily_balance)
        
        reporter.finalize_stats()
        
        # 日次集約でSharpe計算されるべき
        sharpe = reporter.stats["sharpe_ratio"]
        assert sharpe is not None
        assert sharpe > 0  # プラスのリターンなのでSharpe > 0
        assert not np.isnan(sharpe)
        assert not np.isinf(sharpe)
    
    def test_calculate_sharpe_ratio_insufficient_data(self):
        """Sharpe Ratio: データ不足時はNoneに相当する0.0"""
        reporter = BacktestReporter()
        
        # 1日未満のデータ
        for _ in range(100):
            reporter.portfolio_history.append(1000000.0)
        
        reporter.finalize_stats()
        
        # 1日未満なのでNone → 0.0
        assert reporter.stats["sharpe_ratio"] == 0.0
    
    def test_calculate_sharpe_ratio_zero_std(self):
        """Sharpe Ratio: 標準偏差ゼロ時はNoneに相当する0.0"""
        reporter = BacktestReporter()
        
        # 3日分のフラットデータ
        for _ in range(1440 * 3):
            reporter.portfolio_history.append(1000000.0)
        
        reporter.finalize_stats()
        
        # std=0 → None → 0.0
        assert reporter.stats["sharpe_ratio"] == 0.0
    
    def test_expectancy_calculation(self):
        """Expectancy計算の検証"""
        reporter = BacktestReporter()
        
        # 勝ち: 100, 200 (avg=150)
        reporter.record_trade(0.0, 1.0, 100.0, 50000.0, 50100.0, 1.0, 10.0, 5.0)
        reporter.record_trade(1.0, 0.0, 200.0, 50100.0, 50300.0, 1.0, 10.0, 5.0)
        
        # 負け: -50, -100 (avg=75)
        reporter.record_trade(0.0, -1.0, -50.0, 50000.0, 49950.0, 1.0, 10.0, 5.0)
        reporter.record_trade(-1.0, 0.0, -100.0, 49950.0, 49850.0, 1.0, 10.0, 5.0)
        
        reporter.finalize_stats()
        
        # Expectancy = avg_win * win_rate - avg_loss * loss_rate
        # = 150 * 0.5 - 75 * 0.5 = 75 - 37.5 = 37.5
        expected_expectancy = 150.0 * 0.5 - 75.0 * 0.5
        assert abs(reporter.stats["expectancy"] - expected_expectancy) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
