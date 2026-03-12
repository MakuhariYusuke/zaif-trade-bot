"""
Phase 2 P1-3: Reporter統合のテスト

BacktestReporterのclose_reason対応を検証
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from ztb.evaluation.walk_forward.reporter import BacktestReporter


class TestReporterCloseReasonSupport:
    """P1-3: BacktestReporter close_reason対応テスト"""
    
    def test_record_trade_accepts_close_reason(self):
        """record_tradeメソッドがclose_reasonパラメータを受け付ける"""
        reporter = BacktestReporter()
        
        # close_reasonありで呼び出し
        reporter.record_trade(
            position_before=1.0,
            position_after=0.0,
            pnl=100.0,
            entry_price=1000.0,
            exit_price=1100.0,
            size=1.0,
            fee=1.0,
            slippage=0.5,
            timestamp=None,
            close_reason="tp",
        )
        
        # trade_historyに記録されている
        assert len(reporter.trade_history) == 1
        assert reporter.trade_history[0]["close_reason"] == "tp"
    
    def test_close_reason_recorded_for_long_close(self):
        """long_closeでclose_reasonが記録される"""
        reporter = BacktestReporter()
        
        # Long close with SL
        reporter.record_trade(
            position_before=0.8,
            position_after=0.0,
            pnl=-50.0,
            entry_price=1000.0,
            exit_price=950.0,
            size=0.8,
            fee=1.0,
            slippage=0.5,
            close_reason="sl",
        )
        
        assert len(reporter.trade_history) == 1
        assert reporter.trade_history[0]["type"] == "long_close"
        assert reporter.trade_history[0]["close_reason"] == "sl"
    
    def test_close_reason_recorded_for_short_close(self):
        """short_closeでclose_reasonが記録される"""
        reporter = BacktestReporter()
        
        # Short close with TP
        reporter.record_trade(
            position_before=-0.5,
            position_after=0.0,
            pnl=60.0,
            entry_price=1100.0,
            exit_price=1040.0,
            size=0.5,
            fee=1.0,
            slippage=0.5,
            close_reason="tp",
        )
        
        assert len(reporter.trade_history) == 1
        assert reporter.trade_history[0]["type"] == "short_close"
        assert reporter.trade_history[0]["close_reason"] == "tp"
    
    def test_close_reason_recorded_for_reversal(self):
        """reversal tradeでclose_reason="reversal"が記録される"""
        reporter = BacktestReporter()
        
        # Long→Short reversal
        reporter.record_trade(
            position_before=0.8,
            position_after=-0.8,
            pnl=50.0,
            entry_price=1000.0,
            exit_price=1100.0,
            size=1.6,
            fee=2.0,
            slippage=1.0,
            close_reason="reversal",
        )
        
        # reversal は2つに分解される（close + open）
        assert len(reporter.trade_history) == 2
        
        # 最初のclose tradeにclose_reason記録
        close_trade = reporter.trade_history[0]
        assert close_trade["type"] == "long_close"
        assert close_trade["close_reason"] == "reversal"
    
    def test_close_reason_not_recorded_for_open(self):
        """open tradeではclose_reasonが記録されない"""
        reporter = BacktestReporter()
        
        # Long open (close_reasonは無視される)
        reporter.record_trade(
            position_before=0.0,
            position_after=0.8,
            pnl=-2.0,  # エントリーコスト
            entry_price=1000.0,
            exit_price=1000.0,
            size=0.8,
            fee=1.5,
            slippage=0.5,
            close_reason="tp",  # 無視される
        )
        
        assert len(reporter.trade_history) == 1
        trade = reporter.trade_history[0]
        assert trade["type"] == "long_open"
        # close_reasonフィールドが存在しない（openには不要）
        assert "close_reason" not in trade
    
    def test_backward_compatibility_without_close_reason(self):
        """close_reasonなしでも動作する（後方互換性）"""
        reporter = BacktestReporter()
        
        # close_reasonなしで呼び出し
        reporter.record_trade(
            position_before=1.0,
            position_after=0.0,
            pnl=100.0,
            entry_price=1000.0,
            exit_price=1100.0,
            size=1.0,
            fee=1.0,
            slippage=0.5,
        )
        
        # 記録されるが、close_reasonフィールドなし
        assert len(reporter.trade_history) == 1
        assert "close_reason" not in reporter.trade_history[0]
    
    def test_multiple_close_reasons(self):
        """複数のclose_reasonが混在しても記録される"""
        reporter = BacktestReporter()
        
        # TP close
        reporter.record_trade(
            position_before=0.5, position_after=0.0, pnl=50.0,
            entry_price=1000.0, exit_price=1100.0, size=0.5,
            fee=1.0, slippage=0.5, close_reason="tp"
        )
        
        # SL close
        reporter.record_trade(
            position_before=-0.5, position_after=0.0, pnl=-30.0,
            entry_price=1100.0, exit_price=1150.0, size=0.5,
            fee=1.0, slippage=0.5, close_reason="sl"
        )
        
        # Manual close
        reporter.record_trade(
            position_before=0.3, position_after=0.0, pnl=10.0,
            entry_price=1000.0, exit_price=1030.0, size=0.3,
            fee=0.5, slippage=0.3, close_reason="manual"
        )
        
        assert len(reporter.trade_history) == 3
        assert reporter.trade_history[0]["close_reason"] == "tp"
        assert reporter.trade_history[1]["close_reason"] == "sl"
        assert reporter.trade_history[2]["close_reason"] == "manual"
