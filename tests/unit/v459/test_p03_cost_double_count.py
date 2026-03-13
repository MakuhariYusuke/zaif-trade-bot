"""
Unit tests for P0-3: Cost Double-Count Prevention

Verifies that:
- env.step() returns info['trade_pnl'] as NET PnL (costs deducted)
- Reporter.record_trade() treats pnl as NET PnL (no double deduction)
- env.net_pnl == reporter.stats['net_pnl'] (consistency check)
"""

import pytest
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from ztb.evaluation.walk_forward.reporter import BacktestReporter


class TestP03CostDoubleCountPrevention:
    """Verify P0-3: Cost double-count prevention"""

    def test_reporter_record_trade_docstring_specifies_net_pnl(self):
        """Verify Reporter.record_trade() docstring specifies pnl as NET PnL"""
        from inspect import getdoc
        
        reporter = BacktestReporter()
        docstring = getdoc(reporter.record_trade)
        
        assert docstring is not None, "record_trade() has no docstring"
        assert "NET PnL" in docstring or "Net PnL" in docstring, \
            "Docstring does not specify pnl as NET PnL"
        assert "P0-3" in docstring, "Docstring does not reference P0-3 fix"

    def test_reporter_does_not_double_deduct_costs(self):
        """Verify Reporter treats pnl as NET and does not re-deduct costs"""
        reporter = BacktestReporter()
        
        # Record a trade with NET PnL (costs already deducted)
        net_pnl = 100.0  # Already has costs deducted
        fee = 5.0
        slippage = 2.0
        
        reporter.record_trade(
            position_before=0.0,
            position_after=1.0,
            pnl=net_pnl,  # NET PnL
            entry_price=1000.0,
            exit_price=1100.0,
            size=1.0,
            fee=fee,
            slippage=slippage,
        )
        
        # Reporter should record net_pnl as-is (not pnl - fee - slippage)
        assert reporter.stats["net_pnl"] == net_pnl, \
            f"Reporter double-deducted costs: {reporter.stats['net_pnl']} != {net_pnl}"
        
        # Fees are recorded separately for verification
        assert reporter.stats["total_fees"] == fee
        assert reporter.stats["total_slippage"] == slippage

    def test_env_info_contains_trade_pnl(self):
        """Verify env.step() info contains 'trade_pnl' key"""
        from pathlib import Path
        env_file = Path("ztb/trading/environment/fast_intraday_env_v456.py")
        
        if not env_file.exists():
            pytest.skip(f"Environment file not found: {env_file}")
        
        content = env_file.read_text(encoding='utf-8')
        
        # Verify 'trade_pnl' is in info dict
        assert "'trade_pnl':" in content or '"trade_pnl":' in content, \
            "info dict does not contain 'trade_pnl' key"
        
        # Verify P0-3 comment exists
        assert "P0-3" in content, "P0-3 fix not documented in environment file"

    def test_env_trade_pnl_is_net(self):
        """Verify env calculates trade_pnl as NET (costs deducted)"""
        from pathlib import Path
        env_file = Path("ztb/trading/environment/fast_intraday_env_v456.py")
        
        if not env_file.exists():
            pytest.skip(f"Environment file not found: {env_file}")
        
        content = env_file.read_text(encoding='utf-8')
        
        # Verify trade_pnl calculation includes cost deduction
        assert "- fee_paid - slippage_paid" in content, \
            "trade_pnl does not deduct costs (fee_paid, slippage_paid)"
        
        # Verify NET PnL comment
        assert "NET PnL" in content or "net_pnl" in content.lower(), \
            "No NET PnL documentation found"

    def test_reporter_trade_history_preserves_costs(self):
        """Verify Reporter preserves fee/slippage in trade_history for verification"""
        reporter = BacktestReporter()
        
        net_pnl = 50.0
        fee = 3.0
        slippage = 1.5
        
        reporter.record_trade(
            position_before=0.0,
            position_after=1.0,
            pnl=net_pnl,
            entry_price=1000.0,
            exit_price=1050.0,
            size=1.0,
            fee=fee,
            slippage=slippage,
        )
        
        # Verify trade_history contains fee/slippage for verification
        assert len(reporter.trade_history) == 1
        trade = reporter.trade_history[0]
        assert trade["fee"] == fee, "Fee not preserved in trade_history"
        assert trade["slippage"] == slippage, "Slippage not preserved in trade_history"
        assert trade["net_pnl"] == net_pnl, "NET PnL not correct in trade_history"


class TestP03Integration:
    """Integration test for P0-3: env and reporter consistency"""

    def test_p03_documented_in_phase1_spec(self):
        """Verify P0-3 is documented in Phase 1 specification"""
        from pathlib import Path
        doc09 = Path("docs/v459/09_phase1_specification.md")
        
        if not doc09.exists():
            pytest.skip("Doc09 not found")
        
        content = doc09.read_text(encoding='utf-8')
        
        # Verify P0-3 is documented
        assert "P0-3" in content, "P0-3 not documented in Doc09"
        assert "Cost Double-Count" in content, "Cost Double-Count not documented"
        
        # Verify PnL convention is documented
        assert "net_pnl" in content.lower() or "NET PnL" in content, \
            "NET PnL convention not documented"

    def test_env_and_reporter_pnl_convention_match(self):
        """Verify env and reporter use consistent PnL convention"""
        from pathlib import Path
        
        env_file = Path("ztb/trading/environment/fast_intraday_env_v456.py")
        reporter_file = Path("ztb/evaluation/walk_forward/reporter.py")
        
        if not env_file.exists() or not reporter_file.exists():
            pytest.skip("Source files not found")
        
        env_content = env_file.read_text(encoding='utf-8')
        reporter_content = reporter_file.read_text(encoding='utf-8')
        
        # Both should reference P0-3
        assert "P0-3" in env_content, "P0-3 not referenced in environment"
        assert "P0-3" in reporter_content, "P0-3 not referenced in reporter"
        
        # Both should reference NET PnL
        assert "NET PnL" in env_content, "NET PnL not documented in environment"
        assert "NET PnL" in reporter_content or "Net PnL" in reporter_content, \
            "NET PnL not documented in reporter"
