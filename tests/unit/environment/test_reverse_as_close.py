#!/usr/bin/env python3
"""
Unit tests for Reverse-as-Close functionality.

Tests verify that:
1. allow_reverse=True behaves as before (default, backward compatible)
2. allow_reverse=False prevents immediate reversal
3. Position transitions are correct
4. PnL and transaction costs are as expected
"""

import numpy as np
import pandas as pd
import pytest

from ztb.trading.environment.environment import HeavyTradingEnv, EnvironmentConfig


class TestReverseAsClose:
    """Test suite for allow_reverse flag."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "open": [99.5, 100.5, 101.5, 102.5, 103.5, 104.5],
            "high": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [1000, 1100, 1200, 1300, 1400, 1500],
            "sma_5": [100.0] * 6,
            "ema_12": [100.0] * 6,
            "rsi_14": [50.0] * 6,
            "atr_14": [1.0] * 6
        })
    
    def test_allow_reverse_true_default(self, sample_df):
        """Test default behavior: allow_reverse=True."""
        config = EnvironmentConfig(
            allow_reverse=True,
            transaction_cost=0.001,
            max_position_size=1.0,
            curriculum_stage="full"
        )
        
        env = HeavyTradingEnv(df=sample_df, config=config)
        obs, info = env.reset()
        
        # Initial: position=0 (Flat)
        assert env.position == 0.0
        
        # Step 1: BUY → Long
        obs, reward, done, truncated, info = env.step(1)  # BUY
        assert env.position == 1.0, "BUY from Flat should open Long"
        
        # Step 2: SELL → Close Long + Open Short (immediate reversal)
        obs, reward, done, truncated, info = env.step(2)  # SELL
        assert env.position == -1.0, "SELL from Long should reverse to Short (allow_reverse=True)"
    
    def test_allow_reverse_false_no_reversal(self, sample_df):
        """Test reverse禁止モード: allow_reverse=False."""
        config = EnvironmentConfig(
            allow_reverse=False,
            transaction_cost=0.001,
            max_position_size=1.0,
            curriculum_stage="full"
        )
        
        env = HeavyTradingEnv(df=sample_df, config=config)
        obs, info = env.reset()
        
        # Initial: position=0 (Flat)
        assert env.position == 0.0
        
        # Step 1: BUY → Long
        obs, reward, done, truncated, info = env.step(1)  # BUY
        assert env.position == 1.0, "BUY from Flat should open Long"
        
        # Step 2: SELL → Close Long ONLY (no reversal to Short)
        obs, reward, done, truncated, info = env.step(2)  # SELL
        assert env.position == 0.0, "SELL from Long should close to Flat (allow_reverse=False)"
    
    def test_allow_reverse_false_short_to_flat(self, sample_df):
        """Test Short→BUY→Flat (no reversal)."""
        config = EnvironmentConfig(
            allow_reverse=False,
            transaction_cost=0.001,
            max_position_size=1.0,
            curriculum_stage="full"
        )
        
        env = HeavyTradingEnv(df=sample_df, config=config)
        obs, info = env.reset()
        
        # Step 1: SELL → Short
        obs, reward, done, truncated, info = env.step(2)  # SELL
        assert env.position == -1.0, "SELL from Flat should open Short"
        
        # Step 2: BUY → Close Short ONLY (no reversal to Long)
        obs, reward, done, truncated, info = env.step(1)  # BUY
        assert env.position == 0.0, "BUY from Short should close to Flat (allow_reverse=False)"
    
    def test_flat_to_long_short_unaffected(self, sample_df):
        """Test that Flat→Long/Short is unaffected by allow_reverse."""
        config_true = EnvironmentConfig(allow_reverse=True, curriculum_stage="full")
        config_false = EnvironmentConfig(allow_reverse=False, curriculum_stage="full")
        
        # Test with allow_reverse=True
        env_true = HeavyTradingEnv(df=sample_df, config=config_true)
        env_true.reset()
        env_true.step(1)  # BUY
        assert env_true.position == 1.0
        
        # Test with allow_reverse=False
        env_false = HeavyTradingEnv(df=sample_df, config=config_false)
        env_false.reset()
        env_false.step(1)  # BUY
        assert env_false.position == 1.0
        
        # Both should be identical
        assert env_true.position == env_false.position
    
    def test_transaction_cost_count(self, sample_df):
        """Test that allow_reverse=False reduces transaction costs."""
        config_true = EnvironmentConfig(
            allow_reverse=True,
            transaction_cost=0.01,  # 1% fee
            curriculum_stage="full"
        )
        config_false = EnvironmentConfig(
            allow_reverse=False,
            transaction_cost=0.01,  # 1% fee
            curriculum_stage="full"
        )
        
        # Scenario: Flat→BUY→SELL
        # allow_reverse=True: 3 trades (BUY open, SELL close, SELL open)
        # allow_reverse=False: 2 trades (BUY open, SELL close)
        
        env_true = HeavyTradingEnv(df=sample_df, config=config_true)
        env_true.reset()
        env_true.step(1)  # BUY
        env_true.step(2)  # SELL
        trades_true = env_true.trades_count
        
        env_false = HeavyTradingEnv(df=sample_df, config=config_false)
        env_false.reset()
        env_false.step(1)  # BUY
        env_false.step(2)  # SELL
        trades_false = env_false.trades_count
        
        assert trades_true > trades_false, \
            f"allow_reverse=True should have more trades ({trades_true} vs {trades_false})"
    
    def test_position_transitions_detailed(self, sample_df):
        """Test detailed position transitions with both allow_reverse modes."""
        # Test allow_reverse=False: Long→SELL→Flat→SELL→Short
        config_false = EnvironmentConfig(allow_reverse=False, curriculum_stage="full")
        env = HeavyTradingEnv(df=sample_df, config=config_false)
        env.reset()
        
        # Flat → BUY → Long
        env.step(1)
        assert env.position == 1.0, "Should be Long"
        
        # Long → SELL → Flat (no reversal)
        env.step(2)
        assert env.position == 0.0, "Should be Flat (allow_reverse=False)"
        
        # Flat → SELL → Short (normal open)
        env.step(2)
        assert env.position == -1.0, "Should be Short from Flat"
        
        # Short → BUY → Flat (no reversal)
        env.step(1)
        assert env.position == 0.0, "Should be Flat (allow_reverse=False)"
    
    def test_config_from_dict_allow_reverse(self):
        """Test that allow_reverse is correctly parsed from dict."""
        # Test default (True)
        config_default = EnvironmentConfig.from_dict({})
        assert config_default.allow_reverse is True
        
        # Test explicit True
        config_true = EnvironmentConfig.from_dict({"allow_reverse": True})
        assert config_true.allow_reverse is True
        
        # Test explicit False
        config_false = EnvironmentConfig.from_dict({"allow_reverse": False})
        assert config_false.allow_reverse is False
        
        # Test string conversion
        config_str_true = EnvironmentConfig.from_dict({"allow_reverse": "true"})
        assert config_str_true.allow_reverse is True
        
        config_str_false = EnvironmentConfig.from_dict({"allow_reverse": "false"})
        assert config_str_false.allow_reverse is False
    
    def test_backward_compatibility(self, sample_df):
        """Test that existing code without allow_reverse still works."""
        # Old code that doesn't specify allow_reverse
        config = EnvironmentConfig(
            transaction_cost=0.001,
            curriculum_stage="full"
        )
        
        env = HeavyTradingEnv(df=sample_df, config=config)
        obs, info = env.reset()
        
        # Should default to allow_reverse=True (backward compatible)
        assert config.allow_reverse is True
        
        # Test reversal behavior
        env.step(1)  # BUY
        env.step(2)  # SELL
        assert env.position == -1.0, "Should allow reversal by default"


def test_reverse_as_close_integration():
    """Integration test for reverse-as-close functionality."""
    print("\n=== Reverse-as-Close Integration Test ===")
    
    # Create test data
    df = pd.DataFrame({
        "close": [100.0 + i for i in range(10)],
        "open": [99.5 + i for i in range(10)],
        "high": [100.5 + i for i in range(10)],
        "low": [99.0 + i for i in range(10)],
        "volume": [1000 + i * 100 for i in range(10)],
        "sma_5": [100.0] * 10,
        "ema_12": [100.0] * 10,
        "rsi_14": [50.0] * 10,
        "atr_14": [1.0] * 10
    })
    
    # Test scenario: Long→SELL→SELL
    print("\nScenario: Long → SELL → SELL")
    
    # allow_reverse=True
    print("\n1. allow_reverse=True:")
    config_true = EnvironmentConfig(allow_reverse=True, curriculum_stage="full")
    env_true = HeavyTradingEnv(df=df, config=config_true)
    env_true.reset()
    
    env_true.step(1)  # BUY → Long
    print(f"   After BUY: position={env_true.position}")
    
    env_true.step(2)  # SELL → Close + Short
    print(f"   After SELL: position={env_true.position}")
    
    env_true.step(2)  # SELL → No change (already Short)
    print(f"   After SELL again: position={env_true.position}")
    
    # allow_reverse=False
    print("\n2. allow_reverse=False:")
    config_false = EnvironmentConfig(allow_reverse=False, curriculum_stage="full")
    env_false = HeavyTradingEnv(df=df, config=config_false)
    env_false.reset()
    
    env_false.step(1)  # BUY → Long
    print(f"   After BUY: position={env_false.position}")
    
    env_false.step(2)  # SELL → Close only (Flat)
    print(f"   After SELL: position={env_false.position}")
    
    env_false.step(2)  # SELL → Short (from Flat)
    print(f"   After SELL again: position={env_false.position}")
    
    print("\n✅ Integration test complete!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    test_reverse_as_close_integration()
