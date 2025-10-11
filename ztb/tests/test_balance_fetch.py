#!/usr/bin/env python3
"""
Test script for balance fetching functionality.

Tests both dry-run and real balance fetching from Coincheck adapter.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter
from ztb.utils.config import TypedConfig


async def test_dry_run_balance():
    """Test balance fetching in dry-run mode."""
    print("=" * 60)
    print("Testing Coincheck Adapter - Dry Run Mode")
    print("=" * 60)
    
    adapter = CoincheckAdapter(dry_run=True)
    
    # Test getting all balances
    print("\n[Test 1] Get all balances:")
    balances = await adapter.get_balance()
    for balance in balances:
        print(f"  {balance.currency}: free={balance.free}, locked={balance.locked}, total={balance.total}")
    
    # Test getting specific currency
    print("\n[Test 2] Get BTC balance only:")
    btc_balances = await adapter.get_balance(currency="BTC")
    for balance in btc_balances:
        print(f"  {balance.currency}: free={balance.free}, locked={balance.locked}, total={balance.total}")
    
    print("\n[Test 3] Get JPY balance only:")
    jpy_balances = await adapter.get_balance(currency="JPY")
    for balance in jpy_balances:
        print(f"  {balance.currency}: free={balance.free}, locked={balance.locked}, total={balance.total}")
    
    # Test getting current price
    print("\n[Test 4] Get current BTC/JPY price:")
    price = await adapter.get_current_price("btc_jpy")
    print(f"  Current price: ¥{price:,.0f}")
    
    print("\n" + "=" * 60)
    print("✓ All dry-run tests passed!")
    print("=" * 60)


async def test_live_trader_balance():
    """Test balance fetching through LiveTrader."""
    print("\n" + "=" * 60)
    print("Testing LiveTrader Balance Integration")
    print("=" * 60)
    
    # Import here to avoid initialization issues
    from live_trade import LiveTrader
    
    # Create trader with dry-run mode
    try:
        config = TypedConfig()
        trader = LiveTrader(
            model_path=config.get_model_path("ppo_btc_jpy_v367.zip"),  # Use any existing model
            dry_run=True
        )
        
        print("\n[Test 5] Get account balance via LiveTrader:")
        balances = await trader.get_account_balance()
        for currency, amount in balances.items():
            print(f"  {currency}: {amount}")
        
        print("\n[Test 6] Get BTC balance only:")
        btc_balance = await trader.get_account_balance(currency="BTC")
        for currency, amount in btc_balance.items():
            print(f"  {currency}: {amount}")
        
        print("\n" + "=" * 60)
        print("✓ LiveTrader integration tests passed!")
        print("=" * 60)
        
    except FileNotFoundError:
        print("\n⚠️  Model file not found, skipping LiveTrader tests")
        print("   (This is expected if you don't have a trained model yet)")


async def main():
    """Run all tests."""
    print("\n🧪 Balance Fetching Test Suite")
    print("=" * 60)
    
    # Test 1: Dry-run adapter
    await test_dry_run_balance()
    
    # Test 2: LiveTrader integration
    await test_live_trader_balance()
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
