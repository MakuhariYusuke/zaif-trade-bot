#!/usr/bin/env python3
"""
Demo script for Advanced Auto-Stop System.

Shows how the auto-stop system works with simulated market data and trades.
"""

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from ztb.risk.advanced_auto_stop import create_production_auto_stop

if TYPE_CHECKING:
    from ztb.risk.advanced_auto_stop import AdvancedAutoStop


def simulate_market_data(auto_stop: Any, hours: int = 2) -> bool:
    """Simulate market data updates."""
    print("📊 Simulating market data updates...")

    base_price = 1000000.0
    current_time = datetime.now()

    for i in range(hours * 60):  # 1 update per minute
        # Simulate price movement with some volatility
        price_change = 0.001 * (2 * (i % 2) - 1)  # Alternating small changes
        price = base_price * (1 + price_change)

        auto_stop.update_market_data(current_time, price)
        current_time += timedelta(minutes=1)

        # Check stop conditions every 10 minutes
        if i % 10 == 0:
            should_stop, reason, message = auto_stop.check_stop_conditions()
            if should_stop:
                print(f"🚨 STOP TRIGGERED: {reason.value} - {message}")
                return True

    print("✅ Market simulation completed without stops")
    return False


def simulate_trades(auto_stop: Any, num_trades: int = 10) -> bool:
    """Simulate trade results."""
    print("💰 Simulating trade results...")

    for i in range(num_trades):
        # Mix of winning and losing trades
        if i < 3:  # First 3 trades are losses
            pnl = -1000.0 * (i + 1)  # Increasing losses
        elif i < 6:  # Next 3 trades are wins
            pnl = 1500.0 * (i - 2)  # Increasing wins
        else:  # Last trades vary
            pnl = 500.0 if i % 2 == 0 else -800.0

        trade_info = {
            "action": 1 if pnl > 0 else 2,
            "entry_price": 1000000.0,
            "exit_price": 1000000.0 + pnl,
            "position": 1,
            "timestamp": datetime.now(),
        }

        auto_stop.update_trade_result(pnl, trade_info)

        # Check stop conditions after each trade
        should_stop, reason, message = auto_stop.check_stop_conditions()
        if should_stop:
            print(f"🚨 STOP TRIGGERED after trade {i+1}: {reason.value} - {message}")
            return True

        print(
            f"Trade {i+1}: PnL = {pnl:.1f} JPY, Consecutive Losses = {auto_stop.consecutive_losses}"
        )

    print("✅ Trade simulation completed without stops")
    return False


def demonstrate_auto_stop() -> None:
    """Demonstrate the auto-stop system functionality."""
    print("🚀 Advanced Auto-Stop System Demo")
    print("=" * 50)

    # Create production auto-stop system
    auto_stop = create_production_auto_stop()
    print("✅ Auto-stop system initialized")

    # Show initial status
    status = auto_stop.get_status()
    print(
        f"📈 Initial Status: Active={status['is_active']}, Drawdown={status['current_drawdown']:.2%}"
    )

    # Test 1: Normal market conditions
    print("\n🧪 Test 1: Normal market conditions")
    stopped = simulate_market_data(auto_stop, hours=1)
    if not stopped:
        print("✅ No stops triggered under normal conditions")

    # Test 2: High volatility scenario
    print("\n🧪 Test 2: High volatility scenario")
    # Manually set high volatility to trigger stop
    auto_stop.volatility = 0.04  # 4% volatility
    should_stop, _, message = auto_stop.check_stop_conditions()
    if should_stop:
        print(f"🚨 Volatility stop triggered: {message}")
    else:
        print("❌ Expected volatility stop was not triggered")

    # Reset for next test
    auto_stop.resume_trading()

    # Test 3: Trade simulation
    print("\n🧪 Test 3: Trade simulation")
    stopped = simulate_trades(auto_stop, num_trades=8)
    if not stopped:
        print("✅ No stops triggered during trading")

    # Test 4: Consecutive losses scenario
    print("\n🧪 Test 4: Consecutive losses scenario")
    # Create a fresh auto-stop system for this test
    fresh_auto_stop = create_production_auto_stop()
    fresh_auto_stop.consecutive_losses = 3  # Exactly at threshold of 3
    should_stop, _, message = fresh_auto_stop.check_stop_conditions()
    if should_stop:
        print(f"🚨 Consecutive losses stop triggered: {message}")
    else:
        print("❌ Expected consecutive losses stop was not triggered")
        print(f"   Current consecutive losses: {fresh_auto_stop.consecutive_losses}")
        print(f"   Threshold: 3")

    # Final status
    print("\n📊 Final System Status:")
    status = auto_stop.get_status()
    print(f"  Active: {status['is_active']}")
    print(f"  Stop Reason: {status['stop_reason'] or 'None'}")
    print(f"  Current Drawdown: {status['current_drawdown']:.2%}")
    print(f"  Volatility: {status['volatility']:.2%}")
    print(f"  Consecutive Losses: {status['consecutive_losses']}")
    print(f"  Total Trades: {status['total_trades']}")

    print("\n🎉 Demo completed!")


if __name__ == "__main__":
    demonstrate_auto_stop()
