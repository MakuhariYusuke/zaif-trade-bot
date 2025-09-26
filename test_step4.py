#!/usr/bin/env python3
"""
Test script for Step 4: Virtual Trading Bridge
"""

from ztb.trading.bridge import VirtualTradingBridge

class MockTradingEnvironment:
    """Mock trading environment providing market prices"""

    def __init__(self, initial_price: float = 5000000.0):
        self.current_price = initial_price
        self.price_history = [initial_price]

    def get_market_price(self, symbol: str) -> float:
        return self.current_price

    def simulate_price_movement(self, change_percent: float):
        """Simulate price movement"""
        self.current_price *= (1 + change_percent)
        self.price_history.append(self.current_price)

def test_virtual_trading_bridge():
    print('=== Step 4 Dry-run: Virtual Trading Bridge ===')

    # Initialize bridge and mock environment
    bridge = VirtualTradingBridge(initial_balance=100000.0)  # 100k JPY
    env = MockTradingEnvironment(initial_price=5000000.0)  # 5M JPY per BTC

    # Override get_market_price
    bridge.get_market_price = env.get_market_price

    print(f'Initial balance: {bridge.get_balance():.2f} JPY')
    print(f'Initial price: {env.get_market_price("BTC/JPY"):,.0f} JPY')

    # Simulate trading signals
    signals = [
        ('buy', 0.001),   # Buy 0.001 BTC
        ('buy', 0.002),   # Buy 0.002 BTC
        ('sell', 0.001),  # Sell 0.001 BTC
        ('buy', 0.005),   # Buy 0.005 BTC
        ('sell', 0.003),  # Sell 0.003 BTC
    ]

    for i, (side, quantity) in enumerate(signals):
        # Simulate small price movement
        price_change = 0.001 if side == 'buy' else -0.001  # 0.1% movement
        env.simulate_price_movement(price_change)

        current_price = env.get_market_price("BTC/JPY")
        print(f'\nSignal {i+1}: {side.upper()} {quantity} BTC at {current_price:,.0f} JPY')

        # Place order
        order = bridge.place_market_order("BTC/JPY", side, quantity, current_price)

        print(f'Order ID: {order.order_id}')
        print(f'Status: {order.status}')
        print(f'Execution price: {order.filled_price:,.0f} JPY')
        print(f'Commission: {order.commission:.2f} JPY')
        print(f'Balance: {bridge.get_balance():.2f} JPY')
        print(f'Position: {bridge.get_position("BTC/JPY"):.4f} BTC')

    # Final summary
    print('\n=== Final Summary ===')
    print(f'Final balance: {bridge.get_balance():.2f} JPY')
    print(f'Final position: {bridge.get_position("BTC/JPY"):.4f} BTC')
    print(f'Position value: {bridge.get_position("BTC/JPY") * env.get_market_price("BTC/JPY"):,.0f} JPY')
    print(f'Total orders: {len(bridge.get_order_history())}')

    # Show order history
    print('\n=== Order History ===')
    for order in bridge.get_order_history():
        print(f'{order.timestamp} | {order.side.upper()} {order.quantity:.4f} BTC | '
              f'Price: {order.filled_price:,.0f} | Status: {order.status}')

    # Test safety limits
    print('\n=== Safety Test ===')
    # Try to sell more than position
    try:
        order = bridge.place_market_order("BTC/JPY", "sell", 1.0, env.get_market_price("BTC/JPY"))
        print(f'Oversell order status: {order.status}')
    except Exception as e:
        print(f'Oversell prevented: {e}')

    print('=== Step 4 Test Completed ===')

if __name__ == "__main__":
    test_virtual_trading_bridge()