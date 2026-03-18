"""
Test script for Phase 3 Execution Model.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from ztb.trading.environment.components.position_manager import PositionManager
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.trading.execution.realistic import RealisticExecutionModel


def test_execution_model():
    print("Testing RealisticExecutionModel...")

    # Initialize model
    model = RealisticExecutionModel(
        base_slippage=0.001,
        atr_slippage_factor=1.0,
        base_latency_ms=50,
        latency_jitter_ms=10,
    )

    # Test simulation
    price = 10000.0
    atr = 100.0  # 1% volatility

    # Buy
    result = model.simulate_execution("buy", price, 1.0, current_atr=atr)
    print(
        f"Buy Request: {price}, Executed: {result.executed_price:.2f}, Slippage: {result.slippage_rate:.4f}, Latency: {result.latency_ms:.2f}ms"
    )

    expected_slippage = 0.001 + (100 / 10000 * 1.0)  # 0.001 + 0.01 = 0.011
    assert abs(result.slippage_rate - expected_slippage) < 1e-6
    assert result.executed_price > price

    # Sell
    result = model.simulate_execution("sell", price, 1.0, current_atr=atr)
    print(
        f"Sell Request: {price}, Executed: {result.executed_price:.2f}, Slippage: {result.slippage_rate:.4f}"
    )
    assert result.executed_price < price

    print("RealisticExecutionModel test passed!")


def test_position_manager_integration():
    print("\nTesting PositionManager integration...")

    config = EnvironmentConfig(initial_portfolio_value=100000.0, transaction_cost=0.001)

    model = RealisticExecutionModel(base_slippage=0.01)  # 1% slippage

    # Mock price callback
    current_price = 10000.0

    def get_price():
        return current_price

    pm = PositionManager(config, get_price, execution_model=model)

    # Open Long
    # Price 10000, Slippage 1% -> Exec Price 10100
    # Fee 0.1% of 10100 * size
    entry_cost = pm.open_position(1, 0, atr=0.0)

    print(f"Position: {pm.position}")
    print(f"Entry Price: {pm.entry_price}")
    print(f"Entry Cost: {entry_cost}")

    assert pm.position > 0
    assert pm.entry_price > 10000.0  # Should include slippage

    # Close Long
    # Price 10000, Slippage 1% -> Exec Price 9900
    pnl = pm.close_position(0, atr=0.0)
    print(f"Realized PnL: {pnl}")

    # Expected PnL: (9900 - 10100) * size - entry_cost - exit_cost
    # Roughly negative due to spread/slippage
    assert pnl < 0

    print("PositionManager integration test passed!")


if __name__ == "__main__":
    test_execution_model()
    test_position_manager_integration()
