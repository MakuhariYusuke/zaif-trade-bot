# Live Trading Module

This module provides paper trading capabilities and broker interfaces for realistic trading simulation without real exchange connectivity.

## Features

- **SimBroker**: Realistic trading simulation with slippage and commissions
- **Paper Trading**: Risk-free strategy testing with live-like conditions
- **Broker Interfaces**: Abstracted exchange connectivity for easy swapping
- **Dual Modes**: Replay historical data or connect to live streaming

## Quick Start

### Paper Trading Replay

```bash
python -m ztb.live.paper_trader --mode replay --policy sma_fast_slow --duration-minutes 60
```

### Live-Lite Paper Trading

```bash
python -m ztb.live.paper_trader --mode live-lite --policy rl --streaming-config config/streaming.yaml
```

## Trading Modes

### Replay Mode

- Consumes historical data at controlled pace
- Deterministic results for testing
- Configurable duration and speed
- Perfect for strategy validation

### Live-Lite Mode

- Connects to streaming data pipeline
- Real-time simulation without exchange orders
- Risk management integration
- Pre-production testing

## Output Files

Paper trading generates artifacts in `results/paper/<timestamp>/`:

- `pnl.csv`: Time series of P&L and balance
- `trade_log.json`: Individual trade records
- `summary.json`: Session summary statistics

## SimBroker Features

### Realistic Simulation

- **Slippage**: Configurable basis point impact on execution
- **Commissions**: Exchange fee simulation
- **Market Impact**: Position size effects on pricing
- **Latency**: Network delay simulation

### Account Management

- **Balance Tracking**: JPY and BTC balances
- **Position Management**: Real-time position updates
- **P&L Calculation**: Mark-to-market valuations
- **Order Lifecycle**: Complete order state management

## Broker Interfaces

### IBroker Protocol

All brokers implement the same interface:

```python
class IBroker(ABC):
    async def place_order(self, symbol, side, quantity, price=None, order_type='market')
    async def cancel_order(self, order_id)
    async def get_positions(self)
    async def get_balance(self)
    async def get_current_price(self, symbol)
```

### SimBroker Implementation

- Immediate order execution
- Perfect liquidity assumptions
- Configurable trading costs
- Deterministic for testing

### ZaifAdapter (Stub)

- Interface defined for future implementation
- Raises `NotImplementedError` for safety
- Same interface as SimBroker
- Easy production swap

## Risk Integration

Paper trading integrates with risk management:

- **Pre-trade Validation**: Position limits and loss checks
- **Real-time Monitoring**: Volatility and drawdown tracking
- **Automatic Stops**: Trailing stops and take profit
- **Trade Frequency**: Rate limiting and cooldowns

## Configuration Options

- `--mode`: `replay` or `live-lite`
- `--policy`: Trading strategy (`rl`, `sma_fast_slow`, `buy_hold`)
- `--duration-minutes`: Replay session length
- `--initial-balance`: Starting JPY balance
- `--risk-profile`: Risk management profile

## Integration Points

### With Backtesting

- Same strategy adapters
- Consistent metrics calculation
- Unified reporting format

### With Risk Management

- Pre-trade validation hooks
- Real-time position monitoring
- Stop loss automation

### With Data Pipeline

- Streaming data consumption
- Historical replay support
- Feature computation integration

## Example Output

```json
{
  "trades_executed": 12,
  "total_pnl": 1250.50,
  "duration_minutes": 60,
  "final_balance_jpy": 11250.50,
  "final_balance_btc": 0.001
}
```

## Safety Features

- **No Real Orders**: Never submits to actual exchanges
- **Balance Protection**: Prevents negative balances
- **Error Handling**: Graceful failure recovery
- **Logging**: Comprehensive audit trails

## Order State Machine

The live trading module includes a comprehensive order state machine for reliable trade execution and idempotency guarantees.

### Order States

Orders progress through the following states:

1. **CREATED**: Order initialized but not submitted
2. **SUBMITTED**: Order sent to broker
3. **ACCEPTED**: Order acknowledged by exchange
4. **PARTIAL_FILL**: Order partially executed
5. **FILLED**: Order completely executed
6. **CANCELLED**: Order cancelled by user/system
7. **REJECTED**: Order rejected by exchange
8. **EXPIRED**: Order expired
9. **FAILED**: Order failed due to error

### Idempotency Rules

- **Client Order ID Generation**: Automatic generation using `symbol_side_quantity_price_timestamp_uuid`
- **Duplicate Detection**: Prevents duplicate orders within same session
- **State Validation**: Ensures valid state transitions only
- **Local Ledger**: Maintains order history for reconciliation

### Usage Example

```python
from ztb.live.order_state import get_order_state_machine, OrderData

# Create order with automatic idempotency
order_data = OrderData(
    order_id="ord_123",
    symbol="BTC_JPY",
    side="buy",
    quantity=0.001,
    price=3000000.0
)

state_machine = get_order_state_machine()
order_record = state_machine.create_order(order_data)

# Transition states
state_machine.transition_order("ord_123", "submit")
state_machine.transition_order("ord_123", "accept")
state_machine.transition_order("ord_123", "fill", filled_quantity=0.001)
```
