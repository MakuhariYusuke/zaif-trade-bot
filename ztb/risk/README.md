# Risk Management Module

This module provides comprehensive risk controls for trading operations, including position limits, loss management, and trade validation.

## Features

- **Risk Profiles**: Conservative, balanced, and aggressive presets
- **Pre-trade Validation**: Position size and loss limit checks
- **Real-time Monitoring**: Drawdown and volatility tracking
- **Stop Loss Automation**: Trailing stops and take profit rules
- **Trade Frequency Control**: Rate limiting and cooldown periods

## Quick Start

### Using Risk Manager

```python
from ztb.risk import RiskManager

# Create risk manager with balanced profile
risk_mgr = RiskManager(profile_name="balanced")

# Validate trade before execution
allowed, result, message = risk_mgr.validate_and_execute_trade(
    trade_func=my_broker.place_order,
    trade_notional=50000,
    position_notional=100000,
    peak_value=105000,
    symbol="BTC_JPY",
    side="buy",
    quantity=0.001
)
```

### CLI Integration

```bash
# Backtest with risk controls
python -m ztb.backtest.runner --policy rl --risk-profile conservative

# Paper trading with risk management
python -m ztb.live.paper_trader --mode replay --risk-profile balanced
```

## Risk Profiles

### Conservative Profile

- Max position: ¥50,000
- Max single trade: 2% of capital
- Daily loss limit: 2%
- Max drawdown: 5%
- Trades per hour: 2
- Minimum interval: 30 minutes

### Balanced Profile (Default)

- Max position: ¥100,000
- Max single trade: 5% of capital
- Daily loss limit: 5%
- Max drawdown: 10%
- Trades per hour: 5
- Minimum interval: 10 minutes

### Aggressive Profile

- Max position: ¥200,000
- Max single trade: 10% of capital
- Daily loss limit: 10%
- Max drawdown: 20%
- Trades per hour: 10
- Minimum interval: 5 minutes

## Risk Rules

### Position Limits

- **Notional Limits**: Maximum position size in JPY
- **Percentage Limits**: Maximum % of portfolio per position
- **Concentration**: Single asset exposure limits

### Loss Management

- **Daily Loss Limits**: Automatic stop after percentage loss
- **Drawdown Control**: Pause trading during large declines
- **Volatility Checks**: Reject trades in high volatility

### Trade Controls

- **Frequency Limits**: Maximum trades per time period
- **Size Limits**: Maximum trade size as % of capital
- **Cooldown Periods**: Waiting time after losses

### Stop Loss Rules

- **Trailing Stops**: Dynamic stop levels following profits
- **Take Profit**: Automatic profit taking at targets
- **Hard Stops**: Fixed percentage stop losses

## Integration Points

### Backtesting

- Pre-trade validation for each simulated order
- Daily loss limit enforcement
- Position size restrictions

### Paper Trading

- Real-time risk monitoring
- Automatic stop execution
- Trade frequency throttling

### Live Trading (Future)

- Same validation as paper trading
- Real-time position monitoring
- Emergency stop capabilities

## Configuration

### Environment Variables

```bash
# Risk profile selection
ZTB_RISK_PROFILE=balanced

# Custom limits
ZTB_MAX_POSITION_NOTIONAL=100000
ZTB_DAILY_LOSS_LIMIT_PCT=0.05
ZTB_MAX_TRADES_PER_HOUR=5
```

### Programmatic Configuration

```python
from ztb.risk.profiles import create_custom_risk_profile

custom_limits = create_custom_risk_profile(
    max_position_notional=75000,
    daily_loss_limit_pct=0.03,
    max_trades_per_hour=3
)

risk_mgr = RiskManager()
risk_mgr.checker = RiskChecker(custom_limits)
```

## Monitoring

### Risk Status

```python
status = risk_mgr.get_status_report()
print(f"Daily Loss: {status['current_status']['daily_loss']:.2%}")
print(f"Trades This Hour: {status['current_status']['trades_this_hour']}")
```

### Alerts and Callbacks

```python
def on_risk_violation(message: str):
    print(f"RISK ALERT: {message}")
    # Send notification, pause trading, etc.

risk_mgr.on_risk_violation = on_risk_violation
```

## Safety Features

- **Fail-Safe Defaults**: Conservative limits if configuration fails
- **Graceful Degradation**: Continue with warnings if monitoring fails
- **Audit Logging**: All risk decisions logged for review
- **Emergency Stops**: Manual override capabilities

## Performance Impact

- **Minimal Latency**: Risk checks add <1ms to trade decisions
- **Memory Efficient**: Lightweight state tracking
- **Configurable Overhead**: Can disable checks for high-frequency strategies

## Testing

Risk rules include comprehensive validation:

```python
# Test risk rule edge cases
from ztb.risk.rules import RiskRuleEngine
from ztb.risk.profiles import get_risk_profile

engine = RiskRuleEngine(get_risk_profile("conservative"))

# Test various scenarios
assert engine.check_daily_loss_limit()[0]  # Should pass initially
assert not engine.check_trade_frequency()[0]  # Should fail if over limit
```
