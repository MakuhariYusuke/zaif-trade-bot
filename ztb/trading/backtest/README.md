# Backtest Module

This module provides comprehensive backtesting capabilities for trading strategies, including performance metrics calculation and reporting.

## Features

- **Strategy Adapters**: RL policy, SMA crossover, buy & hold
- **Performance Metrics**: Sharpe ratio, Sortino ratio, max drawdown, CAGR, win rate
- **Comprehensive Reporting**: JSON metrics, CSV data, Markdown summaries
- **Deterministic Results**: Fixed seeds for reproducible testing

## Quick Start

### Run SMA Crossover Backtest

```bash
python -m ztb.backtest.runner --policy sma_fast_slow --dataset btc_usd_1m
```

### Run RL Policy Backtest

```bash
python -m ztb.backtest.runner --policy rl --dataset btc_usd_1m --slippage-bps 5
```

## Output Files

Backtests generate the following artifacts in `results/backtest/<timestamp>/`:

- `metrics.json`: Comprehensive performance metrics
- `report.md`: Executive summary
- `equity_curve.csv`: Portfolio value over time
- `orders.csv`: Individual trade records

## Metrics Explained

### Risk-Adjusted Returns

- **Sharpe Ratio**: Average return per unit of volatility (annualized)
- **Sortino Ratio**: Sharpe ratio using only downside volatility

### Returns

- **Total Return**: Overall percentage gain/loss
- **CAGR**: Compound Annual Growth Rate
- **Annualized Return**: Time-weighted annual return

### Risk Metrics

- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Standard deviation of returns (annualized)

### Trade Statistics

- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit divided by gross loss
- **Turnover**: Portfolio turnover rate

## Strategy Adapters

### RL Policy Adapter

- Wraps trained PPO models for inference
- Falls back to momentum strategy if no model available
- Configurable confidence thresholds

### SMA Crossover Adapter

- Fast/slow moving average crossover signals
- Configurable periods (default: 10/20)
- Includes trend confirmation

### Buy & Hold Adapter

- Passive benchmark strategy
- Enters at start, holds throughout

## Configuration Options

- `--policy`: Strategy to test (`rl`, `sma_fast_slow`, `buy_hold`)
- `--dataset`: Data source identifier
- `--slippage-bps`: Trading slippage in basis points
- `--initial-capital`: Starting portfolio value
- `--output-dir`: Results directory

## Integration

The backtest engine integrates with:

- **Risk Management**: Pre-trade validation and position limits
- **Data Pipeline**: Historical price feeds and features
- **Reporting**: Automated artifact generation

## Example Output

```json
{
  "sharpe_ratio": 1.23,
  "max_drawdown": -0.15,
  "total_return": 0.45,
  "win_rate": 0.62,
  "total_trades": 156
}
```
