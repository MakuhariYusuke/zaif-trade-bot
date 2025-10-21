#!/usr/bin/env python3
"""
SAC v431 Backtest Simulation
Simulates backtesting for SAC v431 with market-adaptive rewards
"""

import json
import numpy as np
import time
from pathlib import Path

def load_config():
    """Load v431 configuration"""
    config_path = Path(__file__).parent.parent / "configs" / "v431" / "sac_v431_1_enhanced.json"
    with open(config_path, 'r') as f:
        return json.load(f)

def simulate_market_data(num_steps=10000):
    """Generate synthetic market data for backtesting"""
    np.random.seed(42)  # For reproducible results

    # Generate price data with trends and volatility
    prices = [100.0]  # Starting price

    for i in range(num_steps - 1):
        # Market regime changes
        regime = np.random.choice(['bull', 'bear', 'sideways'], p=[0.3, 0.3, 0.4])

        if regime == 'bull':
            change = np.random.normal(0.001, 0.02)  # Slight upward bias
        elif regime == 'bear':
            change = np.random.normal(-0.001, 0.02)  # Slight downward bias
        else:  # sideways
            change = np.random.normal(0.0, 0.015)  # No bias, higher volatility

        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.1))  # Prevent negative prices

    return np.array(prices)

def simulate_sac_v431_action(price, prev_price, position, config):
    """Simulate SAC v431 action based on market conditions and position"""
    # Calculate price change
    price_change = (price - prev_price) / prev_price

    # Market condition detection
    volatility = np.random.uniform(0.01, 0.05)  # Simulated volatility
    trend_strength = abs(price_change)

    if volatility > 0.03:
        market_condition = 'high_vol'
    elif trend_strength < 0.005:
        market_condition = 'sideways'
    elif price_change > 0.01:
        market_condition = 'bull'
    elif price_change < -0.01:
        market_condition = 'bear'
    else:
        market_condition = 'neutral'

    # Get reward parameters
    rewards = {
        'sell': config['reward_function']['sell_bonus'],
        'hold': config['reward_function']['hold_bonus'],
        'buy': config['reward_function']['buy_bonus']
    }

    # Apply market multipliers
    market_multipliers = config['reward_function']['market_adaptive']
    if market_condition == 'sideways':
        multiplier = market_multipliers.get('sideways_multiplier', 1.0)
    elif market_condition == 'high_vol':
        multiplier = market_multipliers.get('high_vol_multiplier', 1.0)
    else:
        multiplier = 1.0

    # Adjust rewards based on market condition
    adjusted_rewards = {k: v * multiplier for k, v in rewards.items()}

    # Position-based decision making (SAC v431 logic)
    if position > 0.5:  # Long position - consider selling
        if market_condition in ['bear', 'high_vol']:
            action_prob = [0.6, 0.3, 0.1]  # Favor SELL
        else:
            action_prob = [0.2, 0.6, 0.2]  # Favor HOLD
    elif position < -0.5:  # Short position - consider covering
        if market_condition in ['bull', 'high_vol']:
            action_prob = [0.1, 0.3, 0.6]  # Favor BUY
        else:
            action_prob = [0.2, 0.6, 0.2]  # Favor HOLD
    else:  # Flat position
        if market_condition == 'bull':
            action_prob = [0.1, 0.2, 0.7]  # Favor BUY
        elif market_condition == 'bear':
            action_prob = [0.7, 0.2, 0.1]  # Favor SELL
        else:
            action_prob = [0.3, 0.4, 0.3]  # Balanced

    # Add some randomness (exploration)
    action_prob = np.array(action_prob)
    action_prob += np.random.normal(0, 0.1, 3)  # Add noise
    action_prob = np.clip(action_prob, 0.01, 0.99)
    action_prob = action_prob / np.sum(action_prob)

    # Choose action
    actions = ['SELL', 'HOLD', 'BUY']
    action = np.random.choice(actions, p=action_prob)

    return action, adjusted_rewards[action]

def run_backtest_simulation(config, num_steps=10000, initial_capital=10000.0):
    """Run backtest simulation"""
    print("=== SAC v431 Backtest Simulation ===")
    print(f"Steps: {num_steps}, Initial Capital: ${initial_capital}")

    # Generate market data
    prices = simulate_market_data(num_steps)
    print(f"Generated {len(prices)} price data points")

    # Initialize trading variables
    capital = initial_capital
    position = 0.0  # -1 (short) to 1 (long)
    entry_price = 0.0
    trades = []
    transaction_cost = 0.0005  # 0.05%

    actions_count = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    total_reward = 0.0

    print("\nStarting simulation...")

    for step in range(1, num_steps):
        current_price = prices[step]
        prev_price = prices[step - 1]

        # Get SAC v431 action
        action, reward = simulate_sac_v431_action(current_price, prev_price, position, config)
        actions_count[action] += 1
        total_reward += reward

        # Execute trade logic
        if action == 'BUY':
            if position <= 0:  # Open long or cover short
                if position < 0:  # Cover short
                    # Calculate P&L for closing short
                    pnl = (entry_price - current_price) * abs(position) * capital
                    pnl *= (1 - transaction_cost)  # Transaction cost
                    capital += pnl
                    trades.append({
                        'type': 'COVER_SHORT',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'step': step
                    })

                # Open long position
                position = 1.0
                entry_price = current_price
                capital *= (1 - transaction_cost)  # Transaction cost

        elif action == 'SELL':
            if position >= 0:  # Open short or close long
                if position > 0:  # Close long
                    # Calculate P&L for closing long
                    pnl = (current_price - entry_price) * position * capital
                    pnl *= (1 - transaction_cost)  # Transaction cost
                    capital += pnl
                    trades.append({
                        'type': 'CLOSE_LONG',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'step': step
                    })

                # Open short position
                position = -1.0
                entry_price = current_price
                capital *= (1 - transaction_cost)  # Transaction cost

        # HOLD action - no position change

        if step % 1000 == 0:
            progress = step / num_steps * 100
            print(".1f")

    # Close any open position at the end
    if position != 0:
        final_price = prices[-1]
        if position > 0:  # Close long
            pnl = (final_price - entry_price) * position * capital
            pnl *= (1 - transaction_cost)
            capital += pnl
            trades.append({
                'type': 'CLOSE_LONG',
                'entry_price': entry_price,
                'exit_price': final_price,
                'pnl': pnl,
                'step': num_steps
            })
        else:  # Cover short
            pnl = (entry_price - final_price) * abs(position) * capital
            pnl *= (1 - transaction_cost)
            capital += pnl
            trades.append({
                'type': 'COVER_SHORT',
                'entry_price': entry_price,
                'exit_price': final_price,
                'pnl': pnl,
                'step': num_steps
            })

    # Calculate metrics
    total_return = (capital - initial_capital) / initial_capital * 100
    num_trades = len(trades)
    winning_trades = len([t for t in trades if t['pnl'] > 0])
    win_rate = winning_trades / num_trades * 100 if num_trades > 0 else 0

    # Calculate Sharpe ratio (simplified)
    returns = [t['pnl'] for t in trades]
    if returns:
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0
    else:
        sharpe_ratio = 0

    # Calculate max drawdown
    cumulative_returns = np.cumsum([t['pnl'] for t in trades])
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = running_max - cumulative_returns
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

    print("\n=== Backtest Results ===")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Final Capital: ${capital:.2f}")
    print(f"Total Trades: {num_trades}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: ${max_drawdown:.2f}")

    print("\nAction Distribution:")
    total_actions = sum(actions_count.values())
    for action, count in actions_count.items():
        pct = count / total_actions * 100
        print(f"  {action}: {count} ({pct:.1f}%)")

    print("\nMarket-Adaptive Rewards:")
    avg_reward = total_reward / num_steps
    print(f"Average Reward per Step: {avg_reward:.4f}")
    print(f"Total Reward Points: {total_reward:.2f}")

    print("\n=== Performance Analysis ===")
    if win_rate > 50:
        print("✅ Strong performance with high win rate")
    elif win_rate > 40:
        print("⚠️ Moderate performance - needs improvement")
    else:
        print("❌ Poor performance - significant issues detected")

    if sharpe_ratio > 1.0:
        print("✅ Good risk-adjusted returns")
    elif sharpe_ratio > 0.5:
        print("⚠️ Moderate risk-adjusted returns")
    else:
        print("❌ Poor risk-adjusted returns")

    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_trades': num_trades,
        'final_capital': capital,
        'trades': trades
    }

def main():
    # Load configuration
    config = load_config()

    # Run backtest
    results = run_backtest_simulation(config, num_steps=10000, initial_capital=10000.0)

    print("
=== SAC v431 Backtest Complete ===")
    print("Ready for validation and further analysis")

if __name__ == "__main__":
    main()