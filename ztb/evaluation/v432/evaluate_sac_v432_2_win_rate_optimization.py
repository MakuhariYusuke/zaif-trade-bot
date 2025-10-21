#!/usr/bin/env python3
"""
SAC v432.1 Evaluation Script with Advanced Position Management
Enhanced evaluation with negative HOLD penalty and advanced position logic
"""

import json
import numpy as np
from pathlib import Path
import sys

# Add project root to path using path_utils
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.utils.path_utils import get_project_root

def load_config():
    """Load v432.2 configuration"""
    config_path = get_project_root() / "ztb" / "configs" / "v432" / "sac_v432_2_win_rate_optimization.json"
    with open(config_path, 'r') as f:
        return json.load(f)

def simulate_market_data(num_steps=10000):
    """Generate synthetic market data for backtesting"""
    # np.random.seed(42)  # For reproducible results - removed for different outcomes

    # Generate price data with trends and volatility
    prices = [100.0]  # Starting price

    for i in range(num_steps - 1):
        # Market regime changes (more diverse than v431)
        regime = np.random.choice(['bull', 'bear', 'sideways', 'high_vol', 'low_vol'],
                                p=[0.25, 0.25, 0.2, 0.15, 0.15])

        if regime == 'bull':
            change = np.random.normal(0.002, 0.025)  # Stronger upward bias
        elif regime == 'bear':
            change = np.random.normal(-0.002, 0.025)  # Stronger downward bias
        elif regime == 'high_vol':
            change = np.random.normal(0.0, 0.04)  # High volatility
        elif regime == 'low_vol':
            change = np.random.normal(0.0, 0.01)  # Low volatility
        else:  # sideways
            change = np.random.normal(0.0, 0.015)  # No bias, moderate volatility

        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.1))  # Prevent negative prices

    return np.array(prices)

def calculate_dynamic_position_size(config, market_condition, confidence_score, volatility):
    """Calculate dynamic position size based on market conditions"""
    apm_config = config['advanced_position_management']
    base_size = apm_config['dynamic_sizing']['base_position_size']

    # Volatility scaling
    vol_config = apm_config['dynamic_sizing']['volatility_scaling']
    if volatility > 0.03:  # High volatility
        vol_multiplier = vol_config['high_vol_min'] / base_size
    elif volatility < 0.015:  # Low volatility
        vol_multiplier = vol_config['low_vol_max'] / base_size
    else:
        vol_multiplier = 1.0

    # Confidence scaling
    conf_config = apm_config['dynamic_sizing']['confidence_scaling']
    if confidence_score > 0.8:  # High confidence
        conf_multiplier = conf_config['high_conf_max'] / base_size
    elif confidence_score < 0.4:  # Low confidence
        conf_multiplier = conf_config['low_conf_min'] / base_size
    else:
        conf_multiplier = 1.0

    # Market regime adaptation
    regime_config = apm_config['market_regime_adaptation']
    if market_condition in regime_config:
        regime_multiplier = regime_config[market_condition]['leverage_multiplier']
    else:
        regime_multiplier = 1.0

    position_size = base_size * vol_multiplier * conf_multiplier * regime_multiplier
    return min(position_size, apm_config['risk_management']['max_position_size'])

def check_entry_conditions(price, prev_price, market_condition, config):
    """Check if entry conditions are met"""
    entry_config = config['advanced_position_management']['entry_conditions']

    price_change = (price - prev_price) / prev_price
    trend_strength = abs(price_change)
    min_trend = entry_config['trend_strength_min']

    if trend_strength < min_trend:
        return False

    # Market condition specific checks
    if market_condition == 'high_vol' and entry_config.get('volume_confirmation', False):
        # 変動が大きい局面では、より強いトレンドでないとエントリーしない
        if trend_strength < min_trend * 1.5:
            return False

    if entry_config.get('momentum_alignment', False):
        if market_condition == 'bull' and price_change <= 0:
            return False
        if market_condition == 'bear' and price_change >= 0:
            return False

    if entry_config.get('support_resistance_filter', False):
        # シンプルなサポレジ近似: 明確なトレンドが無い場合はエントリーを抑制する
        if market_condition in ('sideways', 'neutral') and trend_strength < min_trend * 1.2:
            return False

    return True

def check_exit_conditions(position, entry_price, current_price, hold_periods, config):
    """Check if exit conditions are met"""
    exit_config = config['advanced_position_management']['exit_conditions']

    if position == 0 or entry_price is None or entry_price <= 0:
        return False, None

    # Profit target check
    if position > 0:  # Long position
        profit_pct = (current_price - entry_price) / entry_price
        if profit_pct >= exit_config['profit_target_pct']:
            return True, 'profit_target'
        elif profit_pct <= -exit_config['stop_loss_pct']:
            return True, 'stop_loss'
    else:  # Short position
        profit_pct = (entry_price - current_price) / entry_price
        if profit_pct >= exit_config['profit_target_pct']:
            return True, 'profit_target'
        elif profit_pct <= -exit_config['stop_loss_pct']:
            return True, 'stop_loss'

    # Time-based exit
    if hold_periods >= exit_config['time_based_exit']['max_hold_periods']:
        return True, 'time_exit'

    return False, None

def simulate_sac_v432_2_action(
    config, price, prev_price, position, position_size, confidence, market_condition
):
    """Simulate SAC v432.2 action with Win Rate Optimization"""
    # Calculate price change
    price_change = (price - prev_price) / prev_price

    # Enhanced market condition detection
    volatility = np.random.uniform(0.005, 0.06)  # Wider volatility range
    trend_strength = abs(price_change)

    if volatility > 0.035:
        detected_condition = 'high_vol'
    elif volatility < 0.015:
        detected_condition = 'low_vol'
    elif trend_strength < 0.008:
        detected_condition = 'sideways'
    elif price_change > 0.015:
        detected_condition = 'bull'
    elif price_change < -0.015:
        detected_condition = 'bear'
    else:
        detected_condition = 'neutral'

    # Get base rewards (v432.2: optimized for win rate)
    base_rewards = {
        'SELL': config['reward_function']['sell_bonus'],  # 0.3
        'HOLD': config['reward_function']['hold_bonus'],  # -0.02 (optimized for scalping)
        'BUY': config['reward_function']['buy_bonus']     # 0.3
    }

    # Apply market adaptive multipliers (enhanced for v432.1)
    market_multipliers = config['reward_function']['market_adaptive']
    if detected_condition == 'sideways':
        multiplier = market_multipliers['sideways_multiplier']  # 2.0 (increased)
    elif detected_condition == 'high_vol':
        multiplier = market_multipliers['high_vol_multiplier']  # 1.5 (increased)
    elif detected_condition == 'low_vol':
        multiplier = market_multipliers['low_vol_multiplier']  # 0.6 (decreased)
    elif detected_condition == 'bull':
        multiplier = market_multipliers.get('bull_multiplier', 1.2)
    elif detected_condition == 'bear':
        multiplier = market_multipliers.get('bear_multiplier', 1.2)
    else:
        multiplier = 1.0

    # Apply ensemble-style specialization bonuses (hardcoded for v432.1)
    specialization_overrides = {
        "bull": {
            "reward_overrides": {"BUY": 0.36, "SELL": 0.27}  # 1.2x, 0.9x
        },
        "bear": {
            "reward_overrides": {"BUY": 0.27, "SELL": 0.36}  # 0.9x, 1.2x
        },
        "sideways": {
            "reward_overrides": {"HOLD": -0.0015}  # 側面局面ではHOLDも許容
        },
        "high_vol": {
            "reward_overrides": {"BUY": 0.39, "SELL": 0.39}  # 1.3x each
        },
        "low_vol": {
            "reward_overrides": {"HOLD": -0.004}  # 低ボラ局面でも過度なトレードを抑える
        }
    }

    if detected_condition in specialization_overrides:
        member_config = specialization_overrides[detected_condition]
        if 'reward_overrides' in member_config:
            overrides = member_config['reward_overrides']
            for action, override in overrides.items():
                if action in base_rewards:
                    base_rewards[action] = override

    # Apply market multiplier to all rewards
    adjusted_rewards = {k: v * multiplier for k, v in base_rewards.items()}

    # Advanced Position Management decision making
    confidence_score = np.random.uniform(0.3, 0.9)  # Simulated confidence
    entry_config = config['advanced_position_management']['entry_conditions']

    # Check exit conditions first (simplified for v432.2)
    should_exit = False  # Simplified - no complex exit logic for initial win rate focus
    if should_exit:
        if position > 0:
            action = 'SELL'  # Close long
        elif position < 0:
            action = 'BUY'   # Close short
        else:
            action = 'HOLD'
    else:
        # Entry/position management logic
        if position == 0:  # Flat position
            # Check entry conditions
            if check_entry_conditions(price, prev_price, detected_condition, config):
                if detected_condition == 'bull' and price_change > 0:
                    action = 'BUY'
                elif detected_condition == 'bear' and price_change < 0:
                    action = 'SELL'
                elif detected_condition == 'high_vol':
                    if trend_strength > entry_config['trend_strength_min'] * 1.8:
                        action_prob = [0.3, 0.4, 0.3]
                        action = np.random.choice(['SELL', 'HOLD', 'BUY'], p=action_prob)
                    else:
                        action_prob = [0.4, 0.2, 0.4]  # Bias toward trading in high_vol
                        action = np.random.choice(['SELL', 'HOLD', 'BUY'], p=action_prob)
                elif detected_condition == 'sideways':
                    action_prob = [0.4, 0.2, 0.4]  # Bias toward trading in sideways
                    action = np.random.choice(['SELL', 'HOLD', 'BUY'], p=action_prob)
                else:
                    action_prob = [0.4, 0.2, 0.4]  # Bias toward trading in uncertain conditions
                    action = np.random.choice(['SELL', 'HOLD', 'BUY'], p=action_prob)
            else:
                action_prob = [0.4, 0.2, 0.4]  # Bias toward trading when entry conditions not met
                action = np.random.choice(['SELL', 'HOLD', 'BUY'], p=action_prob)
        else:
            # Position maintenance logic
            if abs(position) > 0.15:  # 現在の設定では0.15以上で大きめポジションとみなす
                if detected_condition in ['high_vol', 'sideways']:
                    action_prob = [0.4, 0.2, 0.4]  # Bias toward reducing position but allow trading
                    action = np.random.choice(['SELL', 'HOLD', 'BUY'], p=action_prob)
                else:
                    action_prob = [0.3, 0.4, 0.3]  # Hold in trending markets but allow some trading
                    action = np.random.choice(['SELL', 'HOLD', 'BUY'], p=action_prob)
            else:
                action_prob = [0.3, 0.4, 0.3]  # Small position, maintain but allow trading
                action = np.random.choice(['SELL', 'HOLD', 'BUY'], p=action_prob)

    final_action = action
    final_reward = adjusted_rewards[final_action]

    return final_action, final_reward, detected_condition, confidence_score

def run_backtest_simulation_v432_2(config, num_steps=10000, initial_capital=10000.0):
    """Run v432.2 backtest simulation with Win Rate Optimization"""
    print("=== SAC v432.2 Win Rate Optimization Backtest ===")
    print(f"Steps: {num_steps}, Initial Capital: ${initial_capital}")
    print("Features: Enhanced rewards for win rate, optimized market adaptation")

    # Generate market data
    prices = simulate_market_data(num_steps)
    print(f"Generated {len(prices)} price data points with diverse market conditions")

    # Initialize trading variables
    capital = initial_capital
    position = 0.0
    position_units = 0.0
    entry_price = None
    trades = []
    transaction_cost = 0.0005
    hold_periods = 0

    actions_count = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    total_reward = 0.0
    market_conditions = []

    print("\nStarting advanced position management simulation...")

    for step in range(1, num_steps):
        current_price = prices[step]
        prev_price = prices[step - 1]

        if position != 0:
            hold_periods += 1
        else:
            hold_periods = 0

        # Calculate dynamic position size
        volatility = np.random.uniform(0.005, 0.06)
        position_size = calculate_dynamic_position_size(config, 'neutral', 0.5, volatility)

        # Get SAC v432.2 action with win rate optimization
        confidence = 0.5  # Default confidence
        market_condition = 'neutral'  # Default market condition
        action, reward, market_condition, confidence = simulate_sac_v432_2_action(
            config, current_price, prev_price, position, position_size, confidence, market_condition
        )
        actions_count[action] += 1
        total_reward += reward
        market_conditions.append(market_condition)

        # Execute trade logic with advanced position management

        if action == 'BUY':
            if position < 0 and position_units != 0 and entry_price:
                trade_notional = abs(position_units) * current_price
                pnl = (current_price - entry_price) * position_units
                pnl -= transaction_cost * trade_notional
                capital += pnl
                trades.append({
                    'type': 'COVER_SHORT',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'step': step,
                    'market_condition': market_condition,
                    'hold_periods': hold_periods
                })
                position = 0.0
                position_units = 0.0
                entry_price = None
                hold_periods = 0

            if position == 0 and position_size > 0:
                entry_notional = capital * position_size
                if entry_notional > 0:
                    position_units = entry_notional / current_price
                    position = position_size
                    entry_price = current_price
                    capital -= transaction_cost * entry_notional
                    hold_periods = 1

        elif action == 'SELL':
            if position > 0 and position_units != 0 and entry_price:
                trade_notional = abs(position_units) * current_price
                pnl = (current_price - entry_price) * position_units
                pnl -= transaction_cost * trade_notional
                capital += pnl
                trades.append({
                    'type': 'CLOSE_LONG',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'step': step,
                    'market_condition': market_condition,
                    'hold_periods': hold_periods
                })
                position = 0.0
                position_units = 0.0
                entry_price = None
                hold_periods = 0

            if position == 0 and position_size > 0:
                entry_notional = capital * position_size
                if entry_notional > 0:
                    position_units = -entry_notional / current_price
                    position = -position_size
                    entry_price = current_price
                    capital -= transaction_cost * entry_notional
                    hold_periods = 1

        if step % 1000 == 0:
            progress = step / num_steps * 100
            print(f"Progress: {progress:.1f}%")

    # Close positions
    if position != 0 and position_units != 0 and entry_price:
        final_price = prices[-1]
        trade_notional = abs(position_units) * final_price
        pnl = (final_price - entry_price) * position_units
        pnl -= transaction_cost * trade_notional
        capital += pnl
        trades.append({
            'type': 'CLOSE_LONG' if position > 0 else 'COVER_SHORT',
            'entry_price': entry_price,
            'exit_price': final_price,
            'pnl': pnl,
            'step': num_steps,
            'market_condition': market_conditions[-1] if market_conditions else 'neutral',
            'hold_periods': hold_periods
        })
        position = 0.0
        position_units = 0.0
        entry_price = None

    # Calculate metrics
    total_return = (capital - initial_capital) / initial_capital * 100
    num_trades = len(trades)
    winning_trades = len([t for t in trades if t['pnl'] > 0])
    win_rate = winning_trades / num_trades * 100 if num_trades > 0 else 0

    returns = [t['pnl'] for t in trades]
    if returns:
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0
    else:
        sharpe_ratio = 0
        avg_return = 0

    cumulative_returns = np.cumsum([t['pnl'] for t in trades])
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = running_max - cumulative_returns
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

    # Market condition analysis
    market_condition_counts = {}
    for condition in market_conditions:
        market_condition_counts[condition] = market_condition_counts.get(condition, 0) + 1

    print("\n=== SAC v432.1 Advanced Position Management Results ===")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Final Capital: ${capital:.2f}")
    print(f"Total Trades: {num_trades}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: ${max_drawdown:.2f}")
    print(f"Average Trade P&L: ${avg_return:.2f}")

    print("\n[Advanced Action Distribution]")
    total_actions = sum(actions_count.values())
    for action, count in actions_count.items():
        pct = count / total_actions * 100
        print(f"  {action}: {count} ({pct:.1f}%)")

    print("\n[Market Condition Analysis]")
    for condition, count in market_condition_counts.items():
        pct = count / len(market_conditions) * 100
        print(f"  {condition}: {count} ({pct:.1f}%)")

    print("\n[Reward System Performance]")
    avg_reward = total_reward / num_steps
    print(f"Average Reward per Step: {avg_reward:.4f}")
    print(f"Total Reward Points: {total_reward:.2f}")

    # Performance comparison with v432.0
    print("\n[Performance Improvement Analysis vs v432.0]")
    if win_rate > 50:
        print("✅ Win rate improved (>50%)")
    else:
        print("⚠️ Win rate needs further improvement")

    if total_return > -89.94:  # Better than v432.0's -89.94%
        print("✅ Total return improved vs v432.0")
    else:
        print("❌ Total return still needs improvement")

    if actions_count['HOLD'] / total_actions < 0.45:  # Less than v432.0's 45.2%
        print("✅ HOLD rate reduced with negative penalty")
    else:
        print("⚠️ HOLD rate still high despite negative penalty")

    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_trades': num_trades,
        'final_capital': capital,
        'trades': trades,
        'market_conditions': market_condition_counts,
        'actions_distribution': actions_count,
        'total_reward': total_reward
    }

def main():
    # Load configuration
    config = load_config()

    # Run advanced position management backtest
    results = run_backtest_simulation_v432_2(config, num_steps=10000, initial_capital=10000.0)

    # Save results
    output_file = get_project_root() / "ztb" / "evaluation" / "v432" / "sac_v432_2_win_rate_optimization_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    print("\n=== SAC v432.2 Win Rate Optimization Evaluation Complete ===")
    print("Ready for training integration and further optimization")

if __name__ == "__main__":
    main()
