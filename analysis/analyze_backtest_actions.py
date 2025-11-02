import json

import numpy as np

# Load the backtest results
with open("backtest_results_after_training.json", "r") as f:
    data = json.load(f)

# Extract actions history
actions = np.array(data["actions_history"])

# Classify actions: assuming 0 is HOLD, positive is BUY, negative is SELL
# Based on the summary, threshold might be around 0
hold_threshold = 0.1  # Small threshold for HOLD

hold_count = np.sum(np.abs(actions) < hold_threshold)
buy_count = np.sum(actions > hold_threshold)
sell_count = np.sum(actions < -hold_threshold)

total_actions = len(actions)

print(f"Total actions: {total_actions}")
print(f"HOLD: {hold_count} ({hold_count/total_actions*100:.1f}%)")
print(f"BUY: {buy_count} ({buy_count/total_actions*100:.1f}%)")
print(f"SELL: {sell_count} ({sell_count/total_actions*100:.1f}%)")

# Also check the metrics
metrics = data["metrics"]
print("\nMetrics:")
print(f"Total return: {metrics['total_return_pct']:.2f}%")
print(f"Max drawdown: {metrics['max_drawdown_pct']:.2f}%")
print(f"Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Initial portfolio: ¥{metrics['initial_portfolio_value']:,.0f}")
print(f"Final portfolio: ¥{metrics['final_portfolio_value']:,.0f}")
print(f"Total trades: {metrics['total_trades']}")
