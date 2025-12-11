import json
import os


def analyze_backtest(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, "r") as f:
        data = json.load(f)

    trade_pnls = data.get("trade_pnls", [])
    num_trades = len(trade_pnls)

    print(f"Number of trades: {num_trades}")

    if num_trades > 0:
        avg_pnl = sum(trade_pnls) / num_trades
        print(f"Average PnL per trade: {avg_pnl:.2f}")
        print(f"Total PnL from trades: {sum(trade_pnls):.2f}")

        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p <= 0]
        print(f"Win Rate: {len(wins)/num_trades*100:.1f}% ({len(wins)}/{num_trades})")

    # Check action history if available
    if "actions" in data:
        actions = data["actions"]
        # Assuming actions are floats [-1, 1] or ints
        # Let's just print some stats
        print(f"Total actions recorded: {len(actions)}")

        # Simple binning
        # Note: actions might be a list of lists if it's [action_dim]
        # Let's handle that
        flat_actions = []
        for a in actions:
            if isinstance(a, list):
                flat_actions.append(a[0])
            else:
                flat_actions.append(a)

        buys = sum(1 for a in flat_actions if a > 0.33)
        sells = sum(1 for a in flat_actions if a < -0.33)
        holds = len(flat_actions) - buys - sells

        print("Action Distribution (Backtest):")
        print(f"  BUY  (> 0.33): {buys} ({buys/len(flat_actions)*100:.1f}%)")
        print(f"  SELL (< -0.33): {sells} ({sells/len(flat_actions)*100:.1f}%)")
        print(f"  HOLD (rest)  : {holds} ({holds/len(flat_actions)*100:.1f}%)")

        # Print some raw values to see if they are close to 0
        print(f"Sample actions: {flat_actions[:20]}")


if __name__ == "__main__":
    analyze_backtest(
        r"c:\Users\Admin\dev\zaif-trade-bot\backtest_results\phase6_hft_backtest.json"
    )
