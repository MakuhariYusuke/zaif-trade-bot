import json
from pathlib import Path


def main():
    data_path = Path(__file__).resolve().parent.parent / "backtest_results_sac_v446.json"
    data = json.loads(data_path.read_text())
    prices = data["price_history"]
    actions = data["actions"]
    samples = []
    for i in range(len(actions) - 1):
        act = actions[i]
        curr = prices[i]
        nxt = prices[i + 1]
        if act == 1 and nxt > curr:
            samples.append(("BUY", i, curr, nxt, nxt - curr))
        elif act == -1 and nxt < curr:
            samples.append(("SELL", i, curr, nxt, nxt - curr))
        if len(samples) >= 8:
            break
    print(f"collected {len(samples)} favorable samples")
    for typ, idx, curr, nxt, delta in samples:
        print(
            f"{typ} step={idx} price_before={curr:.2f} price_after={nxt:.2f} delta={delta:.2f}"
        )


if __name__ == "__main__":
    main()