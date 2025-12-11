import json

with open("signal_guidance_backtest_results_20251112_135639.json", "r") as f:
    data = json.load(f)

signals = data["results"][0]["guidance_signals"]
print("Total signals:", len(signals))
print("Sample signals:")
for i in range(min(20, len(signals))):
    s = signals[i]
    print(
        "  step",
        s["step"],
        ": score",
        s["guidance_score"],
        "-> guidance_action",
        s["guidance_action"],
        "(orig",
        s["original_action"],
        ")",
    )

# アクション分布を確認
actions = [s["guidance_action"] for s in signals]
unique_actions = set(actions)
print("Unique guidance_actions:", unique_actions)
for action in unique_actions:
    count = actions.count(action)
    print(f"  Action {action}: {count} times")
