from collections import defaultdict

path = r"c:\Users\Admin\dev\zaif-trade-bot\logs\temp_training_1000_run.log"
with open(path, "rb") as f:
    text = f.read().decode("utf-16")

action_counts_first = defaultdict(int)
action_counts_second = defaultdict(int)
step_idx = 0
for line in text.splitlines():
    if "SAC continuous action" in line:
        step_idx += 1
        # parse discrete action
        try:
            da_idx = line.index("discrete action:")
            after = line[da_idx + len("discrete action:") :]
            discrete_str = after.strip().split()[0]
            discrete_action = int(float(discrete_str))
        except Exception:
            continue
        if step_idx <= 500:
            action_counts_first[discrete_action] += 1
        else:
            action_counts_second[discrete_action] += 1

print("Total SAC lines parsed:", step_idx)
print("\nFirst 500:")
for k in sorted(action_counts_first.keys()):
    print(k, action_counts_first[k])
print("\nSecond 500:")
for k in sorted(action_counts_second.keys()):
    print(k, action_counts_second[k])

# percentages
f_total = sum(action_counts_first.values())
s_total = sum(action_counts_second.values())
print("\nPercentages:")
for k in sorted(
    set(list(action_counts_first.keys()) + list(action_counts_second.keys()))
):
    f_pct = action_counts_first.get(k, 0) / f_total * 100 if f_total > 0 else 0
    s_pct = action_counts_second.get(k, 0) / s_total * 100 if s_total > 0 else 0
    print(k, f"{f_pct:.1f}%", "->", f"{s_pct:.1f}%", "(delta", f"{s_pct-f_pct:+.1f}%)")
