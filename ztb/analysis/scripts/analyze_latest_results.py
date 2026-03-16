from ztb.io.json_io import read_json

data = read_json("results/sac_v444_training_results_20251106_044114.json")

print('Keys in root:', list(data.keys()))
if 'training_history' in data:
    print('training_history length:', len(data['training_history']))
    if data['training_history']:
        print('First training_history item:', data['training_history'][0])
        print('Last training_history item:', data['training_history'][-1])

# Count actions
actions = [step['action'] for step in data.get('training_history', [])]
if actions:
    from collections import Counter
    action_counts = Counter(actions)
    total = len(actions)
    print('Action distribution:')
    for action, count in action_counts.items():
        pct = (count / total) * 100
        action_name = {0: 'HOLD', 1: 'BUY', -1: 'SELL'}.get(action, f'UNKNOWN({action})')
        print(f'  {action_name}: {count} ({pct:.1f}%)')

# Check balance penalties
balance_penalties = [step['balance_penalty'] for step in data.get('training_history', [])]
if balance_penalties:
    unique_penalties = set(balance_penalties)
    print('Unique balance penalties:', sorted(unique_penalties))
    non_zero_penalties = [bp for bp in balance_penalties if bp != 0.0]
    print('Non-zero balance penalties count:', len(non_zero_penalties))
    if non_zero_penalties:
        print('Non-zero balance penalty values:', sorted(set(non_zero_penalties)))

# Check rewards
rewards = [step['reward'] for step in data.get('training_history', [])]
if rewards:
    unique_rewards = set(rewards)
    print('Unique rewards:', sorted(unique_rewards))
    print('Average reward:', sum(rewards)/len(rewards))
    print('Min reward:', min(rewards))
    print('Max reward:', max(rewards))
