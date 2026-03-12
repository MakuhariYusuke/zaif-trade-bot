import os
from collections import Counter
from pathlib import Path

from ztb.io.json_io import read_json
# プロジェクトルートを取得
project_root = Path(__file__).parent.parent.parent.parent

# Find the latest training results
results_dir = project_root / 'results'
if os.path.exists(results_dir):
    files = [f for f in os.listdir(results_dir) if f.startswith('sac_v444') and f.endswith('.json')]
    if files:
        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
        print(f'Analyzing latest results: {latest_file}')

        data = read_json(os.path.join(results_dir, latest_file))

        # Extract actions from the last 2000 steps
        actions = []
        rewards = []
        balance_penalties = []

        for step_data in data.get('training_history', []):
            if 'action' in step_data:
                actions.append(step_data['action'])
            if 'reward' in step_data:
                rewards.append(step_data['reward'])
            if 'balance_penalty' in step_data:
                balance_penalties.append(step_data['balance_penalty'])

        if actions:
            action_counts = Counter(actions)
            total_actions = len(actions)

            print(f'\n📊 Action Distribution (last {total_actions} steps):')
            for action, count in sorted(action_counts.items()):
                percentage = (count / total_actions) * 100
                action_name = {0: 'HOLD', 1: 'BUY', -1: 'SELL'}.get(action, f'UNKNOWN({action})')
                print(f'  {action_name}: {count} ({percentage:.1f}%)')

            print('\n💰 Reward Statistics:')
            if rewards:
                print(f'  Average reward: {sum(rewards)/len(rewards):.2f}')
                print(f'  Min reward: {min(rewards):.2f}')
                print(f'  Max reward: {max(rewards):.2f}')
                positive_rewards = [r for r in rewards if r > 0]
                print(f'  Positive rewards: {len(positive_rewards)} ({len(positive_rewards)/len(rewards)*100:.1f}%)')

            print('\n⚖️ Balance Penalty Statistics:')
            if balance_penalties:
                avg_penalty = sum(balance_penalties)/len(balance_penalties)
                print(f'  Average balance penalty: {avg_penalty:.2f}')
                print(f'  Balance penalty was 0: {balance_penalties.count(0.0)} times')
                print(f'  Balance penalty > 0: {len([p for p in balance_penalties if p > 0])} times')
        else:
            print('No action data found in training results')
    else:
        print('No SAC v444 training result files found')
else:
    print('Results directory not found')
