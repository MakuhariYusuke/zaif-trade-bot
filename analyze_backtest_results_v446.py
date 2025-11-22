import json

with open('backtest_results_sac_v446.json', 'r') as f:
    data = json.load(f)

print('=== バックテスト結果サマリー ===')
print(f'モデル: {data["model_name"]}')
print(f'総ステップ数: {data["total_steps"]}')
print(f'初期ポートフォリオ: {data["initial_portfolio"]:,.0f}円')
print(f'最終ポートフォリオ: {data["final_portfolio"]:,.0f}円')
print(f'総リターン: {data["total_return_pct"]:.1f}%')
print(f'総報酬: {data["total_reward"]:.2f}')
print(f'アクション分布: {data["action_distribution"]}')

# ポートフォリオ履歴の分析
portfolio = data['portfolio_history']
print(f'ポートフォリオ履歴サンプル数: {len(portfolio)}')
print(f'最大ポートフォリオ: {max(portfolio):,.0f}円')
print(f'最小ポートフォリオ: {min(portfolio):,.0f}円')

# アクション分布の詳細
actions = data.get('actions', [])
if actions:
    buy_count = actions.count(0)
    sell_count = actions.count(2)
    hold_count = actions.count(1)
    total_actions = len(actions)
    print(f'アクション詳細: BUY={buy_count} ({buy_count/total_actions*100:.1f}%), HOLD={hold_count} ({hold_count/total_actions*100:.1f}%), SELL={sell_count} ({sell_count/total_actions*100:.1f}%)')
else:
    print('アクション履歴なし')