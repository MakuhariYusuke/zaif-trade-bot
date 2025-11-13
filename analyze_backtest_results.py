import json
import numpy as np
import glob
import os
from pathlib import Path

# 必要なインポート
from ztb.trading.environment.constants import continuous_to_discrete_action

def analyze_unified_backtest_results(results_file=None):
    """統一されたバックテスト結果を解析"""
    if results_file is None:
        # 最新のunified_backtest_resultsファイルを検索
        pattern = "unified_backtest_results_*.json"
        files = list(Path(".").glob(pattern)) + list(Path("backtest_results").glob(pattern)) if Path("backtest_results").exists() else []
        if not files:
            print("No unified backtest results found.")
            return
        results_file = max(files, key=os.path.getctime)

    print(f"=== Unified Backtest Analysis: {results_file} ===")

    with open(results_file, 'r') as f:
        summary = json.load(f)

    print(f'Mode: {summary["mode"]}')
    print(f'Episodes: {summary["n_episodes"]}')
    print(f'Periods: {summary["n_periods"]}')
    print(f'Signal Guidance: {summary["enable_signal_guidance"]}')
    print()

    episodes = summary['results']
    final_balances = [ep['final_balance'] for ep in episodes]
    print(f'Final Portfolio Values: {final_balances}')
    print(f'Average Final Balance: {np.mean(final_balances):.2f}')
    print(f'Std Final Balance: {np.std(final_balances):.2f}')
    print(f'Average Return: {summary["avg_return_pct"]:.2f}%')
    print(f'Std Return: {summary["std_return_pct"]:.2f}%')
    print(f'Win Rate: {summary["win_rate"]:.1f}%')
    print(f'Sharpe Ratio: {summary["sharpe_ratio"]:.4f}')
    print()

    # SIGNAL_GUIDANCE分析（有効な場合）
    if summary.get('enable_signal_guidance', False):
        print('=== SIGNAL_GUIDANCE Analysis ===')
        all_guidance_scores = []
        all_original_actions = []
        all_guidance_actions = []

        for ep in episodes:
            if 'guidance_signals' in ep:
                for signal in ep['guidance_signals']:
                    all_guidance_scores.append(signal['guidance_score'])
                    all_original_actions.append(signal['original_action'])
                    all_guidance_actions.append(signal['guidance_action'])

        if all_guidance_scores:
            print(f'Number of signals: {len(all_guidance_scores)}')
            print(f'Average guidance score: {np.mean(all_guidance_scores):.2f}')
            print(f'Score std: {np.std(all_guidance_scores):.2f}')
            print(f'Min score: {min(all_guidance_scores):.2f}')
            print(f'Max score: {max(all_guidance_scores):.2f}')
            print()

            # アクション分布
            print('=== Action Distribution ===')
            orig_discrete = []
            guide_discrete = []
            for a in all_original_actions:
                if isinstance(a, list) and len(a) > 0:
                    orig_discrete.append(continuous_to_discrete_action(a[0]))
                elif isinstance(a, (int, float)):
                    orig_discrete.append(a)
                else:
                    orig_discrete.append(0)  # default

            for a in all_guidance_actions:
                if isinstance(a, list) and len(a) > 0:
                    guide_discrete.append(continuous_to_discrete_action(a[0]))
                elif isinstance(a, (int, float)):
                    guide_discrete.append(a)
                else:
                    guide_discrete.append(0)  # default

            print(f'Original actions - Hold: {orig_discrete.count(0)}, Buy: {orig_discrete.count(1)}, Sell: {orig_discrete.count(-1)}')
            print(f'Guidance actions - Hold: {guide_discrete.count(0)}, Buy: {guide_discrete.count(1)}, Sell: {guide_discrete.count(-1)}')

            differences = sum(1 for o, g in zip(orig_discrete, guide_discrete) if o != g)
            print(f'Actions where guidance differed from original: {differences}/{len(orig_discrete)} ({differences/len(orig_discrete)*100:.1f}%)')
            print()

            # スコア vs ポートフォリオ価値の相関
            portfolio_values = []
            for ep in episodes:
                if 'guidance_signals' in ep:
                    portfolio_values.extend([s['portfolio_value'] for s in ep['guidance_signals']])

            if len(portfolio_values) == len(all_guidance_scores):
                correlation = np.corrcoef(all_guidance_scores, portfolio_values)[0,1]
                print(f'Correlation between SIGNAL_GUIDANCE score and portfolio value: {correlation:.3f}')
            print()

if __name__ == "__main__":
    import sys
    results_file = sys.argv[1] if len(sys.argv) > 1 else None
    analyze_unified_backtest_results(results_file)