#!/usr/bin/env python3
"""
Check action distribution over time from training results.
"""

import json

def main():
    with open('results/sac_v444_training_results_20251106_053352.json', 'r') as f:
        data = json.load(f)

    # Extract action distribution over time
    if 'training_metrics' in data:
        metrics = data['training_metrics']
        print('=== Action Distribution Over Time (Last 20 entries) ===')
        for i, metric in enumerate(metrics[-20:]):  # Last 20 entries
            if 'action_distribution' in metric:
                dist = metric['action_distribution']
                step = metric.get('step', 'unknown')
                hold_pct = dist.get('HOLD', 0)
                buy_pct = dist.get('BUY', 0)
                sell_pct = dist.get('SELL', 0)
                print(f'Step {step}: HOLD={hold_pct:.1f}%, BUY={buy_pct:.1f}%, SELL={sell_pct:.1f}%')

        # Check if SELL is consistently high
        sell_ratios = []
        for metric in metrics:
            if 'action_distribution' in metric:
                sell_ratios.append(metric['action_distribution'].get('SELL', 0))

        if sell_ratios:
            avg_sell = sum(sell_ratios) / len(sell_ratios)
            max_sell = max(sell_ratios)
            min_sell = min(sell_ratios)
            print(f'\nSELL ratio statistics:')
            print(f'Average: {avg_sell:.1f}%')
            print(f'Maximum: {max_sell:.1f}%')
            print(f'Minimum: {min_sell:.1f}%')

            if avg_sell > 60:
                print('⚠️  Severe SELL bias detected throughout training!')
            elif avg_sell > 40:
                print('⚠️  Moderate SELL bias detected')
            else:
                print('✅ SELL ratio appears balanced')

if __name__ == "__main__":
    main()