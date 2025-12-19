#!/usr/bin/env python3
"""Check recent training reports for reward_components."""
import json
from pathlib import Path
from datetime import datetime

reports_dir = Path('reports')
reports = sorted(reports_dir.glob('training_report_*.json'), 
                 key=lambda p: p.stat().st_mtime, reverse=True)[:5]

print("="*80)
print("Recent Training Reports (Last 5)")
print("="*80)

for i, report_path in enumerate(reports, 1):
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        stats = data.get('training_stats', {})
        mod_time = datetime.fromtimestamp(report_path.stat().st_mtime)
        
        print(f"\n{i}. {report_path.name}")
        print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Timesteps: {stats.get('total_timesteps', 'N/A')}")
        
        ad = stats.get('action_distribution', {})
        buy = ad.get('BUY', 0)
        sell = ad.get('SELL', 0)
        hold = ad.get('HOLD', 0)
        print(f"   Actions: BUY={buy:.1%}, SELL={sell:.1%}, HOLD={hold:.1%}")
        
        # Check reward_components
        has_components = 'reward_components' in data or 'reward_components' in stats
        print(f"   reward_components: {'✓ YES' if has_components else '✗ NO'}")
        
        if has_components:
            components = data.get('reward_components', stats.get('reward_components', {}))
            print("   Components:")
            for key, value in list(components.items())[:5]:
                print(f"     {key}: {value:.6f}")
        
        print(f"   Config: {data.get('configuration', {}).get('version', 'unknown')}")
        
    except Exception as e:
        print(f"\n{i}. {report_path.name}")
        print(f"   Error: {e}")

print("\n" + "="*80)
