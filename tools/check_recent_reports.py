#!/usr/bin/env python3
"""Check recent training reports for reward_components."""
from pathlib import Path
from datetime import datetime

from ztb.reporting.services.catalog import (
    extract_action_distribution_from_payload,
    extract_reward_components_from_payload,
    get_recent_training_reports,
    load_training_report,
)
from ztb.utils.safety import ensure_dict

reports_dir = Path('reports')
reports = get_recent_training_reports(limit=5, reports_dir=reports_dir)

print("="*80)
print("Recent Training Reports (Last 5)")
print("="*80)

for i, report_path in enumerate(reports, 1):
    try:
        data = load_training_report(report_path)
        if data is None:
            raise ValueError("Could not load JSON payload")
        
        stats = ensure_dict(data.get('training_stats'))
        mod_time = datetime.fromtimestamp(report_path.stat().st_mtime)
        
        print(f"\n{i}. {report_path.name}")
        print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Timesteps: {stats.get('total_timesteps', 'N/A')}")
        
        ad = extract_action_distribution_from_payload(data)
        buy = ad.get('BUY', 0)
        sell = ad.get('SELL', 0)
        hold = ad.get('HOLD', 0)
        print(f"   Actions: BUY={buy:.1%}, SELL={sell:.1%}, HOLD={hold:.1%}")
        
        # Check reward_components
        components = extract_reward_components_from_payload(data)
        has_components = bool(components)
        print(f"   reward_components: {'✓ YES' if has_components else '✗ NO'}")
        
        if has_components:
            print("   Components:")
            for key, value in list(components.items())[:5]:
                print(f"     {key}: {value:.6f}")
        
        config = ensure_dict(data.get('configuration'))
        print(f"   Config: {config.get('version', 'unknown')}")
        
    except Exception as e:
        print(f"\n{i}. {report_path.name}")
        print(f"   Error: {e}")

print("\n" + "="*80)
