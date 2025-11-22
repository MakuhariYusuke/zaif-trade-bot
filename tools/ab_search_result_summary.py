#!/usr/bin/env python3
import json
from pathlib import Path

reports = Path('reports')
res = {}
for p in reports.glob('training_report_*.json'):
    try:
        obj = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        continue
    ab_tag = obj.get('metadata', {}).get('ab_tag')
    if not ab_tag or not ab_tag.startswith('ab_balance_search_'):
        continue
    env = obj.get('configuration', {}).get('environment', {})
    rs = env.get('reward_settings', {})
    skew = rs.get('skewness_penalty_value')
    balance = rs.get('balance_shaping_value')
    dist = obj.get('training_stats', {}).get('action_distribution', {})
    key = (skew, balance)
    if key not in res:
        res[key] = {'dists': [], 'reports': []}
    res[key]['dists'].append(dist)
    res[key]['reports'].append(p.name)

summary = []
for key, v in res.items():
    dists = v['dists']
    avg = {'HOLD':0.0, 'BUY':0.0, 'SELL':0.0}
    for d in dists:
        for k in avg:
            avg[k] += float(d.get(k, 0.0))
    for k in avg:
        avg[k] /= max(1, len(dists))
    summary.append({'skew': key[0], 'balance': key[1], 'avg': avg, 'count': len(dists), 'reports': v['reports']})

# sort by |BUY-SELL| (balance objective ascending)
summary.sort(key=lambda x: abs(x['avg']['BUY'] - x['avg']['SELL']))
print(json.dumps(summary, ensure_ascii=False, indent=2))
