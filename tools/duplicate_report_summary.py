import json
from pathlib import Path

p = Path('reports/duplicate_report.json')
if not p.exists():
    print('Report not found:', p)
    raise SystemExit(1)

r = json.loads(p.read_text(encoding='utf-8'))
exact = r.get('exact_groups', {})
similar = r.get('similar_pairs', [])
print(f"exact_groups: {len(exact)}")
print(f"similar_pairs: {len(similar)}")

groups = sorted([(k, len(v)) for k, v in exact.items()], key=lambda x: -x[1])
print('\nTop exact groups:')
for k, c in groups[:10]:
    print(f'  {k} -> {c}')

print('\nTop similar pairs:')
for p in sorted(similar, key=lambda x: -x['score'])[:10]:
    print(f"  {p['h1']} ~ {p['h2']} | score={p['score']:.3f}")
