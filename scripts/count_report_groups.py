import json,sys
p = sys.argv[1] if len(sys.argv) > 1 else 'reports/duplicate_report.json'
with open(p,'r',encoding='utf-8') as f:
    obj=json.load(f)
print('exact groups:', len(obj.get('exact_groups',{})))
print('similar pairs:', len(obj.get('similar_pairs',[])))
