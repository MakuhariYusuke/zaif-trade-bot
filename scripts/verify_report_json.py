import json,sys
p = sys.argv[1] if len(sys.argv) > 1 else 'reports/duplicate_report.json'
try:
    with open(p,'r',encoding='utf-8') as f:
        obj=json.load(f)
    print('JSON ok:', len(obj.get('exact_groups',{})), 'exact_groups,', len(obj.get('similar_pairs',[])), 'similar_pairs')
except Exception as e:
    print('JSON load error',e)
    sys.exit(1)
