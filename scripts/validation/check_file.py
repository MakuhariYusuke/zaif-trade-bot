import sys
filename = sys.argv[1] if len(sys.argv) > 1 else 'ztb/tests/unit/training/test_unified_trainer.py'
with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()
print(f'Total lines: {len(lines)}')
for i, line in enumerate(lines[-10:], len(lines)-9):
    print(f'{i}: {repr(line)}')