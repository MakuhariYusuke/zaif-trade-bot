with open('ztb/tests/unit/training/test_paper_trade.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
# 最後の数行を表示
for i, line in enumerate(lines[-10:], len(lines)-9):
    print(f'{i}: {repr(line)}')