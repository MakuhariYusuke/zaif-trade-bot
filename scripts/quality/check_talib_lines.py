from pathlib import Path
p=Path('ztb/utils/talib_wrapper.py')
s=p.read_text(encoding='utf-8')
for i,line in enumerate(s.splitlines(),start=1):
    if 1498<=i<=1512 or 1768<=i<=1792 or 1708<=i<=1720:
        print(i, line.encode('unicode_escape'))
