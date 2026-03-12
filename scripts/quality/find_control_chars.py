from pathlib import Path
p=Path('ztb/utils/talib_wrapper.py')
s=p.read_text(encoding='utf-8',errors='replace')
for i,line in enumerate(s.splitlines(),start=1):
    for j,ch in enumerate(line):
        if ord(ch)<32 and ord(ch) not in (9,10,13):
            print(f'Line {i} col {j} ord {ord(ch)} char {repr(ch)}')
