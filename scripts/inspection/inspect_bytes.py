from pathlib import Path
p=Path('ztb/utils/talib_wrapper.py')
b=p.read_bytes()
s=b.decode('utf-8',errors='replace')
lines=s.splitlines(True)
for i,l in enumerate(lines, start=1):
    if 1504<=i<=1508:
        print('LINE',i,repr(l))
        print([ord(c) for c in l])
