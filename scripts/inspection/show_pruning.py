from pathlib import Path
p=Path('ztb/training/model_compression.py')
s=p.read_text()
lines=s.splitlines()
for i,line in enumerate(lines, start=1):
    if 'class PruningCompressor' in line:
        print('\n---',i)
        for j in range(max(1,i-4), i+20):
            if j-1 < len(lines):
                print(f'{j:4}: {lines[j-1]}')
