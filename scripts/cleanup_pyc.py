import os
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
removed = []
for p in root.rglob('__pycache__'):
    try:
        for child in p.rglob('*'):
            child.unlink()
        p.rmdir()
        removed.append(str(p))
    except Exception:
        pass
for p in root.rglob('*.pyc'):
    try:
        p.unlink()
        removed.append(str(p))
    except Exception:
        pass
print('Removed:', len(removed), 'items')
