"""Cleanup script to remove __pycache__ directories under the project root."""
import os
import shutil

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
removed = 0
for dirpath, dirnames, filenames in os.walk(root):
    if '__pycache__' in dirnames:
        p = os.path.join(dirpath, '__pycache__')
        try:
            shutil.rmtree(p)
            print('removed', p)
            removed += 1
        except Exception as e:
            print('failed to remove', p, e)

print('total removed', removed)
