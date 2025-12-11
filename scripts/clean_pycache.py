import os
import shutil

root = r"c:\Users\Admin\dev\zaif-trade-bot"
for dirpath, dirnames, filenames in os.walk(root):
    for d in list(dirnames):
        if d == "__pycache__":
            full = os.path.join(dirpath, d)
            try:
                shutil.rmtree(full)
                print("Removed", full)
            except Exception as e:
                print("Failed to remove", full, "->", e)
