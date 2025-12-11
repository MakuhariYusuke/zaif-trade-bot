filename = r"c:\Users\Admin\dev\zaif-trade-bot\ztb\features\models\sac\sac_v427_feature_engineering.py"

with open(filename, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "def _apply_adaptive" in line:
        print(f"Line {i+1}: {line.strip()}")
