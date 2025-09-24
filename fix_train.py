with open('scripts/train.py', 'r') as f:
    lines = f.readlines()

# 重複した関数を削除
new_lines = []
skip_until = None
seen_functions = set()

for i, line in enumerate(lines):
    if line.startswith('def '):
        func_name = line.split('(')[0]
        if func_name in seen_functions:
            skip_until = 'def '
            continue
        else:
            seen_functions.add(func_name)

    if skip_until:
        if line.startswith('def ') and skip_until == 'def ':
            skip_until = None
        else:
            continue

    new_lines.append(line)

with open('scripts/train_fixed.py', 'w') as f:
    f.writelines(new_lines)

print("Fixed train.py saved as scripts/train_fixed.py")