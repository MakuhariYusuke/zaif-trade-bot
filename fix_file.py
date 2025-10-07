with open('ztb/tests/unit/training/test_paper_trade.py', 'r', encoding='utf-8') as f:
    content = f.read()
# 231行目までを保持
lines = content.split('\n')
clean_content = '\n'.join(lines[:232]) + '\n'
with open('ztb/tests/unit/training/test_paper_trade.py', 'w', encoding='utf-8') as f:
    f.write(clean_content)
print('File fixed')