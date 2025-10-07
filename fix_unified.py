with open('ztb/tests/unit/training/test_unified_trainer.py', 'r', encoding='utf-8') as f:
    content = f.read()
# 316行目までを保持
lines = content.split('\n')
clean_content = '\n'.join(lines[:317]) + '\n'
with open('ztb/tests/unit/training/test_unified_trainer.py', 'w', encoding='utf-8') as f:
    f.write(clean_content)
print('File fixed')