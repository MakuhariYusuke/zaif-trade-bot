with open('ztb/tests/unit/training/test_unified_trainer.py', 'r', encoding='utf-8') as f:
    content = f.read()
# </content> を含む行を削除
lines = content.split('\n')
clean_lines = [line for line in lines if '</content>' not in line and '<parameter name' not in line]
clean_content = '\n'.join(clean_lines) + '\n'
with open('ztb/tests/unit/training/test_unified_trainer.py', 'w', encoding='utf-8') as f:
    f.write(clean_content)
print('File cleaned')