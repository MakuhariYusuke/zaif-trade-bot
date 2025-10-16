import os

# ztb/multimodal以下の全ディレクトリに__init__.pyを作成
for root, dirs, files in os.walk('ztb/multimodal'):
    init_file = os.path.join(root, '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            module_name = os.path.basename(root)
            f.write('"""' + module_name + ' module"""\n')
            f.write('__version__ = "1.0.0"\n')

print('Created __init__.py files in all multimodal subdirectories')