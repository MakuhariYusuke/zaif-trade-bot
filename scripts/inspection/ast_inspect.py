import ast
from pathlib import Path
s=Path('ztb/training/model_compression.py').read_text()
module=ast.parse(s)
for node in module.body:
    if isinstance(node, ast.ClassDef):
        print('Class:', node.name)
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                print('  method:', item.name, 'line', item.lineno)
    elif isinstance(node, ast.FunctionDef):
        print('Top-level function:', node.name)
