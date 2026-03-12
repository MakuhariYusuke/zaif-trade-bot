import ast
import hashlib
import os
from collections import defaultdict


def get_function_hash(func_node: ast.FunctionDef) -> str:
    """Get hash of function body for comparison"""
    # Remove docstrings and comments for better matching
    body_lines = []
    for node in func_node.body:
        if not isinstance(node, (ast.Expr, ast.Pass)) or not isinstance(
            getattr(node, "value", None), ast.Str
        ):
            body_lines.append(ast.unparse(node))
    return hashlib.md5("".join(body_lines).encode()).hexdigest()


# Find all Python files
py_files = []
for root, dirs, files in os.walk("ztb"):
    for file in files:
        if file.endswith(".py") and not file.startswith("test_"):
            py_files.append(os.path.join(root, file))

# Analyze functions
functions = defaultdict(list)
for file_path in py_files[:30]:  # Analyze first 30 files
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_hash = get_function_hash(node)
                functions[func_hash].append(
                    (file_path, node.name, len(ast.unparse(node)))
                )
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")

# Find duplicates
duplicates = {k: v for k, v in functions.items() if len(v) > 1}
print(f"Found {len(duplicates)} duplicate function groups")

# Show top duplicates
sorted_duplicates = sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True)
for i, (hash_val, funcs) in enumerate(sorted_duplicates[:10]):
    print(f"\nDuplicate group {i+1}: {len(funcs)} functions")
    for file_path, func_name, size in funcs[:5]:  # Show first 5
        print(f"  {file_path}:{func_name} ({size} chars)")
