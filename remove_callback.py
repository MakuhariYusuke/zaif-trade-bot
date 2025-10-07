import ast
import re
from typing import Union

# Read the file
with open('ztb/training/curriculum_transition.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Parse the AST
tree = ast.parse(content)

# Find the TrainingCallback class
training_callback_node: Union[type, ast.ClassDef]

for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == 'TrainingCallback':
        training_callback_node = node
        break
else:
    training_callback_node = type(None)  # fallback

# Get the source lines for the class
lines = content.split('\n')
if isinstance(training_callback_node, ast.ClassDef):
    start_line = training_callback_node.lineno - 1  # 0-indexed
    end_line = (training_callback_node.end_lineno or training_callback_node.lineno) - 1
else:
    raise ValueError("TrainingCallback class not found")

# Remove the class
new_content = '\n'.join(lines[:start_line] + lines[end_line:])

# Write back
with open('ztb/training/curriculum_transition.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print('Removed TrainingCallback class from curriculum_transition.py')