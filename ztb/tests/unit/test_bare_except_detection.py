"""
Tests for detecting bare except clauses in the codebase.

Ensures standardized error handling without bare except clauses.
"""

import ast
import os
from pathlib import Path
from typing import List, Tuple


def find_bare_except_clauses(source_code: str) -> List[Tuple[int, str]]:
    """
    Find bare except clauses in Python source code.

    Args:
        source_code: Python source code as string

    Returns:
        List of (line_number, line_content) tuples for bare except clauses
    """
    violations = []

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return violations  # Skip files with syntax errors

    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            # Check if except clause has no exception type specified
            if node.type is None:
                # Get the line content
                lines = source_code.splitlines()
                if node.lineno <= len(lines):
                    line_content = lines[node.lineno - 1].strip()
                    violations.append((node.lineno, line_content))

    return violations


def test_no_bare_except_clauses():
    """Test that no bare except clauses exist in the codebase."""
    ztb_path = Path(__file__).parent.parent.parent / "ztb"
    violations = []

    # Walk through all Python files in ztb directory
    for root, dirs, files in os.walk(ztb_path):
        # Skip __pycache__ directories
        if "__pycache__" in dirs:
            dirs.remove("__pycache__")

        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    file_violations = find_bare_except_clauses(content)
                    for line_no, line_content in file_violations:
                        violations.append((str(file_path), line_no, line_content))

                except (IOError, UnicodeDecodeError):
                    # Skip files that can't be read
                    continue

    # Report violations
    if violations:
        violation_messages = []
        for file_path, line_no, line_content in violations:
            violation_messages.append(f"{file_path}:{line_no}: {line_content}")

        violation_text = "\n".join(violation_messages)
        raise AssertionError(
            f"Found {len(violations)} bare except clause(s):\n{violation_text}\n"
            "Please replace bare 'except:' with specific exception types or use 'except Exception:'"
        )

    # No violations found
    assert len(violations) == 0
