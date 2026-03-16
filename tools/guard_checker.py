"""
Guard checker: scans the repo for unguarded `sys.exit` calls and suggests recommended changes.
It does NOT modify files by default; it's intended to be used interactively to review candidates.

Usage:
    python tools/guard_checker.py
"""

import argparse
import ast
from pathlib import Path


def find_unguarded_sys_exit(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    class MainGuardVisitor(ast.NodeVisitor):
        def __init__(self):
            self.unguarded_sys_exits = []
            self.in_main_guard = False

        def visit_If(self, node: ast.If):
            # detect __name__ == '__main__' guard and treat nested calls as guarded
            try:
                left = node.test.left
                comparators = node.test.comparators
                if hasattr(node.test, "left") and hasattr(node.test, "comparators"):
                    left = node.test.left
                    comp = node.test.comparators[0]
                    if (
                        isinstance(left, ast.Name)
                        and left.id == "__name__"
                        and isinstance(comp, ast.Constant)
                        and comp.value == "__main__"
                    ):
                        # This block is guarded; ignore sys.exit occurrences inside
                        return
            except Exception:
                pass
            self.generic_visit(node)

        def visit_Expr(self, node: ast.Expr):
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call):
            try:
                if isinstance(node.func, ast.Attribute):
                    if (
                        getattr(node.func.value, "id", None) == "sys"
                        and node.func.attr == "exit"
                    ):
                        self.unguarded_sys_exits.append((node.lineno, node.col_offset))
                elif isinstance(node.func, ast.Name) and node.func.id == "exit":
                    # built-in exit() (rare), treat as potential unguarded exit
                    self.unguarded_sys_exits.append((node.lineno, node.col_offset))
            except Exception:
                pass
            self.generic_visit(node)

    v = MainGuardVisitor()
    v.visit(tree)
    return v.unguarded_sys_exits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=".", help="Repository root to scan")
    args = parser.parse_args()

    repo_root = Path(args.path)
    py_files = list(repo_root.rglob("*.py"))
    candidates = []
    for pf in py_files:
        pfs = str(pf)
        if any(
            x in pfs
            for x in (
                "/venv/",
                "\\venv\\",
                "/venv311/",
                "\\venv311\\",
                "/.venv/",
                "\\.venv\\",
                "/archived/",
                "\\archived\\",
                "/node_modules/",
                "\\node_modules\\",
            )
        ):
            continue
        if "/tests/" in pfs or "\\tests\\" in pfs:
            continue

        violations = find_unguarded_sys_exit(pf)
        if violations:
            candidates.append((pf, violations))

    if candidates:
        print(f"Found {len(candidates)} files with unguarded sys.exit() calls:")
        for pf, vs in candidates:
            print(f"  {pf} -> {len(vs)} occurrences: {vs[:5]}")
        print(
            "\nSuggestion: For each file, move any execution logic into a `main()` function and add:\n"
        )
        print(
            "    if __name__ == '__main__':\n        from ztb.utils.cli import run_main\n        run_main(main)\n"
        )
        return 1
    else:
        print("No unguarded sys.exit() calls found")
        return 0


if __name__ == "__main__":
    import sys

    rc = main()
    sys.exit(rc)
