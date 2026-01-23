import ast
from pathlib import Path


class MainGuardVisitor(ast.NodeVisitor):
    def __init__(self):
        self.unguarded_sys_exits = []
        self.guard_stack = []

    def visit_If(self, node: ast.If):
        # If the test compares __name__ to '__main__', treat this block as main guard
        is_main = False
        try:
            # Check for __name__ == '__main__'
            if isinstance(node.test, ast.Compare):
                left = node.test.left
                if isinstance(left, ast.Name) and left.id == "__name__":
                    for comp in node.test.comparators:
                        if isinstance(comp, ast.Constant) and comp.value == "__main__":
                            is_main = True
            # Also allow if __name__ in ('__main__',)
        except Exception:
            is_main = False

        self.guard_stack.append(
            is_main or (self.guard_stack[-1] if self.guard_stack else False)
        )
        # Visit children
        self.generic_visit(node)
        self.guard_stack.pop()

    def visit_Call(self, node: ast.Call):
        # Detect sys.exit calls
        try:
            if isinstance(node.func, ast.Attribute):
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "sys"
                    and node.func.attr == "exit"
                ):
                    # If current guard stack top is not True, it's unguarded
                    guarded = any(self.guard_stack)
                    if not guarded:
                        # Record location
                        self.unguarded_sys_exits.append((node.lineno, node.col_offset))
        except Exception:
            pass
        self.generic_visit(node)


def test_no_unguarded_sys_exit_across_repo():
    repo_root = Path(__file__).resolve().parents[2]
    py_files = list(repo_root.rglob("*.py"))
    violations = []
    for pf in py_files:
        # Skip files under virtualenv, archived, node_modules, or other non-repo sources
        pfs = str(pf)
        if any(
            p in pfs
            for p in (
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
        # Skip test files or files used as module __main__ (they are allowed to exit in practice) - skip tests path
        if "/tests/" in pfs or "\\tests\\" in pfs:
            continue
        try:
            tree = ast.parse(pf.read_text(encoding="utf-8"))
            v = MainGuardVisitor()
            v.visit(tree)
            if v.unguarded_sys_exits:
                violations.append((str(pf), v.unguarded_sys_exits))
        except SyntaxError:
            # Skip files with syntax errors or python-only scripts that can't be parsed
            continue
        except Exception:
            continue

    assert not violations, f"Found unguarded sys.exit calls in: {violations}"
