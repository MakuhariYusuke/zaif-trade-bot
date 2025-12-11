import ast
from pathlib import Path


def is_guarded_sys_exit(node: ast.AST) -> bool:
    """Return True if sys.exit call is under if __name__ == '__main__' guard.
    We traverse AST parents (approx) to check if any enclosing If node is comparing __name__ to '__main__'"""
    # find parent nodes by walking AST; we approximate by checking ancestors
    # We'll implement by scanning if there's an If node that contains this node with test comparing __name__
    return False  # placeholder: we'll detect unguarded sys.exit via simpler string scanning


def test_no_sys_exit_unguarded_in_root_scripts():
    repo_root = Path(__file__).resolve().parents[3]
    script_paths = [
        repo_root / "train_sac_v446_fixed.py",
        repo_root / "train_sac_v449_corrected.py",
        repo_root / "backtest" / "signal_guidance_backtest.py",
        repo_root / "quick_analysis_v446.py",
    ]
    failures = []
    for p in script_paths:
        if not p.exists():
            continue
        content = p.read_text(encoding="utf-8")
        # naive check: ensure sys.exit isn't present outside __main__ by seeing if 'sys.exit' occurs
        # and check that an `if __name__ == "__main__"` exists and the sys.exit is within that block
        if "sys.exit" in content:
            # quick check: find location and ensure prior if __name__ exists
            main_guard_index = content.find('if __name__ == "__main__"')
            exit_index = content.find("sys.exit")
            if main_guard_index == -1 or exit_index < main_guard_index:
                failures.append(str(p))

    assert not failures, f"Found unguarded sys.exit in scripts: {failures}"
