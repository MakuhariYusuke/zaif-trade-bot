"""Project-level import fixes executed very early on interpreter startup.

379# 修正: ローカル stable_baselines3 スタブ (_sb3_test_stub/) が
本物の pip 版 SB3 をシャドウしていたため、SAC.learn() が何もせず
SAC.predict() が常に int(0) を返していた。

_prefer_local_package() を無効化し、pip版 SB3 を使用する。
テスト用スタブが必要な場合は tests/conftest.py で注入する。
"""

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 1. Local SB3 stub preference — DISABLED (379#)
#    ローカルスタブが本物のSB3をシャドウし、訓練が無操作になっていた。
# ---------------------------------------------------------------------------
def _prefer_local_package() -> bool:
    """Disabled: no longer prefers local stub over pip-installed SB3."""
    return False


_prefer_local_package()


# ---------------------------------------------------------------------------
# 2. Safety-net: replace lingering in-memory stubs with file-backed modules
# ---------------------------------------------------------------------------
def _replace_stub_with_filebacked(fullname: str, candidate_path: Path) -> None:
    """If *fullname* is in ``sys.modules`` but lacks ``__file__`` / ``__spec__``,
    load the file-backed module from *candidate_path* instead."""
    try:
        m = sys.modules.get(fullname)
        if m is None:
            return
        if getattr(m, "__file__", None) and getattr(m, "__spec__", None):
            return  # already file-backed
        if not candidate_path.exists():
            return
        spec_fb = importlib.util.spec_from_file_location(fullname, str(candidate_path))
        if spec_fb is None:
            return
        real_mod = importlib.util.module_from_spec(spec_fb)
        if candidate_path.name == "__init__.py":
            real_mod.__path__ = [str(candidate_path.parent)]
        try:
            spec_fb.loader.exec_module(real_mod)  # type: ignore[union-attr]
            sys.modules[fullname] = real_mod
        except Exception:
            try:
                m.__file__ = str(candidate_path)
                m.__spec__ = importlib.util.spec_from_loader(fullname, loader=None)
            except Exception:
                pass
    except Exception:
        pass


try:
    # 379# DISABLED — ローカルスタブは _sb3_test_stub/ にリネーム済み
    pass
except Exception:
    pass
