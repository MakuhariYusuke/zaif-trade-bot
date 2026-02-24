"""FillTestRunner ソースファイル読込ヘルパー.

mixin 分割 (163#) 後、FillTestRunner のロジックは 4 ファイルに分散している。
ソース解析テストで全体を検索する場合はこのヘルパーを使用すること。

WARNING (God Object 防止):
    FillTestRunner に新メソッドを追加する場合は、適切な mixin ファイルに配置せよ。
    run_fill_test.py 本体は __init__ + main のみ。詳細は各 mixin のヘッダを参照。
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_FILL_TEST_RUNNER_SOURCES: list[Path] = [
    _PROJECT_ROOT / "scripts" / "v460" / "run_fill_test.py",
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "fill_record_helpers.py",
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "fill_cycle_executor.py",
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "fill_loop_orchestrator.py",
]

# 個別ファイルパス (Option A テスト用)
FILL_LOOP_ORCHESTRATOR = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "fill_loop_orchestrator.py"
)
FILL_CYCLE_EXECUTOR = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "fill_cycle_executor.py"
)
FILL_RECORD_HELPERS = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "fill_record_helpers.py"
)
FILL_TEST_RUNNER_MAIN = (
    _PROJECT_ROOT / "scripts" / "v460" / "run_fill_test.py"
)


def read_fill_test_runner_source() -> str:
    """FillTestRunner 全ソース (本体 + 3 mixin) を連結して返す.

    NOTE: ファイル間の行番号は不連続。位置比較テストでは個別ファイルを使用せよ。
    """
    return "\n".join(
        p.read_text(encoding="utf-8") for p in _FILL_TEST_RUNNER_SOURCES
    )
