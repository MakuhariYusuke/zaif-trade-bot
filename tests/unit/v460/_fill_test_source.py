"""FillTestRunner ソースファイル読込ヘルパー.

mixin 分割 (163#, 323#) 後、FillTestRunner のロジックは 6 ファイルに分散している。
ソース解析テストで全体を検索する場合はこのヘルパーを使用すること。

WARNING (God Object 防止):
    FillTestRunner に新メソッドを追加する場合は、適切な mixin ファイルに配置せよ。
    run_fill_test.py 本体は __init__ + main のみ。詳細は各 mixin のヘッダを参照。
"""

from __future__ import annotations

import ast
import inspect
from functools import lru_cache
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_FILL_TEST_RUNNER_SOURCES: list[Path] = [
    _PROJECT_ROOT / "scripts" / "v460" / "run_fill_test.py",
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "fill_record_helpers.py",
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "fill_cycle_executor.py",
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "fill_record_builder.py",
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "pre_order_adjustments.py",
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "offset_pipeline.py",          # 460#
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "orchestrator_guards.py",
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "orchestrator_lifecycle.py",
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "orchestrator_post_cycle.py",
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "orchestrator_pre_cycle.py",
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "orchestrator_balance.py",      # 332#
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "orchestrator_mid_cycle.py",    # 332#
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
FILL_RECORD_BUILDER = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "fill_record_builder.py"
)
PRE_ORDER_ADJUSTMENTS = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "pre_order_adjustments.py"
)
OFFSET_PIPELINE = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "offset_pipeline.py"
)
ORCHESTRATOR_GUARDS = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "orchestrator_guards.py"
)
ORCHESTRATOR_LIFECYCLE = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "orchestrator_lifecycle.py"
)
ORCHESTRATOR_POST_CYCLE = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "orchestrator_post_cycle.py"
)
ORCHESTRATOR_PRE_CYCLE = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "orchestrator_pre_cycle.py"
)
ORCHESTRATOR_MID_CYCLE = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "orchestrator_mid_cycle.py"
)
ORCHESTRATOR_BALANCE = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "orchestrator_balance.py"
)
FILL_TEST_RUNNER_MAIN = (
    _PROJECT_ROOT / "scripts" / "v460" / "run_fill_test.py"
)
CYCLE_GATE_AGGREGATOR = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "cycle_gate_aggregator.py"
)
FILL_TEST_CLI = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "fill_test_cli.py"
)
MAKER_PRICE = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "maker_price.py"
)
MAKER_RISK_GUARDS = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "maker_risk_guards.py"
)
MAKER_REGIME_BOOST = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "maker_regime_boost.py"
)
MAKER_MICROSTRUCTURE = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "maker_microstructure.py"
)
ADAPTATION_ENGINE = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "adaptation_engine.py"
)
BALANCE_CHECKER = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "balance_checker.py"
)
ORDER_MONITOR = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "order_monitor.py"
)
OB_RECORDER = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "ob_recorder.py"
)
FILL_CONFIG_PARSER = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "fill_config_parser.py"
)
FILL_CONFIG = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "fill_config.py"
)
CONFIG_HOT_RELOAD = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "config_hot_reload.py"
)
EVENT_LOGGER = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "event_logger.py"
)
FAST_FILL_DEFENSE = (
    _PROJECT_ROOT / "ztb" / "trading" / "risk" / "fast_fill_defense.py"
)
REGIME_DETECTOR = (
    _PROJECT_ROOT / "ztb" / "trading" / "signal" / "regime" / "regime_detector.py"
)
OB_UTILS = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "ob_utils.py"
)
SKIP_GATE_EVALUATOR = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "skip_gate_evaluator.py"
)
SKIP_GATE_MODEL_LOADER = (
    _PROJECT_ROOT / "scripts" / "v460" / "lib" / "skip_gate_model_loader.py"
)
HINDSIGHT_FILTER = (
    _PROJECT_ROOT / "scripts" / "v460" / "analysis" / "hindsight_filter.py"
)


@lru_cache(maxsize=None)
def read_source_text(path: Path) -> str:
    """UTF-8 テキスト読込をキャッシュする (BOM 自動除去)."""
    return path.read_text(encoding="utf-8-sig")


@lru_cache(maxsize=None)
def parse_source_tree(path: Path) -> ast.AST:
    """ソースの AST をキャッシュする."""
    return ast.parse(read_source_text(path))


@lru_cache(maxsize=None)
def _method_source_index() -> dict[str, str]:
    """FillTestRunner 本体 + mixin 群の method source を一括 index 化する."""
    index: dict[str, str] = {}
    for path in _FILL_TEST_RUNNER_SOURCES:
        source = read_source_text(path)
        lines = source.splitlines()
        tree = parse_source_tree(path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
                continue
            index.setdefault(
                node.name,
                "\n".join(lines[node.lineno - 1:node.end_lineno]),
            )
    return index


def read_fill_test_runner_source() -> str:
    """FillTestRunner 全ソース (本体 + 3 mixin) を連結して返す.

    NOTE: ファイル間の行番号は不連続。位置比較テストでは個別ファイルを使用せよ。
    """
    return "\n".join(
        read_source_text(p) for p in _FILL_TEST_RUNNER_SOURCES
    )


@lru_cache(maxsize=None)
def read_fill_test_method_source(method_name: str) -> str:
    """FillTestRunner 本体または mixin 群からメソッド source を返す."""
    try:
        return _method_source_index()[method_name]
    except KeyError as exc:
        raise KeyError(f"FillTestRunner method not found: {method_name}") from exc


@lru_cache(maxsize=None)
def read_class_method_source(path: Path, class_name: str, method_name: str) -> str:
    """任意ファイルから class method source を返す."""
    source = read_source_text(path)
    lines = source.splitlines()
    tree = parse_source_tree(path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for child in node.body:
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if child.name != method_name:
                continue
            return "\n".join(lines[child.lineno - 1:child.end_lineno])
    raise KeyError(f"{class_name}.{method_name} not found in {path}")


@lru_cache(maxsize=None)
def read_function_source(path: Path, function_name: str) -> str:
    """任意ファイルから top-level function source を返す."""
    source = read_source_text(path)
    lines = source.splitlines()
    tree = parse_source_tree(path)
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != function_name:
            continue
        return "\n".join(lines[node.lineno - 1:node.end_lineno])
    raise KeyError(f"{function_name} not found in {path}")


@lru_cache(maxsize=None)
def read_inspect_source(obj: object) -> str:
    """inspect.getsource() の共有キャッシュ."""
    return inspect.getsource(obj)
