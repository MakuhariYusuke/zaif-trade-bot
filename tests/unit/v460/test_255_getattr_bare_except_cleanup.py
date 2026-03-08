"""255# getattr 排除 + bare except → debug log 改善テスト.

対象:
- skip_gate_evaluator: _gate_buy/_gate_sell, hot_reload_check_interval_sec getattr 排除
- order_monitor: stale_reprice_skip_gate_offset getattr 排除
- resilience / pnl_measurer / lock_manager / ob_utils / fill_cycle_executor: bare except → logger.debug
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

import pytest
from tests.unit.v460._fill_test_source import read_class_method_source

_LIB = Path(__file__).resolve().parents[3] / "scripts" / "v460" / "lib"
_SKIP_GATE_EVALUATOR = _LIB / "skip_gate_evaluator.py"
_ORDER_MONITOR = _LIB / "order_monitor.py"


@lru_cache(maxsize=None)
def _read_lib_source(module_path: str) -> str:
    return (_LIB / f"{module_path}.py").read_text(encoding="utf-8-sig")


@lru_cache(maxsize=None)
def _parse_lib_source(module_path: str) -> ast.AST:
    return ast.parse(_read_lib_source(module_path))

def _code_lines(source: str) -> list[str]:
    """コメント・空行を除いた実コード行のみ返す."""
    return [
        ln
        for ln in source.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


class TestGetAttrRemoval:
    """getattr 排除の確認."""

    def test_select_gate_for_side_no_getattr(self) -> None:
        """_select_gate_for_side が getattr を使っていない."""
        src = read_class_method_source(
            _SKIP_GATE_EVALUATOR,
            "SkipGateEvaluator",
            "_select_gate_for_side",
        )
        code = _code_lines(src)
        assert not any("getattr" in ln for ln in code)

    def test_evaluate_no_gate_getattr(self) -> None:
        """evaluate() 内の _gate_buy/_gate_sell 参照が getattr でない."""
        src = read_class_method_source(
            _SKIP_GATE_EVALUATOR,
            "SkipGateEvaluator",
            "evaluate",
        )
        code = _code_lines(src)
        gate_lines = [ln for ln in code if "_gate_buy" in ln or "_gate_sell" in ln]
        assert gate_lines  # gate 参照行が存在
        assert not any("getattr" in ln for ln in gate_lines)

    def test_hot_reload_no_getattr(self) -> None:
        """_check_and_reload_model が hot_reload_check_interval_sec の getattr を使わない."""
        src = read_class_method_source(
            _SKIP_GATE_EVALUATOR,
            "SkipGateEvaluator",
            "_check_and_reload_model",
        )
        code = _code_lines(src)
        interval_lines = [ln for ln in code if "hot_reload" in ln or "interval" in ln]
        assert not any("getattr" in ln for ln in interval_lines)

    def test_order_monitor_reprice_no_getattr(self) -> None:
        """_should_block_reprice_with_skip_gate が config getattr を使わない."""
        src = read_class_method_source(
            _ORDER_MONITOR,
            "OrderMonitor",
            "_should_block_reprice_with_skip_gate",
        )
        code = _code_lines(src)
        assert not any("getattr" in ln and "stale_reprice" in ln for ln in code)


class TestBareExceptImproved:
    """bare except → logger.debug 改善の確認."""

    @pytest.mark.parametrize(
        "module_path,func_name",
        [
            ("resilience", "get_health_status"),
            ("pnl_measurer", "measure_pnl"),
            ("lock_manager", "heartbeat"),

        ],
    )
    def test_no_bare_except_pass(self, module_path: str, func_name: str) -> None:
        """except Exception: の直後が pass でなく logger.debug を含む."""
        source = _read_lib_source(module_path)
        tree = _parse_lib_source(module_path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                func_src = ast.get_source_segment(source, node)
                assert func_src is not None
                # "except Exception:" の後に "pass" + 改行のみ (= bare) がない
                lines = func_src.splitlines()
                for i, ln in enumerate(lines):
                    if "except Exception:" in ln:
                        # 次の非空行が pass だけではないことを確認
                        for j in range(i + 1, min(i + 4, len(lines))):
                            stripped = lines[j].strip()
                            if stripped and stripped != "#":
                                assert stripped != "pass", (
                                    f"{module_path}.{func_name}: bare 'except Exception: pass' found at line {j}"
                                )
                                break

    @pytest.mark.parametrize(
        "module_path,func_name",
        [
            ("ob_utils", "bid_depth_volume"),
            ("ob_utils", "ask_depth_volume"),
        ],
    )
    def test_ob_utils_no_bare_except(self, module_path: str, func_name: str) -> None:
        """ob_utils の depth_volume 系が bare except pass でない."""
        source = _read_lib_source(module_path)
        tree = _parse_lib_source(module_path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                func_src = ast.get_source_segment(source, node)
                assert func_src is not None
                assert "logger.debug" in func_src, f"{func_name} should use logger.debug in except"
