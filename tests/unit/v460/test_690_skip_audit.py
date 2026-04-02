from __future__ import annotations

import ast
from dataclasses import asdict
from pathlib import Path

from scripts.v460.lib import cancel_reasons as CR
from scripts.v460.lib.cancel_reason_taxonomy import (
    CANCEL_REASON_REGISTRY,
    SkipCategory,
    validate_skip_consistency,
)


ORCHESTRATOR_FILES = [
    "scripts/v460/lib/orchestrator_pre_cycle.py",
    "scripts/v460/lib/orchestrator_mid_cycle.py",
    "scripts/v460/lib/orchestrator_balance.py",
]


def _resolve_reason(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "CR"
    ):
        return getattr(CR, node.attr, None)
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "gate_result"
        and node.attr == "cancel_reason"
    ):
        return "__dynamic_gate_reason__"
    return None


def _resolve_bool(node: ast.AST) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def _iter_execute_skip_calls() -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    for rel_path in ORCHESTRATOR_FILES:
        path = Path(rel_path)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "_execute_skip"
            ):
                continue
            kw_map = {
                kw.arg: kw.value
                for kw in node.keywords
                if kw.arg is not None
            }
            calls.append(
                {
                    "path": rel_path,
                    "line": node.lineno,
                    "cancel_reason": _resolve_reason(kw_map["cancel_reason"]),
                    "update_last_side": _resolve_bool(kw_map["update_last_side"]) if "update_last_side" in kw_map else None,
                    "heartbeat": _resolve_bool(kw_map["heartbeat"]) if "heartbeat" in kw_map else None,
                    "source_line": source.splitlines()[node.lineno - 1],
                    "source_block": "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno]),
                }
            )
    return sorted(calls, key=lambda item: (str(item["path"]), int(item["line"])))


class TestCancelReasonRegistry:
    def test_all_reasons_have_category(self) -> None:
        for reason, meta in CANCEL_REASON_REGISTRY.items():
            payload = asdict(meta)
            assert payload["category"] is not None
            assert payload["description"] != ""
            assert payload["reason"] == reason

    def test_env_halt_reasons_are_false(self) -> None:
        for meta in CANCEL_REASON_REGISTRY.values():
            if meta.category == SkipCategory.ENV_HALT:
                assert meta.expected_update_last_side is False

    def test_side_and_gate_block_reasons_are_true(self) -> None:
        for meta in CANCEL_REASON_REGISTRY.values():
            if meta.category in {SkipCategory.SIDE_BLOCK, SkipCategory.GATE_BLOCK, SkipCategory.RESOURCE}:
                assert meta.expected_update_last_side is True

    def test_no_duplicate_registry_reasons(self) -> None:
        assert len(CANCEL_REASON_REGISTRY) == len(set(CANCEL_REASON_REGISTRY))


class TestCallSiteAudit:
    def test_all_call_sites_have_explicit_update_last_side(self) -> None:
        calls = _iter_execute_skip_calls()

        assert calls
        missing = [c for c in calls if c["update_last_side"] is None]
        assert missing == [], missing

    def test_cancel_reason_matches_registry(self) -> None:
        calls = _iter_execute_skip_calls()

        unknown = [
            c for c in calls
            if c["cancel_reason"] not in CANCEL_REASON_REGISTRY
            and c["cancel_reason"] != "__dynamic_gate_reason__"
        ]
        assert unknown == [], unknown

    def test_update_last_side_matches_category(self) -> None:
        calls = _iter_execute_skip_calls()

        mismatches = [
            c for c in calls
            if c["cancel_reason"] != "__dynamic_gate_reason__"
        ]
        mismatches = [
            c for c in mismatches
            if not validate_skip_consistency(
                str(c["cancel_reason"]),
                bool(c["update_last_side"]),
            )
        ]
        assert mismatches == [], mismatches

    def test_inline_comments_present(self) -> None:
        calls = _iter_execute_skip_calls()

        assert all("#" in str(c["source_block"]) for c in calls)


class TestSkipCeremonyConsistency:
    def test_env_halt_always_has_heartbeat(self) -> None:
        for call in _iter_execute_skip_calls():
            if call["cancel_reason"] == "__dynamic_gate_reason__":
                continue
            meta = CANCEL_REASON_REGISTRY[str(call["cancel_reason"])]
            if meta.category == SkipCategory.ENV_HALT:
                assert call["heartbeat"] is True, call

    def test_side_block_preserves_sleep_default(self) -> None:
        for call in _iter_execute_skip_calls():
            if call["cancel_reason"] == "__dynamic_gate_reason__":
                continue
            meta = CANCEL_REASON_REGISTRY[str(call["cancel_reason"])]
            if meta.category in {SkipCategory.SIDE_BLOCK, SkipCategory.GATE_BLOCK, SkipCategory.RESOURCE}:
                assert call["update_last_side"] is True
