#!/usr/bin/env python3
"""Inventory `typing.Any` usage for staged type-debt reduction.

This script is intentionally lightweight and dependency-free so it can run in
CI and local Windows/Linux shells with the same behavior.

Focus:
- Measure current `Any` usage before refactoring.
- Separate "type debt" (`Any` in annotations / aliases) from runtime fallback.
- Provide top files/directories to prioritize incremental cleanup.

Examples:
    python scripts/quality/any_inventory.py
    python scripts/quality/any_inventory.py --roots ztb scripts/v460 --top 20
    python scripts/quality/any_inventory.py --json-out results/type_any_inventory.json
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from tokenize import NAME, TokenError, tokenize


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
DEFAULT_ROOTS = ("ztb", "scripts/v460")
DEFAULT_EXCLUDES = (
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
)


@dataclass
class FileAnyStats:
    path: str
    any_total_tokens: int
    any_import_tokens: int
    any_annotation_tokens: int
    any_alias_tokens: int
    any_runtime_tokens: int

    @property
    def any_type_debt_tokens(self) -> int:
        return self.any_annotation_tokens + self.any_alias_tokens


def _contains_excluded_part(path: Path, excluded_parts: tuple[str, ...]) -> bool:
    parts = set(path.parts)
    return any(part in parts for part in excluded_parts)


def _iter_python_files(roots: list[str], excluded_parts: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for raw_root in roots:
        root_path = (PROJECT_ROOT / raw_root).resolve()
        if not root_path.exists():
            continue
        if root_path.is_file() and root_path.suffix == ".py":
            if not _contains_excluded_part(root_path.relative_to(PROJECT_ROOT), excluded_parts):
                files.append(root_path)
            continue
        for path in root_path.rglob("*.py"):
            rel = path.relative_to(PROJECT_ROOT)
            if _contains_excluded_part(rel, excluded_parts):
                continue
            files.append(path)
    return sorted(set(files))


def _count_any_tokens(text: str) -> int:
    try:
        token_stream = tokenize(BytesIO(text.encode("utf-8")).readline)
        return sum(1 for tok in token_stream if tok.type == NAME and tok.string == "Any")
    except (TokenError, UnicodeEncodeError):
        return 0


def _count_any_refs(expr: ast.AST | None) -> int:
    if expr is None:
        return 0
    count = 0
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and node.id == "Any":
            count += 1
        elif (
            isinstance(node, ast.Attribute)
            and node.attr == "Any"
            and isinstance(node.value, ast.Name)
            and node.value.id == "typing"
        ):
            count += 1
    return count


def _count_import_any(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {"typing", "typing_extensions"}:
            count += sum(1 for alias in node.names if alias.name == "Any")
    return count


def _is_alias_assignment(node: ast.Assign | ast.AnnAssign) -> bool:
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    else:
        targets = [node.target]
    for target in targets:
        if isinstance(target, ast.Name):
            name = target.id
            if name and (name[0].isupper() or name.isupper()):
                return True
    return False


def _count_annotation_and_alias_any(tree: ast.AST) -> tuple[int, int]:
    annotation_any = 0
    alias_any = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotation_any += _count_any_refs(node.returns)
            for arg in (
                list(node.args.posonlyargs)
                + list(node.args.args)
                + list(node.args.kwonlyargs)
            ):
                annotation_any += _count_any_refs(arg.annotation)
            if node.args.vararg:
                annotation_any += _count_any_refs(node.args.vararg.annotation)
            if node.args.kwarg:
                annotation_any += _count_any_refs(node.args.kwarg.annotation)
        elif isinstance(node, ast.AnnAssign):
            annotation_any += _count_any_refs(node.annotation)
            if _is_alias_assignment(node):
                alias_any += _count_any_refs(node.value)
        elif isinstance(node, ast.Assign) and _is_alias_assignment(node):
            alias_any += _count_any_refs(node.value)

    return annotation_any, alias_any


def _directory_key(path: str) -> str:
    p = Path(path)
    if len(p.parts) >= 3:
        return str(Path(p.parts[0]) / p.parts[1])
    if len(p.parts) >= 2:
        return str(Path(p.parts[0]) / p.parts[1])
    return p.parts[0] if p.parts else "."


def _collect_stats(roots: list[str], excluded_parts: tuple[str, ...]) -> list[FileAnyStats]:
    stats: list[FileAnyStats] = []
    for path in _iter_python_files(roots, excluded_parts):
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue

        any_total = _count_any_tokens(text)
        any_import = _count_import_any(tree)
        any_annotation, any_alias = _count_annotation_and_alias_any(tree)
        any_runtime = max(any_total - any_import - any_annotation - any_alias, 0)

        stats.append(
            FileAnyStats(
                path=str(path.relative_to(PROJECT_ROOT)),
                any_total_tokens=any_total,
                any_import_tokens=any_import,
                any_annotation_tokens=any_annotation,
                any_alias_tokens=any_alias,
                any_runtime_tokens=any_runtime,
            )
        )

    return stats


def _print_report(stats: list[FileAnyStats], top_n: int) -> None:
    scanned_files = len(stats)
    total_any = sum(s.any_total_tokens for s in stats)
    import_any = sum(s.any_import_tokens for s in stats)
    ann_any = sum(s.any_annotation_tokens for s in stats)
    alias_any = sum(s.any_alias_tokens for s in stats)
    runtime_any = sum(s.any_runtime_tokens for s in stats)
    type_debt_any = ann_any + alias_any

    print("Any Inventory")
    print("=============")
    print(f"scanned_files: {scanned_files}")
    print(f"any_total_tokens: {total_any}")
    print(f"any_import_tokens: {import_any}")
    print(f"any_annotation_tokens: {ann_any}")
    print(f"any_alias_tokens: {alias_any}")
    print(f"any_type_debt_tokens: {type_debt_any}")
    print(f"any_runtime_tokens: {runtime_any}")
    print("")

    if not stats:
        return

    ranked_files = sorted(
        stats,
        key=lambda s: (s.any_type_debt_tokens, s.any_total_tokens, s.path),
        reverse=True,
    )[:top_n]
    print(f"Top {len(ranked_files)} files by type-debt Any")
    print("path|type_debt|annotation|alias|runtime|import|total")
    for item in ranked_files:
        print(
            f"{item.path}|{item.any_type_debt_tokens}|{item.any_annotation_tokens}|"
            f"{item.any_alias_tokens}|{item.any_runtime_tokens}|"
            f"{item.any_import_tokens}|{item.any_total_tokens}"
        )
    print("")

    by_dir: dict[str, int] = {}
    for item in stats:
        key = _directory_key(item.path)
        by_dir[key] = by_dir.get(key, 0) + item.any_type_debt_tokens

    ranked_dirs = sorted(by_dir.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:top_n]
    print(f"Top {len(ranked_dirs)} directories by type-debt Any")
    print("directory|type_debt")
    for directory, count in ranked_dirs:
        print(f"{directory}|{count}")


def _write_json(path: Path, roots: list[str], excluded_parts: tuple[str, ...], stats: list[FileAnyStats]) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "roots": roots,
        "excluded_parts": list(excluded_parts),
        "totals": {
            "scanned_files": len(stats),
            "any_total_tokens": sum(s.any_total_tokens for s in stats),
            "any_import_tokens": sum(s.any_import_tokens for s in stats),
            "any_annotation_tokens": sum(s.any_annotation_tokens for s in stats),
            "any_alias_tokens": sum(s.any_alias_tokens for s in stats),
            "any_type_debt_tokens": sum(s.any_type_debt_tokens for s in stats),
            "any_runtime_tokens": sum(s.any_runtime_tokens for s in stats),
        },
        "files": [asdict(s) | {"any_type_debt_tokens": s.any_type_debt_tokens} for s in stats],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory typing.Any usage.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=list(DEFAULT_ROOTS),
        help=f"Relative roots to scan (default: {' '.join(DEFAULT_ROOTS)})",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Top N files/directories to print (default: 20)",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional output path for JSON report.",
    )
    parser.add_argument(
        "--max-type-any",
        type=int,
        default=None,
        help="Fail if total type-debt Any (annotation + alias) exceeds this value.",
    )
    parser.add_argument(
        "--max-total-any",
        type=int,
        default=None,
        help="Fail if total Any tokens exceeds this value.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    roots = [r.strip() for r in args.roots if r.strip()]
    excluded_parts = tuple(DEFAULT_EXCLUDES)
    stats = _collect_stats(roots=roots, excluded_parts=excluded_parts)

    _print_report(stats, top_n=max(args.top, 1))

    total_any = sum(s.any_total_tokens for s in stats)
    type_debt_any = sum(s.any_type_debt_tokens for s in stats)

    if args.json_out:
        _write_json(
            path=(PROJECT_ROOT / args.json_out).resolve(),
            roots=roots,
            excluded_parts=excluded_parts,
            stats=stats,
        )
        print(f"\nJSON report written: {args.json_out}")

    failed = False
    if args.max_total_any is not None and total_any > args.max_total_any:
        print(
            f"ERROR: any_total_tokens={total_any} exceeds --max-total-any={args.max_total_any}"
        )
        failed = True
    if args.max_type_any is not None and type_debt_any > args.max_type_any:
        print(
            f"ERROR: any_type_debt_tokens={type_debt_any} exceeds "
            f"--max-type-any={args.max_type_any}"
        )
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
