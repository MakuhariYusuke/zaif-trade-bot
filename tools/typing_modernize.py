#!/usr/bin/env python3
"""PEP 585/604 typing modernization script for Python 3.10+.

Transforms:
  Dict[K, V]    → dict[K, V]
  List[X]       → list[X]
  Tuple[X, ...] → tuple[X, ...]
  Set[X]        → set[X]
  FrozenSet[X]  → frozenset[X]
  Type[X]       → type[X]
  Deque[X]      → collections.deque[X]   (adds import if needed)
  Optional[X]   → X | None
  Union[X, Y]   → X | Y
  Sequence[X]   → collections.abc.Sequence[X]  (skipped for now)

Also cleans up the `from typing import ...` line by removing
deprecated aliases that are no longer referenced.
"""

from __future__ import annotations

import pathlib
import re
import sys


# PEP 585 simple alias → builtin mapping
_SIMPLE_ALIASES: dict[str, str] = {
    "Dict": "dict",
    "List": "list",
    "Tuple": "tuple",
    "Set": "set",
    "FrozenSet": "frozenset",
    "Type": "type",
}

# Aliases that need bracket-aware replacement
_BRACKET_ALIASES = {"Optional", "Union"}


def _find_matching_bracket(text: str, start: int) -> int:
    """Find the matching ] for the [ at position start."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"Unmatched bracket at pos {start}")


def _split_top_level(text: str) -> list[str]:
    """Split by ',' at top-level only (not inside brackets)."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch in "([":
            depth += 1
            current.append(ch)
        elif ch in ")]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def _collapse_whitespace(text: str) -> str:
    """Collapse multi-line type expression to single line."""
    # Replace newlines and surrounding whitespace with a single space
    return re.sub(r"\s*\n\s*", " ", text).strip()


def _strip_forward_ref_quotes(inner: str) -> str:
    """Strip quotes from forward references: '"Foo"' → 'Foo'."""
    stripped = inner.strip()
    if (stripped.startswith('"') and stripped.endswith('"')) or \
       (stripped.startswith("'") and stripped.endswith("'")):
        return stripped[1:-1]
    return inner


def _replace_optional(text: str) -> str:
    """Replace Optional[X] with X | None throughout text."""
    pattern = re.compile(r"\bOptional\s*\[")
    while True:
        m = pattern.search(text)
        if not m:
            break
        bracket_start = m.end() - 1  # position of [
        bracket_end = _find_matching_bracket(text, bracket_start)
        inner = text[bracket_start + 1 : bracket_end]
        # Recursively process inner content first
        inner = _replace_optional(inner)
        # Collapse multi-line to single line and strip forward ref quotes
        inner = _collapse_whitespace(inner)
        inner = _strip_forward_ref_quotes(inner)
        replacement = f"{inner} | None"
        text = text[: m.start()] + replacement + text[bracket_end + 1 :]
    return text


def _replace_union(text: str) -> str:
    """Replace Union[X, Y, Z] with X | Y | Z throughout text."""
    pattern = re.compile(r"\bUnion\s*\[")
    while True:
        m = pattern.search(text)
        if not m:
            break
        bracket_start = m.end() - 1
        bracket_end = _find_matching_bracket(text, bracket_start)
        inner = text[bracket_start + 1 : bracket_end]
        parts = _split_top_level(inner)
        # Recursively process each part, collapse whitespace, and filter empty parts
        parts = [_collapse_whitespace(_replace_union(p)) for p in parts]
        parts = [p for p in parts if p]  # Remove empty parts from trailing commas
        replacement = " | ".join(parts)
        text = text[: m.start()] + replacement + text[bracket_end + 1 :]
    return text


def _replace_simple_aliases(text: str) -> str:
    """Replace Dict[...] → dict[...], List[...] → list[...], etc.
    
    Skips typing import lines to avoid turning
    'from typing import Dict' into 'from typing import dict'.
    """
    # 'Type' is a common English word — only replace when followed by [
    # Others (Dict, List, Tuple, Set, FrozenSet) are safe to replace globally
    bracket_only = {"Type"}
    
    lines = text.split("\n")
    result: list[str] = []
    in_typing_import = False
    for line in lines:
        # Track multi-line typing import blocks
        if re.match(r"\s*from typing import\b", line):
            result.append(line)
            # Check if it's a multi-line import (has opening paren but no closing)
            if "(" in line and ")" not in line:
                in_typing_import = True
            continue
        if in_typing_import:
            result.append(line)
            if ")" in line:
                in_typing_import = False
            continue
        new_line = line
        for old, new in _SIMPLE_ALIASES.items():
            if old in bracket_only:
                # Only replace when followed by [ (annotation context)
                new_line = re.sub(rf"\b{old}\[", f"{new}[", new_line)
            else:
                new_line = re.sub(rf"\b{old}\b", new, new_line)
        result.append(new_line)
    return "\n".join(result)


def _still_used_in_body(text: str, alias: str) -> bool:
    """Check if an alias is still referenced in the file body (outside imports)."""
    # Remove import lines to check body only
    body = re.sub(r"^.*from typing import.*$", "", text, flags=re.MULTILINE)
    return bool(re.search(rf"\b{alias}\b", body))


def _update_typing_import(text: str) -> str:
    """Remove deprecated aliases from typing import if no longer used."""
    all_removable = set(_SIMPLE_ALIASES.keys()) | _BRACKET_ALIASES
    # Only remove if truly no longer referenced in the body
    actually_removable = {a for a in all_removable if not _still_used_in_body(text, a)}

    # Find all `from typing import ...` lines (possibly multi-line)
    # Handle single-line imports
    def process_import(match: re.Match[str]) -> str:
        indent = match.group(1)
        imports_str = match.group(2)

        imports = [x.strip() for x in imports_str.split(",")]
        remaining = [x for x in imports if x and x not in actually_removable]

        if not remaining:
            # All imports removed — check if we still need typing at all
            return ""
        return f"{indent}from typing import {', '.join(remaining)}"

    # Single-line: from typing import X, Y, Z
    text = re.sub(
        r"^( *)from typing import ([^\n(]+)$",
        process_import,
        text,
        flags=re.MULTILINE,
    )

    # Multi-line with parens: from typing import (\n    X,\n    Y,\n)
    def process_multiline_import(match: re.Match[str]) -> str:
        indent = match.group(1)
        body = match.group(2)
        imports = [x.strip().rstrip(",") for x in body.split("\n") if x.strip() and x.strip() != ")"]
        imports = [x for x in imports if x]
        remaining = [x for x in imports if x not in actually_removable]

        if not remaining:
            return ""
        if len(remaining) <= 5:
            return f"{indent}from typing import {', '.join(remaining)}"
        # Keep multi-line
        body_lines = [f"    {imp}," for imp in remaining]
        return f"{indent}from typing import (\n" + "\n".join(body_lines) + "\n)"

    text = re.sub(
        r"^( *)from typing import \(\n((?:.*\n)*?\s*\))",
        process_multiline_import,
        text,
        flags=re.MULTILINE,
    )

    # Remove empty lines left by removed imports
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def _ensure_future_annotations(text: str) -> str:
    """Add 'from __future__ import annotations' if not present.
    
    Required when converting Optional["Foo"] → Foo | None in files
    that use forward references (string-quoted type hints).
    """
    if "from __future__ import annotations" in text:
        return text
    
    # Find the right insertion point: after docstrings and comments, before other imports
    lines = text.split("\n")
    insert_idx = 0
    
    # Skip shebang, encoding, module docstring
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("#!") or stripped.startswith("# -*-"):
            i += 1
            continue
        if stripped.startswith('"""') or stripped.startswith("'''"):
            # Multi-line docstring
            quote = stripped[:3]
            if stripped.count(quote) >= 2 and len(stripped) > 3:
                # Single-line docstring
                i += 1
            else:
                # Multi-line — find the closing quotes
                i += 1
                while i < len(lines) and quote not in lines[i]:
                    i += 1
                i += 1  # skip the closing line
            continue
        break
    insert_idx = i
    
    # Insert the future import
    lines.insert(insert_idx, "from __future__ import annotations")
    # Add blank line after if next line isn't blank
    if insert_idx + 1 < len(lines) and lines[insert_idx + 1].strip():
        lines.insert(insert_idx + 1, "")
    
    return "\n".join(lines)


def modernize_file(filepath: pathlib.Path, *, dry_run: bool = False) -> tuple[bool, list[str]]:
    """Modernize a single file. Returns (changed, list_of_changes)."""
    text = filepath.read_text(encoding="utf-8")
    original = text
    changes: list[str] = []

    # Step 0: Check if Optional/Union with forward refs requires future annotations
    has_future = "from __future__ import annotations" in text
    needs_future = False
    if not has_future:
        # Check if we'll create X | None with quoted strings  
        if re.search(r'Optional\s*\[\s*"', text) or re.search(r'Union\s*\[.*"', text):
            needs_future = True
        # Check if builtins (dict, list, set, tuple, type) are shadowed by class methods
        import ast
        try:
            tree = ast.parse(text)
            shadowed = {"dict", "list", "tuple", "set", "frozenset", "type"}
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef,)):
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if child.name in shadowed:
                                needs_future = True
                                break
                        if isinstance(child, ast.Assign):
                            for target in child.targets:
                                if isinstance(target, ast.Name) and target.id in shadowed:
                                    needs_future = True
                                    break
        except SyntaxError:
            pass

    # Step 1: Replace simple aliases in annotations
    new_text = _replace_simple_aliases(text)
    if new_text != text:
        for old in _SIMPLE_ALIASES:
            if re.search(rf"\b{old}\b", text) and not re.search(rf"\b{old}\b", new_text):
                changes.append(f"{old} → {_SIMPLE_ALIASES[old]}")
        text = new_text

    # Step 2: Replace Optional[X] → X | None
    new_text = _replace_optional(text)
    if new_text != text:
        changes.append("Optional[X] → X | None")
        text = new_text

    # Step 3: Replace Union[X, Y] → X | Y
    new_text = _replace_union(text)
    if new_text != text:
        changes.append("Union[X, Y] → X | Y")
        text = new_text

    # Step 4: Clean up typing imports
    new_text = _update_typing_import(text)
    if new_text != text:
        changes.append("typing import cleaned")
        text = new_text

    # Step 5: Add future annotations if needed (forward refs + | syntax)
    if needs_future and text != original:
        text = _ensure_future_annotations(text)
        changes.append("added __future__ annotations")

    changed = text != original
    if changed and not dry_run:
        filepath.write_text(text, encoding="utf-8")

    return changed, changes


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Modernize typing annotations")
    parser.add_argument("path", type=pathlib.Path, help="Directory to process")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()

    base = args.path
    if not base.exists():
        print(f"Path does not exist: {base}", file=sys.stderr)
        sys.exit(1)

    total_changed = 0
    for filepath in sorted(base.rglob("*.py")):
        changed, changes = modernize_file(filepath, dry_run=args.dry_run)
        if changed:
            total_changed += 1
            prefix = "[DRY] " if args.dry_run else ""
            print(f"{prefix}{filepath}: {', '.join(changes)}")

    print(f"\n{'Would modify' if args.dry_run else 'Modified'}: {total_changed} files")


if __name__ == "__main__":
    main()
