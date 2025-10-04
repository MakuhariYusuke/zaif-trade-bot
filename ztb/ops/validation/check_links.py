#!/usr/bin/env python3
"""
Document link validator for README.md, docs/, and ztb/ directories.

Checks relative links and anchors in Markdown files.
"""

import argparse
import os
import re
from pathlib import Path
from typing import Optional


def find_markdown_files(
    roots: list[Path], exclude_patterns: Optional[list[str]] = None
) -> list[Path]:
    """Find all Markdown files under the given roots, excluding patterns."""
    if exclude_patterns is None:
        exclude_patterns = []
    exclude_patterns = [os.path.normpath(p) for p in exclude_patterns]
    files = []
    for root in roots:
        path = Path(root)
        if path.is_file() and path.suffix == ".md":
            if not any(
                os.path.normpath(str(path)).startswith(pattern)
                for pattern in exclude_patterns
            ):
                files.append(path)
        elif path.is_dir():
            for md in path.rglob("*.md"):
                if not any(
                    os.path.normpath(str(md)).startswith(pattern)
                    for pattern in exclude_patterns
                ):
                    files.append(md)
    return files


def extract_links(content: str) -> list[tuple[str, str]]:
    """Extract [text](url) links from content."""
    pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    return re.findall(pattern, content)


def extract_headers(content: str) -> list[tuple[int, str]]:
    """Extract headers and their anchors from content."""
    headers = []
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            # Count # for level
            level = 0
            for char in line:
                if char == "#":
                    level += 1
                else:
                    break
            title = line[level:].strip()
            # Generate anchor: lowercase, spaces to -, remove non-word chars
            anchor = re.sub(r"[^\w\-]", "", title.lower().replace(" ", "-"))
            headers.append((level, anchor))
    return headers


def check_link(file_path: Path, link: str, headers: list[tuple[int, str]]) -> bool:
    """Check if a link is valid."""
    if link.startswith(("http://", "https://", "mailto:")):
        # External links, skip
        return True

    if link.startswith("#"):
        # Anchor in same file
        anchor = link[1:]
        return any(h[1] == anchor for h in headers)

    # Relative path
    try:
        resolved = (file_path.parent / link).resolve()
        return resolved.exists()
    except (OSError, RuntimeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check links in Markdown files")
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="Root directories/files to check (e.g., README.md docs ztb)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Patterns to exclude (e.g., ztb/reports/)",
    )
    args = parser.parse_args()

    files = find_markdown_files(args.roots, args.exclude)
    broken = []

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            print(f"Warning: Could not read {file_path} (encoding issue)")
            continue

        links = extract_links(content)
        headers = extract_headers(content)

        for _, link in links:
            if not check_link(file_path, link, headers):
                broken.append((str(file_path), link))

    if broken:
        print("Broken links found:")
        for file, link in broken:
            print(f"  {file}: {link}")
        return 1
    else:
        print("All links are valid.")
        return 0


if __name__ == "__main__":
    exit(main())
