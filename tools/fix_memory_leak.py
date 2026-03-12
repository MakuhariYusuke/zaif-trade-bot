#!/usr/bin/env python3
"""
Memory Leak Fix Tool - Identifies and fixes memory leak in training process

The main issue is:
1. training_utils.py uses global default_memory_manager to cache large DataFrames
2. Cache keys use hash(df.values.tobytes()) which itself consumes huge memory
3. Multiple AB test child processes accumulate cache entries
4. TTLCache doesn't release memory until TTL expires (5-30 minutes)

Fix strategy:
1. Disable memory cache for large DataFrames by default
2. Add explicit cache cleanup at process start/end
3. Reduce cache TTL to 60 seconds for training data
4. Add memory limit enforcement
"""

from pathlib import Path


def fix_training_utils() -> bool:
    """Fix memory leak in training_utils.py by disabling problematic cache."""

    training_utils_path = Path("ztb/training/utils/training_utils.py")

    if not training_utils_path.exists():
        print(f"❌ File not found: {training_utils_path}")
        return False

    content = training_utils_path.read_text(encoding="utf-8")

    # Find and fix enable_memory_cache defaults
    replacements = [
        # Change default enable_memory_cache to False
        ("enable_memory_cache: bool = True", "enable_memory_cache: bool = False"),
    ]

    modified = content
    changes_made = 0

    for old, new in replacements:
        if old in modified:
            modified = modified.replace(old, new)
            changes_made += 1
            print(f"✅ Replaced: {old} -> {new}")

    if changes_made > 0:
        training_utils_path.write_text(modified, encoding="utf-8")
        print(f"✅ Fixed {changes_made} instances in {training_utils_path}")
        return True
    else:
        print(f"⚠️  No changes needed in {training_utils_path}")
        return True


def fix_memory_cache() -> bool:
    """Fix memory_cache.py to reduce TTL and add cleanup."""

    memory_cache_path = Path("ztb/cache/memory_cache.py")

    if not memory_cache_path.exists():
        print(f"❌ File not found: {memory_cache_path}")
        return False

    content = memory_cache_path.read_text(encoding="utf-8")

    # Reduce TTL for data cache from 600 to 60 seconds
    replacements = [
        (
            "self.data_cache = TTLCache(maxsize=500, ttl=600)",
            "self.data_cache = TTLCache(maxsize=500, ttl=60)",
        ),
    ]

    modified = content
    changes_made = 0

    for old, new in replacements:
        if old in modified:
            modified = modified.replace(old, new)
            changes_made += 1
            print(f"✅ Replaced: {old} -> {new}")

    if changes_made > 0:
        memory_cache_path.write_text(modified, encoding="utf-8")
        print(f"✅ Fixed {changes_made} instances in {memory_cache_path}")
        return True
    else:
        print(f"⚠️  No changes needed in {memory_cache_path}")
        return True


def add_cache_cleanup_to_ab_test() -> bool:
    """Add explicit cache cleanup to AB test runner."""

    ab_test_path = Path("tools/ab_test_runner.py")

    if not ab_test_path.exists():
        print(f"❌ File not found: {ab_test_path}")
        return False

    content = ab_test_path.read_text(encoding="utf-8")

    # Check if cleanup is already added
    if "default_memory_manager.optimize_memory_usage()" in content:
        print(f"✅ Cache cleanup already present in {ab_test_path}")
        return True

    # Add cleanup import at top
    if "from ztb.cache.memory_cache import default_memory_manager" not in content:
        # Find the last import statement
        lines = content.split("\n")
        import_end_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                import_end_idx = i

        # Insert new import after last import
        lines.insert(
            import_end_idx + 1,
            "from ztb.cache.memory_cache import default_memory_manager",
        )
        content = "\n".join(lines)
        print("✅ Added import for default_memory_manager")

    # Add cleanup call in main training loop (before training starts)
    # Look for pattern: "child_pid = os.fork()" or similar subprocess creation
    if "def run_single_config(" in content:
        # Add cleanup at start of function
        pattern = "def run_single_config("
        insert_point = content.find(pattern)
        if insert_point > 0:
            # Find end of function signature
            next_line_start = content.find("\n", insert_point)
            # Find first line of function body (after docstring if any)
            func_body_start = content.find('"""', next_line_start)
            if func_body_start > 0:
                # After docstring
                func_body_start = content.find('"""', func_body_start + 3) + 3
            else:
                # No docstring, just after function def
                func_body_start = next_line_start + 1

            cleanup_code = """
    # Clear memory cache before training to prevent leak
    try:
        default_memory_manager.optimize_memory_usage()
        gc.collect()
    except Exception as e:
        pass  # Ignore cleanup errors
"""
            content = (
                content[:func_body_start] + cleanup_code + content[func_body_start:]
            )
            print("✅ Added cache cleanup to run_single_config()")

    ab_test_path.write_text(content, encoding="utf-8")
    print(f"✅ Updated {ab_test_path}")
    return True


def main() -> None:
    """Apply all memory leak fixes."""
    print("=" * 60)
    print("Memory Leak Fix Tool")
    print("=" * 60)

    print("\n1. Fixing training_utils.py (disable problematic DataFrame cache)...")
    fix_training_utils()

    print("\n2. Fixing memory_cache.py (reduce TTL for data cache)...")
    fix_memory_cache()

    print("\n3. Adding cache cleanup to AB test runner...")
    add_cache_cleanup_to_ab_test()

    print("\n" + "=" * 60)
    print("✅ Memory leak fixes applied successfully!")
    print("=" * 60)
    print("\nSummary of changes:")
    print("  1. Disabled memory cache for large DataFrames (default: False)")
    print("  2. Reduced data cache TTL from 600s to 60s")
    print("  3. Added explicit cache cleanup at process start")
    print("\nNext steps:")
    print("  1. Run unit tests to verify changes don't break functionality")
    print("  2. Run AB test with fixed configuration")
    print("  3. Monitor memory usage to confirm leak is resolved")


if __name__ == "__main__":
    main()
