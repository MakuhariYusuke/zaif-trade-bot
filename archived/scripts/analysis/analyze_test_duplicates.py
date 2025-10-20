import os
import re
from collections import defaultdict


def find_test_duplicates() -> None:
    """Find duplicate patterns in test files"""
    test_files = []
    for root, dirs, files in os.walk("ztb"):
        for file in files:
            if file.startswith("test_") or "__tests__" in root:
                test_files.append(os.path.join(root, file))

    # Common test patterns
    patterns = [
        r"@patch\([^)]+\)",
        r"def test_.*\(.*\):",
        r"assert.*==.*",
        r"self\.assertEqual\(.*\)",
        r"env = HeavyTradingEnv\(.*\)",
        r"config = .*Config\(.*\)",
        r"def setUp\(self\):",
        r"import unittest",
        r"from unittest.mock import",
    ]

    duplicates: dict[str, list[str]] = defaultdict(list)

    for file_path in test_files[:20]:  # Check first 20 test files
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            for pattern in patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    duplicates[match].append(file_path)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Filter to only duplicates
    duplicates = {k: list(set(v)) for k, v in duplicates.items() if len(v) > 1}

    print(f"Found {len(duplicates)} duplicate test patterns")

    # Show top duplicates
    sorted_dup = sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (pattern, files) in enumerate(sorted_dup[:10]):
        print(f"\nPattern {i+1} (used in {len(files)} files): {pattern}")
        for file in files[:3]:  # Show first 3 files
            print(f"  {file}")


def find_config_duplicates() -> None:
    """Find duplicate configuration patterns"""
    config_patterns = [
        r"learning_rate.*=.*",
        r"batch_size.*=.*",
        r"n_epochs.*=.*",
        r"gamma.*=.*",
        r"gae_lambda.*=.*",
        r"ent_coef.*=.*",
        r"vf_coef.*=.*",
        r"max_grad_norm.*=.*",
        r"use_sde.*=.*",
        r"sde_sample_freq.*=.*",
        r"rollout_buffer_class.*=.*",
        r"policy_kwargs.*=.*",
        r"tensorboard_log.*=.*",
        r"verbose.*=.*",
        r"seed.*=.*",
        r"device.*=.*",
    ]

    py_files = []
    for root, dirs, files in os.walk("ztb"):
        for file in files:
            if file.endswith(".py") and "training" in root:
                py_files.append(os.path.join(root, file))

    duplicates: dict[str, list[str]] = defaultdict(list)

    for file_path in py_files[:20]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            for pattern in config_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    duplicates[match].append(file_path)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    duplicates = {k: list(set(v)) for k, v in duplicates.items() if len(v) > 1}

    print(f"\nFound {len(duplicates)} duplicate config patterns")

    sorted_dup = sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (pattern, files) in enumerate(sorted_dup[:5]):
        print(f"\nConfig {i+1} (used in {len(files)} files): {pattern}")
        for file in files[:3]:
            print(f"  {file}")


if __name__ == "__main__":
    find_test_duplicates()
    find_config_duplicates()
