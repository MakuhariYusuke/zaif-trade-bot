#!/usr/bin/env python3
"""Fix broken JSON training reports with extra closing braces."""
import json
import re
from pathlib import Path

from ztb.reporting.services.catalog import list_training_reports


def _build_backup_path(file_path: Path) -> Path:
    candidate = file_path.with_suffix(".json.bak")
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = file_path.with_suffix(f".json.bak{index}")
        if not candidate.exists():
            return candidate
        index += 1


def fix_json_file(file_path: Path) -> bool:
    """Fix a single JSON file by removing extra closing braces."""
    try:
        content = file_path.read_text(encoding="utf-8")
        
        # Try to parse first - if successful, no fix needed
        try:
            json.loads(content)
            return False  # Already valid
        except json.JSONDecodeError as e:
            # Check if it's the "Extra data" error
            if "Extra data" not in str(e):
                print(f"Different JSON error in {file_path.name}: {e}")
                return False
        
        # Remove extra closing braces at the end
        # Pattern: }}\n at the end when there should be only }\n
        fixed_content = re.sub(r"\}\}\s*$", "}", content)
        if fixed_content == content:
            print(f"✗ Could not fix {file_path.name}: no trailing extra brace pattern")
            return False
        
        # Verify the fix
        try:
            json.loads(fixed_content)
            # Backup original
            backup_path = _build_backup_path(file_path)
            file_path.rename(backup_path)
            # Write fixed version
            file_path.write_text(fixed_content, encoding="utf-8")
            print(f"✓ Fixed: {file_path.name}")
            return True
        except json.JSONDecodeError as e:
            print(f"✗ Could not fix {file_path.name}: {e}")
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    reports_dir = Path("reports")
    
    if not reports_dir.exists():
        print(f"Reports directory not found: {reports_dir}")
        return
    
    # Find all training report JSON files
    json_files = list_training_reports(reports_dir=reports_dir)
    
    print(f"Found {len(json_files)} training report files")
    print("Checking for broken JSON files...\n")
    
    fixed_count = 0
    for json_file in json_files:
        if fix_json_file(json_file):
            fixed_count += 1
    
    print(f"\n{'='*60}")
    print(f"Fixed {fixed_count} files")
    print("Backup files created with .json.bak extension")


if __name__ == "__main__":
    main()
