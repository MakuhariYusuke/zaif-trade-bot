#!/usr/bin/env python3
"""Remove exact duplicate code blocks based on the duplicate report.

Keeps one occurrence per group and removes the rest.
"""
import json
import os
from pathlib import Path
from typing import Dict, List

def load_report(report_path: Path) -> Dict:
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def remove_duplicates(report: Dict, root: Path):
    """Remove duplicate occurrences, keeping one per group."""
    exact_groups = report['exact_groups']
    
    for group_hash, occurrences in exact_groups.items():
        if len(occurrences) <= 1:
            continue
        
        # Keep the first occurrence, remove the rest
        keep = occurrences[0]
        to_remove = occurrences[1:]
        
        print(f"Processing group {group_hash}: keeping {keep['path']}:{keep['start']}-{keep['end']}, removing {len(to_remove)} duplicates")
        
        for occ in to_remove:
            file_path = root / occ['path']
            if not file_path.exists():
                print(f"Warning: {file_path} does not exist, skipping")
                continue
            
            # Read the file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
            
            # Find the duplicate block
            start_line = occ['start'] - 1  # 0-based
            end_line = occ['end'] - 1      # 0-based
            
            if start_line < 0 or end_line >= len(lines) or start_line > end_line:
                print(f"Invalid line range for {file_path}: {start_line+1}-{end_line+1}")
                continue
            
            # Remove the block
            del lines[start_line:end_line + 1]
            
            # Write back
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                print(f"Removed duplicate from {file_path}")
            except Exception as e:
                print(f"Error writing {file_path}: {e}")

def main():
    root = Path('.')
    report_path = root / 'reports' / 'duplicate_report.json'
    
    if not report_path.exists():
        print(f"Report not found: {report_path}")
        return
    
    report = load_report(report_path)
    remove_duplicates(report, root)
    print("Duplicate removal complete")

if __name__ == '__main__':
    main()