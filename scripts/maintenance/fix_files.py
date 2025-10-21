#!/usr/bin/env python3
"""
Fix all Phase 5 files by removing invalid content at the end
"""

import os

def fix_file(filepath):
    """Fix a single file by removing invalid content at the end"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    if '</content>' in content or '<parameter name' in content:
        print(f"Fixing {filepath}")
        # Remove invalid content at the end
        lines = content.split('\n')
        valid_lines = []
        for line in lines:
            if not line.startswith('<'):
                valid_lines.append(line)
            else:
                break
        # Write back only valid lines
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(valid_lines))
        print(f"Fixed {filepath}")
        return True
    return False

def main():
    """Main function"""
    # Get the project root directory (parent of scripts directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    production_dir = os.path.join(project_root, 'ztb', 'trading', 'production')

    if not os.path.exists(production_dir):
        print(f"Production directory not found: {production_dir}")
        return

    files = [f for f in os.listdir(production_dir) if f.endswith('.py')]

    fixed_count = 0
    for file in files:
        filepath = os.path.join(production_dir, file)
        if fix_file(filepath):
            fixed_count += 1

    print(f"Fixed {fixed_count} files")

if __name__ == '__main__':
    main()