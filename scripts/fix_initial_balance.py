#!/usr/bin/env python3
import json
import os
import glob

def fix_initial_balance_in_config_files():
    """Fix initial_balance from int to float in all config JSON files"""
    config_files = glob.glob('config/*.json') + glob.glob('configs/*.json')

    for config_file in config_files:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Check if initial_balance exists and is int
            if 'environment' in config and 'initial_balance' in config['environment']:
                balance = config['environment']['initial_balance']
                if isinstance(balance, int):
                    print(f"Fixing {config_file}: {balance} -> {float(balance)}")
                    config['environment']['initial_balance'] = float(balance)

                    # Write back the fixed config
                    with open(config_file, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2, ensure_ascii=False)
                else:
                    print(f"Already float in {config_file}: {balance}")

        except Exception as e:
            print(f"Error processing {config_file}: {e}")

if __name__ == "__main__":
    fix_initial_balance_in_config_files()