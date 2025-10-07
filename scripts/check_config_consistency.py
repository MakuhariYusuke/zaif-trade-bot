#!/usr/bin/env python3
"""
設定ファイルの一貫性チェックスクリプト
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def check_config_consistency(config_dir: Path) -> Dict[str, Any]:
    """設定ファイルのキー名の一貫性をチェック."""

    all_keys: Dict[str, List[str]] = {}
    all_values: Dict[str, Dict[str, str]] = defaultdict(dict)

    # 設定ファイルを読み込み
    for config_file in config_dir.glob("*.json"):
        try:
            with open(config_file, encoding='utf-8') as f:
                data = json.load(f)

            all_keys[config_file.name] = list(data.keys())

            # 各キーの型を記録
            for key, value in data.items():
                all_values[key][config_file.name] = str(type(value))

        except Exception as e:
            print(f"Error reading {config_file.name}: {e}")
            continue

    if all_keys:
        # 共通キーを計算
        key_sets = [set(keys) for keys in all_keys.values()]
        common_keys = list(set.intersection(*key_sets)) if key_sets else []

        # ユニークキーを計算
        unique_keys = {}
        for name, keys in all_keys.items():
            unique = set(keys) - set(common_keys)
            unique_keys[name] = list(unique)
    else:
        common_keys = []
        unique_keys = {}

    # 型の一貫性チェック
    type_inconsistencies = {}
    for key, type_dict in all_values.items():
        types = set(type_dict.values())
        if len(types) > 1:
            type_inconsistencies[key] = type_dict

    return {
        "common_keys": common_keys,
        "unique_keys": unique_keys,
        "type_inconsistencies": type_inconsistencies,
        "all_files": list(all_keys.keys())
    }


def main():
    """メイン実行関数."""

    config_dir = Path("configs/train")

    print("🔍 設定ファイル一貫性チェック")
    print("=" * 50)

    if not config_dir.exists():
        print(f"❌ 設定ディレクトリが見つかりません: {config_dir}")
        return

    # 設定ファイルの一貫性チェック
    consistency = check_config_consistency(config_dir)

    print(f"\n📁 対象ファイル数: {len(consistency['all_files'])}")
    print(f"📄 共通キー数: {len(consistency['common_keys'])}")

    if any(consistency['unique_keys'].values()):
        print("\n⚠️  ユニークキー（不整合の可能性）:")
        for file_name, keys in consistency['unique_keys'].items():
            if keys:
                print(f"  {file_name}: {keys}")

    # レポート生成
    report = {
        "consistency_check": consistency,
        "summary": {
            "total_config_files": len(consistency['all_files']),
            "common_keys_count": len(consistency['common_keys']),
            "unique_keys_files": len([f for f, k in consistency['unique_keys'].items() if k]),
            "type_inconsistencies_count": len(consistency['type_inconsistencies'])
        }
    }

    with open("config_consistency_report.json", "w", encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n📊 レポート保存: config_consistency_report.json")
    print(f"📈 要改善項目: {len([f for f, k in consistency['unique_keys'].items() if k]) + len(consistency['type_inconsistencies'])}件")


if __name__ == "__main__":
    main()
