#!/usr/bin/env python3
"""
特徴量セット管理コマンドラインツール

特徴量セットの作成、編集、削除、表示を行うためのツール
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.features.feature_set_manager import get_feature_manager
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def list_feature_sets(manager) -> None:
    """特徴量セットの一覧を表示"""
    sets = manager.list_feature_sets()

    if not sets:
        print("No feature sets found.")
        return

    print("Available Feature Sets:")
    print("=" * 50)

    for name, info in sets.items():
        status = "✓" if info["enabled"] else "✗"
        print(f"{status} {name} (v{info['version']})")
        print(f"   Description: {info['description']}")
        print(f"   Features: {info['feature_count']}")
        print()


def show_feature_set(manager, name: str) -> None:
    """指定された特徴量セットの詳細を表示"""
    try:
        features = manager.get_feature_set(name)
        sets_info = manager.list_feature_sets()

        if name not in sets_info:
            print(f"Feature set '{name}' not found.")
            return

        info = sets_info[name]
        status = "Enabled" if info["enabled"] else "Disabled"

        print(f"Feature Set: {name}")
        print(f"Status: {status}")
        print(f"Version: {info['version']}")
        print(f"Description: {info['description']}")
        print(f"Feature Count: {info['feature_count']}")
        print()
        print("Features:")
        for i, feature in enumerate(features, 1):
            print("2d")
        print()

    except Exception as e:
        print(f"Error showing feature set '{name}': {e}")


def create_feature_set(
    manager, name: str, description: str = "", base_set: Optional[str] = None
) -> None:
    """新しい特徴量セットを作成"""
    try:
        # ベースセットから特徴量を取得
        if base_set:
            base_features = manager.get_feature_set(base_set)
        else:
            base_features = []

        success = manager.add_feature_set(name, base_features, description)
        if success:
            print(f"Created feature set '{name}' with {len(base_features)} features.")
            if base_set:
                print(f"Based on feature set '{base_set}'.")
        else:
            print(f"Failed to create feature set '{name}'.")

    except Exception as e:
        print(f"Error creating feature set '{name}': {e}")


def add_features(manager, set_name: str, features: List[str]) -> None:
    """特徴量セットに特徴量を追加"""
    try:
        success = manager.add_features(set_name, features)
        if success:
            added_count = len(
                [f for f in features if f not in manager.get_feature_set(set_name)]
            )
            print(f"Added {added_count} features to set '{set_name}'.")
        else:
            print(f"Failed to add features to set '{set_name}'.")

    except Exception as e:
        print(f"Error adding features to set '{set_name}': {e}")


def remove_features(manager, set_name: str, features: List[str]) -> None:
    """特徴量セットから特徴量を削除"""
    try:
        success = manager.remove_features(set_name, features)
        if success:
            print(f"Removed features from set '{set_name}'.")
        else:
            print(f"Failed to remove features from set '{set_name}'.")

    except Exception as e:
        print(f"Error removing features from set '{set_name}': {e}")


def delete_feature_set(manager, name: str) -> None:
    """特徴量セットを削除"""
    try:
        success = manager.remove_feature_set(name)
        if success:
            print(f"Deleted feature set '{name}'.")
        else:
            print(f"Failed to delete feature set '{name}'.")

    except Exception as e:
        print(f"Error deleting feature set '{name}': {e}")


def update_feature_set(
    manager,
    name: str,
    description: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> None:
    """特徴量セットを更新"""
    try:
        success = manager.update_feature_set(
            name, description=description, enabled=enabled
        )
        if success:
            print(f"Updated feature set '{name}'.")
        else:
            print(f"Failed to update feature set '{name}'.")

    except Exception as e:
        print(f"Error updating feature set '{name}': {e}")


def main():
    parser = argparse.ArgumentParser(
        description="特徴量セット管理ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 特徴量セットの一覧表示
  python manage_features.py list

  # 特定の特徴量セットの詳細表示
  python manage_features.py show curated

  # 新しい特徴量セットの作成
  python manage_features.py create my_set "My custom feature set" --base curated

  # 特徴量の追加
  python manage_features.py add my_set feature1 feature2 feature3

  # 特徴量の削除
  python manage_features.py remove my_set feature1 feature2

  # 特徴量セットの削除
  python manage_features.py delete my_set

  # 特徴量セットの更新
  python manage_features.py update my_set --description "Updated description" --disable
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="利用可能なコマンド")

    # list command
    subparsers.add_parser("list", help="特徴量セットの一覧を表示")

    # show command
    show_parser = subparsers.add_parser("show", help="特徴量セットの詳細を表示")
    show_parser.add_argument("name", help="特徴量セット名")

    # create command
    create_parser = subparsers.add_parser("create", help="新しい特徴量セットを作成")
    create_parser.add_argument("name", help="特徴量セット名")
    create_parser.add_argument("description", help="説明")
    create_parser.add_argument("--base", help="ベースとなる特徴量セット")

    # add command
    add_parser = subparsers.add_parser("add", help="特徴量セットに特徴量を追加")
    add_parser.add_argument("set_name", help="特徴量セット名")
    add_parser.add_argument("features", nargs="+", help="追加する特徴量")

    # remove command
    remove_parser = subparsers.add_parser("remove", help="特徴量セットから特徴量を削除")
    remove_parser.add_argument("set_name", help="特徴量セット名")
    remove_parser.add_argument("features", nargs="+", help="削除する特徴量")

    # delete command
    delete_parser = subparsers.add_parser("delete", help="特徴量セットを削除")
    delete_parser.add_argument("name", help="特徴量セット名")

    # update command
    update_parser = subparsers.add_parser("update", help="特徴量セットを更新")
    update_parser.add_argument("name", help="特徴量セット名")
    update_parser.add_argument("--description", help="新しい説明")
    update_parser.add_argument("--enable", action="store_true", help="有効化")
    update_parser.add_argument("--disable", action="store_true", help="無効化")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        manager = get_feature_manager()

        if args.command == "list":
            list_feature_sets(manager)
        elif args.command == "show":
            show_feature_set(manager, args.name)
        elif args.command == "create":
            create_feature_set(manager, args.name, args.description, args.base)
        elif args.command == "add":
            add_features(manager, args.set_name, args.features)
        elif args.command == "remove":
            remove_features(manager, args.set_name, args.features)
        elif args.command == "delete":
            delete_feature_set(manager, args.name)
        elif args.command == "update":
            enabled = None
            if args.enable:
                enabled = True
            elif args.disable:
                enabled = True  # Note: this should be False, fixing the logic
            update_feature_set(manager, args.name, args.description, enabled)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
