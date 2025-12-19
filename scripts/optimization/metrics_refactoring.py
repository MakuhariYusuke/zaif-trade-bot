#!/usr/bin/env python3
"""
リファクタリングスクリプト: 重複メトリクス計算の共通化
プロジェクト全体で重複しているメトリクス計算関数をztb/metrics/metrics.pyの共通関数に置き換える
"""

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# 共通メトリクス関数とそのマッピング
METRICS_MAPPING = {
    'calculate_sharpe_ratio': 'sharpe_ratio',
    'calculate_sortino_ratio': 'sortino_ratio',
    'calculate_max_drawdown': 'max_drawdown',
    'calculate_volatility': 'volatility',
    'calculate_win_rate': 'win_rate',
    'calculate_total_return': 'total_return'
}

# 各メトリクス関数が必要とするインポート
REQUIRED_IMPORTS = {
    'sharpe_ratio': 'from ztb.metrics.metrics import sharpe_ratio',
    'sortino_ratio': 'from ztb.metrics.metrics import sortino_ratio',
    'max_drawdown': 'from ztb.metrics.metrics import max_drawdown',
    'volatility': 'from ztb.metrics.metrics import volatility',
    'win_rate': 'from ztb.metrics.metrics import win_rate',
    'total_return': 'from ztb.metrics.metrics import total_return'
}

class MetricsRefactorer:
    """メトリクス計算のリファクタリングクラス"""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.modified_files: Set[str] = set()

    def find_duplicate_functions(self) -> Dict[str, List[str]]:
        """重複メトリクス関数を含むファイルを検索"""
        duplicates = {func: [] for func in METRICS_MAPPING.keys()}

        # git grepで重複関数を検索
        for func_name in METRICS_MAPPING.keys():
            try:
                result = subprocess.run(
                    ['git', 'grep', '-l', f'def {func_name}', '--', 'ztb/**/*.py'],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
                files = set()
                if result.stdout.strip():
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            # 相対パスを絶対パスに変換
                            abs_path = str((self.repo_path / line.strip()).resolve())
                            files.add(abs_path)
                duplicates[func_name] = list(files)
            except subprocess.CalledProcessError:
                # 関数が見つからない場合は空リスト
                duplicates[func_name] = []

        return duplicates

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """ファイルを分析して重複関数と使用状況を特定"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        analysis = {
            'duplicate_functions': [],
            'function_calls': [],
            'imports': [],
            'has_common_import': False
        }

        # 重複関数の定義を検出
        for func_name in METRICS_MAPPING.keys():
            if re.search(rf'def {func_name}\s*\(', content):
                analysis['duplicate_functions'].append(func_name)

        # 関数呼び出しを検出
        for func_name in METRICS_MAPPING.keys():
            calls = re.findall(rf'\b{func_name}\s*\(', content)
            if calls:
                analysis['function_calls'].extend([func_name] * len(calls))

        # インポートを検出
        import_lines = []
        for line in content.split('\n'):
            if line.strip().startswith('from ztb.metrics.metrics import'):
                import_lines.append(line.strip())
                analysis['has_common_import'] = True

        analysis['imports'] = import_lines

        return analysis

    def refactor_file(self, file_path: str) -> bool:
        """ファイルをリファクタリング"""
        print(f"リファクタリング中: {file_path}")

        analysis = self.analyze_file(file_path)

        if not analysis['duplicate_functions'] and not analysis['function_calls']:
            print(f"  変更不要: {file_path}")
            return False

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        modifications = []

        # 1. 重複関数の削除
        for func_name in analysis['duplicate_functions']:
            # 関数定義全体を削除
            pattern = rf'def {func_name}\s*\([^)]*\):\s*"""[\s\S]*?"""\s*(?:[^\n]*\n)*?(?=def|\nclass|\n@|\n\w|\Z)'
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                content = content.replace(match.group(0), '')
                modifications.append(f"削除: {func_name}関数")
                print(f"  削除: {func_name}関数")

        # 2. 関数呼び出しの置き換え
        for old_func in METRICS_MAPPING.keys():
            new_func = METRICS_MAPPING[old_func]
            if old_func in analysis['function_calls']:
                # 関数呼び出しを置き換え
                content = re.sub(rf'\b{old_func}\s*\(', f'{new_func}(', content)
                modifications.append(f"置き換え: {old_func} -> {new_func}")
                print(f"  置き換え: {old_func} -> {new_func}")

        # 3. インポートの更新
        required_imports = set()
        for func_name in analysis['function_calls']:
            if func_name in METRICS_MAPPING:
                required_imports.add(METRICS_MAPPING[func_name])

        if required_imports:
            # 既存のmetricsインポートを更新
            existing_import_line = None
            for line in content.split('\n'):
                if 'from ztb.metrics.metrics import' in line:
                    existing_import_line = line
                    break

            if existing_import_line:
                # 既存のインポートを拡張
                current_imports = re.search(r'from ztb\.metrics\.metrics import (.+)', existing_import_line)
                if current_imports:
                    current_funcs = set(re.split(r',\s*', current_imports.group(1)))
                    current_funcs.update(required_imports)
                    new_import_line = f"from ztb.metrics.metrics import {', '.join(sorted(current_funcs))}"
                    content = content.replace(existing_import_line, new_import_line)
                    modifications.append("インポート更新")
            else:
                # 新しいインポートを追加
                lines = content.split('\n')
                # 適切な位置にインポートを追加（他のインポート文の後）
                insert_pos = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        insert_pos = i + 1
                    elif line.strip() and not line.strip().startswith('#'):
                        break

                new_import = f"from ztb.metrics.metrics import {', '.join(sorted(required_imports))}"
                lines.insert(insert_pos, new_import)
                content = '\n'.join(lines)
                modifications.append("インポート追加")

        # 変更がある場合のみファイルを更新
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.modified_files.add(str(file_path))
            print(f"  完了: {len(modifications)}件の変更")
            return True

        return False

    def run_refactoring(self, target_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """リファクタリングを実行"""
        print("=== メトリクス計算リファクタリング開始 ===")

        # 重複関数を検索
        duplicates = self.find_duplicate_functions()

        total_duplicates = sum(len(files) for files in duplicates.values())
        print(f"重複関数が見つかったファイル数: {total_duplicates}")

        # 対象ファイルを決定
        if target_files:
            files_to_process = [f for f in target_files if Path(f).exists()]
        else:
            files_to_process = set()
            for func_files in duplicates.values():
                files_to_process.update(func_files)
            files_to_process = list(files_to_process)

        print(f"処理対象ファイル数: {len(files_to_process)}")

        # 各ファイルをリファクタリング
        successful_refactors = 0
        failed_files = []

        for file_path in files_to_process:
            try:
                if self.refactor_file(file_path):
                    successful_refactors += 1
            except Exception as e:
                print(f"  エラー: {file_path} - {e}")
                failed_files.append((file_path, str(e)))

        # 結果サマリー
        result = {
            'total_files_processed': len(files_to_process),
            'successful_refactors': successful_refactors,
            'failed_files': failed_files,
            'modified_files': list(self.modified_files)
        }

        print("\n=== リファクタリング完了 ===")
        print(f"処理ファイル数: {result['total_files_processed']}")
        print(f"成功: {result['successful_refactors']}")
        print(f"失敗: {len(result['failed_files'])}")
        print(f"変更ファイル数: {len(result['modified_files'])}")

        if result['failed_files']:
            print("\n失敗したファイル:")
            for file_path, error in result['failed_files']:
                print(f"  {file_path}: {error}")

        return result

    def validate_refactoring(self) -> Dict[str, Any]:
        """リファクタリングの検証"""
        print("\n=== リファクタリング検証 ===")

        validation_results = {
            'syntax_errors': [],
            'import_errors': [],
            'remaining_duplicates': []
        }

        # 変更されたファイルを検証
        for file_path in self.modified_files:
            try:
                # 構文チェック
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                compile(content, file_path, 'exec')
                print(f"  ✓ 構文OK: {file_path}")

            except SyntaxError as e:
                validation_results['syntax_errors'].append((file_path, str(e)))
                print(f"  ✗ 構文エラー: {file_path} - {e}")
            except Exception as e:
                validation_results['import_errors'].append((file_path, str(e)))
                print(f"  ✗ インポートエラー: {file_path} - {e}")

        # 残っている重複関数をチェック
        remaining_duplicates = self.find_duplicate_functions()
        total_remaining = sum(len(files) for files in remaining_duplicates.values())

        if total_remaining > 0:
            print(f"\n残っている重複関数: {total_remaining}件")
            for func, files in remaining_duplicates.items():
                if files:
                    validation_results['remaining_duplicates'].extend([(func, f) for f in files])
                    print(f"  {func}: {len(files)}ファイル")
        else:
            print("\n✓ 重複関数はすべて削除されました")

        return validation_results


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description='メトリクス計算のリファクタリング')
    parser.add_argument('--target-files', nargs='*', help='対象ファイル（指定しない場合は自動検出）')
    parser.add_argument('--dry-run', action='store_true', help='ドライラン（実際の変更を行わない）')
    parser.add_argument('--validate-only', action='store_true', help='検証のみ実行')

    args = parser.parse_args()

    # リポジトリのルートディレクトリを取得
    repo_path = str(Path(__file__).parent.parent)
    refactorer = MetricsRefactorer(repo_path)

    if args.validate_only:
        # 検証のみ
        results = refactorer.validate_refactoring()
        return

    if args.dry_run:
        # ドライラン
        duplicates = refactorer.find_duplicate_functions()
        print("=== ドライラン結果 ===")
        for func, files in duplicates.items():
            print(f"{func}: {len(files)}ファイル")
            for f in files[:5]:  # 最初の5つだけ表示
                print(f"  {f}")
            if len(files) > 5:
                print(f"  ... 他{len(files) - 5}ファイル")
        return

    # リファクタリング実行
    results = refactorer.run_refactoring(args.target_files)

    # 検証実行
    validation = refactorer.validate_refactoring()

    # 最終レポート
    print("\n=== 最終レポート ===")
    if not validation['syntax_errors'] and not validation['remaining_duplicates']:
        print("✓ リファクタリング成功")
    else:
        print("⚠ リファクタリングで問題が発生")

    if validation['syntax_errors']:
        print(f"構文エラー: {len(validation['syntax_errors'])}件")

    if validation['remaining_duplicates']:
        print(f"残存重複: {len(validation['remaining_duplicates'])}件")


if __name__ == '__main__':
    main()