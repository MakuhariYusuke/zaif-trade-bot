# Codex Task: 701# T1 — ワークスペースクリーンアップスクリプト

## ゴール

IDE パフォーマンスを劣化させている不要ファイルの検出・削除を行う `scripts/v460/tools/cleanup_workspace.py` を新規作成する。
また実行時の安全措置として dry-run モードをデフォルトにする。

## 背景

701# IDE 監査で以下の不要ファイル蓄積が発見された:
- `config/ab_search_temp_*.json`: 5,402 ファイル (古い A/B 検索一時ファイル, .gitignore 対象)
- `data/temp/.mypy_cache/`: 12,740 ファイル (temp 環境での mypy キャッシュ残骸)
- `data/temp/.ruff_cache/`: 183 ファイル
- `data/temp/.hypothesis/`: 70 ファイル
- `data/temp/tmp-*`: 各種テンポラリディレクトリ

## 実装仕様

### scripts/v460/tools/cleanup_workspace.py

```python
# CLI: python -m scripts.v460.tools.cleanup_workspace [--execute] [--verbose]
# デフォルト: dry-run (削除は --execute フラグが必要)
```

**クリーンアップ対象** (ハードコードされたルール):

1. `config/ab_search_temp_*.json` — glob マッチで全削除
2. `data/temp/.mypy_cache/` — shutil.rmtree
3. `data/temp/.ruff_cache/` — shutil.rmtree
4. `data/temp/.hypothesis/` — shutil.rmtree
5. `data/temp/.pytest_cache/` — shutil.rmtree
6. `data/temp/tmp-*` — ディレクトリ + 中身を shutil.rmtree

**出力フォーマット**:
```
[DRY-RUN] Would remove 5402 files from config/ab_search_temp_*.json (1.2 MB)
[DRY-RUN] Would remove data/temp/.mypy_cache/ (12740 files, 48.3 MB)
...
Total: 18,395 files, 50.7 MB  (add --execute to actually delete)
```

**安全措置**:
- `--execute` なしでは何も削除しない
- tracked ファイルは絶対に削除しない (`git ls-files` で確認)
- 削除前にサマリを表示して 1 秒待機
- 削除結果のサマリログを標準出力に表示

### テスト

`tests/unit/v460/test_701_cleanup_workspace.py`:

1. `test_dry_run_does_not_delete` — dry-run で実際に削除されないことを確認
2. `test_execute_removes_ab_search_temp` — tmp_path に ab_search_temp ファイルを作成し、execute モードで削除確認
3. `test_tracked_files_never_deleted` — git tracked ファイルが削除対象に含まれないことを確認
4. `test_empty_workspace_no_error` — 対象ファイルが存在しない場合にエラーにならないことを確認
5. `test_summary_output_format` — 出力フォーマットが仕様通りであることを確認

## 制約

- `shutil.rmtree` は `onerror` コールバックで権限エラーをスキップ
- Windows / Linux 両対応 (Path オブジェクトを使用)
- `argparse` を使用した CLI
- 型注釈必須、Any 型禁止
- 既存の `ztb/utils/` ユーティリティがあれば再利用

## 成果物

- `scripts/v460/tools/cleanup_workspace.py`
- `tests/unit/v460/test_701_cleanup_workspace.py`
