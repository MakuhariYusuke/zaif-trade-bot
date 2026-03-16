# AGENTS.md

## プロジェクト概要
Coincheck BTC/JPY マーケットメイカー（SAC強化学習）。短期高収益が大義、長期健全性とのバランスを取る。

## 技術スタック
- Python 3.11 / venv: `.venv/`
- テスト: `python -m pytest tests/ -x --tb=short`
- 型検査: `mypy --config-file mypy.ini`
- コミット: `git commit --no-verify -m "..."` （pre-commitフック回避）

## ディレクトリ構造
| パス | 内容 |
|------|------|
| `ztb/` | コアライブラリ（metrics, ml, trading, utils 等） |
| `scripts/v460/` | 現行バージョンのスクリプト群 |
| `scripts/v460/lib/` | fill_test 実行系モジュール |
| `scripts/v460/ml/` | ML パイプライン |
| `tests/unit/v460/` | v460 ユニットテスト |
| `configs/v460/` | YAML 設定 |
| `docs/v460/` | バージョン別ドキュメント |

## コーディング規約
- **DRY / SOLID / SRP** を徹底。既存実装を最大限再利用し、新規作成は最小限
- **Any 型禁止**。型安全を優先し mypy を活用
- 例外処理・メモリリーク・パフォーマンスに常時注意
- god object を避け、ファイルは適度に分割

## git 運用ルール
- `git add` は対象ファイルを**個別指定**。`git add .` 禁止
- コミット前に `git diff --cached --stat` で差分確認必須
- ドキュメントは毎回更新してコミットに含める
- ドキュメント番号 `NNN#` はインクリメンタルに採番（既存最大+1）

## ⚠ 既知の問題: git 追跡ファイル消失
session037 コミット群（116個）が `git add` 漏れにより追跡ファイルを 5,002→105 に激減させた。ディスク上のファイルは無事。修復手順は `temp/prompt_git_fix.md` を参照。
