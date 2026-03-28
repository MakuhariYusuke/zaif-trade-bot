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
- 日常確認は `git status -uno` を基本にし、untracked 確認が必要な時だけ通常 `git status` を使う
- コミット前に `git diff --cached --stat` で差分確認必須
- ドキュメントは毎回更新してコミットに含める
- ドキュメント番号 `NNN#` はインクリメンタルに採番（既存最大+1）

## 現行アーキテクチャ要点 (649#時点)
- **Safety層**: SAD / MCB は `enabled: true`（606#）。607# で hot-reload 対応済み（`mcb_enabled`/`sad_enabled` 変更時にコンポーネント再構築、状態継承あり）
- **entry_gate**: observe モード（`enabled: false` + CalibrationMap 接続済み）
- **retrain_scheduler**: 649# でデータ鮮度チェックを retrain trigger から分離。`data_freshness_check_interval_sec` (1h) で独立して OHLCV 更新
- **提案文書の検証**: AI 生成の「現在値」は必ず YAML + コードで検証すること（592#/605# 教訓）
