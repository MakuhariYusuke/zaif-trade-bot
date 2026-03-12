# 327# fix: プロアクティブバグハント

**日付**: 2026-03-08
**種別**: fix (バグ修正・改善)
**コミット**: `4b1aa540e`
**前提**: 326# (`401ac0116`)

---

## 1. 目的

326# (Mixin Audit) 完了後、コードベースの予防的品質点検を実施。
ゼロ除算リスク・ファイルハンドルリーク・ログ蓄積問題を事前に発見・修正。

---

## 2. 修正内容

### 2.1 CRITICAL: `loss_cap_ratio` ゼロ除算防止

**ファイル**: `scripts/v460/lib/fill_config.py`

`loss_cap_ratio` は被除数として使用されるが、`__post_init__` でのバリデーションが
欠如しており、`0.0` 設定時に `ZeroDivisionError` でプロセスが即死するリスクがあった。

**対処**:
- `loss_cap_ratio > 0` バリデーション追加
- `soft_loss_cap_ratio >= 0` バリデーション追加
- テスト 3 件追加 (`test_168_daily_drawdown_guard.py`)

### 2.2 HIGH: `event_logger.py` ファイルハンドルリーク

**ファイル**: `scripts/v460/lib/event_logger.py`

`setup_stderr_mirror()` で `stderr_file` を `open()` した後の例外パスで
ファイルハンドルがリークする可能性があった。

**対処**:
- `stderr_file = None` 初期化を `try` ブロック前に移動
- `except` ブロック内で `stderr_file.close()` を追加

### 2.3 HIGH: `fill_test_cli.py` ファイルハンドルリーク

**ファイル**: `scripts/v460/lib/fill_test_cli.py`

`_start_retrain_scheduler()` で `retrain_stderr_fh` のセットアップ失敗時に
ファイルハンドルがリークする可能性があった。

**対処**:
- `except` ブロック内で `retrain_stderr_fh.close()` と `None` 代入を追加

---

## 3. テスト

### 追加テスト (3 件)

| テスト名 | 内容 |
|---|---|
| `test_loss_cap_ratio_zero_raises` | `loss_cap_ratio=0.0` → `ValueError` |
| `test_loss_cap_ratio_negative_raises` | `loss_cap_ratio=-0.01` → `ValueError` |
| `test_soft_loss_cap_ratio_negative_raises` | `soft_loss_cap_ratio=-0.01` → `ValueError` |

### 全テスト結果

- fill-related: **678 passed**, 10 skipped
- 全体: **4,072 passed**

---

## 4. 変更ファイル一覧

| ファイル | 変更 | 概要 |
|---|---:|---|
| `scripts/v460/lib/fill_config.py` | +9 | `__post_init__` バリデーション追加 |
| `scripts/v460/lib/event_logger.py` | +4 | ファイルハンドルリーク防止 |
| `scripts/v460/lib/fill_test_cli.py` | +4 | ファイルハンドルリーク防止 |
| `tests/unit/v460/test_168_daily_drawdown_guard.py` | +15 | バリデーションテスト 3 件 |
| **合計** | **+32** | |

---

## 5. 運用影響

- **本番Bot**: 影響なし — `loss_cap_ratio` は YAML で正値設定済み
- **ファイルハンドル**: 長時間稼働 (22 日, 42 fills, PID 58008) でのリーク防止
- **リスク**: 最小 — バリデーション追加のみ、既存ロジック変更なし
