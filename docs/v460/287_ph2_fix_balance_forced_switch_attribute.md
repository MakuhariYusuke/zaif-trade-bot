# 287# CRITICAL fix: balance_forced → balance_forced_switch 属性名修正

**Phase**: ph2 G1.1-exec  
**Type**: fix  
**Date**: 2026-03-06  
**Parent**: 286# (`804a5a1bdc2c`)  
**Commit**: `e7d2f50d9`  

---

## 概要

286# P1-5（強制買い KPI 分離トラッキング）で導入された `_process_post_cycle()` 内で、
`FillRecord` の属性名を `balance_forced` と誤記していた。正しくは `balance_forced_switch`。
属性名不一致により `AttributeError` が発生し、fill_test プロセスがクラッシュ。

watchdog による自動再起動が機能したが、再起動後のコードにも同一バグが残っていたため、
buy fill 成功時に再度クラッシュする可能性があった。hot swap で修正版を即時反映。

---

## クラッシュ詳細

- **発生日時**: 2026-03-05T17:14:07 (UTC)
- **stop_reason**: `crash:AttributeError`
- **PID**: 54240
- **git_sha**: `804a5a1bdc2c`

### トレースバック

```
File "scripts/v460/lib/fill_loop_orchestrator.py", line 1105, in _process_post_cycle
    if record.balance_forced:
       ^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'FillRecord' object has no attribute 'balance_forced'
```

### 原因分析

`FillRecord` dataclass（`ztb/metrics/fill_quality.py` L118）の正しいフィールド名:
```python
balance_forced_switch: bool | None = None  # 残高不足で side が強制切替されたか
```

286# 実装時に `_process_post_cycle()` の KPI 分離ロジック（L1105）で `_switch` サフィックスを
落として `record.balance_forced` と記述。dataclass ベースのため `__getattr__` fallback なく
即 `AttributeError`。

---

## 修正内容

### コード修正（1行）

**ファイル**: `scripts/v460/lib/fill_loop_orchestrator.py` L1105

```python
# Before (286#):
if record.balance_forced:

# After (287#):
if record.balance_forced_switch:
```

### 回帰テスト（2件追加）

**ファイル**: `tests/unit/v460/test_286_comprehensive_resolution.py`

1. `test_fill_record_balance_forced_switch_attribute` — `FillRecord` が `balance_forced_switch` を持ち、`balance_forced` を持たないことを検証
2. `test_process_post_cycle_uses_balance_forced_switch` — `_process_post_cycle` のソースに `record.balance_forced:` が含まれず `record.balance_forced_switch` が含まれることを検証

### 網羅監査

FillRecord の全 72 フィールドに対し、`scripts/v460/` 内のドットアクセス約 60 箇所、
辞書アクセス約 30 箇所を照合。**他に属性名ミスマッチは存在しないことを確認**。

---

## 反映方法

- hot swap 再起動（`ops/windows/hot_swap_restart.ps1`）で即時反映
- 再起動後の `git_sha`: `e7d2f50d9bda`
- watchdog で稼働継続を確認済み

---

## 教訓と再発防止

本バグは 279# (`config.min_lot` → `config.min_order_btc`) および 281# (`_HALT_PERSIST_INTERVAL`
未定義) と同系統の「属性名/変数名の取り違え」パターン。

**再発防止策**:
1. **mypy strict**: FillRecord アクセスを mypy で静的検査（現在 mypy で検知可能な状態）
2. **回帰テスト**: ソースコード内の属性名をテストで直接検証するパターンの確立
3. **ドキュメントへの反映**: 286# doc に修正注記を追記し、index.md に本 287# を登録

---

## 変更ファイル一覧

1. `scripts/v460/lib/fill_loop_orchestrator.py` — L1105 属性名修正
2. `tests/unit/v460/test_286_comprehensive_resolution.py` — 回帰テスト 2 件追加
3. `docs/v460/286_ph2_fix_282_284_comprehensive_resolution.md` — 修正注記追記
4. `docs/v460/index.md` — 287# エントリ追加
5. `docs/v460/287_ph2_fix_balance_forced_switch_attribute.md` — 本ドキュメント (NEW)
