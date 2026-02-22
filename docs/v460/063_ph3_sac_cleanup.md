# 063# ph3 SAC重複実装の整理

**Phase**: ph3 (準備フェーズ)  
**Date**: 2025-02-15  
**Commit**: (pending)

## 概要

015# SAC調査ドキュメントに基づき、V433/V435世代のデッドコード削除、
DynamicLRScheduler移設、deprecationマーキングを実施。

## 変更内容

### 1. Dead Code削除: trainer.py V433ブロック (-246行)

| メソッド | 行数 | 状態 |
|---|---|---|
| `_initialize_v433_components()` | ~107L | 削除 (呼出箇所なし) |
| `_setup_v433_adaptive_training()` | ~55L | 削除 (enable_v433_adaptive=False) |
| `_execute_v433_adaptive_training()` | ~84L | 削除 (同上) |
| `enable_v433_adaptive` フィールド | - | 削除 |
| V433 conditional blocks (train()) | ~12L | 削除 |
| TYPE_CHECKING: UnifiedOptimizer | - | 削除 |

**根拠**: `enable_v433_adaptive` はline 283で常にFalse、呼び出し元なし。
Importされていたモジュールは前回030#で既にアーカイブ済み。

### 2. リネーム: `_create_v433_training_environment` → `_create_training_environment`

- V433固有ではない汎用環境作成メソッド
- `_create_backtest_environment()` から呼び出されるため保持
- ログメッセージも更新

### 3. DynamicLRScheduler移設

| 項目 | 内容 |
|---|---|
| 移設元 | `ztb/training/sac_v430_training_optimizations.py` |
| 移設先 | `ztb/training/unified_trainer/base/lr_scheduler.py` |
| 利用箇所 | `callbacks.py`, `base_trainer.py` (import更新済み) |

移設元ファイル全体を `archived/training/` にアーカイブ。
`GradientAccumulator`、`EarlyStopping` は利用箇所なし。

### 4. V435スクリプトアーカイブ (5ファイル)

```
scripts/training/train_sac_v435_3.py  → archived/training/v435/
scripts/training/train_sac_v435_4.py  → archived/training/v435/
scripts/training/train_sac_v435_5.py  → archived/training/v435/
scripts/training/train_sac_v435_5_direct.py → archived/training/v435/
scripts/training/train_sac_v435_6.py  → archived/training/v435/
```

0 importers (完全孤立)

### 5. Deprecation通知

| ファイル | 状態 | 理由 |
|---|---|---|
| `ztb/training/trainers/sac_trainer.py` | deprecated通知追加 | 7 active importers |
| `ztb/training/sac_trainer.py` | deprecated通知追加 | Facade layer |

### 6. 既知バグ修正

- `market_regime.py:473`: `"ActionSignal" | None` → `Optional["ActionSignal"]`
  - Python 3.10以前で `|` union構文はランタイムエラー
  - `from __future__ import annotations` なしでは使用不可

## 削減量

| カテゴリ | 行数 |
|---|---|
| trainer.py dead code削除 | -246行 |
| V435スクリプトアーカイブ | -5ファイル |
| sac_v430_optimizations移設→アーカイブ | -422行 (1クラス移設) |
| **合計** | ~-668行, -6ファイル |

## SAC実装整理状況 (015# 更新)

| # | ファイル | 状態 |
|---|---|---|
| 1 | `unified_trainer/algorithms/sac_trainer.py` | ✅ KEEP (canonical) |
| 2 | `trainers/sac_trainer.py` | ⚠️ deprecated (7 importers) |
| 3 | `sac_trainer.py` | ⚠️ deprecated (facade) |
| 4 | `sac.py` (SACSuite CLI) | ✅ KEEP |
| 5 | `adaptive_sac_core.py` | ✅ archived (030#) |
| 6 | V435 scripts | ✅ archived (063#) |
| - | `sac_v430_training_optimizations.py` | ✅ archived (063#) |

## テスト結果

- 637 passed (v460テスト全通過)
- SB3依存テストは環境非対応で別途対応予定
