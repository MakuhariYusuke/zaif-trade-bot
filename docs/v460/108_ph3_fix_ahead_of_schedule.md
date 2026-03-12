# 108# ph3 先行改善 — 018#/021#/106# 残課題の前倒し実施

| key | value |
|-----|-------|
| type | fix/impl |
| phase | ph3 (先行) |
| status | committed |
| parent | 018#, 021#, 106# |
| tests | 827 passed (v460 unit tests, リグレッションなし) |

---

## §1 背景

107# fill_test を 48h 観察中。並行して ph3 以降のドキュメントを全面レビューし、
fill_test に影響せず先行着手可能なタスクを特定・実施した。

## §2 実施内容

### M5: `LivePositionConfig` 重複定義の解消

- **ファイル**: `ztb/trading/live_trader/live_trader.py`
- **問題**: `__init__` 内に同一クラスが2箇所（L224 dry-run用, L425 live用）で重複定義
- **対応**: モジュールレベルに1つだけ定義し、両箇所から参照に統一
- **効果**: ~30行削減、定義の一貫性確保

### C3: `SACAlgorithmTrainer.train()` の `vec_env.close()` 欠落修正

- **ファイル**: `ztb/training/trainers/sac_trainer.py`
- **問題**: `DummyVecEnv` 作成後に `close()` が呼ばれず環境リソースがリーク
- **対応**: vec_env 作成以降を `try/finally` で囲み、`finally` で `vec_env.close()` を確実に呼出
- **効果**: 訓練終了時・例外時のリソースリーク防止

### M1: `_get_current_market_regime()` デバッグコード除去

- **ファイル**: `ztb/trading/environment/heavy_env/core.py`
- **問題**: コメントアウト済みのデバッグ出力があるが、every 1000 steps で統計計算だけは実行中
- **対応**: デバッグブロック全体を削除（~10行）
- **効果**: 不要な計算コストの排除、コード可読性向上

### DUP2: sac_utils 関連の整理

- **問題1**: `ztb/training/__pycache__/sac_utils.cpython-311.pyc` が古い削除済みモジュールのキャッシュとして残存
- **対応**: キャッシュファイルを削除
- **問題2**: `ztb/training/examples.py` と `ztb/training/compare_methods.py` が存在しない `ztb.optimization.sac_utils` を import（デッドコード + 壊れた import）
- **対応**: `archived/` に移動

## §3 影響範囲

| 変更対象 | fill_test 影響 | リスク |
|----------|--------------|--------|
| `live_trader.py` (M5) | なし — fill_test は `run_fill_test.py` | 極小 |
| `sac_trainer.py` (C3) | なし — 訓練時のみ | 極小 |
| `heavy_env/core.py` (M1) | なし — 訓練環境のみ | 極小 |
| デッドファイル移動 (DUP2) | なし | ゼロ |

## §4 018# 残課題ステータス更新

| ID | 内容 | ステータス |
|----|------|-----------|
| C3 | `vec_env.close()` 欠落 | **✅ 108# で実施** |
| H3 | `_market_regime_cache` reset 時未 clear | 後日 (HeavyTradingEnv reset 大改修時) |
| H5 | `_get_info()` 毎 step features/config 含む | 後日 (SB3 info 挙動確認後) |
| M1 | DataFrame → numpy slicing | **✅ 108# でデバッグコード除去** (※ `detect_regime()` が DataFrame 必須のため完全numpy化は不可) |
| M5 | `LivePositionConfig` 重複 | **✅ 108# で実施** |
| DUP2 | `sac_utils` 2ファイル | **✅ 108# で確認・残存整理** (統合は 063# で実施済み) |
| DUP3 | `UnifiedTrainer` 2835L God Object | 後日 (ph3 本格アーキテクチャ再設計) |

## §5 106# R1-R10 残課題ステータス

変更なし（R1/R4/R5/R6/R7 は大規模リファクタ or v461 移行時に実施予定）。

---

> **文書管理**
> - 作成日: 2026-02-18
> - フェーズ: ph3 先行 (107# fill_test 観察中)
> - 前提文書: 018#, 021#, 106#
