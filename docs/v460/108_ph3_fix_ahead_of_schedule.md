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

108# 当時は「変更なし」としていたが、その後の session037 で状況は大きく進んだ。

- R1:
  - `run_fill_test.py` 分割後の抽出先について、`scripts/v460/lib` に残った domain logic の canonical 化を継続
- R3:
  - `SkipGate` の runtime / result metadata / FillRecord payload 境界まで unit/migration test を補強
- R5:
  - `lib -> ztb` 移行は `v461` 以降送りではなく、主要部分を session037 で前倒し実施
  - canonical 化済み:
    - `cancel_reasons`
    - `param_adapter`
    - `lot_sizer`
    - `fast_fill_defense`
    - `sac_common`
    - `regime_detector`
    - `bayesian_regime_filter`
- R6/R7:
  - repo 全体の構造整理自体は依然として別テーマだが、少なくとも `lib -> ztb` の責務分離は前進済み

---

> **文書管理**
> - 作成日: 2026-02-18
> - フェーズ: ph3 先行 (107# fill_test 観察中)
> - 前提文書: 018#, 021#, 106#

## 2026-03-21 補遺

108# で「ph3 以降の前倒し」として着手したテーマは、その後さらに進み、
`Phase 3/4` の実装と `lib -> ztb` 再配置が現実の作業トラックになった。

特に次の点は、108# 時点の見立てより前に進んでいる。

- `scripts/v460/lib` の domain logic を `ztb` へ寄せる方針が、計画ではなく実装段階へ移行
- `skip_gate_evaluator` と `maker_price` は God Object のまま放置せず、
  pure math / result assembly / runtime contract を段階的に抽出
- test-side でも canonical import 収束を進め、shim 依存を大きく削減

そのため、108# の「v461 以降で本格対応」という表現は、現状では
「当時の保守的な見積もり」として読むのが適切である。
