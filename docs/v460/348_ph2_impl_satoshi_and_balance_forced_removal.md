# 348# Satoshi 精度化 + balance_forced 撤廃

## 概要

348# セッションで実施した 2 つの構造的改善:

1. **Satoshi 精度化**: ロットサイズの精度を mBTC (0.001) から satoshi (1e-8) に変更
2. **balance_forced 撤廃**: 存在意義が疑問視されていた balance_forced 機構を全面廃止

## 1. Satoshi 精度化 (lot_step: 0.001 → 0.00000001)

### 背景

347# の 1mBTC 最低ロット制約分析で特定された 6 つの死蔵メカニズムへの対応の第一歩。
Zaif/Coincheck BTC の最小単位は satoshi (1e-8 BTC) であり、ロットステップをこれに合わせることで
より柔軟なポジションサイジングが可能になる。

### 変更内容

| ファイル | 変更 |
|---|---|
| `scripts/v460/lib/lot_sizer.py` | `lot_step: 0.001 → 0.00000001`, `clamp_lot` 丸め精度 4桁→8桁 |
| `configs/v460/fill_test.yaml` | `lot_step: 0.001 → 0.00000001` |
| `scripts/v460/lib/maker_price.py` | Dead `_MIN_ORDER_BTC: Final[float] = 0.001` 削除 |
| `scripts/v460/lib/adaptation_engine.py` | Dead `_MIN_ORDER_BTC: Final[float] = 0.001` 削除 |
| `scripts/v460/lib/balance_checker.py` | `MIN_ORDER_BTC` コメント更新 |
| `tests/unit/v460/test_lot_sizer.py` | satoshi 精度のテスト期待値更新 |

## 2. balance_forced 撤廃

### 背景

balance_forced は片方のサイドの残高不足時に強制的にサイド切り替えを行う機構だった。
しかし以下の理由から存在自体が疑問視されていた:

- **複雑性の温床**: 20+ ソースファイル、30+ テストファイルに跨る巨大な機構
- **デッドロック要因**: balance_forced + per-side halt + kill の三重デッドロックが頻発
- **品質劣化**: 強制トレードは本質的に不利な PnL を生む
- **代替手段の成熟**: inventory_escape, quiescence 等の洗練された代替機構が既に稼働

### 削除したソースファイル変更 (~20 ファイル)

| 対象 | 削除内容 |
|---|---|
| `orchestrator_pre_cycle.py` | CycleContext から `balance_forced`, `is_rescue` フィールド |
| `orchestrator_balance.py` | `_track_balance_forced_frequency()`, 強制切替ロジック |
| `orchestrator_mid_cycle.py` | `_handle_balance_forced_skip`, `_handle_forced_buy_delay` (~150 行) |
| `cycle_gate_aggregator.py` | `balance_forced: bool` パラメータ、ゲートバイパスロジック |
| `fill_cycle_executor.py` | `balance_forced_switch`, rescue offset パラメータ |
| `fill_record_builder.py` | `balance_forced_switch` フィールド構築 |
| `fill_record_helpers.py` | `_balance_forced_skip_count` リストア |
| `fill_loop_orchestrator.py` | RunSessionState の 10+ 強制関連フィールド |
| `fill_config.py` | 13+ 設定フィールド (`forced_fill_pnl_downweight` 含む) |
| `fill_config_parser.py` | 全 balance_forced YAML パース |
| `fill_config_validation.py` | `balance_forced_cooldown_sec` バリデーション |
| `config_hot_reload.py` | 全 balance_forced/forced_buy_delay エントリ |
| `cancel_reasons.py` | `BALANCE_FORCED_SKIP`, `FORCED_BUY_DELAY` 定数 |
| `orchestrator_guards.py` | forced fill downweight ロジック |
| `orchestrator_lifecycle.py` | warmup forced downweight |
| `orchestrator_post_cycle.py` | forced buy/sell KPI トラッキング |
| `guard_reason_classifier.py` | `balance_forced_halt_block`, `forced_buy_delay` 分類 |
| `hindsight_filter.py` | `CR.BALANCE_FORCED_SKIP` 参照 |
| `run_fill_test.py` | `_balance_forced_skip_count` 属性 |
| `configs/v460/fill_test.yaml` | 全 balance_forced 設定エントリ |

### 削除/修正したテストファイル (~30 ファイル)

ASTベースの自動スクリプトを使用して、10 クラス全体削除、15+ メソッド個別削除、
5+ 部分修正 (assert行削除、parametrize エントリ削除) を実施。

### 残存設計

- `degraded_liquidation_*` 設定: inventory_escape パス経由で依然有効
- `balance_forced_switch` FillRecord フィールド: 過去のレコード互換性のため FillRecord dataclass に残存
- Guard fire 名 `balance_forced_halt_block` → `per_side_halt_block` にリネーム

## セルフレビュー結果

### BUG: 0件

### WARN → 修正済み (3件)

| # | 内容 | 対処 |
|---|---|---|
| W-1 | `per_side_halt_block` が guard_reason_classifier に未登録 | RECOVERY として登録 + test_244 にテスト追加 |
| W-2 | lot_sizer 有効化に必要な最低残高がコメント未記載 (347# L-4) | LotSizingConfig docstring に目安追記 |
| W-3 | test_158 に `BALANCE_FORCED_SKIP` デッドコード参照 | コメント化 |

### 未対処 (Coincheck 検証待ち)

- **347# L-3**: `min_order_btc: 0.001 → 0.0005` — Coincheck 板取引の最小注文量要確認

### 後方互換性確認

- `FillRecord.balance_forced_switch`: ztb/metrics/fill_quality.py に残存 (設計通り)
- 分析ツール (tail_loss_analysis, retrain_scheduler): 既存データ遡及分析のため参照を維持
- 新規レコードでは常に `False`/`None` が設定される

## テスト結果

```
4209 passed, 0 failed (test_ml_pipeline 除外: 既知のデータ不足)
```

## ドキュメント命名規則

346#/347# のファイル名を `NNN_phX_TYPE_description.md` 規則に統一:
- `346_test_coverage_and_tail_loss_analysis.md` → `346_phg_rpt_test_coverage_and_tail_loss_analysis.md`
- `347_min_lot_constraint_analysis.md` → `347_ph2_rpt_min_lot_constraint_analysis.md`
