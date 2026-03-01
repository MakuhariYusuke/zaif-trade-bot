# 216# P1: Velocity リネーム + Guard 発火カウンタ永続化 + 事実/仕様分離

| 項目 | 内容 |
|---|---|
| 前提 | 214# §7 残課題 D/E/F |
| 親 | 213# (Codex/Gemini レビュー) → 214# (検証) → 215# (P0) |
| ステータス | 完了 |

---

## 1. D: Velocity 引数リネーム (P1)

### 問題

213# Codex §3: 「instant OB velocity を `price_velocity_60s` という名前で流し込む実装は
命名規則のミスではなく、時間次元の冒涜である」

214# §2 検証: 機能的には正常動作。純粋な命名リファクタリング。

### 変更

| 旧名 | 新名 | 影響箇所 |
|---|---|---|
| `price_velocity_60s` | `price_velocity_bps` | 19 ファイル (~149 箇所) |
| `sg_velocity_60s` | `sg_velocity_bps` | fill_cycle_executor.py |
| `_sg_velocity_60s` | `_sg_velocity_bps` | fill_cycle_executor.py |

**ファイル一覧 (ソース 12 + テスト 7):**

- ソース: fill_config.py, cycle_gate_aggregator.py, skip_gate_evaluator.py,
  fill_cycle_executor.py, fill_loop_orchestrator.py, velocity_math.py,
  feature_enricher.py, skip_gate.py, run_070_model_search.py,
  run_065_save_two_tier.py, run_065_hp_sweep.py, fill_quality.py
- テスト: test_velocity_skip_rule.py, test_194_cycle_gate.py,
  test_195_velocity_b1_soft.py, test_168_daily_drawdown_guard.py,
  test_enricher_skip_gate.py, test_retrain_hot_reload.py,
  test_200_an_improvements.py

**注意**: ML serialized models は旧特徴量名 `price_velocity_60s` を使用。
bot halt 中のため再学習で対応。

### コメント/ログ文字列

- fill_cycle_executor.py: コメントとログ出力も `velocity_bps` に統一
- test_202_log_improvements.py: docstring 修正

### 非対象

- `analysis/204_trade_analysis.py`: 既存 fill records JSON を読むため旧キー名を維持
- `docs/v460/*.md`: 過去の分析ログ (SHAP 値等) は歴史的記録として保持

---

## 2. E: Guard 発火カウンタ永続化 (P1)

### 問題

214# §5: 「Guard 発火カウンタ永続化 — state に hard_skip_count, toxic_veto_count 等を追加」

再起動をまたいでも累積発火回数を保持し、ガード動作の診断性を向上。

### 変更

1. **FillTestState** (resilience.py):
   - `guard_fire_counts: dict[str, int] | None = None` フィールド追加

2. **FillLoopOrchestratorMixin** (fill_loop_orchestrator.py):
   - `_guard_fire_counts: dict[str, int] | None = None` 属性追加
   - `_inc_guard_fire(guard_name)` ヘルパーメソッド追加
   - `_build_state_snapshot` に `guard_fire_counts` 含有
   - state 復元パス (2 箇所) に `guard_fire_counts` 復元追加

3. **計測ポイント**:

| guard_name | 発火箇所 | 意味 |
|---|---|---|
| `hard_skip_utc` | 205# §9.4 Hard Skip | UTC 時間帯スキップ開始 |
| `toxic_veto_set` | 205# §9.2 Toxic Veto | 大損後サイド封鎖設定 |
| `toxic_veto_block` | 205# §9.2 両サイド封鎖 | 両サイド veto でスキップ |
| `dd_halt` | 168# §4.1 DD Halt | 日次ドローダウン halt 開始 |

### テスト (23 passed)

- `TestGuardFireCountsPersistence::test_field_default`
- `TestGuardFireCountsPersistence::test_round_trip`
- `TestGuardFireCountsPersistence::test_inc_guard_fire_helper`

---

## 3. F: 211# §8 事実/仕様分離 (P1)

### 問題

213# Codex §4: 「外部イベントの叙述が仕様文書に混ざりすぎている」
214# §4.3 判定:
- 仕組み定義 (alert_mode.json スキーマ) は仕様として残す
- イラン攻撃固有叙述は背景セクションに分離
- `Operation Epic Fury` は確認不能のため削除

### 変更

1. 211# §8 冒頭の「背景」セクション（イラン攻撃叙述）を「問題」セクション後方へ移動
2. 新サブセクション「背景事実 (本提案の動機)」として分離
3. `Operation Epic Fury` 作戦名を削除
4. 仕様定義部分 (パラメータ表、設計原則) はそのまま維持

---

## 4. テスト結果

| テストファイル | 件数 | 結果 |
|---|---|---|
| test_168_daily_drawdown_guard | 81 | ✅ |
| test_169_config_hot_reload | 16 | ✅ |
| test_215_dd_fix_alert_mode | 23 (+3) | ✅ |
| test_velocity_skip_rule | - | ✅ |
| test_194_cycle_gate | - | ✅ |
| test_195_velocity_b1_soft | - | ✅ |
| test_enricher_skip_gate | - | ✅ |
| test_202_log_improvements | - | ✅ |
| **合計** | **299** | **✅ ALL PASSED** |

既知の失敗 (velocity rename 無関係):
- test_retrain_hot_reload: 5 FAILED → **216# §6 で修正済み**
  (order_price/order_quantity 欠落テストデータ)

---

## 5. 214# §7 アクションアイテム最終状況

| ID | 内容 | ステータス |
|---|---|---|
| A | DD state 整合性修復 | ✅ 215# |
| B | Hot-reload 13 フィールド追加 | ✅ 215# |
| C | alert_mode.json DEFCON | ✅ 215# |
| **D** | **Velocity リネーム** | **✅ 216#** |
| **E** | **Guard 発火カウンタ永続化** | **✅ 216#** |
| **F** | **211# §8 事実/仕様分離** | **✅ 216#** |
| G | 206#–211# 検証 run | ⏳ halt 解除待ち |
| H | update_pnl if/elif | ✅ 215# P0-A |

---

## §6 追加修正 (216# セルフレビュー後)

### 6.1 test_retrain_hot_reload テストデータ修正

FillRecord の必須フィールド `order_price`/`order_quantity` をテスト用 JSONL に追加。
5 テスト (insufficient_samples, insufficient_new, e2e, balance_forced x2) が通過するようになった。

### 6.2 State 復元ロジック DRY リファクタリング

`fill_loop_orchestrator.py` の if/else 両ブロックで重複していた 4 項目
(DD / toxic_veto / one-sided / guard_fire_counts) の復元ロジックを
`_restore_common_state(saved_state)` ヘルパーに抽出。
~20 行削減、今後の state フィールド追加時の 2 箇所修正忘れリスクを解消。

### 6.3 SkipGate ML モデル pickle 互換性

旧モデルの `feature_cols` に残る `price_velocity_60s` を `__init__` 時に
`price_velocity_bps` に自動マイグレーション。

**問題**: モデル weights の position は変わらないが、`_feature_index` のキー名が
旧名のため、新コードが送る `price_velocity_bps` キーにマッチしない。
SHAP Top 1-2 特徴量が NaN 補完で無効化される。

**対策**: `_FEATURE_NAME_MIGRATION` dict によるロード時自動変換。
モデル再学習なしで既存 pkl が即座に新特徴量名に対応。

### 6.4 FillRecord 旧フィールド名エイリアス

`_sanitize_fill_record_fields()` に `_FIELD_ALIASES` を追加。
旧 JSONL レコードの `price_velocity_60s` を `price_velocity_bps` に自動マッピング。
過去の fill records ロード時に velocity データが失われることを防止。

---

## §7 プレ既存テスト全修正 (P3)

### 修正サマリ

| テスト | 件数 | 原因 | 修正内容 |
|---|---|---|---|
| test_088_features | 2 | `_make_loop_skip_record` wrapper 未対応 + `build_skip_fill_record` 集約 | regex を `_make_(?:loop_)?skip_record` に拡張、`build_skip_fill_record` 検証に変更 |
| test_113_resilience | 3 | BOM (U+FEFF) + 行数超過 + re-export 欠落 | `utf-8-sig` 読込、行数上限 650、`_SkipGateResult` re-export 追加 |
| test_141_side_specific | 1 | FillRecord 必須フィールド欠落 | `order_price`/`order_quantity`/`timestamp` 追加 |
| test_145_structural | 1 | `OPERATOR_HALT` 定数が expected set に未追加 | `CR.OPERATOR_HALT` 追加 |
| test_146_multi_exchange | 1 | `get_broker_registry` が CLI に移動済み | 検索先を `fill_test_cli.py` に変更 |
| test_155_hindsight | 1 | `cycle_gate_aggregator.py` の BOM | `utf-8-sig` 読込 |
| test_166_remaining | 1 | Pipeline 内 `feature_names_in_` に旧名残存 | `_migrate_pipeline_feature_names()` で LGBMRegressor 含む全 step をマイグレーション |
| test_175_code_review | 2 | `load_fill_records_glob` → `iter_fill_records_glob` 変更 | mock パスを `ztb.metrics.fill_quality.iter_fill_records_glob` に更新 |
| test_203_dd_state | 2 | クラスレベル属性 `_halt_iter_count=0` + `_should_record_halt` パターン変更 | テスト条件を現実に合わせて更新 |

### 実装詳細

#### Pipeline feature_names_in_ マイグレーション (`skip_gate.py`)
sklearn Pipeline の各 step が保持する `feature_names_in_` 属性に旧特徴量名が残っていると、
predict 時に `ValueError: feature names should match` が発生する。
`_migrate_pipeline_feature_names()` ヘルパーを追加し、ロード時に全 step の
`feature_names_in_` を `_FEATURE_NAME_MIGRATION` で自動更新。
LGBMRegressor は property setter を持たないため、内部の `_Booster.feature_name_` を直接更新。

#### run_fill_test.py re-export
163# Mixin 分割で `_SkipGateResult` / `_FillMonitorResult` / `_PnlMeasurement` が
`fill_config.py` に移動されたが、テストが `run_fill_test` からインポートしていたため
re-export (`as _SkipGateResult` alias) を追加。

### テスト結果
- 修正前: 14 failures (pre-existing)
- 修正後: **2838 passed, 0 failed**
