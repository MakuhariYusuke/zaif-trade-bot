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
- test_retrain_hot_reload: 5 FAILED (fill_records_20260220.jsonl の
  FillRecord schema 不一致。order_price/order_quantity 欠落。pre-existing)

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
