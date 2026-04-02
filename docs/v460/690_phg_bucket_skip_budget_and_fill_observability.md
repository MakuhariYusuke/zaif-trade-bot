# 690# Skip Budget + Codex 5 件一括投入 + Bot 停止調査

| 項目 | 内容 |
|------|------|
| 日付 | 2026-04-02 |
| SHA | b5f7828b1 (torch DLL fix + skip budget prompt) |
| 前提 | 689# (abff47b79) timeout_regime + trace_id |

## §1 Bucket 別 Skip Budget (Codex 実装済み)

### 目的

- `skip_gate` の連続 skip 安全弁を global counter だけで扱うのをやめ、`regime × side` の bucket で制御する
- `bypass_mode=true` でも skip 統計は失わず、FillRecord 側で追えるようにする
- `primary_max_consecutive_skip` はグローバル緊急ブレーキとして残し、budget と独立させる

## 今回の実装 (Skip Budget)

### 1. runtime

- 新規: [skip_gate_budget.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/skip_gate_budget.py)
  - `BucketKey`
  - `BucketState`
  - `BucketedSkipBudget`
- [skip_gate_evaluator.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/skip_gate_evaluator.py)
  - model decision 後、primary safety valve 前に budget check を挿入
  - budget 枯渇時は `budget_exhausted_pass`
  - budget 統計は final block 可否ではなく raw skip 判定で記録
  - これにより `bypass_mode=true` でも統計だけは積める

### 2. config

- [fill_config.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/fill_config.py)
  - `skip_gate_budget_enabled`
  - `skip_gate_budget_window_min`
  - `skip_gate_budget_limits`
  - `get_budget_limit(...)`
- [fill_config_parser.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/fill_config_parser.py)
- [fill_config_validation.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/fill_config_validation.py)
- [config_hot_reload.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/config_hot_reload.py)

### 3. observability

- [fill_config_results.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/fill_config_results.py)
- [skip_gate_result_fields.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/ztb/ml/skip_gate_result_fields.py)
- [skip_gate_fill_record.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/ztb/ml/skip_gate_fill_record.py)
- [fill_record_builder.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/fill_record_builder.py)
- [fill_cycle_executor.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/fill_cycle_executor.py)
- [fill_quality.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/ztb/metrics/fill_quality.py)

追加した field:

- `skip_gate_budget_regime`
- `skip_gate_budget_remaining`
- `skip_gate_budget_exhausted`

## 設計上の判断

- budget は `skip_gate_primary_max_consecutive_skip` を置き換えない
  - budget: bucket 別制御
  - primary safety valve: global 緊急ブレーキ
- budget 統計は `raw skip` で数える
  - `bypass_mode=true` でも統計が残る
  - budget 枯渇で PASS 強制になっても、skip 意図そのものは統計に残る
- `scripts/` に runtime wiring を残しつつ、generic budget state は小さい独立 helper に分けた

## テスト

- 新規: [test_690_skip_budget.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/tests/unit/v460/test_690_skip_budget.py)
  - budget disabled
  - budget exhaustion
  - window rotation
  - regime×side independence
  - default fallback
  - config mutation/hot-reload 相当で ceiling 更新・counter 維持
  - primary safety valve coexist
  - FillRecord observability
  - bypass mode での budget statistics
- 更新:
  - [test_169_config_hot_reload.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/tests/unit/v460/test_169_config_hot_reload.py)
  - [test_346_fill_config_validation.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/tests/unit/v460/test_346_fill_config_validation.py)
  - [test_336_yaml_code_drift_prevention.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/tests/unit/v460/test_336_yaml_code_drift_prevention.py)
  - [test_fill_quality.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/tests/unit/v460/test_fill_quality.py)

## 今後の残り

1. `fill_quality` の judgment/report 側の残分割
2. heavy test setup の grouped sweep
3. PPO/SAC scheduler の shared safety helper 継続整理

---

## §2 Codex 5 件一括投入

689# (timeout_regime + trace_id) の Grade A- レビュー結果と、
P1 残タスク audit を踏まえ、以下 5 件のプロンプトを一括設計・投入。

| # | プロンプトファイル | 概要 | 根拠 | リスク |
|---|---|---|---|---|
| 1 | `690_codex_task_entry_gate_enable.md` | entry_gate 有効化 + EntryGateGuard 安全装置 | 555#/606# CalibrationMap 1,816+ records 蓄積済 | 中 |
| 2 | `690_codex_task_skip_audit.md` | _execute_skip 22 call site 形式的 audit + cancel_reason taxonomy | 689# フォローアップ | 低 |
| 3 | `690_codex_task_timeout_integration_test.md` | regime_timeout_overrides 4段 priority chain 統合テスト | 689# フォローアップ | 低 |
| 4 | `690_codex_task_offset_pipeline_simplify.md` | 9 段 offset pipeline のステージ disable フラグ追加 | 672# P2、§6 offset≈無効 | 低 |
| 5 | `690_codex_task_analysis_protocol.md` | 688# 層別分析を CLI プロトコルとして統合 | 分析再現性確保 | 低 |

### 2-1. Entry Gate 有効化 (555# / 606# 完結)
- CalibrationMap が EWMA τ=100 で十分データ蓄積
- Safety Guard 3 重 (consecutive block 上限 / session block rate / staleness guard)
- `entry_gate_enabled: false → true` へ YAML 変更
- FillRecord に `entry_gate_ev`, `entry_gate_blocked`, `entry_gate_guard_suppressed` 追加

### 2-2. _execute_skip Audit 形式化 (689# follow-up)
- `cancel_reason_taxonomy.py`: SkipCategory enum + CANCEL_REASON_REGISTRY
- AST 解析テスト: 全 call site の `update_last_side` 整合性を自動検証
- 新規 call site 追加時の regression 防止

### 2-3. Timeout Integration Test (689# follow-up)
- `get_timeout_with_reason()` の 4 段 priority chain テスト 15 件+
- 全 regime × side parametrize テスト
- ランタイムコード変更なし (テスト追加のみ)

### 2-4. Offset Pipeline 簡素化 (672# P2)
- 3 ステージの disable フラグ: `offset_ev_stage_enabled`, `offset_toxicity_stage_enabled`, `offset_vg_supplement_enabled`
- 672# §6 理論根拠: offset が PnL に有意な影響なし
- `_exec_stages` JSON 後方互換性維持
- pipeline_stats サマリーログ (100 cycles ごと)

### 2-5. Analysis Protocol CLI (688# 分析再利用化)
- `scripts/v460/analysis/protocols/` ディレクトリ + registry パターン
- Protocol688: 7 section (basic/nfq/as/spread/hour/sha/regime)
- `--protocol 688 --days 4` で 688# 分析を再現
- 既存 analysis スクリプト変更なし

---

## §3 ボット停止原因の調査結果

PID 9192 の「謎の停止」を調査した結果:
- **クラッシュではなく hot-swap restart** だった
- hot_swap_restart.ps1 が 07:53:09 に起動し、PID 9192 を graceful → 強制終了
- `taskkill /PID 9192` が失敗 (2 プロセス構成: PID 9192=Python311 launcher, PID 54256=venv worker)
- 30s 後に強制終了、新 PID=33608 (SHA abff47b79) で正常起動
- fill_test.log に Traceback/Exception なし、正常な cycle 20742 完了後に停止

---

## §4 残タスク (690# 時点)

### P0: 全完了 (20/20)

### P1 (投入済み/進行中)
- [x] skip budget (§1 Codex 実装済み)
- [x] entry_gate 有効化 (§2-1)
- [x] _execute_skip audit (§2-2)
- [x] timeout integration test (§2-3)
- [x] offset pipeline simplify (§2-4)
- [x] analysis protocol CLI (§2-5)
- [ ] AS 予測モデル再構築 (SkipGate 代替、大型タスク)
- [ ] OFI-Lite (605# T2-1、AS model 依存)

### P2
- [x] offset pipeline 簡素化 (§2-4 で対応)

### M (blocked)
- A-S 最適 spread / SAC action 幅 / eDRC / lib→ztb 統合 / walk-forward (全て blocked)
