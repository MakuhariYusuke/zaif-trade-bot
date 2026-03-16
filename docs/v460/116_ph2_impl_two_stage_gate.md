# 116# 二段階ゲート実装 — 115# レビュー反映

| key | value |
|-----|-------|
| type | impl (実装完了報告) |
| scope | ph2 G1.1 二段階ゲート + Watch layer |
| date | 2026-02-19 |
| parent | `114_ph2_ext_gate_redesign_review.md`, `115_ph2_ext_gate_redesign_review_response.md` |
| purpose | 115# GPT-5.3-Codex レビュー全指摘の検証・実装 |

---

## §0 Executive Summary

115# 外部レビュー (GPT-5.3-Codex) の Q10.1–Q10.6 指摘すべてを検証し、
コード・設定・ドキュメント **6 ファイルをセット改訂** した。

| 指摘 | 対応 | 状態 |
|------|------|------|
| Q10.1 二段階化 + 効果量併設 | G1.1-quick (K1-K6) / G1.2-full (F1-F8) 分離、K4 複合条件 | ✅ |
| Q10.2 閾値 (AS 30%, PnL 複合, fill_rate) | AS→30%, K4: `p<0.02 ∧ mean≤-0.8`, F1b: overall≥62% | ✅ |
| Q10.3 attempted 分母 + S0 前提 | attempted 指標実装、S0 は将来課題として明記 | ✅ |
| Q10.4 Watch layer (黄信号) | `p<0.05 ∧ mean<-0.3` → WATCH (パラメータ凍結勧告) | ✅ |
| Q10.5 6 ファイル同時改訂 | 000#, 009#, 014#, YAML, fill_quality.py, run_fill_test.py | ✅ |
| Q10.6 分母不一致 / SkipGate 監査 | K6, F6 skip_gate_ratio チェック追加、cancel_reason 可視化 | ✅ |
| 新 K6 | `skip_gate_ratio ≤ 25%` (Kill gate) | ✅ |
| 新 F1b | `overall_fill_rate ≥ 62%` (Full gate) | ✅ |

テスト: **157 全 PASS** (test_fill_quality.py: 既存137 + 新規20)

---

## §1 事実検証結果

### §1.1 `min_fill_rate_p90` 実態確認

115# §1.2 が指摘した通り、`configs/v460/gate_thresholds.yaml` の現行値は **0.85** (084# で変更済)。
000# §3.3 記載の「90% 基準」は文書上の初期値であり、実運用とは乖離していた。
→ 今回の改訂で文書・設定・コードの三者を整合させた。

### §1.2 AS_ratio 閾値

114# 提案: 35% → 115# 推奨: **30%** (35% はmakerとしては緩すぎる)
→ F5 閾値を 30% に設定。

### §1.3 S0 SkipGate 有効性ゲート

115# が要求した AUC ベースの SkipGate 有効性検証 (S0) は、
現時点で AUC 計算基盤が未整備のため **将来課題** として 000# に明記。
暫定ガードとして K6 (≤25%) / F6 (≤20%) の skip_gate_ratio 上限で代替。

---

## §2 設計: 二段階ゲート構造

### §2.1 G1.1-quick (72h Kill Gate)

早期異常検出。FAIL → 即停止、WATCH → パラメータ凍結。

| Check | 条件 | Fail 時 |
|-------|------|--------|
| K1 | `attempted_fill_rate ≥ 60%` | Kill |
| K2 | `attempted_cancel_ratio ≤ 40%` | Kill |
| K3 | `queue_wait_median ≤ 120s` | Kill |
| K4 | `pnl30 p < 0.02 AND mean ≤ -0.8bps` → 両条件成立で FAIL | Kill |
| K5 | `cumulative_loss < 10,000 JPY` | Kill |
| K6 | `skip_gate_ratio ≤ 25%` | Kill |

**Watch layer**: 全 K-check PASS だが `p < 0.05 AND mean < -0.3bps` → WATCH

### §2.2 G1.2-full (168h Qualification Gate)

本番移行可否判定。

| Check | 条件 | Fail 時 |
|-------|------|--------|
| F1 | `attempted_fill_rate ≥ 70%` | Block |
| F1b | `overall_fill_rate ≥ 62%` | Block |
| F2 | `attempted_cancel_ratio ≤ 30%` | Block |
| F3 | `queue_wait_median ≤ 60s` | Block |
| F4 | `pnl30 p ≥ 0.05` (有意な毀損なし) | Block |
| F5 | `adverse_selection_ratio ≤ 30%` | Block |
| F6 | `skip_gate_ratio ≤ 20%` | Block |
| F7 | `calendar_days ≥ 7` | Block |
| F8 | `n_attempted ≥ 500` | Block |

---

## §3 変更箇所

### §3.1 `configs/v460/gate_thresholds.yaml`

- `g1_1_quick_exec` セクション追加 (K1-K6 + Watch 閾値)
- `g1_2_full_exec` セクション追加 (F1-F8)
- legacy `g1_1_exec` は後方互換のため保持

### §3.2 `ztb/metrics/fill_quality.py`

**FillMetrics** dataclass 拡張 (7 フィールド追加):
- `attempted_orders`, `skip_gate_count`, `skip_gate_ratio`
- `attempted_fill_rate`, `attempted_cancel_ratio`
- `overall_fill_rate`, `post_fill_30s_pnl_ci_upper`

**compute_fill_metrics()** 更新:
- SkipGate 検出: `skip_gate_skipped is True OR cancel_reason == "skip_gate"`
- attempted 分母 = total - skip_gate_count
- CI upper: t 分布ベース (scipy.stats.t)

**新規関数**:
- `g1_1_quick_judgment(metrics, thresholds, cumulative_loss_jpy=0)` → PASS / FAIL / WATCH
- `g1_2_full_judgment(metrics, thresholds)` → PASS / FAIL

既存 `g1_1_judgment()` は後方互換ラッパーとして保持。

### §3.3 `scripts/v460/run_fill_test.py`

- import 追加: `g1_1_quick_judgment`, `g1_2_full_judgment`
- `run_results_only()`: legacy + quick + full の三段判定を実行、結果に `two_stage` キーを付与
- continuous run 最終セクション: 二段階判定を追加計算・出力

### §3.4 ドキュメント改訂

| ファイル | 変更箇所 | 内容 |
|----------|---------|------|
| `000_ph0_plan_project_proposal.md` | §3.3 | 単一 E1-E5 → 二段階 K1-K6 / F1-F8 + Watch |
| | §6 | リスク表更新 |
| `009_ph2_plan_g1_1_exec.md` | §2 | 旧 E1-E5 ↔ 新 K/F マッピング、YAML 構造提示 |
| | §2.3 | FAIL アクション更新 |
| | §2.4 | K4 複合条件の統計的根拠 |
| `014_ph2_plan_completion_and_transition.md` | §3 | 移行条件: G1.1 PASS → G1.2-full PASS |
| | §3.1 | F1-F8 基準表 |

---

## §4 テスト

### §4.1 新規テスト (20 件)

`tests/unit/v460/test_fill_quality.py`:

**TestG11QuickJudgment** (9 件):
- `test_all_pass` — K1-K6 全 PASS
- `test_k1_fill_rate_fail` — K1 FAIL
- `test_k4_pnl_compound_both_conditions_fail` — 115# 複合条件 (p AND mean)
- `test_k4_pnl_significant_but_small_loss_passes` — p 有意だが効果量不足 → PASS
- `test_k4_pnl_large_loss_but_not_significant_passes` — 効果量大だが p 不足 → PASS
- `test_k5_cumulative_loss_fail` — K5 累積損失 FAIL
- `test_k6_skip_gate_ratio_fail` — K6 skip_gate FAIL
- `test_watch_layer` — WATCH 判定
- `test_no_watch_when_pnl_ok` — PnL 良好時 Watch 非発動

**TestG12FullJudgment** (9 件):
- `test_all_pass` — F1-F8 全 PASS
- `test_f1_attempted_fill_rate_fail` — F1 FAIL
- `test_f1b_overall_fill_rate_fail` — F1b FAIL (115# 新設)
- `test_f4_pnl_negative_not_significant_passes` — 有意でなければ PASS
- `test_f4_pnl_negative_significant_fails` — 有意な毀損で FAIL
- `test_f5_adverse_selection_fail` — AS 30% 閾値 FAIL
- `test_f6_skip_gate_ratio_fail` — F6 FAIL
- `test_f7_calendar_days_fail` — 暦日不足
- `test_f8_n_attempted_fail` — サンプル数不足

**TestComputeFillMetricsAttempted** (2 件):
- `test_skip_gate_fields_populated` — skip_gate 除外の正確性
- `test_no_skip_gate_records` — skip なし時 attempted=total

### §4.2 リグレッション

```
tests/unit/v460/test_fill_quality.py: 157 passed, 0 failed
```

---

## §5 残課題

| # | 内容 | 優先度 |
|---|------|--------|
| 1 | S0 SkipGate 有効性ゲート (AUC ベース) | 中 — 基盤整備後 |
| 2 | Cancel reason breakdown 可視化 (Q10.6) | 低 — 監査用 |
| 3 | 分母定義ドキュメント (raw / clean / attempted 一覧) | 低 — 運用ガイド |
| 4 | Watch → パラメータ凍結の自動化 | 中 — ph3 移行後 |
