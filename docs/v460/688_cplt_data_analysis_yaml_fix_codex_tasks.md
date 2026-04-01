# 688# データ分析 + YAML 緊急修正 + Bot 再起動 + Codex タスク設計

| 項目 | 内容 |
|------|------|
| 日付 | 2026-04-02 |
| SHA | 2d12433a9 |
| 対象データ | `fill_records_20260401.jsonl` (n=443, fills=111, 4/1 09:00~4/2 05:53 JST) |

## §1 688# データ分析サマリ

### 全体指標 (vs 686# ベースライン 3/29~4/1)

| 指標 | ベースライン | 今回 | Δ |
|------|------------|------|---|
| BUY 30s PnL | -0.02 | **+0.57** | +0.59 |
| SELL 30s PnL | -0.94 | **-1.46** | -0.52 |
| AS率 全体 | 25.0% | **21.6%** | -3.4pp |
| BUY AS | — | 12.7% | — |
| SELL AS | — | 30.4% | — |
| Fill rate | — | 111/443 = 25.1% | — |

### Cancelベースライン

| Cancel Reason | 件数 | 割合 |
|--------------|------|------|
| preflight_insufficient | 156 | 46.9% |
| spread_too_narrow | 65 | 19.6% |
| skip_gate | 60 | 18.1% |
| timeout | 17 | 5.1% |
| sell_dynamic_kill | 10 | 3.0% |
| その他 | 24 | 7.2% |

### §1.1 SELL 壊滅帯: JST 11h / 13h

| 時間(JST) | UTC | boost値 | sell fills | AS率 | avgPnL |
|-----------|-----|---------|-----------|------|--------|
| 11h | UTC2 | 2.5 | 6 | 50% | -5.19 |
| 13h | UTC4 | 2.5 | 3 | **100%** | -8.85 |

**結論**: boost=2.5 では不十分。fill が来る時点で AS fill（急騰中に約定するサバイバーシップバイアス）。

### §1.2 BUY 側の改善
- BUY avg PnL: +0.57 (ベースライン -0.02 から大幅改善)
- BUY AS率: 12.7% (全体 21.6% を大きく下回る)
- BUY 60s PnL: +2.19 (良好なリバーサル)

### §1.3 未稼働機能 (コード変更はhot-reload不可)
- `skip_gate_bypassed=0` → 306664e32 のコード変更は再起動必要
- `trend_5s_sell_guard=0` → 同様に再起動必要
- state separation (72bd8e713) → 再起動必要

## §2 YAML 緊急修正 (commit: 2d12433a9)

| 変更 | 旧値 | 新値 | 根拠 |
|------|------|------|------|
| sell_hour_offset_boost UTC2 | 2.5 | **5.0** | JST11h AS50% PnL=-5.19, soft-kill レベル |
| sell_hour_offset_boost UTC4 | 2.5 | **5.0** | JST13h AS100% PnL=-8.85, soft-kill レベル |
| hour_ceiling_mult UTC2 | 2.0 | **3.0** | boost=5.0 対応で ceiling 拡大 |
| hour_ceiling_mult UTC4 | 2.5 | **3.5** | boost=5.0 対応で ceiling 拡大 |
| offset_ceiling_ratio_sell | 0.40 | **0.50** | 684# A1, sell AS30.4% 全時間帯防御 |

## §3 Bot 再起動 (SHA: 2d12433a9)

- 旧 PID: 39004 (4/1 09:21 JST～)
- 新 PID: 9192 (4/2 06:18 JST～)
- retrain_scheduler: PID 54664 (再起動済)
- 反映される機能:
  - 688# YAML 変更 (§2)
  - 686# SG bypass_mode (306664e32)
  - 687# state separation (72bd8e713)
  - 687# metrics split (bd0fecc40)
  - 685# trend_5s_sell_guard

## §4 Codex 成果レビュー

### 72bd8e713: state separation (Grade: B+)
- `last_executed_side` (実約定のみ) と `last_attempted_side` (全試行) の分離
- テスト: 135 行の新規テスト (test_687_state_separation.py)
- 全 110 テスト PASS
- 注意点: resume fallback (last_attempted_side=None → side フォールバック) は実質安全と判断
  - cancelled 時の side = 注文は試行されたので "attempted" として妥当
  - filled=False 時の executed_side → None にフォールバック (安全)

### bd0fecc40: metrics split
- fill_metrics_core.py を新設、fill_quality.py から scan ヘルパーを抽出
- perf_runner を tests/conftest.py に昇格
- skip_gate_bypassed observability 拡張

## §5 新 Codex タスク

| # | タスク | 複雑度 | プロンプト |
|---|--------|--------|-----------|
| C1 | Timeout regime×side 別短縮 | S | `688_codex_task_timeout_regime_side.md` |
| C2 | _execute_skip 監査 + Decision Trace ID | S | `688_codex_task_skip_audit_trace_id.md` |

### 残 Codex バックログ
- 638# P1: bucket 別 skip budget (M) — regime bucketing 基盤が必要
- 672# P1: AS 予測モデル再構築 (L) — 長期課題
- 687# torch test isolation — Codex 投入済み、結果待ち
