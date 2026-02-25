# 166# レビュー対応 + 162/163 残課題消化

## 概要

165# 末尾 7 のレビュー指摘 (P02, P12) への対応と、162#/163# 残課題の消化を実施。

## 1 レビュー指摘事項 (165# 7.5)

| ID | 優先度 | 指摘内容 | 対応状況 |
|----|--------|----------|----------|
| R-1 | P0 | 再現性固定 (run_id/period/files を1セット) |  stopgap_health.py に pply_filters() 追加、CLI 引数対応 |
| R-2 | P0 | 163出口判定の運用化 (閾値逸脱自動アラート) |  generate_alerts() + AlertItem (critical/warning/info) |
| R-3 | P1 | AS-R1 閾値校正 (8.0bps, 機会費用測定) |  velocity log 蓄積待ち (fill_test SHA 955a7818842a で2件skip確認) |
| R-4 | P1 | model_used 経路別レポート (side_sell/side_buy/unified 分離) |  section_model_used() + compute_model_used_metrics() |

## 2 実施内容

### 2.1 stopgap_health.py (584  762 lines)

**追加機能:**
- pply_filters(records, *, run_id, git_sha, date_from, date_to)  analyze_fill_logs.py と同一ロジック
- ModelUsedMetrics dataclass  経路別 AS率/PnL メトリクス
- AlertItem dataclass  severity (critical/warning/info), stopgap_id, message
- compute_model_used_metrics()  model_used 経路別の AS率/PnL 算出
- generate_alerts()  退出基準閾値逸脱の自動アラート生成
- generate_health_report() に ilters_applied 引数追加
- print_health_summary() に Model Used 表 + Alerts セクション追加

### 2.2 stopgap_daily_report.py (95  115 lines)

**追加:**
- --run-id, --git-sha, --date-from, --date-to CLI 引数
- pply_filters() 呼び出し + フィルタ後空チェック

`
# 使用例: 再現性固定分析
.venv\Scripts\python.exe scripts/v460/analysis/stopgap_daily_report.py \
  --git-sha 955a78 --date-from 2026-02-25 --date-to 2026-02-28
`

### 2.3 nalyze_fill_logs.py (675  709 lines)

**追加:**
- section_model_used()  model_used (skip_gate_model_used) 経路別分析セクション
  - 各経路: N (約定数), AS# (AS 件数), AS% (AS 率), PnL30 (平均PnL30s), AS_Loss (AS平均損失)
  - 対象経路: 
one, primary:side_buy, primary:side_sell, primary:unified 等

### 2.4 テスト

| ファイル | 追加テスト数 | 合計 |
|----------|-------------|------|
| 	est_stopgap_health.py | +20 (6 apply_filters + 5 model_used + 4 alerts + 4 report + 1 intent) | 52 |
| 	est_analyze_fill_logs.py | +3 (section_model_used) | 21 |
| **合計** | +23 | 73 passed |

## 3 162/163 残課題ステータス

### 162# 4 改善提案

| ID | 内容 | 状態 |
|----|------|------|
| P0-A1 | SkipGate AS 検知改善 |  AS-R1 velocity rule (165#) |
| P0-A2 | balance_forced_skip 根因 |  IS enabled=true (5a5b9ba42) |
| P0-A3 | sell guard 緩和 |  offset_floor 0.100.20 (164#) |
| P1-B1 | 時間帯フィルタ |  107# dynamic gating |
| P1-B2 | null regime fallback |  unknown regime skip |
| P1-B3 | Retrain data 蓄積 |  quality gate モデル拒否中 |
| P1-B4 | git_sha 8ba101953 diff |  新SHA で supersede |
| P2-C1 | orderbook_error sell_guard 閾値 |  未着手 |
| P2-C2 | Reprice logic tuning |  未着手 |
| P2-C3 | stale_skip_gate_blocked 閾値 |  未着手 |

### 162# 7.3 レビュー提案

| 提案 | 状態 |
|------|------|
| --run-id/--git-sha/--date CLI |  完了 (analyze + stopgap 両方) |
| Stopgap exit 表 |  163# + 165# stopgap_health.py |
| 160# 判定ロジック統合 |  ab_judgment.py + stopgap_health.py |
| IS staging plan |  163# 文書化済 (S0S1 完了, S2 待機) |

### 163# ロードマップ

| 項目 | 状態 |
|------|------|
| IS enabled=true |  S1 (5a5b9ba42) |
| Dynamic gating |  regime_adaptive_enabled=true |
| Sell offset 動的最適化 | SO-1 , SO-2/SO-3  |
| AS 根因 + モデル改善 |  AS-R1 (165#) + model_used (本166#) |
| sell_guard 閾値動的化 |  P2 未着手 |

## 4 165# 受入条件チェック (7.6)

| 条件 | 状態 | 備考 |
|------|------|------|
| 同一母集団の再現検証 |  | apply_filters で固定可能 |
| model_used 経路別 AS 源泉説明 |  | section_model_used + compute_model_used_metrics |
| Lock-free 連続 run log で AS-R1 効果/副作用評価 |  | SHA 955a7818842a 稼働中 (2 velocity skip/29 cycles) |
| 162/163 stopgap exit 判断と整合 |  | generate_alerts で自動化 |

## 5 現在のシステム状態

- **Fill Test**: PIDs 139024/139880 (SHA 955a7818842a), 安定稼働中
- **Retrain**: PIDs 120328/147464
- **Velocity Skip**: 29 cycles 中 2 件 skip (7.3%)
- **テスト**: 73 passed (関連テストのみ), PPO integration test は無関係の既知障害
- **Git HEAD**: 955a78188

## 6 次ステップ

1. **R-3 AS-R1 閾値校正**: velocity log が 100+ records 蓄積次第、price_velocity_60s 分布 vs AS outcome を分析
2. **P2-C1/C2/C3**: orderbook_error 閾値、reprice tuning、stale_skip_gate 閾値  IS S2 移行後に着手
3. **IS S1S2 移行判定**: nalyze_fill_logs.py --git-sha 5a5b9ba --date-from 2026-02-26 で評価
