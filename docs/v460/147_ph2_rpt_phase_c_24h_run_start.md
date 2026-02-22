# 147# Phase C 24h 連続 run 開始報告

**日時**: 2026-02-23  
**種別**: rpt (報告)  
**Phase**: ph2 (maker 執行可能性検証)  
**前セッション**: 146# (multi-exchange registry §11.4 fixes)

---

## §1 概要

134# で定義された Phase C「再計測」を開始。R-1 (短期施策) + R-2a (regime_weighting) を有効化した状態で 24h 連続 run を実行し、Gate 判定基盤データを収集する。

### Phase C の位置付け (134# §7 より)

```
Phase A: データインフラ復旧 (Day 0-1) ← ✅ 135# 完了
Phase B: 観測性強化 (Day 1-2)         ← ✅ 135# 完了
Phase C: 再計測 (Day 2-3)             ← 🔄 本ドキュメント (24h run 中)
Phase D: retrain 再始動 (Day 3-5)     ← ✅ 136# 先行実施済み
Phase E: P1 群着手 (Day 5+)           ← ✅ 137#-141# 先行実施済み (全 9 項目)
```

**注**: Phase D/E は Phase C に先行して実施済み。詳細は [144# §7](144_ph2_impl_regime_reprice_timeout.md#7-134-ロードマップ位置確認) 参照。

---

## §2 Phase C 開始状況

### 2.1 起動パラメータ

| 項目 | 値 |
|------|-----|
| 開始日時 | 2026-02-23 03:37:33 JST |
| PID | 108148 |
| 設定ファイル | `configs/v460/fill_test.yaml` |
| 実行時間 | 24h (`--hours 24`) |
| Git SHA | `7e8a3e981` (145# Phase C: R-2a config) |

### 2.2 R-2a 設定 (145# で追加)

```yaml
regime_weighting_enabled: true
regime_sample_weights:
  high_vol: 1.2
  trending: 0.8
  ranging: 1.0
  unknown: 0.5
regime_current_boost: 1.5
regime_current_lookback: 10
regime_weight_floor: 0.1
```

### 2.3 起動時警告

| 警告 | 内容 | 影響 |
|------|------|------|
| trades_health | UNHEALTHY: stale=83.1h (threshold=36.0h) | retrain 品質低下リスク |
| quarantine | 307 records excluded (blank git_sha) | ゾンビプロセス由来データ除外済 |

### 2.4 累積データ資産

| 日付 | 件数 |
|------|------|
| 2026-02-13 | 211 |
| 2026-02-14 | 220 |
| 2026-02-15 | 60 |
| 2026-02-16 | 21 |
| 2026-02-17 | 205 |
| 2026-02-18 | 277 |
| 2026-02-19 | 250 |
| 2026-02-20 | 217 |
| 2026-02-21 | 377 |
| 2026-02-22 | 252 |
| **合計** | **2,090** |

---

## §3 前回 fill_test 停止原因調査

### 3.1 事実

| 項目 | 値 |
|------|-----|
| 最終起動 | 2026-02-21 19:26:54 JST |
| 設定時間 | 168h (7日間) |
| 予定終了 | 2026-02-28 19:26 JST |
| 実際停止 | 2026-02-23 02:36:18 JST |
| 稼働時間 | 約 31h (予定の 18.5%) |

### 3.2 ログ調査結果

**明示的な停止ログなし**:

| 検索パターン | 結果 |
|--------------|------|
| `LOSS CAP REACHED` | ❌ 該当なし |
| `KillSwitch` | ❌ 該当なし |
| `Stopping fill test` | ❌ 該当なし |
| `completed` / `gracefully` | ❌ 該当なし |
| `Exception` / `Traceback` | ❌ 致命的エラーなし |

**最終ログ内容**:
```
2026-02-23 02:36:18 [skip_gate] SKIP: sell order skipped (score=-1.168)
2026-02-23 02:36:18 [fast_fill_defense] Reset on unfilled (sell)
```

→ 正常なサイクル処理中に突然停止

### 3.3 停止原因の推定

| 可能性 | 確度 | 根拠 |
|--------|------|------|
| **外部要因 (PC再起動/電源断/ターミナル終了)** | 高 | 明示的停止ログなし、正常処理中に停止 |
| hours 経過 | ❌ | 168h 設定に対し 31h で停止 |
| hard_loss_cap 到達 | ❌ | `LOSS CAP REACHED` ログなし |
| KillSwitch 発動 | ❌ | `preflight_skip_exceeded` 等のログなし |
| time_filter による停止 | ❌ | time_filter は sleep only、プロセス終了しない |

### 3.4 本番稼働への影響評価

**リスク**: 外部要因による予期せぬ停止は本番稼働で重大な問題となる。

| 問題 | 影響度 | 対策案 |
|------|--------|--------|
| プロセス死活監視なし | 高 | systemd / Windows Service 化 |
| 自動再起動なし | 高 | supervisor / restart policy |
| 停止通知なし | 中 | Slack/Discord webhook |
| 累積データ消失リスク | 低 | batch_flush で緩和済 (107#) |

### 3.5 推奨アクション (P2 優先度)

| ID | 施策 | 工数 | 備考 |
|----|------|------|------|
| P2-A | プロセス死活監視 cron | 0.2日 | `ps aux | grep fill_test` → Slack 通知 |
| P2-B | Windows Task Scheduler 自動再起動 | 0.3日 | 異常終了時のみ再起動 |
| P2-C | 起動/停止イベントログ永続化 | 0.1日 | `fill_test_events.jsonl` |

---

## §4 125# 「48h 観察」との関係

125# で言及された「S1 効果観察 (48h)」は **自動停止機能ではなく観察計画**:

> "S1 効果観察 (48h): fill_test は既に `skip_utc_hours_sell: [4, 8, 14, 15, 16]` で稼働中"

`--hours` パラメータで 48h を指定した事実はなく、今回の停止とは無関係。

---

## §5 Phase C 完了条件

| 条件 | 閾値 | 検証方法 |
|------|------|----------|
| 連続稼働時間 | ≥ 24h | ログ timestamp 差分 |
| fill_records 件数 | ≥ 200 (新規) | `wc -l fill_records_20260223.jsonl` |
| 異常終了なし | 0 | `KillSwitch` / `Exception` ログなし |
| Gate 判定実行 | 1回以上 | `run_gate_check.py` 実行ログ |

---

## §6 次ステップ

1. **Phase C 完了後 (24h 後)**:
   - Gate 判定 (`run_gate_check.py --run-id <phase_c_run_id>`)
   - Oracle 対比で alpha/beta 切分け初期分析

2. **Phase D (retrain 再始動)**:
   - trades パイプライン健全性確認
   - retrain 定期ループ再開

3. **死活監視導入** (P2-A/B/C):
   - 本番稼働前に実装必須

---

## §7 Codex レビュー依頼事項

本ドキュメントについて以下の観点でレビューを依頼:

### 7.1 停止原因調査の妥当性

- §3.2 のログ調査パターンは網羅的か
- §3.3 の推定ロジックに論理的欠陥はないか
- 他に考慮すべき停止原因はあるか

### 7.2 本番稼働リスク評価

- §3.4 の影響評価は適切か
- §3.5 の対策案で十分か、追加施策の提案

### 7.3 Phase C 設計

- §5 の完了条件は適切か
- R-2a 設定値 (§2.2) の妥当性

### 7.4 コードベース確認依頼

以下のコード箇所を確認し、停止原因の手がかりを探索:

```
scripts/v460/run_fill_test.py:
  - L1109-1600: run_continuous() メインループ
  - L1240: "Starting fill test" ログ出力箇所
  - KillSwitch 使用箇所全般

ztb/risk/circuit_breakers.py:
  - KillSwitch クラスの kill() 条件
```

---

## 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `docs/v460/147_ph2_rpt_phase_c_24h_run_start.md` | NEW: 本ドキュメント |

