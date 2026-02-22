# 149# Phase C 並行作業計画

**日時**: 2026-02-23  
**種別**: plan (計画)  
**Phase**: ph2 (maker 執行可能性検証)  
**前セッション**: 147# (Phase C 開始報告)

---

## §1 概要

Phase C (24h 連続 run) 実行中の並行作業として、P2/P3 残項目の実施可否を検討する。

### 1.1 147# 補足: Phase D/E 先行実施

147# §1 で「Phase C: 再計測 (Day 2-3)」を開始したが、実際には **Phase D/E は既に先行実施済み**:

```
Phase A (Data Infra)      : ✅ 135#
Phase B (Observability)   : ✅ 135#
Phase C (Re-measurement)  : 🔄 147# で開始 (24h run 中)
Phase D (Retrain restart) : ✅ 136# で先行実施済み
Phase E (P1 group)        : ✅ 137#-141# で先行実施済み (全 9 項目完了)
```

**参照**: [144# §7](144_ph2_impl_regime_reprice_timeout.md#7-134-ロードマップ位置確認)

### 1.2 R-1 / R-2 ステータス

| 施策群 | ステータス | 実装セッション |
|--------|----------|---------------|
| R-1a (offset regime adaptation) | ✅ 完了 | 143# |
| R-1b (lot regime adaptation) | ✅ 完了 | 143# |
| R-1c (reprice regime adaptation) | ✅ 完了 | 144# |
| R-1d (timeout regime adaptation) | ✅ 完了 | 144# |
| R-2a (regime_weighting_enabled) | ✅ config 有効化 | 145# |
| R-2b (regime sample weights) | ✅ config 設定済み | 145# |

---

## §2 P2/P3 残項目ステータス (146# §12.5 より)

| ID | 施策 | 判定 | 理由 | Phase C 中実施可否 |
|---|---|---|---|---|
| P2-01 | WalkForward → retrain | ⚠️ 保留 | SAC/PPO 用。LGBM アダプタ層が必要。工数 ~1日 | ❌ 工数大 |
| P2-02 | v459 統計 gate 常時化 | ❌ 不要 | 既に fill_quality.py 内で Holm-Bonferroni 使用中 | N/A |
| P2-03 | run_observation 同時運転 | ✅ 解決済 | P0-04 TradesRecorder で fill_test 内蔵化により二重系化 | N/A |
| P2-04 | oracle 日次 KPI 化 | ✅ **実装済** | 146# `daily_health_check.py` に統合 | ✅ タスク登録のみ |
| P2-06 | worst hour-side ルール | ⚠️ P1-04 統合 | regime×時間帯分析が先 | ❌ 分析待ち |
| P2-07 | execution trace 因果ログ | ⚠️ 保留 | FillRecord で部分対応済み。完全標準化は大改修 | ❌ 工数大 |
| P2-08 | shadow model A/B | ⚠️ P2 維持 | hot-reload アトミック性確保済み。工数大 | ❌ 工数大 |
| P3-01 | hft_proxies boardless fallback | ❌ 優先度低 | fill_test は tick 板データ直接保有 | N/A |
| P3-02 | advanced_regime_detector AB | ⚠️ P3 維持 | unknown レジーム削減に有望だが P0-09 で応急対応済み | ⚠️ 調査可 |
| P3-03 | dynamic_position_sizer | ⚠️ 保留 | 固定ロット設計との整合要検討 | ❌ 設計検討必要 |
| P3-04 | pnl_monte_carlo 日次実行 | ✅ **実装済** | 146# `daily_health_check.py` に統合 | ✅ タスク登録のみ |
| P3-05 | venue 横断比較 | ⚠️ P3 維持 | 146# multi-exchange で基盤は整備済み | ⚠️ 調査可 |

---

## §3 Phase C 期間中の推奨作業

### 3.1 即時実行可能 (工数 < 0.2日)

| 項目 | 内容 | 工数 | 効果 |
|------|------|------|------|
| **A1** | daily_health_check タスクスケジューラ登録 | 0.1日 | P2-04/P3-04 運用化 |
| **A2** | daily_health_check 初回実行 + レポート確認 | 0.05日 | 現状把握 |

### 3.2 調査・分析 (工数 0.2-0.5日)

| 項目 | 内容 | 工数 | 効果 |
|------|------|------|------|
| **B1** | P3-02 advanced_regime_detector 評価 | 0.3日 | unknown レジーム削減可否 |
| **B2** | P3-05 venue 横断比較データ収集 | 0.2日 | Coincheck vs 他取引所スプレッド |
| **B3** | 144# §8.1 CRITICAL 指摘対応設計 | 0.3日 | preflight-lot 整合性 |

### 3.3 非推奨 (Phase C 後に実施)

| 項目 | 理由 |
|------|------|
| P2-01 WalkForward | SAC/PPO 前提、工数大 |
| P2-07 因果ログ標準化 | 大改修、Phase C 完了後に検討 |
| P2-08 shadow model | hot-reload 安定後に検討 |
| P3-03 dynamic_position_sizer | 固定ロット設計との整合検討必要 |

---

## §4 daily_health_check 運用化計画

### 4.1 現在の実装状態 (146# §12.4)

```python
# scripts/v460/daily_health_check.py
# 統合チェック: trades_health + feature_freshness + gate_judgment + oracle_baseline
# Monte Carlo: pnl_monte_carlo 5000 シミュレーション
```

### 4.2 タスクスケジューラ登録

```powershell
# ops/windows/daily_health_check.ps1 (146# で作成済み)
# - 毎日 09:00 JST 実行
# - 7日以上古いレポート自動削除
# - 出力: reports/daily/YYYY-MM-DD.json
```

**登録コマンド**:
```powershell
$Action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -File C:\Users\Admin\dev\zaif-trade-bot\ops\windows\daily_health_check.ps1"
$Trigger = New-ScheduledTaskTrigger -Daily -At 09:00
Register-ScheduledTask -TaskName "ZTB-DailyHealthCheck" -Action $Action -Trigger $Trigger -Description "v460 daily health check"
```

### 4.3 期待出力

```json
{
  "timestamp": "2026-02-23T09:00:00+09:00",
  "trades_health": {"healthy": true, "stale_hours": 12.3},
  "feature_freshness": {"fresh": true, "latest_file": "20260223"},
  "gate_judgment": {"g1_1": "PASS", "g1_2": "WATCH", "latest_run": "FAIL"},
  "oracle_baseline": {"daily_pnl_bps": 2.3, "theoretical_max_bps": 15.2},
  "monte_carlo": {"mean_pnl": -0.5, "var_95": -12.3, "prob_positive": 0.42}
}
```

---

## §5 144# §8.1 CRITICAL 指摘対応

### 5.1 問題概要

| # | 重大度 | 問題 |
|---|---|---|
| 1 | CRITICAL | preflight 前に regime lot が反映されていない |
| 2 | HIGH | `_current_lot` の乗算的増加リスク |
| 3 | HIGH | 縮小レジームで preflight が過大ロット基準 |

### 5.2 対応方針

Phase C 完了後に対応予定。設計案:

```python
# 現状: preflight → regime_lot の順
# 改善案: regime_lot → preflight の順

# run_continuous() 内
effective_lot = self._regime_adjusted_lot(next_side)  # 先に算出
balance_ok = self._balance_checker.check(side=next_side, required_lot=effective_lot)
if not balance_ok:
    # skip or shrink
```

### 5.3 工数見積

| 作業 | 工数 |
|------|------|
| 設計レビュー | 0.1日 |
| 実装 | 0.3日 |
| テスト追加 | 0.2日 |
| **合計** | **0.6日** |

---

## §6 Codex レビュー依頼事項

### 6.1 Phase C 並行作業の妥当性

- §3 の作業優先度は適切か
- Phase C 実行中に避けるべき作業はあるか

### 6.2 daily_health_check 運用

- §4.2 のタスクスケジューラ設定は適切か
- 追加すべきヘルスチェック項目はあるか

### 6.3 144# CRITICAL 対応

- §5.2 の設計方針は妥当か
- 他に考慮すべき点はあるか

### 6.4 コードベース確認依頼

```
scripts/v460/daily_health_check.py:
  - 全体構造とエラーハンドリング
  - Monte Carlo の n=10 下限の妥当性

ztb/features/advanced_regime_detector.py:
  - P3-02 評価のための現状確認
  - unknown 判定ロジック

scripts/v460/run_fill_test.py:L281, L809, L1298:
  - 144# §8.1 指摘箇所の詳細確認
```

---

## §7 次ステップ

1. **即時**: A1 (タスクスケジューラ登録) + A2 (初回実行)
2. **Phase C 中**: B1-B3 の調査・分析
3. **Phase C 後**: 144# §8.1 CRITICAL 対応

---

## 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `docs/v460/149_ph2_plan_phase_c_parallel_work.md` | NEW: 本ドキュメント |

---

## §8 148# P2 実装ログ

### 8.1 P2-A: fill_test 死活監視 (watchdog)

**実装**: `ops/windows/fill_test_watchdog.ps1`

機能:
- プロセス存在確認 (`Get-Process` で `run_fill_test` を検索)
- lock heartbeat 鮮度確認 (300s stale 閾値)
- 最新 fill_record タイムスタンプ確認
- `fill_test_events.jsonl` へのアラートイベント記録
- Discord webhook 通知 (オプション)

**タスクスケジューラ登録**:

```powershell
# 5分間隔で実行
$Action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -File C:\Users\Admin\dev\zaif-trade-bot\ops\windows\fill_test_watchdog.ps1"
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5)
Register-ScheduledTask -TaskName "ZTB-FillTestWatchdog" -Action $Action -Trigger $Trigger -Description "147# P2-A: fill_test watchdog"
```

**手動実行**:
```powershell
cd C:\Users\Admin\dev\zaif-trade-bot
.\ops\windows\fill_test_watchdog.ps1
# Discord 通知付き
.\ops\windows\fill_test_watchdog.ps1 -Notify
```

### 8.2 daily_health_check 初回実行結果

**実行日時**: 2026-02-23 04:27:59 JST

| チェック | 結果 | 詳細 |
|---------|------|------|
| trades_health | ⚠️ UNHEALTHY | missing_days=['20260221', '20260220'] |
| feature_freshness | ✅ OK | - |
| gate_judgment | ❌ ERROR | API 引数不整合 (既知) |
| oracle_baseline | ✅ done | mean=0.000 bps |

**Oracle レポートサマリ**:

| セグメント | n | 実績 PnL30s | Oracle PnL30s | 月間推定 (JPY) |
|-----------|---|-------------|---------------|----------------|
| all | 1182 | -0.24 bps | +2.64 bps | -7,622 / +40,133 |
| buy | 591 | -0.00 bps | +2.82 bps | -104 / +45,428 |
| sell | 591 | -0.47 bps | +2.44 bps | -15,141 / +34,838 |

**所見**:
- sell 側の実績 PnL が大きくマイナス (逆選別リスク継続)
- Oracle 上限と実績の乖離が大きく、SkipGate 品質向上余地あり
- trades_health の missing_days は TradesRecorder 収集問題の可能性

### 8.3 watchdog 動作確認

```
PS> .\ops\windows\fill_test_watchdog.ps1
08-02-23 04:31:17 RUNNING [watchdog] fill_test 稼働中: PID=108148, uptime=0.00:53:44 | heartbeat STALE (703s ago)
```

- プロセス検出: ✅ PID=108148 を正しく検出
- heartbeat: STALE (古いコードで動作中のため 60s 周期更新なし)
- 次回 run 以降は heartbeat が 60s 周期で更新されるため正常動作予定

### 8.4 P2 ステータスまとめ

| ID | 施策 | 状態 |
|----|------|------|
| P2-A | プロセス死活監視 | ✅ 実装完了 `fill_test_watchdog.ps1` |
| P2-B | Task Scheduler 自動再起動 | ❌ 未着手 (要設計検討) |
---

## 変更履歴

| 日時 | 変更内容 |
|------|----------|
| 2026-02-23 04:00 | 初版作成 |
| 2026-02-23 04:31 | §8 追加: 148# P2 実装ログ (watchdog, daily_health_check結果) |

## 変更ファイル (更新)

| ファイル | 変更内容 |
|----------|----------|
| `docs/v460/149_ph2_plan_phase_c_parallel_work.md` | §8 追加: P2 実装ログ |
| `docs/v460/147_ph2_rpt_phase_c_24h_run_start.md` | P2-A ステータス更新 |
| `docs/v460/148_ph2_rev_147_phase_c_stop_cause_and_side_issues.md` | P2 追加 |
| `ops/windows/fill_test_watchdog.ps1` | NEW: 死活監視スクリプト |