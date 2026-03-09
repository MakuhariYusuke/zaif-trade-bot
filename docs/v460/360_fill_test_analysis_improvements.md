# 360# Fill Test ログ分析 & 改善提案

| 項目 | 値 |
|------|-----|
| 対象期間 | 2026-03-05 ～ 2026-03-10 (5日間) |
| 分析時点 | 2026-03-10 01:00 JST |
| データ源 | `results/v460/fill_test/fill_records_20260305-09.jsonl`, `fill_test_events.jsonl`, `fill_test.log` |
| Git HEAD | `79409e8a5` (359# P3A-2 + self-review) |
| Bot SHA | 最終デプロイ `d4db8277e0` → `819ec73b2081` |

---

## §1 集計概要 (03-05 ～ 03-09)

| 指標 | 値 |
|------|-----|
| 総レコード | 3,885 |
| 約定 (filled) | 865 (22.3%) |
| キャンセル | 3,020 (77.7%) |
| サイクル数 | 9,090 |
| 累積 PnL | **−717.92 JPY** |

### キャンセル理由 TOP-7

| 理由 | 件数 | 割合 | 備考 |
|------|------|------|------|
| `buy_dynamic_kill` | 506 | 16.8% | buy 側 kill |
| `skip_gate` | 430 | 14.2% | ML スキップ判定 |
| `sell_dynamic_kill` | 425 | 14.1% | sell 側 kill (§3 詳述) |
| `forced_buy_delay` | 389 | 12.9% | 348# 撤廃済？要確認 |
| `per_side_dd_halt` | 340 | 11.3% | 日次 DD per-side halt |
| `spread_too_narrow` | 245 | 8.1% | スプレッド狭小 |
| `stale_adverse_drift` | 189 | 6.3% | 価格ドリフトによるキャンセル |

### side 別パフォーマンス

| 指標 | Buy | Sell |
|------|-----|------|
| レコード | 2,196 | 1,568 |
| 約定 | 433 (19.7%) | 432 (27.6%) |
| PnL30 平均 | −0.18 bps | −0.08 bps |
| EV-PnL 平均 | +0.02 bps | −0.02 bps |
| Queue Wait median | 11.2s (全体) | - |
| Queue Wait P90 | 38.7s (全体) | - |

---

## §2 安定性問題: サイレントクラッシュ

### 2.1 概要

3月の watchdog NOT_RUNNING アラート: **13件** (13回の自動再起動)。
lock_conflict (dual-spawn): **12件**。

### 2.2 クラッシュパターン

| 特徴 | 詳細 |
|------|------|
| エラーログ | **なし** — プロセスが静かに消滅 |
| 最後のログ | 通常のサイクル実行中 (PnL 待機 sleep 中など) |
| stderr | 空 — 例外未捕捉型のクラッシュではない |
| 発生間隔 | 不定 (0.1h ～ 24h) |

**具体例 (03-09 17:59)**:
```
L28399: 17:59:04 INFO [pnl_measurer] Waiting 16.6s for PnL measurement...
--- (ここでプロセス消滅) ---
L28400: 17:59:58 INFO [fill_test_cli] Log file: ... (新プロセス)
```

**具体例 (03-09 09:48)**:
```
L25930: 09:48:07 DUAL KILL quiescence: both buy/sell killed — resting
--- (ここでプロセス消滅) ---
L25931: 09:49:22 fill_test_cli Log file: ... (新プロセス)
```

### 2.3 推定原因

| 仮説 | 確度 | 根拠 |
|------|------|------|
| **OOM (メモリ不足)** | 高 | stderr 空、ログなし終了 = OS kill。RSS 制限 2500MB。retrain_scheduler が同一環境でメモリ使用。 |
| スレッド内未捕捉例外 | 中 | heartbeat/health_monitor スレッドの例外でメインスレッドが気付かず? |
| Windows タスクスケジューラ干渉 | 低 | 外部的な kill |

### 2.4 改善案

1. **[OPS-1] atexit / signal handler でクラッシュ直前のメモリ使用量をログ** — `psutil.Process().memory_info().rss` を shutdown hook に追加
2. **[OPS-2] health_monitor の RSS チェック間隔を短縮** — 現在 300s (5分) → 60s に変更。`rss_critical_mb` 超過時に即座にログ + state 保存
3. **[OPS-3] Windows イベントログ連携** — `Application` ログの "python.exe" terminated エントリを watchdog で確認

---

## §3 sell_dynamic_kill (SDK) 影響分析

### 3.1 設定値

| パラメータ | 値 | 意味 |
|-----------|-----|------|
| `enabled` | **true** | 有効 |
| `window` | 50 | 直近 50 fill の rolling |
| `threshold_bps` | −0.3 | base kill 閾値 |
| `ewma_alpha` | 0.05 | EWMA モード (effective window ≈ 20) |
| `ewma_time_decay_tau_sec` | 600 | kill 中の EWMA 減衰 (半減期 ≈ 7分) |
| `max_kill_duration_sec` | 1800 | 30分で自動解除 |
| `inv_relaxation.enabled` | true | 在庫連動緩和 |
| `inv_relaxation.max_bps` | 0.5 | effective range: −0.3 ～ −0.8 |
| `regime_thresholds` | trending_up=−0.3, trending_down=−1.0, ranging=−0.5 | |

### 3.2 日次推移

| 日付 | SDK 件数 | 全キャンセル | 約定率 | 備考 |
|------|---------|-------------|--------|------|
| 03-05 | 0 | 606 | 12.3% | SDK 未発動 |
| 03-06 | 0 | 399 | 26.0% | SDK 未発動 |
| 03-07 | 0 | 506 | 20.6% | SDK 未発動 |
| 03-08 | **220** | 520 | 21.9% | 突然発動 |
| 03-09 | **135** | 297 | 30.4% | 継続 |

### 3.3 なぜ 03-08 から突然発動?

SDK は `sell_dynamic_kill_enabled: true` (YAML §598)。
03-07 まで kill が発動しなかったのは、sell EWMA が閾値 (−0.3 bps) を上回っていたため。
03-08 時点で sell の rolling PnL が閾値以下に低下 → kill 発動 → sell 全停止。

**同一 SHA (eb24cf4a74)** で 03-07: SDK=0, 03-08: SDK=42 → **コード変更ではなく PnL 状態変化が原因**。

### 3.4 現在の kill 状態 (state.json)

- pnl_history: 144 件
- ewma_value: **+0.67** (正値 → kill 解除済み)
- kill_activated_at: None (現在 kill なし)
- total_kills: **41** (累計)
- recent 50 mean: **+1.12 bps** (回復済み)

### 3.5 SDK の KPI 影響

sell 側の 425 件 SDK キャンセルは、sell キャンセル全体の **37.4%** を占める。
SDK を除外した場合の attempted fill rate: 22.3% → **34.3%**。

---

## §4 Watchdog Dual-Spawn (lock_conflict)

### 4.1 メカニズム

```
[Task Scheduler] → watchdog.ps1 → Start-Process (PID A)
                                 ↓ (restart.lock 解放)
[Task Scheduler] → watchdog.ps1 → Start-Process (PID B)  ← 5分後
                                   PID A が fill_test.lock 保持中
                                   PID B → lock_conflict → 即 exit
```

しかし場合によっては watchdog が 2 連続発動し、2つの Python プロセスが**同時に**起動:
- 一方が lock 取得成功
- もう一方が `lock_conflict` で即停止

### 4.2 実例 (03-09 00:10)

```
00:10:10 [event] stop: reason=lock_conflict  (Process A 敗北)
00:10:12 Starting fill test: 168.0h          (Process B 勝利)
00:10:57 [event] stop: reason=lock_conflict  (Process C 敗北)
00:12:04 Starting fill test: 168.0h          (Process D 勝利)
```

### 4.3 原因

- Task Scheduler が「前回の実行が完了していなくても新規実行を許可」設定になっている可能性
- `restart.lock` は watchdog スクリプト終了時に即解放されるため、次の invocation までの保護がない
- Python プロセスの起動～lock 取得に数秒かかる間に次の watchdog が同じ判定を下す

### 4.4 改善案

1. **[OPS-4] restart.lock の保持期間を延長** — watchdog 終了後も 60s 保持 (lockfile に PID + expiry ts 記録)
2. **[OPS-5] Task Scheduler の「既存実行中は新規実行しない」設定を確認・有効化**
3. **[OPS-6] fill_test.lock 取得成功まで watchdog が待機** — Start-Process 後 10s sleep して lock 存在確認

---

## §5 G1.1 ゲート判定 (暫定)

| 基準 | 閾値 | 現状 | 判定 |
|------|------|------|------|
| K1 attempted_fill_rate ≥ 60% | ≥ 60% | 25.0% (excl skip_gate) | **FAIL** |
| K1 (excl SDK) | ≥ 60% | 34.3% | **FAIL** |
| K3 queue_wait_median ≤ 120s | ≤ 120s | 11.2s | **PASS** |
| K2 (72h kill-switch なし) | 72h 連続 | クラッシュにより中断多数 | **要再評価** |

**K1 FAIL の主因**:
1. `buy_dynamic_kill` (16.8%) + `sell_dynamic_kill` (14.1%) = 30.9% のレコードが kill
2. `per_side_dd_halt` (11.3%) — 日次 drawdown halt
3. `forced_buy_delay` (12.9%) — 348# で撤廃済みだが残存?
4. `spread_too_narrow` (8.1%) — 市場構造要因

**SDK + BDK を除外しても 34.3%** → kill 以外の要因 (halt, delay, spread) も大きい。

---

## §6 改善提案まとめ

### 即時対応可能 (コード変更不要)

| ID | 内容 | 期待効果 |
|----|------|---------|
| OPS-5 | Task Scheduler dual-run 防止設定 | lock_conflict 解消 |

### 短期コード改善

| ID | 内容 | 影響範囲 | 期待効果 |
|----|------|---------|---------|
| OPS-1 | atexit で RSS/状態ダンプ | `fill_test_cli.py` | クラッシュ原因特定 |
| OPS-2 | health_monitor RSS チェック頻度 UP | `fill_test.yaml` | OOM 事前検知 |
| OPS-4 | restart.lock 保持期間延長 | `watchdog.ps1` | dual-spawn 防止 |
| OPS-6 | restart 後の lock 確認待ち | `watchdog.ps1` | dual-spawn 防止 |

### 中期パラメータ調整 (次期 fill test)

| ID | 内容 | 備考 |
|----|------|------|
| ~~TUNE-1~~ | ~~`forced_buy_delay` 残存確認~~ | 348# 撤廃済み。389 件は pre-348 SHA (e7d2f50d9b 等) のレコード。03-09 以降は 0 件。**解決済み** |
| TUNE-2 | `per_side_dd_halt` 閾値見直し | 11.3% は高すぎる |
| TUNE-3 | `sell_dynamic_kill_threshold_bps` 緩和検討 (−0.3 → −0.5) | ewma_alpha=0.05 で反応が速すぎる可能性 |

---

## §7 ph3 ブロッカー状況 (359# からの引き継ぎ)

| ブロッカー | 状態 | 備考 |
|-----------|------|------|
| B1 yaml | ✅ 完了 | g2_sac_train.yaml 作成済み |
| B2 features | ✅ 暫定完了 | 12 features 選定。精度は G2 実行後に評価 |
| B3 feature_names | ✅ 完了 | HeavyTradingEnv に注入 |
| B4 multi-seed | ✅ 完了 | dispatch_multi_seed 実装 |
| B5 trainer 3-way | ✅ 設計決定 | C path (FillTestLoop 内蔵) |

**結論**: 全 ph3 コードレベルブロッカーは解消済み。次ステップは SAC 訓練実行 (G2 ゲート試行)。
fill test は安定性問題の解消が先決。

---

## §8 次のアクション

1. **OPS-5**: Task Scheduler 設定確認 (手動)
2. **OPS-1**: atexit hook 追加 (`fill_test_cli.py`)
3. クラッシュ原因特定後、fill test 再起動して G1.1 168h 再計測
4. G1.1 PASS 後 → G2 SAC 訓練開始
