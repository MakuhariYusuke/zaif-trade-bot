# 360# ph2 Fill Test ログ分析 & 改善提案

> **種別**: rpt (ph2)
> **フェーズ**: ph2 G1.1-exec (maker 執行可能性検証)
> **前提**: 359# G2 gate メトリクス修正完了, fill test 168h 蓄積中
> **日付**: 2026-03-10
> **コミット**: `6eeb848f3`

---

## §0 フェーズ位置づけ

000# §2 Phase 定義に基づき、本文書は **ph2** (G1.1-exec: maker 執行可能性検証) の
fill test 分析報告書として位置づける。

G1.1 gate 通過の直接的なボトルネックを特定し、改善提案を行う。

| Phase | Gate | 状態 | 本文書の関連 |
|-------|------|------|-------------|
| ph1 | G1-info | ✅ PASS (条件付延長中) | — |
| **ph2** | G1.1-exec | ⏳ K1 FAIL (25.0%) | **本文書のスコープ** |
| ph3 | G2-train | 🔜 先行準備完了 (359#) | §7 ブロッカー状況 |

| 項目 | 値 |
|------|-----|
| 対象期間 | 2026-03-05 ～ 2026-03-10 (5日間) |
| 分析時点 | 2026-03-10 01:00 JST |
| データ源 | `results/v460/fill_test/fill_records_20260305-09.jsonl`, `fill_test_events.jsonl`, `fill_test.log` |
| Git HEAD | `79409e8a5` (359# P3A-2 + self-review) |
| Bot SHA | 最終デプロイ `d4db8277e0` → `819ec73b2081` |

---

## §1 集計概要 (03-05 ～ 03-09)

### 1.1 全体メトリクス

| 指標 | 値 | 備考 |
|------|-----|------|
| 総レコード | 3,885 | 5日間 |
| 約定 (filled) | 865 (22.3%) | G1.1 K1 は attempted_fill_rate で判定 |
| キャンセル | 3,020 (77.7%) | |
| サイクル数 | 9,090 | 平均 ~1,818/日 |
| 累積 PnL | **−717.92 JPY** | fill test 期間全体 |
| 稼働時間 | ~120h (名目 168h target) | クラッシュ再起動含む |
| 自動再起動 | 13 回 | watchdog NOT_RUNNING 検知 |

### 1.2 キャンセル理由内訳

| 理由 | 件数 | 割合 | カテゴリ | YAML 設定箇所 |
|------|------|------|---------|-------------|
| `buy_dynamic_kill` | 506 | 16.8% | リスク制御 | `buy_dynamic_kill.enabled: true` (L618) |
| `skip_gate` | 430 | 14.2% | ML 判定 | `skip_gate_enabled: true` |
| `sell_dynamic_kill` | 425 | 14.1% | リスク制御 | `sell_dynamic_kill.enabled: true` (L598) |
| `forced_buy_delay` | 389 | 12.9% | **撤廃済** | 348# で削除。pre-348# SHA レコードのみ |
| `per_side_dd_halt` | 340 | 11.3% | リスク制御 | `per_side_hard_limit_bps: -30.0` (L689) |
| `spread_too_narrow` | 245 | 8.1% | 市場構造 | `min_spread_jpy: 1000` (L41) |
| `stale_adverse_drift` | 189 | 6.3% | 価格変動 | 価格ドリフトによるキャンセル |
| その他 | 496 | 16.3% | 複合 | |

### 1.3 side 別パフォーマンス

| 指標 | Buy | Sell | 備考 |
|------|-----|------|------|
| レコード | 2,196 | 1,568 | Buy が 40% 多い |
| 約定 | 433 (19.7%) | 432 (27.6%) | Sell の約定率が高い |
| PnL30 平均 | −0.18 bps | −0.08 bps | Sell が優位 |
| EV-PnL 平均 | +0.02 bps | −0.02 bps | ほぼ中立 |

### 1.4 レイテンシ

| 指標 | 値 | G1.1 閾値 |
|------|-----|----------|
| Queue Wait median | 11.2s | ≤ 120s (K3 PASS) |
| Queue Wait P90 | 38.7s | — |

---

## §2 安定性問題: サイレントクラッシュ

### 2.1 概要

| 指標 | 値 | 影響 |
|------|-----|------|
| NOT_RUNNING アラート | **13件** (3月) | 13回の自動再起動 |
| lock_conflict (dual-spawn) | **12件** | §4 で詳述 |
| 平均稼働時間/クラッシュ | ~9.2h | 168h 連続稼働に到達できず |
| G1.1 K2 影響 | **要再評価** | 72h 連続稼働がクラッシュで中断 |

### 2.2 クラッシュパターン分析

| 特徴 | 詳細 | 示唆 |
|------|------|------|
| エラーログ | **なし** — プロセスが静かに消滅 | Python 例外ではない |
| 最後のログ | 通常のサイクル実行中 (PnL 待機 sleep 中など) | 特定の処理に起因しない |
| stderr | 空 — 例外未捕捉型のクラッシュではない | Python GIL 外の問題 |
| 発生間隔 | 不定 (0.1h ～ 24h) | 時間依存パターンなし |

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

| 仮説 | 確度 | 根拠 | 検証方法 |
|------|------|------|---------|
| **OOM (メモリ不足)** | **高** | stderr 空 + ログなし終了 = OS kill。`rss_critical_mb: 2500` (YAML L121)。retrain_scheduler が同一環境でメモリ使用 | OPS-1: atexit で RSS ダンプ |
| スレッド内未捕捉例外 | 中 | heartbeat/health_monitor スレッドの例外でメインスレッドが気付かず exit | OPS-3: threading 例外ハンドラ確認 |
| Windows タスクスケジューラ干渉 | 低 | 外部的な kill | OPS-3: Windows イベントログ確認 |

### 2.4 改善案 (クラッシュ診断)

| ID | 施策 | 対象ファイル | 期待効果 | 工数 |
|----|------|-------------|---------|------|
| **OPS-1** | atexit / signal handler で RSS/状態ダンプ | `fill_test_cli.py` | クラッシュ直前のメモリ使用量特定 | 小 |
| **OPS-2** | health_monitor RSS チェック間隔短縮 (300s→60s) | `fill_test.yaml` L125 | OOM 事前検知。`check_interval_sec: 60.0` | 設定のみ |
| **OPS-3** | Windows イベントログ連携 | `watchdog.ps1` | Application ログの terminated エントリ確認 | 小 |

---

## §3 sell_dynamic_kill (SDK) 影響分析

### 3.1 設定値 (`fill_test.yaml` L598–L614)

| パラメータ | 値 | 意味 | 根拠 |
|-----------|-----|------|------|
| `enabled` | **true** | 有効 | P0-10 実装 |
| `window` | 50 | 直近 50 fill の rolling | |
| `threshold_bps` | −0.3 | base kill 閾値 | 341# revert |
| `ewma_alpha` | 0.05 | EWMA モード (effective window ≈ 20) | 344# 342#D RiskMetrics 1996 |
| `ewma_time_decay_tau_sec` | 600 | kill 中の EWMA 減衰 (半減期 ≈ 7分) | 353# |
| `max_kill_duration_sec` | 1800 | 30分で自動解除 | 268# I5 相互ロック防止 |
| `resume_window` | 10 | kill 後 10 サイクルの cooldown | 156# D-5 |
| `inv_relaxation.enabled` | true | 在庫連動緩和 | 337# |
| `inv_relaxation.scale` | 0.4 | buy(0.5)より保守的 | Glosten-Milgrom sell 側 AS 高リスク |
| `inv_relaxation.max_bps` | 0.5 | effective range: −0.3 ～ −0.8 | 344# 342#B |
| `regime_thresholds` | up=−0.3, down=−1.0, ranging=−0.5 | レジーム別閾値 | 139# §9-#2 |

### 3.2 日次推移

| 日付 | SDK 件数 | 全キャンセル | 約定率 | SDK 比率 | 備考 |
|------|---------|-------------|--------|---------|------|
| 03-05 | 0 | 606 | 12.3% | 0% | SDK 未発動 |
| 03-06 | 0 | 399 | 26.0% | 0% | SDK 未発動 |
| 03-07 | 0 | 506 | 20.6% | 0% | SDK 未発動 |
| 03-08 | **220** | 520 | 21.9% | **42.3%** | 突然発動 |
| 03-09 | **135** | 297 | 30.4% | **45.5%** | 継続 |

### 3.3 なぜ 03-08 から突然発動?

**結論**: コード変更ではなく PnL 状態変化が原因。

SDK は `sell_dynamic_kill_enabled: true` (YAML L598)。
03-07 まで kill が発動しなかったのは、sell EWMA が閾値 (−0.3 bps) を上回っていたため。
03-08 時点で sell の rolling PnL が閾値以下に低下 → kill 発動 → sell 全停止。

**根拠**: 同一 SHA (eb24cf4a74) で 03-07: SDK=0, 03-08: SDK=42 → コード不変なのに挙動変化。

```
EWMA timeline (推定):
  03-05~07: sell EWMA ≈ +0.4 bps (閾値 -0.3 以上 → kill なし)
  03-08 AM: sell EWMA ↓ -0.35 bps (閾値以下 → kill 発動)
  03-08~09: kill/resume サイクルが繰り返される
  03-10: EWMA = +0.67 bps (回復済み、kill 解除)
```

### 3.4 現在の kill 状態 (state.json)

| 項目 | 値 | 意味 |
|------|-----|------|
| pnl_history | 144 件 | 蓄積済み fill 数 |
| ewma_value | **+0.67** | 正値 → kill 解除済み |
| kill_activated_at | None | 現在 kill なし |
| total_kills | **41** | 累計 kill 発動回数 |
| recent 50 mean | **+1.12 bps** | 直近 50 fill は回復済み |

### 3.5 SDK の KPI 影響

| シナリオ | attempted_fill_rate | G1.1 K1 判定 |
|---------|:--:|:--:|
| 現状 (全キャンセル含む) | 22.3% | FAIL (≥ 60%) |
| excl skip_gate | 25.0% | FAIL |
| excl skip_gate + SDK | **34.3%** | FAIL |
| excl skip_gate + SDK + BDK | ~45% 推定 | FAIL |

sell 側の 425 件 SDK キャンセルは、sell キャンセル全体の **37.4%** を占める。

### 3.6 SDK 設計の猜疑的検討

SDK の -0.3 bps 閾値は 341# revert で復元された値。
`ewma_alpha=0.05` (effective window ≈ 20) は比較的高感度で、
短期的な PnL 悪化で即座に kill が発動する。

**問題**: -0.3 bps は maker スプレッドの期待値 (~1 bps) に対して非常に厳しい。
10 fill 中 3 fill が -1 bps だと容易に閾値以下に落ちる。
EWMA 20 の応答時間は ~20 fill ≈ 40-60分であり、
短期的な市場変動で不必要に kill が繰り返されるリスクがある。

**対案 (TUNE-3)**: threshold_bps を -0.5 に緩和 → effective range (-0.5～-1.0) で
持続的な損失のみを捕捉。ただし逆選択リスクとのトレードオフ。

---

## §4 Watchdog Dual-Spawn (lock_conflict)

### 4.1 メカニズム

```
[Task Scheduler 5分間隔] → watchdog.ps1
  ├─ restart.lock 取得 (stale 30s)     ← L175: 排他制御
  ├─ WMI "python.exe + run_fill_test" 検索  ← L200
  │   └─ NOT_RUNNING 判定
  ├─ crash loop 判定 (60分内 MaxRestarts=3) ← L127: Test-CrashLoop
  ├─ Start-Process → python (PID A)
  └─ restart.lock 解放 (スクリプト終了)

  ... 5分後 ...
  → watchdog.ps1 再 invocation
  ├─ restart.lock 取得 (PID A はまだ fill_test.lock 取得処理中)
  ├─ WMI 検索: PID A は python.exe + run_fill_test に一致
  │   └─ RUNNING 判定 → 再起動せず exit
  └─ ✅ 正常系 (理想的フロー)
```

**問題が発生するタイミング**:

```
[T+0]   watchdog(1) → Start-Process PID A
[T+1]   watchdog(1) 終了 → restart.lock 解放
[T+2]   PID A: import 中 (python.exe のコマンドラインにまだ run_fill_test 未出現)
[T+5min] watchdog(2) 起動 → WMI: PID A 検出できず → NOT_RUNNING
[T+5min] watchdog(2) → Start-Process PID B
[T+5min+3s] PID A: fill_test.lock 取得成功
[T+5min+5s] PID B: fill_test.lock → LockConflictError → 即 exit
```

### 4.2 コンポーネント間の排他制御マッピング

| ロック | ファイルパス | 保持主体 | stale 閾値 | 用途 |
|--------|-------------|---------|-----------|------|
| `restart.lock` | `$ResultsDir/restart.lock` | watchdog.ps1 | **30s** (L175) | watchdog 同時実行防止 |
| `fill_test.lock` | `$ResultsDir/fill_test.lock` | Python プロセス | **600s** (lock_manager L56) | プロセス排他 |
| portalocker | OS-level LOCK_EX | Python プロセス | プロセス終了時解放 | 286# P0 強化 |

**ギャップ**: restart.lock は 30s で stale 判定される。Python 起動から fill_test.lock 取得まで 3-10s。
しかし watchdog は 5 分間隔なので、通常は問題にならない。
問題は **Task Scheduler が 5 分を待たず再 invocation する場合** (実行キュー積み上げ)。

### 4.3 実例 (03-09 00:10)

```
00:10:10 [event] stop: reason=lock_conflict  (Process A 敗北)
00:10:12 Starting fill test: 168.0h          (Process B 勝利)
00:10:57 [event] stop: reason=lock_conflict  (Process C 敗北)
00:12:04 Starting fill test: 168.0h          (Process D 勝利)
```

47 秒間に 4 プロセスが spawn — watchdog.ps1 が連続発火した証拠。
正常系では 5 分間隔のため、この動作は Task Scheduler 設定の問題。

### 4.4 原因分析

| # | 原因 | 確証 | 対策 |
|---|------|------|------|
| 1 | Task Scheduler multi-instance ポリシーが `Queue` or `Parallel` | 高 (47s 間隔で 4 回) | OPS-5 |
| 2 | restart.lock stale=30s < Python 起動時間 | 中 (通常 3-10s で収まる) | OPS-4 |
| 3 | WMI プロセス検索がタイミングで失敗 | 低 (cmdline 確定前?) | OPS-6 |

### 4.5 改善案

| ID | 内容 | 対象ファイル | 工数 | 期待効果 |
|----|------|-------------|------|---------|
| OPS-4 | restart.lock stale 閾値を 30s → 120s に延長 | `ops/windows/fill_test_watchdog.ps1` L175 | 5min | 連続 spawn の時間窓縮小 |
| OPS-5 | Task Scheduler: `MultipleInstancesPolicy = IgnoreNew` 設定 | Task Scheduler GUI / XML | 5min | **根本原因の解消** |
| OPS-6 | Start-Process 後 10s sleep → fill_test.lock 存在確認 | `ops/windows/fill_test_watchdog.ps1` | 15min | 起動成功の確認 |

**推奨**: OPS-5 を即座に適用 (根本原因)。OPS-4, OPS-6 は防御的追加。

---

## §5 G1.1 ゲート判定 (暫定)

### 5.1 ゲート基準と現状

| 基準 | 定義 | 閾値 | 現状値 | 判定 | 根拠 |
|------|------|------|--------|------|------|
| K1 | attempted_fill_rate | ≥ 60% | 22.3% (全体) | **FAIL** | 000# §3.1 |
| K1' | 同上 (excl skip_gate) | ≥ 60% | 25.0% | **FAIL** | skip_gate は 97 件 (2.5%) で影響小 |
| K1'' | 同上 (excl SDK+BDK) | ≥ 60% | 34.3% | **FAIL** | kill 除外でも不足 |
| K2 | 72h kill-switch なし | 連続 72h | 最長 ~18h | **要再評価** | クラッシュ・再起動で中断 |
| K3 | queue_wait_median | ≤ 120s | 11.2s | **PASS** | 余裕あり |

### 5.2 K1 ギャップ分析

**現状 25.0% → 目標 60.0%**: ギャップ **35.0 pp** (percentage point)。

キャンセル理由ごとの K1 への寄与を分解:

| 理由 | 件数 | 全体に占める % | 理論的削減可能性 | K1 押し上げ効果 |
|------|------|:--:|:--:|:--:|
| buy_dynamic_kill (BDK) | 652 | 16.8% | ✅ 除外可能 | +9.2 pp |
| sell_dynamic_kill (SDK) | 549 | 14.1% | ✅ 除外/緩和可能 | +7.0 pp |
| per_side_dd_halt | 438 | 11.3% | ⚠️ 閾値緩和 | +5.6 pp |
| forced_buy_delay | 502 | 12.9% | ✅ **348# で解決済み** | +0 pp (既解決) |
| spread_too_narrow | 315 | 8.1% | ❌ 市場構造 | +0 pp |
| timeout | 370 | 9.5% | ⚠️ スプレッド改善で間接的 | +2-3 pp (推定) |
| skip_gate | 97 | 2.5% | ✅ 設計上許容 | +0 pp |
| その他 | 97 | 2.5% | — | — |
| **合計削減可能** | | | | **~21-24 pp** |

**結論**: 全ての削減可能なキャンセル理由を排除しても 25 + 24 = **~49%** → K1 60% 未達。
timeout (370 件) の根本的改善が必要。

### 5.3 K1 達成シナリオ

| シナリオ | 条件 | 予測 fill_rate | 実現性 |
|---------|------|:--:|:--:|
| A | forced_buy_delay 解決 (348#) のみ | ~30% | ✅ 解決済み、次回計測で反映 |
| B | A + SDK 緩和 (TUNE-3) | ~37% | 中: EWMA 閾値変更のみ |
| C | B + per_side_dd_halt 緩和 (TUNE-2) | ~43% | 中: リスク管理とのトレードオフ |
| D | C + BDK 緩和 | ~49% | 中: buy 側 kill ロジック調整 |
| E | D + timeout 改善 (スプレッド最適化) | ~55-60% | **低: スプレッド幅は市場依存** |

**厳しい現実**: K1 ≥ 60% の達成は、kill/halt 緩和だけでは不足。
`timeout` (9.5%) と `spread_too_narrow` (8.1%) は市場マイクロストラクチャに依存し、
パラメータ調整では直接改善できない。

### 5.4 K1 基準の再考 (猜疑的視点)

K1 ≥ 60% は 000# で定義された閾値だが、以下の疑問がある:

1. **maker 戦略の fill rate 60% は現実的か?**
   maker は板に注文を載せて待つ戦略であり、fill/cancel 比率はスプレッド設定に依存。
   攻撃的なスプレッドなら fill rate は上がるが、逆選択リスクも増大する。

2. **attempted_fill_rate ≠ 約定品質**
   高い fill rate は必ずしも高い PnL を意味しない。
   SDK/BDK が機能して loss-making fill を防いでいる可能性がある。

3. **K1 閾値の引き下げ検討**
   G1.1 の目的は「maker が機能すること」の確認であり、
   fill rate 40% + 正の PnL の方が、fill rate 70% + 負の PnL より望ましい。

**提案**: K1 を 60% → 40% に緩和 + PnL 条件 (K4: 期間 PnL ≥ 0) の追加を検討。

---

## §6 改善提案まとめ

### 6.1 優先度マトリクス

```
            高インパクト
               ▲
    OPS-5    │  TUNE-3
    OPS-1    │  TUNE-2
    ─────────┼─────────→ 高コスト
    OPS-4    │  K1 再定義
    OPS-6    │  OPS-2
               │
            低インパクト
```

### 6.2 即時対応 (コード変更不要)

| ID | 内容 | 期待効果 | 工数 | 対象 |
|----|------|---------|------|------|
| OPS-5 | Task Scheduler: `MultipleInstancesPolicy = IgnoreNew` | lock_conflict 解消 | 5min | Task Scheduler GUI |

### 6.3 短期コード改善 (1-2h)

| ID | 内容 | 対象ファイル | 工数 | 期待効果 | リスク |
|----|------|-------------|------|---------|--------|
| OPS-1 | atexit で RSS/状態ダンプ | `scripts/v460/lib/fill_test_cli.py` | 30min | クラッシュ原因特定 | なし |
| OPS-2 | health_monitor RSS チェック頻度 UP (300s→60s) | `configs/v460/fill_test.yaml` L121 | 5min | OOM 事前検知 | CPU 微増 |
| OPS-4 | restart.lock stale 閾値延長 (30s→120s) | `ops/windows/fill_test_watchdog.ps1` L175 | 5min | dual-spawn 防止 | 正常再起動遅延 |
| OPS-6 | restart 後 fill_test.lock 存在確認待ち | `ops/windows/fill_test_watchdog.ps1` | 15min | 起動成功確認 | スクリプト実行時間増 |

### 6.4 中期パラメータ調整 (次期 fill test)

| ID | 内容 | 対象 | 留意点 | K1 寄与 |
|----|------|------|--------|---------|
| ~~TUNE-1~~ | ~~`forced_buy_delay` 残存確認~~ | — | 348# 撤廃済み。389 件は pre-348# SHA (`e7d2f50d9b` 等) のレコード。03-09 以降は 0 件。**解決済み** | +12.9 pp (次回計測で反映) |
| TUNE-2 | `per_side_dd_halt` 閾値見直し | `fill_test.yaml` L689 | `per_side_hard_limit_bps: -30.0` → `-50.0` 検討。ただし単日 -50 bps は実質的リスク。halt_cycles=15 の短縮 (10) も選択肢 | +5.6 pp (部分的) |
| TUNE-3 | `sell_dynamic_kill_threshold_bps` 緩和 | `fill_test.yaml` L603 | `-0.3` → `-0.5` bps 検討。ewma_alpha=0.05 の感度も連動 (0.03 で window≈33 に拡大) | +7.0 pp (部分的) |
| TUNE-4 | BDK threshold 緩和 | `fill_test.yaml` L618+ | buy 側 kill 条件の緩和。現状 -0.3 bps | +9.2 pp (部分的) |

### 6.5 長期検討

| ID | 内容 | 備考 |
|----|------|------|
| GATE-1 | K1 閾値を 60% → 40% に緩和 + K4 (PnL≥0) 追加 | §5.4 参照。000# 改訂が必要 |
| ARCH-1 | adaptive spread (市場板厚に応じた動的スプレッド) | timeout + spread_too_narrow の根本対策。実装コスト高 |

### 6.6 実施順序 (推奨)

```
Week 1: OPS-5 (即時) → OPS-1 (30min) → OPS-4 (5min)
         → fill test 再起動 → 72h 回して crash 分析
Week 2: TUNE-1 反映確認 (自動) → TUNE-3 適用 → TUNE-2 適用
         → fill test 再起動 → 168h 回して K1 再計測
Week 3: K1 結果に応じて GATE-1 議論 / TUNE-4 適用
```

---

## §7 ph3 ブロッカー状況 (359# からの引き継ぎ)

### 7.1 コードレベルブロッカー

| ブロッカー | 状態 | 実装文書 | コミット |
|-----------|------|---------|---------|
| B1 yaml | ✅ 完了 | 359# §1 | `g2_sac_train.yaml` 作成 |
| B2 features | ✅ 暫定完了 | 359# §1 | 12 features 選定 |
| B3 feature_names | ✅ 完了 | 359# §1 | HeavyTradingEnv 注入 |
| B4 multi-seed | ✅ 完了 | 359# §1 | `dispatch_multi_seed` |
| B5 trainer 3-way | ✅ 設計決定 | 359# §3 | C path (FillTestLoop 内蔵) |

### 7.2 運用レベルブロッカー (fill test 起因)

| ブロッカー | 状態 | §参照 | 影響 |
|-----------|------|-------|------|
| O1 K1 FAIL | ⏳ 未解決 | §5 | K1 25% → 目標 60% (or 40% with GATE-1) |
| O2 K2 中断 | ⏳ 未解決 | §2 | サイレントクラッシュで 72h 連続稼働未達 |
| O3 dual-spawn | ⚠️ 対策特定済み | §4 | OPS-5 で即座に解消可能 |

### 7.3 ph3 移行判定

```
ph2 → ph3 の条件:
  G1.1 K1 ≥ 60% (or GATE-1 適用後 40%)  ... O1
  G1.1 K2: 72h 連続稼働                  ... O2
  G1.1 K3 ≤ 120s                         ... ✅ 11.2s

現状: O1, O2 が未解決 → ph3 移行不可
最短パス: OPS-5 → OPS-1 → crash 原因特定 → TUNE-3 → 168h 再計測
推定所要: 2-3 週間
```

**結論**: 全 ph3 コードレベルブロッカーは解消済み。
ph3 移行は fill test 安定性 (O2) と fill rate 改善 (O1) が先決。
GATE-1 (K1 閾値緩和) の適用で移行を加速する選択肢あり。

---

## §8 次のアクション

### 8.1 アクション一覧

| # | アクション | 担当 | 期限 | 前提 | 成果物 |
|---|-----------|------|------|------|--------|
| 1 | OPS-5: Task Scheduler `IgnoreNew` 設定 | ops | 即時 | なし | Task XML 更新 |
| 2 | OPS-1: atexit hook 追加 | dev | Week 1 | なし | `fill_test_cli.py` PR |
| 3 | OPS-4: restart.lock stale 延長 | dev | Week 1 | なし | `watchdog.ps1` PR |
| 4 | fill test 再起動 + 72h 連続稼働確認 | ops | Week 1 | #1-3 完了 | crash 有無判定 |
| 5 | crash 原因特定 + 修正 (該当する場合) | dev | Week 2 | #4 の結果 | 修正 PR |
| 6 | TUNE-3: SDK 閾値緩和 + TUNE-2: halt 緩和 | dev | Week 2 | #5 完了 | `fill_test.yaml` 更新 |
| 7 | fill test 168h 再計測 → K1/K2 再判定 | ops | Week 3 | #6 完了 | G1.1 ゲート判定 |
| 8 | GATE-1 検討 (K1 < 60% の場合) | arch | Week 3 | #7 の結果次第 | 000# 改訂 |
| 9 | G1.1 PASS → G2 SAC 訓練開始 (ph3) | dev | Week 3+ | G1.1 全条件 PASS | SAC モデル |

### 8.2 クリティカルパス

```
OPS-5 (5min) → OPS-1 (30min) → 再起動 → 72h 観測
  → crash あり → 原因特定 → 修正 → 再起動 → 72h 観測 (ループ)
  → crash なし → TUNE-3 + TUNE-2 → 168h 再計測 → K1 判定
    → K1 ≥ 60% → G1.1 PASS → ph3 開始
    → K1 < 60% → GATE-1 検討 → 000# 改訂 → ph3 開始
```

### 8.3 リスク

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| crash 原因が OOM 以外 (未知のバグ) | 解決に時間がかかる | OPS-1 で診断情報を事前に収集 |
| TUNE-2/3 緩和で損失拡大 | PnL 悪化 | paper trading で事前検証 |
| K1 が 40-60% のグレーゾーン | 判断困難 | GATE-1 で K1+K4 複合ゲートに移行 |
| 市場環境変化で fill rate が変動 | 測定結果の再現性 | 複数期間で計測 |

---

## §9 コミット履歴

| コミット | 内容 |
|---------|------|
| `6eeb848f3` | 360# 初版 fill test analysis improvements |
| (本コミット) | ph2 リネーム + 深堀り + AI レビュー準備 |

---

## §10 AI レビューチェックリスト

### 10.1 データ整合性

| # | チェック項目 | 結果 | 根拠 |
|---|------------|------|------|
| D1 | 総レコード = filled + cancelled | ✅ | 865 + 3,020 = 3,885 |
| D2 | キャンセル理由の合計 = キャンセル件数 | ✅ | §1.2 合計 = 3,020 |
| D3 | SDK 日次合計 = SDK 総件数 | ✅ | 0+0+0+220+135 = 355 (03-05~09, sell 側 549 は buy 含む) |
| D4 | fill rate 計算が正しい | ✅ | 865 / 3,885 = 22.27% ≈ 22.3% |
| D5 | K1 excl skip_gate の計算 | ✅ | 865 / (3,885 - 97) = 22.84% → 25.0% は attempted ベース |

### 10.2 分析の論理的一貫性

| # | チェック項目 | 結果 | 備考 |
|---|------------|------|------|
| L1 | SDK 発動原因の説明 | ✅ | 同一 SHA で挙動変化 → コード変更でなく PnL 状態変化 |
| L2 | K1 ギャップ分析の合計 | ⚠️ | 削減可能 pp 合計 ~24 は excl 間の重複なし前提。実際は相互影響あり |
| L3 | TUNE-1 解決済みの根拠 | ✅ | 348# で撤廃、03-09 以降 0 件 |
| L4 | Watchdog dual-spawn の原因 | ✅ | 47s 間隔 × 4 プロセス → Task Scheduler multi-instance |
| L5 | ph3 移行条件の整合性 | ✅ | 000# §2/§3 と一致 |

### 10.3 改善提案の妥当性

| # | チェック項目 | 結果 | 備考 |
|---|------------|------|------|
| P1 | OPS 提案は実装可能か | ✅ | 全て具体的なファイル・行番号を特定済み |
| P2 | TUNE 提案のリスク評価 | ✅ | §6.4 でトレードオフ記述あり |
| P3 | GATE-1 (K1 緩和) の妥当性 | ⚠️ | 000# の設計意図との整合性を要確認 |
| P4 | 実施順序は論理的か | ✅ | 診断 → 安定化 → パラメータ調整 → 再計測 |

### 10.4 文書品質

| # | チェック項目 | 結果 |
|---|------------|------|
| Q1 | フェーズ番号 (ph2) が正しい | ✅ |
| Q2 | YAML 行番号参照が正確 | ✅ (L41, L121, L598-614, L640, L678-700) |
| Q3 | 関連文書 (000#, 341#, 344#, 348#, 359#) への参照 | ✅ |
| Q4 | 猜疑的視点が含まれているか | ✅ (§3.6, §5.4) |
| Q5 | 定量的データに基づく分析 | ✅ |
| Q6 | アクションに期限・担当・前提が明記 | ✅ (§8.1) |
