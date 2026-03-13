# 154# Dry-Run 10h ログ分析 & 改善提案

> **対象期間**: 2026-02-23 00:00 〜 13:57 (JST)  
> **対象ログ**: `results/v460/fill_test/logs/fill_test.log`  
> **分析時点**: 2026-02-23 13:57 JST

---

## §1 概要指標

### 1.1 サイクル実行サマリ (2/23 全日)

| 指標 | 値 | 備考 |
|---|---|---|
| サイクル開始 (=== Cycle) | **36** | 2071–2105 |
| skip_gate SKIP | **15** (42%) | buy=9, sell=6 |
| 発注 (Placed) | **20** | |
| 約定 (Filled) | **15** | |
| 未約定キャンセル | **3** | timeout 92–93s |
| P0-08 deadlock skip | **278** | ★ 04:36 以降 9.5h 空転 |
| Spread too narrow (ERROR) | **7** | min 1200 JPY 未達 |
| ERROR 合計 | **10** | |

### 1.2 PnL 実績 (Filled 15 件)

| Side | 件数 | 平均 PnL (bps) | 最大 | 最小 |
|---|---|---|---|---|
| **buy** | 7 | **+1.91** | +7.25 | -0.01 |
| **sell** | 8 | **-1.54** | +1.40 | -6.66 |
| **全体** | 15 | **+0.01** | +7.25 | -6.66 |

### 1.3 Progress (cumPnL 推移)

```
02-22 03:11  cycle=1750  fill=63.3%  cumPnL=-234.2 JPY  regime=ranging
02-22 22:17  cycle=2050  fill=60.5%  cumPnL=-299.0 JPY  regime=ranging
02-23 04:19  cycle=2100  fill=60.5%  cumPnL=-295.3 JPY  regime=trending
```

---

## §2 CRITICAL 発見: P0-08 デッドロック

### 2.1 経緯

1. **04:29 Cycle 2104 (buy)**: 0.001 BTC @ ¥10,463,311 — 6.1s で fast fill
2. **04:31**: JPY 残高 → **¥2,117** (BTC=0.002) — buy に必要な最低 ~¥10,116 を大幅に下回る
3. **04:33 Cycle 2105 (sell)**: sell limit @ ¥10,472,681 — **92.9s timeout** 未約定
4. **04:36〜13:57**: すべてのサイクルが以下のループで空転:

```
check balance(buy) → Insufficient JPY (2117 < ~10,116)
→ switch to sell (091#)
→ freeze buy 3 cycles (120#)
→ P0-08: balance_forced_switch=True → SKIP (avg -1.98bps)
→ sleep 120s → repeat
```

### 2.2 根本原因

- **P0-08 の無条件スキップ**: `skip_balance_forced: true` のとき、forced switch サイクルを **常に** スキップする
- 買えない（JPY 枯渇）→ 売りに切替 → P0-08 が売りもスキップ → **永久デッドロック**
- 0.002 BTC を保持しているが、P0-08 が sell を一切許可しない
- **9.5 時間** (278 サイクル分) 完全に機能停止

### 2.3 preflight / balance_shrink との関係

- preflight 不足判定後に `opposite` side (sell) が OK → **preflight_skip_count はリセットされる**
- つまり preflight_pause / SAFE_STOP は発動しない（preflight 自体は "成功" 扱い）
- balance_shrink も発動しない（同じ理由）
- **あらゆるフェイルセーフを迂回**する盲点

---

## §3 その他の発見事項

### 3.1 Retrain Scheduler 停止

```
03:37 trades unhealthy: missing_days=['20260221','20260220'], stale=83.1h
05:37 consecutive_skips=2, next_check_in=14400s (4.0h)
09:37 consecutive_skips=3
13:37 consecutive_skips=4
```

- モデル最終学習: **2026-02-21 20:56** → **40h+ 陳腐化**
- trades データ収集が 2/20 以降停止している可能性
- 指数 backoff (1h→2h→4h→4h) で回復チャンスが減少

### 3.2 Skip Gate 攻撃性

- 36 サイクル中 **15 が skip_gate で棄却 (42%)**
- buy skip score: -0.924 〜 -5.549 (6 件)
- sell skip score: -0.664 〜 -1.168 (3 件 + unified model for remaining)
- 一方で、通過した buy は平均 +1.91bps → モデルは buy 方向の判別はある程度有効
- **問題**: unified model への fallback 後 (03:37〜) の精度が不明

### 3.3 Sell 側の脆弱性

- Sell PnL 平均: **-1.54bps** (buy は +1.91bps)
- Cycle 2075: -6.66bps (wait 45.1s — 長い fill time と逆選択の典型)
- Cycle 2091: -4.28bps (wait 70.7s — 同様)
- **Sell の fill time が長い ≒ 逆選択リスク高** パターンが再現

### 3.4 Fast Fill Defense 発動

- Cycle 2104 (buy, 6.1s): fast_fill_defense 発動 → multiplier×2.00
- これが JPY 枯渇の直接原因ではないが、fast fill 自体は逆選択リスクの兆候

### 3.5 Watchdog Heartbeat Stale

```
05:37 RUNNING PID=108148 | heartbeat STALE (4701s ago, threshold=300s)
```

- プロセスは生存しているが、**1時間以上** heartbeat が更新されていない
- watchdog は "RUNNING" と判定しているため、restart は発動しない
- deadlock サイクルでは heartbeat 更新がスキップされている可能性

### 3.6 Model Path Mismatch

```
WARNING: target='pnl30' but model_path contains 'pnl120'
```

- retrain_scheduler が毎回出力しているが、実害は不明
- target と model_path の不一致は混乱の元

### 3.7 Spread Too Narrow

- 756〜1,174 JPY (min 1,200 JPY) で 7 回 ERROR
- 主に 04:17〜04:23 の時間帯 (trending regime)
- trending 時のスプレッド縮小でエッジが消える

---

## §4 改善提案

### 4.1 P0-Critical: デッドロック解消 (即時修正)

| ID | 提案 | 優先度 | 工数 |
|---|---|---|---|
| **C-1** | P0-08 を「buy/sell 両方残高 OK の場合のみスキップ」に変更 | ★★★ | 0.2h |
| **C-2** | P0-08 デッドロック検出カウンタ追加 (N回連続 forced_skip → sell を強制実行) | ★★★ | 0.3h |
| **C-3** | SAFE_STOP に「P0-08 連続 N 回」も含める (preflight 迂回の穴を塞ぐ) | ★★☆ | 0.2h |

**推奨**: C-1 + C-2 の併用。C-1 は根本修正、C-2 はフォールバック。

```python
# C-1 案: run_fill_test.py L1691 付近
if _balance_forced and self.config.skip_balance_forced:
    # 154# C-1: 反対 side も残高不足の場合のみスキップ
    #   → 片側しか取引できない状況では forced でも実行する (deadlock 防止)
    opposite_side = "sell" if next_side == "buy" else "buy"
    opposite_also_insufficient = await self._check_balance_for_side(
        opposite_side, regime_mult=_regime_mult
    )
    if not opposite_also_insufficient:
        # 反対 side (= next_side に forced switch された側) のみ実行可能
        # → デッドロック防止のためスキップしない
        logger.info(
            f"[154# C-1] balance_forced but opposite side also insufficient "
            f"— proceeding with {next_side} to avoid deadlock"
        )
    else:
        # 両方 OK → 従来通りスキップ
        logger.info(
            f"[133# P0-08] Skipping cycle — balance_forced_switch=True ..."
        )
        ...
        continue
```

### 4.2 P0-High: Retrain 回復

| ID | 提案 | 優先度 | 工数 |
|---|---|---|---|
| **R-1** | trades データ収集の健全性確認・修復 (missing_days の原因特定) | ★★★ | 0.5h |
| **R-2** | retrain backoff 上限を設定 (max 4h → max 2h に変更) | ★★☆ | 0.1h |
| **R-3** | trades_health が UNHEALTHY でも fill_records だけで retrain を試みるオプション | ★★☆ | 0.3h |

### 4.3 P1-Medium: Sell 品質向上

| ID | 提案 | 優先度 | 工数 |
|---|---|---|---|
| **S-1** | fill_time > 30s の sell は「slow fill → 逆選択」として skip_gate 学習に重み付け | ★★☆ | 0.5h |
| **S-2** | sell offset 動的拡張: trending regime で sell offset を 1.5x にする (スプレッド内エッジ確保) | ★★☆ | 0.3h |
| **S-3** | sell の PnL moving average が閾値以下なら自動で P0-10 (sell_dynamic_kill) 発動 | ★☆☆ | 0.2h |

### 4.4 P1-Medium: Watchdog 強化

| ID | 提案 | 優先度 | 工数 |
|---|---|---|---|
| **W-1** | deadlock ループ内でも heartbeat を更新する (stale 誤検知防止) | ★★☆ | 0.1h |
| **W-2** | watchdog に「heartbeat stale + process running」時の restart 機能追加 | ★★☆ | 0.3h |
| **W-3** | P0-08 連続 skip を watchdog metrics として公開 | ★☆☆ | 0.2h |

### 4.5 P2-Low: その他

| ID | 提案 | 優先度 | 工数 |
|---|---|---|---|
| **M-1** | model path / target の不一致を起動時に ERROR 昇格 (WARNING から変更) | ★☆☆ | 0.1h |
| **M-2** | spread too narrow 時の fallback offset (min_spread → offset 自動調整) | ★☆☆ | 0.3h |

---

## §5 即時対応の推奨

1. **C-1 コード修正をデプロイ** → deadlock 解消
2. **残高へ JPY 追加入金** → buy 再開可能にする (or BTC 0.001 を手動 sell して JPY 回復)
3. **trades データ収集の原因調査** → retrain 復旧

---

## §6 実装記録

### 6.1 C-1 + C-2 実装 (2026-02-23)

**変更ファイル**:
- `scripts/v460/lib/fill_config.py`: `balance_forced_deadlock_limit` 追加 (default=3)
- `scripts/v460/run_fill_test.py`: P0-08 ブロックを C-1/C-2 対応に書き換え
- `configs/v460/fill_test.yaml`: `balance_forced_deadlock_limit: 3` 追加
- `tests/unit/v460/test_154_deadlock_prevention.py`: 15 テスト追加 (全 PASS)

**ロジック**:
1. **C-1**: `_balance_forced=True` 時、元の side も残高不足なら forced side で実行を許可
2. **C-2**: 連続 forced skip が `balance_forced_deadlock_limit` (default=3) 回に達した場合もフォールバック実行
3. 実サイクル実行時にカウンタリセット

**テスト結果**: 1538 passed, 15 new (154#), 1 pre-existing failure (unrelated)

---

## §7 レビュー・追記欄

*(レビュー結果をここに追記)*
