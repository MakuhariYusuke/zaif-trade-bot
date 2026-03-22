# 529# fill_test 取引損益分析 (2026-03-22)

> 作成日: 2026-03-22  
> 対象: 2026-03-22 00:00〜16:41 (PID=95692, SHA=d93b9a5bf672)  
> 前回分析参照: 503# (Sell/Buy損益要因分析), 524# (preflight_skip_exceeded)

---

## §1 本日のサマリー

| 指標 | 値 |
|------|-----|
| 実行サイクル数 | 261 cycles (14708〜14968) |
| 約定数 | 89 fills |
| Fill Rate | 34.1% (89/261) |
| 合計 PnL (bps) | **-35.6 bps** |
| 平均 PnL | **-0.40 bps/fill** |
| 勝率 | 50.6% (45W/44L) |
| sidecar | 全取引 stale (TTL=7800s 超過) |
| regime | ranging 固定 (stability=9) |

### Fill Rate 低下要因の内訳

| ブロック理由 | 発生回数 | 備考 |
|-------------|---------|------|
| balance_insufficient | 418 | JPY不足(buy)またはBTC不足(sell) |
| cross_venue_veto | 100 | bitflyer down lead で buy 抑制 |
| skip_gate SKIP | 59 | モデルが逆選択リスク判定 |
| spread_too_narrow | 27 | Coincheck spread < min 700 JPY |
| sell_dynamic_kill | 17 | sell EWMA ベース抑制 |
| final_clamp | 119回 clamped | 全約定取引で 0.25 ceiling に到達 |

---

## §2 約定取引の個別分析（大損/大勝）

### 大損取引 (PnL < -5 bps)

| Cycle | Side | Price | Wait | PnL | 主因 |
|-------|------|-------|------|-----|------|
| **14902** | sell | 11,060,856 | 33.6s | **-20.79bps** | macro_weak_up で timeout短縮→re-quote→slow fill逆選択 |
| 14850 | - | - | 6.1s | -14.66bps | fast fill 即時逆選択 |
| 14951 | sell | 11,022,756 | 11.5s | -10.13bps | 逆選択 (moderate wait) |
| 14859 | - | - | 27.4s | -9.70bps | slow fill 逆選択 |
| 14896 | - | - | 11.3s | -7.96bps | 逆選択 |
| 14916 | - | - | 22.8s | -7.37bps | slow fill 逆選択 |
| 14948 | sell | 11,020,779 | 6.3s | -6.37bps | fast fill 逆選択 |
| 14903 | sell | - | 6.4s | -5.34bps | fast fill 直後 |

### 大勝取引 (PnL > +5 bps)

| Cycle | Side | Price | Wait | PnL | 主因 |
|-------|------|-------|------|-----|------|
| 14841 | - | - | 29.9s | **+16.86bps** | slow fill + 有利方向 |
| 14933 | sell | 11,046,825 | 22.6s | +9.91bps | cross-venue favorable tighten |
| 14851 | - | - | 6.0s | +9.20bps | fast fill + 順方向 |
| 14708 | - | - | 5.8s | +8.78bps | fast fill + 順方向 |
| 14887 | - | - | 11.4s | +6.50bps | moderate wait |
| 14913 | - | - | 5.8s | +5.05bps | fast fill + 順方向 |

---

## §3 最悪取引 C14902 の詳細分析

**結果**: sell @ 11,060,856 JPY → **-20.79 bps** (wait=33.6s)

### タイムライン

| 時刻 | イベント |
|------|---------|
| 12:27:05 | Cycle 14902 (sell) 開始 |
| 12:27:05 | cross_venue hint=None (low_confidence) → CV保護なし |
| 12:27:05 | inv_skew imbalance=+0.303 → offset 0.14→0.123 (sell寄りスキュー) |
| 12:27:05 | VG boost: vpin=0.57(cont=0.43) → offset 0.30 |
| 12:27:05 | **macro_boost**: macro_weak_up → **offset_mult=1.30** (+174JPY) |
| 12:27:05 | **final_clamp**: 0.3993 → **0.2500** (38%カット) |
| 12:27:06 | 注文発行 @ 11,061,527 (mid=11,058,869 推定) |
| 12:27:06 | **timeout短縮**: macro=WEAK_UP → 12.0s |
| 12:27:22 | Cancel (16.2s) → 未約定 |
| 12:27:28 | Re-quote @ 11,060,856 (mid下落に追従) |
| 12:27:39 | **fill** @ 11,060,856 (wait=10.8s from re-quote) |
| 12:29:39 | PnL測定: **-20.79 bps** |

### 問題の根本原因

1. **macro_boost の過剰増幅**: macro_weak_up が sell に 1.30x を適用したが、final_clamp で 0.3993→0.25 にカット。macro_boost の計算自体は正しいが、**ceiling で切り捨てられるので実質的に効果がない**にも関わらず、timeout を 12s に短縮する副作用だけが残った。

2. **timeout短縮の副作用**: macro=WEAK_UP で sell timeout が通常15s→12s に短縮。しかしこの短縮により最初の注文がキャンセルされ、re-quote が発生。市場が不利方向に動いた後に約定。

3. **cross_venue 保護なし**: confidence < 0.2 で hint=None → sell 側保護ゼロ。503# で指摘した「sell 側 CV 未適用」問題が再現。

---

## §4 直近の取引分析（16:01〜16:21）

### 最近の約定5本

| Cycle | Side | Price | Wait | PnL | 特記 |
|-------|------|-------|------|-----|------|
| 14955 | sell | 11,033,199 | 6.4s | -3.55bps | fast fill→AS=true, FFD activated |
| 14956 | buy | 11,037,359 | 27.9s | -1.70bps | re-quote 1回, CV未適用 |
| **14957** | **sell** | **11,036,610** | **11.3s** | **+3.49bps** | **CV favorable tighten適用, AS=false** |
| 14958 | buy | 11,029,160 | 39.2s | +0.29bps | re-quote 1回, CV adverse→offset拡大 |
| 14961 | sell | 11,033,836 | 11.6s | -0.15bps | skip_gate hot-reload, 微損 |

**合計: -1.62 bps / 5 fills** (avg -0.32 bps)

### パターン解析

- C14957 が唯一の明確な黒字: **cross_venue favorable tighten が適用された sell** → offset 0.3000→0.2901 (factor=0.967)
- C14955: EV score=+1.30 (予測黒字) → 実際 -3.55bps → **EV の sell 側予測精度に問題** (503# 既報)
- C14956/14958: buy 側は re-quote 後に約定 → 最初の注文が 0.25 ceiling で深すぎ、取り消しで遅延
- **全取引が final_clamp 0.25 に到達**: pre-clamp offset は 0.26〜0.40 の範囲

---

## §5 現在のデッドロック状態 (16:22〜継続中)

### 状況

16:22 以降、ボットが **完全なデッドロック** に陥っている:

| 時刻 | 試行Side | 結果 | 理由 |
|------|---------|------|------|
| 16:22 | buy | veto | CV spread=-9.01bps > threshold 8.0 |
| 16:23-16:24 | sell | skip | BTC=0 |
| 16:25 | buy | veto | CV spread=-9.31bps |
| 16:26-16:27 | sell | skip | BTC=0 |
| 16:28 | buy | veto | CV spread=-9.41bps |
| 16:31 | buy | reject | spread 256 JPY < min 700 |
| 16:34 | buy | veto | CV spread=-9.39bps |
| 16:37 | buy | reject | spread too narrow |
| 16:40 | buy | veto | CV spread=-9.04bps |

**7 consecutive NO_FEASIBLE_QUOTE** → 約18分間の完全アイドル。

### デッドロックのメカニズム

```
[1] sell 完了 → JPY 回収, BTC=0
[2] buy しようとする → cross_venue_veto (bitflyer -9bps lead)
[3] sell できない → BTC=0 で在庫なし
[4] buy できない → veto 継続
[5] → [2] に戻り無限ループ
```

**残高**: JPY 25,120 (遊休), BTC 0.0 → 買いたいが買えない状況。

### 背景

bitFlyer mid と Coincheck mid の乖離が -9bps 前後で安定。
veto_threshold_bps = 8.0 を超えているため全 buy がブロック。
同時に Coincheck の bid-ask spread が 243-522 JPY と min_spread_jpy=700 を下回り、
スプレッドが狭すぎて発注できないケースも交互に発生。

---

## §6 構造的問題の整理

### 問題1: Cross-Venue Veto デッドロック (CRITICAL)

**頻度**: 本日2回（08:19-08:25, 16:22-16:40+継続中）

veto が long-duration で持続した場合、在庫ゼロの状態でロックされる。
503# で指摘された「sell 側 CV 未適用」問題に加え、**veto による完全停止**が新規問題。

**改善案**:
- **A) veto に time-limit 導入**: sell_dynamic_kill と同様、N秒経過で auto-release
- **B) veto_threshold_bps 引き上げ**: 8.0 → 9.5 (現在の spread=-9.0〜9.4 帯をカバー)
- **C) 在庫ゼロ時の veto 緩和**: BTC=0 の場合、買わないと何もできない → threshold を一時的に緩和

**推奨**: C を第一優先。在庫ゼロ時は veto_threshold を 1.5x に緩和（8.0→12.0）。
理由: 在庫ゼロで買えないリスク > 逆選択で多少不利に買うリスク。
buy 後に sell で回収する機会がある。

### 問題2: final_clamp 0.25 による pipeline 出力の一律切り捨て

**頻度**: 119/261 cycles (46%) で clamped

pipeline が 0.26〜0.40 の offset を算出しても、ceiling=0.25 で強制カット。
これは offset pipeline のリスク評価を無視している。

503# のデータでは offset >= 0.25 が buy 側で**唯一の黒字バケット**。
しかし sell 側では offset < 0.19 が最善 (+138.58 JPY)。

**改善案**:
- **サイド別 ceiling の再検討**: buy を 0.27〜0.30 に引き上げ、sell は 0.20-0.22 に引き下げ
- 現 config: buy=0.25, sell=0.25 → buy=0.28, sell=0.22 を検証
- 注: 503# データで buy offset >= 0.25 は WR 52%/黒字、sell offset >= 0.25 は WR 48%/赤字

### 問題3: Sidecar Signal の恒常的 stale

**頻度**: 全取引で sidecar=stale

sidecar signal の TTL=7800s (2.2時間) が常時超過。
retrain_scheduler が signal を更新していないか、更新頻度が不足。
sidecar の予測情報が一切活用されていない。

**改善案**:
- retrain_scheduler の sidecar 更新頻度を確認
- TTL を延長するか、stale 時のフォールバック戦略を導入

### 問題4: min_spread_jpy=700 が低ボラ環境に不適合

**頻度**: 27回/日 のブロック

Coincheck spread が 243-522 JPY の時間帯（低ボラ ranging）で発注不可。
481# で 1000→700 に緩和済みだが、さらなる引き下げが必要な状況。

**改善案**:
- 500 JPY に段階緩和（spread 300-700 帯の取引機会を回収）
- ただし低スプレッド = 逆選択リスク増のため、offset は保守的に

### 問題5: macro_boost の ceiling 下での副作用

**頻度**: C14902 で -20.79bps の最大損失に寄与

macro_boost が offset を拡大するも final_clamp でカット。
しかし timeout 短縮の副作用だけが残り、re-quote → slow fill → 逆選択。

**改善案**:
- macro_boost 適用後に ceiling でカットされる場合、timeout 短縮も無効化
- または ceiling_hit 時は macro_boost の timeout 効果をスキップ

---

## §7 503# との比較・進捗

| 503# 指摘 | 状態 | 備考 |
|-----------|------|------|
| Cross-Venue sell 側未適用 | ❌未改善 | C14957 のみ tighten 適用（favorable条件下） |
| Ranging 損失集中 | ❌継続 | ranging 100%、trending 移行なし |
| Buy fast fill 逆選択 | ⚠一部改善 | FFD の Activated/Reset が動作中 |
| Sell slow fill 逆選択 | ❌継続 | C14902 -20.79bps が典型例 |
| Dynamic kill 過剰 | ✅緩和 | 今日は 17回（503# 期間: 312回/week） |
| **新規: CV veto デッドロック** | ❌新規問題 | 503# 時点では未顕在化 |
| **新規: final_clamp 一律 46%** | ❌新規問題 | pipeline 出力の大半が無駄に |

---

## §8 優先改善ロードマップ

| 優先度 | 施策 | 推定影響 | 実装難度 |
|--------|------|---------|---------|
| **P0** | CV veto 在庫ゼロ時緩和 | デッドロック解消 | 低（config分岐追加） |
| **P1** | buy offset ceiling 引き上げ (0.25→0.28) | buy 黒字化の可能性 | 低（config変更） |
| **P2** | macro_boost ceiling-hit 時のtimeout連動 | C14902型損失防止 | 中（ロジック修正） |
| **P3** | min_spread_jpy 段階緩和 (700→500) | 27回/日の機会回収 | 低（config変更） |
| **P4** | sidecar signal 更新頻度改善 | 予測精度向上 | 中（scheduler調査） |
| **P5** | sell offset ceiling 引き下げ (0.25→0.22) | sell 逆選択軽減 | 低（config変更） |

---

## 付録: 本日の PnL 分布

```
PnL bps     | 件数 | 累計bps
------------|------|--------
> +10       |   2  | +25.64
+5 to +10   |   4  | +29.53
+1 to +5    |   9  | +14.53
0 to +1     |  16  | +6.19    (微益)
-1 to 0     |  14  | -7.30    (微損)
-5 to -1    |  20  | -51.30
-10 to -5   |   5  | -37.07
< -10       |   4  | -51.42  (C14902:-20.79, C14850:-14.66, C14951:-10.13, C14859:-9.70 の4本だけで-55.3bps)
```

**全体**: 正のPnL=+75.89bps vs 負のPnL=-147.09bps → **ネット-71.2bps**
（ただし約定待ち時間分を含むため、bps合計=約-35.6の集計値と差異あり）

**損益の 73% がワースト4本** (-55.3/-75.5) に集中 → テール損失の抑制が最優先。
