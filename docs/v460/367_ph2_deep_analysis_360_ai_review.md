# 367# ph2 深堀り分析 — 360# AIレビュー準備

**日付**: 2026-03-10
**フェーズ**: ph2 G1.1-exec
**対象**: fill test 360#→364# 後の状態評価 + AIレビュー準備
**ベース文書**: `docs/v460/360_ph2_rpt_fill_test_analysis.md`

---

## §1 360# 改善提案の実施状況と効果検証

### 1.1 TUNE 実施結果

| ID | 内容 | コミット | 結果 | 検証データ |
|:--:|------|---------|:----:|-----------|
| TUNE-1 | `forced_buy_delay` 撤廃 | 348# | ✅ **成功** | 03-09: 0件, 03-10: 0件 (03-08: 50件は pre-348# SHA) |
| TUNE-2 | `per_side_dd_halt` -30→-50bps, cycles 15→10 | 364# `22a4fc583` | ✅ **成功** | 03-08~10: 全日 0件 (完全抑制) |
| TUNE-3 | `sell_dynamic_kill` -0.3→-0.5bps + regime緩和 | 364# `edaab98e0` | ✅ **劇的成功** | 03-08: 220kills → 03-09: 140 → **03-10: 0kills** |
| TUNE-4 | BDK threshold 緩和 | 364# **SKIP** | ❌ **判断ミス** | 03-10: BDK=21kills, **#1 cancel reason (36.8%)** |

### 1.2 TUNE-3 効果の定量分析

```
SDK kills推移:
  03-08 (pre-364#):  220 kills / 520 cancels = 42.3% ← 最大キャンセル理由
  03-09 (mixed SHA):  140 kills / 415 cancels = 33.7% ← 改善中
  03-10 (364# SHA):     0 kills /  57 cancels =  0.0% ← 完全解消 ✅

YAML変更:
  threshold_bps: -0.3 → -0.5
  regime_thresholds.trending_up: -0.3 → -0.5
  regime_thresholds.ranging: -0.5 → -0.7
  inv_relaxation.max_bps: 0.3 → 0.5
```

**TUNE-3 は 360# で最もインパクトの高い改善提案であり、完全に成功した。**

### 1.3 TUNE-4 SKIP の再検証 (猜疑的視点)

364# コミットメッセージ: `SHA 819ec73b 0件BDK cancel, 変更不要`

**しかしこの判断は以下の理由で問題**:

1. **時点バイアス**: SHA `819ec73b` での BDK=0 は、その時点で SDK が大量にキャンセルしていたため BDK に到達する前に SDK で kill されていた可能性が高い
2. **TUNE-3 の玉突き効果**: SDK kill を解消 → より多くの注文が BDK チェックに到達 → BDK kill 顕在化
3. **実データ証拠**: 03-10 (SDK=0, BDK=21) は TUNE-3 成功の裏返しとして BDK が表面化

### 1.4 OPS 実施状況

| ID | 内容 | 360# 推奨 | 実施有無 | 効果 |
|:--:|------|---------|:------:|------|
| OPS-1 | atexit RSS ダンプ | Week 1 | ⏳ 未確認 | crash 原因特定に必要 |
| OPS-2 | health_monitor 頻度UP | 5min | ⏳ 未確認 | OOM 事前検知 |
| OPS-4 | restart.lock stale 延長 | 5min | ⏳ 未確認 | dual-spawn 防止 |
| OPS-5 | Task Scheduler IgnoreNew | 即時 | ⏳ 未確認 | lock_conflict 解消 |

**run_id 分析による crash 状況**:
```
03-08: 3 runs (2 回 crash/restart) — 09:00~00:00, 00:12~05:05, 05:13~08:59
03-09: 4 runs (3 回 crash/restart) — 09:01~09:41, 09:51~17:52, 18:00~03:09, 03:18~08:58
03-10: 1 run  (0 回 crash/restart) — 09:00~14:53 (ただし 6h 分のみ)

SHA遷移:
  03-08: 5 SHA混在 (bb59fb1, fea7911, e5d4937, d4db827, eb24cf4)
  03-09: 5 SHA混在 (06f0ba2, 22a4fc5, 0d22298, 819ec73, d4db827)
  03-10: 単一SHA (22a4fc583078 = 364# TUNE-2/3)
```

03-10 は **crash なしで安定稼働** (6h 時点)。ただし 72h 連続稼働 (K2) の判定には不十分。

---

## §2 現状の構造的問題分析

### 2.1 Cancel Reason トポロジーの時間変化

```
                03-08          03-09          03-10
─────────────────────────────────────────────────────
#1 cancel    SDK (42.3%)    SDK (33.7%)    BDK (36.8%) ← 首位交代
#2 cancel    SG  (12.1%)    SG  (23.4%)    SG  (24.6%)
#3 cancel    FBD  (9.6%)    STN (17.1%)    STN (15.8%)
#4 cancel    STN  (8.3%)    SAD  (8.4%)    SAD (14.0%)
#5 cancel    BDK  (7.1%)    BDK  (8.2%)    DDH  (5.3%)
─────────────────────────────────────────────────────
Fill Rate     21.9%          31.2%          32.1%
```

**重要な構造変化**:
- SDK が解消 → fill rate は 21.9% → 32.1% へ改善 (+10.2pp)
- しかし BDK が台頭し、K1 60% にはまだ 28pp の gap
- skip_gate (SG) が安定して #2 — モデル品質の問題
- stale_adverse_drift (SAD) が増加傾向 (5.8% → 14.0%)

### 2.2 BDK Kill の根本原因

**BDK config** (`configs/v460/fill_test.yaml` L618-631):
```yaml
buy_dynamic_kill:
  threshold_bps: -0.8     # default (ranging で使用)
  regime_thresholds:
    trending_down: -0.5   # 厳しい (逆張り)
    trending_up: -1.5     # 寛容 (順張り)
    high_vol: -0.5        # 厳しい
  # ranging は明示なし → default -0.8 を使用
buy_dynamic_kill_inv_relaxation:
  scale: 0.5
  max_bps: 0.3            # effective: -0.8 - 0.3 = -1.1 (最大緩和時)
```

03-10 BDK 分布: **ranging=14, trending_up=7** (全て buy 側)

**問題**: ranging regime で threshold=-0.8bps は、SDK の regime_thresholds.ranging=-0.7bps と比較して厳しい。SDK で -0.3→-0.5→-0.7 と段階的に緩和した実績を考えると、BDK.ranging にも明示的な閾値設定が必要。

**EWMA 状態推定**: BDK kill が集中発生 (21件 / 84レコード = 25%) → EWMA が threshold 付近を推移し、kill/resume を繰り返している可能性。`ewma_time_decay_tau_sec: 600` (半減期≈7分) で自然回復はするが、根本的にレジーム ranging での実績 PnL が -0.8bps 付近にあることを示唆。

### 2.3 Skip Gate モデルの深刻な calibration drift

```
Skip Gate Score分布 (03-10):
  FILLED 記録: n=27, min=-5.79, max=3.40, mean=+0.14
  SKIPPED 記録: n=14, min=-5.97, max=-1.11, mean=-3.07

問題点:
  (1) sg=-5.79 で FILLED — これはモデルが「高確率で逆選択」と判定したのに通過
  (2) sell 側で sg < -2.0 の FILLED が日常的: 03-08=28件, 03-09=40件, 03-10=4件
  (3) 全て sell 側 — buy 側では発生しない
```

**根本原因仮説**:
1. **max_skip_rate safety valve**: 直近20件の skip 率が上限を超えると force-pass する。sell 側の SG score が全般的に低いため、多くの sell 注文が force-pass で通過している可能性
2. **sell 側モデルの訓練データ偏り**: SDK kill 期間中 (03-08~09) に sell 側 fill が少なく、モデル用の学習サンプルが不足→calibration drift
3. **adaptive_threshold の暴走**: 目標 skip 率に合わせて閾値が自動調整 → score が全般的に低い sell 側で閾値が下がりすぎ

**影響**: SG score < -2.0 で fill された注文の PnL は全て `N/A` (post_fill 計測なし) → 実際の損益影響が不明

### 2.4 post_fill PnL 計測の完全欠落

```
03-10 post_fill PnL coverage:
  30s:  0/27 (0%)
  60s:  0/27 (0%)
  120s: 0/27 (0%)
```

**これは 360# §1.3 で指摘されていた問題が悪化している**。360# 時点では ~50% だった coverage が 0% に低下。

**原因候補**:
- fill 後の mid_price 取得タスクがクラッシュ後に再起動されない
- 364# SHA での何らかのリグレッション
- asyncio タスク漏れ (OOM の一因にもなりうる)

**影響**: 
- EWMA のフィードバックが入らない → dynamic_kill/skip_gate の判定精度低下
- online_monitor の `pass_mean_pnl` が正しく計算されない → DEGRADED 状態の信頼性低下

### 2.5 Reprice 機構の休眠

全データ (03-08~10) で `reprice_count > 0` = 0件。

360# では言及されていないが、reprice は timeout 削減の有力手段。板の移動に追従して注文価格を更新することで、fill probability を高められる。

---

## §3 Online Monitor — DEGRADED 状態の分析

```
最新 (2026-03-10 14:18):
  n=100 (pass=63, skip=37)
  pass_mean_pnl = -1.090 bps   [DEGRADED: threshold=-0.3bps]
  pass_win_rate = 39.7%
  skip_precision = 100.0%

  buy: n=47 (pass=31, skip=16, skip_rate=34.0%), pass_pnl=-1.059bps, win_rate=35.5%
  sell: n=53 (pass=32, skip=21, skip_rate=39.6%), pass_pnl=-1.120bps, win_rate=43.8%

時系列推移:
  03-09 19:50: pass_mean_pnl=+1.724bps   [HEALTHY]
  03-10 01:00: pass_mean_pnl=-0.340bps    [DEGRADED]  ← HEALTHY→DEGRADED 遷移
  03-10 03:13: pass_mean_pnl=-0.860bps    [DEGRADED]
  03-10 14:18: pass_mean_pnl=-1.090bps    [DEGRADED]  ← 悪化継続
```

**重要**: pass_mean_pnl は post_fill PnL に依存するが、03-10 は coverage=0%。
→ DEGRADED 判定自体の信頼性が疑わしい。post_fill PnL 欠落が原因で、
   過去の負の PnL のみが残存している可能性。

---

## §4 改善提案 — 優先度再評価

360# の §6 改善提案マトリクスを、03-10 のデータに基づき再評価。

### 4.1 最優先 (Critical Path)

| # | 内容 | 理由 | 工数 | K1 寄与 |
|:--:|------|------|:----:|:------:|
| **FIX-0** | post_fill PnL 計測修復 | 全判定ロジックの基盤。0% coverage は致命的 | 1-2h | 間接的 (判定精度向上) |
| **TUNE-4R** | BDK ranging 閾値追加 `-1.0` | 03-10 #1障壁。SDK TUNE-3 と同様のアプローチ | 5min | +6-9pp |
| **SG-1** | Skip Gate sell 側 calibration 修正 | sg<-2.0 で fill=逆選択リスク | 2-3h | +3-5pp |

### 4.2 高優先 (Week 1)

| # | 内容 | 理由 | 工数 |
|:--:|------|------|:----:|
| **OPS-5** | Task Scheduler IgnoreNew | dual-spawn 根本解決 | 5min |
| **OPS-1** | atexit RSS ダンプ | crash 原因特定の前提条件 | 30min |
| **SAD-1** | `stale_adverse_drift` 閾値見直し | 14% に増加、timeout の先行指標 | 30min |

### 4.3 中期 (Week 2)

| # | 内容 | 理由 |
|:--:|------|------|
| **H1** | Reprice 機構の有効化 | timeout + fill rate 改善 |
| **GATE-1** | K1 閾値 60% → 40% + K4 (PnL≥0) | 360# §5.4 の提案 |

### 4.4 K1 達成シミュレーション (更新版)

```
現状 (03-10): 32.1%

Step 1: TUNE-4R (BDK ranging 緩和)
  → BDK 21件のうち ranging 14件 解消 → 推定 41/84 = 48.8%

Step 2: SG-1 (Skip Gate sell calibration)
  → SG 14件のうち sell 7件 解消 → 推定 48/84 = 57.1%

Step 3: SAD-1 (stale_adverse_drift 閾値緩和)
  → SAD 8件のうち 4件 解消 → 推定 52/84 = 61.9%  ← K1 60% 達成 ✅

Step 4: FIX-0 (post_fill 修復) → 判定精度向上で間接的改善
```

**結論**: TUNE-4R + SG-1 + SAD-1 の 3段改善で K1 60% 達成が理論的に可能。
ただし GATE-1 (K1 40% → +K4) の方が現実的かつリスクが低い。

---

## §5 AI レビューチェックリスト (360# §10 更新版)

### 5.1 データ整合性

| # | チェック | 結果 | 根拠 |
|:--:|---------|:----:|------|
| D1 | 360# TUNE-3 効果: SDK 0件 | ✅ | 03-10 fill_records で確認 |
| D2 | TUNE-4 SKIP の副作用: BDK 顕在化 | ✅ | TUNE-3 で SDK 解消 → BDK 表出 |
| D3 | per_side_dd_halt = 0件 (TUNE-2) | ✅ | 03-08~10 全日 0件 |
| D4 | forced_buy_delay = 0件 (TUNE-1) | ✅ | 03-09~10 で 0件 |
| D5 | post_fill PnL coverage = 0% | ⚠️ | 360# 時点 ~50% → 0% に悪化 |
| D6 | online_monitor DEGRADED 信頼性 | ⚠️ | PnL coverage 0% での判定は不正確 |

### 5.2 分析の論理的一貫性

| # | チェック | 結果 | 備考 |
|:--:|---------|:----:|------|
| L1 | BDK 顕在化の因果推論 | ✅ | SDK 解消 → BDK 到達率上昇 (gates の直列評価) |
| L2 | SG sell 側 anomaly の説明 | ⚠️ | max_skip_rate force-pass 仮説は要実証 |
| L3 | K1 シミュレーションの独立性前提 | ⚠️ | cancel 理由間の相互影響は未モデル化 |
| L4 | DEGRADED と PnL coverage の因果 | ⚠️ | coverage 0% → DEGRADED は相関だが因果不明 |

### 5.3 改善提案の妥当性

| # | チェック | 結果 | 備考 |
|:--:|---------|:----:|------|
| P1 | FIX-0 (post_fill 修復) の実装可能性 | ✅ | 既存コードの asyncio タスク調査で特定可能 |
| P2 | TUNE-4R の SDK 移行類推 | ✅ | -0.3→-0.5 で SDK 完全解消の実績があり、BDK にも適用妥当 |
| P3 | SG-1 の calibration drift 対策 | ⚠️ | 根本原因が max_skip_rate か training data か未特定 |
| P4 | GATE-1 vs 実改善のトレードオフ | ✅ | 360# §5.4 の猜疑的視点が有効 |

### 5.4 文書品質 (360# 原文)

| # | チェック | 結果 | 備考 |
|:--:|---------|:----:|------|
| Q1 | §3 SDK 分析の予測的中率 | ✅ | TUNE-3 で SDK 完全解消 — 分析は正確 |
| Q2 | §5.3 K1 シナリオの更新必要性 | ✅ | 03-10 データで再計算完了 (本レポート §4.4) |
| Q3 | §6 実施順序の妥当性 | ⚠️ | FIX-0 (post_fill 修復) が 360# に含まれていない — 追加必要 |
| Q4 | TUNE-4 SKIP 判断の反省 | ✅ | 時点バイアスの教訓 (§1.3) |

---

## §6 次のアクション (更新版)

```
即時:   FIX-0 (post_fill PnL 修復) — 判定基盤の復旧
        OPS-5 (Task Scheduler) — 5min
Week 1: TUNE-4R (BDK ranging 閾値 -1.0) — 5min YAML 変更
        OPS-1 (atexit hook) — 30min
        SG-1 調査 (max_skip_rate / adaptive_threshold ログ追加)
Week 2: SAD-1 (stale_adverse_drift 閾値見直し)
        72h 連続稼働 K2 再計測
        K1 再計測 → GATE-1 要否判定
```

---

## §7 360# からの key takeaway

1. **TUNE-3 は完全成功**: SDK kills 220→0。段階的閾値緩和 + regime 別設計が有効
2. **TUNE-4 SKIP は判断ミス**: 時点バイアスに起因。gates 直列評価では上流 kill 解消が下流 kill を顕在化させる
3. **post_fill PnL 0% は致命的**: EWMA, online_monitor, skip_gate 全ての判定が空回りしている
4. **K1 60% の厳しい現実は変わらず**: 360# §5.4 の猜疑的視点 (GATE-1) が現実的
5. **fill rate 改善トレンドは positive**: 21.9% → 31.2% → 32.1% — TUNE-2/3 の効果は確実

---

*本レポートは 360# の AI レビュー (§10) を更新し、03-10 データでの cross-reference を実施したものである。*
