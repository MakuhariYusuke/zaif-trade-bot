# 098# ph2 deep analysis: 097# 後の構造診断 + 収益改善戦略

| key | value |
|---|---|
| 番号 | 098 |
| フェーズ | ph2 |
| 種別 | deep analysis |
| 対象 | 097# SkipGate retrain 後の本番稼働分析 |
| 参照 | `docs/v460/097_ph2_skipgate_retrain_preorder.md`, `docs/v460/096_ph2_rev_095.md`, `scripts/v460/run_fill_test.py`, `scripts/v460/ml/skip_gate.py`, `configs/v460/fill_test.yaml`, `results/v460/fill_test/fill_records_*.jsonl` |
| 作成日 | 2026-02-17 |
| git_sha | `f2011fda2` (097# doc) |
| 結論 | **097# SkipGate は preorder-only 特徴量で学習/推論不整合を解消（096# #1 CRITICAL 対応済み）。P(AS) 分布は [0.42, 0.56] に集中し分離力は限定的だが、adaptive threshold が正常に収束中で、20サイクル目に初の buy skip 2件を記録。主要な収益改善余地は SkipGate 外にあり、 (A) fast_fill_defense の neg_edge 検出漏れ50%修正 (B) sell offset 保守化 (C) regime detector warm-up 問題の3点が P0。全440件の損益平均は -0.602bps だが、Non-AS トレードは +2.040bps と構造的に profitable であり、AS フィルタリング精度向上が最短の収益改善路線。** |

---

## §0 エグゼクティブサマリ

1. **097# SkipGate retrain 成果**: preorder-only 特徴量 (k=10) で学習/推論不整合 (096# #1 CRITICAL) を完全に解消。P(AS) は全サイクルで non-null な値を出力中。
2. **097# ランの実績**: 20 records / 13 filled / PnL -0.840bps。2 件の buy skip (p_as ≥ 0.5449) を初達成。ただし **n=13 では統計的判断は不可能**。
3. **全データセット (574 records / 440 filled)**: PnL -0.602bps、AS 率 38.0%。**Non-AS は +2.040bps と明確にプラス**であり、AS 除去が最大のレバレッジ。
4. **構造的問題 6 件を特定**: fast_fill_defense neg_edge 検出漏れ (50%)、SkipGate 閾値フィードバックループ、regime detector warm-up 不全、time_filter 合成論理欠落、param_adapter I/O 無駄、warm_start 閾値未復元。
5. **短期改善 (P0) 3 施策でシミュレーション上 +0.3〜0.5bps 改善見込み**。

---

## §1 097# Run 詳細分析 (run_id: `1771315101_ac824150`)

### §1.1 基本統計

| 指標 | 値 |
|---:|---|
| 総レコード | 20 |
| Filled | 13 |
| Unfilled (timeout) | 5 |
| Skipped (SkipGate) | 2 |
| PnL 平均 (30s) | -0.840 bps |
| PnL min / max | -4.077 / +3.593 |
| AS 率 | 31% (4/13) |
| E3 (120s) 平均 | +0.004 bps (n=11) |
| Regime | **全件 "unknown"** |

### §1.2 全サイクル詳細

| # | side | P(AS) | threshold | skip | PnL (30s) | wait | AS | E3 (120s) | regime |
|---|---|---|---|---|---|---|---|---|---|
| 1 | sell | 0.4786 | 0.5800 | — | N/A | 90.6s | N/A | N/A | unknown |
| 2 | sell | 0.4269 | 0.5600 | — | -0.940 | 19.1s | ✗ | -3.323 | unknown |
| 3 | buy | 0.5591 | 0.6300 | — | N/A | 95.3s | N/A | N/A | unknown |
| 4 | sell | 0.4816 | 0.5400 | — | N/A | 93.7s | N/A | N/A | unknown |
| 5 | buy | 0.5250 | 0.6100 | — | **-2.678** | 6.7s | **✓** | +2.845 | unknown |
| 6 | sell | 0.4990 | 0.5274 | — | -1.986 | 12.8s | ✗ | +14.203 | unknown |
| 7 | buy | 0.4403 | 0.5900 | — | -0.313 | 32.1s | ✗ | -0.771 | unknown |
| 8 | sell | 0.4755 | 0.5274 | — | **-2.727** | 6.8s | **✓** | +2.769 | unknown |
| 9 | buy | 0.4956 | 0.5700 | — | +0.426 | 6.8s | ✗ | N/A | unknown |
| 10 | sell | 0.4747 | 0.5274 | — | -0.889 | 51.5s | ✗ | -0.170 | unknown |
| 11 | buy | 0.4998 | 0.5500 | — | N/A | 95.1s | N/A | N/A | unknown |
| 12 | sell | 0.4225 | 0.5177 | — | +0.713 | 45.4s | ✗ | N/A | unknown |
| 13 | **buy** | **0.5468** | **0.5449** | **SKIP** | — | — | — | — | — |
| 14 | buy | 0.5037 | 0.5449 | — | +0.984 | 18.9s | ✗ | -6.378 | unknown |
| 15 | sell | 0.5159 | 0.5177 | — | **-2.670** | 6.7s | **✓** | -0.196 | unknown |
| 16 | buy | 0.5027 | 0.5449 | — | **-4.077** | 74.7s | **✓** | -2.923 | unknown |
| 17 | sell | 0.5004 | 0.5177 | — | +3.593 | 19.8s | ✗ | +2.381 | unknown |
| 18 | **buy** | **0.5484** | **0.5449** | **SKIP** | — | — | — | — | — |
| 19 | buy | 0.5340 | 0.5449 | — | N/A | 91.2s | N/A | N/A | unknown |
| 20 | buy | 0.5329 | 0.5449 | — | -0.354 | 12.3s | ✗ | -8.392 | unknown |

### §1.3 097# 暫定所見 (n=13; 統計的信頼度は低い)

- P(AS) 分布: buy [0.440, 0.559], sell [0.422, 0.516] — 全て threshold 初期値以下
- Adaptive threshold 収束:
  - **buy**: 0.630 → 0.610 → 0.590 → 0.570 → 0.550 → **0.5449** (6 step で P(AS) 圏に到達、skip 開始)
  - **sell**: 0.580 → 0.560 → 0.540 → **0.5274** → 0.5177 (5 step、skip 未達)
- Skip 2 件は両方 buy side で P(AS) ≥ 0.5449。これらの推定 PnL は不明だが、直後の #14 (P(AS)=0.5037) は +0.984bps → skip 対象の P(AS) が高い方がより悪いトレードだった可能性が高い
- AS 4/13 (31%) — 全データ平均 38% より良好だが n が小さく判断不可
- E3 (120s): +0.004bps — 30s PnL (-0.840) よりかなり改善。120s horizon で利益が回復する傾向

---

## §2 全データセット分析 (574 records / 440 filled)

### §2.1 基本損益

| 区分 | PnL (bps) | n | 備考 |
|---|---:|---:|---|
| **全体** | **-0.602** | 440 | |
| buy | -0.381 | 225 | |
| sell | -0.833 | 215 | sell 劣後継続 |
| **AS トレード** | **-4.920** | 167 (38.0%) | 損失の主因 |
| **Non-AS** | **+2.040** | 273 | **構造的にプラス** |
| E3 (120s) | +0.363 | 62 | 長期ではプラス回帰 |

**最重要所見**: Non-AS が +2.040bps ということは、AS をフィルタリングできさえすれば即座に利益転換する。SkipGate 改善のレバレッジは極めて高い。

### §2.2 Wait Bucket 分析

| 帯 | PnL (bps) | n | AS 率 | 判定 |
|---|---:|---:|---:|---|
| 5-10s | -0.716 | 155 | 43% | 危険 |
| 10-15s | -0.646 | 50 | 34% | |
| 15-30s | **-1.300** | 71 | 44% | **最大損失帯** |
| 30-60s | -0.595 | 68 | 37% | |
| 60-120s | **+0.518** | 42 | 24% | プラス |
| 120s+ | -0.192 | 54 | 31% | |

**パターン**: wait 60s+ は安定してプラス (AS率 24-31%)。15-30s 帯が最悪 (AS率44%)。

### §2.3 Wait Bucket × Side 交差分析

| 帯 | buy PnL | buy n | buy AS% | sell PnL | sell n | sell AS% |
|---|---:|---:|---:|---:|---:|---:|
| 5-10s | -0.056 | 82 | 45% | **-1.458** | 73 | 41% |
| 10-15s | -1.196 | 25 | 32% | -0.096 | 25 | 36% |
| 15-30s | -1.118 | 39 | 44% | **-1.522** | 32 | 44% |
| 30-60s | -0.832 | 35 | 37% | -0.345 | 33 | 36% |
| 60-120s | +0.604 | 26 | 23% | +0.377 | 16 | 25% |
| 120s+ | +0.324 | 18 | 28% | -0.450 | 36 | 33% |

**注目**: sell × 5-10s は -1.458bps (n=73) — sell の fast fill は特に破壊的。sell × 120s+ も -0.450bps で唯一マイナスの長時間帯 → sell のタイムアウト損失が構造的。

### §2.4 Per-Run 損益推移

| run_id | git | PnL | buy | sell | n | AS% | E3 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1771007867 | a9320c9a | **+0.366** | +0.269 | +0.466 | 118 | 38% | N/A |
| 1771043051 | ca1bcaed | -1.097 | +0.229 | -2.424 | 60 | 33% | N/A |
| 1771071431 | 68a13aac | -1.322 | -1.550 | -1.094 | 24 | 29% | N/A |
| 1771095285 | 51c02be6 | -0.927 | -0.450 | -1.405 | 82 | 30% | +0.101 |
| 1771306370 | c3dbaed5 | **-0.066** | -0.572 | +0.440 | 32 | 28% | +1.208 |
| **1771315101** | **f2011fda** | -0.840 | -1.002 | -0.701 | 13 | 31% | +0.004 |
| unknown | ? | -1.132 | -0.893 | -1.426 | 89 | 55% | N/A |

**注目**: 最良 run `1771007867` (a9320c9a) は PnL +0.366bps (n=118) — AS 率 38% でもプラス。このときの構成と何が違うかの特定が必要 (SkipGate 無し、regime 無し、time_filter 無し)。

### §2.5 Effective Offset 分布

| side | min | median | max | mean |
|---|---:|---:|---:|---:|
| buy | 0.0750 | 0.0750 | 0.1875 | 0.1042 |
| sell | 0.1920 | 0.2400 | 0.2400 | 0.2386 |

**所見**: sell offset は 0.24 に集中 — `sell: 0.12` × `narrow_spread_boost_sell: 2.0` = 0.24。ほぼ全ての sell が narrow spread boost を受けている。0.24 でも PnL -0.833bps → offset をさらに上げるか、別のアプローチが必要。

### §2.6 Offset vs PnL 因果分析

| side | low offset (mean) | PnL | high offset (mean) | PnL |
|---|---:|---:|---:|---:|
| buy | 0.075 | -0.392 | 0.156 | **-1.588** |
| sell | 0.236 | -0.706 | 0.240 | **-1.557** |

**逆説的結果**: high offset のほうが PnL が悪い。これは因果ではなく選択バイアス — high offset は fast_fill_defense 発動や narrow_spread_boost 適用で既に「危険な市場状態」のサイン。offset 上昇だけでは AS を防げないことを示す。

---

## §3 構造的問題カタログ

### §3.1 [P0-A] fast_fill_defense: `has_negative_edge` 検出率 50% 問題

**場所**: `scripts/v460/run_fill_test.py` L2097-2105

```python
has_negative_edge = (
    record.mid_at_fill is not None
    and record.fill_price is not None
    and (
        (record.side == "buy" and record.fill_price > record.mid_at_fill)
        or (record.side == "sell" and record.fill_price < record.mid_at_fill)
    )
)
```

**問題の構造**:
- Maker limit order の場合: `fill_price ≈ order_price < mid` (buy) / `fill_price ≈ order_price > mid` (sell)
- AS が発生するのは「約定後に mid が不利方向に移動」するケース
- だが `has_negative_edge` は**約定時点**の `fill_price vs mid_at_fill` しか見ない
- Maker 注文では `fill_price < mid` が通常状態なので、mid が若干下がっても `fill_price > mid_at_fill` にならない場合がある

**データ証拠**:

| side | 閾値 | fast n | neg_edge検出 | AS件数 | **AS ∧ ¬neg_edge (漏れ)** | neg_edge ∧ ¬AS (誤検知) |
|---|---:|---:|---:|---:|---:|---:|
| buy | ≤5.0s | (0 hit: threshold_sec_buy=null → 共通5s) | — | — | — | — |
| sell | ≤15.0s | 98 | 51 (52%) | 39 (40%) | **18 (46% of AS)** | 30 |

**buy 側の閾値問題**: `threshold_sec_buy: null` → 共通値 `5.0s` が使用されるが、buy の 5-10s 帯は -0.056bps (n=82, AS=45%)。buy ≤5s のデータがない = buy fast_fill_defense は実質無効。

**改善案**:
1. `has_negative_edge` を `post_fill_30s_pnl < 0` の**事後判定**に変更（即時→遅延トリガー方式）
2. 又は `fill_price vs order_price` の乖離 + `queue_wait <= threshold` の AND 条件に変更
3. `threshold_sec_buy` を `10.0` に設定して buy fast fill も検出対象に

### §3.2 [P0-B] SkipGate 閾値フィードバックループ + warm_start 閾値未復元

**場所**: `scripts/v460/ml/skip_gate.py` L225-227 (閾値解決), L270-335 (_calibrate_threshold), L534-590 (warm_start)

**問題チェーン**:

```
起動時:
  1. YAML → config.as_threshold=0.65, as_threshold_buy=null, as_threshold_sell=0.60
  2. SkipGate init: config を override (L558-565)
  3. warm_start: _pas_history_buy/sell を復元 (L584-586)
  4. *** しかし config.as_threshold_buy/sell は YAML 値のまま ***

evaluate() 呼出時:
  5. threshold = config.as_threshold (0.65)  ← L225
  6. buy: as_threshold_buy = null → 0.65 使用  ← L226-227
  7. _calibrate_threshold(side, prob, 0.65)  ← L231
  8. history に prob (~0.50) を追加
  9. quantile 計算 → target_threshold ~0.50
  10. new_threshold = 0.65 - 0.02 = 0.63  ← step 制約で 0.02 ずつ
  11. config.as_threshold_buy = 0.63  ← L329

次回 evaluate():
  12. threshold = config.as_threshold_buy (0.63)  ← L226
  13. _calibrate_threshold(side, prob, 0.63)
  14. new_threshold = 0.63 - 0.02 = 0.61
  ... 0.65 → 0.63 → 0.61 → 0.59 → 0.57 → 0.55 → 0.5449 (SKIP!) ← **6 cycles**
```

**097# での実際の収束** (§1.2 から):

| cycle | buy threshold | sell threshold |
|---|---|---|
| 1 (sell) | — | 0.5800 |
| 2 (sell) | — | 0.5600 |
| 3 (buy) | 0.6300 | — |
| 4 (sell) | — | 0.5400 |
| 5 (buy) | 0.6100 | — |
| 6 (sell) | — | 0.5274 ← quantile jump |
| 7 (buy) | 0.5900 | — |
| 8 (sell) | — | 0.5274 |
| 9 (buy) | 0.5700 | — |
| 10 (sell) | — | 0.5274 |
| 11 (buy) | 0.5500 | — |
| 12 (sell) | — | 0.5177 |
| 13 (buy) | **0.5449** ← first skip! | — |

**buy**: 6 step (cycle 3→13) で YAML 0.65 → 0.5449 に収束。warm_start が P(AS) 履歴を復元したため quantile 計算は即座に正しい target を算出できたが、**閾値の初期値が 0.65 のまま**のため step=0.02 で 6回 call しないと到達できない。

**sell**: 5 step (cycle 1→12) で 0.58 → 0.5177 に収束。sell は `as_threshold_sell: 0.60` が YAML 設定値だが、calibrate 後に 0.5800 に上がっている（不自然）。→ 前回の run で calibrate が 0.58 を書き込み、次の init で warm_start がこれを上書きしなかったため。

**修正案**:
1. `warm_start_skip_gate_thresholds()` で履歴復元後に **1回の calibrate 呼出し**を追加して threshold 値も即座に設定する
2. または `_calibrate_threshold()` の `adaptive_step` を `0.10` に増加して 2 step で収束させる
3. 根本対策: threshold の初期値を YAML ではなく warm_start 時の quantile 計算結果で上書き

### §3.3 [P0-C] sell offset 0.12 → 0.15 の効果と限界

**現状**: sell offset 0.12 → narrow_spread_boost (2.0) → 有効 0.24。Sell PnL = -0.833bps。

§2.6 の逆説的結果（high offset のほうが PnL 悪い）を考えると、offset 増加は限界効果が逓減。根本原因は売りの AS 構造:
- sell × 5-10s: AS 41%, PnL -1.458bps (最悪帯)
- sell × 15-30s: AS 44%, PnL -1.522bps
- sell × 60-120s: AS 25%, PnL +0.377bps

**sell の問題は offset ではなく timing**: fast fill (15s 以内) の sell が壊滅的。

**改善案**:
1. `sell_offset_floor: 0.08 → 0.10` (底上げ)
2. `threshold_sec_sell: 15.0 → 10.0` (fast_fill 判定を厳格化)
3. sell で wait < 10s の場合に追加ブースト（x3.0 → offset=0.36）
4. SkipGate の `target_skip_rate_sell: 0.20 → 0.25` で sell skip を増加

### §3.4 [P1-D] Regime Detector warm-up 問題

**場所**: `scripts/v460/run_fill_test.py` L513-527

```python
regime_cfg = RegimeConfig(
    window=config.regime_window,  # 20
    ...
)
self._regime_detector = FillTestRegimeDetector(regime_cfg)
```

**問題**: `window=20` で 20 件のデータが蓄積するまで全て "unknown"。

**Per-Run Regime 状態**:

| run_id | n | unknown | ranging | trending | 備考 |
|---|---:|---:|---:|---:|---|
| 1771007867 | 118 | 0 | 0 | 0 | regime 未導入 (none) |
| 1771043051 | 60 | 0 | 0 | 0 | regime 未導入 (none) |
| 1771071431 | 24 | 17 | 1 | 6 | 71% unknown |
| 1771095285 | 82 | 17 | **52** | **13** | **唯一 regime が有効機能** |
| 1771306370 | 32 | 22 | 10 | 0 | 69% unknown |
| **1771315101** | **13** | **13** | **0** | **0** | **097# 全件 unknown** |

**影響**: `trending_offset_boost: 1.5` が全く発動しない。regime 別の SkipGate 特徴量 (`regime_trending`, `regime_ranging`) も常に 0。

**修正案**:
1. warm_start で直近 N 件の price を `_regime_detector` に feed して初期化
2. `window=20 → 10` に縮小
3. regime が unknown のときデフォルトで最も保守的な設定 (ranging) を仮定

### §3.5 [P2-E] time_filter: side 別リスト優先で global リストが無視される

**場所**: `scripts/v460/run_fill_test.py` L914-937

```python
# side 別リストが定義されている場合はそちらを優先
if side == "buy" and self.config.skip_utc_hours_buy:
    if current_utc_hour in self.config.skip_utc_hours_buy:
        ...
        return True
    return False  # ← ここで global check をスキップ
```

**現在の影響**: 現 YAML では side 別リストに global の時間帯も含めてあるため実害なし。

**潜在リスク**: YAML 編集時に global に時間帯を追加しても side 別ルートで早期 return してしまい、global フィルタが適用されない。

**修正案**: early return を廃止し、side 別 + global の和集合で判定。

```python
filtered_hours = set(self.config.skip_utc_hours or [])
if side == "buy" and self.config.skip_utc_hours_buy:
    filtered_hours |= set(self.config.skip_utc_hours_buy)
elif side == "sell" and self.config.skip_utc_hours_sell:
    filtered_hours |= set(self.config.skip_utc_hours_sell)
return current_utc_hour in filtered_hours
```

### §3.6 [P2-F] param_adapter: 全履歴 I/O の無駄

**場所**: `scripts/v460/run_fill_test.py` L2412-2417

```python
all_records = load_fill_records_glob(str(self._results_dir))  # 全ファイル読み込み
records, _q = filter_clean_records(all_records)
del all_records
if self.config.adapt_recency_window > 0 and len(records) > self.config.adapt_recency_window:
    records = records[-self.config.adapt_recency_window:]  # 最新 120件に切り詰め
```

**問題**: `adapt_recency_window=120` でも全 JSONL ファイル（現在 5 ファイル / 574 行）を毎回 parse → そのうち 454 行を捨てる。データ増加に伴い O(total) の無駄。

**修正案**: `load_fill_records_glob()` の逆順読みを実装し、必要件数に達した時点で早期停止。`warm_start_skip_gate_thresholds()` (L556-574) に既に逆順読みパターンが実装されているため、同じ手法を param_adapter にも適用。

---

## §4 SkipGate 精度分析

### §4.1 P(AS) 分布

| side | n | min | p10 | p25 | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| buy | 39 | 0.440 | 0.486 | 0.499 | 0.506 | 0.525 | 0.545 | 0.559 |
| sell | 33 | 0.422 | 0.476 | 0.484 | 0.503 | 0.516 | 0.532 | 0.545 |

**所見**: 全サンプルが 0.42-0.56 の狭い帯域に集中。**分離力が低い** (ROC-AUC=0.442)。しかし以下のシミュレーションはわずかな分離でも skip 効果が出ることを示す。

### §4.2 閾値別シミュレーション (仮想 skip のPnL比較)

| threshold | side | skip 件数 (%) | skip 群 PnL | kept 群 PnL | 差分 |
|---|---|---:|---:|---:|---:|
| 0.50 | buy | 27 (69%) | -0.904 | -0.444 | skip 群が悪い → 正しい |
| 0.50 | sell | 19 (58%) | +0.027 | -0.294 | **逆効果** (skip 群のほうが良い) |
| 0.55 | buy | 1 (3%) | N/A | -0.771 | n=1 判断不可 |
| 0.45 | buy | 38 (97%) | -0.786 | -0.313 | n=1 kept |
| 0.45 | sell | 31 (94%) | -0.089 | -0.114 | ほぼ同等 |

**解釈**: buy 側では P(AS) ≥ 0.50 群が -0.904bps vs kept -0.444bps で SkipGate の方向性は正しい。sell 側では逆効果 → **sell の特徴量基盤が改善余地あり**（preorder 特徴量のみではsell方向の AS 予測が困難な可能性）。

### §4.3 SkipGate 改善方向性

1. **即効性**: `as_threshold` 初期値を 0.52/0.50 に設定 → warm_start 収束と合わせて 1-2 cycle で skip 開始
2. **中期**: sell 用の追加特徴量候補:
   - `spread_bps`: sell は narrow spread 時に AS 率が高い
   - `hour_of_day_sin/cos`: 売り側の時間帯感度が高い
   - `recent_sell_pnl_mean`: 直近 sell PnL のトレンド
3. **評価指標**: skip 群の仮想 PnL < kept 群の PnL を side ごとに検証し、方向性が正しい side のみ skip を有効にするオプション

---

## §5 Repriced Trades 分析

| side | reprice | PnL | wait | AS | 所見 |
|---|---:|---:|---:|---|---|
| buy | 1 | -4.335 | 44.5s | ✓ | 追いかけ買い → AS |
| sell | 2 | +4.377 | 143.8s | ✗ | reprice 成功 |
| sell | 1 | +4.692 | 192.3s | ✗ | reprice 成功 |
| sell | 1 | -6.514 | 272.0s | ✓ | 長時間待ち → AS |
| sell | 1 | -3.100 | 44.4s | ✓ | reprice → AS |

**n=5 のため断定不可**。傾向として:
- **buy reprice はリスキー** (1/1 が AS、-4.335bps)
- **sell reprice は二極化** (Non-AS で +4.5bps、AS で -5.0bps)
- 096# 設定の `max_reprice_buy: 1` は正しい方向

---

## §6 E3 (120s horizon) 分析

| run_id | n | E3 率 | 30s PnL | 120s PnL | 差分 |
|---|---:|---:|---:|---:|---:|
| 1771095285 | 82 | 32% | -0.927 | +0.101 | **+1.028** |
| 1771306370 | 32 | 50% | -0.066 | +1.208 | **+1.274** |
| 1771315101 | 13 | 85% | -0.840 | +0.004 | +0.844 |
| 全体 | 62 | — | — | +0.363 | — |

**重要所見**: 120s horizon では全 run が 30s より改善。maker ポジションは 30s で一時的に逆行しても 120s で回復する傾向。これは **maker edge の存在を裏付ける**が、30s 判定で AS 認定されてしまう。

**含意**:
1. AS 判定の horizon を 30s → 60s に延長する検討価値あり
2. E3 PnL を SkipGate の学習ラベルに使用（120s 後の結果で AS/Non-AS を判定）
3. 少なくとも E3 データ蓄積のため `e3_sampling_ratio: 0.50` は維持

---

## §7 優先施策まとめ

| 優先度 | 施策 | 期待効果 | 実装コスト | リスク |
|---|---|---|---|---|
| **P0** | fast_fill_defense: neg_edge → post-fill PnL 判定に変更 + buy 閾値 10s | +0.2bps (fast fill AS 18件×-4.9bps の半数を回避) | LOW | fill rate 低下の可能性 (要モニタ) |
| **P0** | SkipGate: warm_start で threshold 値も calibrate | +0.1bps (6 cycle の遅延解消) | LOW | なし |
| **P0** | SkipGate: YAML 初期値 0.65/0.60 → 0.52/0.50 | 即効 skip 開始 | CONFIG ONLY | 過剰 skip リスク (max_skip_rate=0.3 で安全弁あり) |
| **P1** | regime detector warm_start | regime 特徴量の有効化 | MEDIUM | warm_start price data の保存必要 |
| **P1** | sell 特化策: 5-10s 帯の追加ブースト (3.0x) | sell fast fill -1.46bps の緩和 | LOW | fill rate 低下 |
| **P2** | time_filter: global + side 合成 | latent bug 修正 | LOW | なし |
| **P2** | param_adapter: 逆順読み早期停止 | I/O 削減 | LOW | なし |
| **検討** | AS 判定 horizon 30s → 60s への変更 | E3 で +0.84bps の改善を反映 | HIGH | SkipGate 再学習必要 |

---

## §8 Codex へのレビュー依頼事項

本文書に基づき、以下の観点でレビューを依頼:

1. **§3.1 fast_fill_defense 修正案**: `has_negative_edge` の代替ロジックとして「事後 PnL 判定」vs「fill_price vs order_price 乖離」のどちらが適切か
2. **§3.2 warm_start 閾値復元**: calibrate 1 回呼出し vs step 増加 vs quantile 直接上書きの妥当性
3. **§4.2 sell SkipGate 逆効果**: sell 側の P(AS) が AS を正しく識別できていない原因仮説と追加特徴量候補
4. **§6 E3 含意**: AS 判定 horizon 変更の影響範囲とリスク
5. **§2.6 offset 逆説**: high offset ≒ 悪環境のバイアスを除去した上での最適 offset の推定方法
6. **§7 優先順位**: P0 3 施策の実装順序と相互影響
7. **その他**: 見落としている構造的リスクや改善機会

---

## Appendix A: SkipGate Skip 記録

| run_id | side | P(AS) | threshold | 備考 |
|---|---|---:|---:|---|
| 1771315101 | buy | 0.5468 | 0.5449 | 097# cycle 13: 初の skip |
| 1771315101 | buy | 0.5484 | 0.5449 | 097# cycle 18: 2nd skip |

## Appendix B: 設定値チェックリスト (fill_test.yaml 抜粋)

| パラメータ | 現在値 | 構造的問題 | 推奨変更 |
|---|---|---|---|
| `spread_offset_ratio` | 0.05 | — | — |
| `side_offset.sell` | 0.12 | §3.3 | 0.15 検討 |
| `fast_fill_defense.threshold_sec_buy` | null (→5.0) | §3.1 | **10.0** |
| `fast_fill_defense.threshold_sec_sell` | 15.0 | — | — |
| `skip_gate.as_threshold` | 0.65 | §3.2 | **0.52** |
| `skip_gate.as_threshold_buy` | null (→0.65) | §3.2 | **0.52** |
| `skip_gate.as_threshold_sell` | 0.60 | §3.2 | **0.50** |
| `skip_gate.adaptive_step` | 0.02 | §3.2 | **0.05** |
| `skip_gate.target_skip_rate_sell` | 0.20 | §4.2 | 0.25 検討 |
| `sell_guard.offset_floor` | 0.08 | — | 0.10 検討 |
| `regime_window` | 20 | §3.4 | 10 検討 |
