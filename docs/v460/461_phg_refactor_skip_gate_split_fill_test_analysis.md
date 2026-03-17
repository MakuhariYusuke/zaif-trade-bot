# 461# skip_gate_evaluator Mixin 分割 + MAX LINES + Fill Test 10 日間深堀り分析

> **種別**: refactor / rpt  
> **フェーズ**: phg (フェーズ横断)  
> **日付**: 2026-03-17 〜 03-18  
> **前提**: 460# offset pipeline 抽出完了後

---

## 概要

4 つの作業を実施:

1. **skip_gate_evaluator.py Mixin 分割** (1362→866 行)
2. **MAX LINES 宣言追加** (3 ファイル)
3. **Fill Test 5 日間ログ分析** (初版)
4. **Fill Test 10 日間深堀り分析** — 再起動前後比較 + 設計面・市場理論面考察

---

## 1. skip_gate_evaluator.py 分割

### 背景

`SkipGateEvaluator` は 1362 行に達し、model loading/hot-reload (~300 行) と ev_weighted 判定ロジック (~210 行) が evaluate() 本体と混在。既存の Mixin パターン (322#/323#/325#/332#/460#) に倣い分割。

### 抽出結果

| 新ファイル | クラス | 行数 | 責務 |
|---|---|---|---|
| `skip_gate_model_loader.py` | `SkipGateModelLoaderMixin` | ~300 | モデルパス解決, ロード, config overrides, warm_start, calibrator注入, side/alt モデル, hot-reload |
| `skip_gate_ev_weighted.py` | `SkipGateEvWeightedMixin` | ~210 | ev_weighted 統合判定 (188#/190#/193#), offset 変換 |

### クラス継承

```python
class SkipGateEvaluator(SkipGateModelLoaderMixin, SkipGateEvWeightedMixin):
    """MAX LINES: 900"""
```

### 移動メソッド一覧

**SkipGateModelLoaderMixin**:
- `_resolve_model_path`, `_read_model_hash`, `_load_gate_from_path`
- `_apply_config_overrides`, `_apply_warm_start`, `_inject_calibrator`
- `_load_side_models`, `_load_alt_models`
- `_check_and_reload_model`, `_check_and_reload_side_models`

**SkipGateEvWeightedMixin**:
- `_try_ev_weighted_decision`, `_ev_weighted_as_offset`

### テスト修正

5 テストファイルでソース読み取りパスを更新:
- `_fill_test_source.py`: `SKIP_GATE_MODEL_LOADER` パス定数追加
- `test_141_side_specific_models.py`: logger パッチパス変更
- `test_143_regime_utilization.py`: ソース読み取り先変更
- `test_139_review_fixes.py`: hot_reload ソース読み取り先変更
- `test_255_getattr_bare_except_cleanup.py`: hot_reload ソース読み取り先変更

---

## 2. MAX LINES 宣言追加

| ファイル | 現在行数 | MAX LINES |
|---|---|---|
| `skip_gate_evaluator.py` | 866 | 900 |
| `cycle_gate_aggregator.py` | 733 | 800 |
| `fill_config_parser.py` | 1024 | 1100 |

---

## 3. Fill Test 5 日間ログ分析 (2026-03-13 〜 2026-03-17)

### 3.1 基本統計 (3/17 最新)

| 指標 | 値 |
|---|---|
| 総レコード | 506 |
| Fill (約定) | 83 (16.4%) |
| Cancel | 423 (83.6%) |

### 3.2 Fill Rate 推移

```
3/13: 28% → 3/14: 26% → 3/15: 17% → 3/16: 14% → 3/17: 16%
```

低下傾向。`ranging_low_vol_skip` が日々増加し 3/17 で 51.3% を占有。

### 3.3 Cancel Reason 内訳 (3/17)

| Reason | 割合 |
|---|---|
| ranging_low_vol_skip | 51.3% |
| no_feasible_quote | 13.9% |
| spread_too_narrow | 9.2% |
| timeout | 9.0% |
| skip_gate | 4.3% |
| sell_dynamic_kill | 3.5% |
| その他 | 8.8% |

### 3.4 Adverse Selection

| 指標 | Buy | Sell | 全体 |
|---|---|---|---|
| AS Rate (processed) | 24% | 39% | 31.3% |
| AS Rate (raw 50.6%) | — | — | 50.6% |

Sell 側が buy の 1.6 倍。

### 3.5 PnL (bps)

| Horizon | Mean | Median |
|---|---|---|
| 30s | -1.12 | -0.34 |
| 60s | -1.91 | -0.74 |
| 120s | -1.82 | -0.59 |

全ホライズンで平均マイナス。

### 3.6 異常検出

- **route_to_kill_deadlock**: 0→43→13→53→0 (5 日間)。3/14-3/16 で PID 特有に発生、3/17 で解消済み (421# final clamp 修正後 SHA)
- **status_unknown_fast**: 6→0→0→3→8。増加傾向。`pending_reconciliation` と相関
- **early_exit**: 全 5 日間で 0 件。実質トリガーされない設定

### 3.7 EV Score と逆選択の関係

| グループ | Mean EV Score |
|---|---|
| 逆選択あり | 0.82 |
| 逆選択なし | 1.44 |

EV score が低いトレードで逆選択が集中。

### 3.8 改善提案

| 優先度 | 施策 |
|---|---|
| P0 | `status_unknown_fast` 増加の原因調査 + reconciliation 改善 |
| P1 | `ranging_low_vol_skip` 閾値の見直し (fill rate 低下の主因) |
| P1 | Sell 側 AS 防御の強化 (39% は高い) |
| P2 | EV score 低スコア帯 (< 0.8) でのスキップ強化 |
| P2 | early_exit の閾値調整 (現状発火せず) |

---

## 4. 品質監査 (付帯)

### Bare except 監査

4 箇所すべて適切にハンドリング済み確認:
- `fill_probability_model.py:322` — logger.exception
- `maker_risk_guards.py:482` — exc_info=True
- `sidecar_signal_io.py:72` — pass + fallback (意図的)
- `skip_gate_evaluator.py:633` — logger.warning + exc_info

### asyncio.sleep 直接呼出し

13 箇所。大半は計測/監視モジュール(pnl_measurer, order_monitor)でレジーム対応不要。変更なし。

### 低優先: ab_judgment.py

986 行。オフライン分析モジュールのため分割は低優先。

---

## テスト結果

```
v460 全テスト: 4470 passed, 9 skipped, 0 failed
skip_gate 関連: 648 passed, 28 skipped, 0 failed
```

---

## 変更ファイル一覧

### 新規

| ファイル | 行数 |
|---|---|
| `scripts/v460/lib/skip_gate_model_loader.py` | ~300 |
| `scripts/v460/lib/skip_gate_ev_weighted.py` | ~210 |

### 修正

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/skip_gate_evaluator.py` | Mixin 化 + MAX LINES: 900 (1362→866) |
| `scripts/v460/lib/cycle_gate_aggregator.py` | MAX LINES: 800 追加 |
| `scripts/v460/lib/fill_config_parser.py` | MAX LINES: 1100 追加 |
| `tests/unit/v460/_fill_test_source.py` | SKIP_GATE_MODEL_LOADER 定数追加 |
| `tests/unit/v460/test_141_side_specific_models.py` | logger パッチパス修正 |
| `tests/unit/v460/test_143_regime_utilization.py` | ソース読み取り先修正 |
| `tests/unit/v460/test_139_review_fixes.py` | ソース読み取り先修正 |
| `tests/unit/v460/test_255_getattr_bare_except_cleanup.py` | ソース読み取り先修正 |

---

## 5. Fill Test 10 日間深堀り分析 (2026-03-08 〜 03-17)

> セクション 3 の 5 日間分析を 10 日間に拡張し、再起動前後のパフォーマンス差異を
> 設計面・市場理論面から多角的に考察する。

### 5.1 分析対象

- **期間**: 2026-03-08 〜 2026-03-17 (10 日間)
- **レコード総数**: 4,825
- **約定数**: 1,002
- **ユニーク SHA**: 35+
- **ユニーク PID**: 20+
- **分析スクリプト**: `temp/analyze_fill_test_deep_v2.py`

### 5.2 SHA パフォーマンス比較 (上位 8 SHA)

| SHA | コミット | 日付 | n | Fill Rate | PnL 30s | AS% | Win% | Buy PnL | Sell PnL | Buy Ceiling% | Sell Ceiling% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **0d22298** | — | 3/9 | 212 | 29.2% | **+1.21** | 27.4 | 56.5 | +1.60 | +0.85 | **N/A†** | **N/A†** |
| **eb24cf4** | — | 3/8 | 246 | 36.2% | -0.02 | 22.5 | 52.8 | +0.57 | -0.62 | **N/A†** | **N/A†** |
| 819ec73 | — | 3/9 | 252 | 38.1% | -0.07 | 34.4 | 46.9 | -0.10 | -0.05 | N/A† | N/A† |
| 92c588e | — | 3/12 | 258 | **37.2%** | -0.22 | 25.0 | 49.0 | -0.76 | +0.32 | N/A† | N/A† |
| 5c3238f | — | 3/13-14 | 541 | 34.6% | -0.27 | 27.8 | 48.1 | -0.60 | +0.07 | N/A† | N/A† |
| bff652e | — | 3/12-13 | 379 | 27.4% | -0.47 | 28.8 | 48.1 | -0.49 | -0.45 | N/A† | N/A† |
| d0769f2 | 458# hot-reload | 3/16-17 | 317 | **3.8%** | -1.78 | 41.7 | 33.3 | -1.78 | -1.78 | 100% (5/5) | 100% (3/3) |
| **f840d0e** | 459# fix | 3/17 | 307 | 28.7% | **-1.23** | 31.8 | 47.7 | -0.32 | **-2.15** | **100% (41/41)** | **100% (17/17)** |

> **†** `execution_pre_clamp_offset` フィールドが 3/8-3/14 のレコードには不在。ceiling rate は計測不能であり「0%」ではない。463# 検証で確定。

**発見 1** ~~(修正済)~~: 正の PnL を達成した SHA (0d22298, eb24cf4) では ceiling clamp が**計測不能 (N/A)**。`execution_pre_clamp_offset` フィールドが 3/8-3/14 レコードに不在のため、clamp の有無は判断できない。ただし offset 値自体は低い (buy=0.117-0.119, sell=0.296-0.323) ため、pipeline が計算値をそのまま使用していた可能性は残る。

**発見 2**: f840d0e (現行) は全 SHA 中**最悪の PnL** (-1.23bps)。Buy ceiling clamp 100% (41/41, pre_clamp 存在 fills 全件) が支配的。

**発見 3**: d0769f2 (458# hot-reload) は fill rate 3.8%。3/17 では **217 レコード全件 `ranging_low_vol_skip` で cancel** = 完全遮断。d0769f2 の ceiling は **100% (buy 5/5, sell 3/3)** — f840d0e と同様に全件 clamp。

### 5.3 日別トレンド

```
日付     レコード  Fill%   PnL 30s  AS%    Win%   特記
──────────────────────────────────────────────────────────────
3/08      666    21.9%   -0.26    25.3   48.6   初日、eb24cf4 主体
3/09      603    31.2%   +0.17    31.4   48.9   0d22298=最良 SHA
3/10      190    26.8%   +0.43    25.5   56.9   データ少
3/11       85    32.9%   +0.48    32.1   53.6   データ最少
3/12      553    39.6%   +0.32    23.3   51.6   ★ ピーク日 (fill/AS)
──────────────────────────────────────────────────────────────
3/13      625    28.5%   -0.79    33.7   46.6   ← ここから悪化
3/14      602    26.4%   -0.47    18.2   47.8   route_to_kill=43
3/15      370    16.8%   -1.44    32.3   33.9   worst win rate
3/16      607    13.7%   +0.86    28.9   48.2   route_to_kill=53
3/17      524    16.8%   -1.23    31.8   47.7   status_unknown=8
```

**3/12 が性能のピーク**。fill rate 39.6%, AS 23.3%, PnL +0.32bps。
3/13 以降は PnL がマイナス圏に転落し、fill rate も 13-17% に急落。

### 5.4 再起動境界分析 (3/16→3/17)

#### 5.4.1 3/16 SHA タイムライン (UTC)

```
f34467b  00:01-03:29  n=79,  fill=6
1d64e64  03:30-03:51  n=10,  fill=3   (443# cross-venue)
c38c15e  03:59-04:09  n=6,   fill=2   (444# cross-venue閾値)
e23a063  04:11-05:07  n=23,  fill=7   (444# 閾値最終調整)
a9714ad  05:17-07:55  n=80,  fill=9   (445# cross-venue EMA)
52627ff  07:56-11:59  n=117, fill=14  (450# DRY cleanup)
c7ebd8c  12:03-19:27  n=192, fill=30  (454# micro-timeout)
d0769f2  19:37-23:59  n=100, fill=12  (458# hot-reload)
```

3/16 だけで **8 SHA = 8 回再起動**。Cross-venue 関連の変更が集中。

#### 5.4.2 3/17 SHA タイムライン (UTC)

```
d0769f2  00:01-07:17  n=217, fill=0   ← 全件 cancel (ranging_low_vol_skip)
f840d0e  07:22-18:20  n=307, fill=88  ← 459# fix 適用後
```

3/17 は前半 7 時間が d0769f2 による**完全遮断**。f840d0e で再起動後に取引再開。

#### 5.4.3 c7ebd8c (3/16) vs f840d0e (3/17) 直接比較

| 指標 | c7ebd8c (3/16) | f840d0e (3/17) | 差分 |
|---|---|---|---|
| Fill Rate | 15.6% (30/192) | 28.7% (88/307) | +13.1pp |
| PnL 30s | **+2.26 bps** | **-1.23 bps** | -3.49 bps |
| AS% | 33.3% | 31.8% | -1.5pp |
| Win% | 53.3% | 47.7% | -5.6pp |
| Buy PnL | -1.39 bps | -0.32 bps | +1.07 |
| Sell PnL | **+5.92 bps** | **-2.15 bps** | **-8.07 bps** |
| Buy Ceil% | 80.0% (12/15) | 93.2% (41/44) | +13.2pp |
| Sell Ceil% | 40.0% (6/15) | 38.6% (17/44) | -1.4pp |
| Buy Pre-clamp | 0.306 | 0.288 | -0.018 |
| Sell Pre-clamp | 0.834 | 0.705 | -0.129 |
| Top Cancel | ranging_low_vol_skip(78) | no_feasible_quote(59) | プロファイル変化 |

**逆説**: f840d0e は fill rate が倍近く高いが PnL は -3.49bps 悪化。「より多く約定している」が「損失を拡大している」。

**Sell 側の壊滅的差異**: c7ebd8c sell=+5.92bps → f840d0e sell=-2.15bps (差=-8.07bps)。c7ebd8c は sell pre-clamp が 0.834 と高く、より広い offset で有利な位置に約定していた。

#### 5.4.4 ベンチマーク SHA との比較 (3/8-3/9)

| 指標 | eb24cf4 (3/8) | 0d22298 (3/9) | f840d0e (3/17) |
|---|---|---|---|
| Fill Rate | 36.2% | 29.2% | 28.7% |
| PnL 30s | -0.02 | **+1.21** | **-1.23** |
| AS% | 22.5% | 27.4% | 31.8% |
| Buy Offset | 0.119 | 0.117 | 0.198 |
| Sell Offset | 0.296 | 0.323 | 0.483 |
| Buy Ceiling | **N/A†** | **N/A†** | 100% (41/41) |
| Sell Ceiling | **N/A†** | **N/A†** | 100% (17/17) |
| Balance Switch | **23.6% (21/89)‡** | 0% (0/62)‡ | 58% (51/88)§ |
| Cross-Venue | 未導入† | 未導入† | 42% (37/88) |
| Top Cancel | sell_dynamic_kill(27%) | sell_dynamic_kill(51%) | no_feasible_quote(27%) |

> **†** 3/8-3/14 のレコードには `execution_pre_clamp_offset`, `cross_venue_lead_lag_applied` フィールドが不在。
> **‡** `balance_forced_switch` フィールドで計測。eb24cf4 は 21/89 が True (PnL=+0.1bps, AS=23.8% = normal より良好)。初版 461# は `resolved_side_reason` (不在フィールド) を使い 0% と誤記していた。
> **§** f840d0e は `resolved_side_reason` フィールドで計測 (`balance_forced_switch` は全件 None)。‡ と § は異なるフィールドによる計測であり直接比較は要注意。

**根本的差異** ~~(修正済)~~: 3/8-3/9 の好成績 SHA はシンプルな構成 — **ceiling clamp は計測不能だが offset 値が低い (buy 0.12, sell 0.30)**。cross-venue は未導入。balance switch は eb24cf4 で 23.6% 発火していたが PnL への悪影響は見られず (+0.1bps)、0d22298 では 0%。sell_dynamic_kill が自然なフィルターとして機能していた点は確認済み。

### 5.5 Cancel 理由のプロファイル変化

```
                eb24cf4  0d22298  5c3238f  bff652e  f840d0e
                (3/8)    (3/9)    (3/13)   (3/12)   (3/17)
─────────────────────────────────────────────────────────────
sell_dyn_kill   26.8%    50.7%    58.2%    53.5%    0%     ← 消滅
skip_gate       24.8%    18.0%    6.8%     17.5%    11.4%
spread_narrow   15.3%    16.7%    13.8%    12.7%    19.6%
no_feasible     1.9%     0.7%    2.0%     —        26.9%  ← 新主因
timeout         7.6%     —       2.8%     2.2%     18.3%  ← 急増
final_clamp     —        —       —        —        4.1%   ← 新登場
cv_veto         —        —       —        —        2.3%   ← 新登場
range_low_vol   —        —       —        —        (d0769f2 で 94%)
```

**sell_dynamic_kill の漸進的減少と no_feasible_quote の台頭** ~~(修正済)~~: sell_dynamic_kill は 458#/459# で突然消滅したのではなく、3/10 以降から同一日のSHA間でも高い変動性を示していた (3/12: 92c588e=6%, 66165ee=0%; 3/13: bff652e=73%, 5c3238f=41%; 3/14: 5c3238f=80%, 1a84c04=0%)。f840d0e で 0% となったのは、この漸進的傾向の帰結であり、458#/459# 単独の因果ではない。一方、no_feasible_quote (26.9%) と timeout (18.3%) の台頭は f840d0e 固有の現象。

### 5.6 時間帯分析 (f840d0e, JST)

| JST | n | PnL 30s | AS% | 評価 |
|---|---|---|---|---|
| 16h | 7 | +0.85 | 14.3% | ✅ 良好 |
| 17h | 9 | +1.64 | 33.3% | ✅ 最良 |
| 19h | 8 | +1.90 | 25.0% | ✅ 良好 |
| 00h | 11 | -1.93 | 18.2% | ⚠ |
| 02h | 11 | -1.50 | 36.4% | ⚠ |
| 20h | 12 | -1.37 | 33.3% | ⚠ |
| 21h | 16 | -1.41 | 18.8% | ⚠ |
| 18h | 4 | -3.19 | 50.0% | ❌ |
| 22h | 7 | -3.18 | 57.1% | ❌ 危険 |
| 03h | 1 | -6.90 | 100% | ❌ |
| 23h | 2 | -12.94 | 100% | ❌ 壊滅 |

**JST 22-23h, 03h**: 逆選択率 57-100%、PnL -3 〜 -13bps。深夜薄板時間帯での約定は壊滅的。

### 5.7 f840d0e 構造分析

#### EV Score の双峰分布

| EV bin | n | PnL 30s | AS% | 備考 |
|---|---|---|---|---|
| < 0.5 | 44 | -2.15 | 40.9% | = sell 側全量 |
| 0.5-1.0 | 0 | — | — | **空白** |
| 1.0-2.0 | 2 | +0.55 | 0% | 微量 |
| > 2.0 | 42 | -0.36 | 23.8% | = buy 側 (高 EV でも負) |

EV < 0.5 と EV > 2.0 の二極化。中間帯 (0.5-2.0) にほぼ約定なし。高 EV (>2.0) でも PnL は -0.36bps = **EV score が正の PnL に変換されていない**。

> **463# 検証追記**: 全母集団 (filled+cancel) で検証した結果、f840d0e の EV 0.5-1.0 帯は **filled=0, cancel=0, total=0** — cancel 母集団にすら存在しない構造的空白。旧 SHA では同帯が 7-13% 存在 (例: bff652e 13.4%, 92c588e 10.2%)。d0769f2 も 0% であり、**458# 以降で EV 計算に不連続が発生した**ことが全母集団で確認された。

#### Balance Switch の影響 ~~(修正済)~~

> **計測方法の注意**: f840d0e は `resolved_side_reason == "balance_switch"` で計測。`balance_forced_switch` フィールドは全件 None。一方、旧 SHA (eb24cf4) は `balance_forced_switch == True` で計測。両者は異なるフィールドであり、直接の比率比較には限界がある。

| 区分 | n | PnL 30s | AS% |
|---|---|---|---|
| balance_switch (resolved_side_reason) | 51 (58%) | -1.46 | 35.3% |
| normal | 37 (42%) | -0.92 | 27.0% |

Balance switch 経由の fill が過半数を占め、PnL も AS も normal より劣悪。在庫管理が不利な方向への約定を強制している。

**参考**: eb24cf4 (3/8) では `balance_forced_switch=True` が 21/89 (23.6%) だが、PnL=+0.1bps, AS=23.8% と**むしろ normal (PnL=-0.06) より良好**。balance switch 自体が問題なのではなく、**f840d0e の market 条件下での balance switch コスト** が問題である可能性。

#### Cross-Venue 適用効果

| 区分 | n | PnL 30s | AS% |
|---|---|---|---|
| cv_applied | 37 | -0.36 | 24.3% |
| cv_not_applied | 51 | -1.86 | 37.3% |

Cross-venue が適用された fill は PnL/AS ともに改善。**有効に機能しているが適用率が 42%** にとどまっている。

> **463# 検証追記 (selection bias)**: Side 分離分析の結果、CV効果は**buy側のみ有効** (buy: cv_on=+0.25bps vs cv_off=-1.43bps)。sell側は cv_on=-2.60bps vs cv_off=-2.05bps と**CV適用時のほうが悪化**。P1-3「適用率 42%→70%」の提案は buy 限定で再評価すべき。

---

## 6. 設計面考察 — 構造的問題の同定

### 6.1 Ceiling Clamp パイプラインの根本矛盾

f840d0e におけるバイサイド ceiling clamp 93.2% は**設計上の致命的矛盾**を示す。

1. **Offset pipeline** が market microstructure に基づき optimal offset を計算 (pre-clamp mean = 0.288)
2. **Ceiling clamp** がそれを 0.200 に強制的に切り詰める（30% の情報を破棄）
3. 結果として、モデルが「ここまで離す必要がある」と判断した情報が ceiling によって却下される

```
パイプライン:  ML Model → EV計算 → Offset = 0.288
                                      ↓ ceiling clamp
                               Actual = 0.200  (△ 30%)
```

この構造は**補正パラメータが制御パラメータを無効化する**典型的な over-constraint。405#/418# で導入された final clamp は本来安全装置だが、現在はほぼ全 fill で発火 = もはや「例外」ではなく「ルール」と化している。

**3/8-3/9 のベンチマーク SHA (0d22298, eb24cf4)** では ceiling clamp は**計測不能** (pre_clamp フィールド不在)。ただし offset 値 (buy 0.12, sell 0.30) が低く、ceiling を必要としない範囲で約定していた可能性がある。「安全装置が不要だった頃のほうが安全だった」という解釈は魅力的だが、**旧SHA市場条件 (3/8-3/9) と新SHA市場条件 (3/17) の比較であり、ceiling の有無だけでは説明できない**。

### 6.2 Cancel プロファイルの構造変化 — 「積極的遮断」から「漂流」へ

```
旧世代 (3/8-3/12):                    現行 f840d0e (3/17):
  sell_dynamic_kill = 27-58%              sell_dynamic_kill = 0%
  (明示的判断でキル)                     no_feasible_quote = 27%
  → スキップ品質が高い                    timeout = 18%
                                         → なぜスキップしたか不明瞭
```

sell_dynamic_kill は「条件不適のため積極的にスキップ」。一方 no_feasible_quote/timeout は「引用可能な価格が存在しないまま時間切れ」。後者は pipeline が実行可能な価格を生成できていないことを示す。

> **463# 検証追記**: sell_dynamic_kill の減少は 458#/459# のみに帰すべきではない。3/10 以降から同一日の SHA 間でも 0% と 80% が混在しており、漸進的かつ SHA 間変動の大きい現象。ただし f840d0e で no_feasible_quote/timeout が台頭した事実は新しく、pipeline の構造変化を示唆する。

**設計上の含意**: 458#/459# の変更が sell_dynamic_kill の条件を変質させた結果、trade or not-trade の判断が曖昧化した。明確な kill 条件が失われ、pipeline が「取引できない状態」に長時間留まる。

### 6.3 ranging_low_vol_skip の全面遮断問題

d0769f2 (458# hot-reload) の 3/17 前半 7 時間：217 レコード全件が `ranging_low_vol_skip`。

しかし f840d0e (459#) の fills を見ると：
- regime=`ranging`: **87/88 fills** (98.9%)
- regime=`trending_up`: 1/88 fills

**本システムの約定のほぼ全量は ranging 環境で発生している**。ranging_low_vol_skip はトレンド不在時に取引を止める意図だが、板取りメーカーは本質的にレンジ内の mean-reversion から利益を得る戦略であり、ranging 環境こそ本来の主戦場。このゲートが全面的に発火する状態は戦略の自己否定に等しい。

### 6.4 Balance Switch の逆機能 ~~(修正済)~~

balance_switch fills が 58% (51/88) を占め (`resolved_side_reason` で計測)、normal fills より PnL/AS ともに劣化 (-1.46 vs -0.92, AS 35.3% vs 27.0%)。

balance switch は在庫偏りを解消するために「不利でも反対方向に約定する」メカニズム。しかし：
- 在庫偏りの解消コストが、偏り放置のコストを上回っている可能性
- 特に buy ceiling 100% の状況下で balance switch → buy を強制すると、ceiling-clamped な不利な価格で在庫を増やすことになる

> **463# 検証追記**: eb24cf4 では `balance_forced_switch=True` が 21/89 (23.6%) だが PnL=+0.1bps, AS=23.8% と**正常 fill より良好**。balance switch 自体が劣化要因なのではなく、**f840d0e 固有の条件 (ceiling 100%, offset 高) と balance switch の組み合わせ**が問題。また、f840d0e の balance_switch 計測は `resolved_side_reason` (新フィールド)、eb24cf4 は `balance_forced_switch` (従来フィールド) であり、**計測方法が異なる**ことにも留意が必要。

### 6.5 EV Score の中間帯空白

EV 0.5-1.0 帯に fill が 0 件。これはスコアリング pipeline の不連続を意味する。旧世代 SHA では 0.5-1.0 帯にも一定の fill が存在した (819ec73: n=10, bff652e: n=17, 92c588e: n=14)。

458#/459# での変更が EV 計算に何らかの離散化もしくは閾値効果をもたらし、中間帯が消失した可能性がある。中間帯は「判断の境界」であり、ここに fill がないことは「迷ったケースを全て拒否」している ≒ **過剰に保守的なフィルタリング** と解釈できる。

---

## 7. 市場理論面考察

### 7.1 Glosten-Milgrom モデルから見た Ceiling の問題

Glosten-Milgrom (1985) の bid-ask spread 理論：

$$
\text{spread} \geq 2 \cdot P(\text{informed}) \cdot E[\Delta V | \text{informed}]
$$

マーケットメーカーの提示 spread は、情報トレーダーの存在確率と期待価格変動の積を補償する必要がある。

f840d0e では：
- AS rate = 31.8% (情報トレーダーとの取引確率)
- 尤度の高い buy 側で ceiling が offset を 0.288 → 0.200 に圧縮
- **情報コストの補償が 30% 不足している**

比較: 0d22298 (best SHA) は offset = 0.117 だが AS = 27.4%。f840d0e は offset = 0.198 (0d22298 より高い) にもかかわらず AS = 31.8%。offset を上げたのに AS が改善していないのは、**市場条件の変化 (スプレッド縮小・流動性変化) が offset 増分を吸収している** ことを示唆。

### 7.2 Avellaneda-Stoikov 在庫リスクプレミアム

Avellaneda-Stoikov (2008) の最適 spread：

$$
\delta^*(q) = \gamma \sigma^2 (T-t) + \frac{2}{\gamma} \ln\left(1 + \frac{\gamma}{k}\right)
$$

在庫 $q$ が偏るほど、偏り解消方向の offset を縮小し、逆方向を拡大する。

f840d0e の balance_switch は Avellaneda-Stoikov の在庫管理に概ね整合するが、**ceiling clamp が $\delta^*(q)$ の上界を強制** している問題がある。在庫リスクが高い局面で必要な offset ($\delta^* = 0.288$) を ceiling ($\delta_{\max} = 0.200$) が遮断すると、リスクプレミアムが不足し逆選択に晒される。

結果: balance_switch fills の PnL = -1.46bps は、在庫リスクプレミアム不足の直接的な反映。

### 7.3 Kyle λ と深夜薄板リスク

Kyle (1985) の λ (market impact 係数)：

$$
\lambda = \frac{\Sigma_0}{2\sigma_u^2}
$$

薄板時間帯 (JST 22-03h) では流動性 ($\sigma_u^2$) が減少し λ が上昇 → 情報トレーダーの 1 注文あたりの価格インパクトが拡大。

データが証明：
- JST 22h: AS = 57.1%, PnL = -3.18bps
- JST 23h: AS = 100%, PnL = -12.94bps
- JST 16-17h (活況時間): AS = 14-33%, PnL = +0.85 〜 +1.64bps

**深夜帯では情報非対称性が極端に拡大し、固定 ceiling (0.200) では全く補償できない**。時間帯別 ceiling もしくは deep-night skip gate の導入が理論的に支持される。

### 7.4 マーケットメーカーの本質的ジレンマ

| 要素 | 旧世代 (良い) | 現行 (悪い) | 理論的解釈 |
|---|---|---|---|
| Offset | 低い (0.12) | 高い (0.20) | 市場条件悪化への防御的反応 |
| Ceiling | なし | 93% 発火 | 防御が過剰 → 逆に脆弱化 |
| Cancel | 明示的 kill | 漂流型 cancel | 判断力低下 |
| 機能追加 | なし | CV/BalSwitch/Ceiling | 複雑性の増加がパフォーマンスを劣化 |

**複雑性のパラドックス**: 安全機構の追加が個々には合理的でも、積層されると相互干渉を起こし全体パフォーマンスを悪化させる。典型的な **defensive programming anti-pattern** 。

### 7.5 Ranging 環境と Mean-Reversion 収益源

板取りメーカーの収益モデルは基本的に mean-reversion ベース:

$$
E[\text{profit}] = \delta \cdot P(\text{fill}) \cdot (1 - P(\text{adverse})) - \text{AS cost}
$$

ranging 環境は mean-reversion が最も期待されるレジームであり、f840d0e の 98.9% の fill が ranging で発生しているのは戦略として正しい。しかし：

1. d0769f2 の `ranging_low_vol_skip` 全面遮断は、この収益源を完全に遮断
2. f840d0e で遮断は解除されたが、ceiling clamp がδを過剰に圧縮
3. 結果として $E[\text{profit}] < 0$ となっている

**理論的処方**: ranging 環境での offset は ceiling-free (もしくは ceiling 大幅緩和) であるべき。Ranging でこそ ML model の計算した offset を信頼し、trending 環境でのみ ceiling で保護するのが合理的。

---

## 8. 総合診断と改善提案

### 8.1 根本原因の階層構造

```
Level 0: PnL = -1.23bps (全 SHA 中最悪)
  ↑
Level 1: Sell 側壊滅 (-2.15bps, tail10=-19.78bps)
  ↑
Level 2a: Ceiling clamp 93.2% → offset が情報コストを補償できない
Level 2b: Balance switch 58% → 不利方向への強制約定
Level 2c: Cancel プロファイル変質 → 漂流型 cancel の増加
  ↑
Level 3: 458#/459# の安全機構追加が積層的に pipeline を制約
```

### 8.2 P0 改善提案 (即時)

| # | 施策 | 期待効果 | 理論的根拠 |
|---|---|---|---|
| P0-1 | **Buy ceiling を 0.200 → 0.350 に緩和** (ranging 限定) | PnL +0.5-1.0bps | Glosten-Milgrom 情報コスト補償 |
| P0-2 | **JST 22-03h の fill を停止もしくは ceiling 2倍化** | tail loss 削減 | Kyle λ 拡大による AS 不可避 |
| P0-3 | **status_unknown_fast 増加の原因調査** (8件、全SHA最多) | 異常検出 | — |

### 8.3 P1 改善提案 (短期)

| # | 施策 | 期待効果 | 理論的根拠 |
|---|---|---|---|
| P1-1 | **ranging_low_vol_skip の閾値見直し** (ソフトモード) | fill rate 回復 | ranging = 主収益レジーム |
| P1-2 | **balance_switch 発動条件の厳格化** | PnL -0.54bps 改善 | A-S 在庫コスト過大 |
| P1-3 | **Cross-venue 適用率の向上** (cv_applied=42%→70%+) | AS -5pp | CV 有効性確認済み |
| P1-4 | **EV 0.5-1.0 帯の空白解消** (計算不連続の修正) | fill 品質向上 | スコアリング連続性 |

### 8.4 P2 改善提案 (中期)

| # | 施策 | 期待効果 | 理論的根拠 |
|---|---|---|---|
| P2-1 | **Ranging 環境で ceiling-free モード導入** | pipeline 信頼回復 | Mean-reversion 収益モデル |
| P2-2 | **sell_dynamic_kill 条件の再設計** (明示的 kill 回復) | cancel 品質向上 | 漂流 → 判断 への回帰 |
| P2-3 | **時間帯別 offset 係数の導入** | 深夜 AS 防御 | Kyle λ 時間変動 |
| P2-4 | **安全機構の体系的レビュー** (ceiling/CV/BalSwitch 相互作用) | 全体最適化 | 複雑性削減 |

---

## 9. Reviewer追記 (Codex)

461# は症状の観察としてはかなり良いが、**因果推定の置き方にまだ危うさが残る**。特に「旧SHAは単純で良かった」「新しい安全機構が利益を壊した」という結論は、現状の分析スクリプトの集計方法だと過剰に強い。

### 9.1 HIGH: スキーマ差分を跨いだ比較が未補正

本稿の最重要問題はこれである。

`temp/analyze_fill_test_deep_v2.py` では、以下の late-added フィールドをそのまま同列比較している。

- `execution_pre_clamp_offset`
- `cross_venue_lead_lag_applied`
- `resolved_side_reason`

しかしこれらは比較的新しい観測項目であり、古い SHA の fill_records では **未記録 = None/False 扱い** になる。

その結果、

- 「旧SHAは ceiling clamp 0%」
- 「旧SHAは balance switch 0%」
- 「旧SHAは cross-venue なし」

という表現のうち、少なくとも前二者は **本当にゼロだった** のではなく、**観測フィールドが存在しないのでゼロに見えているだけ** の可能性が高い。

特に `offset_stats()` は `execution_pre_clamp_offset is not None` のレコードだけを clamp 判定に使いつつ、分母は side fills 全体で計算している。古いレコードに当該列が無ければ、自動的に「unclamped 扱い」で ceiling rate が下がる。

したがって、5.2 / 5.4 / 5.7 の

> 「好成績 SHA の共通項 = ceiling clamp ゼロ」

という主張は、現状のままでは **証明されていない**。ここは強い結論を一段下げるべきである。

### 9.2 HIGH: run_id / start_git_sha を使わず current git_sha だけで切っている

本稿は再起動境界を意識しているのに、分析スクリプトは実質的に `git_sha[:7]` しか使っていない。

しかし現行 FillRecord には、既に

- `run_id`
- `git_sha`
- `start_git_sha`

が入っている。さらに `ztb/metrics/fill_quality.py` 側には run_id / git_sha / date でのフィルタ基盤も存在する。

それにもかかわらず、`temp/analyze_fill_test_deep_v2.py` は

- run 単位の分離をしない
- same-run / restarted-run を混在させる
- hot-reload 前後を current `git_sha` のみで読む

という設計である。これでは「458#/459# が悪化させた」のではなく、「別 run / 別市場局面 / 別 PID を並べたらそう見えた」可能性を排除できない。

特に 5.4 の再起動境界分析は方向性として正しいのに、**識別子の使い方がまだ甘い**。次版では最低でも

- `run_id` 固定
- `start_git_sha` 固定
- できれば日付だけでなく時刻窓固定

が必要である。

### 9.3 HIGH: 「458#/459# が sell_dynamic_kill を変質させた」は現状では言い過ぎ

6.2 や 5.5 では、sell_dynamic_kill の消滅と no_feasible_quote / timeout の増加を、458#/459# の変更による構造変化として読んでいる。

ただし、少なくとも文書ベースで確認できる 458#/459# の中心変更は

- macro sell protection
- hysteresis
- micro-timeout 連動
- ranging_buy_low_vol ソフト化
- hot-reload 到達性の改善

であり、**sell_dynamic_kill そのものの条件を直接変更した記述はない**。

もちろん間接影響はありうる。だが現状の分析だけで

> 「sell_dynamic_kill の条件が変質した」

と書くのは因果が強すぎる。ここは

> 「cancel population が変わり、結果として sell_dynamic_kill の観測比率が低下した可能性」

までに留めるのが妥当である。

### 9.4 MEDIUM: Cross-Venue の効果推定は selection bias を含む

5.7 の

- cv_applied: -0.36 / AS 24.3%
- cv_not_applied: -1.86 / AS 37.3%

は一見かなり有望に見える。しかし cross-venue hint は、もともと

- 参照市場の spread/velocity/EMA 条件を満たし
- confidence gate を通過した

ケースにだけ適用される。つまり `cv_applied` 群は最初から「シグナルが明瞭だったサブセット」である。

そのため、

> 「適用率を 42% → 70% に上げれば AS が 5pp 下がる」

という提案はまだ飛躍がある。必要なのは適用率の単純引上げではなく、

- side/time/regime 固定比較
- same-SHA 内の applied vs not-applied
- できれば hint confidence bin 別の効果測定

である。

### 9.5 MEDIUM: EV 0.5-1.0 帯空白から「不連続」を断定するのは早い

f840d0e の filled 88 件だけを見て EV bin が

- `<0.5`: 44
- `0.5-1.0`: 0
- `1.0-2.0`: 2
- `>2.0`: 42

となっているのは事実として面白い。

ただし、ここから直ちに

> 「458#/459# で EV 計算に離散化が入った」

とまでは言えない。理由は 3 つ。

1. filled だけを見ており、cancel された候補を見ていない
2. sample size が小さい
3. side / regime / balance_switch が混ざっている

したがってここは「不連続の疑い」までに留め、まずは

- 全 processed population
- buy/sell 分離
- pre/post で同じ bin 定義

で再検証すべきである。

### 9.6 本稿の強い点は残る

批判だけではない。本稿で価値が高い点も明確にある。

1. **3/12 をピーク、3/13 以降を悪化局面として整理したこと**
2. **3/16→3/17 の restart boundary を主要観察点に据えたこと**
3. **fill rate 改善と PnL 改善が一致しないことを明示したこと**
4. **deep-night 帯の tail risk を時間帯で可視化したこと**
5. **balance_switch 群の悪化を症状として抽出したこと**

この 5 点は、次の実装優先順位を決める上で有用である。

### 9.7 私の最終判断

461# は、

- 症状の観察: **高品質**
- 因果の切り分け: **まだ粗い**
- 実装提案の方向性: **一部妥当だが強い結論は要減速**

という評価である。

特に次版で優先すべきは新機能追加ではなく、以下の 4 点である。

| 優先 | 追加タスク | 理由 |
|---|---|---|
| P0 | `analyze_fill_test_deep_v2.py` に schema-aware 比較を追加 | late-added field の擬似ゼロ問題を除去 |
| P0 | `run_id` / `start_git_sha` 固定モードを追加 | hot-reload/再起動混線の排除 |
| P1 | cross-venue を side/time/regime 固定で再評価 | selection bias の除去 |
| P1 | EV bin を filled ではなく processed 母集団で再計測 | 不連続仮説の検証 |

結論として、本稿は「利益が出ない理由をかなり正しく嗅ぎ当てている」が、**旧SHA単純系が本当に優れていたのか、安全機構が本当に犯人なのかはまだ確定していない**。ここを確定させずに ceiling 緩和や gating 撤去へ走ると、再現性のない改善に終わる危険がある。

---

## 10. 検証補正サマリー (463# に基づく)

> **検証日**: 2026-03-18
> **検証スクリプト**: `temp/analyze_fill_test_v3.py` (schema-aware, 全 7 パート分析)
> **検証動機**: 462# レビュー指摘 + §9 Reviewer 追記の各論点を実データで検証

### 10.1 修正箇所一覧

| 箇所 | 原文 | 修正後 | 根拠 |
|---|---|---|---|
| §5.2 table Ceiling% (旧SHA全6件) | 0% | **N/A†** | `execution_pre_clamp_offset` が 3/8-3/14 レコードに不在 |
| §5.2 d0769f2 Ceiling | 83.3%/50.0% | **100% (5/5) / 100% (3/3)** | pre_clamp 存在 fills のみで再計測 |
| §5.2 f840d0e Ceiling | 93.2%/38.6% | **100% (41/41) / 100% (17/17)** | 同上 |
| §5.2 発見1 | 「ceiling clamp ゼロ」 | 「計測不能 (N/A)」 | フィールド不在 |
| §5.4.4 eb24cf4 Balance Switch | 0% | **23.6% (21/89)** | `balance_forced_switch=True` で再計測 |
| §5.4.4 eb24cf4/0d22298 Ceiling | 0/45(0%), 0/30(0%) | **N/A†** | フィールド不在 |
| §5.5 sell_dynamic_kill | 「458#/459# で消滅」 | 「漸進的減少 + SHA間変動大」 | v3 PART7 タイムライン分析 |
| §5.7 CV 効果 | 「適用率を上げれば改善」 | 「buy 限定で有効、sell は逆効果」 | v3 PART6 side分離分析 |
| §5.7 EV 0.5-1.0 | 「空白 (filled のみ)」 | 「全母集団で確認、構造的空白」 | v3 PART4 全母集団分析 |
| §6.1 旧SHA ceiling | 「0% だった」 | 「計測不能」 | フィールド不在 |
| §6.4 Balance Switch | 「逆機能」 | 「f840d0e 条件下での逆機能 + 計測方法差異」 | eb24cf4 では正の効果 |

### 10.2 462# 指摘の検証結果

| 462# §節 | 指摘内容 | 判定 | 備考 |
|---|---|---|---|
| §2 Schema Drift | late-added fields で旧SHA比較が不正確 | **CONFIRMED** | 3/8-3/14: 4フィールド全て不在 |
| §3 Run Drift | git_sha[:7] だけで run を識別 | **CONFIRMED** | 28 runs 中、hot-reload で複数SHA持つ run あり |
| §4 Config Drift | config identifier が FillRecord に不在 | **VALID (未検証)** | 構造上の欠損、要改善 |
| §5 Population Drift | filled のみの分析は EV/CV/BalSwitch で不十分 | **PARTIALLY CONFIRMED** | EV は全母集団で構造的空白を確認。CV は selection bias が sell で裏付け |
| §6 Market Regime Drift | 3/8 vs 3/17 は matched-market 比較ではない | **VALID** | 暗黙の前提として Market同一を仮定していた |
| §7 Implementation Caution | ceiling 緩和等の提案は根拠不十分 | **PARTIALLY VALID** | ceiling以外の要因 (market, schema) を排除できていない |

### 10.3 新発見

1. **balance_forced_switch vs resolved_side_reason の二重計測問題**: f840d0e は `balance_forced_switch=None` (全件) だが `resolved_side_reason="balance_switch"` が 51 件。旧 SHA は `balance_forced_switch=True/None`。2つのフィールドは異なる計測であり、cross-schema 比較に注意

2. **d0769f2 ceiling = 100%**: 初版 461# は 83.3%/50.0% としていたが、pre_clamp 存在 fills に限れば buy 5/5, sell 3/3 で 100%。f840d0e と同じく全件 clamp

3. **eb24cf4 balance switch は正の効果**: 21/89 fills が balance_switch=True だが PnL=+0.1bps, AS=23.8% で normal (-0.06bps, 22.1%) より良好。「balance switch = 悪」は f840d0e 限定の観察
