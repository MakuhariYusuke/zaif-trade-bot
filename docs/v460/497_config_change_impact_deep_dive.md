# 497# Config Change Impact Deep Dive — 3/10-3/19 全変更の効果検証

**日時**: 2026-03-20
**対象期間**: 2026-03-10 〜 2026-03-19 (fill_test JSONL ログ)

---

## 1. 目的

v460 で適用された各 config 変更の効果を日次 fill ログから検証し、
「明確に良い / 悪い / 微妙」を判定する。
「微妙」なものについてはコードレベルでの原因調査を実施。

---

## 2. 全体サマリ (日次メトリクス 3/10-3/19)

| 日付 | 主要変更 | fills | fill% | PF | avg_pnl30 | 主要 cancel |
|------|----------|-------|-------|------|-----------|-------------|
| 3/10 | v460 開始 | 11 | 6.6% | 0.33 | -3.08 | timeout多数 |
| 3/11 | - | 16 | 8.8% | 0.95 | -0.12 | timeout |
| 3/12 | - | 22 | 7.3% | 0.25 | -3.72 | timeout |
| 3/13 | SG 改善 | 27 | 7.2% | 0.40 | -4.33 | timeout |
| 3/14 | 454# micro-timeout ON | 57 | 10.3% | 0.40 | -2.28 | timeout |
| 3/15 | 454# 定着 | 95 | 14.7% | 1.28 | +0.61 | timeout |
| 3/16 | 438# ranging hard skip | 83 | 13.7% | 1.28 | +0.71 | timeout |
| 3/17 | 459#+458# soft+macro | 109 | 17.2% | 0.67 | -1.36 | NFQ59 |
| 3/18 | 468# deep-night defense | 108 | 23.8% | 0.95 | -0.08 | preflight87 |
| 3/19 | 481#+491# spread+VPIN | 84 | 33.3% | 0.79 | -0.83 | sdk38+rtk38 |

---

## 3. 各変更の判定

### 3.1 ✅ 454# micro-timeout ON (3/14 適用)
**判定: 明確にプラス**

- fill rate: 7.2% → 10.3% → 14.7% (2日で倍増)
- PF: 0.40 → 1.28
- 理由: TTL-based cancel+re-quote が stale order の adverse selection を大幅低減

### 3.2 ❌ 438# ranging hard skip (3/15→3/16)
**判定: 明確にマイナス**

- fill rate: 14.7% → 13.7% (−1.0%pt)
- ranging regime での fill 機会を hard block → 7h+ のデッドロック発生
- 459# で soft mode に戻して解消

### 3.3 ✅ 468# deep-night defense (3/17→3/18)
**判定: 明確にプラス**

- rlvs: 217 → 0 (完全除去)
- fill rate: 17.2% → 23.8%
- sell PF: 0.57 → 1.41 (sell 側が改善)

### 3.4 ⚠️ 459#/458# ranging soft + macro sell (3/16→3/17)
**判定: 微妙 (sell 側に悪影響、ただし市場要因との混在)**

#### 調査結果:
- 3/16→3/17 で sell PF が 1.37→0.57 に急落
- sell total_pnl30: +55.4 → −83.4 (138 JPY 悪化)
- NFQ (no_feasible_quote) が 0→59 に急増

#### 原因分析:
- **458# macro sell slope_threshold 1.0→0.5**: 検出感度が倍増、sell_boost が頻発
  - sell_boost は offset を 1.3-1.6 倍に拡大 → fill 確率低下 → 逆に stale 化
  - 461# 分析で既に「EV 0.5-1.0 band collapse」と指摘
- **459# ranging soft mode**: hard skip を解除 → low-vol ranging でも fill 実行
  → fill rate は上がるが pnl は混在
- **3/18 で sell PF が 1.41 に回復** → 468# deep-night defense が本質的改善

#### 注意:
458# の macro sell 機構は現在も有効 (`slope_threshold: 0.5`)。
3/19 でも sell 側 PnL は−90.2 と悪い。slope_threshold の再調整を検討すべき。

### 3.5 ⚠️ 481# min_spread 1000→700 + veto 6→8bps (3/18 中)
**判定: 微妙 (marginal、有害ではない)**

#### 調査結果:
- fill rate: 19.0% → 34.3% (改善)
- PF: 0.93 → 0.96 (ほぼ同等)
- spread 700-999 での追加 fill: 6件 (3/19)、pnl@700-999 = −0.26 bps

#### 結論:
- 大きな効果なし。有害でもないので 700 を維持で問題ない。

### 3.6 ⚠️ 491# VPIN 0.60→0.80 + ceiling_buy 0.20→0.25 + composite_risk (3/19 19:35)
**判定: 評価不能 (config が適用されなかった)**

#### 重大な発見:
**491# の変更は 3/19 の主要 fill window で適用されていなかった。**

| run_id | 期間 | n | config_hash | ceiling_buy |
|--------|------|---|-------------|-------------|
| 1773912878 | 18:34-00:55 | 201 | 5d47adbdb9502818 | **0.20** (旧値) |
| 1773938398 | 01:40-03:05 | 31 | (None) | **0.25** (新値) |
| 1773943801 | 03:10-03:42 | 15 | (None) | **0.25** + 496# code |

- run_id `1773912878` は 491# commit 前に起動済み → hot-reload で YAML は反映されたが、
  **ceiling_buy は 0.20 のまま** (52/66 fills が ceiling=0.20)
- ceiling_buy=0.25 が適用されたのは 01:40 以降の再起動後 (4/66 fills のみ)

#### hot-reload の調査結果:
- `offset_ceiling_ratio_buy` は `config_hot_reload.py` L497 で **hot-reload 対象に含まれている**
- `maybe_reload()` は mtime ベースで YAML 変更を検知する仕組み
- 可能性: (a) YAML ファイルの mtime が変わらなかった (git commit と YAML 編集のタイミング問題)、
  (b) `maybe_reload()` が呼ばれる前に fill cycle が完了した、
  (c) `config_hash` がリロード後に再計算されない (ログ表示上の問題)
- **→ 次回 config 変更時はプロセス再起動で確実に適用するのが安全**

---

## 4. 3/19 regression 分析

### 4.1 Slow fills の急増
| 日付 | slow% (≥30s) | quick% (<10s) | avg_queue |
|------|-------------|---------------|-----------|
| 3/16 | 0% | 57% | 6.6s |
| 3/17 | 1% | 46% | 10.4s |
| 3/18 | 1% | 53% | 10.5s |
| **3/19** | **34%** | **37%** | **22.6s** |

- micro-timeout は動作 (rq=1 for all slow fills)
- ただし max_requote=2 (pre-496#) → 2 attempt 合計で 30-71s
- 496# で max_requote=4 + TTL短縮 (15s/10s) に改善済み

### 4.2 sell_dynamic_kill の大量発火
- sell_dynamic_kill: 38 回 (22:51-03:29 の time window)
- regime 分布: ranging=53%, trending_up=24%, trending_down=24%
- sell 側の rolling PnL が threshold 以下に → kill gate 発火は正常動作

### 4.3 route_to_kill_deadlock: 38 回
- buy 残高不足 × sell kill-gated → 両側ブロック
- 496# Recovery Skew で解消済み (kill gate bypass + wide offset ×2.0)
- **3/20 以降のデータで効果検証が必要**

---

## 5. Sell 側 PnL の構造的問題

| 日付 | sell_fills | sell_total30 | sell_PF | 主要因 |
|------|-----------|-------------|---------|--------|
| 3/16 | 30 | +55.4 | 1.37 | 良好 |
| 3/17 | 51 | −83.4 | 0.57 | 458# macro + NFQ |
| 3/18 | 52 | +49.5 | 1.41 | 468# 改善 |
| 3/19 | 41 | −90.2 | 0.66 | sdk38 + 市場要因 |

- sell 側は日によって ±50-90 JPY の振れ幅
- 好調日 (3/16, 3/18) と不調日 (3/17, 3/19) が交互
- **458# slope_threshold=0.5 が sell_boost を過剰に発火** させている可能性
- deep-night defense (468#) は有効だが日中の sell adverse selection には無力

---

## 6. アクション推奨

### P0 (緊急)
1. **3/20 データで 496# 効果を検証**: recovery_skew、TTL短縮、max_requote=4 の効果
2. **hot-reload 対象フィールド確認**: `offset_ceiling_ratio_buy` がリロード対象か検証

### P1 (近日)
3. **458# slope_threshold 0.5→1.0 への再調整検討**:
   - 0.5 は感度が高すぎ、sell_boost が過剰に発火
   - 3/17, 3/19 の sell 側悪化の一因の可能性
4. **sell_dynamic_kill の EWMA/time_decay チューニング**:
   - 38 回の sdk 発火は rolling PnL 悪化の自然な反応
   - ただし rtk deadlock を 38 回引き起こし、有効 trade 時間を大幅削減

### P2 (監視)
5. **ceiling_buy=0.25 の効果測定**: 適用後のデータで clamped 率と PnL を比較
6. **composite_risk の影響測定**: enabled=true + threshold=1.5 の blocking 率

---

## 7. 分析手法
- 使用データ: `results/v460/fill_test/fill_records_YYYYMMDD.jsonl`
- 分析スクリプト: `temp/analyze_config_impact.py`, `temp/analyze_deep_dive.py`, `temp/analyze_deep_dive2.py`, `temp/analyze_deep_dive3.py`, `temp/check_ceiling.py`, `temp/check_runid.py`
- fill record フィールド名: `effective_offset_used` (not `effective_offset_ratio`)、`vg_vpin` (not `vpin`)、`requote_attempts`、`offset_stages` (JSON string with `ceiling` key)
