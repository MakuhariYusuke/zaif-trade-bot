# 309# 307#/308# レビュー対応: 理論倒錯修正 + スキーマ是正 + 交絡分離

> **日付**: 2026-03-06  
> **対象レビュー**: [307#](307_ph2_rev_303_306_systems_market_review.md) (Codex), [308#](308_ph2_gemini_31_pro_review_306_307_inverted_microstructure.md) (Gemini 3.1 Pro)  
> **方針**: 喫緊性の高い P0 指摘を即時修正し、妥当性を検証のうえ記録する

---

## 1. レビュー指摘の妥当性判定

### 1.1 308# 盲点1: L2 Microprice Side Selection — 理論倒錯 (AS Seeker) ✅ **妥当**

**Gemini の主張**: `microprice > mid → sell` は Market Making 理論に反する。買い圧力がある時に sell limit を出すと、価格上昇後に逆選択される。

**検証結果**:

| 条件 | 発生事象 | maker への影響 |
|---|---|---|
| `microprice > mid` | `Qb > Qa` (bid 厚い = 買い圧力) | 価格上昇圧力 |
| 旧実装: sell に回る | 約定率↑ (liveness) | 約定後に価格上昇 = AS 被弾 |
| 理論正答: buy に回る | 約定率↓ (thick queue の後方) | 約定すれば安全 |

- **Glosten-Milgrom (1985)**: informed flow が支配する側に maker が立つと逆選択コストが増大
- **Kyle (1985)**: 買い板の厚みは informed trader の蓄積を示唆しうる
- **実データ**: sell AS率 30.3%、AS-nonAS 差 = -9.09 bps (p=0.0000) — sell 側の AS が最大の損失源
- **Live config**: `microprice_side.enabled: true` で **稼働中だった** → 即時対応が必要

**対応**: ロジック反転 + YAML 無効化 (§2.1)

### 1.2 308# 盲点2: L1 Dynamic Cycle Interval — 理論倒錯 ✅ **妥当**

**Gemini の主張**: `σ_ref / σ` (高 σ → 短 interval) は Taker 戦術であり、Maker は高 σ 時に参加頻度を下げるべき。

**検証結果**:

| σ 水準 | 旧数式 (`σ_ref/σ`) | 新数式 (`σ/σ_ref`) |
|---|---|---|
| σ = 0.001 (高ボラ) | interval × 0.5 (短縮) | interval × 2.0 (延長) |
| σ = 0.00025 (低ボラ) | interval × 2.0 (延長) | interval × 0.5 (短縮) |

- **Avellaneda-Stoikov (2008)**: 高ボラ時の最適戦略は「スプレッド拡大 + 頻度低下」
- 高ボラ時に短 interval = より頻繁に板に出る = AS 被弾機会↑
- 低ボラ時に長 interval = 安全な時に休む = 機会損失
- **Live config**: `dynamic_cycle_interval.enabled: true` で **稼働中だった**

**対応**: 数式を `σ / σ_ref` に反転 (§2.2)

### 1.3 307# F1: 分析スキーマ齟齬 ✅ **妥当**

| スクリプト参照 | FillRecord 実フィールド | 状態 |
|---|---|---|
| `effective_offset_ratio` | `effective_offset_used` | 309# 修正済 |
| `fill_timestamp` | `queue_wait_sec` + `timestamp` | 309# 修正済 |
| `offset_stages` | `offset_stages` (JSON) | 309# 分析追加 |
| `queue_depth_ahead` | `queue_depth_ahead` | 309# 分析追加 |
| `microprice_bias_bps` | `microprice_bias_bps` | 309# 分析追加 |

### 1.4 307# F2: 交絡分離不足 ✅ **妥当**

`balance_forced_switch` と `decision_path` を分離しないまま sell vs buy を論じていた。

**修正後データ** (§3 deep dive §8):
- Sell forced_switch: n=247, pnl=-0.85 bps, AS=34.4%
- Sell normal: n=1017, pnl=-0.25 bps, AS=29.3%
- → forced_switch は PnL -0.6 bps 悪化 + AS +5.1pp 増加 → 交絡は有意

### 1.5 307# F6: param_adapter 理由文矛盾 ✅ **妥当**

`offset 拡大で AS 回避` → offset 拡大は aggressive 化であり AS 回避にならない。  
実際の意図は deadlock break (liveness 優先)。理由文を修正。

### 1.6 307# F3: AS/session/regime が支配的 ✅ **妥当**

deep dive 再実行で確認:
- AS vs non-AS diff: sell -9.09bps, buy -7.60bps (p=0.0000)
- UTC 08h sell: AS 63%, pnl -3.55bps (最悪時間帯)
- UTC 16h sell: AS 61%, pnl -2.25bps
- → side 差よりも AS / 時間帯が圧倒的に支配的

### 1.7 307# F5: `none` レジーム問題 ⚠️ **妥当だが今回は見送り**

`none` レジームを regime テーブルに含めるべきとの指摘。方向性は正しいが、  
`none` の内部分類 (warmup / detector-failure / true-none) は別 issue として扱う。

---

## 2. 実施した修正

### 2.1 L2 Microprice Side Selection — ロジック反転 + 無効化

**ファイル**: `scripts/v460/lib/side_selector.py`, `configs/v460/fill_test.yaml`

旧 (306# AS Seeker):
```python
# microprice > mid → 買い圧力 → sell が有利
if microprice_bias_bps > threshold:
    mp_side = "sell"
elif microprice_bias_bps < -threshold:
    mp_side = "buy"
```

新 (309# Safety Mode):
```python
# microprice > mid → 買い圧力 → buy に回る (safety: 厚い queue の後方)
# Glosten-Milgrom: informed flow 側に立たない
if microprice_bias_bps > threshold:
    mp_side = "buy"
elif microprice_bias_bps < -threshold:
    mp_side = "sell"
```

YAML: `microprice_side.enabled: false` — ロジック修正済だが、  
safety モードの有効性を実データで検証するまで無効化を維持。

### 2.2 L1 Dynamic Cycle Interval — 数式反転

**ファイル**: `scripts/v460/lib/fill_loop_orchestrator.py`

旧 (306# Taker 戦術):
```python
ratio = sigma_ref / sigma  # 高σ→短interval (危険)
```

新 (309# A-S Cooldown):
```python
ratio = sigma / sigma_ref  # 高σ→長interval (Cooldown)
```

DocString に Avellaneda-Stoikov (2008) の理論的根拠を追記。  
YAML: `dynamic_cycle_interval.enabled: true` — 数式修正済、即時反映。

### 2.3 分析スキーマ修正 + 新分析セクション追加

**ファイル**: `analysis/306_deep_dive.py`

| 修正 | 内容 |
|---|---|
| `effective_offset_ratio` → `effective_offset_used` | Offset-PnL 相関が n=0 → n=2179 に |
| `fill_timestamp` → `queue_wait_sec` | Fill Speed が n=0 → n=2552 に |
| §8 新設 | `decision_path` / `balance_forced_switch` 交絡分離 |
| §9 新設 | `offset_stages` / `queue_depth_ahead` / `microprice_bias_bps` 分析 |

### 2.4 param_adapter 理由文修正

**ファイル**: `scripts/v460/lib/param_adapter.py`

旧: `→ offset 拡大で AS 回避 (306# A1)`  
新: `→ liveness 優先 — deadlock break (309# F6 修正)`

### 2.5 maker_price.py DocString 修正

旧: `正 → 買い圧力 (bid 厚い → microprice > mid → sell 有利)`  
新: `正 → 買い圧力 (bid 厚い → microprice > mid)` + 309# 注記

---

## 3. Deep Dive 再実行結果 (309# 修正反映後)

### 3.1 §3 Offset-PnL 相関 (修正前: n=0 → 修正後: 動作)

| Side | n | Pearson r | 解釈 |
|---|---|---|---|
| Sell | 1083 | +0.02 | offset と PnL はほぼ無相関 |
| Buy | 1096 | -0.03 | 同上 |

→ offset 比率そのものの大小は直接 PnL を決定しない。AS 発生パターンの方が支配的。

### 3.2 §7 Fill Speed (修正前: n=0 → 修正後: 動作)

| Side | n | mean | median | p90 | fast_fill(<30s) |
|---|---|---|---|---|---|
| Sell | 1264 | 25.8s | 11.7s | 60.4s | 75.3% |
| Buy | 1288 | 25.7s | 12.6s | 59.5s | 72.4% |

→ sell/buy で約定速度に有意差なし。問題は速度ではなく質 (AS 率)。

### 3.3 §8 交絡分離 (新設)

| Side | Group | n | PnL30 | AS率 |
|---|---|---|---|---|
| Sell | forced_switch | 247 | -0.85 | 34.4% |
| Sell | normal | 1017 | -0.25 | 29.3% |
| Buy | forced_switch | 197 | -0.59 | 22.3% |
| Buy | normal | 1091 | -0.23 | 28.7% |

→ **Sell の forced_switch は明確に悪化** (PnL -0.6bps, AS +5pp)。  
→ Buy は forced_switch の方がむしろ AS が低い (22% vs 29%) — 在庫リバランス効果か。

### 3.4 §9 新可観測性 (データ蓄積開始)

Bot デプロイ後わずかなレコード (sell 2件, buy 3件) だが、パイプラインは機能:
- `offset_stages`: 13 ステージの寄与が JSON 展開できている
- `queue_depth_ahead`: 記録されている (現在は 0 = 最良価格)
- `microprice_bias_bps`: 値が記録されている

→ 307# F1 の「スキーマ齟齬」は完全解消。データ蓄積後に本格分析可能。

---

## 4. 307#/308# 指摘のうち、今回対応しなかった項目

| ID | 指摘 | 理由 |
|---|---|---|
| 307# F4 | buy EV 逆転の n=7 問題 | 交絡除去データ蓄積を待ちたい |
| 307# F5 | `none` regime 分解 | warmup/detector-failure 分類は別 issue |
| 307# F7 | spread capture / AS cost 分解 | 305# 枠で対応予定 |
| 308# P0-2 | L2 反転後の再有効化テスト | データ蓄積後に ab テスト設計 |

---

## 5. テスト結果

| テストスイート | 結果 |
|---|---|
| `test_306_proposals.py` | **51 passed** |
| `tests/unit/v460/` 全体 | **4069 passed**, 19 warnings |
| 変更による regression | **0** |

### テスト変更点
- `test_microprice_overrides_to_sell` → `test_microprice_overrides_to_buy` (反転)
- `test_microprice_overrides_to_buy` → `test_microprice_overrides_to_sell` (反転)
- `test_high_sigma_shortens_interval` → `test_high_sigma_lengthens_interval` (反転)
- `test_low_sigma_lengthens_interval` → `test_low_sigma_shortens_interval` (反転)
- `test_clamped_to_min/max`: 入力値を swap (高 σ → max, 低 σ → min)
- `test_yaml_has_microprice_side`: `enabled is True` → `enabled is False`

---

## 6. 修正ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `configs/v460/fill_test.yaml` | L2 無効化, L1 コメント修正 |
| `scripts/v460/lib/side_selector.py` | L2 ロジック反転 (safety mode) |
| `scripts/v460/lib/fill_loop_orchestrator.py` | L1 数式反転 (`σ/σ_ref`) |
| `scripts/v460/lib/maker_price.py` | DocString 修正 |
| `scripts/v460/lib/param_adapter.py` | 理由文修正 (F6) |
| `analysis/306_deep_dive.py` | スキーマ修正 + §8/§9 新設 |
| `tests/unit/v460/test_306_proposals.py` | テスト反転 (6件) |
| `analysis_results/306_deep_dive.json` | 再集計結果 |

---

## 7. 結論

308# Gemini が指摘した **L2 Microprice の「AS Seeker」問題** と **L1 Dynamic Interval の「Taker 戦術」問題** は、いずれも Market Making 理論 (Glosten-Milgrom 1985, Avellaneda-Stoikov 2008) に照らして妥当であった。

特に L2 は `enabled: true` で稼働中であり、**sell AS 率 30.3% の一因**であった可能性が高い。  
本修正により、理論的に正しい方向 (safety mode / cooldown) に矯正された。

307# Codex の F1 (スキーマ齟齬) も完全に修正され、deep dive が初めて全セクション有効に動作した。  
§8 で交絡分離が行われた結果、**sell forced_switch が AS +5pp / PnL -0.6bps の悪化要因**であることが新たに判明した。

次のステップ:
1. データ蓄積後に L2 safety mode の有効化を検討
2. sell forced_switch の制御強化 (AS 高時間帯では forced を抑制)
3. `offset_stages` の本格分析 (どのステージが AS と相関するか)
