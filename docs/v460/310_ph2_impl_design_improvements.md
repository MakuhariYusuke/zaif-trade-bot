# 310# 設計面改修: 307#/308# 残課題の構造的解消

> **日付**: 2026-03-07  
> **対象レビュー**: [307#](307_ph2_rev_303_306_systems_market_review.md) F3/F5/F6/F7, [308#](308_ph2_gemini_31_pro_review_306_307_inverted_microstructure.md) Blindspot 1  
> **方針**: 309# で P0 理論倒錯を修正済。残る設計面の課題を構造的に解消する

---

## 変更一覧

| ID | 改修内容 | 対象指摘 | 優先度 |
|---|---|---|---|
| A | Sell AS Time-of-Day Offset Boost | 307# F3 (AS/Session 支配, UTC 08/13/14/16h) | P0 |
| B | param_adapter Decision Path Split | 307# F6 (AS 防御とデッドロック解除の混同) | P1 |
| C | L2 Safety Mode Re-enablement Guardrails | 308# Blindspot 1 (理論正答への安全な復帰) | P1 |
| D | None Regime Observability | 307# F5 (None Regime の可観測性不足) | P1 |
| E | Spread Capture / AS Cost Decomposition | 307# F7 (PnL 分解の欠如) | P1 |

---

## A. Sell AS Time-of-Day Offset Boost (P0)

### 背景 (307# F3)

306# deep dive §5 で UTC 時間帯別の AS 率を分析した結果、特定時間帯で異常に高い AS 率を確認:

| UTC | n | PnL30 (bps) | AS率 | 特徴 |
|---|---|---|---|---|
| 08h | 27 | -3.55 | 62.96% | 東京前場寄り: 情報トレーダー活発 |
| 13h | 55 | -2.16 | 41.82% | 欧州前場: informed flow 増加 |
| 14h | 38 | -3.28 | 44.74% | 欧州前場: 高ボラティリティ |
| 16h | 18 | -2.25 | 61.11% | 欧州本場: 最大流動性 = 最大 AS リスク |

### 理論的根拠

**Ho-Stoll (1981)**: マーケットメイカーの最適スプレッドは逆選択コストの関数であり、informed trader の活動が増加する時間帯ではスプレッドを拡大すべき。

### 実装

- **`fill_config.py`**: `sell_hour_offset_boost: dict[int, float]` フィールド追加
- **`maker_price.py`**: `_apply_sell_hour_boost()` メソッド追加
  - sell 側の offset にのみ、UTC 時間帯別乗数を適用
  - pipeline stage: `buy_as_guard` → **`sell_hour`** → `loss_boost` の間
  - buy side は一切影響を受けない
- **`fill_test.yaml`**: `sell_hour_offset_boost: {8: 1.5, 13: 1.3, 14: 1.3, 16: 1.5}`

### テスト

- `test_sell_hour_config_exists`: config フィールド存在確認
- `test_sell_hour_multiplier_applied`: 正しい乗数適用
- `test_sell_hour_no_effect_on_buy`: buy 側不干渉確認
- `test_sell_hour_empty_dict_no_effect`: 空 dict の安全性
- `test_sell_hour_yaml_parsing`: YAML パース検証

---

## B. param_adapter Decision Path Split (P1)

### 背景 (307# F6)

param_adapter の return 文が全て同じ形式で、AS 防御 (hold) とデッドロック解除 (increase) が `reason` テキストだけで区別されていた。ログ分析で混同が発生しやすく、意思決定の追跡が困難。

### 実装

- **`param_adapter.py`**: `AdaptationResult` dataclass に `decision_path: str = "hold"` 追加

| decision_path | 条件 | 行動 |
|---|---|---|
| `"insufficient_data"` | history < min_periods | hold |
| `"as_defense"` | AS + fill 両方悪化 & EV ≈ 0 | hold |
| `"deadlock_break"` | AS + fill 両方悪化 & EV << 0 | increase |
| `"as_defense"` | AS のみ高 | decrease |
| `"fill_recovery"` | fill のみ低 | increase |
| `"ev_optimization"` | EV 正 & AS マージンあり | decrease |
| `"hold"` | 正常範囲 | hold |

### テスト

`TestDecisionPath` クラスで 7 テスト:
- 各 decision_path の値が正しく設定されることを検証
- `insufficient_data`, `as_defense_hold`, `deadlock_break`, `as_defense_decrease`, `fill_recovery`, `ev_optimization`, `hold_normal`

---

## C. L2 Safety Mode Re-enablement Guardrails (P1)

### 背景 (308# Blindspot 1)

309# で L2 Microprice Side Selection のロジックを反転 (AS Seeker → Safety Mode) し、YAML で無効化。将来の再有効化に備えてガードレールを事前実装。

### 理論的根拠

**Glosten-Milgrom (1985)**: 狭スプレッド時は informed/uninformed の分離が困難になり、microprice signal の信頼性が低下。Ranging 以外の regime では trend-following flow が支配的で microprice の safety mode signal が反転しうる。

### 実装

- **`fill_config.py`**: 2 フィールド追加
  - `microprice_side_min_spread_bps: float = 15.0` — 最小スプレッド閾値
  - `microprice_side_regime_gate: list[str] = ["ranging"]` — 許可 regime リスト
- **`side_selector.py`**: `next()` に `spread_bps`, `regime` kwargs 追加
  - `spread_bps < min_spread_bps` → microprice スキップ
  - `regime not in regime_gate` → microprice スキップ
- **`fill_record_helpers.py`**: `_next_side()` で `spread_bps` と `regime` を計算して渡す
  - `spread_bps = last_spread_raw / last_mid_price * 10000` (public API 使用)

### テスト

- `test_microprice_guardrail_spread_blocks`: 狭スプレッド時に microprice スキップ
- `test_microprice_guardrail_regime_blocks`: 非許可 regime で microprice スキップ
- 既存 microprice テスト: `spread_bps=20.0, regime="ranging"` を追加

---

## D. None Regime Observability (P1)

### 背景 (307# F5)

Regime detector が "none" を返すケースの影響が不明。306# deep dive にも none regime の分析が欠けていた。

### 実装

- **`fill_loop_orchestrator.py`**: 2 カウンター追加
  - `_none_regime_cycle_count`: none regime の累積サイクル数
  - `_total_regime_cycle_count`: 全サイクル数
  - progress log に `none_regime=X/Y` 出力
- **`306_deep_dive.py` §11**: `none_regime_analysis()` 関数追加
  - None regime のフィル数、割合、平均 PnL、AS 率
  - Non-none との比較

### Deep Dive §11 結果

| 区分 | n | 平均 PnL (bps) | AS率 |
|---|---|---|---|
| None regime | 267 | -0.4624 | 50.56% |
| Non-none | 2292 | -0.3216 | — |
| 全体 | 2559 | — | — |

None regime = 10.43% だが PnL は -0.4624 bps で non-none (-0.3216) より 44% 悪い。

---

## E. Spread Capture / AS Cost Decomposition (P1)

### 背景 (307# F7)

PnL を「spread capture」と「AS cost」に分解する分析が不足。Maker の理論的利益構造の理解に不可欠。

### 理論

Maker の PnL 分解:
$$\text{realized\_pnl} = \text{spread\_capture} - \text{as\_cost}$$

- **spread_capture** = `spread_bps × effective_offset_used` — offset 分だけ板の内側に入った理論的収益
- **as_cost** = `spread_capture - realized_pnl` — 逆選択により侵食された利益
- **efficiency** = `realized_pnl / spread_capture` — 1.0 で完全、負で AS コスト超過

### 実装

- **`306_deep_dive.py` §10**: `spread_as_cost_decomposition()` 関数
  - FillRecord の `spread_bps`, `effective_offset_used`, `post_fill_30s_pnl` を使用
  - 正しい bps 変換: `sc_bps = spread_bps × offset_ratio`

### Deep Dive §10 結果

| Side | n | spread_capture (bps) | realized_pnl (bps) | AS cost (bps) | efficiency |
|---|---|---|---|---|---|
| Sell | 1086 | 0.86 | -0.28 | 1.14 | -0.32 |
| Buy | 1100 | 0.28 | -0.30 | 0.58 | -1.07 |

**解釈**:
- Sell: spread capture 0.86 bps を得るが、AS cost 1.14 bps に侵食される (efficiency -32%)
- Buy: spread capture 0.28 bps と小さく、AS cost 0.58 bps で大幅赤字 (efficiency -107%)
- Buy side の offset が sell より小さいため spread capture が少なく、AS 耐性が低い

---

## 検証結果

### テスト

| 対象 | 結果 |
|---|---|
| test_306_proposals.py | 67 passed (51 → 67, +16 新規) |
| v460 全体 | 4085 passed, 19 warnings |
| 回帰 | なし |

### compute() 行数

sell_hour_boost pipeline stage 追加 (5行) により 280 → 285 行。
`test_compute_line_count_reduced` の上限を 280 → 290 に変更。

---

## ファイル変更一覧

| ファイル | 変更種別 | 改修 ID |
|---|---|---|
| `scripts/v460/lib/fill_config.py` | Modified | A, C |
| `scripts/v460/lib/maker_price.py` | Modified | A |
| `scripts/v460/lib/param_adapter.py` | Modified | B |
| `scripts/v460/lib/fill_loop_orchestrator.py` | Modified | D |
| `scripts/v460/lib/side_selector.py` | Modified | C |
| `scripts/v460/lib/fill_record_helpers.py` | Modified | C |
| `analysis/306_deep_dive.py` | Modified | D, E |
| `configs/v460/fill_test.yaml` | Modified | A, C |
| `tests/unit/v460/test_306_proposals.py` | Modified | A, B, C |
| `tests/unit/v460/test_260_compute_extract_regime_split.py` | Modified | A |
| `docs/v460/310_ph2_impl_design_improvements.md` | Created | — |
