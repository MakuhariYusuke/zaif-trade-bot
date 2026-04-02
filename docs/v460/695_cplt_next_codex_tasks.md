# 695# 次期 Codex タスク計画

## 概要

694# Codex Task 1-4 の実装結果と 696# 多視点分析レポートの知見を基に、次に Codex に投入する 4 タスクを策定。
すべて 696# で発見された**収益性リスク**への対処。

## 前提: 694# Codex 実装結果（4/2 適用済み）

| Task | 内容 | 結果 |
|------|------|------|
| T1 | AS-aware trailing skip filter | `as_trailing_tracker.py` 追加、skip_gate_evaluator に接続 |
| T2 | Buy-side cross-venue protection | `maker_risk_guards.py` / `maker_price.py` に実装 |
| T3 | Protocol 688 type safety | `Protocol688Config` 追加、spread bucket / AS severity を config 化 |
| T4 | Offset pipeline math validation | public API 通しの math validation、stage-disable テスト |

**テスト結果**: 33 passed (focused), 4428 passed / 15 skipped (broader)
**修正が必要だった既存テスト**: 3件（cancel reason frozenset, CancelReason Literal, stages_json keys）

---

## 695# タスク一覧

### 優先度マトリクス

| 優先度 | タスク | 根拠 | 期待効果 | 工数 |
|--------|--------|------|----------|------|
| **P0-1** | trend_5s veto counterfactual | 14% cancel → 価値不明 | cancel 最適化 ±0.2bps | 3-4h |
| **P0-2** | 低スプレッド AS 防御ゲート | AS 64.3% @ 0-1500 | リスク抑制 +0.3bps | 4-5h |
| **P1-1** | Ranging regime AS 調査 & 適応型ガード | AS 36.5%→56.5% | regime別最適化 | 5-6h |
| **P1-2** | Fill record observability 強化 | guard pipeline 記録不足 | 分析基盤改善 | 3-4h |

---

### Task 1: trend_5s veto counterfactual 分析 (P0-1)

**背景**: 696# 視点3 で判明。`trend_5s_sell_guard_veto` が 4/2 の cancel reason の 14%（15/107件）。
4/1 では 0.8%（3/364件）だったものが急増。threshold_bps=0.5 が aggressive すぎる可能性がある。

**問題**: veto された注文が本当に損失を回避できたのか不明。counterfactual PnL が未計測。

**実装方針**:
- veto'd records の counterfactual PnL を mid 価格変化から推定
- control group（boost で通過した sell fills）との比較
- Net impact = veto benefit − opportunity cost
- Protocol 688 フレームワークに `695_trend5s` として登録

**成果物**:
- `scripts/v460/analysis/sections/section_trend_5s_counterfactual.py`
- `tests/unit/v460/test_695_trend5s_counterfactual.py`

---

### Task 2: 低スプレッド AS 防御ゲート (P0-2)

**背景**: 696# 視点4 が特定した最大リスク。0-1500 JPY spread bucket で AS rate が 64.3%（14/22 fills）、PnL = -0.461 bps。686# の `min_spread_atr_cap` 1.2 への緩和で狭スプレッド参入が増え、逆選択に晒されている。

**仮説**: 低スプレッド = 高流動性 = informed flow が多い。現行ガードはスプレッドで条件分岐しない。

**実装方針**:
- `SpreadConditionalASGuard`: spread < threshold 時に EV penalty を追加
- Hard block ではなく offset 調整（ev_adjustment）
- `enabled: false` で observe → ログ蓄積後に有効化

**成果物**:
- 既存 guard chain への `SpreadConditionalASGuard` 追加
- `tests/unit/v460/test_695_spread_as_guard.py`

---

### Task 3: Ranging regime AS 調査 & 適応型ガード (P1-1)

**背景**: 696# 視点7 が検出。ranging regime で AS rate が 36.5%（4/1）→ 56.5%（4/2）に悪化。PnL も +0.600 → -0.362 bps。trending_down は逆に改善。

**仮説**: `skip_gate_bypass_mode=true`（686#）により、ML が ranging で reject していた注文が通過。skip_gate は全体 MI≈0 だが、ranging 限定では予測力があった可能性。

**実装方針**:
- Part A: regime × spread bucket × AS rate のクロス集計分析
- Part B: `RegimeGuardAdapter` — regime 別の EV threshold premium と penalty multiplier
- 既存ガードの composition（合成）、置換しない

**成果物**:
- `scripts/v460/analysis/sections/section_regime_as_deep_dive.py`
- `tests/unit/v460/test_695_regime_as_analysis.py`

---

### Task 4: Fill record observability 強化 (P1-2)

**背景**: 696# の分析過程でフィールド名の不一致やガード判断のメタデータ不足が発覚。entry_gate の EV 推定値、regime 状態、trend_5s の値など、guard pipeline の判断過程が記録されていない。

**実装方針**:
- `guard_pipeline_result` 構造体をフィールドに追加
- schema_version=2 で後方互換性を維持
- 既に計算済みのデータのシリアライズのみ（新規計算なし）

**成果物**:
- `fill_record_builder.py` への `guard_pipeline_result` 追加
- `tests/unit/v460/test_695_fill_record_enrichment.py`

---

## 実行計画

- Task 1, 2 は独立 → 並列で Codex に投入可能
- Task 3 は Task 2 の guard パターンに依存 → Task 2 完了後
- Task 4 は独立だが Task 1-3 の分析結果を反映可能 → 最後に投入

## Codex プロンプト

- `prompts/codex_695_task1_trend5s_counterfactual.md`
- `prompts/codex_695_task2_spread_as_guard.md`
- `prompts/codex_695_task3_regime_as_investigation.md`
- `prompts/codex_695_task4_fill_record_enrichment.md`

すべて `638970c9f` でコミット済み。
