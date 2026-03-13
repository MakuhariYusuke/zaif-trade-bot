# 240# Toxicity Budget: binary skip → continuous adverse-selection budget (232# §2.2)

> **日付**: 2026-03-03
> **前提**: 239# (`f7e1ba91c`) — InfeasibleQuoteError + 制約前方移動
> **テスト**: 3328 passed (3286 + 42 新規)

---

## 1. 課題

232# §2.2 (Codex) / 233# §2 (Gemini) が共に指摘:

> `dynamic_kill` や `skip_gate` は hard block に寄りがち。
> 情報優位フローは連続的で、二値制御だけだと「止まりすぎる」「止まるべき時に全部外れる」の両方を起こす。

**既存の dynamic_kill**: rolling PnL < threshold → 即座に KILL (全サイクル停止)。
 threshold 以上なら完全に通過。中間状態がなく、Glosten-Milgrom の逆選択プレミアムを段階的に反映できない。

---

## 2. 解決策: Toxicity Budget (4段階 Glosten-Milgrom 応答)

### 2.1 正規化 Toxicity スコア

```
score = max(0, rolling_mean / threshold)    # threshold < 0
```

| score 範囲 | ゾーン | 対応 |
|-----------|--------|------|
| 0 – warn_level (0.3) | GREEN | 通常参加 |
| warn_level – caution_level (0.7) | YELLOW | offset 拡大 |
| caution_level – 1.0 | ORANGE | offset 拡大 + 確率的参加 (1/N) |
| ≥ 1.0 | KILL | 完全停止 (従来互換) |

### 2.2 線形補間

- **YELLOW**: `offset_mult` を `warn_offset` → `caution_offset` で線形補間
- **ORANGE**: `offset_mult` + `participation_rate` を共に線形補間
  - `participation_rate`: 1.0 → `min_participation` (default 0.33)

### 2.3 市場理論根拠

**Glosten-Milgrom (1985)**:
- 情報優位フローが増加 → bid-ask spread (逆選択プレミアム) 拡大で対応
- 二値停止ではなく spread 拡大が第一防衛線

**Kyle (1985)**:
- 情報は漸次的に価格に反映される → 段階的応答が統計的最適

**Kelly Criterion**:
- EV < 0 の取引は bet size = 0 が最適
- EV ≈ 0 (ORANGE zone) では bet size を縮小 (participation rate 低下)

---

## 3. 実装

### 3.1 DynamicKillManager 拡張 (`ztb/risk/sell_dynamic_kill.py`)

- `ToxicityLevel` enum: GREEN / YELLOW / ORANGE / KILL
- `ToxicityAssessment` frozen dataclass: level, score, offset_mult, participation_rate, threshold_used, rolling_mean
- `DynamicKillConfig`: 7 新フィールド (toxicity_budget_enabled, warn/caution levels, offset mults, min_participation)
- `assess_toxicity(regime)` メソッド: **副作用なし** (check_kill と独立)
  - cooldown 中 → KILL 固定
  - レジーム別閾値対応

### 3.2 CycleGateAggregator 段階的応答 (`scripts/v460/lib/cycle_gate_aggregator.py`)

- `CycleGateResult`: `toxicity_offset_mult` (default 1.0), `participation_rate` (default 1.0) 追加
- Gate 4/5: kill 判定 + YELLOW/ORANGE toxicity → `_apply_toxicity_graded()` で段階的応答
  - KILL → 従来通り blocked
  - YELLOW/ORANGE → blocked にせず offset_mult / participation_rate を設定
  - GREEN → GREEN のまま (gate blocked でも theory 上ここに来ない)
- 234# degraded_liquidation との優先順位: toxicity graded > degraded_liquidation > hard block

### 3.3 Orchestrator (`scripts/v460/lib/fill_loop_orchestrator.py`)

- `_assess_buy_toxicity()` / `_assess_sell_toxicity()`: 副作用なしの assessment 取得
- `gate.evaluate()` に `buy_toxicity` / `sell_toxicity` 引数追加
- **participation_rate チェック**: gate 通過後、`random.random() > participation_rate` なら probabilistic skip
- `toxicity_offset_mult` → `run_single_cycle()` に伝搬

### 3.4 Executor (`scripts/v460/lib/fill_cycle_executor.py`)

- `run_single_cycle(toxicity_offset_mult=1.0)` パラメータ追加
- `_apply_offset_multiplier()` で toxicity offset を適用 (trending_offset_mult の直後)

### 3.5 cancel_reasons (`scripts/v460/lib/cancel_reasons.py`)

- `TOXICITY_PARTICIPATION_SKIP = "toxicity_participation_skip"` 追加

---

## 4. デフォルト設定 (後方互換)

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `toxicity_budget_enabled` | `False` | 無効 → 従来 binary kill のまま |
| `toxicity_warn_level` | `0.3` | YELLOW 開始 (score) |
| `toxicity_caution_level` | `0.7` | ORANGE 開始 (score) |
| `toxicity_warn_offset_mult` | `1.0` | YELLOW 入口の offset 乗数 |
| `toxicity_caution_offset_mult` | `2.0` | ORANGE 入口の offset 乗数 |
| `toxicity_kill_offset_mult` | `3.0` | KILL 直前の offset 乗数 |
| `toxicity_caution_min_participation` | `0.33` | ORANGE 最悪時の最低参加率 |

**デフォルト disabled** のため、`toxicity_budget_enabled=True` を YAML config で設定しない限り既存動作に影響なし。

---

## 5. テスト (42 新規)

| クラス | テスト数 | 内容 |
|--------|---------|------|
| TestToxicityLevel | 2 | enum 基本属性 |
| TestToxicityAssessment | 3 | frozen, slots, green defaults |
| TestDynamicKillConfigToxicity | 3 | フィールド存在、デフォルト無効、ゾーン順序 |
| TestAssessToxicityGreen | 4 | データ不足、正の PnL、budget 無効、閾値以下 |
| TestAssessToxicityYellow | 3 | 入口、中間、線形補間 |
| TestAssessToxicityOrange | 3 | 入口、中間補間、min_participation 下限 |
| TestAssessToxicityKill | 3 | 閾値到達、超過、cooldown 中 |
| TestAssessToxicityRegime | 1 | レジーム別閾値でスコア変動 |
| TestAssessToxicityNoSideEffect | 2 | 複数回呼出不変、check_kill 非干渉 |
| TestGateAggregatorToxicityFields | 2 | デフォルト値、cancel_reason マッピング |
| TestGateAggregatorGradedResponse | 4 | YELLOW通過、ORANGE参加率、KILL blocked、None→legacy |
| TestCancelReasons | 2 | 定数存在、AUDIT_CANCEL_REASONS 所属 |
| TestExecutorToxicityParam | 2 | パラメータ存在、ソースコード適用確認 |
| TestOrchestratorToxicityAssess | 3 | メソッド存在、gate 引数、participation ロジック |
| TestGlostenMilgromTheory | 4 | offset 単調増加、participation 単調減少、GREEN=full、KILL=zero |
| TestBuyManagerToxicity | 1 | BuyDynamicKillManager 継承検証 |

---

## 6. 変更ファイル

| ファイル | 変更種別 |
|---------|---------|
| `ztb/risk/sell_dynamic_kill.py` | ToxicityLevel, ToxicityAssessment, config + assess_toxicity() |
| `scripts/v460/lib/cycle_gate_aggregator.py` | toxicity fields + _apply_toxicity_graded() |
| `scripts/v460/lib/fill_loop_orchestrator.py` | _assess_xxx_toxicity() + participation check |
| `scripts/v460/lib/fill_cycle_executor.py` | toxicity_offset_mult param + apply |
| `scripts/v460/lib/cancel_reasons.py` | TOXICITY_PARTICIPATION_SKIP |
| `tests/unit/v460/test_240_toxicity_budget.py` | 42 新規テスト |
| `tests/unit/v460/test_145_structural_fixes.py` | AUDIT_CANCEL_REASONS 期待値更新 |
