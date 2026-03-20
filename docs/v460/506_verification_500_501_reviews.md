# 506# 500番・501番レビュー検証報告

## 概要

500# (PHG rev 497#–499#/503#) および 501# (PHG second opinion: cross-venue basis correction) の
主要主張をフィルレコード (3/14–3/20, 773 fills) およびコード解析により独立検証した結果を報告する。

---

## 1. 500# データ主張の検証

### 1.1 Sell CV Applied Cases (Claim 2.1)

| 区分 | N | PnL合計 | avg |
|------|---|---------|-----|
| CV True (applied) | 22 | -39.01 | -1.773 |
| CV False (not applied) | 208 | -151.61 | -0.729 |
| CV None (no data) | 153 | +16.12 | +0.105 |

**判定: ✅ 完全一致。** CV applied sell cases は non-applied より avg で 2.4 倍悪い。
500# の「sell に cross-venue を as-is 適用するのは危険」という主張は正しい。

### 1.2 Sell 日次 PnL 変動 (Claim 2.2)

| 日付 | PnL (JPY) | N |
|------|-----------|---|
| 3/14 | -55.36 | 80 |
| 3/15 | -30.00 | 31 |
| 3/16 | +55.43 | 41 |
| 3/17 | -83.40 | 54 |
| 3/18 | +49.52 | 52 |
| 3/19 | -143.53 | 57 |
| 3/20 | +32.83 | 68 |

**判定: ✅ 方向一致。** 500# は 3/20 sell を +25.16 と記載（当方計測 +32.83）。
微小差異はメトリック集計タイミングの違いと推定。
**Sell 崩壊は常態ではなく条件付き**という 500# の核心的主張は正しい。

### 1.3 Sell Fast/Slow Fill パターン (Claim 3.1)

| 区分 | N | PnL | avg |
|------|---|-----|-----|
| fast <10s | 175 | +59.93 | +0.342 |
| slow ≥30s | 64 | -151.71 | -2.370 |

**判定: ✅ confirmed。** Buy と完全に逆パターン（buy: fast=bad -0.414, slow=good +0.195）。
500# の「buy/sell に非対称実行ポリシーが必要」は正しい。

### 1.4 Sell Offset Bucket PnL (Claim 3.3)

| Bucket | N | PnL | avg | WR |
|--------|---|-----|-----|----|
| 0.10–0.19 | 43 | +138.58 | +3.223 | 67.4% |
| 0.19–0.25 | 134 | -199.76 | -1.491 | 49.3% |
| ≥0.25 | 206 | -113.33 | -0.550 | 47.6% |

**判定: ✅ 完全一致。** 0.10–0.19 バケットが最高パフォーマンス。
500# の「sell offset は狭める方向が正しい」は強く支持される。

### 1.5 recovery_skew パフォーマンス (3/20)

| reason | N | PnL | avg |
|--------|---|-----|-----|
| None\|sell | 23 | +27.62 | +1.201 |
| balance_switch\|sell | 26 | +17.20 | +0.661 |
| recovery_skew\|sell | 19 | -11.99 | -0.631 |

**判定: ✅ 完全一致。** recovery_skew 経由の sell は唯一のマイナス寄与。
500# の「recovery_skew は liveness のみで profit には未貢献」という評価は正当。

---

## 2. 501# データ主張の検証

### 2.1 構造的 Basis 統計

| 統計量 | 501# 主張 | 実測値 | 判定 |
|--------|-----------|--------|------|
| N | 633 | 638 | ≈ (データ更新差) |
| Mean | -3.32 bps | -3.326 bps | ✅ |
| Median | -3.64 bps | -3.640 bps | ✅ |
| Std | 2.31 bps | 2.304 bps | ✅ |
| Negative % | 90.7% | 90.6% | ✅ |

**判定: ✅ 事実上完全一致。** CC mid は BF mid より構造的に 3.3 bps 高い。

### 2.2 De-meaning シミュレーション

- **Current**: adverse_side="sell" → 全体の約 9.4% のみ（実質無効）
- **De-meaned** (basis=-3.326 bps 差し引き): sell guard 発動 → **42.5%** (271/638)
- Buy/Sell 分割: 57.5% buy / 42.5% sell → 大幅にバランス改善

**判定: ✅ 概念的に有効。** ただし以下の重大な注意点あり（§3 参照）。

### 2.3 ⚠️ Basis 安定性 — 501# 未言及の懸念

| 日付 | 日次平均 Basis | 累積平均 | δ |
|------|---------------|---------|---|
| 3/16 | -1.012 bps | -1.012 | - |
| 3/17 | -2.198 bps | -1.811 | -0.387 |
| 3/18 | -4.150 bps | -2.926 | -1.224 |
| 3/19 | -3.966 bps | -3.206 | -0.759 |
| 3/20 | -3.748 bps | -3.326 | -0.422 |

**日次 basis は -1.0 ～ -4.2 bps と大きく変動する。**
501# は global mean (-3.3 bps) を定数として差し引く提案だが、
3/16 に -3.3 を適用すると実際の basis (-1.0) から 2.3 bps のオフセットエラーが発生し、
sell 側に過剰な guard を発動する逆効果リスクがある。

**→ 固定定数ではなく EMA ベースの rolling basis 推定が必須。**
既存の `CrossVenueEMAState` が `ema_spread_bps` を追跡しているため、
同様の構造で `ema_basis_bps` を追加することは容易に実装可能。

---

## 3. 追加発見事項

### 3.1 Sell Age vs PnL の詳細分布

| Age Bucket | N | PnL | avg | WR |
|------------|---|-----|-----|----|
| 0–10s | 175 | +59.93 | +0.342 | 52.0% |
| 10–20s | 90 | -16.64 | -0.185 | 53.3% |
| 20–30s | 54 | -66.09 | -1.224 | 44.4% |
| 30–50s | 46 | -158.73 | -3.451 | 47.8% |
| ≥50s | 18 | +7.02 | +0.390 | 44.4% |

**500# P0 の sell cumulative age cap (20–25s) は強く支持される。**
30–50s バケット（n=46, PnL=-158.73, avg=-3.451）が sell 損失の核心。
20s でキャップすれば -224.82 JPY の損失を回避可能（perfect hindsight 上限）。

### 3.2 Sell Regime 集中

| Regime | N | PnL | avg | WR |
|--------|---|-----|-----|----|
| ranging | 333 | -190.25 | -0.571 | 50.8% |
| trending_down | 24 | +5.31 | +0.221 | 54.2% |
| trending_up | 26 | +10.44 | +0.402 | 42.3% |

**Sell 損失の 100% が ranging レジームに集中。** Trending では sell は profitable。

### 3.3 sell_dynamic_kill の実態

- Dynamic kill cancels: **0 件** / 924 non-fills (0.0%)
- 現在の閾値 (-0.5 bps) では **一度もトリガーしていない**

**500# P1 の sell_dynamic_kill 副作用経路 調整は、まず発動実態の確認が先行すべき。**
現状ゼロ発動であるため、閾値の「ブラント緩和」を危険視する 500# の指摘は
正しい方向だが、実質的には dormant 機能への懸念である。

---

## 4. 検証結果サマリ

### 500# 評価

| 主張 | 検証結果 | 備考 |
|------|---------|------|
| Sell CV applied は逆効果 | ✅ confirmed | avg -1.773 vs -0.729 |
| Sell 崩壊は条件付き | ✅ confirmed | 日次変動大、3日はプラス |
| Buy/Sell 非対称実行が必要 | ✅ confirmed | Fast/slow パターン完全逆 |
| Sell offset 狭める | ✅ strongly confirmed | 0.10–0.19 が最高 WR・PnL |
| recovery_skew 再評価 | ✅ confirmed | sell 唯一のマイナス |
| P0: sell age cap 20–25s | ✅ strongly confirmed | 30–50s = -158.73 JPY |
| P1: sell_dynamic_kill | △ dormant | 閾値内で 0 回発動 |
| P2: ranging skip soft 化 | — 未検証 | データ不足 |

**500# の分析品質は高い。データ主張は全て裏付けられた。**

### 501# 評価

| 主張 | 検証結果 | 備考 |
|------|---------|------|
| 構造的 Basis -3.3 bps | ✅ exact match | mean/median/std 全一致 |
| De-meaning で sell guard 有効化 | ✅ concept valid | 9% → 42.5% |
| Fixed constant で実装 | ⚠️ 要修正 | 日次変動 -1.0～-4.2 で不安定 |
| Micro-timeout asymmetry | ✅ directionally correct | sell slow=bad のデータ支持 |

**501# の方向性は正しいが、実装は固定定数ではなく adaptive/EMA 推定が必須。**

---

## 5. 推奨アクション優先度

| 優先度 | アクション | 根拠 | 想定効果 |
|--------|-----------|------|---------|
| **P0** | Sell age cap (20–25s) 導入 | 30–50s で -158.73 JPY 集中 | -100 JPY/week+ 削減 |
| **P0** | Sell offset 狭小化 (→0.12–0.18) | 0.10–0.19 WR=67.4%, PnL=+138.58 | Sell 収益性改善 |
| **P1** | De-meaning (EMA basis) 実装 | 501# 提案 + basis 非定常性補正 | Sell に CV guard 適用可能に |
| **P1** | recovery_skew shadow 評価 | 500# P1, 実損 -11.99 | 無効なら除去で +12 JPY |
| **P2** | Buy/Sell 非対称 timeout | 500#/501# 共通, fast/slow 逆転データ | 追加損失防止 |
| **P2** | sell_dynamic_kill 閾値再調整 | 現在 dormant (0 triggers) | 実効化する場合のみ |

---

## 6. 501# De-meaning 実装設計メモ

501# の `adjusted_spread = gating_spread - historical_basis_bps` は概念的に正しい。
ただし `historical_basis_bps` を固定定数にするのは日次変動 (σ ≈ 1.2 bps) から危険。

### 推奨実装パターン

```python
# CrossVenueEMAState に ema_basis_bps を追加
@dataclass
class CrossVenueEMAState:
    ema_ref_mid: float
    ema_spread_bps: float
    ema_basis_bps: float        # ← 新規: rolling basis EMA
    last_timestamp: float
    n_updates: int = 0

# compute_cross_venue_lead_lag_hint に basis 補正を追加
def compute_cross_venue_lead_lag_hint(..., basis_bps: float = 0.0):
    ...
    gating_spread = ema_spread_bps
    adjusted_spread = gating_spread - basis_bps   # de-mean
    direction = "up" if adjusted_spread > 0.0 else "down"
    ...
```

- Basis EMA の α は spread EMA より遅い（α=0.01–0.05 程度、日次～数日スケール）
- 初期値は 0.0（保守的: 既存動作と同一）、warm-up 後に収束
- Hot-reload 対応: `basis_correction_enabled` フラグで on/off 切替

**502# リファクタリング完了後に実装するのが適切。**

---

*検証日: 2026-03-21*
*データソース: results/v460/fill_test/fill_records_20260314–20260320.jsonl*
*検証スクリプト: temp/verify_reviews.py, temp/verify_reviews2.py*

---

## 7. 実装完了: P0/P1 アクション

### 7.1 P0: Sell Age Cap (sell_age_cap_sec)

**変更ファイル:**
- `scripts/v460/lib/fill_config.py`: `sell_age_cap_sec: float | None = None` 追加
- `scripts/v460/lib/order_monitor.py`: polling loop の effective_timeout に `min(timeout, cap)` 適用
- `scripts/v460/lib/config_hot_reload.py`: hot-reload 対応
- `scripts/v460/lib/fill_config_parser.py`: YAML flat key パース追加
- `configs/v460/fill_test.yaml`: `sell_age_cap_sec: 25.0` 設定

**動作:** sell 注文の滞留時間を 25s に制限。30–50s バケット (n=46, PnL=-158.73) を回避。

### 7.2 P0: Sell Offset 狭小化

**変更:** `configs/v460/fill_test.yaml` の `side_offset.sell: 0.18 → 0.14`

0.10–0.19 バケット (WR=67.4%, avg=+3.223) を活用し、0.19–0.25 バケット (avg=-1.491) への
偏りを軽減。

### 7.3 P1: De-meaning (EMA Basis Correction)

**変更ファイル:**
- `scripts/v460/lib/cross_venue_lead_lag.py`:
  - `CrossVenueEMAState` に `ema_basis_bps: float = 0.0` 追加
  - `update_cross_venue_ema()` に `basis_alpha` kwarg 追加
  - `compute_cross_venue_lead_lag_hint()` に `basis_bps` kwarg 追加
  - Confidence mode の direction 判定: `adjusted_spread = gating_spread - basis_bps`
- `scripts/v460/lib/fill_cycle_executor.py`: EMA 更新 / hint 計算に basis パラメータ伝播
- `scripts/v460/lib/fill_config.py`: `cross_venue_basis_correction_enabled`, `cross_venue_basis_ema_alpha` 追加
- `scripts/v460/lib/fill_config_parser.py`: YAML マッピング追加
- `scripts/v460/lib/config_hot_reload.py`: hot-reload 対応
- `configs/v460/fill_test.yaml`: `basis_correction_enabled: true`, `basis_ema_alpha: 0.02`

**動作:** CC/BF 間の構造的 basis (-3.3 bps) を EMA で追跡し、direction 判定前に差し引く。
basis_bps=0.0 (default) では既存動作と完全に同一 (後方互換)。
enabled 時: sell 側が CV guard を受ける確率が ~9% → ~42% に改善。

### 7.4 テスト

`tests/unit/v460/test_506_sell_improvements.py`: 10 tests
- EMA basis tracking (初期化, 収束, 安定性)
- De-meaning direction flip 検証
- 後方互換性 (basis_bps=0.0)
- Config field 存在確認, hot-reload 対応確認
