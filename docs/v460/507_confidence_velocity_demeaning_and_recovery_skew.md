# 507# Confidence/Velocity De-meaning 統一 & Recovery Skew 縮小

## 概要

506# で導入した cross-venue basis correction (de-meaning) の
**confidence 計算と velocity agreement 判定** に残っていた不整合を修正。
加えて 500# P1 の recovery_skew offset multiplier を縮小。

---

## 1. 修正内容

### 1.1 Confidence 計算のバグ (506# 実装の欠陥)

**問題:**

506# de-meaning で `direction` 判定を `adjusted_spread = gating_spread - basis_bps` に
切り替えたが、confidence 計算は以下のまま放置されていた:

```python
base_conf = min(1.0, max(floor, abs(gating_spread) / ref))
```

`gating_spread ≈ -3.3bps` (常に一定) のため:
- `base_conf = |3.3| / 3.0 = 1.0` → **常に最大値にクランプ**
- sell 側 guard の boost 強度が偏差の大きさに関係なく常にフルブースト

つまり basis から +0.1bps しか逸脱していなくても、+3.0bps 逸脱した時と同じ
最大強度の offset boost が適用される。

**修正:**

```python
_conf_spread = adjusted_spread if basis_bps != 0.0 else gating_spread
base_conf = min(1.0, max(floor, abs(_conf_spread) / ref))
```

de-meaning 有効時は `|adjusted_spread|` (basis からの偏差幅) で confidence を計算。

| 状態 | 修正前 conf | 修正後 conf | 効果 |
|------|------------|------------|------|
| adjusted=+0.5bps (微小偏差) | 1.0 | 0.33 (floor) | 適切に抑制 |
| adjusted=+1.5bps (中偏差) | 1.0 | 0.50 | 比例的 |
| adjusted=+3.0bps (大偏差) | 1.0 | 1.00 | 変化なし |

### 1.2 Velocity Agreement 判定のバグ (506# 実装の欠陥)

**問題:**

velocity と spread の方向一致判定が以下のまま:

```python
elif gating_spread * reference_velocity_bps > 0.0:
    vel_factor = 1.0   # agrees
```

`gating_spread ≈ -3.3bps` (常にマイナス) のため:
- BF 上昇中 (velocity > 0): `-3.3 * (+) < 0` → disagreement (vel_factor=0.5)
- BF 下降中 (velocity < 0): `-3.3 * (-) > 0` → agreement (vel_factor=1.0)

De-meaning 後に `adjusted_spread > 0` なら「BF が CC に対して上方逸脱」を意味し、
velocity が正 (BF 上昇) なら**同方向**であるべき。しかし上記では逆判定される。

**修正:**

```python
_vel_spread = adjusted_spread if basis_bps != 0.0 else gating_spread
elif _vel_spread * reference_velocity_bps > 0.0:
    vel_factor = 1.0
```

### 1.3 Recovery Skew Offset Multiplier 縮小 (500# P1)

**変更:** `recovery_skew_offset_mult: 2.0 → 1.5`

**根拠:** 506# 独立検証で recovery_skew|sell の PnL = -11.99 (avg = -0.631)。
`ceiling × 2.0` は保守的すぎて fill 品質を犠牲にしている。
1.5 に縮小してデータ収集を継続し、効果を再評価する。

---

## 2. セルフレビュー

### 2.1 修正の正当性

| 観点 | 判定 | 理由 |
|------|------|------|
| direction 判定 | ✅ 不変 | 506# から adjusted_spread で判定しており、507# は触れない |
| confidence 基準 | ✅ 正当 | de-meaning = basis 除去後の偏差が情報量。偏差比例の boost が合理的 |
| velocity 基準 | ✅ 正当 | 方向一致は「基準からの逸脱方向」と velocity が同方向かで判定すべき |
| 後方互換性 | ✅ 保持 | `basis_bps=0.0` のとき `_conf_spread = gating_spread` (従来通り) |
| gating threshold | ✅ 不変 | `abs(gating_spread) < threshold` は de-meaning 前で判断。情報量ゲートとして正しい |

### 2.2 設計上の検討事項 — 意図的に修正しなかった箇所

#### Gating Threshold (L267)

```python
if abs(gating_spread) < spread_bps_threshold:  # 1.0 bps
    return None
```

これは **de-meaning 前の `gating_spread`** で判定している。

basis ≈ -3.3bps のとき `|gating_spread| ≈ 3.3bps > 1.0` なので、
事実上常に hint が生成される。これは**意図的に正しい**:
- このゲートは「EMA に十分な信号量があるか」のチェック
- 信号量は原始信号のspreads幅であり、basis 補正後の偏差ではない
- 偏差が小さい場合の安全弁は `confidence` と `min_confidence` ゲート (L355) が担う

もし `adjusted_spread` でゲートすると、spread が basis 近傍のとき
信号丸ごと棄却されてしまい、sell 側の guard が無効化される。

#### Microprice Confidence Modifier (L337)

```python
direction_sign = 1.0 if direction == "up" else -1.0
if microprice_spread_bps * direction_sign > 0.0:
    mp_factor = 1.0
```

Microprice は生の板深度情報 (CC/BF) から算出された重み付き mid の差であり、
basis 除去は行われていない。この raw microprice を de-meaning 後の
`direction` と cross-check するのは、**独立データソースによる方向確認**として合理的。

Microprice 自体に basis 補正を加える設計もあり得るが、
microprice は板深度不均衡の反映であり基準水準とは性質が異なるため、
現時点では raw で使うのが妥当。

#### Veto 判定 (maker_risk_guards.py L247)

```python
if abs(hint.spread_bps) >= cfg.cross_venue_lead_lag_veto_threshold_bps:
    # 8.0bps → hard veto
```

`hint.spread_bps` は raw EMA spread (de-meaning 前)。
Veto は**絶対的な乖離量** (CC-BF 間の異常な価格差) で判断すべきであり、
basis 補正後の偏差で判定すると veto が効かなくなる。→ 正しく raw で判定。

### 2.3 リスク評価

| リスク | 評価 | 対策 |
|--------|------|------|
| De-meaning 有効時に sell guard が弱すぎる | ✅ 低い | confidence_floor=0.33 が下限を保証。小偏差でも最低 33% の boost |
| Recovery_skew 1.5 で deadlock 再発 | ⚠️ 中 | hot-reload 対応済。ログで recovery_skew 発動を監視し、問題あれば即時 2.0 に戻す |
| Velocity 判定逆転で buy 側に影響 | ✅ なし | `basis_bps=0.0` (default) では従来ロジックと完全同一 |

---

## 3. 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `scripts/v460/lib/cross_venue_lead_lag.py` | `_conf_spread`, `_vel_spread` 導入 (L280-283) |
| `configs/v460/fill_test.yaml` | `recovery_skew_offset_mult: 2.0 → 1.5` |
| `tests/unit/v460/test_506_sell_improvements.py` | `TestDeMeaningConfidenceCorrection` 3件追加 |
| `docs/v460/506_verification_500_501_reviews.md` | §8 に 507# 内容を追記 |

---

## 4. テスト

`tests/unit/v460/test_506_sell_improvements.py` に 3 件追加 (合計 13 件):

| テスト | 検証内容 |
|--------|---------|
| `test_confidence_proportional_to_deviation_with_basis` | 小偏差 < 大偏差の confidence 比例性 |
| `test_confidence_without_basis_uses_raw_spread` | basis=0.0 で後方互換 (raw 使用) |
| `test_velocity_agreement_with_adjusted_spread` | velocity と adjusted_spread の同方向判定 |

全 49 テスト (cross-venue 36 + sell_improvements 13) パス。
v460 全体 3786 テスト中 0 failed。

---

*実装日: 2026-03-20*
*コミット: `d9c31ff6d`*
