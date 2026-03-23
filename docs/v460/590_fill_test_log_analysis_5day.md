# 590# Fill Test ログ分析 (5日間: 2026-03-19〜24)

## 概要
`results/v460/fill_test/fill_records_YYYYMMDD.jsonl` 5日分を SHA/起動単位/多次元で分析し、改善機会を特定。

- **総サイクル**: 2,504 / **約定数**: 799 / **約定率**: 31.9%
- **PnL30s 総和**: -132.2 bps（5日間）
- **21 SHA** が確認、うち主要 SHA は 7 つ（n≥20）

---

## 1. 日次 PnL トレンド

| 日付 | 約定数 | 約定率 | PnL30s mean | PnL30s sum |
|------|--------|--------|-------------|------------|
| 03-19 | 66 | 38.8% | -0.49 | -32.1 |
| 03-20 | 273 | 48.0% | -0.22 | -60.5 |
| 03-21 | 161 | 26.7% | -0.55 | -88.7 |
| 03-22 | 125 | 18.6% | -0.44 | -55.0 |
| 03-23 | 140 | 34.1% | +0.27 | +38.0 |
| 03-24 | 34 | 43.6% | +1.95 | +66.1 |

**所見**: 3/21-22 が最悪。3/23以降は新SHA（c164d21d, 447b2ec5）投入後に改善。

---

## 2. SHA 別パフォーマンス（n≥10）

| SHA | n | PnL mean | PnL sum | AS率 | offset mean | B/S |
|-----|---|----------|---------|------|-------------|-----|
| 447b2ec5 | 41 | **+1.49** | +61.3 | 39% | 0.285 | 21/20 |
| c164d21d | 73 | **+0.86** | +62.9 | 48% | 0.408 | 41/32 |
| dfbe3b53 | 169 | +0.22 | +36.4 | 49% | 0.210 | 86/83 |
| cc6c9466 | 13 | +0.41 | +5.3 | 38% | 0.193 | 7/6 |
| a81157bd | 12 | +0.46 | +5.5 | 33% | 0.247 | 8/4 |
| e6f2ef62 | 12 | -0.02 | -0.2 | 25% | 0.250 | 8/4 |
| 99ca9511 | 31 | -0.03 | -0.8 | 55% | 0.277 | 18/13 |
| 8e37cf96 | 20 | -0.41 | -8.1 | 60% | 0.392 | 10/10 |
| 5a546923 | 89 | -0.27 | -24.3 | 56% | 0.214 | 45/44 |
| d93b9a5b | 106 | -0.45 | -47.9 | 53% | 0.246 | 67/39 |
| 20d4f778 | 115 | -0.67 | -77.3 | 57% | 0.226 | 72/43 |
| 548dda24 | 58 | **-1.00** | -58.1 | 45% | 0.191 | 29/29 |
| 77cd082e | 10 | **-2.44** | -24.4 | 60% | 0.210 | 5/5 |

**所見**:
- **勝ちSHA**: 447b2ec5（+1.49 mean, AS39%）と c164d21d（+0.86, offset高め 0.408）
- **負けSHA**: 548dda24（-1.00, macro_strong_down で -77.6 集中）、20d4f778（-0.67, n=115 買い偏重）
- c164d21d は offset が高い（0.408）のに profit → additive pipeline 有効の可能性
- 447b2ec5 = `554# Raw data gap fill + CalibrationMap offline batch`
- 548dda24 = `session037: align ml real-data test contract`（古い SHA）

---

## 3. マクロトレンド × PnL

| macro_trend | n | PnL mean | PnL sum | AS率 | B/S |
|-------------|---|----------|---------|------|-----|
| macro_weak_up | 152 | **+0.73** | +110.8 | 44% | 77/75 |
| macro_weak_down | 180 | -0.15 | -27.4 | 51% | 97/83 |
| macro_neutral | 278 | -0.24 | -67.5 | 51% | 166/112 |
| macro_insufficient | 82 | -0.63 | -51.5 | 55% | 48/34 |
| macro_strong_up | 59 | -0.36 | -21.3 | 59% | 33/26 |
| **macro_strong_down** | **45** | **-1.79** | **-80.4** | 49% | 24/21 |

**所見**:
- `macro_strong_down` が最悪（-1.79 mean）— 特に sell 側が損失集中
- `macro_weak_up` のみプラス — MM にとって緩やかな上昇が最適環境
- `macro_strong_up` も意外に負け（-0.36）— 追随できていない可能性

---

## 4. レジーム × PnL

| regime | n | PnL mean | AS率 |
|--------|---|----------|------|
| ranging | 711 | -0.35 | 52% |
| trending_up | 49 | +1.36 | 41% |
| trending_down | 39 | +1.55 | 41% |

**所見**: 89% が ranging で約定、PnL は負。trending のみ profit だが件数少。

---

## 5. オフセット × PnL

| 区分 | n | PnL mean |
|------|---|----------|
| Low offset (≤0.25) | 643 | -0.36 |
| High offset (>0.25) | 153 | **+0.60** |
| Buy × Low | 365 | -0.30 |
| Buy × High | 80 | +0.15 |
| Sell × Low | 278 | -0.44 |
| Sell × High | 73 | **+1.10** |

**所見**: offset が高い方が PnL 良好。現在の median offset 0.25 は低すぎる可能性。sell × high offset の組合せが最も profitable。

---

## 6. スプレッド × PnL

| 区分 | n | PnL mean | AS率 |
|------|---|----------|------|
| Narrow (≤1.95 bps) | 400 | -0.00 | 49% |
| Wide (>1.95 bps) | 399 | -0.33 | 52% |

**所見**: Narrow spread のほうが PnL良好。Wide spread時は AS率も高く、ボラが高い不利な環境での約定が多い。

---

## 7. Volatility Guard

| 状態 | n | PnL mean |
|------|---|----------|
| VG ON | 782 | -0.16 |
| VG OFF | 14 | -1.04 |

**所見**: VG OFF 時は PnL が大幅に悪化（-1.04 vs -0.16）。ただし n=14 と少量。

---

## 8. Sidecar Signal

| status | n | PnL mean |
|--------|---|----------|
| stale | ~481 | -0.30 |
| fresh | ~210 | -0.28 |
| **error** | **~107** | **+0.58** |

**所見**: sidecar error 時に PnL が逆に良い（+0.58）。error でサイドカー補正が skip されること自体が利益になっている — サイドカー補正の方向性に問題がある可能性を示唆。

---

## 9. Cross-Venue Lead-Lag

| 状態 | n | PnL mean |
|------|---|----------|
| CV Applied | 254 | -0.22 |
| CV Not Applied | 542 | -0.15 |

**所見**: CV lead-lag 適用時のほうがわずかに悪い。CV 補正が逆効果の可能性あり。

---

## 10. テール損失分析

- **P5 しきい値**: -9.55 bps
- **最悪 40 fills**: 80% sell 側、95% ranging、VG 98% triggered
- **最悪単体**: PnL30s = -72.7 bps（sell, UTC13, ranging, macro_strong_down, SHA=548dda24）
- **UTC 13-14 (JST 22-23)**: 77 fills で PnL sum = -124.88

---

## 11. テレメトリ欠損

### 11.1 Kissell & Glantz 指標（CRITICAL）
- `adverse_selection_cost_bps`: **0/799 (0%)** — 全レコードで欠損
- `spread_capture_bps`: **0/799 (0%)** — 全レコードで欠損
- `post_fill_90s_mid` 相当フィールド: 未記録

**根本原因**: これらのフィールドは `fill_record_builder.py` の `_build_fill_measurement_fields()` に含まれるが、**稼働中の全 SHA がこのコードを含まないバージョン**。305# で追加されたが、log に残っている SHA は全て 305# 以前のコミット。

**対策**: 最新 SHA をデプロイすれば自然解決。コード修正は不要。

### 11.2 execution_additive_enabled
- **2 SHA のみ記録**（c164d21d, 8e37cf96）— 573# テレメトリ追加後
- 残り 13+ SHA は pre-573# → フィールド未出力

### 11.3 entry_gate_verdict
- **全 2,504 レコードで missing** — 589# で `entry_gate_enabled: false` を YAML 追加済みだが、稼働 SHA には未反映

---

## 12. 改善提案（優先度順）

### P1: macro_strong_down 防御強化 (HIGH)
- **問題**: macro_strong_down で PnL = -1.79 mean、特に sell 側 × 548dda24 で -77.6 集中
- **提案**: strong_down 時の sell 側 offset を引き上げ（+50%）、またはマクロ連動のキル条件追加
- **期待効果**: テール損失 -80.4 → 約半減

### P2: offset 底上げ検討 (HIGH)
- **問題**: 中央値 0.25 だが low offset (≤0.25) → PnL -0.36、high offset → +0.60
- **提案**: base_offset を 0.25 → 0.30 に引き上げ、または eDRC の感度調整
- **期待効果**: AS 率低下、PnL 改善

### P3: サイドカー補正方向の検証 (MEDIUM)
- **問題**: error（=補正なし）時に PnL +0.58、fresh（=補正あり）時に -0.28
- **提案**: サイドカー bias の正負反転 or 係数縮小の検証
- **期待効果**: fresh 210 fills × 0.86 bps 改善 = +180 bps

### P4: CV lead-lag 効果再検証 (MEDIUM)
- **問題**: CV 適用時 PnL -0.22 vs 非適用 -0.15
- **提案**: CV lead-lag の shrinkage factor 調整、または spread condition の閾値再設定
- **期待効果**: 254 fills × 0.07 bps = 小幅改善

### P5: 深夜帯（UTC 13-14 / JST 22-23）の制御 (MEDIUM)
- **問題**: 77 fills で PnL sum = -124.88
- **提案**: `hour_ceiling_mult` の深夜帯設定追加、または offset プレミアム
- **注意**: hour_ceiling_mult フィールドが全レコードで未記録 — 機能が有効か要確認

### P6: 最新 SHA デプロイ (LOW)
- **問題**: Kissell & Glantz 指標、entry_gate verdict、additive pipeline テレメトリが記録されない
- **対策**: 最新コードでの fill test 再起動で自然解決

---

## 13. 次ステップ
1. macro_strong_down × sell の offset 引き上げロジック設計
2. base_offset 底上げの backtest
3. sidecar bias の効果検証（A/B テストまたはログ回帰分析）
4. hour_ceiling_mult 設定の確認と深夜帯パラメータ追加
5. 最新 SHA デプロイ後の AS/SpCap メトリクス確認
