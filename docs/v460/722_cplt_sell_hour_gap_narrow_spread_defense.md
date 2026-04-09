# 722# 720#初動分析: sell_hour_boost gap + narrow spread sell 対策

> 720# int-key fix デプロイ後 8h (UTC 0h–7h) の実績分析から浮上した 2つの改善。
> 714# 六五「公弋取彼在穴」— 回収可能な YAML hot-reload で穴を射抜く。

---

## 背景: 720# デプロイ後の初動 (04/09 UTC 0h–7h)

### 全体実績

| 指標 | 04/09 (n=72) | 04/08 (n=184) | 変化 |
|------|-------------|--------------|------|
| avg pnl30 | −0.40 bps | −0.52 bps | +0.12 (改善) |
| WR | 50% | 46% | +4pt |
| AS率 | 22% | 21% | ≈同等 |
| sell avg | −0.55 bps | −0.95 bps | +0.40 (改善) |

sell_hour_offset_boost が初めて本番で機能 (21/35 sell fills で発火)。
UTC 4h (JST 13h) × 5.0 ブーストで avg **+1.64 bps**（688# 指定の最危険帯）。

### 発見された 2つの穴

---

## 穴 1: UTC 5h / 6h に sell_hour_offset_boost 未設定

### データ (4日間集計: 04/06–04/09)

| UTC (JST) | n | avg pnl30 | total | AS率 | boost |
|-----------|---|-----------|-------|------|-------|
| UTC 5 (14h) | 23 | −1.28 bps | −29.5 bps | 26% | **なし** |
| UTC 6 (15h) | 20 | −1.08 bps | −21.6 bps | 25% | **なし** |

比較: UTC 0h (x1.50) = avg −2.63 → UTC 5/6 はそれより穏やかだが持続的に負。

### 判断根拠

- n ≥ 20 で 4日間一貫して負 → 統計的に有意な傾向
- AS率 25-26% は全体平均(22%)を上回る
- 720# により他の全時間帯にブーストが効いた結果、これら2時間が相対的な穴に

### 変更

```yaml
sell_hour_offset_boost:
  5: 1.3   # 722# JST14h: 4日 sell n=23 avg=-1.28bps AS=26%
  6: 1.3   # 722# JST15h: 4日 sell n=20 avg=-1.08bps AS=25%
```

×1.3 は最小限の防御。UTC 0h (×1.5) や UTC 8h (×1.5) より控えめ。
→ 合計 17 → 19 entries。

---

## 穴 2: Narrow spread (<2bps) sell の構造的損失

### データ (4日間集計)

| Spread bucket | sell total pnl30 | n |
|---------------|-------------------|---|
| < 2bps (narrow) | **−81.8 bps** | 110 |
| 2–4bps (mid) | **+35.9 bps** | 160 |

narrow sell は 4日間で −81.8 bps の損失源。mid sell は +35.9 bps で黒字。

### 時間帯別内訳 (narrow sell 損失 top 5)

| UTC | n | avg | 特徴 |
|-----|---|-----|------|
| UTC 17 (02h JST) | 3 | −7.33 | 極小サンプルだが深刻 |
| UTC 13 (22h JST) | 4 | −4.46 | sell_hour_boost ×2.0 適用済みでも負 |
| UTC 8 (17h JST) | 7 | −3.19 | sell_hour_boost ×1.5 適用済みでも負 |
| UTC 21 (06h JST) | 4 | −3.07 | sell_hour_boost ×1.5 適用済みでも負 |
| UTC 5 (14h JST) | 14 | −1.95 | 穴1 で boost 追加予定 |

### 現状の narrow 防御

```yaml
spread_adaptive:
  narrow_spread_boost_sell: 1.5  # 674#: 2.5→1.5 に引下げ
```

674# で「狭 spread = 低情報非対称」として 2.5→1.5 に引下げたが、4日間の実データは
**狭 spread sell は一貫して損失** を示す。論理的には正しくても、実証が否定している。

### 原因の考察

狭 spread 環境では:
1. **adverse selection 確率が上がる**: 市場が活発で価格変動が速い局面で spread が縮小
2. **offset base が小さくなる**: pipeline 入力の regime/spread_adapt が低く、×1.5 でも不足
3. **vol_guard による追加防御も弱い**: 狭 spread 時は vpin が低めで boost 余地が少ない

### 変更

```yaml
spread_adaptive:
  narrow_spread_boost_sell: 2.0  # 722# 1.5→2.0: 4日 narrow sell=-81.8bps
```

1.5→2.0 は 674# 以前の 2.5 よりまだ控えめ。pipeline の他段階との乗算で
offset は概ね 0.25-0.35 → 0.33-0.47 に上昇（final ceil 0.8 に抵触しない）。

---

## 714# 卦辞との照合

| 卦辞 | 本施策との対応 |
|------|-------------|
| **可小事，不可大事** | YAML 2箇所の微調整。アーキテクチャ変更なし |
| **不宜上，宜下** | offset を上げる（保守的方向）のみ。攻撃的な変更なし |
| **公弋取彼在穴** | hot-reload で回収可能。効果がなければ即時 revert |
| **飛鳥遺之音** | narrow spread での sell AS パターンという「飛鳥の音」を聴いた |

---

## リスク評価

### sell_hour_offset_boost UTC 5/6 追加

- **リスク**: sell fill 率低下（offset 上昇による約定距離拡大）
- **影響範囲**: UTC 5/6 の sell のみ (全 fill の約 8%)
- **緩和**: ×1.3 は最小ブースト。効果不十分なら ×1.5 に引上げ
- **回収**: YAML hot-reload で即時 revert 可能

### narrow_spread_boost_sell 1.5→2.0

- **リスク**: 狭 spread 時の sell fill 率低下
- **影響範囲**: spread < 2.5bps かつ sell (全 fill の約 20%)
- **緩和**: 2.0 は 674# 以前の 2.5 より控えめ。段階的引上げ
- **回収**: YAML hot-reload で即時 revert 可能

### 期待効果

4日データ外挿:
- UTC 5/6: ×1.3 で AS fill の一部を回避 → −51.1 bps/4日 の 15-20% 改善見込み
- narrow sell: ×2.0 で base 防御強化 → −81.8 bps/4日 の 20-30% 改善見込み

---

## 変更一覧

| ファイル | 変更 | 種類 |
|---------|------|------|
| configs/v460/fill_test.yaml | sell_hour_offset_boost に UTC 5/6 追加 | config |
| configs/v460/fill_test.yaml | narrow_spread_boost_sell 1.5→2.0 | config |
