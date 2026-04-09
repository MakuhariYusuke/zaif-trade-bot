# 723# skip_gate bypass パラドックス — 深堀り分析

> **結論**: skip_gate LightGBM モデルは予測力がゼロないし負。bypass_mode および
> rate_limiter による skip 抑制が +66.0bps の純利益を生み出しており、現行設定
> (bypass_mode=true, max_skip_rate=0.30) は合理的。

---

## 1. 背景

686# で skip_gate にバイパスモードを導入した理由は「スコア四分位間の AS 率に差が無い」
= モデルの予測力がゼロであったため。以降 bypass_mode=true が全サイド (buy/sell) で
常時有効となっている。

720# の int-key バグ修正で hour_offsets (9 entries) が復活し、skip_gate スコアの
算出根拠が変化した可能性がある。本分析は 720# デプロイ後 (04/06-09) の 4 日間データで
bypass の実績を精査する。

### メカニズム

skip_gate_evaluator → ztb/ml/skip_gate.py の 2 層構造:

```
[Model predict] → should_skip=True?
        │
        ├─ Yes ─→ [Rate Limiter] skip_rate > max_skip_rate (0.30)?
        │              │
        │              ├─ Yes → forced_pass=True, reason="skip_rate_limit(X%>30%)"
        │              │        should_skip=False → ★ rate_limiter catch
        │              │
        │              └─ No  → should_skip=True のまま
        │                       → [Evaluator] bypass_mode=True?
        │                              └─ Yes → bypassed=True ★ bypass_mode catch
        │
        └─ No ──→ reason="pass" ★ genuine_pass
```

**rolling window** (直近 20 決定/side): rate_limiter 発動後に should_skip=False を記録
→ window が回復 → bypass_mode 発動フェーズ → True 蓄積 → 再び rate_limiter……
という周期的サイクルが形成される。

---

## 2. 4 日間集計 (04/06–09, 575 fills)

### 2.1 二分類: bypass vs normal

| 分類 | n | avg30 (bps) | total30 (bps) | WR | AS |
|------|---|-------------|---------------|----|----|
| bypassed (model=skip, bypass_mode 捕捉) | 183 | **+0.36** | **+66.0** | 48% | 17% |
| normal (rate_limiter + genuine_pass) | 392 | -0.53 | -206.9 | 45% | 24% |

### 2.2 三分類: bypass / rate_limited / genuine_pass

| 分類 | n | avg30 | total30 | WR | AS |
|------|---|-------|---------|----|----|
| bypassed (model=skip, bypass_mode) | 183 | **+0.36** | **+66.0** | 48% | 17% |
| rate_limited (model=skip, rate>30%) | 330 | -0.52 | -172.5 | 45% | 25% |
| genuine_pass (model=pass) | 62 | -0.55 | -34.4 | 47% | 18% |

**注目点**: genuine_pass (model=pass, score 平均 +0.654) が avg=-0.55bps で
rate_limited (model=skip) の avg=-0.52bps と同等。
→ **モデルの pass/skip 判定に有意な予測力は認められない**。

### 2.3 モデル判定別 (skip=全override vs pass)

| モデル判定 | n | avg30 | total30 | WR | AS |
|-----------|---|-------|---------|----|----|
| model=SKIP (全て override) | 513 | -0.21 | -106.5 | 46% | 22% |
| model=PASS | 62 | -0.55 | -34.4 | 47% | 18% |

モデルが「通すべき」と判断した fill が「止めるべき」fill より悪い。
→ モデルの信頼度順序が反転している可能性。

---

## 3. 次元別分析

### 3.1 Side 別

| Side | 分類 | n | avg30 | total30 | WR | AS |
|------|------|---|-------|---------|----|----|
| sell | bypassed | 87 | **+1.01** | **+87.6** | 53% | 17% |
| sell | normal | 195 | -0.65 | -127.5 | 47% | 30% |
| buy | bypassed | 96 | -0.23 | -21.6 | 44% | 17% |
| buy | normal | 197 | -0.40 | -79.4 | 44% | 19% |

**sell bypass が全体の利益の源泉** (+87.6bps)。sell AS=17% vs normal 30% は
bypass fills が AS を回避する傾向を示す。

### 3.2 Regime 別

| Regime | 分類 | n | avg30 | total30 | WR | AS |
|--------|------|---|-------|---------|----|----|
| trending_up | bypassed | 23 | **+2.01** | **+46.3** | 61% | 4% |
| trending_up | normal | 51 | -2.42 | -123.4 | 43% | 22% |
| ranging | bypassed | 123 | +0.34 | +41.4 | 47% | 18% |
| ranging | normal | 270 | -0.32 | -86.8 | 44% | 26% |
| trending_down | bypassed | 37 | -0.59 | -21.7 | 43% | 22% |
| trending_down | normal | 71 | +0.05 | +3.3 | 51% | 20% |

**trending_up bypass が突出**: +2.01bps (WR 61%, AS 4%)。
trending_up normal は -2.42bps の大赤字。差は **4.43bps**。

trending_down は逆転: bypass=-0.59 vs normal=+0.05。
→ bypass の価値は regime 依存性が強い。

### 3.3 skip_gate スコア分布

| 分類 | n | mean | p25 | p50 | p75 |
|------|---|------|-----|-----|-----|
| bypassed | 183 | -0.863 | -1.475 | -0.740 | -0.299 |
| normal | 392 | -0.637 | -1.326 | -0.595 | +0.104 |
| genuine_pass | 62 | +0.654 | +0.325 | +0.705 | +0.852 |

bypassed はスコアが最も低い (モデルが最も強く skip 推奨)。
にも関わらず収益が最も高い → **スコアと収益の逆相関**。

### 3.4 日次一貫性

| 日付 | bypass n | bypass avg | normal n | normal avg | bypass 優位 |
|------|----------|-----------|----------|------------|-----------|
| 04/06 | 42 | +0.49 | 106 | +0.11 | ✅ |
| 04/07 | 49 | +0.46 | 118 | -0.62 | ✅ |
| 04/08 | 64 | +0.11 | 120 | -0.86 | ✅ |
| 04/09 | 28 | +0.57 | 48 | -0.88 | ✅ |

**4 日連続で bypass > normal**。単発の偶然ではない。

### 3.5 skip_rate at bypass time

bypass fills 183件の `skip_gate_side_skip_rate`:
- mean=0.274, min=0.000, max=0.300, p50=0.300

ほぼ全て rate=0.300 (閾値ギリギリ) で発動。
bypass_mode は rate_limiter の直前で介入する構造。

---

## 4. 反事実分析

| シナリオ | total30 (bps) |
|---------|---------------|
| 現状 (bypass_mode=true) | -140.9 |
| bypass fills をスキップした場合 | **-206.9** |
| 差分 (bypass の貢献) | **+66.0** (47% 改善) |

---

## 5. 構造的解釈

### 5.1 なぜ bypass fills は profitable か

仮説群（相互排他ではない）:

1. **モデル逆相関**: LightGBM のスコアが市場の実態と逆相関。
   低スコア (-0.863 avg) fill が高収益 → モデルの学習データ or 特徴量が陳腐化。
2. **Rolling window cycling effect**: bypass は rate_limiter リセット直後の
   「回復フェーズ」に集中。この期間の市場状態が有利な可能性
   (selection bias: 直前のスキップ集中期から市場が回復している)。
3. **trending_up regime 特効**: bypass fills の trending_up は +2.01bps
   (n=23, AS=4%)。モデルは trending_up を「危険」と判断するが、
   sell_hour_boost + regime_boost が重なる局面では maker にとって有利。

### 5.2 trending_up bypass の解剖

trending_up bypass (n=23, avg=+2.01, AS=4%) が全体 bypass 利益の 70%:
- 上昇トレンド中に model が「skip」推奨
- bypass_mode で fill が通る
- 上昇局面での maker sell は本来有利（高い板を取られにくい）
- AS が 4% と極端に低い → model が恐れている AS が実際には発生しない

→ モデルの trending_up 忌避は過学習 or 古い特徴量分布に基づく誤判断の可能性。

### 5.3 trending_down bypass の警告

trending_down bypass (n=37, avg=-0.59) は唯一 normal (+0.05) を下回る。
→ trending_down では model の skip 推奨が**正しい**可能性がある。
→ regime 条件付き bypass 解除は検討余地あり（ただし n=37 で結論は時期尚早）。

---

## 6. max_skip_rate への示唆

714# P1 で「max_skip_rate 0.30→0.35/0.40 検討」が保留されている。
本分析の結論:

| 選択肢 | 効果予測 |
|--------|---------|
| 0.30 維持 (現行) | bypass_mode が ~32% の fill を捕捉。+66.0bps/4d 貢献。✅ 推奨 |
| 0.35 に上げる | rate_limiter 発動減 → bypass_mode 捕捉増 → 潜在的に利益増 |
| 0.40 に上げる | 672# CI=[−0.53,+0.07] で否定済み。fill rate 低下リスク |
| 0.25 に下げる | bypass_mode 捕捉減 → +66.0bps の一部を失う |

**現時点の推奨: 0.30 維持**。
- モデルに予測力がない以上、skip_rate を上げる（=モデルを信じる）のは逆効果
- bypass_mode が十分に機能しているので rate_limiter のさらなる緩和は不要
- 根本対策は **モデル再学習 or bypass_mode の恒久化**

---

## 7. 再現コマンド

```bash
.venv/Scripts/python.exe temp/analyze_bypass.py
```

データソース: `results/v460/fill_test/fill_records_2026040{6,7,8,9}.jsonl`

---

## 8. 今後のアクション

| 優先度 | アクション | 状態 |
|--------|---------|------|
| — | bypass_mode=true 維持 | ✅ 現行で継続 |
| — | max_skip_rate=0.30 維持 | ✅ 714# P1 判断材料として本文書を参照 |
| P2 | trending_down 条件での bypass 解除検討 | ⏳ n=37 で統計的検出力不足。要蓄積 |
| P3 | skip_gate モデル再学習 (720# fix 後データ利用) | ⏳ 十分なデータ蓄積後 |
| P3 | bypass_mode の恒久化 (モデル撤去) 検討 | ⏳ 6 を含む複合判断 |

---

*作成: 2026-04-09, 対象期間: 2026-04-06〜09 (575 fills)*
*関連: 686# (bypass導入), 714# (hexagram P1), 720# (int-key fix)*
