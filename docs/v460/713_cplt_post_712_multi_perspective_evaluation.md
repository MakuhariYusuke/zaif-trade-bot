# 713# Post-712# Multi-Perspective Evaluation

- **前提**: 712# Fix F1-F3 (ceiling引上げ, entry_gate無効化, side_offset引上げ) を 4/8 01:47:41 JST にhot-reload
- **対象期間**: 4/8 01:48 ～ 4/9 04:20 JST (約26時間)
- **Phase分類**: A = pre-reload (750cd71), B = post-reload same SHA, C = auto-restart (90de90f/0a817b2)

---

## 1. Phase比較サマリー

| Phase | fills | sc_avg | win_rate | clamp | AS |
|-------|-------|--------|----------|-------|----|
| A (pre-reload) | 100 | **-0.684** | 27% | 35 | 27 |
| B (post-reload, 深夜) | 50 | **-1.099** | 26% | 16 | 11 |
| C (auto-restart) | 134 | **-0.397** | 35% | 30 | 31 |

**Phase C は Phase A 比で sc_avg +42% 改善** (−0.684 → −0.397)。win_rate も 27% → 35%。

Phase B の悪化 (−1.099) は深夜帯 (01:48～12:37) + MCB halt 3回 + trending_up 集中に起因。市場条件の寄与が大きく、fix自体の悪影響ではない。

## 2. 日次推移

| 日付 | fills | sc_avg | win% | sc_total |
|------|-------|--------|------|----------|
| 04/04 | 131 | −0.315 | 41% | −41.3 |
| 04/05 | 177 | −0.252 | 39% | −44.6 |
| 04/06 | 143 | −0.460 | 40% | −65.8 |
| 04/07 | 166 | −0.618 | 30% | −102.6 |
| 04/08 | 175 | −0.659 | 33% | −115.4 |
| 04/09 (途中) | 27 | −0.508 | 22% | −13.7 |

注意: 04/08全体はPhase B深夜帯を含む。Phase C単独(134fills) は sc_avg=−0.397 で 04/04-06 ベースラインに接近。

---

## 3. Skip Gate 分析 — 最重要発見

### 3.1 Skip Gate カテゴリ別パフォーマンス

| カテゴリ | fills | sc_avg | 備考 |
|---------|-------|--------|------|
| organic_pass | 8 | **+0.028** | モデルが「通せ」→ブレークイーブン |
| forced_pass (rate_limit) | 82 | **−0.462** | モデルは「skip」→rate limiter強制通過 |
| bypassed | 44 | **−0.354** | bypass_mode=true |

**Phase C fills の 61% (82/134) がrate limiter による強制通過**。max_skip_rate=0.30 のため、skip率が30%を超えるとモデル判断を無視して強制的にfillに向かう。

モデルの organic pass はブレークイーブン (+0.028) であり、**モデル自体は機能している**。問題は rate limiter が悪質な機会を強制的に通していること。

### 3.2 典型的パターン

Worst fill #1: 22:42 buy sc=−5.57 (trending_up, CLAMP, skip_rate_limit(31%>30%) FORCED)
→ モデルは「skip」と言ったが rate limiter が 31%>30% で強制通過 → ceiling でさらにCLAMP → 大損

### 3.3 過去ドキュメント参照

- **710#**: 閾値 0.4-0.6 推奨、sell model は inverted (低スコア = 高リターン)
- **708#**: 二峰性係数 0.5595 確認、forced_pass 主因は 35%>30%
- **709#**: CX1 skip_gate quality analysis, threshold 推奨 0.4-0.6
- **599#**: buy skip_gate model は 2月24日から未更新 (stale)

---

## 4. Clamp 分析 — 逆説的発見

### 4.1 Phase C Clamp 状況

| 側 | clamped | unclamped | clamp sc_avg | unclamp sc_avg |
|----|---------|-----------|------------|--------------|
| buy | 7/68 | 61/68 | **−0.791** | −0.366 |
| sell | 23/66 | 43/66 | **−0.113** | −0.530 |

**sell 側で逆説**: clamped fills (−0.113) が unclamped (−0.530) よりも大幅に良好。

売り clamp: pre_clamp_offset avg=1.122 → post_clamp=0.650。パイプラインが 1.12 を出力→ ceiling 0.65 で切り詰め。この強制的なワイドオフセットがむしろ保護的に機能。

買い clamp: Phase A (pre=0.547→post=0.350) から Phase C (pre=0.824→post=0.500) に改善。ceiling 引上げの効果は確認できるが、clamp直後の sc は依然悪い。

### 4.2 含意

売り側: offset パイプラインの有機的出力 (avg=0.466 unclamped) は低すぎる可能性。ceiling 0.65 がむしろセーフティネットとして機能している。

---

## 5. Regime 分析

### 5.1 Phase A → C 改善

| side/regime | A sc_avg | C sc_avg | 改善 |
|------------|----------|----------|------|
| buy/ranging | −0.60 | −0.36 | +40% |
| sell/ranging | −0.47 | −0.35 | +26% |
| buy/trending_down | **−1.44** | **−0.39** | +73% |
| sell/trending_down | −0.94 | −0.60 | +36% |
| sell/trending_up | −0.93 | −0.39 | +58% |
| buy/trending_up | −0.76 | −0.62 | +18% |

**hard_skip_mult overrides (buy/trending_down: 4.0, sell/trending_up: 5.0) は劇的に有効**。buy/trending_down は −1.44 → −0.39 と 73% 改善。

---

## 6. Velocity & OBI 相関

| side | outcome | velocity_avg | OBI_avg |
|------|---------|-------------|---------|
| buy | win (28) | +0.28 | +0.308 |
| buy | loss (40) | +0.71 | +0.008 |
| sell | win (19) | +0.10 | +0.068 |
| sell | loss (47) | −1.04 | +0.097 |

- **Buy**: 勝ちtrade は低velocity (+0.28)。高velocity (+0.71) は危険 → velocity defense 正当化
- **Sell**: 負けtrade は負のvelocity (−1.04 = 価格下落中に売り出し) → AS の典型パターン

---

## 7. Offset Pipeline 分析

### Buy pipeline stages (avg):
```
base=0.10 → regime=0.12 → amihud=0.13 → vol_guard=0.19 → ffd=0.22 → final=0.22
```

### Sell pipeline stages (avg):
```
base=0.18 → as_shift=0.24 → regime=0.29 → spread_adapt=0.37 → vol_guard=0.46 → ffd=0.49 → final=0.49
```

注目: sell の sell_hour と loss_boost ステージは全く寄与していない (vol_guard → sell_hour → loss_boost が全て同値)。sell_hour_offset_boost の乗算対象が売り時間帯でも 0 の場合がある。

---

## 8. SAG / FFD / VG / Entry Gate

### 8.1 SAG (Spread AS Guard)
- 133/134 fills で triggered (実質的に常時ON)
- フラット 0.5 bps ペナルティ = spread 2bps 時に25%税
- **redesign (709#/710#)** は実装済み・無効化中。有効化で +9.4% 改善見込み

### 8.2 FFD Boost
- ON: 20 fills, sc=−0.238 / OFF: 114 fills, sc=−0.425
- FFD作動時は +18.7bps 改善。防御は機能している

### 8.3 VG (Velocity Guard)
- ON(110) sc=−0.399 vs OFF(24) sc=−0.388
- ほぼ差なし。VGの直接的な価値は不明確だが、vg_boost_factor < 1.1 がほとんどで影響が微小

### 8.4 Entry Gate
- 全134 fills の entry_gate_ev < 0 (avg=−1.728, max=−1.164)
- 有効化すれば全fillをブロック → CalibrationMap が系統的に悲観的、または stale
- **retrain が先決** (546# cold-start 問題: 最低50-100 fills 必要)

---

## 9. 改善候補 — 複数アプローチ比較

### I-1: Skip Gate Rate Limiter 緩和 [P0]

**現状**: max_skip_rate=0.30 → 61% fills が forced pass (sc=−0.462)

| 方法 | 内容 | 期待効果 | リスク | 過去参照 |
|------|------|---------|--------|---------|
| A: max_skip_rate 引上げ | 0.30→0.45 | forced_pass を ~40% に削減 | fill数減少 (収入機会↓) | 710# |
| B: threshold 引下げ | 0.8→0.5 | organic_pass 増加 (forced_pass 削減) | 低品質 fill 通過リスク | 708#, 709# |
| C: model retrain | 最新データで skip_gate 再学習 | 根本解決 | 時間コスト | 599# (stale model) |
| D: side別 max_skip_rate | buy=0.40, sell=0.50 | sell model inverted 問題に対応 | 複雑化 | 710# sell inversion |

**推奨**: A (0.30→0.40) を第一弾、効果測定後に B or C を検討。
理由: organic_pass sc=+0.028 はモデルの有効性を示唆。rate limiter 緩和が最も直接的。

### I-2: SAG Redesign 有効化 [P1]

**現状**: flat 0.5 bps tax (133/134 triggered)

| 方法 | 内容 | 期待効果 | リスク | 過去参照 |
|------|------|---------|--------|---------|
| A: redesign_enabled=true | 逆比例ペナルティ (2.0/spread) | +9.4% 全体 | narrow spread で penalty 増 | 709#, 710# |
| B: flat penalty 増加 | ev_penalty_bps 0.5→0.8 | 一律保守化 | ワイドspread時の抑制過剰 | 708# |
| C: SAG 無効化 | side_offset に吸収 | 簡素化 | 動的保護喪失 | — |

**推奨**: A (redesign 有効化)。実装済み・テスト済み。パラメータは 710# で調整済み (ref=2.0, cap=1.5)。

### I-3: Sell Side Offset / Ceiling 再調整 [P1]

**現状**: sell unclamped (offset avg=0.466) が sell clamped (ceiling=0.650) より悪い

| 方法 | 内容 | 期待効果 | リスク | 過去参照 |
|------|------|---------|--------|---------|
| A: sell side_offset 0.18→0.22 | パイプライン底上げ | unclamped sells 改善 | fill率低下 | 685# |
| B: sell ceiling 0.65→0.80 | より広いオフセット許容 | パイプラインの判断を尊重 | 過剰保守化 | 431#, 712# |
| C: sell_hour 17h boost 1.5→2.5 | 最悪時間帯強化 | 17h sc=−1.23 改善 | 他時間影響 | 310# |

**推奨**: 段階的に A (sell side_offset +0.04) を第一弾。SAG redesign と併用で相乗効果期待。

### I-4: 次フェーズ検討事項

- **Entry Gate CalibrationMap retrain**: 50-100 fills 蓄積後に実施 (546#)
- **Skip Gate model retrain**: 599# 以降未更新の buy model 更新
- **trending_up 対策強化**: buy/trending_up は依然 −0.62 で最悪カテゴリの一つ

---

## 10. 712# Fix の総合評価

### 有効性確認

| Fix | 対象 | 効果 |
|-----|------|------|
| F1: ceiling 引上げ | buy 0.35→0.50, sell 0.50→0.65 | buy clamp 35→7件, sell clamp rate 低下。sell clamped fills は paradoxically 良好 |
| F2: entry_gate 無効化 | entry_gate_enabled=false | Phase C entry_gate blocks = 0 (Phase A は 92 blocks)。CPU/log 浪費解消 |
| F3: side_offset 引上げ | buy 0.08→0.10, sell 0.14→0.18 | パイプライン底上げ効果あり、sc_avg 改善に寄与 |

### 残課題

1. **Skip gate forced pass** が最大のパフォーマンス・ドレイン (61% of fills, sc=−0.462)
2. **SAG flat tax** が 25% 相当の一律課金 (redesign 準備済み)
3. **Sell unclamped offset** が低すぎる (avg=0.466 vs ceiling=0.650)
4. **Entry Gate CalibrationMap** は retrain まで使用不可

---

## 付録: 分析スクリプト

```bash
.venv/Scripts/python.exe temp/analyze_713_deep.py
```
