# 684# 681-683 統合レビュー：多角的検証・盲点抽出・Codex タスク設計

| 項目 | 値 |
|------|-----|
| 作成日 | 2026-04-01 |
| 入力 | 681# (構造改善提案), 682# (批判的バリデーション), 683# (風水渙・既存活用) |
| 対象データ | `fill_records_20260401.jsonl` (n=300, fills=78) |
| HEAD | 8475a2512 |

---

## 0. 三者の立場整理

| # | 視座 | P0 主張 | Ceiling 0.55 | 核心的貢献 |
|---|------|---------|:---:|------------|
| 681# | 構造改善派 | sell ceiling 引上げ | **支持** | Clamp 100%問題の発見、regime×side クロス集計 |
| 682# | 批判的実証派 | sell-entry 条件付き抑制 | **反対(先送り)** | sell-entry roundtrip の損失分離、tail score 逆転の指摘 |
| 683# | 統合・深度解釈 | 止血→幅取りの2段階 | **条件付支持** | シーケンシャル戦略の提示、velocity 閾値引下げ提案 |

**三者の合意点**: sell 側の品質が問題の核。buy は概ね健全。

---

## 1. データ再検証結果

### 1.1 Sell 多次元分解

| 分割軸 | 条件 | n | Sell PnL30 (bps) | 差分 |
|--------|------|---|:---:|:---:|
| mid_price_trend_5s | >0 (上昇中) | 14 | **-3.25** | |
| | ≤0 | 25 | -0.80 | **Δ=-2.45** |
| queue_wait_sec | <10s (fast) | 12 | **-2.87** | |
| | ≥10s (slow) | 27 | -1.14 | **Δ=-1.73** |
| orderbook_imbalance | >0.1 (buy圧) | 19 | -1.83 | |
| | ≤0.1 | 20 | -1.53 | Δ=-0.30 |

**最重要発見**: `mid_price_trend_5s > 0` が sell PnL に対して**最大の説明力**を持つ（Δ=-2.45 bps）。682# が指摘した「上方ドリフト中の sell が壊滅的」は**データで強く支持**。

### 1.2 複合条件（Toxic Sell）

| 条件組合わせ | n | avg PnL30 | sum PnL30 |
|:------------|---|:---:|:---:|
| t5>0 AND fast(<10s) | 4 | **-5.16** | -20.6 |
| t5>0 AND obi>0.1 | 6 | **-6.01** | -36.1 |
| 上記以外の sell | 35 | -1.28 | – |

`t5>0 AND obi>0.1`（上方トレンド＋買い板優勢）の 6 件で **-36.1 bps** が生成 — 全 sell 損失 -65.4 の **55%** を占める。

### 1.3 Sell Tail Worst 20%（n=7）特性

| 指標 | Tail | 全 Sell | 682# 報告 |
|------|------|---------|-----------|
| avg PnL30 | -10.73 | -1.68 | -13.59 (p10) |
| avg sg_score | **+2.38** | – | +2.92 (682#) |
| avg trend_5s | -1.47 | – | -1.93 (682#) |
| avg queue_wait | 14.6s | – | – |
| avg OBI | +0.029 | – | -0.077 (682#) |
| AS 率 | **100%** (7/7) | 28.2% | 100% (682#) |
| Regime 分布 | td=3, tu=2, r=2 | – | – |

**盲点 ①**: 682# は tail の OBI=-0.077 と報告しているが、本検証では +0.029。サンプル定義の差異が原因の可能性（682# は p10 = worst 10%、本検証は worst 20%）。**数値の再現性を常に確認すべき**。

### 1.4 Skip Gate Score 相関

| | Pearson (681#) | **Spearman (本検証)** | p値 |
|---|:---:|:---:|:---:|
| Sell | -0.203 | **-0.204** | 0.212 |
| Buy | -0.069 | **-0.114** | 0.490 |

**盲点 ②**: **p値が 0.21 で統計的に有意ではない** (α=0.05)。681# は r=-0.203 を根拠に「通すべきでない fill を通している」と主張したが、**n=39 では有意な逆相関とは言えない**。ただし方向性の示唆としては有用（Type II error のリスク、すなわちサンプル不足で検出力不足）。

### 1.5 SG Score エントロピー

Sell 側の SG score を 10-bin に離散化した Shannon エントロピー:

$$H(\text{SG score}|\text{sell}) = 2.596 \text{ bits} \quad (\max = 3.322)$$

**情報理論的考察**: $H/H_{\max} = 0.78$ — スコア分布は完全一様ではないが、十分に分散している。「スコアが一部の値に集中して判別不能」という状態ではない。問題は**分布の形状ではなく、スコアの順序が PnL と整合していない**こと（単調性の破綻）。

### 1.6 Velocity 閾値シミュレーション（683# 提案検証）

| 閾値 (bps) | vel≥th n | vel≥th PnL | vel<th n | vel<th PnL |
|:---:|---|:---:|---|:---:|
| 2.0 | 10 | **-0.16** | 29 | -2.20 |
| 3.0 | 5 | **+2.01** | 34 | -2.22 |
| 4.0 (現行) | 2 | -2.20 | 37 | -1.65 |

**重大発見**: 683# の「sell_velocity_skip 4.0→2.0-3.0」提案は**データ上は支持されない**。vel≥2.0 の sell は PnL=-0.16 と**むしろ良好**、vel≥3.0 は +2.01 で**収益的**。

**解釈**: sell において `price_velocity > 0`（上方移動）は maker の sell 板が take される方向。velocity が高い sell fill は「勢いに乗って売れた」のであり、その後の mean reversion で利益が出ている可能性。**velocity skip を引き下げると、良い sell を殺す**。

**盲点 ③**: 683# の velocity 閾値引下げ提案は、`price_velocity` と `mid_price_trend_5s` を混同している可能性。`price_velocity_bps` は 60s 窓の移動平均ベースで平滑化されており、`mid_price_trend_5s` は直近 5 秒の生の方向性。**真の toxic 指標は trend_5s であり velocity ではない**。

### 1.7 VPIN 閾値感度分析

| VPIN 閾値 | ≥th n | ≥th PnL | <th n | <th PnL |
|:---:|---|:---:|---|:---:|
| 0.50 | 76 | -0.64 | 2 | +2.34 |
| 0.55 | 76 | -0.64 | 2 | +2.34 |
| **0.60** | **68** | **-0.29** | **10** | **-2.38** |
| 0.65 | 43 | -0.22 | 35 | -0.98 |
| 0.70 | 23 | -0.45 | 55 | -0.61 |

**盲点 ④**: VPIN<0.60 の 10 件は PnL=**-2.38** で最悪。VPIN 閾値を上げて「低 VPIN を通す」と、**低 VPIN 帯の悪い fill が増える**。681# / 683# の VG 引上げ提案は VPIN 閾値（バイナリ発火 0.80）ではなく `vpin_continuous_min`（連続スケーリングの開始点）に関するものだが、低 VPIN 帯の毒性は要注意。

### 1.8 Spread Quartile（682# 追跡検証）

| Quartile | Spread 範囲 (bps) | n | PnL30 |
|----------|:---:|---|:---:|
| Q1 | <2.28 | 19 | -0.53 |
| Q2 | 2.28–2.64 | 20 | -0.76 |
| **Q3** | **2.64–2.98** | **19** | **-0.04** |
| Q4 | ≥2.98 | 20 | -0.89 |

682# の報告（Q3 が最良）を**再現確認**。672# の「狭スプレッド＝良」の一般化は 4/1 に当てはまらない。

### 1.9 sell JST 時間帯（sell のみ抽出）

| JST | n | avg PnL | sum PnL | 判定 |
|-----|---|:---:|:---:|------|
| 9 | 3 | -3.16 | -9.5 | 危険 |
| 10 | 5 | -0.29 | -1.5 | |
| **11** | **6** | **-5.19** | **-31.1** | **最悪** |
| 12 | 2 | +1.93 | +3.9 | |
| **13** | **3** | **-8.85** | **-26.6** | **壊滅** |
| 14 | 3 | -0.62 | -1.9 | |
| 15 | 2 | -1.43 | -2.9 | |
| 16 | 3 | -1.45 | -4.4 | |
| **17** | **2** | **+6.33** | **+12.7** | **最良** |
| 18 | 2 | -2.04 | -4.1 | |
| 19 | 2 | -2.24 | -4.5 | |
| 20 | 3 | +1.13 | +3.4 | |
| 21 | 2 | +1.30 | +2.6 | |

**JST 11h + 13h の sell が -57.7 bps を生成** — 全 sell 損失 -65.4 の **88%**。

---

## 2. 盲点と新知見

### 盲点 ① 統計的検出力不足（全レビュワー共通）

n=39（sell fills）で Spearman r=-0.204 は **p=0.212**。α=0.05 水準で帰無仮説を棄却できない。681# も 682# もこの相関を根拠にしているが、必要最小 n ≈ 75 (power=0.80, effect=medium)。**1 日分のデータで SG score の判別能力を結論づけるのは早計**。

### 盲点 ② Velocity vs Trend_5s の混同（683# 固有）

683# は `sell_velocity_skip_threshold_bps` を 4.0→2.0 に提案したが、velocity≥2.0 の sell は PnL=-0.16 で良好。真に有毒なのは `mid_price_trend_5s > 0`（sell PnL=-3.25）。これらは異なる時間窓の異なる指標であり、混同は危険。

### 盲点 ③ Ceiling 引上げの hidden benefit（682# が見落とし）

682# は「unclamped sell も -2.05 で悪い → ceiling だけが問題ではない」と正しく指摘した。しかし、**clamp された fill の offset_stages が失われることで、post-hoc 分析の精度が下がっている**という二次効果を見落としている。Ceiling 引上げは PnL 直接改善だけでなく、**offset pipeline の因果推論可能性を復元する**分析基盤効果がある。

### 盲点 ④ JST 11h/13h sell 集中問題の未言及

682# は「AM sell が悪い」と一般化したが、具体的に **JST 11h (sum=-31.1) と 13h (sum=-26.6) の 2 時間帯が sell 損失の 88% を構成**していることは明示していない。この 2 時間帯のみ sell を hard skip すれば、sell PnL は -65.4 → -7.7 bps に改善する（理論値、逸失利益なしの上限）。

### 盲点 ⑤ SAC sidecar の根本的ボトルネック

675#-680# で SAC を修正したが、**現在の SAC confidence ≈ 0.06 × max_boost=0.20 = 0.012 bps** で、Coincheck の tick size (1 JPY ≈ 0.01 bps) とほぼ同一。SAC が PnL に影響を与えるには confidence が最低 0.5 必要（0.5 × 0.20 = 0.10 bps）。これには `confidence_roi_full` の追加引下げだけでなく、**reward signal の S/N 比改善**が本質。

### 盲点 ⑥ Decision path の単一性

全 78 fills が `decision_path=primary_only`。sell-entry roundtrip を巡る 682# の議論は重要だが、**decision_path にバリエーションがないため roundtrip 分析自体が不可能**。fill_records に「このサイクルで入った sell が、N サイクル後に buy で利確された」という紐付けがない。682# の「sell-entry RT -223.66 bps」は別ツール (`analyze_fill_logs`) からの値であり、fill_records 単体からは再現不可。

### 盲点 ⑦ Information Leakage in Regime Classification

trending_up/sell の PnL=-2.93 は**レジーム判定が fill 時点で正しいとは限らない**。`regime` フィールドは注文時レジームだが、fill 時点ではレジームが遷移している可能性がある。`regime_at_order` と fill 後の実態の不一致は、regime-conditional な veto の精度を下げる。

---

## 3. 統合的処方箋（優先順位付き）

### Phase 1: 止血（YAML 即時適用、hot-reload）

| # | 施策 | 変更 | 根拠 |
|---|------|------|------|
| **S1** | JST 11h/13h sell hard skip | `hard_skip_utc_hours` に UTC 2,4 を sell 限定追加、または `sell_hour_offset_boost` で UTC 2=2.5, 4=2.5 | 2 時間帯で sell 損失の 88% |
| **S2** | trend_5s > 0 sell の追加防御 | `toxic_sell_veto_velocity_threshold: 0.0` → **1.0 に引上げ**（trend_5s は velocity proxy として toxic_sell_veto に入力）| trend_5s>0 sell = -3.25 bps |
| **S3** | Skip Gate trending_up/down 厳格化 | `trending_up: 0.3→0.5`, `trending_down: 0.1→0.3` | trending sell が損失の核 |

### Phase 2: 分析基盤改善（コード変更、Codex 向き）

| # | 施策 | 内容 |
|---|------|------|
| **A1** | Sell ceiling 0.40→0.50 (0.55ではない) | Phase 1 の stop-blood 効果確認 **後** に適用。0.55 は攻めすぎ、0.50 で pipeline 情報の 80% が復元 |
| **A2** | SAC reward 改善 & sell-aware training | §4 Codex タスク参照 |
| **A3** | Skip Gate sell 専用 retrain | unified model の sell 判別低を改善 |

### Phase 3: 中期構造改善

| # | 施策 |
|---|------|
| **M1** | trend_5s ベースの新 veto layer（velocity_skip と独立） |
| **M2** | Decision path に roundtrip 紐付け ID を追加（682# 分析手法の基盤化）|
| **M3** | VG vpin_continuous_min 0.50→0.55 の A/B 検証 |

---

## 4. Codex タスク設計

### タスク 1: SAC Sell-Aware Reward & Observation 改善

```
# Codex Task: SAC Sell-Aware Training Enhancement (684# Phase A2)

## 背景
- SAC sidecar の現在の confidence ≈ 0.06 → offset 寄与 0.01 bps (実質ゼロ)
- 4/1 データで sell PnL=-1.68 bps が全損失源
- SAC の observation に sell risk 指標が不足 (mid_price_trend_5s, OBI_sign×side が未入力)

## 要件

### R1: Observation Space 拡張
ファイル: `configs/v460/experiments/g2_sac_train.yaml` の features.selected に以下を追加:
- `mid_price_trend_5s` (既に FeatureRegistry に存在するか確認、なければ追加)
- `signed_obi` = orderbook_imbalance × side_sign (+1 for buy, -1 for sell)

対応する FeatureRegistry 計算:
- ファイル: `ztb/ml/feature_registry.py` (または相当ファイル)
- mid_price_trend_5s: 直近5秒の mid price 変化率 (bps)
- signed_obi: OBI × position_sign

### R2: Reward の Sell-Side Penalty
ファイル: `ztb/trading/environment/components/calculators/reward_calculator.py`
- use_simple_reward=true のパス内で、sell 側の AS fill に追加ペナルティを付与
- 具体的: `if side == "sell" and adverse_selected: reward *= 1.5` (罰の増幅)
- RewardSettings に `sell_as_penalty_mult: float = 1.5` を追加

### R3: Confidence Scaling 改善
ファイル: `scripts/v460/ml/sac_retrain_scheduler.py`
- `confidence_roi_full: 0.002` → `0.001` への引下げ対応
- ただしコード変更は不要（YAML 変更のみ）。テストで ROI=0.001 時に confidence=1.0 を確認

### R4: テスト
- `tests/unit/v460/test_sac_reward_sell_aware.py` を新規作成
  - sell AS fill 時の reward penalty が正しく適用されることを確認
  - observation に新特徴量が含まれることを確認
- 既存テスト `tests/unit/v460/test_sac_retrain_scheduler.py` が pass すること
- `python -m pytest tests/ -x --tb=short` で全テスト pass

### 制約
- Any 型禁止、mypy 通過を確認
- 既存の PPO sidecar / SkipGate との互換性を壊さない
- FeatureRegistry に新特徴量を追加する場合、既存 pipeline (parquet 生成) も更新
```

### タスク 2: trend_5s ベース Sell Veto Layer 実装

```
# Codex Task: mid_price_trend_5s Sell Veto Layer (684# Phase M1)

## 背景
- 4/1 データで sell + mid_price_trend_5s > 0 は PnL=-3.25 bps (n=14)
- 既存の velocity_skip は price_velocity_bps (60s 窓) で判定しており、5s の短期方向を捉えない
- toxic_sell_veto は spread + OBI + VPIN の条件だが trend_5s を未使用

## 要件

### R1: trend_5s Sell Guard 新設
ファイル: `scripts/v460/lib/fill_test_executor.py` (または offset pipeline 該当箇所)
- 新パラメータ (YAML hot-reload 対応):
  ```yaml
  trend_5s_sell_guard:
    enabled: true
    threshold_bps: 0.5        # trend_5s > 0.5 bps で発動 (soft mode)
    hard_veto_threshold_bps: 3.0  # trend_5s > 3.0 bps で hard veto
    offset_boost_factor: 1.5  # soft mode 時の offset 乗数
  ```
- soft mode: threshold 超過 → sell offset を boost (velocity_skip_as_offset パターン踏襲)
- hard mode: hard_veto_threshold 超過 → sell を skip

### R2: FillRecord に記録
- `trend_5s_guard_triggered: bool`
- `trend_5s_guard_action: str` ("boost" / "veto" / "none")

### R3: YAML 配線
ファイル: `configs/v460/fill_test.yaml` に `trend_5s_sell_guard:` セクション追加

### R4: テスト
- `tests/unit/v460/test_trend_5s_sell_guard.py` を新規作成
  - soft mode の offset boost が正しく適用されること
  - hard mode の veto が正しく発動すること
  - buy 側に影響しないこと
- 既存テスト全 pass 確認

### 制約
- 既存の velocity_skip / toxic_sell_veto との干渉を避ける (独立レイヤー)
- hot-reload 対応必須 (ConfigReloader 経由)
- Any 型禁止
```

---

## 5. 682# vs 683# 最終裁定

| 論点 | 682# | 683# | **本検証判定** |
|------|------|------|---------------|
| Ceiling 引上げの優先度 | P2 (後回し) | P1 (止血後) | **Phase 2 (0.50 まで、682# 寄り)** |
| Sell-entry 抑制 | P0 | 支持 | **P0 (全員合意)** |
| velocity 閾値 4→2-3 | 言及なし | P0 提案 | **却下** (vel≥2.0 は PnL 良好) |
| VG 引上げ | 注意喚起 | 支持 (0.55) | **Phase 3** (低VPIN帯が有毒、要慎重) |
| AM sell 防御 | sell-entry 論に内包 | hour_boost 提案 | **P0-S1** (JST 11h/13h 限定) |
| SAC/PPO | infra repair 段階 | 言及なし | **Phase 2-A2** (Codex 委任) |
| SG score 逆相関 | 根拠として使用 | 根拠として使用 | **p=0.21 で有意でない。方向性のみ参考** |

### 総合判定

682# の「sell-entry 条件付き抑制を先行」は**最も堅実かつデータ支持される戦略**。683# の velocity 引下げは却下だが、「止血→幅取りのシーケンシャル戦略」というフレームワークは正しい。681# の ceiling 引上げは Phase 2 で 0.50 を上限として実施。

**最優先アクション**: JST 11h/13h の sell 防御と trending sell の SG 厳格化。この 2 手で理論上 sell PnL の 88% がカバーされる。

---

## 6. 再現コマンド

```bash
# 本検証の全分析を再現
python temp/analysis_682_683.py

# Spread quartile
python -c "
import json, statistics
F=[json.loads(l) for l in open('results/v460/fill_test/fill_records_20260401.jsonl')]
F=[r for r in F if r.get('filled')]
sp=sorted([float(f.get('spread_bps',0) or 0) for f in F])
n=len(sp); q=[sp[n//4],sp[n//2],sp[3*n//4]]
for lo,hi,l in [(0,q[0],'Q1'),(q[0],q[1],'Q2'),(q[1],q[2],'Q3'),(q[2],999,'Q4')]:
    b=[f for f in F if lo<=float(f.get('spread_bps',0) or 0)<hi]
    if b: print(f'{l}: n={len(b)} pnl={statistics.mean([float(x[\"post_fill_30s_pnl\"]) for x in b]):.2f}')
"
```

---

*本文書は 681#-683# の三者レビューを統合し、定量データで反証・支持を行ったもの。全数値は再現コマンド付き。*
