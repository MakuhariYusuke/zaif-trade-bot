# 422# 100K Training Result — Deep Forensic Analysis

**Date**: 2026-03-14  
**Experiment**: Baseline A (reward-clean [256,256]) × 100K timesteps  
**Config**: `g2_sac_reward_clean_100k.yaml`  
**Result**: **G3 FAIL** ← 20K では PASS だった Config A が 100K で FAIL

---

## 1. Executive Summary

20K Baseline Aは G3 PASS (ROI median 0.594%, Sharpe 5.70) だったが、同一ハイパーパラメータで
100K に拡張した結果 **G3 FAIL** となった。主因は以下の3つ:

| G3 Check | 20K | 100K | 判定 |
|---|---|---|---|
| pf_median (>1.05) | 1.1451 | **1.0373** | FAIL (margin: -0.013) |
| pf_worst (>0.95) | 1.0893 | **0.8733** | FAIL (margin: -0.077) |
| sharpe_annual (>0.8) | 5.7021 | **0.3406** | FAIL (margin: -0.459) |
| max_drawdown (<15%) | 0.26% | 4.67% | PASS |
| gross_gt_fee | True | True | PASS |
| reward_profit_corr (>0) | N/A | 0.7645 | PASS |

**最重要発見**: 20K G3 PASSは **val_ratio=0.02 (偽陽性の疑い)** によるもので、  
100K の val_ratio=0.20 がより信頼性の高い評価であった可能性が極めて高い。

**423# レビューで追加発見した重大盲点 2件**:
1. **F6 checkpoint 評価 = 5,000 ステップ打切り** → 100K の OOS 243K 行の **2.1%** しか評価せず best_model 選択
2. **G3 最終評価は F6 best_model ではなく 100K 完了後の final_model で実行** → F6 は保存するだけで使われていない

---

## 2. Experimental Setup — 重要な差異

| Parameter | 20K Config | 100K Config | 影響度 |
|---|---|---|---|
| total_timesteps | 20,000 | 100,000 | 直接 |
| **val_ratio** | **0.02** | **0.20** | **★致命的** |
| buffer_size | 200,000 | 200,000 | 同一 |
| checkpoint_interval | 5,000 | 10,000 | 軽微 |
| architecture | [256,256] | [256,256] | 同一 |
| params/data ratio | 10.8x (@20K) | 2.15x (@100K) | 大 |
| **OOS期間** | **~17日** | **~169日** | **★致命的** |

### val_ratio の衝撃

| | val_ratio=0.02 (20K) | val_ratio=0.20 (100K) |
|---|---|---|
| Training rows | 1,192,591 | 973,544 |
| Validation rows | **24,339** | **243,386** |
| OOS coverage | ~17日 (405.6h) | ~169日 (4056.4h) |
| 統計的信頼性 | **低** | **高** |
| F6 checkpoint eval coverage | 5000/24339 = **20.5%** | 5000/243386 = **2.1%** |

> **Critical**: val_ratio=0.02 は市場データの一部切片しか検証していない。  
> 好条件の期間に偏った可能性がある。

---

## 3. Seed別詳細比較

### 3.1 最終評価 (OOS)

| Metric | 20K s42 | 20K s123 | 20K s456 | 20K s789 | 100K s42 | 100K s123 | 100K s456 | 100K s789 |
|---|---|---|---|---|---|---|---|---|
| ROI% | 0.690 | 0.637 | 0.332 | 0.550 | **6.640** | -0.513 | 2.693 | **-3.385** |
| PF | 1.198 | 1.116 | 1.089 | 1.174 | **1.261** | 0.984 | 1.091 | **0.873** |
| Sharpe | 5.76 | 6.00 | 3.02 | 5.64 | 5.20 | -0.23 | 0.91 | **-2.02** |
| MaxDD% | 0.21 | 0.20 | 0.26 | 0.20 | 0.88 | **4.67** | 2.21 | **4.07** |
| Corr | 0.537 | 0.562 | -0.203 | 0.606 | **0.977** | 0.674 | -0.562 | 0.856 |
| Trades | 10,749 | 22,481 | 10,699 | 13,071 | 102,407 | 126,015 | 121,169 | 77,591 |

### 3.2 統計サマリー

| Metric | 20K | 100K | 変化方向 |
|---|---|---|---|
| ROI mean | 0.552% | 1.359% | ▲ (ただし巨大分散) |
| ROI median | 0.594% | 1.090% | ▲ |
| **ROI std** | 0.137% | **3.731%** | ▼▼▼ (27.3倍) |
| **ROI CV** | 0.25 | **2.75** | ▼▼▼ (11倍) |
| PF median | 1.145 | 1.037 | ▼ |
| PF worst | 1.089 | 0.873 | ▼▼ |
| Sharpe median | 5.702 | 0.341 | ▼▼▼ |
| Corr median | 0.549 | 0.765 | ▲ |
| MaxDD worst | 0.26% | 4.67% | ▼▼ (18倍) |

> **重大所見**: ROI の CV が 0.25 → 2.75 (11倍悪化)。  
> 100K ではシード間の分散が爆発的に増大。

---

## 4. 学習曲線分析

### 4.1 F6 Best-Model Selection

| Seed | 20K best_ckpt | 20K OOS_ROI | 100K best_ckpt | 100K OOS_ROI |
|---|---|---|---|---|
| 42 | 20,000 steps | 0.429% | 60,000 steps | 0.113% |
| 123 | 5,000 steps | 0.575% | 40,000 steps | 0.154% |
| 456 | 10,000 steps | 0.465% | 20,000 steps | 0.148% |
| 789 | 10,000 steps | 0.422% | 30,000 steps | 0.064% |

> **衝撃的な事実**: 20K の best OOS ROI (0.42-0.58%) は 100K の best OOS ROI (0.06-0.15%) の
> **3-7倍高い**。val_ratio=0.02 の OOS は明らかに楽観的。

### 4.2 Checkpoint OOS ROI 推移 (100K, 全seed)

```
Step     s42      s123     s456     s789     Mean
10K     0.030%   0.070%   0.050%   0.010%   0.040%
20K     0.010%   0.100%   0.150%★  -0.010%  0.063%
30K    -0.020%   0.070%   0.050%   0.060%★  0.040%
40K     0.010%   0.150%★  0.030%  -0.010%   0.045%
50K     0.110%★  0.080%   0.000%   0.040%   0.058%
60K     0.110%★  0.050%   0.030%   0.010%   0.050%
70K     0.060%   0.030%   0.060%  -0.050%   0.025%
80K    -0.030%   0.060%  -0.000%  -0.020%   0.003%
90K     0.010%   0.090%   0.080%  -0.030%   0.038%
100K    0.040%   0.090%   0.040%  -0.020%   0.038%
```

### 4.3 In-Sample ROI 推移 (100K)

```
Step     s42      s123     s456     s789     Mean
10K     0.220%   0.280%   0.260%   0.220%   0.245%
20K     0.120%   0.310%   0.190%   0.160%   0.195%
30K     0.120%   0.260%   0.170%   0.150%   0.175%
40K     0.150%   0.170%   0.170%   0.120%   0.153%
50K     0.120%   0.250%   0.170%   0.120%   0.165%
60K     0.120%   0.190%   0.100%   0.110%   0.130%
70K     0.110%   0.260%   0.260%   0.140%   0.193%
80K     0.130%   0.220%   0.160%   0.100%   0.153%
90K     0.260%   0.240%   0.090%   0.130%   0.180%
100K    0.130%   0.250%   0.090%   0.090%   0.140%
```

### 4.4 パターン分析

1. **In-Sample ROI**: 10K 時点の 0.245% から 全体的に緩やかに低下 → 100K で 0.140%
2. **OOS ROI**: 散在的に正負を振動、80K 以降は mean 0.003-0.038% に低下
3. **OOS ROI ピーク**: 全シードとも 20K-60K 範囲にベスト、80K以降のベスト更新なし
4. **IS-OOS Gap**: 100K 時点で IS=0.140% > OOS=0.038%、gap=0.102% (軽微な過学習シグナル)

---

## 5. 仮説検証 — 何が起きたのか

### H1: 「100K で過学習した」 → △ 部分的に支持

- IS ROI の低下トレンドは典型的な過学習ではない（過学習なら IS は上がる）
- むしろ **学習飽和** (saturation): 追加データが新情報を提供しない
- F6 が早期チェックポイントを選択 → 100K フルトレーニングの恩恵なし
- 過学習よりも **「無益な学習」** が正確

### H2: 「val_ratio=0.20 が厳しすぎる」 → ◎ 強く支持

| 証拠 | 詳細 |
|---|---|
| OOS ROI 差 | 20K best OOS: 0.42-0.58% vs 100K best OOS: 0.06-0.15% (3-7倍) |
| 事実 | **同じモデル構造、同じ学習アルゴリズム**でこの差は val_ratio 以外で説明困難 |
| メカニズム | val_ratio=0.02 → OOS=~24K行 → 特定期間に偏る → 楽観的評価 |
| 含意 | 20K G3 PASS は **偽陽性 (false positive)** の可能性 |

### H3: 「シード感度が 100K で悪化した」 → ○ 支持だが代替説明あり

- 20K: ROI CV=0.25 (全シード安定的プラス)
- 100K: ROI CV=2.75 (seed42=+6.64%, seed789=-3.39%)
- **仮説**: val_ratio=0.20 の大きな OOS データに対し、シードごとに異なる戦略を学習
  - 好運なシード (42): 市場パターンに合致する戦略を偶然発見
  - 不運なシード (789): 逆方向の戦略を学習

**★423# レビュー追記**: シード分散の増大は「val_ratioの差異」でも説明可能。
val_ratio=0.02 (24K行 = ~17日) は燭い期間のため、どのシードでも似た ROI になりやすい。
val_ratio=0.20 (243K行 = ~169日) は多様な市場環境を含むため、
シードごとの戦略差が ROI に大きく反映される。
つまり **「シード感度が悪化」のではなく「評価期間が広がり真のバラつきが見えた」** 可能性。

### H4: 「params/data ratio の改善 (10.8x→2.15x) が有害」 → ○ 間接的に支持

- 414# の分析では 10.8x の「暗記」が 20K 成功の鍵だった
- 2.15x では暗記ができない → 真の汎化が必要 → しかし汎化に失敗
- ただし val_ratio 0.20 のデータ分割でさらに訓練データ減少 (1.2M → 974K)

### H5: 「学習自体は進んでいるが評価条件が変わった」 → ◎ 最有力

**20K と 100K は同じ実験ではない**:
1. 訓練データ: 1.2M rows → 974K rows (19% 減少)
2. 評価データ: 24K rows → 243K rows (10倍 増加)
3. 評価期間: ~2 週間 → ~4 ヶ月
4. F6 best checkpoint: 5K-20K → 20K-60K

> 結論: 100K G3 FAIL は「100K が悪い」のではなく、  
> **「20K G3 PASS が不当に楽観的だった」** 可能性が最も高い。

---

## 6. 交絡因子分析

### 6.1 val_ratio × timesteps の交絡

現在のデータでは 20K=val_ratio 0.02 と 100K=val_ratio 0.20 が完全に交絡。
分離するには以下の実験が必要:

| 実験 | timesteps | val_ratio | 目的 |
|---|---|---|---|
| A (既存) | 20K | 0.02 | ✅ G3 PASS |
| D (100K) | 100K | 0.20 | ✅ G3 FAIL |
| **E (提案)** | **20K** | **0.20** | val_ratio 効果の分離 |
| **F (提案)** | **100K** | **0.02** | timesteps 効果の分離 |

### 6.2 F6 Best-Model と最終評価の乖離 ★423# レビュー追記

**423# レビューで特定した乖離の2つの根本原因**:

#### 原因 1: F6 checkpoint eval = 5,000 ステップ打切り

`_checkpoint_eval_roi()` (sac_train.py L533-556) は `_CHECKPOINT_EVAL_MAX_STEPS = 5_000` で
OOS 全走査せず先頭 5K ステップのみで ROI を算出。最終 G3 eval (`evaluate_model_oos`) は
`max_steps_per_episode=None` で OOS 全 243K ステップを走査。

| | F6 checkpoint eval | G3 final eval |
|---|---|---|
| 関数 | `_checkpoint_eval_roi()` | `evaluate_model_oos()` |
| OOS steps | **5,000** | **243,386** |
| Coverage | **2.1%** (100K) / 20.5% (20K) | **100%** |
| 指標 | ROI のみ | ROI + PF + Sharpe + MaxDD + Corr |

#### 原因 2: G3 final eval は final_model を使用 (best_model ではない)

`sac_train.py` L195:
```python
eval_metrics = _evaluate_trained_model(model, eval_env, cfg)
```
ここでの `model` は `model.learn(total_timesteps=...)` 完了後の **最終モデル (100K step)**。
F6 が best_model_path に保存した early-stop モデルは **G3 評価に使用されない**。

| Seed | F6 best step | F6 OOS ROI | G3 final ROI | gap ratio |
|---|---|---|---|---|
| 42 | 60K | 0.113% | +6.640% | 58.8x |
| 123 | 40K | 0.154% | -0.513% | -3.3x |
| 456 | 20K | 0.148% | +2.693% | 18.2x |
| 789 | 30K | 0.064% | -3.385% | -53.2x |

> **設計欠陥**: F6 は「過学習回避のための early stopping」を意図しているが、
> 保存するだけで G3 評価には使っていない → best_model の効果が完全に無効化されている。

### 6.3 F6 チェックポイント OOS カバレッジの非対称性

20K (val_ratio=0.02) では F6 は OOS の 20.5% を見ていたが、
100K (val_ratio=0.20) では OOS のわずか 2.1% しか見ていない。
これは F6 の best-model 選択の信頼性が 100K で大きく低下していることを意味する。
ただし上述の通り、現状 best_model は G3 評価に使われないため直接の影響はない。

---

## 7. 方策提案

### 優先度 S: 即座に実行すべき

#### S1: val_ratio 分離実験 (20K × val_ratio=0.20)

```yaml
# 期待: G3 FAIL → 20K G3 PASS が偽陽性だったことの証明
total_timesteps: 20000
val_ratio: 0.20  # ← 変更点
```

- **所要時間**: ~30 分 (既存 20K パイプラインと同等)
- **期待結果**: G3 FAIL → 真の性能は 100K 同等
- **これが PASS なら**: 100K の問題は timesteps にあり、根本的再設計が必要
- **ROOT CAUSE 特定の最重要実験**

### 優先度 A: 戦略的に重要

#### A1: F6 best_model を G3 評価に使用する修正 ★423# 具体化

現在 G3 eval は final_model (100K完了後) で実行されている。
F6 が保存した best_model で G3 eval を実行するよう修正すべき:

```python
# sac_train.py L195 付近: best_model があればそれをロードして評価
if best_model_path and best_model_path.exists():
    from stable_baselines3 import SAC
    model = SAC.load(str(best_model_path), env=eval_env)
    logger.info(f"F6: Loaded best model from {best_model_path} for G3 eval")
eval_metrics = _evaluate_trained_model(model, eval_env, cfg)
```

また F6 checkpoint eval の `max_steps=5000` を増やすか、
val_ratio に応じて動的に調整すべき（100K では 2.1% しか見ていない）。

#### A2: Seed Variance 低減策

100K での CV=2.75 は許容不能。検討すべき施策:
1. **Multi-seed ensemble**: 4 seed の予測を平均化
2. **Checkpoint ensemble**: 各 seed の複数チェックポイントを平均化
3. **G3 gate の見直し**: median ではなく worst-case に近い基準

### 優先度 B: 中期的施策

#### B1: 正則化強化

- Weight decay (M2): 20K 実験 C で corr 改善効果あり、100K で再検証
- Dropout: 未検証、追加検討

#### B2: Architecture 最適化

- 100K で ratio=2.15x は健全だが、汎化失敗
- **Attention-based**: 時系列の重要な特徴に選択的に注目
- **Residual connections**: 勾配消失防止

#### B3: データ拡張・多様化

- 訓練データの質が問題の可能性
- Walk-forward validation の導入

### 優先度 C: 長期的

#### C1: Reward 関数の根本見直し

- reward_profit_corr は 100K で改善 (0.55→0.76)
- しかし PF と Sharpe は悪化
- **Alignment は改善しているが profitability がない** → reward が利益と無関係な行動を強化

#### C2: Alternative RL アルゴリズム

- PPO, TD3 等の安定性の高いアルゴリズム検討
- SAC の entropy regularization が不安定化の原因か

---

## 8. 結論と次ステップ

### 根本的洞察

> **20K G3 PASS は val_ratio=0.02 による偽陽性の可能性が高い。**  
> **100K G3 FAIL は val_ratio=0.20 による、より信頼性の高い評価結果。**

この仮説が正しければ、現在の SAC + [256,256] + reward-clean 構成は
**真のOOS汎化能力を持っていない** ことになる。

### 即時アクション

1. **S1 実験** (20K × val_ratio=0.20) を最優先で実行 → 偽陽性仮説の検証
2. **A1 修正**: F6 best_model で G3 評価を実行するよう sac_train.py を修正
   - これは S1 と独立して実施可能（コード修正のみ）
3. 結果に応じた戦略分岐:
   - **S1 FAIL**: 偽陽性確定 → reward/architecture の根本見直し
   - **S1 PASS**: timesteps 問題 → A1 修正の効果で改善可能性あり

### リスク認識

1. **val_ratio 系統的リスク**: G3 PASS を達成した構成はすべて val_ratio=0.02。
   全ての過去 G3 PASS 結果の信頼性に疑問。
2. **F6 死蔵リスク**: best_model は保存されるが G3 評価に使用されない。
   F6 の early-stopping 効果が完全に無効化されている。
3. **F6 OOS カバレッジ低下**: val_ratio=0.20 時の F6 checkpoint eval は OOS の
   2.1% しか見ておらず、best_model 選択自体の信頼性も低い。

---

## Appendix A: Trade Frequency 安定性 (423# レビュー追加)

トレード頻度 (trades/row) は 20K⇔100K でほぼ一致しており、
モデルの行動パターン自体は大きく変わっていない:

| Seed | 20K trades/row | 100K trades/row |
|---|---|---|
| 42 | 0.442 | 0.421 |
| 123 | 0.924 | 0.518 |
| 456 | 0.440 | 0.498 |
| 789 | 0.537 | 0.319 |

→ 同程度の売買頻度で ROI 分散が爆発 = **トレード方向 (質) の問題**であり
トレード回数 (量) の問題ではない。

## Appendix B: Raw Data Reference

- 100K Result: `results/v460/v460_g2train_seed42_20260314_074228.json`
- 20K Baseline: `results/v460/v460_g2train_seed42_20260313_101021.json`
- Training Log: `temp/reward_clean_100k_log.txt` (UTF-16)
- Config: `configs/v460/experiments/g2_sac_reward_clean_100k.yaml`

