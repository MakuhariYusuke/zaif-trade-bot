# 425# 423#/424# レビュー妥当性評価 + 両者の盲点補完

**Date**: 2026-03-15  
**Phase**: ph4  
**Type**: rev (レビュー妥当性評価)  
**対象**: `docs/v460/423_ph4_rev_422_100k_training_review_and_next_options.md` (Codex), `docs/v460/424_ph4_rev_422_423_gemini_second_opinion_and_self_review.md` (Gemini)  
**方法**: 実装コード照合 + 全実験結果ファイル横断分析

---

## 1. 423# (Codex) 妥当性評価

### 1.1 正しい指摘

| # | 指摘 | 判定 | 検証根拠 |
|---|---|---|---|
| F1 | val_ratio 交絡は正しいが「単独原因は言い切りが強い」 | **◎ 正確** | 下記 §3 で定量的に裏付け |
| F2 | best_model をそのまま差し替えるだけでは不十分 | **◎ 正確** | best_model 自体が 5K step proxy で選ばれており、100K OOS ROI 0.06–0.15% の中での選別はノイズ支配的 |
| F3 | `100K × val_ratio=0.02` は diagnostic only | **◎ 正確** | 短い favorable tail で通る設定に最適化する危険。422# の F 実験候補は昇格させない方がよい |
| F4 | 100K seed 分散爆発は「alpha のレジーム非頑健性」 | **○ 妥当** | ただし §3 の発見で修正が必要 (後述) |
| F5 | Attention/PPO/TD3 へのジャンプは早い | **◎ 正確** | 評価ハーネス修正が先 |
| F6 | SAC を sidecar 化する方が安全 | **○ 方向は妥当** | ただし根拠は「100K FAIL」ではなく「評価系不備で真の実力が不明」の方がより正確 |

### 1.2 過大評価されている点

| # | 指摘 | 問題 |
|---|---|---|
| O1 | 「walk-forward 的評価へ寄せる」を P0 行動リストに入れている (§7, §8) | walk-forward は構造変更が大きい。424# セルフレビューでも指摘の通り、現フェーズでは過エンジニアリング。P0 には不適 |
| O2 | 「regime-conditioned policy selection」を P2 で提案 | regime 識別精度自体が未検証であり、この段階で multi-policy 選択に投資するのは premature |

### 1.3 見落としている点 → §3 で補完

---

## 2. 424# (Gemini) 妥当性評価

### 2.1 正しい指摘

| # | 指摘 | 判定 | 検証根拠 |
|---|---|---|---|
| G1 | 5K step 打切りの「真の恐ろしさ」= Training 直後 ~3.5 日への過学習 | **○ 論理は妥当** | ただし `_checkpoint_eval_roi()` は `oos_eval_env` (val_df 全体) に対して先頭 5K step を実行するため、「Training 終盤の直後」ではなく「OOS 期間の最初の ~3.5 日」が正しい ([sac_train.py L471-493](scripts/v460/lib/tasks/sac_train.py)) |
| G2 | final vs best の二元論ではなく、tail holdout 構造の欠陥を直視すべき | **◎ 正確** | `train_val_split()` は末尾固定 holdout。時系列の single-tail 問題は本質的 ([sac_common.py L296-330](scripts/v460/lib/sac_common.py)) |
| G3 | 169 日 OOS は「モデルの賞味期限」とミスマッチ | **△ 一面的** | セルフレビューで自ら批判している通り、「レジームが変わるから仕方ない」は開発者の甘え。ガードが整っていれば損失ゼロに抑え込めるはず |
| G4 | Walk Forward 欠如 | **○ 正しいが時期尚早** | セルフレビューで自ら認める通り、423# の「まず評価ハーネスを直せ」が現実的 |

### 2.2 過大評価されている点

| # | 指摘 | 問題 |
|---|---|---|
| O1 | 「時間スケールのミスマッチ」を両レビューの欠落観点として提示 | 423# §4.1 は「レジーム非頑健性」として同じ論点を既に提示している。「欠落」ではなく「別の言い方」 |
| O2 | 「SAC 万能主義の限界」→ sidecar ダウングレードに全面同意 | sidecar 化の是非判断は、まず評価系修正で真の実力を見てからが正しい順序。現時点で「限界」と断じるのは早い |

### 2.3 セルフレビューの質

424# のセルフレビュー (§3) は自己批判として機能しており、論理の一方向性を自ら補正している。特に「169 日 OOS での FAIL を重く受け止めた 422# の危機感は正しい」という結論は妥当。

### 2.4 見落としている点 → §3 で補完

---

## 3. 両者が見落としている重大な盲点

### 3.1 ★ 既存実験データに「S1 相当」の evidence が存在する

422# は「S1: 20K × val_ratio=0.20 を最優先実験」として提案し、423#/424# もこれに同意している。  
しかし、**既存の result ファイルを横断分析すると、S1 実験を待たずに val_ratio 仮説を部分検証できるデータが既にある。**

#### 全実験 G3 クロス集計 (result files 横断)

| Config | timesteps | val_ratio | G2 | G3 | pf_med | sharpe |
|---|---|---|---|---|---|---|
| reward_clean (409#) | 20K | 0.02 | PASS | **PASS** | 1.145 | 5.70 |
| reward_clean (dup) | 20K | 0.02 | PASS | **PASS** | 1.145 | 5.70 |
| reward_clean_m1 (414#) | 20K | 0.02 | PASS | **PASS** | 1.120 | 3.88 |
| γ=0.95 reward-tuned (387# fast) | 20K | 0.02 | PASS | **FAIL** | 1.006 | 0.22 |
| reward_clean_m1m2 (414#) | 20K | 0.02 | FAIL | **FAIL** | 1.011 | 1.18 |
| baseline (385#) | 20K | 0.02 | FAIL | **FAIL** | 1.042 | 2.68 |
| **reward_clean_100k** | **100K** | **0.20** | FAIL | **FAIL** | 1.037 | 0.34 |

#### 導出される結論

1. **val_ratio=0.02 でも G3 FAIL する構成は複数ある** (reward-tuned, m1m2, baseline)。  
   → 422# の「20K G3 PASS は偽陽性」は **reward-clean 固有の成功** であり、val_ratio だけの問題ではない。

2. **G3 PASS を達成したのは reward-clean 系 (無印 + m1) のみ**。  
   → val_ratio=0.02 の「楽観」は事実だが、reward-clean の reward 設計自体にも要因がある。

3. **reward-tuned (387# G2 PASS) は val_ratio=0.02 ですら G3 FAIL**。  
   → 422# の「root cause = val_ratio 交絡」は不完全。reward 設計 × val_ratio の **交互作用** が正しい。

### 3.2 ★ S1 実験の「期待結果」の解釈を事前修正すべき

422# は「S1 (20K × val_ratio=0.20) が FAIL なら偽陽性確定」としている。  
423# もこれを採用している。

しかし §3.1 の cross-tab から、以下の解像度上げが必要:

| S1 結果 | 正しい解釈 |
|---|---|
| **FAIL** | val_ratio=0.02 が楽観的だったことの追加確認。だが「reward-clean が悪い」のか「val_ratio=0.02 でしか通らない」のかは区別不能 |
| **PASS** | val_ratio=0.20 でも reward-clean は通る → 100K の問題は timesteps (学習飽和 or checkpoint 選別) 側にある |

つまり **FAIL でも root cause の一意特定にはならない**。  
FAIL の場合の追加実験として、**reward-tuned × 20K × val_ratio=0.20** (= S1') も同時に走らせ、reward 設計の交互作用を分離すべき。

### 3.3 ★ checkpoint eval の 5K step は「OOS の最初の ~3.5日」を評価している

424# G1 は「Training 終盤のレジームへの過学習」と述べているが、実装上は:

```python
# sac_train.py L471-493: _train_with_checkpoints
oos_roi = _checkpoint_eval_roi(model, oos_eval_env)  # oos_eval_env = val_df 全体
```

`oos_eval_env.reset()` は val_df の **先頭** (= train 期間の直後) から走査開始する。  
したがって checkpoint eval が見ているのは「OOS 期間中で最も ancient な ~3.5 日」。

これは 424# の「Training 直後に最も overfit した checkpoint を選ぶ」という批判を **強化** する:  
OOS 先頭は train 期間と regime が連続しやすく、checkpointの初期化段階での in-distribution 成功を過大評価するバイアスがある。

### 3.4 ★ F6 OOS ROI と G3 final ROI の乖離パターン

422# §6.2 の乖離テーブルで:

| Seed | F6 best OOS ROI | G3 final ROI | gap ratio |
|---|---|---|---|
| 42 | 0.113% | +6.640% | 58.8x |
| 123 | 0.154% | -0.513% | -3.3x |
| 456 | 0.148% | +2.693% | 18.2x |
| 789 | 0.064% | -3.385% | -53.2x |

**両レビューとも「乖離が大きい」で終わっているが、パターンを見逃している。**

- F6 OOS ROI は全 seed とも **微小正** (0.06–0.15%)  
- G3 final ROI は seed 42/456 が大幅プラス、123/789 が大幅マイナス

OOS 先頭 3.5 日では全 seed が似た performance なのに、full OOS では diverge する。  
→ **分岐は OOS 後半 (train から遠い期間) で発生**。これは 424# の「時間スケールのミスマッチ」仮説を定量的に裏付ける。

修正の方向は:
1. F6 eval を multi-slice (先頭 5K + 中間 5K + 末尾 5K) にする　← 423# 提案と合致
2. **worst-slice で棄却** するゲートを追加 ← 423# §6.3 のアイデアだが F6 段階に前倒し

### 3.5 ★ 100K の reward_profit_corr 改善は「alignment 改善だが方向の汎化失敗」

422# の corr データ:
- 20K seeds: 0.537, 0.562, **-0.203**, 0.606 (median 0.549)
- 100K seeds: **0.977**, 0.674, **-0.562**, 0.856 (median 0.765)

corr median が改善している (0.549→0.765) のに PF/Sharpe が悪化するという paradox がある。  
423# §4 は「alpha のレジーム非頑健性」で説明し、424# は「時間スケールのミスマッチ」で説明しているが、**いずれも corr paradox を直接扱っていない**。

考えられる説明:
- 100K で学習が進むと reward と PnL の **相関は強まる** が、reward 自体が **非汎化的な行動** を強化している
- 言い換えると、「reward に従う精度は上がったが、reward が正しい方向を指していない期間が long OOS に含まれる」
- つまり **reward 関数自体はレジーム非依存だが、それが示す action が特定レジームでしか profitable でない**

これは 422# C1 (Reward 関数の根本見直し) の優先度を **B→A に上げるべき** ことを示唆する。

---

## 4. 修正された推奨アクション順

### P0: 即時 (evaluation harness fix)

| # | アクション | 根拠 | 工数 |
|---|---|---|---|
| P0-1 | **S1: 20K × val_ratio=0.20** を実行 | 422#/423# 合意。§3.1 により結果解釈の解像度を上げておく | ~30分 |
| P0-2 | **S1': reward-tuned × 20K × val_ratio=0.20** を同時実行 | §3.2: reward 設計 × val_ratio 交互作用の分離 | ~30分 |
| P0-3 | **F6 eval を 5K→multi-slice** (先頭/中間/末尾 各 5K) に変更 | 423#/424# 合意 + §3.4 で定量的に裏付け | コード変更のみ |
| P0-4 | **final_model と best_model の full OOS 並行評価** を G3 パイプラインに追加 | 423# §3.2 の手順が正しい (先に差し替えではなく両方評価) | コード変更のみ |

### P1: S1/S1' 結果待ち

| S1 結果 | S1' 結果 | 次のアクション |
|---|---|---|
| FAIL | FAIL | val_ratio=0.02 は全般的に楽観的 → 過去の G3 PASS 全体の信頼性見直し |
| FAIL | PASS | reward-clean 固有の問題 → reward 関数の再設計が最優先 |
| PASS | FAIL | reward-clean は頑健だが reward-tuned は弱い → architecture/hparam 比較に進む |
| PASS | PASS | 100K の問題は timesteps 側 → F6 multi-slice + best_model 差替えで改善可能性 |

### P2: 評価系修正後の比較実験

1. 同一 multi-slice F6 + full OOS で `[256,256]` vs `[128,128]`
2. `weight_decay` 小レンジ (0 / 1e-5 / 1e-4)
3. Checkpoint selection を ROI 単独 → `ROI - λ·MaxDD` に拡張

### HOLD (423# と同意)

- `100K × val_ratio=0.02` を本命候補に昇格
- `Attention / Residual / PPO / TD3` への即座のジャンプ
- checkpoint `state_dict` 平均
- Walk-forward 実装 (P0-P2 修正後に再検討)
- SAC sidecar 化の即断 (評価系修正後の真の実力を見てから判断)

---

## 5. 結論

### 423# (Codex) の総合評価: **B+ (良)**

- 422# の過強主張を適切に補正し、val_ratio 以外の要因 (F6 proxy 品質、レジーム非頑健性) を正しく指摘
- 推奨アクション順は実務的で現実的
- ただし walk-forward を P0 に入れている点、既存データの横断分析が欠けている点が弱い

### 424# (Gemini) の総合評価: **B (良)**

- 「時間スケールのミスマッチ」の観点追加は有価値だが、423# §4.1 と実質重複
- セルフレビューは論理の一方向性を自ら補正しており質が高い
- 5K step の「Training 直後への過学習」は方向が正しいが実装の詳細 (OOS 先頭から走査) を確認していない
- 「SAC 万能主義の限界」結論は時期尚早

### 両者が共通して見落としていた点

1. **既存データで S1 仮説の部分検証が可能**。reward-tuned は val_ratio=0.02 でも G3 FAIL → val_ratio は唯一の原因ではない
2. **S1 実験に S1' (reward-tuned 版) を併走させないと交互作用が区別不能**
3. **corr paradox** (alignment 改善 × profitability 悪化) を直接扱っていない → reward 関数見直しの優先度を上げるべきシグナル
4. **F6 OOS ROI の seed 間一致 vs G3 ROI の seed 間 diverge** のパターンから、分岐は OOS 後半で起きている (multi-slice の定量的根拠)
