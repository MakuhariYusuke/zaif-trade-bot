# 434# 426#/432#/433# 横断レビュー: 評価設計・fill実態・次期エッジ案の再整列

> **種別**: rev
> **対象**: 426# / 432# / 433#
> **関連**: 423#-431# (SAC評価系・Sidecar/Clamp監査), 057# / 070# (AS分類・walk-forward資産)
> **レビュー日**: 2026-03-15

---

## §0 Executive Summary

3文書はそれぞれ見ている層が違う。

- `426#` は **シミュレーション評価設計** の問題を扱っている
- `432#` は **live fill 実態** の問題を扱っている
- `433#` は **次の alpha / veto backlog** を扱っている

したがって、3つを同じ土俵で「正しい / 間違い」と裁くより、**どこまで実証できているか** と **今どの順で採ると儲けに近いか** に分けて整理する方が実務的である。

今回の結論は以下。

| 対象 | 総合判定 | 主結論 |
|---|---|---|
| `426#` | 概ね妥当 | `reward_clean > reward_tuned` と `long OOS での脆弱化` は裏付けあり。ただし `optimal val_ratio=0.05-0.10` や `1D action space が主犯` は未実証 |
| `432#` | 有益だが一部補正必要 | `Skip Gate 崩壊`、`buy+ranging`、`reprice悪化` は強い。だが `ceiling=0.15 一律` は現行 config と不一致で、因果の言い切りはまだ強い |
| `433#` | 発想は良いがそのまま実装指示には使わない | `toxicity sidecar` が最有望。`BitFlyer lead-lag` と `queue heuristic` は二軍候補。文書内の制御文字混入は先に直すべき |

profit-first に言えば、**最優先は 433 の大改造ではなく、432 の live 実態を使って toxicity veto を既存資産で差し込むこと** である。426 はその補助として「SAC 単体の G3 を勝ち筋の中心に置きすぎない」ことを教えている。

---

## §1 今回の裏取りで確認できた事実

### §1.1 426# の数値は概ね再現できる

結果 JSON を照合すると、426# の主な headline は再現できた。

- `results/v460/v460_g2train_seed42_20260313_055925.json`: original clean は `G3 PASS`, `pf_median=1.145074`
- `results/v460/v460_g2train_seed42_20260312_201227.json`: original tuned は `G3 FAIL`, `pf_median=1.005857`
- `results/v460/v460_g2train_seed42_20260314_173804.json`: `S1` は `G3 FAIL`, `pf_median=1.049216`, `roi_seed_std=0.0360`
- `results/v460/v460_g2train_seed42_20260314_202531.json`: `S1'` は `G3 FAIL`, `pf_median=1.031146`, `roi_seed_std=0.0679`

また multi-slice についても、`slice_metrics` に `early/mid/late` が実際に記録されていた。

- `scripts/v460/lib/tasks/sac_train.py`
- `scripts/v460/run_gate_check.py`

### §1.2 432# の deep dive はスクリプト出力で再現できる

`temp/analyze_432_deep.py` を再実行すると、少なくとも以下は再現できた。

- `skip_gate_score vs 30s PnL: r = -0.0097`
- `buy+ranging`: `PF 0.766`, `PnL -0.397`, `n=1310`
- `sell+trending_up`: `AS 35.5%`, `PnL -0.832`
- `balance_forced`: 30s では悪いが 120s では normal より改善
- `regime mismatch`: 3.0% だが `PnL -2.294`

したがって、432# は「印象論」ではなく、少なくとも集計結果に支えられている。

### §1.3 433# の発想は既存実装に寄せられる

433# の主要アイデアのうち、少なくとも次はゼロからではない。

- toxicity 学習の土台: `scripts/v460/ml/as_classifier.py`
- 板・約定の観測基盤: `scripts/v460/run_observation.py`, `ztb/data/market_data_collector.py`
- Sidecar 注入経路: `scripts/v460/lib/orchestrator_mid_cycle.py`, `scripts/v460/lib/cycle_gate_aggregator.py`, `scripts/v460/lib/sidecar_signal_io.py`
- walk-forward 資産: `ztb/evaluation/walk_forward/splitter.py`

つまり 433# は「妄想」ではなく、**reuse 可能な足場の上にある backlog** と見てよい。

---

## §2 426# レビュー

### §2.1 妥当な点

426# の強い部分は次の3点。

1. `val_ratio=0.02` が楽観寄りであること
2. `reward_clean` が `reward_tuned` より頑健であること
3. `mid` 期崩壊パターンが複数 seed で見えること

特に `426#` の「S1 は marginal fail、S1' は structural fail」という整理は良い。`reward_tuned` は `val_ratio` の被害者というより、現設計のままだと学習信号が濁っている可能性が高い。

### §2.2 補正が必要な点

ただし、426# の結論にはまだ踏み込みすぎの箇所がある。

1. **`optimal val_ratio = 0.05-0.10` は未実証**  
   `426#` は `0.02` と `0.20` の2点比較に過ぎない。間のレンジは合理的仮説ではあるが、まだ実験結果ではない。

2. **`100K でも失敗したので step 増加は解ではない` は強すぎる**  
   423#-425# でも整理した通り、100K 側には `val_ratio` や `best_model/final_model` の交絡がある。`step増加が万能でない` までは言えるが、`step増加は無意味` までは言えない。

3. **`1D continuous_action の表現力不足` は plausible だが未証明**  
   これは理論的にはあり得るが、現在の evidence で直接切れているのは `regime generalization` と `reward design` であり、action-space が主犯かはまだ分離できていない。

### §2.3 設計面からの補強

426# から本当に導くべき設計判断は、「適切な val_ratio を当てること」よりも、**rolling / walk-forward / periodic retrain を標準系にすること** である。

既に資産はある。

- `ztb/evaluation/walk_forward/splitter.py`
- `ztb/evaluation/unified_evaluation.py`
- `scripts/v460/ml/walk_forward_as.py`

profit-first で言えば、`G3 single split` をいじり続けるより、**時系列窓ごとの崩れ方を標準出力にする方が次の負け方を減らす**。

### §2.4 市場理論からの補強

426# の「mid 期崩壊」は、単に学習量の不足というより、**市場参加者構成と microstructure の時間変化を policy が吸収できていない** と読む方が自然である。

- maker に有利な局面と不利な局面は、価格方向よりも `toxicity / flow composition / queue risk` で変わる
- SAC が学んでいる reward は、その変化を十分に観測していない可能性がある
- だから `reward_clean` のような単純系が一時的に勝っても、中期では脆い

ここは 432# と 433# を繋ぐ論点である。

---

## §3 432# レビュー

### §3.1 強い発見

432# の中で、かなり信頼してよいのは次。

1. **Skip Gate は現状ほぼ死んでいる**  
   `r=-0.0097` は「弱い」ではなく、運用判断の主軸に置けない水準。

2. **`buy+ranging` が損失の主エンジン**  
   `PF 0.766` かつボリューム最大なので、ここを放置すると他の改善が相殺されやすい。

3. **`sell+trending_up` は教科書的 adverse selection**  
   上昇フローに対する sell maker は、良い逆張りでなければ単なる逆選択になる。

4. **reprice は基本的に悪い側へ寄っている**  
   追いかけるほど不利になるのは maker の典型的負け方。

5. **balance_forced は短期悪化・中期改善の二面性がある**  
   これは「方向性は合っているが execution quality が悪い」という重要な示唆。

### §3.2 補正が必要な点

432# は有益だが、少なくとも2点は修正が要る。

1. **P0 の現状認識が stale config**  
   432# は `buy ceiling = 0.150, sell ceiling = 0.150` と書いているが、現行は以下。

   - `configs/v460/fill_test.yaml:580` `offset_ceiling_ratio = 0.15`
   - `configs/v460/fill_test.yaml:581` `offset_ceiling_ratio_buy = 0.20`
   - `configs/v460/fill_test.yaml:582` `offset_ceiling_ratio_sell = 0.50`

   したがって「現在も一律 0.15 で ceiling」という読みは事実とずれる。もし 432# の主張が「分析対象期間ではそうだった」なら、**その SHA / date 範囲を本文で明示** した方が良い。

2. **`offset_stages.ceiling` の解釈は注意が要る**  
   431# の self-review で、`offset_stages["ceiling"]` は発火時しか記録されず、sell 側で取りこぼしがあると整理されている。

   - `docs/v460/431_ph2_impl_clamp_observability_and_data_analysis.md:129`
   - `docs/v460/431_ph2_impl_clamp_observability_and_data_analysis.md:137`

   よって、432# の「AS でも非ASでも ceiling は同じ」という結論は、方向性としてはわかるが、**完全に確定的な事実としてはまだ弱い**。

### §3.3 因果の言い切りは少し弱めるべき

432# の §9 は筋が良いが、現状は **observational chain** である。

- `Skip Gate 崩壊` はかなり強い
- `AS 危険時に必要 offset が大きい` もかなり強い
- しかし `ceiling が主犯` は、現行 config と clamp observability の再測定で詰め直したい

より正確には、

> `悪い取引を止める系` と `offset で逃がす系` の両方が十分に働いておらず、その交点で `buy+ranging` と `sell+trending_up` が燃えている

と書く方が堅い。

### §3.4 市場理論からの補足

432# の live 所見は、市場理論的にもかなり筋が通っている。

- `buy+ranging` 悪化: mean reversion を取りにいっているのではなく、**静かな時間に受け身の liquidity を安売りしている** 可能性
- `sell+trending_up` 悪化: 上昇フローに against して有毒フローを受けている可能性
- `reprice` 悪化: 受動 maker から能動 taker まがいに近づくほど、情報劣位で不利になりやすい
- `regime mismatch` 悪化: 注文から fill までの短い間にも order-flow state が変わることを示す

つまり、今のボトルネックは「もっと賢く方向予測する」以前に、**毒のあるフローに参加しないこと** である。

---

## §4 433# レビュー

### §4.1 一番価値が高いのは Toxicity sidecar

433# の4案の中で、最も近くて実利的なのは `toxicity` である。

理由は単純で、既に reuse できるものが多い。

- `scripts/v460/ml/as_classifier.py` に教師あり分類の足場がある
- `fill_records` を正例/負例に使える
- `Sidecar` の注入経路が既にある
- 432# の結果が「AS回避が支配的」と示している

これは 426# の「SAC 単体は long OOS で脆い」とも整合する。SAC に全部やらせるより、**毒を veto する軽量モデルを別系統で載せる方が構造に合う**。

### §4.2 BitFlyer 案は「arb」ではなく lead-lag feature として扱うべき

433# の BitFlyer 案は面白いが、言い方は少し補正した方がいい。

これは厳密な意味での arbitrage ではなく、**cross-venue lead-lag / stale quote exploitation のヒント** である。特に public API ベースでは以下のリスクがある。

- REST / public feed の遅延
- venue 間 clock skew
- Coincheck / BitFlyer の出来高構成差
- 非同期取得による stale fusion

したがって初手は、

- hard directional flip
- aggressive override

ではなく、

- veto
- offset boost / retreat hint
- participation suppressor

の方が安全である。

### §4.3 Queue 管理は「近似ヒューリスティクス」として扱うべき

433# の queue 管理も方向性は良いが、public L2 だけで **真の queue position** を知ることはできない。よって正確には

- `queue position estimation`

というより、

- `front volume depletion heuristic`

である。

ただし、それでも価値はある。特に「自分の前の厚みが急に消えたら retreat」系は、maker の被弾回避としては実用的。

### §4.4 文書衛生の問題は先に直すべき

433# は本文自体は読めるが、一部に制御文字が混ざっており、そのまま backlog source にすると危ない。

- `docs/v460/433_ph4_advanced_microstructure_edge_ideas.md:22`
- `docs/v460/433_ph4_advanced_microstructure_edge_ideas.md:39`
- `docs/v460/433_ph4_advanced_microstructure_edge_ideas.md:41`

例えば `trade_flow`, `bf_cc_spread_bps`, `aggressiveness_hint` の表記が壊れている。これは内容の是非とは別に、**設計メモとしての保守性が低い**。

### §4.5 Committee は hard AND より weighted veto が良い

433# の「委員会」は発想として良いが、初手から

- A, B, C 全員 OK の時だけ参加

にすると、今度は参加率が崩れて別の dead system を作る危険がある。

推奨は以下。

1. `toxicity` だけは veto 権を持つ
2. `BitFlyer lead-lag` は boost / suppress の補助票
3. `queue heuristic` は retreat / cancel の補助票
4. `SAC` は方向ヒントに限定

つまり **hard AND committee ではなく、weighted score + veto** の方が現構造に合う。

---

## §5 3文書を通した共通盲点

### §5.1 sim-live gap をまだ閉じ切れていない

426# は simulator OOS を見ている。432# は live fill を見ている。433# は将来 alpha を見ている。

だが、その間の

- `モデル出力`
- `clamp / gate / route-to-kill`
- `実約定品質`
- `その結果を次学習へ戻す`

という閉ループはまだ弱い。

ここを閉じないと、426# で良さそうなものが 432# で死に、433# の新案も同じ穴に落ちる。

### §5.2 単特徴量思考より interaction 設計が必要

432# が示す通り、単変量相関はほぼ全部弱い。なので、

- 単一閾値を増やす
- 単一 feature だけで hard skip する

より、

- `side × regime × toxicity × clamp_fired`
- `decision_path × balance_forced × queue_wait`
- `lead-lag × spread × velocity`

のような interaction を見る設計が必要である。

### §5.3 収益の本丸は directional alpha ではなく toxic participation 回避

今の証拠を総合すると、勝敗を大きく動かしているのは

- 未来の価格方向を当てる能力

より、

- 不利な相手に板を差し出さない能力

である。

この意味で、433# の toxicity 案は 426# の SAC 改良案より優先度が高い。

---

## §6 Profit-First 優先順位

### P0

1. `433#` を清書し、制御文字混入を除去して backlog として再固定する
2. `432#` の `ceiling` 論点を current-SHA で再集計し、`pre_clamp / post_clamp / clamp_fired / resolved_ceiling` を明示する
3. `426#` は `val_ratio 最適化` より `walk-forward / rolling retrain` に議論を寄せる

### P1

1. `scripts/v460/ml/as_classifier.py` を起点に、`fill_records` から toxicity veto モデルを作る
2. 432# 再集計では `decision_path`, `balance_forced`, `requested_side/resolved_side_reason`, `clamp_fired` で分解する
3. `Committee` は hard AND ではなく `toxicity veto + weighted hints` で最小実装する

### P2

1. BitFlyer は `lead-lag suppressor/boost` として小さく導入する
2. queue 管理は `front-volume depletion` の retreat ルールとして試す
3. SAC は「万能 driver」ではなく、sidecar の一票として責務限定を維持する

---

## §7 最終結論

426#・432#・433# はバラバラに見えるが、実はかなり一つの方向を指している。

- `426#`: SAC の単独 OOS 優位は長期には脆い
- `432#`: live では toxicity / clamp / path 交絡が支配的
- `433#`: だから次に足すべきは end-to-end policy 強化より、毒回避・lead-lag・queue 監視の sidecar 群

したがって、次の一手は

> **SAC をさらに重くすることではなく、live fill で裏付けられた toxic-participation 回避を既存資産で sidecar 化すること**

である。

これが今の3文書を一番無駄なく繋ぐ、profit-first の解釈だと考える。
