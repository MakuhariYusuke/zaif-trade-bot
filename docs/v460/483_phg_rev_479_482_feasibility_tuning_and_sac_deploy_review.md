# 483# 479-482レビュー: feasibility緩和は妥当、SAC投入は探索段階に留まる

> 種別: rev
> 対象: 479#, 480#, 481#, 482# と関連実装
> 日付: 2026-03-19

---

## 0. 結論

今回の一連の対応は二つに分けて評価すべきである。

1. 480# と 481# の feasibility / veto / observability 改善は、現在の fill 低下要因に対する実務的な P0 対応として概ね妥当である。
2. 482# の SAC 100K seed456 デプロイは、探索的投入としては理解できるが、まだ「有意に優れたモデルを見つけた」とまでは言えない。

特に重要なのは、481# の設定変更と 482# の sidecar 差し替えが同時期に入っている点である。現時点で fill 改善や PnL 変化が観測されても、その寄与を

- feasibility 緩和
- cross_venue veto 緩和
- NFQ 可視化改善
- SAC sidecar

に分離して語ることはできない。

したがって本レビューの立場は明確である。

- 481# は前進である
- 482# は「試す価値はある」が「信頼してよい」とはまだ言えない
- 次の優先課題は、さらなる緩和や再学習より先に、寄与分解可能な観測設計を作ること

---

## 1. 479-481 の評価: 修正の方向は正しい

### 1.1 480# の訂正は核心を突いている

480# が指摘した通り、479# の最大の誤りは `no_feasible_quote` を `spread_too_narrow` と同一視したことである。

実際には最新 cohort の NFQ はかなりの部分が `cross_venue_lead_lag_veto` 起因であり、`min_spread_jpy` だけを緩めても全体像は改善しない。ここを `spread` 単独問題から `spread + veto` の二本立てに再構成した 480# の読みは正しい。

この再構成は設計上も重要である。なぜなら現在の Buy 劣位は、単なる狭スプレッド忌避だけでなく、BitFlyer 先行下落に対する veto が Buy 側の機会集合を継続的に削っているからである。

### 1.2 481# の変更は実装済みで、P0 として妥当

以下はコード / config 上で確認できた。

- `configs/v460/fill_test.yaml` で `veto_threshold_bps: 8.0`
- `configs/v460/fill_test.yaml` で `min_spread_jpy: 700`
- `scripts/v460/lib/fill_cycle_executor.py` で NFQ エスカレーションログに `last_reason=` を追加

この 3 点は全て、480# が特定したボトルネックに対して直接効く。

特に `last_reason=` の追加は過小評価すべきではない。NFQ は表面上 1 種類に見えても、背後の infeasible reason が異なると対策が真逆になるため、ここが曖昧なままだと再び 479# 型の誤読を繰り返す。

### 1.3 ただし、次の spread 緩和はまだ早い

481# は Phase 1 として妥当だが、その直後に `min_spread_jpy: 500` へ進むのは時期尚早である。

理由は二つある。

1. まだ 700 への緩和後 cohort の品質評価がない
2. 480# 時点で最新 cohort の約定品質は既に弱く、逆選択率も無視できなかった

つまり、現在の優先課題は「さらに通す量を増やす」ことではなく、「700/8bps で増えた約定候補が本当に良い fill なのか」を 먼저測ることである。

---

## 2. 482 の評価: result JSON は実在するが、証拠強度は弱い

482# の記載内容は、少なくとも実験ファイルと整合している。

- `configs/v460/experiments/g2_sac_reward_clean_100k_vr02.yaml`
  - `val_ratio: 0.02`
  - `checkpoint_interval: 10000`
  - `evaluation.n_episodes: 1`
  - `seeds: [42, 123, 456, 789]`
- `results/v460/v460_g2train_seed42_20260318_114608.json`
  - 4 seed 分の checkpoint / eval / best-model 評価を保持
  - seed456 best checkpoint は 90K

したがって、482# は根拠のない創作ではない。ただし、問題は「ファイルがあるか」ではなく「そこからどこまで強く言えるか」である。

### 2.1 val_ratio=0.02 と n_episodes=1 は、best checkpoint 選定用としては使えても、採用判断には弱い

`results/v460/v460_g2train_seed42_20260318_114608.json` では全 seed の `train_val_split.val_ratio` が 0.02、`best_model_eval_metrics.n_episodes` が 1 である。これは 482# 自身も認めている通り、正式な G3 合格判定や採用判定の土台としては弱い。

この条件で強く疑うべきなのは、seed456 が強いのではなく「末尾 2% の regime にたまたま最も合った checkpoint が選ばれた」可能性である。

### 2.2 OOS > Train が全 seed で強く出ており、validation slice の偏りを疑うべき

結果 JSON では、best checkpoint において全 seed で OOS ROI が in-sample ROI を大幅に上回る。

- seed123: in-sample 0.24% に対し OOS 0.70%
- seed456: in-sample 0.09% に対し OOS 0.54%
- seed789: in-sample 0.09% に対し OOS 0.43%

一般に、これをそのまま「汎化が良い」と解釈するのは危険である。むしろ validation period の有利な regime、または単一 episode 評価の分散を疑うべきである。

### 2.3 seed456 は成績トップ候補だが、signal quality が弱い

seed456 の best-model 指標自体は悪くない。

- best-model gross ROI: 0.597%
- PF: 1.185
- Sharpe: 6.21
- MaxDD: 0.25%

しかし同じ JSON で、reward-profit 相関が非常に弱い。

- seed456 final `reward_profit_corr = -0.0115`
- seed456 best-model `reward_profit_corr = 0.1876`

他 seed と比べても明らかに見劣りする。これは「利益が出た checkpoint」は見つかっていても、「その policy の出力が profit と安定的に整合している」とは言いにくいことを意味する。

さらに slice を見ると、seed456 best-model は

- early: PF 1.565 だが reward-profit 相関 -0.171
- mid: PF 0.883 で失速
- late: PF 1.187 かつ相関 0.975

という非一様な構造を持つ。これは「policy quality が全域で高い」というより、「特定 slice で強く、他では不安定」に近い。

### 2.4 seed456 を即本命視するより、seed123 を対照群として残すべき

seed123 は top ROI ではないが、checkpoint OOS が 10/10 で正、best-model の reward-profit 相関も 0.690 と健全である。つまり、456 は高収益候補、123 はより整合的な基準候補という見方が自然である。

この 2 本を比較せずに 456 単独で「採用モデル」と言い切るのは、探索空間を狭めすぎている。

---

## 3. live sidecar の実効影響はかなり小さい

482# を評価する上で、最も重要なのに過小評価されているのが「モデルが良くても live でどれだけ価格を動かせるか」である。

確認できた sidecar 制約は以下の通り。

- `configs/v460/fill_test.yaml`: `max_boost_bps: 0.15`
- `configs/v460/fill_test.yaml`: `dead_zone: 0.10`
- `scripts/v460/lib/fill_config_validation.py`: `sidecar_max_boost_bps <= 0.20` hard ceiling
- `scripts/v460/lib/sidecar_types.py`: `compute_sidecar_offset_bps_v2` は dead zone 除去後に `max_boost_bps * shaped * confidence`
- `scripts/v460/lib/sidecar_types.py`: signal TTL は 7800 秒
- `scripts/v460/lib/sidecar_signal_io.py`: TTL 超過時は `None` を返し neutral 化
- `scripts/v460/ml/sac_retrain_scheduler.py`: retrain interval は 2h-4h

この構造から分かることは明快である。

1. sidecar はリアルタイム板反応型ではない
2. stale になると neutral に退化する
3. 生きている間も最大寄与は 0.15bps に制限される
4. `|bias| <= 0.10` は完全に無効化される

したがって、481# 直後の fill 改善がもし観測されたとしても、それを SAC の勝利と即断してはいけない。現実には `min_spread_jpy 1000→700` と `veto 6→8bps` の方が寄与ははるかに大きい可能性が高い。

別の言い方をすると、今の 482# は「強いモデルを入れた」というより、「効き幅の小さい advisory signal を tentative に接続した」に近い。

---

## 4. 改善の優先順位

### P0. まず live 効果の寄与分解を可能にする

次の 24h-48h 分析では、少なくとも以下を cohort 固定で持つべきである。

- `run_id`
- `start_git_sha`
- config snapshot または主要パラメータ列
- sidecar signal の stale / fresh 状態
- `sidecar_offset_bps` の分布
- `last_reason` を含む NFQ 内訳

これがないままでは、481 と 482 のどちらが効いたのか、あるいは両方効いていないのかすら判断できない。

### P1. SAC の採用判定は val_ratio>=0.10 か walk-forward でやり直す

seed456 を否定する必要はないが、採用判断はやり直すべきである。

最低でも以下が必要である。

- `val_ratio >= 0.10`
- `n_episodes > 1`
- 同一 reward で seed123 / 456 / 789 を再比較

可能なら walk-forward で regime を跨いだ再評価にしたい。今の 2% holdout は best checkpoint 選定には使えても、採用判定としては脆い。

### P1. next spread relaxation より先に約定品質を点検する

`min_spread_jpy: 700` の次に 500 へ進む前に、以下を観測すべきである。

- filled の PnL30s / PnL60s
- AS 率
- side 別 fill 率
- spread 帯別 fill quality

もし fill 数だけ増えて quality が悪化しているなら、今必要なのはさらなる緩和ではなく quality-conditioned quoting である。

### P2. 123 と 456 の二系統比較を残す

現時点の seed 解釈は次が妥当である。

- 456: 高収益候補だが validation 依存が強く、reward-profit 整合性が弱い
- 123: 収益は控えめだが OOS の符号安定性と相関が比較的健全

よって次にやるべきは 456 固定ではなく、neutral / 123 / 456 の live 比較、または少なくとも offline replay 比較である。

---

## 5. 最終判断

481# までは、かなり筋が良い。ここは素直に進めてよい。

一方 482# は、実験そのものは有意義だが、結論の強さが evidence を少し上回っている。seed456 の投入は「探索的 canary」としてなら妥当であり、現時点で否定する必要はない。しかし、それをもって SAC sidecar の有効性や seed456 の優位性が確認できたとはまだ言えない。

短期収益の観点でも、次の勝ち筋は「さらに緩める」ことや「さらに学習する」ことより、まず 481 と 482 の効果を分離して測ることである。ここを曖昧にしたまま前に進むと、良い変更を捨て、効いていない変更を過大評価する危険が高い。