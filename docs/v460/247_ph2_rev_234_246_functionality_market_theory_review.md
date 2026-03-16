# 247# 234#–246# レビュー — 機能性・市場理論・収益性の再点検

> **日付**: 2026-03-03  
> **対象**: `234#`〜`246#`, `prompts/246_codex_gemini_review_prompt.md`, 現行コード, `results/v460/fill_test/` 実ログ  
> **目的**: 234#〜246# の実装が本当に根本解決へ近づいているかを、機能・市場理論・運用実態の3面で検証する

---

## 0. 総括

結論は明確です。

1. **234#〜241# は「危険な bypass を減らし、分類と防御を細かくした」という意味で前進です。**
2. **ただし、242#〜246# を含む新設計は、現時点では本番 run に未投入です。**
3. **したがって 245# の実績を使って 246# の有効性を論じるのは、因果としてまだ弱いです。**
4. **本質的な未解決点は「在庫中立を前提にしすぎていること」と「評価指標が directional carry を見ていないこと」です。**
5. **最も危険な残課題は、246# の DD cooldown release が hard halt を“再武装しない一回限りの停止”へ変えている点です。**

全体評価としては、

> **構造改善は進んだが、運用・計測・在庫哲学の3つがまだ揃っていない**

です。

---

## 1. 主要 findings

### 1.1 [HIGH] 242#〜246# はまだ live で検証されていない

現行のディスク上コードは最新ですが、稼働中プロセスはそれを使っていません。

### 実ログの事実

- `2026-03-03 16:30:22` の起動ログでは `git_sha=c141be4a2947`
- これは `241#` の SHA で、`242#`〜`246#` の後ではありません
- 現在の run `1772523008_b7f1ace2` の `fill_records_20260303.jsonl` は
  - `2 records`
  - `0 fills`
  - `cancel_reason = daily_drawdown_halt` のみ

つまり、**242# の quiescence も、246# の cooldown release も、本番挙動としてはまだ見えていません。**

### 含意

- 234#〜241# の設計評価はできる
- 242#〜246# の運用効果は、現時点ではまだ仮説段階

246# の「2時間後に再開するはず」という主張に対し、  
実ログでは `18:10 JST` 時点でもなお `daily_drawdown_halt` のみです。  
これは**効果がなかった**のではなく、**そのコードがまだ動いていない**という意味です。

---

### 1.2 [HIGH] 245# の18日集計は、241# 単体の実績として読むには因果が混ざりすぎている

245# の分析自体は有用です。  
ただし、**241# の性能評価として読むのは危険**です。

### 実ログの事実

`results_analyzer` の `2026-03-03 16:27` 出力では、18日集計の中に多数の `run_id` が混在しています。  
同じログ中で、`1772436274_49bdd24a`, `1772429659_653404f3`, `1772425929_9e37aef3` など、複数 run の分解が出ています。

さらに、`schema_health` ログ上でも 2/28〜3/3 の間に複数の `git_sha` が入れ替わっています。

### 含意

245# の

- `累積 -792 JPY`
- `sell -1.316bps`
- `trending_up -0.919bps`

といった数字は、**戦略全体の傾向を見る材料**としては良いです。  
しかし、**241# の設計だけが悪い／246# で直る**という因果の根拠にはなりません。

### 推奨対応

246# の評価は必ず

1. `run_id` 固定
2. `git_sha` 固定
3. `date_from / date_to` 固定

で切るべきです。  
今の 245# は「全体傾向の診断」として使い、**パラメータ正当化の直接証拠には使いすぎない**方が良いです。

---

### 1.3 [HIGH] 242# は「No Trade = 正常」を完全には実装していない

242# の方向性自体は正しいです。  
ただし、コードはまだ **probe を廃止していません**。

### コード上の事実

- `ztb/risk/sell_dynamic_kill.py:392`
  - 依然として stale probe 判定が残る
- `ztb/risk/sell_dynamic_kill.py:395`
  - `toxic_kill_stale_multiplier` を掛けて probe 間隔を延長するだけ
- `ztb/risk/sell_dynamic_kill.py:398`
  - 閾値超過で probe 実行
- `ztb/risk/sell_dynamic_kill.py:405`
  - さらに force release も残る

### 含意

242# は実態として

> **probe 廃止**

ではなく、

> **probe の延期**

です。

この違いは重要です。  
「No Trade を正常化した」と見せつつ、最悪時にはまだ**時間が来たら穴を開ける**構造が残っています。

---

### 1.4 [HIGH] `dual_kill_bypass` が残っており、242# の思想と衝突している

これは 242# の根本思想と、現在のコードの間にある最も大きい整合性ギャップです。

### コード上の事実

- `scripts/v460/lib/cycle_gate_aggregator.py:231`
  - `is_buy_killed and is_sell_killed` で `dual_kill` 判定
- `scripts/v460/lib/cycle_gate_aggregator.py:233`
  - そのまま `dual_kill_bypass = True`
- `scripts/v460/lib/cycle_gate_aggregator.py:239`
  - `allowing {side} through to break deadlock`

### 問題

242# は「No Trade = 正常」を認める方針です。  
それなのに、**両側が kill されている最悪の状態で、なお 1 つ通す**ロジックが残っています。

これは実務的には

> **“市場が毒だ” と両モデルが言っているのに、とにかく何かを出す**

という意味です。

### 推奨対応

`dual_kill_bypass` は、少なくとも `toxicity_budget_enabled=True` のときは原則停止し、

- `quiescence`
- `operator override`
- `超長時間 no-fill でも directional alpha が未成立`

などの明示条件付きに格下げすべきです。

---

### 1.5 [CRITICAL] 246# の DD cooldown release は、hard halt を「2時間の一回停止」に変えてしまっている

ここは今回の最重要の設計リスクです。

### コード上の事実

- `scripts/v460/lib/daily_drawdown_guard.py:219`
  - `is_halted()` が halt 判定を返す
- `scripts/v460/lib/daily_drawdown_guard.py:237`
  - 一定時間経過で `cooldown_released = True`
- `scripts/v460/lib/daily_drawdown_guard.py:244`
  - `cooldown_released` なら `False` を返す
- `scripts/v460/lib/daily_drawdown_guard.py:254`
  - 以後は lot 縮小だけを返す

しかし、このとき

- `_state.halted` は `True` のまま
- `halt_triggered_at` もそのまま
- 再度 hard halt を張り直す再武装ロジックがありません

### 実質の挙動

246# の hard halt は、

> **「その日の hard stop」**

ではなく、

> **「その日の最初の2時間だけ止まる」**

に変わっています。

これは **機会損失の削減** には効きますが、同時に **二段目の絶対停止線を消す** ことになります。

### 推奨対応

246# をそのまま入れるなら、最低でも次のどちらかが必要です。

1. `cooldown_release` 後の再悪化で `halt_triggered_at` を再設定し、二度目の halt を許可
2. `cooldown_release` 後は別の tighter hard limit（例: 追加 -10bps）を持つ

現状のままだと、**“再開” ではなく “片方向の解除”** です。

---

### 1.6 [HIGH] 収益評価が「実現スプレッド損益」と「保有ポジション価値」を分離できていない

これは市場理論上の根本問題です。  
245# が気づき始めている論点ですが、実装側はまだ追いついていません。

### 実ログの事実

- `2026-03-03 16:30:22` の起動ログで `loss_cap` 用総資産は `22611 JPY`
- 同時に `cumulative_pnl_jpy = -792.30`
- 残高は `JPY 1100.61 + BTC 0.002`

つまり、運用上は **総資産（mark-to-market）** を見ている一方で、  
戦略評価は依然として **約定ベースの短期 realized PnL** に強く寄っています。

### 問題

上昇トレンド中に BTC を持つこと自体が alpha なら、

- `JPY が減る`
- `buy が増える`
- `sell が減る`

ことは必ずしも失敗ではありません。

それなのに評価系が

- `post_fill_30s_pnl`
- `cumulative_pnl_jpy`

中心だと、**保有中の directional gain を“存在しないもの”として扱う**ため、

> 正しく積んだ BTC ポジションを、在庫異常として解消しにいく

方向へ設計が歪みます。

### 推奨対応

P/L を最低でも 3 つに分けるべきです。

1. `spread_capture_pnl`
2. `inventory_carry_mtm`
3. `toxicity / adverse_selection_loss`

この分離なしに 246# の sell 防御だけ強めると、**何を守れて何を捨てているか**が見えません。

---

### 1.7 [MEDIUM] 239# は「本当の feasible quote 計算」までは到達していない

239# は悪くありません。  
ただし、**232# で要求された“真の feasible set”の一部だけ**です。

### コード上の事実

- `scripts/v460/lib/maker_price.py:886`
  - `spread < min_spread_jpy` を早期 reject
- `scripts/v460/lib/maker_price.py:894`
  - `sell_max_spread_jpy` を早期 reject

これは「spread 制約の前倒し」であって、  
まだ以下を含んでいません。

- post-only 非交差制約
- trend / toxicity / degraded の offset 後価格
- sell offset floor
- 最終的な admissible price interval

### 結論

239# は

> **“feasible quote proactive” の第1段階**

であり、

> **“真の制約交点計算”**

ではありません。

名称ほど完了していない点は、明確に認識しておくべきです。

---

### 1.8 [MEDIUM] PhantomPositionGuard は「不確定時に保持する」より「一回で捨てる」寄りになっている

237#/238# の方向性は正しいです。  
しかし、今の実装は **曖昧なまま pending を消しやすい** です。

### コード上の事実

- `scripts/v460/lib/phantom_position_guard.py:298`
  - order recheck 失敗は warning のみ
- `scripts/v460/lib/phantom_position_guard.py:332`
  - balance check 失敗も warning のみ
- `scripts/v460/lib/phantom_position_guard.py:246`
  - その後、全 pending を `clear()`

### 問題

取引所 API が不安定なときこそ phantom は起きやすいのに、  
その **“最も怪しい時” に一回の再照合失敗で quarantine を手放す** ことになります。

### 推奨対応

`_reconcile_single()` が

- `definitely_clean`
- `definitely_phantom`
- `inconclusive`

の三値を返すようにし、`inconclusive` は pending 維持にすべきです。

---

### 1.9 [MEDIUM] PhantomPositionGuard の buy 側残高照合は、まだ配線が半端

これは実装と文書のズレです。

### コード上の事実

- `scripts/v460/lib/phantom_position_guard.py:63`
  - `balance_snapshot_jpy` フィールドはある
- しかし使用されていません
- `scripts/v460/lib/balance_checker.py:44`
  - `_last_jpy_free` は保持している
- しかし公開プロパティがありません
- `scripts/v460/lib/fill_cycle_executor.py:173`
  - `_maybe_register_phantom()` は `last_btc_free` だけ渡している

### 含意

buy 側の Phase 2 は、実質的に

- stale な BTC snapshot
- もしくは `None`

に頼りやすく、**buy phantom の取りこぼし** が起きやすいです。

### 推奨対応

1. `BalanceChecker.last_jpy_free` を公開
2. buy 時は `JPY decrease` も照合
3. side 別に `BTC-snapshot / JPY-snapshot` を使い分ける

---

### 1.10 [MEDIUM] one-sided freeze / cooldown が「どの side を凍結したか」を持っていない

234#/235# のエスカレーション自体は良いですが、実装は side-bound ではありません。

### コード上の事実

- `scripts/v460/lib/fill_loop_orchestrator.py:78`
  - 保存しているのは残数カウンタだけ
- `scripts/v460/lib/fill_loop_orchestrator.py:1458`
  - 現在の `next_side` を skip
- どの side が原因で freeze/cooldown に入ったかは保持していません

### 問題

ドキュメントは「当該 side を凍結」と書いています。  
しかし実装は、**その cycle で選ばれた side を止めるだけ**です。

したがって、在庫やレジームが変わった後に

- 本来止めたかった side ではなく
- 既に健全化した反対 side

を止める可能性があります。

### 推奨対応

`_one_sided_frozen_side: str | None` を持ち、  
凍結・cooldown は side に紐付けて扱うべきです。

---

### 1.11 [LOW] 新設パラメータの validation がまだ甘い

`fill_config.py` には多くの validation がありますが、  
今回増えた重要パラメータの一部が未検証です。

特に以下は値域検証がありません。

- `degraded_liquidation_lot_mult`
- `degraded_liquidation_offset_mult`
- `degraded_liquidation_duty_cycle`
- `dd_cooldown_release_lot_scale`

負値や極端値が入ると、**縮退清算や halt 解除が逆方向**に壊れます。  
ここは小さな追加で事故率をかなり下げられます。

---

### 1.12 [LOW] 複雑性は再び上がっており、God Object 回帰の兆候がある

現在の行数は以下です。

- `scripts/v460/lib/fill_loop_orchestrator.py`: `2356`
- `scripts/v460/lib/fill_cycle_executor.py`: `1369`
- `scripts/v460/lib/fill_config.py`: `1569`

234#〜246# は個々の防御としては合理的です。  
ただし、**守りの追加が orchestrator / config に再集中**しており、

> 「バグを潰すたびに制御塔が太る」

構造に戻りつつあります。

これはすぐの破綻ではありませんが、次の大型改修でまた同じ複雑性問題が出ます。

---

## 2. 市場理論から見た方向修正

### 2.1 在庫目標を「ゼロ」ではなく「レジーム依存の目標帯」に変えるべき

ここが最重要です。

Avellaneda-Stoikov の在庫中立は、短期 MM と mean-reverting 前提では有効です。  
しかし、245# が示すように `trending_up` で sell が大きく悪化しているなら、

> **在庫をゼロへ戻すこと自体が逆張り**

になります。

したがって次段階は、

- `trending_up`: BTC 目標在庫を正に持つ
- `trending_down`: BTC 目標在庫を縮める
- `ranging`: 中立に戻す

という **target inventory band** へ移るべきです。

---

### 2.2 Sell defence hardening は“延命”であって“根治”ではない

246# の

- `sell_offset_floor ↑`
- `sell_dynamic_kill threshold ↑`
- `trending_sell_offset_boost ↑`
- `toxic_veto threshold ↑`

は、方向としては理解できます。  
ただし、これは

> **「売るけど、もっと守って売る」**

であって、

> **「そもそも今は売るべきか」**

には踏み込んでいません。

`trending_up` の directional alpha を認めるなら、本筋は

- sell を harder にする

よりも、

- **sell を inventory target 超過時に限定する**
- **reversal evidence が出るまで BTC を持つ**

です。

---

### 2.3 Spread alpha と carry alpha を会計上で分離しない限り、最適化方向を誤る

今のままでは、

- MM のスプレッド収益
- BTC 保有の含み損益
- 逆選択コスト

が混ざったまま議論されがちです。

これでは

- `sell` が悪いのか
- `forced neutralization` が悪いのか
- `inventory hold` が実は正しいのか

が判定できません。

まずは ledger を分けるべきです。  
これは収益改善というより、**収益改善を正しく評価するための前提**です。

---

## 3. 次にやるべき P0

### P0-1

**242#〜246# を含む SHA で、新しい `run_id` を切って再実測する。**

今の本番 run は `241#` 止まりです。  
246# のレビューは、まずデプロイしないと始まりません。

### P0-2

**246# の cooldown release に “再武装” を入れる。**

少なくとも

- second hard stop
- re-arm threshold
- post-release extra loss budget

のどれかは必要です。

### P0-3

**`dual_kill_bypass` を quiescence 方針と整合させる。**

「両側 toxic なら休む」を原則に戻し、  
どうしても通す条件を限定すべきです。

### P0-4

**PhantomPositionGuard を三値化し、`inconclusive` を保留にする。**

一回の API 失敗で pending を捨てるのは危険です。

### P0-5

**inventory target band を導入し、ゼロ在庫前提をやめる。**

ここを変えない限り、sell 防御をどれだけ積んでも  
「上昇局面で利益側ポジションを壊す」問題は残ります。

---

## 4. 最終判断

234#〜246# は、前回レビューで指摘した

- bypass
- liveness 強迫
- phantom
- feasibility

の各論にかなり真面目に対応しています。  
その点は評価できます。

ただし、現時点での本質は次の 3 つです。

1. **まだ新設計が live に入っていない**
2. **入ったとしても、246# の halt 解除設計には再武装不足がある**
3. **最も大きい収益機会は、sell 防御強化より “在庫中立前提の修正” にある**

したがって、次の一手は

> **246# の微調整を続けることではなく、在庫目標・P/L 会計・halt 再武装を一つの設計としてまとめ直すこと**

です。ここをやれば、ようやく「守りのパッチ群」から「儲けるための設計」へ移れます。
