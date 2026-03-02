# 232# 222#–231# レビュー — 本筋投入前の先回り点検、FFD評価、残存リスク

> **日付**: 2026-03-03  
> **対象**: `222#`〜`231#`, `prompts/230_231_review_prompt.md`, 現行コード, `results/v460/fill_test/` 実ログ  
> **目的**: 何度も投入して失敗する流れを断ち、mainline へ戻す前に「まだ踏む穴」を先回りで潰す

---

## 0. 総括

結論は 5 点です。

1. **230#/231# の FFD 強化そのものは、静的コードレビュー上は概ね妥当です。**
2. **ただし、現在回っている実運用 run はまだ旧コードであり、230#/231# は live で未検証です。**
3. **現時点の主障害は FFD ではなく、`balance_forced` 起点の片側強制ループと、そもそも発注可能価格帯が空になる問題です。**
4. **古典的な lock/mutex デッドロックは見当たりません。残っているのは logical deadlock / liveness trap です。**
5. **本筋投入前の優先順位は、FFD 微修正よりも「在庫都合で危険側を通常運転してしまう構造」の是正です。**

言い換えると、現状は

> **防御ロジックは増えたが、最終的な執行モードの設計がまだ粗い**

という評価が最も正確です。

---

## 1. 主要 findings

### 1.1 [CRITICAL] 230#/231# はまだ live 検証されていない

これは今回の最重要事実です。  
現行コードは `git rev-parse` 上 `ebd2592e1057` ですが、現在の fill test run はそれを使っていません。

### 実ログの事実

- `2026-03-02 16:24:45`: lock 取得
- `2026-03-02 16:24:47`: `run_id=1772436274_49bdd24a, git_sha=2243c90f44cf`
- 以後の `fill_records_20260302.jsonl` でも、この run の `git_sha` は全件 `2243c90f44cf`

つまり、**今見えている live の挙動は 230#/231# 後のものではありません。**

### 含意

- 230#/231# の「コードとしての妥当性」は見られる
- しかし「実運用で効いたか」は、現ログではまだ裏付けられない

この状態で「230#/231# まで入れたから安全」と判断すると、**未デプロイの防御を前提に判断する運用事故**になります。

### 推奨対応

本筋投入前に最低限これを固定すべきです。

1. `run_id` ごとに `git_sha` を固定して評価する
2. 230#/231# を含む SHA で新 run を切り直す
3. レビューは「旧SHAの実ログ評価」と「新SHAの静的レビュー」を混ぜない

---

### 1.2 [HIGH] `balance_forced` が依然として「危険側を通常運転で通す」設計になっている

`223#` で `per-side halt` 破りは潰れています。  
しかし、その次の問題として **dynamic kill / trending sell 抑制を在庫都合でほぼ無効化する** 構造が残っています。

### コード上の根拠

- `scripts/v460/lib/cycle_gate_aggregator.py:351`
  - `balance_forced` 中の `trending_sell` は block せず offset 保護のみ
- `scripts/v460/lib/cycle_gate_aggregator.py:447`
  - `buy_dynamic_kill` は `not balance_forced` のときしか block しない
- `scripts/v460/lib/cycle_gate_aggregator.py:476`
  - `sell_dynamic_kill` も `not balance_forced` のときしか block しない

### 実ログの事実

`2026-03-02 23:53`〜`2026-03-03 00:57` の現行 run で、以下が繰り返し発生しています。

- `buy insufficient, switching to sell immediately`
- `balance_forced but one_sided_balance — proceeding with sell`
- 同じ cycle で `sell kill: regime=trending_up, threshold_used=-0.5 ...`

つまり、**「sell は危険」と判定しているのに、在庫都合で sell を続けている**状態です。

### 問題の本質

これは deadlock 回避としては理解できます。  
しかし執行設計としては、

> **停止すべき side を、通常と近い cadence / 通常に近い lot / 通常フローで通している**

ため、`halt bypass` より一段マシなだけで、まだ十分に危険です。

### 推奨対応

`balance_forced` を「execute anyway」ではなく、**degraded liquidation mode** に落とすべきです。

最低でも次の 3 段階に分けるのが良いです。

1. `balance_forced + kill active`: lot を floor 近くまで縮小
2. `balance_forced + kill active`: offset をさらに拡大
3. `balance_forced + kill active`: N 回に 1 回だけ参加 (duty cycle 制限)

在庫都合は理解できますが、**在庫都合が統計的危険判定に勝ってはいけません。**

---

### 1.3 [HIGH] `one_sided_consecutive_limit` は“制限”ではなく、実態としては“遅延”でしかない

`207#` で入れた片側連続実行制限は、現在のコードだと hard breaker ではありません。

### コード上の根拠

- `scripts/v460/lib/fill_loop_orchestrator.py:1555`
  - 連続回数を数える
- `scripts/v460/lib/fill_loop_orchestrator.py:1560`
  - 上限超過時も warning を出すだけ
- 同ブロックでは `interval ×3.0` のログは出るが、**その場で stop / freeze / skip はしていない**

### 実ログの事実

現行 run では以下が連続しています。

- `5/5`
- `6/5`
- `7/5`
- ...
- `14/5`

つまり、上限を超えても**止まらず増え続けています**。

### 推奨対応

`one_sided_consecutive_limit` は次のように段階化すべきです。

1. 上限到達: interval 乗数
2. 上限 + 2: 強制 cool-down
3. 上限 + 4: その side を N サイクル凍結
4. 上限 + 6: `one_sided_liveness_halt` を記録して operator review

今のままだと、これは deadlock 対策ではなく、**じわじわ損するループの遅回し**です。

---

### 1.4 [HIGH] 230#/231# の FFD 強化は正しいが、現行ボトルネックには刺さっていない

`230#` の `l2_deadzone_bps`、`231#` の TTL refresh / streak 修正は、**FFD 単体としては筋が良い**です。  
ただし、今の run で主に止まっている理由は FFD ではありません。

### 現行 run (`run_id=1772436274_49bdd24a`) の集計

- 対象 record: `95`
- filled: `31`
- side: `buy=39`, `sell=54`, `none=2`
- 30s PnL 合計:
  - `buy = +5.826bps`
  - `sell = -10.733bps`

非約定理由の上位は以下です。

- `skip_gate`: `24`
- `stale_adverse_drift`: `11`
- `sell_guard_reject`: `7`
- `spread_too_narrow`: `7`
- `postonly_crossing_skip`: `4`
- `sell_dynamic_kill`: `3`
- `status_unknown_fast`: `2`

一方で `ffd_boost_active` は **2 record** しかありません。

### 含意

FFD を詰めても無駄ではありません。  
ただし、**今の主損失源・主停滞源はそこではない**です。

### 推奨対応

230#/231# は「入れてよい」。  
しかし優先順位は

1. quote feasibility
2. one-sided degraded mode
3. status unknown / phantom fill 対策
4. その後に FFD 微調整

です。

---

### 1.5 [HIGH] 発注可能な価格帯が空になる “feasible set collapse” が未整理

ここは今回の盲点として強く指摘します。  
今の設計は guard が個別に正しくても、**全部合わせると実行可能な価格が存在しない**場面があります。

### 実ログの事実

同じ run の直近ログで、以下が同時多発しています。

- `Spread too narrow: 284 / 475 / 481 / 862 JPY < min 1000`
- `sell_guard: spread 5272 / 5488 / 7120 > max 5000`
- `postonly_crossing_skip`

### コード上の根拠

- `scripts/v460/lib/maker_price.py:868`
  - `min_spread_jpy` 未満なら reject
- `scripts/v460/lib/maker_price.py:916`
  - `sell_max_spread_jpy` 超過なら reject
- さらに `post_only` 非交差制約、offset boost、trend boost、rescue boost が乗る

この結果、

> **「狭すぎてもダメ、広すぎてもダメ、cross してもダメ、offset はさらに広げる」**

となり、**許容価格区間が空集合**になる瞬間があります。

### 推奨対応

`maker_price` の前段に、明示的な `feasible quote interval` 計算を入れるべきです。

やることは単純です。

1. best bid/ask から post-only 非交差の上限/下限を作る
2. `min_spread_jpy` / `sell_max_spread_jpy` を制約として重ねる
3. offset 系を適用した希望価格帯を計算する
4. 積集合が空なら、`no_feasible_quote` として早期 skip

これを入れれば、「色々計算した末に最後に reject」ではなく、**構造的に無理な cycle** を早く切れます。

---

### 1.6 [HIGH] `status_unknown` + cancel失敗系は、まだ phantom position の温床

`225#` でも残課題になっていた箇所で、まだ本質解決していません。

### 実ログの事実

例:

- `2026-03-03 00:11:54` `sell` 発注
- `00:12:11` `status unknown after 3 retries — treating as cancelled`
- 直後に cancel 400: `Failed to cancel the order`
- 再確認で `order not found in transactions either`
- そのまま `filled=False`

### コード上の根拠

- `scripts/v460/lib/order_monitor.py:320`
  - `status unknown` を cancelled 扱いに寄せる
- `scripts/v460/lib/order_monitor.py:523`
  - cancel 失敗時に再確認するが、見つからなければそのまま抜ける

取引所 API の遅延・整合遅れがある以上、  
`not found now` は `never filled` と同義ではありません。

### 推奨対応

ここは `filled=False` 即断ではなく、**quarantine state** を持つべきです。

最低限、次を追加すべきです。

1. `pending_reconciliation` 状態を FillRecord に残す
2. 残高差分と open orders を次 cycle で再照合
3. 照合完了まで同 side の在庫計算を慎重側へ寄せる

これをやらないと、**「フラットだと思っていたのに実は片持ち」**が起こり得ます。

---

### 1.7 [MEDIUM] 230# の「hasattr 排除」は部分達成で、まだ匂いが残る

これは致命傷ではありません。  
ただし、「全部片付いた」という理解は危険です。

### 事実

- `230#` は `fill_cycle_executor.py` の 10 箇所中 8 箇所を削減
- ただし `scripts/v460/lib/fill_cycle_executor.py:343`
- そして `scripts/v460/lib/fill_cycle_executor.py:554`

この 2 箇所は依然として

- `hasattr(self, "_current_regime_value")`

を使っています。

これは文書どおり「mixin method 存在確認なので据え置き」で、即バグではありません。  
ただ、**mixin 契約が暗黙のまま**という意味で、コード衛生の課題はまだ残っています。

### 推奨対応

- `_current_regime_value()` を Protocol / base mixin で契約化
- もしくは `FillCycleExecutorMixin` 側に no-op 実装を持たせる

これで `hasattr` の最後の正当化を外せます。

---

### 1.8 [LOW] FFD は still “副作用 getter” の匂いがある

`229#` で `consume_recovery_cycle()` へ改名したのと同じ観点です。  
FFD も API 名と副作用が少し噛み合っていません。

### コード上の事実

- `scripts/v460/lib/fast_fill_defense.py:102`
  - `get_boost_multiplier()` の中で TTL 期限切れなら state を変更する
- `scripts/v460/lib/fast_fill_defense.py:296`
  - `import_state()` は `state.get(...) or default` を使う

前者は「getter なのに mutate」、後者は現状の型では問題ないものの、将来フィールド追加時に意味論が崩れやすい書き方です。

### 推奨対応

- `get_boost_multiplier()` は将来的に `refresh_and_get_boost_multiplier()` か、明示的 `tick()` 分離を検討
- `import_state()` は `None` 判定を個別に行う helper に寄せる

優先度は低いですが、**防御ロジックほど API の意味論を曖昧にしない方がよい**です。

---

## 2. 市場理論に基づく提案

### 2.1 Avellaneda-Stoikov: 在庫圧を「強制売買」ではなく「予約価格と参加率」に落とす

今の `balance_forced` は、実質的に

> 在庫が危ない → 危険 side でも入る

になっています。  
MM 理論では、在庫圧は本来 **reservation price と quote width** に反映すべきです。

したがって、`balance_forced` 時は

1. reservation price を在庫解消方向へ寄せる
2. ただし spread は広げる
3. lot は縮める
4. 参加頻度も下げる

という **価格・サイズ・時間の 3 軸調整** にすべきです。

---

### 2.2 Glosten-Milgrom / Kyle: 逆選択リスクは “skip or go” の二値より participation budget で扱う

`dynamic kill` や `skip_gate` は hard block に寄りがちです。  
しかし情報優位フローは連続的で、二値制御だけだと

- 止まりすぎる
- 止まるべき時に在庫都合で全部外れる

の両方を起こします。

そこで、

1. 危険度が軽いときは quote を広げる
2. 危険度が中くらいなら 1/N 参加
3. 危険度が高いときだけ hard stop

という **toxicity budget** にした方が、収益機会と防御の両立がしやすいです。

---

### 2.3 2階層制御: 大相場の方向判断と micro execution を分離する

「大値動きについていけない」問題の延長線上で見ると、現行系はまだ microstructure guard に比重が寄っています。

本来は

1. macro/mid horizon: 今どちらに参加すべきか
2. micro horizon: その side をどれだけ保守的に出すか

を分けるべきです。

`230#/231#` は後者の改善ですが、前者が弱いままだと「うまく守るが、大局を取れない」状態が続きます。

---

### 2.4 “No-Quote Zone” を正式概念として導入する

今回の `min_spread` / `max_spread` / post-only / offset 競合は、実務的には

> **今は板が悪く、そもそも出してはいけない**

というだけです。

これを異常系ではなく、**平常の市場状態の一種**として扱うべきです。

`no_feasible_quote` を first-class reason にすると、ログ解析も一気にやりやすくなります。

---

## 3. デッドロック観点の判定

### 3.1 古典的なデッドロック

レビューした範囲では、**lock/mutex/await の相互待ちによる古典的デッドロックは見当たりません。**

---

### 3.2 残っているのは logical deadlock / liveness trap

現実に危険なのは次の 3 つです。

1. **片側強制ループ**
   - `buy` 不足 → `sell` へ強制 → `sell` 側 guard で削られつつ継続
2. **feasible set collapse**
   - 出したいが、制約を全部満たす価格が存在しない
3. **phantom flat 誤認**
   - `status_unknown` を未約定扱いし、内部状態だけフラット認定

これは「止まる」より厄介で、**動き続けながら悪い状態を維持する**タイプの不具合です。

---

## 4. 本筋投入前の優先順位

### P0

1. **230#/231# を含む SHA で新 run を開始し、live 検証ログを作る**
2. **`balance_forced` を degraded liquidation mode 化する**
3. **`no_feasible_quote` を明示導入する**
4. **`status_unknown` 系に reconciliation quarantine を入れる**

### P1

1. **`one_sided_consecutive_limit` を hard breaker まで昇格する**
2. **`skip_gate` / `dynamic kill` / `balance_forced` を participation budget 化する**
3. **FFD の API 意味論 (`getter` の副作用) を整理する**

### P2

1. **mixin 契約の明示化 (`_current_regime_value`)**
2. **guard reason を「市場都合」と「システム都合」で分類し直す**

---

## 5. Gemini に投げると良い論点

セカンドオピニオンを取るなら、次の 4 点に絞ると良いです。

1. `balance_forced` を hard bypass から graded liquidation へ落とす最適設計
2. `min_spread` / `sell_max_spread` / post-only を統合した `no_feasible_quote` の定式化
3. `status_unknown + cancel fail` のときの在庫再照合プロトコル
4. 230#/231# の FFD 改善が、実際に優先度上位か、それとも局所最適か

---

## 6. 最終判断

`222#`〜`231#` の流れで、コード品質と防御の細部は確実に前進しています。  
ただし、**本筋投入を止めている本当の原因は FFD の精密化不足ではありません。**

今のボトルネックは

- 旧 SHA のまま live が走っていること
- 在庫都合で危険 side を通常運転してしまうこと
- 発注可能価格帯が空になる設計を明示的に扱っていないこと
- `status_unknown` をまだ安全に畳めていないこと

です。

したがって、次にやるべきは

> **FFD の追加細工より先に、執行可能性と在庫圧の扱いを一段抽象化して整理すること**

です。ここをやると、mainline に戻したときの失敗確率が目に見えて下がります。
