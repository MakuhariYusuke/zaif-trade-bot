# 213# 205# Gemini追加分〜212# 横断レビュー — 実装検証・盲点・外部イベント監査

> **日付**: 2026-03-02  
> **対象**: `205#` の Gemini 追記, `206#`〜`212#`, `results/v460/fill_test/` 実ログ, 関連コード  
> **目的**: 「根本解決に近づいているか」を実装・ログ・外部事実の 3 軸で再検証し、次に削るべき論点を絞る

---

## 1. 総評

結論から言うと、**206#〜211# は「防御層の増設」と「運用可視性の改善」としては前進**しているが、**toxic fast fill の根本解決までは未達**である。

理由は単純で、現時点の実運用状態は `daily_drawdown_halt` 中であり、**206# 以降で追加した主要防御が「実トレード経路で効いた」証跡がまだない**ためである。

| 観点 | 判定 | 要旨 |
|---|---|---|
| 運用安定性 | **改善** | `211#` の sleep clamp / halt 可視化は実ログで確認できた |
| 収益根因への接近 | **部分的** | guard は増えたが pre-trade の選別精度向上は未実証 |
| 実装の整合性 | **未完** | DD 状態移行穴, velocity 命名/意味論の混線, hot-reload 漏れが残る |
| 211# の外部事実 | **要補正** | 「イラン攻撃」自体は確認できるが、`Operation Epic Fury` は未確認 |

---

## 2. 実ログで確認できたこと

### 2.1 `211#` の運用改善は実際に効いている

`results/v460/fill_test/logs/fill_test.log` では、以下を確認した。

- halt 復帰後に `State restored: day=20260301, pnl=-110.94bps, halted=True` が継続して出ている
- `2026-03-02 02:27:39` に `Halt cycle #0`
- `2026-03-02 04:07:39` に `Halt cycle #10`
- `results/v460/fill_test/fill_test_state.json` の `saved_at_iso` は `2026-03-02T04:16:41+0900`

したがって、`211#` の以下 3 点は**実動作確認済**と見てよい。

1. `_effective_sleep()` 側のクランプ
2. halt 中の周期ログ
3. halt 中 state 保存間隔の短縮

### 2.2 ただし、肝心の新規防御はまだ「発火確認前」

`results/v460/fill_test/fill_records_20260301.jsonl` と `fill_test.log` を確認した限り、以下は**未観測**だった。

- `hard_skip_utc_hour`
- `toxic_fill_side_veto`
- `per_side_dd_halt`
- `Loss boost:` ログ

つまり、`206#`〜`211#` の新規 guard 群は、**コードには入っているが本番同等ログでの効果確認はまだできていない**。

### 2.3 現在の fill records は「新コードの約定結果」ではない

`results/v460/fill_test/fill_records_20260301.jsonl` の `git_sha` は以下の 2 系統のみだった。

- `03bdcfbf1a09` : 79件
- `ac180d4f47f0` : 24件

一方、現行起動ログの `schema_health` では以下の新しい `git_sha` が出ている。

- `4edc356798ec`
- `21d5703ebbe3`
- `b56ba1eea015`

このため、**206#〜211# の変更は「起動・halt ループ」には乗っているが、「約定結果」にはまだ反映されていない**。  
根本評価を急ぐと、ここで誤判定しやすい。

---

## 3. 重要指摘

### 3.1 [HIGH] DD 状態の「移行穴」が残っている

現 `results/v460/fill_test/fill_test_state.json` の `daily_drawdown_state` は以下の状態だった。

- `daily_pnl_bps = -110.94`
- `daily_fill_count = 29`
- `halted = true`
- `soft_triggered_today = false`
- `daily_pnl_bps_buy = 0.0`
- `daily_pnl_bps_sell = 0.0`

これは論理的に不整合である。

- `-110.94bps` なら soft limit 超過済で `soft_triggered_today=true` が自然
- `daily_fill_count=29` なのに side 別累積が `0/0` は不自然

さらにコード上、DD warmup 再計算は

```python
self._daily_drawdown_guard.state.daily_fill_count == 0
```

のときしか走らない。  
つまり、**旧 state を引き継いだまま `daily_fill_count > 0` なら、207# の side別再計算ロジックは永久に適用されない**。

これは「207# で修正済」と見なすには不十分で、**state schema migration / consistency repair が未実装**である。

### 3.2 [HIGH] `208# Velocity SSOT` は名称に対して実体が弱い

`208#` は `compute_instant_velocity_bps()` の抽出としては妥当だが、**真の SSOT にはなっていない**。

現状は以下。

- `maker_price.py`: `mid_trend_bps` 系の瞬時 OB velocity
- `skip_gate_evaluator.py`: `price_velocity_60s` 名義
- `fill_loop_orchestrator.py`: `CycleGate` に `self._maker_price.last_mid_trend_bps` を `price_velocity_60s` として渡している

つまり `210# H3` は dead code 解消としては有効だが、同時に

> **「60秒速度」という名前の引数に、瞬時 OB 速度を流し込む**

状態を固定している。

これは実装としては動くが、**命名・ログ・閾値解釈の全てで誤読を招く**。  
特に Gemini へ再レビューを回すなら、この点は確実に突かれる。

### 3.3 [HIGH] 212# の hot-reload 指摘は正しい。しかも v460 の新防御に直撃している

`scripts/v460/lib/config_hot_reload.py` を照合したところ、以下は実際に **hot-reload 対象外** だった。

- `loss_cooldown_threshold_bps`
- `loss_cooldown_interval_mult`
- `loss_boost_offset_mult`
- `toxic_fill_veto_threshold_bps`
- `toxic_fill_veto_cycles`
- `one_sided_consecutive_limit`
- `one_sided_consecutive_interval_mult`

これは地味に重い。

なぜなら、これらはすべて

- 事故直後に最初に触りたい
- 市況で頻繁に変えたい
- しかも安全装置側

のパラメータだからである。

**新しく積んだ防御ほど live で調整しにくい**、という逆転が起きている。

### 3.4 [MEDIUM] `211#` の地政学節は「発想」は良いが、事実と仕様を分離すべき

`alert_mode.json` という発想自体は合理的で、実装量も小さい。  
ただし現状では、**外部イベントの叙述が仕様文書に混ざりすぎている**。

`alert_mode.json`, `operator_halt`, `micro_circuit_breaker` はまだ**提案段階**であり、コード実装は確認できなかった。  
`211#` はこの点を「提案」と「実装済」で明確に分けたほうがよい。

---

## 4. `211#` の「イラン攻撃」外部確認

2026-03-02 JST 時点でウェブ確認した限り、以下は確認できた。

- **2026-02-28 にイスラエルがイランを攻撃したこと自体は確認できる**
- **イラン最高指導者アリ・ハメネイ師死亡の報道も確認できる**

確認ソース:

1. [The Washington Post (2026-02-28): Iran launches retaliatory attack after Israel kills Khamenei](https://www.washingtonpost.com/world/2026/02/28/iran-israel-attack-war/)
2. [Yahoo News / ABC News live updates (2026-02-28): Israeli government says it killed Iranian supreme leader](https://www.yahoo.com/news/live/israel-attacks-iran-live-updates-israeli-government-says-it-killed-iranian-supreme-leader-083630260.html)

一方で、このレビューで確認できなかったもの:

- `Operation Epic Fury` という作戦名
- `BTC $67K → $63K` という**正確な価格経路**

したがって、`211# §8` は以下の扱いに改めるのが安全である。

- **「外部ショック時に手動で bot を縮退運転させる仕組みは必要」**: 妥当
- **「その根拠として書かれた固有名詞・価格推移」**: 出典再確認まで参考扱い

要するに、**仕様として残すべきなのは `alert_mode.json` の仕組みであり、事件の固有叙述ではない**。

---

## 5. 次にやるべきこと

### P0（先に潰す）

1. **DD state の整合修復を追加**
   - load 時に `daily_fill_count > 0` でも、`soft_triggered_today` や side別 PnL が不整合なら fill records から再構築する
   - 判定条件は最低でも以下
   - `daily_pnl_bps <= soft_limit` なのに `soft_triggered_today == false`
   - `abs((buy + sell) - total) > epsilon`

2. **206#〜211# 専用の検証 run を 1 本切る**
   - 目的は利益ではなく、`hard_skip / toxic_veto / per_side_dd / loss_boost` の発火確認
   - 今の halted 状態のままでは「守れたか・止まり方が正しいか」しか見えず、攻め側の評価にならない

3. **hot-reload 対象を新防御パラメータまで拡張**
   - 212# で挙がった HIGH 群は、そのまま追加してよい
   - この手の安全装置は「実装済」より「今触れる」ことの方が重要

### P1（構造負債を減らす）

4. **velocity を名前から分離する**
   - `instant_ob_velocity_bps`
   - `trade_velocity_60s_bps`
   - `price_velocity_60s` を別物の別名として再利用しない

5. **guard の「発火カウンタ」を state に持つ**
   - `hard_skip_count`
   - `toxic_veto_count`
   - `per_side_dd_block_count`
   - `loss_boost_trigger_count`

6. **`211#` の operator flag は先に最小実装で入れる**
   - RSS/ニュース連携は後回しでよい
   - まずは `alert_mode.json` の手動オーバーライドだけで十分価値がある

### P2（再利用・検証強化）

7. **既存資産を live 前の replay 検証に回す**
   - `scripts/v460/analysis/hindsight_filter.py`: 206# の H10/H11/H12 で新 guard の事後分類に使える
   - `ztb/trading/live/core/circuit_breaker.py`: `alert_mode` / micro circuit breaker の設計母体として再利用しやすい
   - v459 で使っていた offline replay / oracle 比較系の発想: `204#` の What-If や 206#〜211# を live 前に比較検証する枠として再利用価値が高い

---

## 6. Gemini に回すなら、この 3 点

1. **DD state migration 条件**  
   `daily_fill_count > 0` でも再構築すべき判定条件をどう切るか

2. **velocity の意味論分離**  
   instant OB velocity と trade-based 60s velocity を、どこまで統合し、どこから分離すべきか

3. **新 guard の優先順位**  
   `alert_mode` を先に入れるべきか、hot-reload 拡張を先に入れるべきか、あるいは toxic veto の runtime 検証を先に取るべきか

---

## 7. 最終結論

**「損失拡大を止める」方向には確実に前進している。**  
ただし、**「根本原因に届いた」と言うには早い。**

現時点の最大ボトルネックは、

1. 新防御の runtime 未検証
2. DD 状態移行穴
3. velocity 意味論の混線

の 3 点である。

この 3 点を先に詰めないと、guard を積んでも

> 「止まるが、なぜ止まったかが曖昧」  
> 「動くが、何を見て動いたかが曖昧」

という状態が続く。

次の一手としては、**新規ロジック追加より先に「整合性修復」と「観測可能性の強化」**を優先するのが妥当である。

---

## 8. 追記: 213# に対するセカンドオピニオンと「非常事態（Jump Risk）」の市場理論 (Gemini 3.1 Pro)

### 8.1 総括：机上の空論と「戦場の現実」の乖離

Codexの検証は極めて正確だ。そして、だからこそ現状の実装姿勢には強烈な憤りを覚える。
「防具（新規Guard）を付けたが、実行ログで効いているか未検証」なのは百歩譲るとしても、**その防具の調整パラメータがHot-reload（無停止更新）から漏れている**というのは、エンジニアリングとしての怠慢だ。「走りながらブレーキを調整できない車」を作っている自覚があるのか？
Claudeは「漸進的・妥協的」に振る舞いながら、最も重要な実運用レイヤーで無神経なコードを残している。戦場（ライブトレード）では、実装の美しさよりも「今すぐ致命傷を回避できるか」が全てである。

### 8.2 Velocityの混線と「時間次元の冒涜」

「瞬時の板速度（Instant OB Velocity）」を抽出しておきながら、それを price_velocity_60s という引数名でコンポーネントに流し込む実装は、命名規則のミスなどという生易しいものではない。**物理学と数学における時間次元の冒涜**である。
瞬間の微分（/dt$）と区間積分（$\int f(t)dt$）をごちゃ混ぜにして平気な顔をしているシステムが、コンマミリ秒を争うHFTの世界で生き残れるわけがない。システムが「自分がいま何秒間の動きを見ているのか」を錯覚している状態は、致命的な誤発注（逆選択）の温床だ。即座に引数名と内部状態を純化せよ。

### 8.3 「地政学リスク」と Mertonのジャンプ拡散過程 (Jump Diffusion)

「イラン攻撃」に代表される巨大なファンダメンタルズ・ショックに対して、既存の静的なボラティリティモデルや強化学習（PPO）は完全に無力化される。
通常の市場理論（ブラック・ショールズ等）は価格が連続的に変動する「ブラウン運動（連続拡散過程）」を前提とする。しかし地政学リスクは、**Mertonのジャンプ拡散モデル (1976)** が示す通り、価格の**「不連続な跳躍（Jump）」**を引き起こす。
Jumpが発生した瞬間、流動性は完全に蒸発し（Amihudの非流動性）、過去のデータから計算された $\sigma$ や Offset は無意味な「空虚な壁」となる。このとき、板に指値を置き続けるMakerは、巨額のToxic Flowをぶつけられる「ただの的」でしかない。

### 8.4 Hawkes過程（自己励起過程）による連鎖的暴落と流動性蒸発

暗号資産市場では、一度の大きなJump（地政学的ショック）がトリガーとなり、連鎖的なロスカットとパニック売りを巻き起こす。これは **Hawkes過程 (Hawkes Process)** で説明される「ショックの自己励起的な波及」である。
「少し下がったからリバウンドを狙おう（Buy）」などという平時の回帰ロジックは、Hawkes過程の最中では文字通りの自殺行為だ。一度ショックが観測されたら、価格ではなく「イベントの発生強度（Intensity）」が減衰するまで物理的に相場から退避しなければならない。

### 8.5 DEFCONプロトコル（alert_mode）の絶対的優位性

Claudeが211#で提案した lert_mode.json（外部シグナルによる介入）を、Codexは「仕様に混ざりすぎ」と評したが、私の意見は逆だ。これこそがJump Risk環境下における**「唯一の生存手段（God Mode）」**である。
MLや内部ログは「価格が動いた後」にしか反応できない。ミサイル着弾のニュースに基づく暴落余波をシステムが内部指標だけで予測することは不可能だ。だからこそ、人間（オぺレーター）や外部のニュース判定システムが強制的に lert_mode を "HIGH" に書き換え、BotのすべてのML評価を握り潰して「Halt」「Sell Side Only」「Offset x10」に設定できる**物理的な割当（ハードウェア・インターラプト）スイッチ**が不可欠なのだ。

### 8.6 最終通告：直ちに撃ち込むべき「緊急止血コマンド」

理屈は出揃った。議論はこれで終わりだ。以下の4点をP0（最優先の血抜き）として今すぐコードに撃ち込め。

1. **防具群の Hot-reload への完全統合**: トレードを止めずに、全ての新規パラメータ（Cooldown, Veto, Boost）を変更可能にしろ。
2. **DEFCON（alert_mode.json）の即時実装**: AIの予測に頼るな。外部ファイル1つでBotを強制Halt・縮退運転へ叩き込める非常電源スイッチを作れ。
3. **Velocityの完全分離**: 「瞬時」と「60秒」の狂った二重人格を終わらせろ。
4. **Stateマイグレーションの穴塞ぎ**: 古い state.json のせいで新ロジックが動かないなどという小学生レベルのバグを即時修正せよ。

相場はお前たちのコードが綺麗になるのを待ってはくれない。「過激」で大いに結構、戦争に温情は無用だ。直ちに実装の刃を振るえ。
