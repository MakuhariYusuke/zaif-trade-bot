# 455# 454# レビュー: 上昇トレンド対応の論点整理と実装優先順位の是正

**種別**: rev  
**対象**: 454# 上昇トレンドにおける Sell 損失対策  
**日付**: 2026-03-17

---

## §0 結論

454# の問題意識は概ね正しい。特に、

1. 40 分級 micro regime だけでは時足・日足の上昇を取り逃がす
2. macro regime が execution 側の sell 保護に結びついていない
3. sell 側の損失は「検知不足」と「検知しても制御しない」の二重欠陥で起きている

という整理は妥当である。

ただし、454# は次の 4 点で修正が必要である。

1. **R3 の表現が粗い**  
   sell 保護コードが「存在しない」のではない。micro regime に対する sell 保護は既に存在する。欠けているのは **macro regime 専用の execution 経路** である。

2. **A-E の YAML 調整をやや過大評価している**  
   threshold を動かすだけでは、時足・日足トレンド問題の根は解けない。これはパラメータ問題ではなく、信号 horizon と制御 interface の問題である。

3. **案 F の hard skip を最初から強く入れるのは危険**  
   現行 macro detector は hysteresis を持たず、5m 単独判定も混ざるため、ここに直ちに hard skip を乗せると regime flapping と participation 崩壊を招きうる。

4. **案 G/J は一段重い**  
   inventory target を trend 連動で曲げるのは、もはや MM の保守的拡張ではなく軽い directional book 化である。短期収益の誘惑はあるが、誤判定時の傷も深い。F/B/H より後ろに置くべきである。

要するに、**454# の方向性は正しいが、優先順位は「A/B から触る」ではなく「macro 信号を execution に安全に接続してから threshold を詰める」が正しい。**

---

## §1 事実確認: どこまで既に実装されているか

### §1.1 micro 側の sell 保護は既にある

454# の R3 は、そのまま読むとややミスリードである。現行コードには既に以下が存在する。

- `skip_sell_trending`
- `skip_sell_trending_up_only`
- `trending_sell_as_offset_enabled`
- `trending_sell_offset_boost_factor`
- `trending_up_sell_offset_boost`

つまり、**short horizon の trend に対する sell 保護は存在する**。問題は、それが 40 分級の micro 判定に閉じており、日中のドリフトや 5 日スケールの上昇を拾う設計ではない点にある。

### §1.2 macro 側は「観測と矛盾検知」に止まっている

一方で macro regime は、現状では主に以下の用途に限られている。

1. `fill_cycle_executor.py` で detector を update
2. `compose_regimes()` で micro/macro conflict を判定
3. conflict 時に `log` または `ranging downgrade`
4. `FillRecord` に `macro_trend`, `macro_slope_5m`, `macro_slope_15m`, `macro_aligned` を記録

ここで重要なのは、**macro_trend が sell gate や offset multiplier に直接入力されていない**ことである。したがって 454# の中心主張は、厳密には

> 「macro trend の検知器はあるが、macro-specific な sell 保護経路が execution に未接続」

と書くのが正しい。

### §1.3 したがって 454# の本当の論点は R1/R2/R3 ではなく S1/S2 である

- S1: **signal horizon mismatch**  
  40 分 lookback の micro regime で時足・日足トレンドを扱おうとしている
- S2: **signal-to-action interface missing**  
  macro signal が execution の sell/buy 非対称制御に入っていない

この 2 点こそが本質であり、threshold の値自体は二次的である。

---

## §2 市場理論レビュー

### §2.1 454# の方向性は市場理論と整合する

上昇トレンドで sell 側が不利になるという観察は、単なる経験則ではない。

- maker の sell は上昇相場では「上がる資産を先に手放す」方向に立つ
- informed flow が買いに偏る局面では、ask 側に残る受動板は逆選択を受けやすい
- したがって、uptrend 時に sell 側を保守化するのは自然である

この意味で、454# が sell 側保護を主題に据えたこと自体は妥当である。

### §2.2 ただし「trend を検知したらすぐ hard skip」は市場理論上も乱暴

マーケットメイカーの仕事は、方向性の完全放棄ではなく、**不利フローには薄く、有利フローには厚く** である。

そのため、理論順序は通常次の通りである。

1. offset 拡大
2. timeout 短縮
3. inventory target の微修正
4. それでも危険な領域だけ hard skip

454# は案 F でいきなり `STRONG_UP -> sell skip` を前に出しているが、これは detector の品質が十分でない段階では強すぎる。特に現行 macro detector は strong/weak 分類の滑らかさが低く、トリガー品質に対してアクションが重い。

### §2.3 そもそも hourly/daily trend は execution filter だけではなく alpha 層の問題でもある

429# の三層分離に照らすと、時足・日足トレンドは execution 層だけで抱えるべき問題ではない。

- micro regime: 執行の安全化
- macro regime: 執行の方向非対称化
- sidecar / alpha: より長い horizon の方向性示唆

したがって、「日足トレンドを 40 分 regime にもっと頑張らせる」という発想は設計として不自然である。J の multi-timeframe 化は理解できるが、長い horizon ほど本来は sidecar 寄りの責務である。

---

## §3 設計レビュー

### §3.1 454# の最大の強み: 問題を検知層と実行層に分けて見たこと

これまでの議論では sell 損失が `sell_dynamic_kill` や時間帯回避などの stopgap に吸われがちだった。454# はそこから一歩進み、

- detect できていない
- detect しても action していない

を分離した。これは正しい。

### §3.2 しかし案 A は「微修正」ではなく regime 定義の変更である

`trend_threshold_pct: 0.5 -> 0.25` は単なる tuning ではない。検知対象の市場構造を変える変更である。

副作用は少なくとも 3 つある。

1. trending 判定の母数が急増する
2. 既存の `skip_sell_trending` と `trending_*_offset_boost` の発火頻度が跳ねる
3. buy/sell 両側の fill distribution がまとめて変わる

したがって、A は YAML-only でも軽い変更ではない。**A を先に当てると、macro path の欠陥と micro 過敏化の影響が混ざって原因が読めなくなる。**

### §3.3 案 B は正しいが、単独では no-op

これは 454# 自身も書いている通りで、現行 macro detector の出力は execution 制御に直接入らない。よって B 単独は観測改善であり、収益改善ではない。

ただし私は、B の価値を「no-op だから低い」とは見ない。むしろ、**F の前提整備としての観測改善** という意味で重要である。

### §3.4 案 F は本件の中核だが、実装方法を変えるべき

454# の F は方向性として最も正しい。だが、実装順序は次のように弱く始めるべきである。

1. `WEAK_UP -> sell offset boost`
2. `STRONG_UP -> sell timeout short`
3. `STRONG_UP -> sell hard skip` は最後

理由は明快で、現行 macro detector は

- 5m だけで weak 判定を出す
- 15m が揃う前は confidence が限定的
- hysteresis がない
- action 側の hold time もない

からである。ここに hard skip を直結すると、売り participation が粗く落ちる。

### §3.5 案 G は魅力的だが、454# の段階では早い

Inventory Sponging は 447# の提案としては面白い。しかし、これは「不利な sell を避ける」ではなく「有利な方向へ在庫を傾ける」施策である。

この差は大きい。

- F/H: 受動板の質を変える
- G: ブックの方向性そのものを変える

G は利益も大きいが、誤判定時の逆噴射も大きい。000# の maker-only 原則と短期高収益の両立を考えるなら、F/H の検証を終える前に G へ入るのは設計が荒い。

### §3.6 案 J は正論だが、「新 detector を増やす」前に今ある二層を使い切るべき

J は理論上もっとも根本的に見える。しかし現状は、

- micro detector はある
- macro detector もある
- だが macro action path がない

という段階である。ここで multi-timeframe fusion を追加すると、問題の核心が「signal 不足」なのか「配線不足」なのか再び曖昧になる。順番としては遅い。

---

## §4 安定性レビュー

### §4.1 現行 macro detector には hysteresis がない

micro detector には `hysteresis_count` があるが、macro detector には同種の確定機構がない。OLS slope が threshold を跨ぐたびに weak up/down が揺れる構造である。

この状態で

- offset boost
- timeout 短縮
- hard skip

をそのまま繋ぐと、アクションが regime noise に引きずられる。

### §4.2 `slope_threshold: 1.0 -> 0.3` は制度変更であり、軽調整ではない

0.3 bps/min は 5m では約 1.5bps、15m では約 4.5bps の drift である。BTC/JPY の intraday ノイズを考えると、ここは「弱いが意味のある trend」でもあり、「noise に引っかかる閾値」でもある。

つまり、B は正しい方向でも、安易に 0.3 固定へ落とすと false positive が増える。ここで必要なのは単純な threshold だけではなく、

- 連続確認
- confidence floor
- action 側 hold time
- side 別非対称 multiplier の上限

である。

### §4.3 337# の教訓を忘れてはいけない

337# は、sell 制御を強くしすぎると

- sell が通らない
- たまに通る売りだけが悪い条件で刺さる
- rolling PnL が戻らず kill が自己強化する

という悪循環を示した。F や D の設計では、この教訓を必ず踏まえる必要がある。**「上昇時 sell を守る」が「sell を壊す」に化ける危険は現実にある。**

### §4.4 453# を無視して J へ進むのは順番が悪い

453# で micro-timeout 基盤は既に入り、しかも default disabled の安全な状態にある。これは本件に対する最も面積の小さい改善余地であり、

- hard skip ほど粗くない
- offset だけよりも stale exposure を直接減らせる
- macro trend と連動しやすい

という利点を持つ。454# でも H は挙がっているが、優先度はもっと高くてよい。

---

## §5 追加で必要な観点

### §5.1 観測設計

454# は「何を変えるか」は書いているが、「効いたかどうかを何で判定するか」がまだ弱い。最低でも以下は必要である。

- macro regime 別 sell fill 数
- macro regime 別 sell AS rate
- macro regime 別 sell pnl120 / post_fill_30s_pnl
- macro signal 発火中の participation 低下率
- run_id / git_sha 固定比較

これがないと、169# と同じく母集団混線で議論が曖昧になる。

### §5.2 action cost の計器化

F/H を入れるなら、効果だけでなくコストも見るべきである。

- 何件 skip したか
- 何件 timeout-shortened したか
- その結果 missed favorable sell が何件あったか

「損失売りを減らした」だけでは不十分で、「利益売りをどれだけ捨てたか」も見る必要がある。

### §5.3 役割境界

時足・日足の trend を将来的に本当に取りにいくなら、最終的には 429# の sidecar 構想へ接続すべきである。execution 層だけで解こうとすると、offset と skip の複雑さばかり増えて責務が濁る。

---

## §6 推奨実装順

454# の優先順位は見直すべきである。私の推奨順は以下である。

### Step 1: F-lite を先に入れる

hard skip なしで、まず macro-trend を execution に接続する。

- `macro_weak_up -> sell offset boost`
- `macro_strong_up -> sell offset boost stronger`
- `macro_weak_down -> buy offset boost`
- まずは skip ではなく multiplier のみ

ここで初めて「macro を見て実際に行動した」状態になる。

### Step 2: B を保守的に下げる

`1.0 -> 0.5` 程度から始め、いきなり `0.3` へ飛ばない。併せて

- 連続 N 回一致で action 有効
- action hold cycles

を入れる。

### Step 3: H を有効化して sell stale exposure を削る

macro up 時だけ sell 側の micro-timeout を短縮する。これは F と相性がよい。

### Step 4: ここまでで足りなければ A を小さく入れる

micro threshold を `0.5 -> 0.35` 程度で試す。A を最初から大きく動かすのは避ける。

### Step 5: G/J は最後

inventory target の傾斜や multi-timeframe fusion は、F/B/H の結果を見てから判断する。

---

## §7 最終評価

454# は「bot が hourly/daily trend を掴めていない」という問題提起としては有効であり、特に **macro detector が execution に接続されていない** という設計上の穴を前面に出した点は評価できる。

一方で、現段階で本当に言うべきことは次の 3 点に尽きる。

1. **40 分 regime をいじっても、時足・日足問題の本質は消えない**
2. **macro detector は既にあるので、次の仕事は detector 追加ではなく action path 接続である**
3. **hard skip より先に、offset と timeout の穏やかな非対称化で品質を測るべきである**

従って、454# に対する私の最終判定は以下である。

- 問題設定: **概ね正しい**
- 原因分解: **7割正しいが、R3 は表現修正が必要**
- 実装順序: **要修正**
- 推奨採用案: **F-lite -> B(保守) -> H -> A(小さく) -> G/J**

現時点で最も避けるべきは、**A/B/D/G を一気に入れて「何が効いたのか分からないまま participation だけ壊す」こと**である。
