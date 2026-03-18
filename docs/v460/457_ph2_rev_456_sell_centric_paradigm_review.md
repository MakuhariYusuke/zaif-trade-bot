# 457# 456# セカンドレビュー: Sell-Centric Paradigm 提案 A/B/C の評価

**種別**: rev  
**対象**: 456# 上昇トレンド＝「売り手市場」という逆転パラダイムと対策  
**日付**: 2026-03-17

---

## §0 結論

456# の問題提起は 454# より一段進んでいる。454#/455# が「上昇トレンド時の sell をどう守るか」を主題にしていたのに対し、456# は

- 上昇トレンドを sell 回避局面ではなく sell premium 回収局面として見る
- hard skip ではなく高プレミアム ask を置く
- 在庫の利確と micro-burst を取りにいく

という、より収益志向の発想を提示している。

この発想自体は面白いし、完全には間違っていない。実際、上昇トレンドでは買い手が流動性を欲しがるため、ask 側で premium を抜くという視点には市場理論上の正当性がある。

ただし、現行 v460 実装に照らすと ABC の評価は明確に分かれる。

1. **提案A: 条件付き採用余地あり**  
   ただし「3-5倍の法外な ask」をそのまま実装する案には反対。現行の final clamp と ratio ceiling に正面衝突するため、現実的には **A-lite として moderate な macro sell offset premium** に落とすべきである。

2. **提案B: 発想は良いが、今は重すぎる**  
   これは offset 改良ではなく、実質的に **laddered execution engine** の新設である。現行の single-order / single-monitor 前提とは相性が悪く、今このタイミングで入れるべきではない。

3. **提案C: 現時点では非推奨**  
   もっとも面白いが、もっとも再現性が低い。現行 cross-venue は micro burst を 1-3 秒で狙う超短期 strike 系ではなく、数秒スケールの hint/retreat 補助票である。現システムの責務境界から外れすぎている。

要するに、**456# のパラダイムは「攻めの sell premium 発想」としては評価するが、実装に採るのは A を縮小解釈したものだけで十分であり、B/C はまだ早い**というのが私の結論である。

---

## §1 事実確認: 現行実装と 456# の距離

### §1.1 現行 offset は ratio-based であり、極端な ask をそのまま表現しにくい

現行 executor は、最終的に `effective_offset_ratio` を使って price を決め、最後に side 別 ceiling を再適用する構造になっている。さらに ceiling を大きく超えると hard skip になる。

- sell ceiling: `0.50`
- final clamp hard skip: `ceiling × 2.5`、つまり sell は `1.25` 超で skip

したがって、456# A の「通常の 3倍〜5倍を法外な位置に置く」は、そのままでは表現不能か、表現できても clamp/hard skip に吸われやすい。

これは単なる実装都合ではない。現行システムは、極端な ask を長く晒すより、**暴走した multiplier を抑制する**方向で安全化されているからである。

### §1.2 現行 micro-timeout は「1 本の注文を短く持つ」設計であり、ladder ではない

453#/454# で有効化された micro-timeout は、現在

- 1 回に 1 本だけ発注
- timeout 後に cancel
- 同じ lot を re-quote

という設計である。部分約定の本格利用や複数価格レイヤーの同時管理はまだ入っていない。

そのため、456# B の「+0.20 / +0.40 / +0.60 と切り替えながら小分けに打診する」は、現行の micro-timeout を少し捻れば済む話ではない。**execution engine の責務を拡張する別件**である。

### §1.3 現行 cross-venue は hollowing strike 用ではない

cross-venue は現状、reference venue の

- mid spread
- EMA spread
- velocity
- microprice
- depth imbalance

を用いた hint 生成器であり、もともとの思想は **directional override ではなく adverse-side retreat / veto の補助票** である。これは設定定義にも明記されている。

よって、456# C のような「波が来る直前の数秒だけゲリラ的に ask を晒す」戦略は、今の cross-venue の責務から大きく飛躍している。

---

## §2 提案Aの評価: The Liquidity Mirage

### §2.1 良い点

456# の中では、A がもっとも筋が良い。

理由は 3 つある。

1. **市場理論と整合する**  
   上昇トレンドで ask 側 premium を拡大するのは、単に逃げるより MM 的である。

2. **455# の路線と矛盾しない**  
   hard skip ではなく、まず offset premium を載せるのは 455# で推した F-lite の延長である。

3. **大改造なしで実験できる**  
   「法外な ask」そのものは無理でも、macro up 時の sell offset premium を一段広げるだけなら実装面積は小さい。

### §2.2 問題点

ただし、456# の書き方のままでは 2 つの問題がある。

1. **3-5倍という倍率設定が荒すぎる**  
   sell floor が `0.30`、sell ceiling が `0.50` の現状では、3倍で `0.90`、5倍で `1.50` になりうる。これは現行 clamp 設計と噛み合わない。5倍側は hard skip 域に入りうる。

2. **「めったに刺さらない高値 ask」を狙う設計は、現状では soft skip と区別しにくい**  
   micro-timeout は現状 sell 側 20 秒、max_requote 2 回であり、長時間の mirage order を残して局所バーストを待つ設計ではない。よって極端な ask は、実態として「ほぼ約定しない注文」になりやすい。

### §2.3 判定

**A は採用候補。ただし A-lite に落とすべき** である。

推奨する具体形は次の通り。

- `macro_weak_up -> sell offset ×1.2〜1.5`
- `macro_strong_up -> sell offset ×1.5〜1.8`
- ceiling は現行維持、hard skip 追加なし
- まずは ratio multiplier として実装し、abs bps 指定や専用 mirage mode は後回し

要するに、A の価値は「法外 ask」そのものではなく、**sell を stop ではなく premium 化して扱う思想**にある。そこだけ抽出すれば十分に価値がある。

---

## §3 提案Bの評価: Dynamic Inventory Offloading

### §3.1 良い点

B の発想は理解できる。上昇トレンド時の MM は、単に spread を取るだけでなく、抱えた在庫をどう高く捌くかという問題を持つ。したがって

- 在庫利確を主目的にする
- 天井を点で当てず、面で売り抜ける

という発想自体は自然である。

### §3.2 しかし今のコードベースでは別プロダクトに近い

問題は、これがもはや「macro signal を execution に繋ぐ軽改修」ではない点である。

B を真面目にやるには少なくとも以下が必要になる。

1. 複数価格レイヤーの order 管理
2. 残量追跡
3. partial fill を前提にした再配分
4. cancel race と phantom order の複雑化対応
5. layer ごとの FillRecord 観測設計

現行 micro-timeout は 1 本の注文を短時間で re-quote するだけで、複数レイヤー売りや仮想 ladder を扱うための execution kernel ではない。

### §3.3 市場理論上のリスク

B は理論的にも注意がいる。なぜなら、これは「悪い sell を減らす」ではなく、**上昇局面で意図的に売り在庫を捌く**設計だからである。

つまり、誤判定時には

- 上昇初動で在庫を手放しすぎる
- 買い戻しが高くなる
- MM 収益ではなく directional miss が損益の中心になる

危険がある。これは 455# で後順位に置いた G にかなり近い。

### §3.4 判定

**B は現時点では不採用** が妥当である。

将来的にやるなら、いきなりサブサイクル ladder ではなく、まずは

- cycle 単位で sell 側 lot を在庫連動で少し増やす
- 価格は 1 本のまま
- レイヤリングはしない

という縮小版から入るべきである。

---

## §4 提案Cの評価: Cross-Venue Hollowing Strike

### §4.1 面白さは ABC で最大

C は発想としては最も鋭い。reference venue の板の薄化を「危険信号」ではなく「局所的 premium 回収の予告」と読むのは、単なるトレンド追随よりずっと面白い。

ただし、面白いことと今やるべきことは別である。

### §4.2 現行基盤との差が大きすぎる

この案が成立するには、少なくとも次が必要である。

1. **1-3 秒級の極短 TIF**  
   現行設定は sell 20 秒であり、C の想定スケールより一桁長い。

2. **event-driven cancellation**  
   現在の `cancel_on_cross_venue_flip` は設定だけあり、実ロジックは将来拡張扱いである。

3. **hollowing を直接測る指標**  
   現行は depth imbalance や microprice は見ているが、「板が急速に引っ込んだ」というイベント検知器そのものではない。

4. **spoof / fake thinning 耐性**  
   L5 深度だけで hollowing を読むのは危うい。見せ板や一時的な板消失に過敏反応しやすい。

### §4.3 これはもはや latency-sensitive strike engine である

C の本質は、伝統的 MM よりも **短命な event-driven strike** に近い。しかも reference venue の先行板変化を使うため、実質的には軽い latency arb / micro-event trading の性格を持つ。

そのため、現行の

- 120 秒 cycle
- rule-based safe execution
- maker-only quality observation

という v460 の中核思想からかなり逸脱する。

### §4.4 判定

**C は現時点では非推奨** である。

理由は、悪いからではなく **実証コストに対して観測可能性が低すぎる** からである。今これを始めると、効いたかどうかより先に

- trigger が本物か
- 板薄化が spoof ではないか
- Coincheck 側で本当に premium fill できたか
- その fill 後に mean reversion が起きたか

の検証だけで長く溶ける。

---

## §5 実装担当に渡すなら何を残すべきか

456# の ABC をそのまま実装依頼に変換するのは危険である。実装担当に渡すなら、私は次のように圧縮する。

### §5.1 採用する核

- 上昇トレンドでは sell を「停止対象」ではなく「premium 付け対象」とみなす
- まずは macro signal を sell offset premium にだけ接続する
- hard skip / ladder / strike engine は入れない

### §5.2 採用しない部分

- A の 3-5倍という極端倍率
- B のサブサイクル ladder execution
- C の hollowing strike 本体

### §5.3 実装順

1. A-lite: macro up 時の sell premium multiplier
2. その状態で micro-timeout を短くして stale exposure を削る
3. 観測が揃ってから threshold や hold time を詰める

---

## §6 最終判定

456# は、単なる保守的 stopgap から一歩出て「どう攻めて抜くか」を考えた文書として価値がある。特に A の思想は、455# の F-lite をより収益寄りに翻訳したものとして読むと有益である。

しかし、提案の成熟度にはかなり差がある。

- **A**: 採用価値あり。ただし縮小して使う
- **B**: 将来研究テーマ。今は重すぎる
- **C**: 面白いが、現システムには早すぎる

従って、実装担当に渡すべき結論は次の一文で足りる。

> 456# の本当の収穫は「上昇トレンド時の sell を hard skip ではなく premium 化して扱う」という思想であり、実装に落とすなら A-lite だけを抽出して F-lite/H の延長として入れるべきである。B/C は現段階では研究案に留める。
