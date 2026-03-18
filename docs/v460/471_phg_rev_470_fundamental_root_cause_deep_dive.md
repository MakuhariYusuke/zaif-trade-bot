# 471# 470# レビュー: 根本原因分析の妥当性確認と追加深掘り

**種別**: rev  
**対象**: 470# 根本原因分析：なぜBotは構造的に負けるのか  
**日付**: 2026-03-18

---

## §0 結論

470# の中心主張には、**当たっている部分と外している部分が両方ある**。

当たっている点は明確である。

1. **現行 sell 側は base price 生成時点で mid に寄りすぎている**
2. **`sell_offset_floor: 0.30` は売りを防御するどころか、最低限の攻撃性を保証してしまっている**
3. **sell 側の fill 品質が悪く、逆選択コストが重い**
4. **EV の売り側はほぼ縮退しており、品質フィルタとして期待できない**

しかし、470# が言う

> 「全ての防御的ブーストが逆方向に効いている」

は、コード事実としては **不正確** である。実際にはもっと悪い。現在の実装は、

- `maker_price.py` では「ratio が大きいほど mid に近い」
- `offset_pipeline.py` では「multiplier > 1 で sell price を mid から遠ざける」
- `micro_timeout` 再クオートでは「ratio を価格比率のように mid に直接掛ける」

という **少なくとも3種類の offset 意味論が混在** している。

したがって、470# の「単一の構造的バグ」という整理は惜しい。より正確には、

> **売り側 floor 問題は確かに重いが、本当の根は「offset ratio の意味がモジュール間で一致していないこと」にある**

である。

ロック誤動作については、469# の修正で主因からは外れたと見てよい。少なくとも 470# の損失構造を説明する主犯ではない。今の主戦場はロックではなく、**価格組立てと再クオートの整合性** である。

---

## §1 470# の正しい点

### §1.1 sell floor が危険なのは事実

`maker_price.py` の基底価格生成では、sell 価格は

```text
sell_price = best_ask - spread * effective_offset_ratio
```

で作られている。これは ratio が大きいほど ask から mid 側に寄ることを意味する。

さらに compute パイプラインの早い段階で

- side base offset
- inventory skew
- regime boost
- spread adaptive
- volatility / liquidity 補正

を通した後、sell については `sell_offset_floor` が再適用される。

つまり 470# の

> `sell_guard.offset_floor: 0.30` が売りの最低攻撃性として作用する

という指摘は妥当である。ここは reviewer として明確に支持する。

### §1.2 sell 側のフィルタ非対称も実務上の問題である

470# が指摘するように、`skip_ranging_buy_low_vol` に相当する hard quality filter は buy 側に偏っている。売り側は

- offset 調整
- hour boost
- macro boost
- cross-venue retreat

のような price-based 防御が多く、buy のような単純な hard skip が薄い。

そのため、品質の悪い sell 機会が price control の失敗時にそのまま約定へ流れやすい。この問題設定は正しい。

### §1.3 EV 縮退の疑いも強い

470# の EV 評価は強い主張だが、少なくとも

- 売り側の EV 分布が極端に狭い
- 高 EV が高 PnL に繋がっていない

という観察は、現行の制御において EV が良い指標として機能していないことを示唆する。売り側 EV を信用して offset 調整するのは危険、という警告としては有効である。

---

## §2 470# の最大の誤差: 「全ブースト逆効果」は正確ではない

### §2.1 executor 側の multiplier は sell を mid から遠ざける実装である

470# は、pipeline の各段がすべて「mid に向かって押し込む」と書いている。これは base `maker_price` の意味論をそのまま executor に延長してしまっている。

しかし `pre_order_adjustments.py::_apply_offset_multiplier()` の既定モードは、

- buy: multiplier > 1 で price を下げる
- sell: multiplier > 1 で price を上げる

という実装であり、**sell については mid から遠ざける方向** に動く。

つまり、`trending_sell`, `toxicity`, `VG supplement`, `macro boost`, `alert mode` などの executor multiplier は、470# が書くような「全部が ask を mid に寄せる」ものではない。

### §2.2 だが、これで安心ではない

むしろ問題は深い。price は遠ざけているのに、`effective_offset_ratio` の数値はそのまま乗算で増えている。

するとシステム内部では、

- **price の意味**: 保守的に遠ざかった
- **ratio の意味**: 数値だけ見ると大きくなった

というズレが発生する。

このズレは

- final clamp
- observability (`execution_pre_clamp_offset`)
- degraded liquidation
- micro-timeout re-quote

など、ratio を後続で再利用する箇所を汚染する。

470# は「方向が全部逆」と見ているが、実態は **方向が逆なのではなく、方向定義が層ごとに割れている** である。

---

## §3 HIGH: 470# が見落としている本命1 — `_recalc_price_with_new_offset()` の式が基底価格式と整合していない

これは 470# が拾えていない、かなり重い問題である。

### §3.1 base price 生成の式

`maker_price.py` では、最終価格は

```text
buy_price  = best_bid + spread * ratio
sell_price = best_ask - spread * ratio
```

で作られる。

### §3.2 しかし再計算ヘルパーは half-spread 前提になっている

`pre_order_adjustments.py::_recalc_price_with_new_offset()` は、コメントでも実装でも

```text
order_price = mid ∓ spread * ratio / 2
```

を前提に mid を逆推定している。

これは base の price 生成式と一致しない。

### §3.3 含意

このヘルパーは少なくとも以下で使われる。

- degraded liquidation
- offset pipeline final clamp

つまり、**offset を後から再計算している箇所の price が、基底価格式と別の幾何学で更新されている** 可能性が高い。

470# は sell floor を主犯にしているが、レビュー観点ではこちらの方がむしろ危険である。sell のみならず buy も巻き込む、価格再計算の根本的な不整合だからである。

---

## §4 HIGH: 470# が見落としている本命2 — micro-timeout re-quote が spread ratio ではなく価格比率として再利用している

これもかなり危険である。

`fill_cycle_executor.py` の micro-timeout 再クオートでは、再注文価格を

```text
buy  = mid * (1 - effective_offset_ratio)
sell = mid * (1 + effective_offset_ratio)
```

で計算している。

しかし `effective_offset_ratio` は、これまで一貫して

> spread に対する比率

として使われてきた値である。たとえば `0.20` は「mid の 20%」ではなく「spread の 20%」である。

### §4.1 何が起きるか

mid が 11.8M JPY で ratio が 0.20 なら、sell re-quote は

```text
11.8M × 1.20 ≒ 14.16M
```

となる。これは通常の maker 調整ではなく、**完全に別物の遠方価格** である。

### §4.2 470# への含意

470# は 3/16-3/18 を「sell offset floor の問題」と読んでいるが、その期間は micro-timeout が既に有効である。すると

- 初回注文は base semantics
- 再クオートは price-percentage semantics

が混在することになる。

このため、470# の観測結果の一部は sell floor だけでなく、**micro-timeout 再クオートの価格異常** によって悪化している可能性がある。

これは 470# の単一原因説を弱める、非常に重要な見落としである。

---

## §5 HIGH: 470# は「単一の構造的バグ」と言い切っているが、実際は三重の意味論破綻である

レビューとして最も重要な整理はこれである。

### §5.1 現在の offset 意味論は少なくとも3通りある

1. **maker_price 基底層**  
   ratio 大 = mid に近い = 攻撃的

2. **offset_pipeline 調整層**  
   multiplier > 1 で sell price を遠ざける = 保守的

3. **micro-timeout 再クオート層**  
   ratio を価格比率として mid に直接掛ける

### §5.2 含意

470# は 1 の問題を正しく突いているが、2 と 3 を見落としている。その結果、

> 「設計は全部逆」

と書いてしまっている。実際は

> **設計が逆なのではなく、設計が一致していない**

がより正確である。

この差は大きい。前者なら sell parameter を縮めれば済むが、後者なら **価格計算契約の再定義** が必要になる。

---

## §6 ログ分析の見落とし補足

### §6.1 470# は same-SHA / same-config 固定の強さが弱い

470# は対象 SHA を `dec605767` としているが、3/16-3/18 の fill_records 比較においては、前回 461#/462# で指摘した通り

- run drift
- config drift
- population drift

を常に警戒すべきである。

今回の 470# は 461# よりは絞っているが、それでも

- micro-timeout 有効化後
- deep-night 調整進行中
- lock fix 後

の複合局面であり、「単一バグが全損失を説明する」と断ずるにはまだ慎重さが必要である。

### §6.2 売りの高 fill rate は「設計成功」ではなく「悪い fill を通しすぎ」の疑いが強い

Buy 10.2% / Sell 26.5% という数字は、470# が言うように quality asymmetry の可能性が高い。レビューとしては、ここを次のように補足したい。

- sell fill rate の高さは優位性ではない
- sell で hard quality filter が弱く、しかも価格が浅いなら、むしろ危険信号
- 「売りがよく刺さる」のではなく「悪い相手に売らされている」可能性が高い

これは 470# の方向性を補強する点である。

### §6.3 即約定の悪さは、単なる Glosten-Milgrom の一般論以上に「自ら最前に出す実装」と結びついている

470# は <10秒 fill の悪さを逆選択一般論として説明している。これは正しいが、さらに一段補足すると、現状は

- sell floor
- sell ceiling の高さ
- quality skip の薄さ

により、sell が **自ら情報優位フローの最前列に立つ** 構造になっている。つまり、単に市場が悪いのではなく、システムが bad selection を能動的に招いている疑いが強い。

---

## §7 ロック誤動作について

469# の lock 修正は筋が通っている。少なくとも今回レビューした 470# の内容からは、**lock conflict が現在の損失主因として残っている痕跡は薄い**。

残るとすれば lock そのものではなく、lock 周辺で派生する

- `pending_reconciliation`
- `status_unknown_fast`
- state persistence / restart 境界の歪み

の観測問題である。したがって 471# の立場としては、

> lock 誤動作は主犯から降りたが、運用境界のノイズ源としては引き続き監視対象

が妥当である。

---

## §8 470# への最終判定

### §8.1 正しい点

- sell floor が危険
- sell 側品質フィルタが薄い
- sell fill が逆選択に偏っている
- EV 売り側はほぼ壊れている

### §8.2 言い過ぎな点

- 「全ブーストが逆方向」
- 「単一の構造的バグに帰着」
- 「P0 で sell offset 縮小だけを即やれば最大改善」

### §8.3 私の最終結論

470# は、**sell 側 floor 問題を主犯候補として炙り出したレポートとしては価値が高い**。ただし reviewer としては、根本原因を次のように言い換える。

> 現在の真の根本問題は、sell floor 単体ではなく、offset ratio の意味が `maker_price`・`offset_pipeline`・`micro-timeout` の間で一致していないことである。sell floor はその中でも最悪の初期条件を作る増幅因子である。

つまり、470# の P0 は方向としては近いが、まだ十分ではない。もし実装判断をするなら、本来の優先順位は

1. **offset 契約の一貫化**
2. **micro-timeout re-quote 式の見直し**
3. **`_recalc_price_with_new_offset()` の数式整合性修正**
4. **その上で sell floor / sell ceiling / sell quality filter を再設計**

である。

---

## §9 reviewer メモ

470# は「単純すぎる結論」に見えるが、これはむしろ良い兆候でもある。コードベースの複雑さが高い今、こういう単純化仮説を出してみること自体は価値がある。

ただし今回の仮説は、review の結果として次の形に精密化されるべきである。

> 売り側のセマンティック反転は部分的に事実だが、真の深層原因は offset semantics の層間不整合であり、sell floor はその破綻を最も露骨に露出させている箇所である。
