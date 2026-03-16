# 428# 427レビュー — 全体方針・Sidecar・Final Clamp の再整理

**Date**: 2026-03-15  
**Target**: `docs/v460/427_global_quant_cultures_and_sidecar_plan.md`  
**Scope**: 全体方針、設計妥当性、市場理論、既存実装活用、見落とし補完

---

## 1. 主要所見

| # | 重要度 | 観点 | 指摘 | 根拠 |
|---|---|---|---|---|
| 1 | HIGH | 実装現実 | `427#` は `Sidecar` と `Final Clamp` を「これから入れる新方針」として書いているが、**骨格は既に入っている** | `scripts/v460/lib/fill_cycle_executor.py:742`, `scripts/v460/lib/fill_config.py:324`, `scripts/v460/lib/orchestrator_mid_cycle.py:135`, `scripts/v460/lib/sidecar_types.py:1` |
| 2 | HIGH | 本当のブロッカー | 現在の課題は「未実装」ではなく、**Sidecar が live で実質死んでいること** と **効能未検証** | `cache/sidecar_signal.json` は `2026-03-11` の neutral で停止。`2026-03-09`〜`2026-03-14` の fill_records `2467` 行で `sidecar_offset_bps` / `sidecar_bias` 非 null が `0` |
| 3 | HIGH | 方針決定 | `427# §2` の国別文化論は発想源としては有益だが、**設計判断の主根拠にはしない方がよい** | 国別分類は引用や定義がなく、一般化が強い。v460 の設計は「国」ではなく「Coincheck BTC/JPY の microstructure」「小規模運用の保守性」「観測可能性」で決めるべき |
| 4 | HIGH | 進め方 | `427# §4` は「実装計画」より **検証計画 / 統合計画** に改めるべき | Final Clamp は既に稼働し、ログでも発火中。Sidecar も配線済みだが stale で無効。重複実装より live 証明と効果測定が先 |
| 5 | MEDIUM | 安全設計 | Final Clamp は必要だが、**安全装置だけで upstream の失敗を隠す危険** がある | clamp 発火率、pre/post-clamp 条件付きPnL、hard-skip率を見ないと「損失は防げたが alpha が死んでいる」を見逃す |
| 6 | MEDIUM | ML責務 | SAC に「ボラティリティ推定値」まで持たせるのは初手として重い。**まずは bounded な directional_bias / aggressiveness 補助に留める方がよい** | RL出力は解釈しやすく audit しやすいほど実運用に向く。vol推定はルール/監督学習で分離しやすい |
| 7 | MEDIUM | 見落とし | `427#` は 422-425 で出ていた **reward 設計問題 / corr paradox** をほぼ踏まえていない | architecture を sidecar 化しても、reward が long-OOS でズレる問題は別途残る |
| 8 | MEDIUM | 既存資産 | walk-forward は新規実装しなくてよい。`ztb` と `scripts/v460` に再利用資産がある | `ztb/evaluation/walk_forward/splitter.py:1`, `scripts/v460/ml/walk_forward_as.py:90`, `ztb/trading/strategies/action_signal_guide/components/sac_integration.py:105` |

---

## 2. 427# で良かった点

### 2.1 「完全 End-to-End より分離型へ寄せる」は方向として正しい

これは支持できます。v460 の現在地では、SAC に発注価格・数量・offset を丸ごと握らせるより、

- 方向バイアス
- 攻撃性の微修正
- 参加見送りの補助

のような **限定責務** に落とす方が現実的です。

特に `fill_test` の本丸は execution quality なので、

- alpha の当たり外れ
- queue / fill / timeout / AS
- ルールベースの安全装置

を分離した方が、どこで損しているかが見えます。

### 2.2 Final Clamp を最後段に置く思想は正しい

これもその通りです。しかも思想だけでなく、既に実装も入っています。

- executor 側の最終 clamp: `scripts/v460/lib/fill_cycle_executor.py:742`
- 設定定義: `scripts/v460/lib/fill_config.py:324`

実ログでも `2026-03-14`〜`2026-03-15` に clamp 発火が継続しており、

- buy `0.23-0.40 -> 0.20`
- sell `0.8339 -> 0.50`

のように、今も必要な防波堤として機能しています。

---

## 3. 強く補正したい点

### 3.1 「文化圏で設計を選ぶ」は比喩に留めた方がよい

`427# §2` は面白いのですが、設計会議の根拠としては粗いです。

特に次の書き方は補正した方が安全です。

- 「米国型 = 完全 End-to-End SAC」
- 「中韓型 = Rigid Clamp 文化」
- 「日本型 = LightGBM + 分離」

これらは方向感の比喩にはなりますが、実際の設計はもっと混ざります。  
ここを強く書き過ぎると、**国別ステレオタイプで技術選択を正当化する文章** になりやすいです。

428の立場では、ここは次のように扱うのがよいです。

- `§2` は **Appendix / inspiration** へ降格
- 本文の判断基準は **市場構造・保守性・観測性・責務分離** に置く

### 3.2 427# の Phase 1/2/3 は「実装計画」ではなく「統合・検証計画」

現状を見ると、`427#` の各フェーズはかなりの部分が既にあります。

- Final Clamp: 実装済み、ログ発火あり
- Sidecar 型: 実装済み
- Sidecar 注入配線: 実装済み
- stale safe read: 実装済み

一方で不足しているのは次です。

1. sidecar signal の鮮度
2. sidecar が本当に fill_test に効いた証跡
3. clamp が「安全」だけでなく「収益改善」にどう関与したか

したがって、`427#` のフェーズ名も次のように直すと実態に合います。

- Phase 1: `Final Clamp Coverage Audit`
- Phase 2: `Sidecar Live Presence & Efficacy Proof`
- Phase 3: `Robust Evaluation Harness`

### 3.3 Sidecar は今、構想よりも「不在証明」が先

これが今回の最重要補正です。

現時点の runtime では:

- `cache/sidecar_signal.json` は存在するが、`2026-03-11` の neutral で停止
- `fill_test.log` には `Sidecar signal stale` が継続出力
- 直近 `fill_records_20260309-20260314.jsonl` 合計 `2467` 行で `sidecar_offset_bps` / `sidecar_bias` 非 null は `0`

つまり、今の v460 にとって Sidecar は

- 「導入を検討している新機能」でもなく
- 「live で効いている機能」でもなく
- **配線済みだが実質 inert な機能**

です。

この状態では、Sidecar の良し悪しを市場理論で論じる前に、

1. signal freshness
2. non-neutral push
3. fill_record 反映
4. 条件付きPnL

の 4 点を先に埋める必要があります。

---

## 4. 市場理論からの補強

### 4.1 v460 の問題は「国の文化」より「maker microstructure の責務分割」

Coincheck BTC/JPY の maker 系で効く論点は、文化圏より以下です。

- spread が十分あるか
- timeout が増えていないか
- adverse selection が増えていないか
- inventory repair と alpha 参加が混ざっていないか
- clamp 発火時に実質どの市場状態だったか

つまり、重要なのは

- 誰がどの国で好むアーキテクチャか

ではなく、

- **どの責務を ML に渡すと edge を増やし、どの責務を渡すと execution を壊すか**

です。

この意味で、`427#` の Sidecar 方向は市場理論とも整合します。  
ただし根拠は「日本っぽいから」ではなく、**maker 執行では価格・数量・停止条件を ML に丸投げしない方が良いから** です。

### 4.2 Sidecar の first target は「方向予測」より「参加質の改善」

SAC を繋ぐなら、いきなり「未来価格を当てる主役」にするより、次の問いに答えさせる方が勝ちやすいです。

- 今は少し aggress してよいか
- 今は一段保守的にすべきか
- 今は参加しない方がよいか

これは 1 分 maker の現実に合っています。  
逆に「ボラティリティ推定値」まで actor 出力へ積むと、

- 解釈が曖昧
- reward との対応が弱い
- audit が難しい

ので、初手としては重いです。

---

## 5. 設計面からの提案

### 5.1 全体方針は「3層分離」で決めるとぶれにくい

427# はここを文章で言っているのですが、図式として固定した方が今後の判断が楽です。

1. **Alpha / Meta layer**  
   directional bias, skip/participation, aggressiveness hint
2. **Execution layer**  
   maker price, lot, timeout, side switch, inventory repair
3. **Safety / Audit layer**  
   final clamp, hard skip, guard, observability, post-mortem

この3層で考えると、SAC は当面 `1` に限定するのが自然です。  
`2` と `3` はルール側に残した方が事故を減らせます。

### 5.2 方針判断の基準を文化論ではなく scorecard 化する

今後どの方式を採るかは、次の5軸で判定すると実務的です。

| 軸 | 問い | 使い方 |
|---|---|---|
| Profit leverage | それは本当に bps 改善余地が大きいか | 小さいなら後回し |
| Controllability | 壊れた時に clamp/kill で抑え込めるか | 抑え込めないなら危険 |
| Observability | pre/post の差分を記録できるか | 記録不能なら本番に乗せない |
| Data parity | train と live で同じ情報が使えるか | 乖離が大きいなら sidecar止まり |
| Maintenance cost | 小規模運用で回せるか | 運用できないなら棄却 |

この scorecard で見ると、現時点のおすすめ順はこうです。

1. ルールベース execution + clamp 維持
2. SAC sidecar を directional bias / aggressiveness 補助として限定導入
3. full autonomous driver は hold

### 5.3 walk-forward は既存資産を使えばよい

427# の Phase 3 は正しいですが、新規発明は不要です。

既存資産:

- `ztb/evaluation/walk_forward/splitter.py:1`
- `scripts/v460/ml/walk_forward_as.py:90`
- `ztb/analysis/evaluation/walk_forward_integration_pipeline.py:58`

つまり、評価基盤は「作る」より **つなぐ** 方が正解です。

### 5.4 `ztb` の SAC 統合資産も活かせる

`ztb/trading/strategies/action_signal_guide/components/sac_integration.py:105` には、

- action alignment
- confidence correlation
- timing alignment
- market alignment

を統合する `SACSignalValidator` があります。  
これはそのまま使えなくても、**sidecar の有効性監査指標** として発想を流用できます。

---

## 6. 427# で抜けている重要論点

### 6.1 reward 問題は architecture 変更だけでは消えない

422-425 で出ていた本丸は、

- `val_ratio` 交絡
- single-tail holdout
- 5K step checkpoint proxy
- reward-profit corr paradox

でした。427# はここをかなり薄く扱っています。

つまり、Sidecar へ寄せるのは良いとしても、

- 何を reward で学ばせるのか
- long OOS で何が崩れたのか
- reward が profitable regime と non-profitable regime をどう混同したのか

は残ります。

### 6.2 Final Clamp の次は「Clamp-Driven Development」を避ける

Clamp は必要です。  
ただし、clamp が頻発する構造になると、開発が

- upstream の失敗を clamp で吸収
- そのまま運用継続
- 本質原因の修正が後回し

になりがちです。

これを防ぐには、最低でも以下を取るべきです。

- `clamp_fire_rate`
- `hard_skip_rate`
- `pre_clamp_offset` 分布
- `clamp fired` 条件付きの fill / pnl / AS

### 6.3 SAC 以外の別解も残すべき

ユーザーが歓迎している「範囲外の改善」としては、Sidecar の役割自体を RL 以外で置き換える案も残した方がよいです。

候補:

1. **Meta-labeling / contextual bandit** で参加可否だけ学習
2. 監督学習で toxicity / volatility / fill-prob を別々に推定
3. それらをルール executor へ統合し、RL は使わない

1分 maker の duty が限定的なら、こちらの方が保守しやすい可能性があります。

---

## 7. 推奨アクション順

### P0: すぐやる

1. **427# の文書位置づけを「実装計画」から「統合・検証計画」へ修正**
2. **Sidecar live presence を証明** する  
   `signal fresh` / `non-neutral` / `fill_records non-null` / `log trace`
3. **Clamp observability を定例指標化** する  
   `clamp_fire_rate`, `hard_skip_rate`, `pre_clamp_offset`
4. **walk-forward は既存資産流用で着手** する

### P1: Sidecar の責務を固定

1. SAC 出力はまず `directional_bias` + `small bounded boost` に限定
2. `volatility estimate` はルール or 監督学習へ分離
3. Sidecar の KPI を `fill_rate`, `post_fill_30s_pnl`, `AS`, `timeout` に固定

### P2: その後の分岐

- Sidecar が有効なら: bias / aggressiveness の2軸へ拡張
- Sidecar が無効なら: RL に固執せず meta-labeling / supervised overlay へ切替

### HOLD

- 国別文化論を主根拠にしたアーキテクチャ選択
- full end-to-end SAC への回帰
- Sidecar に価格・数量・停止条件まで持たせる拡張
- bare-metal / 自作 engine 系の大工事

---

## 8. 結論

427# の核は悪くありません。  
**「SAC を主役から降ろし、execution と safety を分離する」** という方向は、v460 の現在地にかなり合っています。

ただし、今の実態に合わせるなら結論は次の形に直すのが正確です。

> **v460 の次方針は「Sidecar / Final Clamp を新規導入すること」ではない。**  
> **既に入っている Sidecar / Final Clamp を、live に存在し、効いており、利益に寄与する構造として証明すること** である。

その上で、全体方針は文化論ではなく、

- microstructure 適合性
- 観測可能性
- 安全性
- profit leverage
- 保守コスト

の 5 軸で決めるのが一番ぶれません。
