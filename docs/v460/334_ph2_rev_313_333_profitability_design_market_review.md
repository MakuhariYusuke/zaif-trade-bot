# 334# 313#–333# 横断レビュー — 収益性最優先の再整理

> **種別**: rev  
> **対象**: 313#–333#  
> **起票**: 2026-03-08  
> **観点**: profitability-first / systems engineering / market theory  
> **補足確認**: `analysis/311_observational_rerun.py`, `scripts/v460/lib/maker_price.py`, `scripts/v460/lib/fill_config.py`, `scripts/v460/lib/fill_config_parser.py`, `scripts/v460/lib/fill_cycle_executor.py`, `configs/v460/fill_test.yaml`  
> **テスト**: `./.venv/Scripts/python.exe -m pytest tests/unit/v460 --no-cov` → **4105 passed**

---

## §1 結論

313#–333# は一時かなり「ratio セマンティクス論争」に流れたが、最終的には 333# が本丸をかなり明確にした。

**現時点の最重要論点は sell 側の追加防御ではなく、buy 側の行動空間が kill / delay / duty skip で潰れすぎていること**である。  
333# の `buy fill_rate = 9.3%`、`buy_dynamic_kill = 216`, `forced_buy_delay = 100`, `degraded_liquidation_duty_skip = 95` は、もはや観測上のノイズではなく構造問題と見てよい。

一方で、320#/321# の side-specific ceiling 修正、318# の none/unknown 修正、319#/316# の `mid_at_order` 導入は、**利益に直結する本物の修正**だった。ここは評価してよい。

総合すると、現段階でやるべきことは次の 3 点に収束する。

1. **構造変更をいったん止め、同一 SHA 条件で buy 側 suppressor 群を順に検証すること**
2. **sell 側は「まだ悪い」のではなく「閾値近辺まで戻った」ものとして、過剰反応しないこと**
3. **ranging でのみ見えている正の期待値を先に取り切り、trending 追随は別問題として切り分けること**

---

## §2 313#–333# で正しかった点

| 範囲 | 判定 | レビュアー所見 |
|---|---|---|
| 313#–315# | ✅ 妥当 | 旧 `spread_capture` / `AS cost` 分解の解釈は崩れていた。`effective_offset_used` を主指標に据えた評価は信用し過ぎない方がよい |
| 316#–321# | ✅ 非常に価値が高い | `mid_at_order`, none/unknown 修正, side-specific ceiling, YAML parse 修正は、理屈ではなく実害修正だった |
| 322#–332# | ✅ 方向性は正しい | God Object 分割は silent miswire 防止に効く。321# の YAML 未パース事故は、まさに分割不足が招いた事故だった |
| 333# | ✅ 現時点で最重要 | mixed-SHA ではなく同一 SHA 相当の 24h を切り出した点が最も重要。意思決定価値が最も高い |

特に 315# の「ceiling は正常に効いているが、`effective_offset_used` は post-processing 後なので価格位置の proxy としては危うい」という整理は重要だった。  
実際、現行 `analysis/311_observational_rerun.py` も old ratio 中心の見方から、`fill_price` と `mid_at_fill` ベースの距離を見る方向に寄っており、**議論は一応収束方向に向かっている**。

---

## §3 迷走していた点と、今は優先度を下げるべき点

### §3.1 ratio セマンティクス論争は、もう主戦場ではない

313#/314# の「boost の向きが理論的に逆ではないか」という問題提起は有益だった。  
ただし 315# 以降の整理と現行コードを見る限り、**今すぐ全 boost の符号をひっくり返すのは優先度が低い**。

理由は 3 つある。

1. `maker_price.py` 内の ceiling と `fill_cycle_executor.py` 側の post-ceiling 補正を分けて見ないと、議論がすぐ誤る
2. 320#/321# の side-specific ceiling 修正の方が、実利益にはるかに大きく効く
3. 333# が示した本丸は ratio の向きではなく **buy 側 suppressor の過剰発火** だから

### §3.2 319# の「当面 executor 側で吸収」は、今見ると短期逃げだった

319# 時点ではそれでも実務判断として理解できるが、320#/321# の後では評価を改めるべきである。  
**sell 側 ceiling を side-specific に分けた判断は正しかった**。  
ここを戻す理由はない。

### §3.3 333# の好結果を一般化し過ぎるのは危険

333# は有望だが、以下の制約が大きい。

| 項目 | 333# の状態 | 含意 |
|---|---|---|
| 期間 | 24h | Gate 判定には短い |
| fills | 100 | `p10` 判定にはまだ細い |
| ranging 比率 | 90.3% | 有利な相場に偏っている |
| trending fills | 10 | ほぼ未評価 |
| none | 0 | warmup / detector 異常が起きなかっただけで、頑健性の証明ではない |

したがって 333# は「**正の期待値の芽がある**」ことの証拠にはなるが、「**持続的に勝てる**」ことの証拠にはまだならない。

### §3.4 再現可能性はまだ弱い

`analysis/311_observational_rerun.py` はあるが、**333# に対応する専用スクリプトや JSON 出力が repo に見当たらない**。  
333# の結論自体は重要だが、再現性という意味では 311# より弱い。  
これは設計論というより **実験基盤の問題** である。

---

## §4 収益性ファーストで見た本丸

### §4.1 今の本丸は「buy 側の過剰抑制」

333# の skip/cancel 理由はかなり露骨である。

| 理由 | 件数 | 示唆 |
|---|---:|---|
| `buy_dynamic_kill` | 216 | buy を最も強く殺している主因 |
| `forced_buy_delay` | 100 | buy を hard skip 寄りに遅延 |
| `degraded_liquidation_duty_skip` | 95 | 本来は安全弁だが、参加不能要因になっている疑い |
| `balance_forced_switch` | 170 | サイクルの 26.7% で action space が歪んでいる |

`buy avg_pnl = +0.372`、`buy p10 = -3.819` は見た目だけなら悪くない。  
しかしこれは **「通した buy が優秀」ではなく「通した buy だけが生き残った」** 可能性が高い。  
つまり survivorship bias をかなり疑うべきである。

### §4.2 市場理論的には、二面参加を殺すと MM の源泉を失う

今の 333# は ranging 優勢日で利益が出ている。これは不自然ではない。  
むしろ passive maker は **ranging で bid/ask の両面参加を繰り返して稼ぐ** べきであり、そこで利益が見えたのは良い兆候である。

ただし、その環境で buy 側を kill し過ぎると何が起こるか。

1. 在庫修復が遅れる
2. 片側だけ参加する時間が増える
3. MM ではなく「条件付き片側参加システム」になる
4. 取れるはずの mean reversion を取り逃す

高頻度化以前に、**二面参加の品質が崩れている**。  
ここを直さずに取引頻度だけ上げても、毒を高速で食うだけになる。

### §4.3 sell 側は「再度大工事」より「監視対象」

sell 側は 333# で `avg_pnl +0.889`, `p10 -5.207`。  
これは良いとは言わないが、**完全な失敗ではなく閾値近辺まで戻している**。

ここで further defense を何層も積むより、優先順位は下がる。  
むしろ side-specific ceiling 修正後の長めデータで、

1. `fill_rate >= 30%` を維持できるか
2. `p10` が本当に -5bps 近辺に張り付くのか
3. `trending_up sell` が依然として壊れているのか

を見た方がよい。

---

## §5 設計レビュー

### §5.1 322#–332# の分割は概ね正しい

`maker_price.py`, `fill_cycle_executor.py`, `fill_loop_orchestrator.py`, `fill_config.py` の分割は、方向として正しい。  
保守性だけでなく、**設定漏れ・ parse 漏れ・ private state 直叩き** の事故を減らす効果がある。

326# と 331# の自己監査も健全で、単なる分割で終わらず、

- 型修正
- encapsulation 回復
- validation 追加
- early return による監視欠落修正

まで踏み込めている。

### §5.2 ただし「God Object の解体」と「複雑性の消滅」は別

注意点は、**複雑性が減ったのではなく、分散しただけの部分もある**ことだ。

例として 332# 後の `run_continuous` は短くなったが、実際の意思決定は複数 Mixin に広がっている。  
これは可読性改善には効く一方で、**状態遷移の追跡コスト** を上げる。

今後必要なのは追加分割ではなく、むしろ次の 2 つである。

1. `pre_cycle -> balance -> mid_cycle -> execute -> post_cycle` の責務地図を 1 枚に固定すること
2. どのフィールドをどの層が更新してよいかを明文化すること

### §5.3 小さな設計リスク

| 項目 | 所見 |
|---|---|
| default/YAML drift | `unknown_regime_max_consecutive` はコード既定値と YAML 実値がずれており、将来の programmatic config で事故になりうる |
| source-inspection tests | 分割耐性は上がるが、設計改善のたびにテストが文字列依存で追随するのはやや重い |
| 333# 再現基盤 | 結論は重要だが script 化されておらず、追試しづらい |

---

## §6 市場理論レビュー

### §6.1 333# は「ranging では戦える」証拠

333# の利益の 97% が ranging 由来というのは、むしろ自然である。  
受動的 MM が一番勝ちやすいのは、方向感が弱く、往復し、スプレッドを取りやすい局面だからである。

従って、「ranging で勝っているのは偏りだからダメ」とは言わない。  
むしろ **そこが本来の収益源** である。

### §6.2 ただし「大値動きを取りたい」は別戦略

ユーザーの問題意識として「大値動きに乗れない」「もっと儲けたい」は正しい。  
ただし、それを今の passive quoting 系ロジックだけで解決しようとすると無理が出る。

受動的 MM と trend capture は、要求する行動がかなり違う。

| 目的 | 望ましい行動 |
|---|---|
| passive MM | 毒を避けつつ両面参加し、細かく回す |
| trend capture | 片側へ寄せる、追いかける、場合によっては受動性を捨てる |

したがって、「高頻度で儲ける」を本気で狙うなら、まずは

1. ranging 用 MM を強くする
2. trending は回避するか、別 policy に分ける

の二段構えにした方がよい。  
**一つの passive maker 設定で両方を取ろうとするのが、ここまでの迷走の一因** である。

### §6.3 高頻度化それ自体は正義ではない

高頻度であることは、正の期待値があるときだけ意味がある。  
今の段階で本当に必要なのは HFT 化ではなく、

- 行動空間の無駄な抑制を外す
- toxic session を避ける
- 勝てる局面で二面参加できるようにする

ことである。

---

## §7 次にやるべきこと

### P0

1. **buy_dynamic_kill の限定的緩和実験**  
   global 緩和ではなく、`ranging` かつ spread/volatility 条件を満たす区間だけで 1 ノッチずつ緩めるべき。

2. **forced_buy_delay を hard skip から「まずは守備的 quote」に落とす実験**  
   最初の数サイクルだけは skip ではなく offset 拡大で参加させる価値がある。

3. **degraded_liquidation_duty_skip の分解**  
   本当に liquidation duty なのか、単なる保守過剰なのかを分けるべき。  
   inventory repair の buy まで同じ理由で殺しているなら設計ミス寄り。

4. **同一 SHA 条件を固定して 168h 蓄積**  
   これ以上の大きな refactor は、明確なバグ修正以外はいったん止めた方がよい。

### P1

5. **333# 相当の分析を script + JSON 出力に昇格**  
   document だけでなく、`analysis/333_*.py` と `analysis_results/333_*.json` 相当の再現資産を残すべき。

6. **buy を alpha / repair に分離して観測**  
   今の buy kill が「攻めの buy」を殺しているのか、「在庫修復 buy」まで殺しているのかで結論が変わる。

7. **sell は監視継続、追加工事は保留**  
   `n>=200` 程度の同一条件データで再び `p10 < -5` が続くなら手を入れる。今は buy より優先度が低い。

### P2

8. **decision flow の SSOT 作成**  
   322#–332# の分割は悪くないが、責務境界が文書化されていないと再び混乱する。

9. **metric glossary の固定**  
   `effective_offset_used`, `mid_at_order`, `mid_at_fill`, `mid_distance_bps`, `decision_path` の意味を 1 枚に固定した方がよい。

---

## §8 レビュアー判断の最終要約

313#–333# の流れは、遠回りはあったが無駄ではなかった。  
**解析の誤り修正 → 実害修正 → 構造整理 → 同一 SHA 検証** という順に、最低限の土台は整ってきている。

ただし今後は、次の線引きが重要である。

- **儲けに直結する変更**: buy 側 suppressor 群、same-SHA 長期観測、再現可能な分析基盤
- **今は後回しでよい変更**: 追加の大規模分割、ratio 理論の再論争、sell 側のさらなる複雑防御

厳しめに言えば、ここから先は「何でも改善する」段階ではない。  
**「何を触ると本当に利益に効くか」をかなり狭く絞る段階**である。  
333# まで来た今、その絞り込み先は buy 側の過剰抑制と、ranging 優位を本当に取り切れているかの検証だと判断する。
