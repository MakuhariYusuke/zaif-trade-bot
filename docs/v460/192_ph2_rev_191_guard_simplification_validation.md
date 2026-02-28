# 192# 191レビュー: Guard複雑性分析の検証 + 根本簡素化方針

> **目的**: 191# の主張を実ログと現行コードで照合し、複雑化の真因を切り分ける。  
> **結論**: 191# の「`ev_weighted` が最大 blocker」という認識は概ね正しい。だが、根本問題はガード数そのものではなく、**同じ「取引するか否か」の責務が複数層に分散し、hard block / threshold補正 / 執行拒否として重複していること**にある。  
> **方針**: `ev_weighted` の単純無効化は止血としては有効。ただし本命は「判定責務の整理」であり、そこを外すと別の guard が次の詰まり箇所になる。

---

## 1. 実ログによる事実確認

### 1.1 191# の主要数値の検証

`results/v460/fill_test/logs/fill_test.log` を `2026-02-28 13:03:00–15:54:59 JST` で直接集計した結果:

| 指標 | 実測 | 191# 記載 | 判定 |
|---|---:|---:|---|
| Cycle 開始 | 69 | 70 | ほぼ一致 |
| Fill | 12 | 12 | 一致 |
| `ev_weighted_skip` (buy) | 11 | 11 | 一致 |
| `ev_weighted_skip` (sell) | 16 | 16 | 一致 |
| `velocity_buy_skip` | 3 | 3 | 一致 |
| `velocity_sell_skip` | 8 | 8 | 一致 |
| `Spread too narrow < 1000` | 8 | 8 | 一致 |
| `ev_weighted` safety valve | 2 | 2 | 一致 |
| `stale_order` reprice | 4 | 4 | 一致 |
| `balance_forced but one_sided_balance` | 28 | 77 events | 指標定義が異なる |

**判断**:
- `191#` の中核主張、すなわち **`ev_weighted` が最大の blocking 要因**という点は、ログで裏付けられる。
- `70 cycles` は、timestamp 境界の取り方次第で `69` になる。ここは重大な誤りではないが、**次回以降は time range に加えて cycle range を明記すべき**。
- `balance_insufficient 77 events` は、今回確認した `balance_forced but one_sided_balance` 28 件とは別定義。`balance_checker` 警告や side switch も含めた合算である可能性が高い。**ログ断片だけでは再現不能なので、集計条件の明記が必要**。

### 1.2 再現性に関する注意

現時点の `results/v460/fill_test/fill_test_state.json` はすでに次 run に進んでおり、`run_id=1772251439_5604f769`。  
そのため、191# の point-in-time 集計は **現在の `fill_records_20260228.jsonl` 全体集計だけでは再現できない**。

現行 run の上位 cancel reason は以下:

| cancel_reason | 件数 |
|---|---:|
| `skip_gate` | 31 |
| `ranging_low_vol_skip` | 12 |
| `filled` | 15 |
| `spread_too_narrow` | 10 |
| `skip_gate_rule_velocity_sell` | 10 |

**含意**:
- `ev_weighted` を切っても、それだけで複雑性問題は終わらない。  
- 現在も `ranging_low_vol_skip` / `spread_too_narrow` / `velocity_sell_skip` が十分大きい。  
- よって `191# S1` は有効だが、**構造問題の本丸ではない**。

---

## 2. 191# の妥当な指摘

### 2.1 `ev_weighted` は「第2 SkipGate」化している

`scripts/v460/lib/skip_gate_evaluator.py` では、`_try_ev_weighted_decision()` が以下を一箇所で担っている。

1. alt horizon モデル評価
2. `pnl30/pnl120` の加重合成
3. threshold 判定
4. `one_sided_balance` 時の threshold 緩和
5. consecutive skip safety valve
6. 最終 `SkipDecision` の reason/model 生成

これは **単一責務原則に反する**。  
`ev_weighted` は単なる「補助スコア」ではなく、**独立した hard gate** として振る舞っており、しかも安全弁まで内包している。

### 2.2 `stale_order` の追随 reprice は、複雑性だけでなく損益的にも危険

191# が挙げた `stale_order reprice` の懸念は妥当。  
実ログでも `2026-02-28 15:54` 台に sell で 2 回 reprice が発生しており、**「待つべき注文」と「追ってはいけない注文」の線引きが現状は回数制限だけに寄っている**。

### 2.3 ファイル肥大の認識は大筋正しい

現行行数 (`wc -l`) は以下:

| ファイル | 行数 |
|---|---:|
| `scripts/v460/lib/fill_loop_orchestrator.py` | 1308 |
| `scripts/v460/lib/fill_config.py` | 1162 |
| `scripts/v460/lib/skip_gate_evaluator.py` | 967 |
| `scripts/v460/lib/fill_cycle_executor.py` | 828 |
| `scripts/v460/lib/maker_price.py` | 761 |

`191#` の数値は `fill_config.py` が 1 行差、`maker_price.py` が 1 行差で、実務上は誤差の範囲。  
認識すべき本質は、**大きいこと自体より「判定責務が横断して散っていること」**。

---

## 3. 191# がまだ捉え切れていない根本原因

### 3.1 真の複雑化源は「guard の分散所有」

現在、同じ「この cycle で注文すべきか」の判断が少なくとも 4 箇所に分散している。

| 層 | 主ファイル | 役割 |
|---|---|---|
| Layer A | `scripts/v460/lib/fill_loop_orchestrator.py` | balance/rescue/regime/dynamic kill |
| Layer B | `scripts/v460/lib/fill_cycle_executor.py` | spread pause / min spread / placement前 reject |
| Layer C | `scripts/v460/lib/maker_price.py` | price算出内 sell_guard / spread妥当性 |
| Layer D | `scripts/v460/lib/skip_gate_evaluator.py` | velocity / hour offset / narrow spread strict化 / ML / ev_weighted |

問題は guard 数ではなく、**同じ概念が別のレイヤーで別の形式で再評価される**こと。

具体例:
- spread は `maker_price.py` で拒否される
- 同時に `fill_cycle_executor.py` でも `min_spread_jpy` / `narrow_spread_pause` が走る
- さらに `skip_gate_evaluator.py` でも `narrow_spread_offset` として threshold 厳格化される

これは **同一ドメイン概念に対する三重管理** であり、保守コストと挙動の非直感性を生む。

### 3.2 stateful override が非局所的な挙動を作っている

挙動を読みにくくしている主因は、単なる if 文の多さではなく、以下のような状態変数が層をまたいで効く点にある。

- `_balance_forced`
- `one_sided_balance`
- `_ev_consecutive_skip_count`
- `_trending_sell_skip_count`
- `_narrow_spread_consecutive`

このため、エージェントは「同じ market 状況でも直前の数 cycle によって別行動を取る」。  
これ自体は悪ではないが、**状態の持ち主が散っているため因果追跡が難しい**。

### 3.3 「なぜその行動を取ったのか」が一意に残っていない

現状のログは豊富だが、判定理由が以下のように分散している。

- 早期 return 型 (`velocity_skip`)
- threshold 補正型 (`hour_offset`, `narrow_spread_offset`)
- safety valve 解除型 (`ev_weighted`)
- 執行時 reject 型 (`sell_guard_reject`, `spread_too_narrow`)

この構造だと、**最終的に filled した注文ですら「どの guard を通過し、どの guard が緩和されたか」が一目で分からない**。  
ここが「複雑化して見える」最大の要因。

---

## 4. 191# の S1-S4 に対する再評価

### 4.1 S1: `ev_weighted_enabled: false`

**評価**: 有効。ただし「診断用の止血策」であって、根治策ではない。

理由:
- 現時点のログでは `ev_weighted_skip=27` と最大 blocker
- まず外して基準線を取り直す価値はある
- ただし外した瞬間、次は `ranging_low_vol_skip` / `velocity_sell_skip` / `spread_too_narrow` が前面化する

**結論**:
- 実施するなら賛成
- ただし **「複雑性削減」とは呼ばない方が良い**
- 位置づけは「dominant blocker の一時切り離し」

### 4.2 S2: velocity 閾値緩和

**評価**: 収益検証としては妥当。だが、構造簡素化にはほぼ寄与しない。

閾値が `6.0 -> 10.0` になっても、`rule_velocity_*` という **別 hard gate が残る構造自体は変わらない**。  
複雑性を減らすなら、閾値変更より先に「これは hard skip であるべきか、それとも offset 強化に降格すべきか」を決めるべき。

### 4.3 S3: B1' low-vol 緩和

**評価**: これも挙動調整であって簡素化ではない。

`low_vol_threshold` を下げても、guard の責務配置は変わらない。  
むしろ `B1'` と `SkipGate` の役割境界が曖昧なまま、通過量だけ増える。

### 4.4 S4: reprice 最大回数 1

**評価**: 実害軽減としては妥当。だが本筋ではない。

複雑性を本当に減らすなら、回数制限より先に:

1. 「追随 reprice を許す条件」
2. 「cancel-only に切り替える条件」
3. 「そもそも stale にしないための再価格戦略」

を分けるべき。  
単なる `2 -> 1` は傷を浅くするが、意思決定構造はそのまま残る。

---

## 5. 根本解決に近い簡素化方針

### 5.1 まず「hard blocker」と「soft modifier」を分離する

現在は同じ `skip` 系の文脈に、性質の違うものが混在している。  
最低限、以下の 2 類型に分けるべき。

| 類型 | 例 | あるべき振る舞い |
|---|---|---|
| **Hard Blocker** | 残高不足 / Exchange reject / `min_spread_jpy` | 即中止して良い |
| **Soft Modifier** | `hour_offset` / `narrow_spread_offset` / `ev_weighted` / velocity警戒 | 価格・threshold を補正し、原則として単独では即 skip しない |

現状の最大問題は、`ev_weighted` と `velocity` が **soft にすべき局面でも hard block として使われている**こと。

### 5.2 `ev_weighted` を「gate」から「補正器」に降格する

最小コストで効く本命はこれ。

`_try_ev_weighted_decision()` の責務を次のように縮小する:

1. alt horizon を評価
2. `ev_score` を計算
3. `delta_threshold` または `delta_offset` を返す
4. 最終 PASS/SKIP は既存 primary gate が一元決定

つまり:
- **現状**: `ev_weighted` 自身が PASS/SKIP を確定する
- **修正後**: `ev_weighted` は「厳しくする/緩める」補助信号に留める

これで以下を同時に解消できる。

- safety valve という二重例外が不要になる
- `one_sided_balance` 緩和の置き場を一本化しやすい
- 「通したのは primary なのか ev_weighted なのか」が曖昧でなくなる

### 5.3 spread 系 guard は一箇所を親にする

spread は、少なくとも次の 3 系統に分けて整理すべき。

1. **市場品質の下限**: `min_spread_jpy`
2. **価格 aggressiveness 補正**: `narrow_spread_boost` / `narrow_spread_offset`
3. **一時停止ロジック**: `narrow_spread_pause`

このうち、即 skip 権限を持つ親を 1 箇所に固定しない限り、同じ spread 条件が複数理由で弾かれ続ける。

### 5.4 `guard_trace` を「filled 時にも」必ず残す

191# の分析が成立したのはログが多いからであって、構造が明快だからではない。  
次にやるべきは、各 cycle について以下を構造化して残すこと。

- どの guard を通過したか
- どの guard が threshold を変更したか
- どの guard が bypass / safety / rescue したか
- 最終決定者は誰か

`187#` で `guard_trace` 記録は始まっているが、**filled 時の意思決定経路の統一記録**としてはまだ足りない。  
「skip 時だけ理由が残る」状態をやめるべき。

---

## 6. 実装順の提案（低リスク順）

### 6.1 P0: 今すぐやるべきこと

1. `191# S1` を診断用に実施し、`ev_weighted` を一時停止
2. 同時に `run_id` / `cycle_start-end` を固定して比較ログを取る
3. `skip_gate`, `ranging_low_vol_skip`, `spread_too_narrow`, `rule_velocity_sell` の順位変化を確認

これで「`ev_weighted` を外した後の真の主 blocker」が見える。

### 6.2 P1: 次にやるべき最小リファクタ

1. `scripts/v460/lib/skip_gate_evaluator.py` から `EvWeightedPolicy` を抽出
2. `ev_weighted` を PASS/SKIP 判定者ではなく threshold 補正器に変更
3. safety valve を `ev_weighted` 内から撤去し、必要なら上位で統一管理

これは **挙動変更を最小限に抑えながら、責務だけを正す** リファクタとして成立する。

### 6.3 P2: その後にやるべき整理

1. spread 系 guard の所有者を 1 箇所に寄せる
2. `velocity_skip` を hard skip から soft strictness へ落とせるか再評価
3. `balance_forced` / `one_sided_balance` の救済条件を 1 箇所に集約

ここまで進めて初めて、「複雑性を減らした」と言える。

---

## 7. 既存の再利用候補

複雑性削減の進め方として、既存の整理実績は再利用価値が高い。

| 既存資料 | 再利用価値 |
|---|---|
| `docs/v460/080_phg_dedup_and_inheritance.md` | 重複排除・継承/共通化の観点 |
| `docs/v460/113_ph2_impl_resilience_r1_split.md` | God Method 分割の実績 |
| `docs/v460/153_ph2_refactor_test_stabilization.md` | runner 分割設計の雛形 |
| `docs/v460/161_phg_impl_code_quality_and_structural_improvements.md` | DRY 統合・責務分離の直近実績 |

今回の論点は新規発明ではなく、**すでにこのリポジトリで一度うまくやった整理を、guard 周辺へ横展開するだけ**とも言える。

---

## 8. 最終判断

191# は、**「どこが一番詰まっているか」を見つける分析としては有効**。  
一方で、S1-S4 だけでは「詰まりの順番を入れ替える」可能性が高く、**過度な複雑化の根本是正には届かない**。

本件の本質は次の一文に尽きる。

> **問題は guard が多いことではない。guard の判定権限が分散し、しかも同じ概念を別レイヤーで重複評価していることが問題である。**

従って、次の一手は:

1. まず `ev_weighted` を一時停止して真の詰まり順を再測定  
2. その後 `ev_weighted` を hard gate から補正器へ降格  
3. spread / balance / rescue の ownership を整理

この順が最も安全で、かつ根本解決に近い。

---

## 9. 追記: 192# に対するセカンドオピニオンと「過保護化の解体方針」 (Gemini 3.1 Pro)

### 9.1 Codexの分析への同意と「思い込み」の指摘
Codexの**「同じ概念が別のレイヤーで重複評価されている（§3.1）」という指摘は完全に同意**する。現在のアーキテクチャは「何か新しい危険に気づくたびに、新しい層にif文を増築した」結果生み出されたフランケンシュタインのような過保護システムである。
一方で、Codexが「`ev_weighted` を一時停止してから真の詰まり順を再測定せよ（§6.1）」と述べている点は、**研究者的で悠長すぎる思い込み**である。「どこが一番詰まっているか」を順番に探す時間は無駄だ。直近ログを見れば明らかなように、`ev_weighted` がブロックの主因であり、さらに**残高不足（balance_insufficient）による片側強制（one_sided_balance）**がそれに拍車をかけている。

### 9.2 実ログが示す「もう一つの狂気」 — 残高ガードの無限ループ
直近の稼働ログ（`2026-02-28 16:04` 付近）を直接確認したところ、以下のような**無駄な無限ループ**が起きている。
1. JPYが枯渇している（`1718 < min 10046`）ため、Buyがスキップされる。
2. 強制的にSellにスイッチする（`freeze buy for 3 cycles`）。
3. Sellに行くと `ev_weighted` が常時負スコアを出してSkipする（`score=-2.025`）。
4. 1に戻る。
この無限ループを抜ける唯一の道は「190#で実装したsafety valve（5連続Skipで強制PASS）」のみである。つまり、今のシステムは**「ガードが全てをブロックし、安全弁（バグ技のような例外）で無理やり取引をこじ開けている」**という狂気の沙汰である。

### 9.3 「ev_weighted」は悪か？ — 評価軸の誤謬
`ev_weighted` が全て負の予測（-0.1 ～ -3.8 bps）を出していることを「悪」とし、これを排除・降格（補正器化）しようとする流れがあるが、これは危険な認識である。
機械学習系モデル（ev_weighted）が常にマイナスを予測するということは、**「いま現在の相場とoffset設定では、Makerとしてどうやっても勝てない（エッジがない）」**とシステムが悲鳴を上げているのである。それを「ガードが厳しすぎる」とみなし、安全弁で無理やりPASSさせる（190#）のは、温度計がマイナスを指している時に「温度計が壊れている」と言って叩き割るのと同じである。

### 9.4 本当の「根本簡素化」と収益への道（ネクストアクション）
「複雑なガードを整理して、コードを綺麗にする」ことは目的ではない。目的は「勝てる局面でのみ取引し、儲ける」ことである。

1. **Safety Valve（190#）の即時撤廃**: 5連続Skipで強制PASSさせる機能は、負け戦に無理やり突撃させる自殺行為。即座に消せ。
2. **ev_weighted を「執行の全権」から外し、「Offsetの根拠」に格上げする**:
   Codexの§5.2に一部同意するが、補正器に降格させるのではない。**「EVがマイナスなら、EVがプラスになるまでOffsetを意図的に広げる（価格を奥に置く）」**という物理的な価格調整の源泉として使うべきである。「勝てないなら、勝てる価格まで引いて待つ」のがMakerの王道である。
3. **Guardの階層一元化（即時実行）**:
   「Spread」と「Velocity」のガード判定を全て `skip_gate_evaluator.py` の**外部（手前）**に引き剥がし、「ルール層（Run/Halt）」と「ML予測層（EV）」の2層のみに物理分割せよ。
4. **在庫Skewの容認（JPY枯渇の解消）**:
   片側残高枯渇による無限ループを防ぐため、トレンド方向に対しては残高限界までポジションを傾ける（Inventory Skew）ことを許可し、一方的な相場での機会損失を防げ。
