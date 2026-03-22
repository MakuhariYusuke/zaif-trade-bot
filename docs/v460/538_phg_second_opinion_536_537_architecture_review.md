# 538# 536#-537# セカンドオピニオン — 易占由来の方向修正を設計・市場理論・実装現況で再検証

> 更新: 2026-03-22
> 対象: 536#, 537#, `configs/v460/fill_test.yaml`, `scripts/v460/lib/pre_order_adjustments.py`, `scripts/v460/lib/cycle_gate_aggregator.py`, `scripts/v460/lib/sidecar_types.py`, 関連実装
> 注: 本レビューは**易占そのもの**の当否ではなく、536#/537# がそこから引いた設計判断を、現行コード・設定・市場理論で検証したものである。

## 0. 総評

536# は、硬直化した議論をほどくための**発想転換の文書**としては有益である。537# はそれを実装可能な論点へ落とし直しており、その意味では 2 文書はうまく役割分担できている。

ただし、セカンドオピニオンとして強く言いたいのは次の 3 点である。

1. **536# は比喩としては良いが、根拠文書としては使い過ぎない方がよい**
2. **537# はかなり良い翻訳だが、現行 config / code とズレた前提が混ざっている**
3. **両文書とも「簡素化」の成功条件をまだ定義していない**

つまり、今回の主題は「易占を信じるか」ではない。**設計の何を減らし、何を中核として残すかを、測定可能な形に翻訳できているか**である。

---

## 1. Findings

| # | 重要度 | 対象 | 指摘 | 推奨対応 |
|---|---|---|---|---|
| 1 | HIGH | `docs/v460/537_phg_review_536_architecture_simplification.md`, `configs/v460/fill_test.yaml`, `scripts/v460/lib/sidecar_types.py` | 537# には現行状態とズレた前提が混ざっている。`min_spread_jpy` は既に `500`、`composite_risk_enabled` は既に `true`、sidecar TTL は既に `7800s` | 537# を backlog 化する前に「すでに適用済 / 未適用 / 再校正のみ必要」を 3 列で棚卸しする |
| 2 | HIGH | `docs/v460/537_phg_review_536_architecture_simplification.md`, `scripts/v460/lib/pre_order_adjustments.py`, `configs/v460/fill_test.yaml` | 537# の `buy:0.30, sell:0.40` を「buy を aggressive」と説明する箇所は、オフセットの意味論と噛み合っていない。現行の multiplier は通常、`>1` で mid から遠ざけるため、ceiling を上げるほど保守化する | 非対称化するなら、まず「高オフセット=より保守的」という意味を固定し、buy/sell の aggressiveness は **価格方向ではなく offset 距離**で定義する |
| 3 | HIGH | 536# シナリオA/C, 537# Phase 0/1 | 両文書は clamp 飽和を正しく問題視しているが、`ceiling 0.25→0.35` を先にやると upstream の multiplier 爆発を広い価格帯でそのまま通す危険がある | 先に stage 別の膨張源を抑える。少なくとも `pre-clamp distribution` と `stage contribution` を見てから ceiling を動かす |
| 4 | MEDIUM | 536# §1-3, 537# §4-5 | 両文書とも「固定ルール vs ML」に議論を寄せすぎている。実際にはその中間、すなわち**低次元の learned calibration + hard invariants**が抜けている | ルール全面撤去でも ML 丸投げでもなく、少数ノブを学習で較正する設計を別軸で置く |
| 5 | MEDIUM | 536# §3, 537# §6 | `Stage Max Mult は1行`, `OFI-Lite は低コスト`, `kill 統合はすぐ` という表現はやや楽観的。実際には config, hot reload, observability, tests, state semantics まで触る | 工数表記を「ロジック変更量」ではなく「影響面」で見積もる。小改修に見える構造変更を甘く見ない |
| 6 | MEDIUM | 536# 全体, 537# §7 | 「時間帯ハードコードは過学習だから捨てる」は半分正しいが、時間帯は人間活動・取引所間フロー・流動性供給者行動の proxy でもある。完全撤去は早い | 時間帯ルールは hard block から降ろし、まず feature / weight / prior として残す |
| 7 | MEDIUM | 537# §4, `scripts/v460/lib/cycle_gate_aggregator.py` | 537# は `composite_risk` を簡素化の切り札寄りに置くが、これは gate を減らすのでなく**別の集約 gate を上に積む**設計でもある。使い方を誤ると不透明さが増える | simplification を目指すなら、`composite_risk` は他 soft gate を置換する前提で使い、単なる上乗せにしない |
| 8 | LOW | 536#/537# 全体 | 「何をもって簡素化成功とするか」が未定義。議論が美しい比喩と個別施策に散っている | `hard gate数`, `active multiplier数`, `clamp率`, `stale率`, `1 cycleあたり block理由数` などの reduction scorecard を作る |

---

## 2. まず支持したい点

### 2.1 536# の「Cから入り、Aを目標にする」勾配

これは良い。いきなり OFI/A-S 全面移行へ飛ぶと、現状の混乱にさらに新層を足す危険がある。まず明らかに飽和している部分、死んでいる部分、説明不能な部分を削るという順番は妥当である。

### 2.2 537# の「catastrophic guard は残す」修正

536# は「ceiling 廃止」に寄り気味だったが、537# が

- ceiling は段階的に動かす
- catastrophic guard は残す
- OFI は最初から全面置換せず bridge として導入する

と補正したのは良い。これは市場理論というより**運用工学として正しい**。

### 2.3 「MLを最終意思決定者にしない」という方向

この点も支持する。Sidecar は、現状の設計でも「最終判断」ではなく「情報要約」として置かれている。これは本システムの失敗史とも整合する。

---

## 3. 536#/537# が見落としている現況差分

### 3.1 537# の一部は、すでに current config に反映済み

現行 `configs/v460/fill_test.yaml` と関連実装を見ると、次は既に変わっている。

- `configs/v460/fill_test.yaml:42`
  - `min_spread_jpy: 500`
- `configs/v460/fill_test.yaml:1160`
  - `composite_risk_enabled: true`
- `scripts/v460/lib/sidecar_types.py:36`
  - sidecar TTL は既に `7800s`
- `scripts/v460/lib/fill_config.py:402-403`
  - `cross_venue veto` には `max_consecutive` と `inventory_zero_threshold_mult` が既にある

このため、537# の

- `composite_risk_enabled: true`
- `TTL 600→7200`
- `min_spread 700→500`

を「これからやる施策」として扱うのは、現況との差分管理として危うい。

ここで必要なのは新提案ではなく、**状態棚卸し**である。

### 3.2 今の本当の論点は「導入済みか否か」ではなく「効いているか」

例えば `composite_risk` は enabled だが、それが

- soft gate の硬直を本当に減らしているのか
- 逆に reason taxonomy を不透明にしていないか

は別問題である。536#/537# は「未導入の善策」として語るが、現状では **既導入・未検証** のものが混ざっている。

この差は大きい。設計上、

- 未導入なら「入れるか」
- 既導入なら「残すか、弱めるか、置換するか」

で議論の仕方が変わるからである。

---

## 4. 強く反論したい点

### 4.1 `buy=0.30, sell=0.40` を「buy aggressive」と呼ぶのは危ない

`pre_order_adjustments.py` の `_apply_offset_multiplier()` は、通常モードでは multiplier が大きいほど **mid から遠ざかる**。これは現行システムで「高 offset = より保守的」の意味である。

したがって、`offset_ceiling_ratio_buy=0.30` を現行 `0.25` から引き上げるのは、buy を aggressive にするのではなく、むしろ **より深く待つ / より遠ざける余地を増やす**方向である。

もし 537# が言いたいことが

- buy は取りに行きたい
- sell はより厳しく守りたい

なのであれば、設計表現は

- buy ceiling は lower / or unchanged
- sell ceiling は higher

のように、**意味論を固定してから**書いた方がよい。

### 4.2 `ceiling 0.25→0.35` は first move としては少し危ない

531# 系の「100% clamp 飽和」は確かに強い警告である。だが、そこから即 ceiling を上げると、今までは 0.25 で止まっていた multiplier 連鎖が、より広い範囲で実価格へ出てくる。

もし upstream の主因が

- macro
- toxicity
- trend
- velocity

の掛け算暴走にあるなら、ceiling 引き上げは「隠れていた過剰防衛」を市場へ露出させるだけになりうる。

順番としては、

1. stage ごとの実効寄与を可視化
2. 必要なら stage cap を入れる
3. その後 ceiling を ladder で上げる

の方が安全である。

### 4.3 `composite_risk` を simplification の本命にし過ぎない方がよい

`composite_risk` は確かに便利だが、設計上は

- 複数の soft gate をスコアへ圧縮する
- そのスコアでさらに block する

という構造であり、**単純化というより集約**である。

これはうまく使えば良いが、悪く使うと

- 元 gate も残る
- composite も追加される
- block reason が二段化する

という、今の「硬直化」そのものを強化する。

simplification を狙うなら、`composite_risk` は

- 既存 soft gate を置換する
- 置換対象を明示する
- 「なぜ通した/止めた」を 1 行で説明できる

という条件付きで使うべきである。

---

## 5. 補強したい点

### 5.1 時間帯ルールは「捨てる」より「降格」がよい

536# が時間帯 hardcode を嫌うのは妥当だが、時間帯は単なる overfit ではない場合もある。

- 流動性供給者の交代
- 海外市場の開閉
- 昼休み・薄商い
- 取引所間の先行遅行パターン

の proxy になることがあるからである。

したがって、完全撤去よりよい順序は次である。

1. hard block から外す
2. feature / prior / weight として残す
3. OFI/CV 等のより本質的な指標が十分に置換できたら削る

537# はここを一部補っているが、538# ではより明確に支持したい。

### 5.2 A-S は「直接実装」より「参照モデル」として使う方が安全

537# の A-S 提案は理論として正しい。ただし live へそのまま接続するには、まだ前提が足りない。

特に弱いのは

- `kappa` の推定安定性
- latency / cancel-replace を含む実執行コスト
- Coincheck 特有の fill 到着分布

である。したがって A-S はまず

- reference spread
- sanity bound
- 現行 base spread の比較対象

として導入するのがよい。

### 5.3 OFI も CV も「一つのハブに潰し切らない」方がよい

536# は「OFI ハブに束ねる」と書くが、実務上は

- OFI: ローカル板の需給圧
- CV: 他 venue からの先行
- VPIN / toxicity: flow の不均衡蓄積

で見ているものが違う。

これらを早い段階で 1 スカラーに潰すと、簡素化はできても **診断可能性**を失う。

おすすめは

- まずは orthogonal な 3 軸として残す
- そのうえで最終スコアへどう落とすかを later phase で決める

である。

---

## 6. 536#/537# に足したい「第三の道」

両文書はやや

- 固定ルールを捨てる
- かといって ML 丸投げもしない

という二項対立で書かれているが、実際には中間案がある。

それは、**低次元ノブだけを学習で較正し、 hard invariant は固定する**方式である。

例えば学習対象を

- `min_spread floor`
- `offset ceiling`
- `toxicity threshold`
- `CV veto threshold`

の 4-6 個に限定し、

- monotonicity を守る
- 安全域外へ出ない
- same-SHA walk-forward でしか更新しない

という制約を付ければ、「ML への全委譲」とは全く別物になる。

これは 536#/537# が見落としている、かなり実務的な逃げ道である。

---

## 7. いま本当に必要なもの

### 7.1 reduction scorecard

簡素化を言うなら、先に測るべきである。最低限ほしいのは以下。

- active hard gate 数
- active soft gate 数
- active multiplier 段数
- clamp rate
- stale sidecar rate
- 1 cycle あたり平均 block reason 数
- 1 fill あたり平均 explanation token 数

これがないと、「すっきりした気がする」だけで終わる。

### 7.2 current-state matrix

537# のように良い提案でも current config とズレると、作業が空転しやすい。必要なのは、各施策について

- already on
- already off
- partially on
- proposed
- verified effective
- verified ineffective

を 1 表にした matrix である。

### 7.3 simplification の定義

いま曖昧なのはここである。簡素化とは

- コード行数削減
- gate 数削減
- tuning ノブ削減
- live decision path 短縮
- PnL attribution の明瞭化

のどれを優先するのか。これを決めないと、536# の詩的な整理も 537# の具体策も、結局は別方向へ広がる。

---

## 8. 結論

536# は、硬直した議論をほぐすための**よい起爆剤**である。537# は、それを実装言語へ落とし直した**かなり良い一次翻訳**である。

そのうえで、セカンドオピニオンとしての判定は次の通りである。

1. **支持**
   - C→A の段階移行
   - catastrophic guard 温存
   - ML を情報要約に留める方針

2. **反論**
   - ceiling 引き上げ先行
   - `buy 0.30 / sell 0.40 = buy aggressive` という説明
   - composite_risk を simplification の切り札と見ること

3. **追加提案**
   - reduction scorecard
   - current-state matrix
   - low-dimensional learned calibration という第三の道

要するに、536#/537# の価値は「易占が当たった」ことではない。**複雑化の本質を、捨てる対象・残す対象・測る対象へ分ける視点を出したこと**にある。

538# 時点で次に進めるなら、最初の一手は大改造ではなく、

- current-state matrix 作成
- reduction scorecard 作成
- stage 寄与を見てから ceiling を動かす

の 3 つである。ここが整うと、断捨離が思想で終わらず、実装判断に変わる。
