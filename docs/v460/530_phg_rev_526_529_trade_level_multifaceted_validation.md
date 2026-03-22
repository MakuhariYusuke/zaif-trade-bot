# 530# 526#-529# レビュー — 取引単位再検証と sell 崩れ / deadlock の構造整理

> 更新: 2026-03-22
> 対象: 526#-529#, `results/v460/fill_test/logs/fill_test.log`, `results/v460/fill_test/fill_records_20260322.jsonl`, 関連実装
> 注: 新規分析スクリプトは追加せず、既存ログ・JSONL・実装のみで検証した。主対象は 529# が切り出した `Cycle 14708-14968`。

## 0. 総評

526#-528# の実装は、全体としては良い。特に 526# の可観測性強化は、今回の 529# のような取引単位レビューを成立させる意味で有効だった。

一方、529# は問題意識そのものはかなり良いが、**集計の母集団と粒度が混ざっている**ため、数字の一部はそのままでは再現できない。ここを補強しないと、正しい改善案に辿り着いていても優先順位を誤る。

レビュー者としての結論は次の通りである。

- 本当に重い損失源は **sell 側** であり、この認識は 529# を支持できる
- ただし 529# の headline 数値は、log 直読みでは **`87 fills / -41.9bps / 43W-44L`** で、文書記載の `89 fills / -35.6bps / 45W-44L` と一致しない
- `cross_venue favorable tighten` は、529# が書くより **sell 側で効いている**
- 16:22 以降の **在庫ゼロ + buy veto + sell freeze** は、実際に deadlock 的な膠着を作っている
- `final_clamp 0.25` は確かに飽和しているが、**単独犯**というより upstream が ceiling に張り付いていることの症状として扱う方が正確

---

## 1. Findings

| # | 重要度 | 対象 | 指摘 | 推奨対応 |
|---|---|---|---|---|
| 1 | HIGH | `docs/v460/529_phg_rpt_fill_test_trade_analysis_20260322.md`, `results/v460/fill_test/logs/fill_test.log`, `results/v460/fill_test/fill_records_20260322.jsonl` | 529# の headline 数値は log 直読みで再現しない。`261 cycles / 87 fills / -41.9bps / 43W-44L` であり、文書記載の `89 fills / -35.6bps / 45W-44L` とズレる。また `fill_records_20260322.jsonl` の `cycle_id` は数値 cycle ではなく UUID 風で、log cycle と直接一致しない | 529# / 以後の分析では `source=log-cycle` か `source=fill_records` かを明示し、`event count` と `distinct cycle count` を分ける。中期的には `FillRecord` に `log_cycle_no` を追加して照合可能にする |
| 2 | HIGH | `results/v460/fill_test/logs/fill_test.log`, 529# 全体 | 529# の「sell が本丸」という認識は支持できる。log 直読みでは buy `55 fills / +2.35bps / avg +0.043`、sell `32 fills / -44.25bps / avg -1.383` で、実質的なドライバーは sell 側 | 次の改善は sell 全面停止ではなく、**toxic sell の条件分離**に寄せる。`favorable tighten 有無`、待ち時間、CV hint 有無で分解する |
| 3 | HIGH | `results/v460/fill_test/logs/fill_test.log`, 529# §4 | 529# は `C14957` を「唯一の明確な黒字」としているが、sell の `cross_venue favorable tighten` は aggregate でもかなり良い。該当 sell fills は `13 fills / +9.94bps / avg +0.765 / WR 61.5%`、non-favorable sell は `19 fills / -54.19bps / avg -2.852 / WR 42.1%` | `favorable tighten` を sell に残す。評価は単発成功談ではなく、**条件付き平均**で続ける |
| 4 | HIGH | 529# §5, `results/v460/fill_test/logs/fill_test.log`, `scripts/v460/lib/balance_checker.py` | 16:22 以降の「完全デッドロック」は表現としてやや強いが、構造としては支持できる。`BTC=0` で sell が止まり、buy は `cross_venue veto` や `spread_too_narrow` で止まり、在庫回復経路が消えている | `BTC=0` 時の buy veto 緩和、または veto の time-decay / release を P0 で入れる。これは profit-first でも正当化できる |
| 5 | MEDIUM | 529# §6 問題2, `scripts/v460/lib/offset_pipeline.py`, `configs/v460/fill_test.yaml` | `final_clamp 0.25` は実際に高頻度で発生しているが、ここから即「clamp が主犯」と断定するのは強い。該当窓では `118 cycle` で clamp、filled trade でも `83/87` が clamp 済みで、むしろ**説明変数としての識別力が低い** | ceiling の議論は継続してよいが、同時に `pre-clamp offset` の分布と `cap_hit 後の PnL` を side 別に見る。原因というより saturation 指標として扱う |
| 6 | MEDIUM | 529# §3, `results/v460/fill_test/logs/fill_test.log` | `C14902` の「macro_boost が timeout 短縮だけを残した」という診断は plausibile だが、まだ一部仮説である。`re-quote 後に不利 fill` は支持できるが、「元の注文なら助かった」までは log だけでは言い切れない | `macro timeout shortener` は `ceiling hit 時のみ無効化` を候補にしてよいが、因果は replay 的に確認する。文書では断定を少し弱める |
| 7 | MEDIUM | 529# §2, §4 | 529# は fast fill をやや毒視し過ぎている。実データでは sell `0-10s` は `15 fills / avg -0.324 / WR 60%`、sell `30s+` は `1 fill / -20.79` と、**速さそのものより stale exposure と tail** が重い | 「fast fill = 悪」ではなく、`sell 10-20s / 30s+` と `CV favorable tighten なし` を危険帯として整理する |
| 8 | LOW | `docs/v460/526_phg_impl_log_observability_and_dead_code.md`, `docs/v460/527_phg_impl_jpy_precision_and_silent_except.md`, `docs/v460/528_phg_impl_codex_import_cleanup_and_review_docs.md` | 526#-528# は方向として健全。ただし 527# の JPY 精度改善や 528# の import cleanup は **可観測性・保守性の改善**であり、短期 PnL を直接押し上げる施策ではない | 収益改善 attribution では 526-528 を「診断基盤改善」と位置づける。alpha 改善と混ぜない |

---

## 2. 529# の数値再現性について

### 2.1 log 直読みで再現できた値

`Cycle 14708-14968` を `results/v460/fill_test/logs/fill_test.log` から直接読み直すと、次だった。

- cycles: `261`
- fills: `87`
- pnl_sum: `-41.9bps`
- wins / losses: `43 / 44`

side 別:

- **buy**: `55 fills / +2.35bps / avg +0.043bps`
- **sell**: `32 fills / -44.25bps / avg -1.383bps`

このため、529# の

- `89 fills`
- `-35.6bps`
- `45W / 44L`

は、そのままでは再現しない。

### 2.2 何が起きているか

`results/v460/fill_test/fill_records_20260322.jsonl` は

- rows: `234`
- run_id: 単一 (`1774095355_5b72a73f`)
- git_sha: 単一 (`d93b9a5bf672`)

で、run/sha の純度自体は良い。

しかし `cycle_id` は `1774137728_2b2ef230` のような形式で、529# が使っている `14902` のような **log の連番 cycle** とは別物である。つまり今の per-trade 分析は、

- log では連番 cycle
- fill_records では timestamp/hash 系 cycle_id

を見ており、**母集団の join key がない**。

統計的にはここが一番危ない。count のズレそのものより、**どの trade を何で数えたかが曖昧**なことが問題である。

### 2.3 改善案

最低限、今後の分析では以下を明記した方がよい。

- `source=log-cycle` または `source=fill_records`
- `distinct cycle count` か `event count` か
- `run_id`, `git_sha`, `cycle range`

そのうえで実装側は、`FillRecord` に `log_cycle_no` を持たせるのが最も効く。これで 529# のような「特定 trade の原因追跡」がずっと安定する。

---

## 3. 529# で支持できる論点

### 3.1 sell 側が損失の中心

これは 529# の認識を支持してよい。buy がほぼ収支トントンなのに対し、sell はこの窓で `-44.25bps` まで沈んでいる。市場理論的にも、ask が stale なまま残ると informed buyer に hit されやすく、maker の sell は adverse selection を受けやすい。

つまり今の本丸は「system 全体が equally bad」ではなく、**sell quality の条件付き崩れ**である。

### 3.2 16:22 以降の膠着

529# の「完全デッドロック」は少し強い言い方だが、構造はほぼその通りである。log では

- 直前に sell fill が入り
- BTC が 0 になり
- sell は在庫不足で自然停止し
- buy は `cross_venue veto` と `spread_too_narrow` で止まる

という流れが続いていた。これは classic な **inventory recovery path の消失**であり、profit-first の観点でも強く是正対象である。

### 3.3 526# の可観測性強化は効いている

今回のレビューが回ったのは、526# が以下を入れていたからでもある。

- `scripts/v460/lib/fill_cycle_executor.py:572`
  - cycle result logging の統一
- `scripts/v460/lib/order_monitor.py:557`
  - cancel log に `order_id` と `side`
- `scripts/v460/lib/balance_checker.py:178`, `scripts/v460/lib/balance_checker.py:249`
  - 残高不足ログの文脈強化

これは収益直接施策ではないが、**誤診を減らす施策**として価値がある。

---

## 4. 529# で補正したい論点

### 4.1 `cross_venue favorable tighten` は過小評価しない方がよい

529# は `C14957` を強調しているが、実際には favorable tighten が入った sell は aggregate でも良い。

- sell + favorable tighten: `13 fills / +9.94bps / avg +0.765 / WR 61.5%`
- sell + non-favorable: `19 fills / -54.19bps / avg -2.852 / WR 42.1%`

つまり、**sell を守るなら favorable tighten の方向は正しい**。ここを「例外的な黒字 trade」として片付けるのはもったいない。

また 14933 のように、14957 以外にも favorable 条件下の黒字 sell は見えている。今やるべきは機能撤去ではなく、**どの条件で tighten が効き、どこで no-op / late になるか**の絞り込みである。

### 4.2 `fast fill = toxic` は一般化し過ぎ

待ち時間 bucket を side 別に見ると、像が少し違う。

**buy**

- `0-10s`: `17 fills / -1.36 / avg -0.080`
- `10-20s`: `10 fills / +2.37 / avg +0.237`
- `20-30s`: `6 fills / +8.49 / avg +1.415`
- `30s+`: `22 fills / -7.15 / avg -0.325`

**sell**

- `0-10s`: `15 fills / -4.86 / avg -0.324`
- `10-20s`: `9 fills / -13.32 / avg -1.480`
- `20-30s`: `7 fills / -5.28 / avg -0.754`
- `30s+`: `1 fill / -20.79 / avg -20.79`

ここから読めるのは、「速いから悪い」より

- sell の **10-20s** がかなり悪い
- `30s+` はサンプル少だが catastrophic
- buy は `20-30s` がむしろ最良

ということだ。したがって、待ち時間対策は side 非対称でやるべきである。

### 4.3 `final_clamp 0.25` は確かに飽和しているが、単独犯とは言いにくい

この窓では `118 cycle` で clamp が発生し、filled trade でも `83/87` が clamp 済みだった。しかもすべて `0.25` への着地である。529# が「pipeline 出力が一律切り捨てられている」と感じたのは理解できる。

ただし、ここまで ubiquitous だと、`clamped vs unclamped` の比較だけでは因果が弱い。むしろ

- upstream multipliers が offset を常に ceiling 近傍まで押し上げる
- その結果、final clamp が **ほぼ常時発火する safety rail** になっている

と見る方が正確である。

設計的には、問題は `final_clamp` そのものより、**pre-clamp offset の識別力が失われている**ことにある。

### 4.4 C14902 は「macro_boost 単独犯」とまでは言わない方がよい

14902 について、529# の

- macro_weak_up
- timeout 短縮
- re-quote 後の不利 fill

という流れはかなり plausibile である。

ただし、log から確実に言えるのは

- 最初の注文が未約定で cancel された
- その後に re-quote され
- その 2 本目が悪い fill になった

までであり、「最初の注文を残せば助かった」は反実仮想である。金融工学・市場理論の観点では、trend against maker の最中に stale quote を残せばもっと悪くなる可能性もある。

したがって文書上は、

- `macro timeout shortener が損失を悪化させた疑いは強い`
- ただし `macro_boost 単独犯` や `timeout短縮だけが悪い` とまでは断定しない

くらいが妥当である。

---

## 5. 526#-528# 個別レビュー

### 5.1 526# は良い改善

526# は profit を直接押し上げる施策ではないが、レビュー品質を上げる意味で良い。

確認できた点:

- `scripts/v460/lib/fill_cycle_executor.py:572`
  - `_log_cycle_result()` による cycle logging 整理
- `scripts/v460/lib/order_monitor.py:557`
  - unfilled cancel log の `order_id` / `side` 付与
- `scripts/v460/lib/balance_checker.py:178`, `scripts/v460/lib/balance_checker.py:249`
  - 残高不足ログの明確化
- `scripts/v460/lib/config_hot_reload.py:803`
  - non-fatal な config_hash update failure を debug 化

この種の改善は、今後の誤診を減らす。

### 5.2 527# は診断基盤としては妥当

確認できた点:

- `ztb/trading/live/exchanges/coincheck/adapter.py:317`
  - `market_buy_amount = str(round(jpy_amount, 8))`
- `scripts/v460/lib/skip_gate_evaluator.py` の例外観測強化
- 残高表示精度の改善

評価としては、これは **precision / observability の改善**であって、直接の alpha 改善ではない。良い修正だが、PnL の改善と混線させない方がよい。

### 5.3 528# は無害な整理

528# の import cleanup と review docs 追加は、保守性としては無難である。ただし収益系レビューでは、ここを大きな改善項目として扱う必要はない。

---

## 6. 優先アクション

### P0

1. **trade join key を整備する**
   - `FillRecord` に `log_cycle_no` を追加し、log cycle と JSONL を同一 trade で追えるようにする。

2. **在庫ゼロ時の buy veto を緩和する**
   - `BTC=0` かつ sell freeze 中だけ threshold を緩める、または veto に time-decay を入れる。

3. **sell favorable tighten を維持し、条件付き集計を標準化する**
   - 機能を切るのではなく、`favorable / non-favorable / no-op / veto` で分けて観測する。

### P1

1. **sell の stale exposure を side 別に詰める**
   - 特に `10-20s` と `30s+` を危険帯として扱う。

2. **final_clamp を saturation 指標として観測する**
   - `pre-clamp offset`, `post-clamp offset`, `cap_hit` を side 別・PnL 条件付きで見る。

3. **C14902 型の macro timeout 副作用を replay 的に確認する**
   - `ceiling hit 時のみ timeout shortener 無効` は候補として妥当。

### P2

1. **529# の block 理由表を粒度別に書き直す**
   - `line event count`
   - `distinct cycle count`
   - `filled trade count`

2. **526#-528# は診断基盤改善として別枠管理する**
   - 収益改善 attribution とは切り分ける。

---

## 7. 結論

529# は、問題の当たり方そのものはかなり良い。特に

- sell が本丸
- 16:22 以降の在庫ゼロ膠着
- final_clamp の飽和感

は、レビュー側でも支持できた。

そのうえで、今回いちばん補強したいのは次の 3 点である。

1. **数値の母集団を揃えること**
2. **sell を一括で悪者にせず、favorable tighten など効いている条件を残すこと**
3. **deadlock を「市場が悪い」で済ませず、inventory recovery path の欠落として直すこと**

つまり次にやるべきは、新機構を雑に足すことではない。まずは

- join key 整備
- veto release
- sell 条件分離

である。ここを詰めると、以後の fill test 論点がかなりクリアになる。
