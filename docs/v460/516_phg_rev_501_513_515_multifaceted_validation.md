# 516# 501#-513#/515# レビュー — Fill Test 劣化メカニズムの再検証と補強

> 更新: 2026-03-21 16:10 JST
> 対象: 501#-513#, 515#, `results/v460/fill_test/fill_records_20260320.jsonl`, `results/v460/fill_test/fill_records_20260321.jsonl`, `results/v460/fill_test/logs/fill_test.log`, 関連実装
> 注: 今回は**分析スクリプトを新規追加せず**、既存実装・既存ログ・既存 JSONL の照合のみで検証した。

## 0. 総評

501#-513# と 515# には、かなり当たっている指摘がある。特に以下は評価してよい。

- 501#/506# の **cross-venue basis を生値ではなく de-mean すべき**という方向性
- 508#/509# の **可観測性強化と sell_age_cap の修正**
- 513# の **dormant mechanism 監査**
- 515# の **sell 側崩壊・XV veto 連鎖・EV/offset 相互作用への危機感**

一方で、515# の結論の一部は**現行 runtime と照合すると強すぎる**。現時点の本質は、単一犯人ではなく次の複合である。

1. **sell fill 自体は toxic で負けている**
2. **buy は cross_venue veto / no_feasible_quote で過剰抑制されている**
3. **offset pipeline が飽和し、cross_venue の retreat が no-op 化する場面がある**
4. **507# 設定と現行 runtime が一致しておらず、原因帰属を汚している**

つまり、「sell を守れていない」だけでも、「新機構を足せばよい」だけでもない。**runtime drift / config mismatch を先に止めたうえで、buy の過剰 filtration と sell の toxic fill を分けて扱う**のが筋である。

---

## 1. 重要所見

| # | 重要度 | 所見 | 検証結果 |
|---|---|---|---|
| 1 | HIGH | 507# の `recovery_skew_offset_mult: 2.0→1.5` が現行 runtime と不一致 | `configs/v460/fill_test.yaml` は `1.5`、しかし `fill_test.log` では 2026-03-21 に `config=2.0×` が繰り返し出力 |
| 2 | HIGH | 515# の `sell_dynamic_kill 不活性` は一般論としては強すぎる | 2026-03-21 の log では `sell dynamic kill activated` と `Cycle gate blocked: sell_dynamic_kill` が複数回確認できる |
| 3 | HIGH | 515# の `FFD/VG 0%稼働` は少なくとも FFD については現行 runtime と不一致 | 2026-03-21 の log で buy/sell 両側の `fast_fill_defense Activated` を確認 |
| 4 | HIGH | 現行 run の本丸は「buy 過剰抑制」と「sell toxic fill」の二重苦 | `20260321` same-run で buy cancel の主因は `no_feasible_quote=67`、sell fill は `18 fills / -40.06 / avg -2.23 / AS 77.8%` |
| 5 | HIGH | cross_venue は無効ではなく、**遅すぎるか飽和して no-op** になっている | log に `sell adverse hint ... offset 0.3000->0.3000 ... CAP_HIT=NO-OP` を確認 |
| 6 | MEDIUM | 515# の `sell_offset 0.14 主犯論` は単独原因としては弱い | 現行 run の sell fill は主に `effective_offset_used=0.15-0.25` に分布し、その帯でも負けている |
| 7 | MEDIUM | 515# の broad before/after 比較は mixed-SHA 汚染を受ける | `20260320` は `run_id=1773953473_cd76a319` でも `dfbe3b539eaa` と `5a546923a96c` が混在 |
| 8 | LOW | 502#-505# の設計整理は方向性として正しい | ただし今は profit-first の P0 ではなく、runtime 整合と fill quality の方が優先 |

---

## 2. same-run で見た現行劣化の中身

### 2.1 現行 run (`run_id=1774021842_804f05db`, `git_sha=20d4f778ef67`) の実態

`results/v460/fill_test/fill_records_20260321.jsonl` の same-run 集計:

- rows = 191
- fills = 51
- PnL30 total = **-41.67**

side 別:

- **buy**: `33 fills / -1.61 / avg -0.049 / WR 48.5% / AS(raw) 51.5%`
- **sell**: `18 fills / -40.06 / avg -2.226 / WR 22.2% / AS(raw) 77.8%`

ここから読めることは明快である。

- **sell は fill した時点でかなり toxic**
- しかし system 全体としては、**buy 側が veto/no_feasible で止まり過ぎている**

unfilled cancel の side 別内訳:

- **buy**: `no_feasible_quote=67`, `cross_venue_lead_lag_veto=13`, `skip_gate=9`
- **sell**: `timeout=15`, `skip_gate=11`, `spread_too_narrow=8`, `sell_dynamic_kill=4`

従って、現行の崩れ方は

- buy: **入れなさ過ぎる**
- sell: **入ると負けやすい**

という**非対称の二重障害**である。ここを一つの仮説で説明し切ろうとすると無理が出る。

### 2.2 待ち時間と PnL の関係

same-run の queue wait bucket:

- `0-10s`: `13 fills / -11.11 / avg -0.854 / AS 69.2%`
- `10-30s`: `22 fills / -33.13 / avg -1.506 / AS 63.6%`
- `30-60s`: `9 fills / +0.89 / avg +0.099 / AS 55.6%`
- `60s+`: `7 fills / +1.67 / avg +0.239 / AS 42.9%`

これは 515# の「fast fill が毒化している」という問題意識を**概ね支持**する。少なくとも現行 run では、**早く刺さる注文ほど toxic** である。

市場理論的にもこれは自然である。maker の即約定は、しばしば「有利だから約定した」のではなく、**相手が情報優位だから約定させられた**ことを意味する。Glosten-Milgrom / Kyle 的には、ここで無理に participation を取りに行くと負けやすい。

---

## 3. 515# に対する是々非々レビュー

### 3.1 当たっている点

#### A. `sell` の崩壊を主因に据えたこと

これは正しい。same-run でも sell は `avg -2.226bps`, `AS 77.8%` で、buy より明確に悪い。

#### B. `cross_venue veto → no_feasible_quote` 連鎖に注目したこと

これも正しい。2026-03-21 の log では、`last_reason=cross_venue_lead_lag_veto` を伴う `NO_FEASIBLE_QUOTE` が buy 側で連続している。

#### C. `basis_correction` の副作用を疑ったこと

これも論点として妥当である。ただし結論は少し補正が必要で、**basis correction 自体が悪い**というより、**basis correction 後の hint を受けた downstream の実行系が飽和している**と見る方が実態に近い。

### 3.2 補正すべき点

#### A. `sell_dynamic_kill の事実上の不活性化`

これは**その分析窓においては成立しうる**が、一般化は危険である。

現行 log では 2026-03-21 に:

- `sell dynamic kill activated`
- `Cycle gate blocked: sell_dynamic_kill`

が繰り返し出ている。従って「機構が死んでいる」というより、**分析に使った run/window では効きにくかった**と書くべきである。

#### B. `FFD/VG 0%稼働`

少なくとも **FFD については現行 runtime と不一致**である。

2026-03-21 の log には:

- `Activated (buy)`
- `Activated (sell)`
- `Reset on unfilled`
- `TTL expired`

が複数回出ている。

加えて、コード順序上 `scripts/v460/lib/fill_record_builder.py` の `ffd_boost_active` は **その注文を出した時点の状態**を保存し、`ztb/trading/risk/fast_fill_defense.py` の `evaluate_fill()` は **fill 後に** `scripts/v460/lib/orchestrator_post_cycle.py` から呼ばれる。よって FillRecord 上の `FFD active=0` は、「その fill をきっかけに起動しなかった」ではなく、**そのサイクル開始時点で boost 中ではなかった**という意味合いが強い。

したがって、515# のこの節は「安全機構が無効」ではなく、**観測の意味論がずれている可能性**として補強すべきである。

#### C. `sell_offset 0.14 主犯論`

ここも単独犯にはしにくい。

same-run の sell fills は主に:

- `[0.15, 0.20)`: `4 fills / -9.79 / avg -2.45 / AS 100%`
- `[0.20, 0.25)`: `14 fills / -30.27 / avg -2.16 / AS 71.4%`

で負けている。つまり、**base sell_offset=0.14 を出発点にしても、実際の fill は inv_skew / recovery_skew / EV / clamp の結果として 0.15-0.25 に再配置されたうえで負けている**。

言い換えると、問題は「0.14 が悪い」だけではなく、**sell 側の realized quote が toxic flow に対してまだ甘い**ことである。

#### D. `basis correction paradox`

問題意識は良いが、機序の書き方は少し直したい。

現行 log では sell 側にも hint は出ている。にもかかわらず:

- `sell adverse hint ... offset 0.3000->0.3000 ... CAP_HIT=NO-OP`

が出ている。これは「sell hint が出ない」のではなく、**出た時には既に offset が ceiling 近傍まで膨らんでおり、追加 retreat 余地がない**ことを示す。

従って、より正確な表現は:

- basis correction 導入後、sell 側 hint 自体は見えるようになった
- しかし downstream の offset pipeline が先に飽和し、cross_venue guard の retreat が no-op 化する
- 一方 buy では veto だけが強く残り、`no_feasible_quote` が積み上がる

である。これは filtration paradox というより、**signal integration の遅さと saturation 問題**である。

---

## 4. 501#-513# のうち評価すべきもの / 優先度を落とすもの

### 4.1 評価すべきもの

#### 501#/506# — de-meaning の方向

これは支持する。固定 basis は危険だが、`basis_ema_alpha=0.02` の適応型補正は、構造偏差を生値で誤検知するより合理的である。

#### 508# — observability 強化

有効。`basis_bps`, `adjusted_spread_bps`, `sell_age_cap` ログは、今回のレビューでも直接役に立った。これは良い投資だった。

#### 509# — sell_age_cap 修正

有効。現行 log に `micro_timeout sell_age_cap exceeded` が複数回あり、少なくとも **sell が無限にぶら下がる**類のバグは抑えられている。

#### 513# — dormant mechanism 監査

概ね妥当。`SAD/MCB` をはじめ「設定はあるが今の利益改善の主戦場ではない」ものを棚卸しした意義はある。

### 4.2 優先度を落とすもの

#### 502#-505# — `scripts/v460/lib` → `ztb` 移行 / object split

方向性はよい。ただし現状では **P0 ではない**。

今のボトルネックは設計美ではなく、

- runtime と YAML の不整合
- fill quality の side 非対称崩れ
- no_feasible_quote の過積載

である。設計整理は、これらを一段落させてからでも遅くない。

金融工学的にも、設計改善は長期の保守性を上げるが、短期 PnL を直接押し上げる力は弱い。今は **quote quality / participation control / toxic flow avoidance** の方が優先である。

---

## 5. 追加で拾うべき盲点

### 5.1 recovery_skew の runtime drift / config mismatch

これは今回の最重要 blind spot である。

- `configs/v460/fill_test.yaml` は `recovery_skew_offset_mult: 1.5`
- `scripts/v460/lib/fill_config.py` の dataclass default は `2.0`
- log は `config=2.0×`

この組み合わせは、**YAML override がどこかで落ちている**か、**runtime が旧 config を握ったまま**である可能性を強く示す。

この状態では 507# 以降の attribution が汚れる。先に直すべきである。

### 5.2 sidecar が stale のまま

2026-03-21 の log では `Cached sidecar signal is stale (TTL=7800.0s exceeded)` が繰り返し出ている。今回の主因ではないが、**本来別の制御で緩和できる局面が neutral/stale fallback で潰れている**可能性はある。

### 5.3 buy の over-filtering と sell の toxic fill を混ぜて議論しない

今の構図は対称ではない。

- buy は `no_feasible_quote / veto` が重い
- sell は `fill した後の質` が悪い

このため、「両 side に同じ保護を足す」「両 side の offset を同じ方向に動かす」は雑である。**buy は participation 回復、sell は quote quality 改善**という切り分けが必要である。

---

## 6. 優先アクション

### P0

1. **`recovery_skew_offset_mult` の YAML → runtime 反映不整合を潰す**
   - ここが崩れたままだと 507# 以降のレビュー全体が汚れる。

2. **cross_venue の no-op 率を first-class に観測する**
   - 既存可観測性で足りる。新スクリプトは増やさず、既存集計に `pre_offset/post_offset/cap_hit/vetoed` を足して見る。

3. **same-run / same-SHA 前提で 515# の統計を再記述する**
   - `20260320` は mixed-SHA 汚染があるため、比較根拠としては一段弱い。

### P1

1. **buy の `no_feasible_quote` 連鎖を緩和する**
   - 現行 run では最頻原因であり、利益機会を先に削っている。

2. **sell の fast-to-mid fill をさらに保守化する**
   - ただし `base sell_offset` 単独ではなく、`inv_skew / recovery_skew / EV / clamp` の合成後 offset を見て調整する。

3. **FFD の評価方法を修正する**
   - `FillRecord` 上の `ffd_boost_active` を「同一 fill での起動率」と読まない。既存ログと突き合わせる運用に改める。

### P2

1. **502#-505# の構造整理**
   - 今すぐではないが、中長期的には必要。

2. **513# dormant mechanism の再投入検討**
   - これは P0/P1 が片付いてからで十分。

---

## 7. 結論

515# は、問題の匂いをかなり正しく嗅いでいる。しかし、その結論を現行 runtime にそのまま適用すると、次の 3 点で危ない。

1. **`FFD 0%` や `sell_dynamic_kill 不活性` を一般論化してしまうこと**
2. **`sell_offset 0.14` を単独主犯にして、実際の saturated pipeline を見落とすこと**
3. **`basis correction` 自体を悪者にして、実際には downstream integration / cap saturation が問題である点を外すこと**

現時点の最も筋の良い整理は以下である。

- **buy**: veto/no_feasible で止まり過ぎ
- **sell**: fill quality が悪すぎる
- **共通**: runtime と config の不整合があり、レビュー対象を汚している

従って、次は新機構を増やすより先に、**runtime 整合性の回復 → same-run 再集計 → buy/sell 非対称対策の分離**で進むべきである。
