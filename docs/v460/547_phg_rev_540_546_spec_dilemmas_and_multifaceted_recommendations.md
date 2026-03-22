# 547# 540#-546# レビュー — 仕様上の悩みの再整理と next step の優先順位付け

> 更新: 2026-03-23
> 対象: 540#-546#, `configs/v460/fill_test.yaml`, `scripts/v460/lib/maker_price.py`, `scripts/v460/lib/cycle_gate_aggregator.py`, `scripts/v460/lib/pre_order_adjustments.py`, `ztb/risk/toxicity_budget.py`
> 確認: `tests/unit/v460/test_405_offset_ceiling_pipeline.py`, `tests/unit/v460/test_sidecar_sac_integration.py`, `tests/unit/v460/test_266_market_theory_protocol.py` — **131 passed**
> 注: 546# にある通り、fill_test はまだ旧 SHA 稼働中であり、540#-545# の変更は live 未反映である。よって本レビューは「設計妥当性 + 反映前の止血順位」を主題とする。

## 0. 総評

540#-545# は全体として前進である。特に 540# の **pre-order pipeline と executor pipeline の分離**は、536#-539# の議論をかなり健全化した。ここは強く評価してよい。

そのうえで、いま一番気を付けるべきは次の 3 点である。

1. **540# 自身が示した最大 blocker は `preflight_insufficient` なのに、その後の施策が offset/sidecar に寄り過ぎている**
2. **542#-545# は ceiling, OFI, δ*, sidecar を一気に積んでおり、live 未検証のまま次の動的化へ進むのは危ない**
3. **546# の悩みはかなり筋が良いが、A/B/C/D/E のうち幾つかは「やるかどうか」より「今やるべきではない」が答えになる**

結論から言うと、次は「さらに賢い制御を足す」より、**540# が見つけた構造を same-SHA で live 検証し、最大 blocker と sidecar 実効値の分布を確認する段**である。

---

## 1. Findings

| # | 重要度 | 対象 | 指摘 | 推奨対応 |
|---|---|---|---|---|
| 1 | HIGH | 540# §3, 541#-545# 全体 | 540# の block 分布では `preflight_insufficient=31.1%` が最大で、`no_feasible_quote=13.6%`、`spread_too_narrow=7.9%` が続く。にもかかわらず後続は offset/sidecar 動的化へ重心が寄り過ぎている | 次の P0 は inventory / preflight / no-feasible の再点検。動的化の追加は live 反映後の block 構成を見てから |
| 2 | HIGH | 542#-545#, 546# | `ceiling 0.30`, `δ*→spread_adapt`, `OFI boost`, `toxicity→confidence`, `δ*→sidecar ceiling`, `quadratic shaping` を短期間に積んでいる。個々は理屈があるが、合成後の実効分布がまだ見えていない | これ以上 sidecar / δ* / OFI を足す前に、same-SHA live で `clamp率`, `hard_skip率`, `no_feasible_quote率`, `sidecar非ゼロ率` を測る |
| 3 | HIGH | 546# A, `scripts/v460/lib/sidecar_types.py`, `scripts/v460/lib/cycle_gate_aggregator.py` | `sidecar max_boost_bps 0.20→0.30` は今すぐの優先度が低い。quadratic shaping と toxicity attenuation が既に中間帯をかなり潰すため、ボトルネックが ceiling ではない可能性が高い | 546# A は **(b) 維持** を基本にし、live データ後に **(c) 0.25→0.30 の ladder** を検討する |
| 4 | MEDIUM | 540# §2, 541# §4, 542# §1 | `spread_adapt 主犯` という整理は有力だが、やや強い。中央値で identity の段でも、tail / 危険 regime では支配的になりうる | `global median` だけでなく `loss tail 条件付き`, `sell only`, `adverse OFI only` で stage 寄与を切る |
| 5 | MEDIUM | 543# §3, `ztb/risk/toxicity_budget.py` | `Toxicity Budget 独立化` は方向は正しいが、まだ `ToxicityAssessment` / `ToxicityLevel` 型は `sell_dynamic_kill` に依存しており、完全独立ではない | 将来 kill を外すなら、型定義を shared module に切り出して真に分離する |
| 6 | MEDIUM | 546# B, 545# B, `scripts/v460/lib/cycle_gate_aggregator.py` | CalibrationMap を入れる方向は良いが、Toxicity attenuation と同じ confidence 経路に直接足すと二重減衰になりやすい | 546# B は **(b) オフライン batch → 起動時 load** を推奨。最初は `regime×side` 程度に抑え、confidence 本体ではなく「ceiling補正」か「size補正」へ逃がす案も検討 |
| 7 | MEDIUM | 546# C, 544#-545# | δ* は既に pre-order と sidecar ceiling に入っている。executor へ第三の注入を行うと、同一理論値の多重適用になりやすい | 546# C は **(d) 保留** を支持。少なくとも現段階で executor へは入れない |
| 8 | LOW | 541#-545#, `scripts/v460/lib/maker_price.py` | 理論の実装先が `MakerPrice` に集中し、状態・テレメトリ・学術モデル接続が肥大化し始めている | `MakerPrice` から microstructure state/telemetry を分離する設計メモを早めに作る |

---

## 2. 支持できる点

### 2.1 540# の「二重構造の発見」は重要

これは今回の最大の収穫である。pre-order と executor を分けて見たことで、少なくとも

- どこで ratio が作られるか
- どこで追加乗算されるか
- どこで clamp されるか

が分かった。以後の議論はこの地図を外してはいけない。

### 2.2 541# の最適化は妥当

lazy import の引き上げと disabled stage skip は、派手さはないが堅実である。少なくとも

- hot path を軽くする
- disabled 段のノイズを減らす
- それでも stage 記録は残す

という方向は正しい。

### 2.3 543#-545# は「計測→接続」の順番を守っている

543# で OFI-Lite / Toxicity / δ* をまず観測可能にし、544#-545# で接続した流れ自体は悪くない。いきなり live control に飛び込まず、まず stage telemetry を整えたのは良い進め方である。

---

## 3. 強く補正したい点

### 3.1 最大 blocker を取り違えない方がよい

540# の数字をそのまま読めば、最大 blocker は `sell_dynamic_kill` ではない。

- `preflight_insufficient`: 31.1%
- `skip_gate`: 16.1%
- `no_feasible_quote`: 13.6%
- `timeout`: 11.5%
- `spread_too_narrow`: 7.9%
- `sell_dynamic_kill`: 5.9%

この並びで、541#-545# がほぼ offset/sidecar に振れているのは、少し焦点が先に進みすぎている。

金融工学というよりシステム設計の観点で言うと、今は「pricing intelligence」より前に「参加可能性」の方が詰まっている。参加できない系の blocker が大きい限り、賢い pricing を足しても効果は限定される。

### 3.2 542# の ceiling 0.30 は妥当候補だが、そこで一度止まるべき

542# の論理は理解できる。`spread_adapt` が 0.30 を出し、ceiling 0.25 が切っていたなら、0.30 へ上げるのは自然な候補である。

ただし、そこから直ちに

- δ* が narrow 閾値を上げる
- OFI が boost を強める
- sidecar ceiling が δ* で拡がる

まで積むと、「0.30 に上げた結果どうなったか」が見えなくなる。

したがって 542# の次に必要だったのは、本来は 544#/545# の追加動的化ではなく、**0.30 化単独の same-SHA 観測**である。

### 3.3 sidecar の実効値がかなり sparse になる可能性が高い

現行 sidecar はすでに次を持っている。

- `dead_zone=0.10`
- `shaping=quadratic`
- `toxicity attenuation`
- `δ* dynamic ceiling`

この組合せでは、中間帯 bias のかなりの部分が小さくなる。

例えば normalized bias が 0.5 なら、quadratic で 0.25 になる。`max_boost_bps=0.20` なら 0.05bps。さらに Toxicity ORANGE で 0.3 倍されると 0.015bps まで落ちる。

つまり 546# A の本質は「0.20→0.30 に上げるか」より、**いま sidecar が実際どれだけ非ゼロで効いているか**である。

---

## 4. 546# の保留論点への回答

### A) Sidecar max_boost_bps 0.20→0.30

判定: **今は (b)、次点で (c)**

- 今の base 推奨は **(b) δ* dynamic ceiling に委任**
- live 実測後にやるなら **(c) 0.25 → 0.30 の ladder**
- **(a) いきなり 0.30 固定**はまだ早い

理由:

- quadratic + toxicity で ceiling 以前に signal が潰れている可能性がある
- δ* ceiling が already-on なので、理論的に必要な時だけ 0.30 相当へ上がる経路は既にある
- まずは `effective sidecar offset` の分布を見るべき

### B) CalibrationMap → sidecar confidence 統合

判定: **(b) オフライン batch → 起動時 load** を支持

ただし、そのまま confidence 本体へ掛けるのはやや危ない。

おすすめ順:

1. `regime×side` 程度の低次元で開始
2. JSON / YAML へエクスポートして起動時 load
3. 最初は confidence 直掛けより、`sidecar ceiling scalar` か `size scalar` として使う

これなら 545# の toxicity attenuation と役割がぶつかりにくい。

### C) δ* → executor pipeline 伝搬

判定: **(d) 保留** を支持

理由:

- δ* は既に pre-order と sidecar ceiling に入っている
- executor は 540# の整理でも「比較的 tame」側
- 同一理論値の三重適用は、説明性より先に相関制御の問題を起こす

いま executor へ足すのは、理論の不足より**注入箇所の増え過ぎ**が問題になる。

### D) Drift Detection

判定: **(a) 先に Toxicity 分布カウンタ**, OFI PSI は後

これは 546# の感覚に賛成である。

- Toxicity はカテゴリ分布なので baseline が薄くても見やすい
- OFI は baseline 未確立の段で PSI を見ても、行動に落ちにくい

したがって、まずは

- ORANGE + KILL の持続比率
- OFI mean の偏り
- sidecar stale / non-zero rate

程度の軽量監視から始めるのがよい。

### E) その他

- `VG/Macro` executor 死亡段は、今すぐ削除より **quarantine 扱い**がよい
- `sell_hour` は「ハードコード悪」ではなく、現状は YAML ベースの条件付き保護として扱えばよい
- `pre-order identity 段` については、541# の 5 段 skip で十分合理的。ここから先は削除より「責務の整理」が大事

---

## 5. 追加で提案したいこと

### 5.1 まず same-SHA live validation を 1 本通す

546# が明示している通り、fill_test は旧 SHA 稼働中である。ならば今の優先順位は明確で、**新しい理論追加ではなく反映後の 1 本目の live 検証**である。

そこで最低限見るべき指標は次の 8 つ。

| 指標 | 理由 |
|---|---|
| `clamp_rate` | 542# の直接効果確認 |
| `final_clamp_hard_skip_rate` | ceiling 解放で hard skip が減るか |
| `no_feasible_quote_rate` | OFI/δ* 動的化で悪化していないか |
| `preflight_insufficient_rate` | 最大 blocker が改善しているか |
| `sidecar_nonzero_rate` | 545#/546# 後に sidecar が死んでいないか |
| `sidecar_attentuated_rate` | toxicity 減衰のかかり過ぎ検知 |
| `delta_star_ceiling_hit_rate` | δ* ceiling が本当に使われているか |
| `post_fill_30s_pnl by side` | 収益面の最終確認 |

### 5.2 telemetry schema に version を入れる

540#-545# で `offset_stages` の項目はかなり増えた。

- `ofi_lite`
- `ofi_mean`
- `as_delta_star`
- `delta_star_bps`
- `spread_bps`

ここで schema version が無いと、後で mixed-SHA 集計をした時に「キーがある run と無い run」が混ざる。

簡単でもよいので、`offset_stages_schema_version` 的なものを入れる価値がある。

### 5.3 `MakerPrice` の責務分割メモを先に作る

540-545 の実装で、`MakerPrice` はさらに多くの責務を持った。

- pre-order pricing
- OFI state
- δ* telemetry
- spread_adapt modulation
- state history

このまま live で効いたとしても、次にまた詰まる。今すぐ分割実装までは不要だが、

- pricing core
- microstructure state
- telemetry recorder

の 3 分割くらいのメモは先に作っておくと後が楽である。

---

## 6. 結論

540#-545# は、単なる思いつきの動的化ではなく、かなり筋の通った再設計である。特に

- 540# の pipeline 実態把握
- 543# の計測チャネル整備
- 546# の保留論点整理

は支持できる。

そのうえで、レビュー者としての結論は次である。

1. **いま最優先なのは、これ以上の追加動的化ではなく live 検証**
2. **546# A は保留寄り、B はオフライン load、C は保留、D は軽量監視から**
3. **最大 blocker は still `preflight_insufficient` 系であり、offset の賢さだけでは勝ち切れない**

要するに、次の一手は「もっと理論を足す」ではない。**540# が見つけた構造を、現在の 542#-546# 実装で same-SHA 検証し、どこが本当に改善しどこが依然詰まっているかを測ること**である。
