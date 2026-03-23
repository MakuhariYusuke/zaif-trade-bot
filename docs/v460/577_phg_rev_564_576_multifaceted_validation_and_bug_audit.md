# 577# 564–576番レビュー: eDRC・計測基盤・加法化移行の多角的検証

- 日付: 2026-03-23
- 目的: 564#–576# の論点を、コード・ログ・既存分析スクリプト・金融工学の観点から再検証し、論証の薄い箇所、設計過剰、実装ミス、統計解釈の危うさを整理する
- 使用した既存分析/検証:
  - `scripts/v460/analysis/analyze_fill_logs.py --date-from 2026-03-20 --date-to 2026-03-23`
  - `scripts/v460/analysis/analyze_fill_logs.py --date-from 2026-03-23 --date-to 2026-03-23 --git-sha c164d21d367b`
  - `results/v460/fill_test/fill_records_20260323.jsonl` の直接確認
  - `results/v460/fill_test/logs/fill_test.log` の TypeError 検索
  - `tests/unit/v460/test_571_robust_stats.py`, `tests/unit/v460/test_467_remaining_issues.py` (`55 passed`)

---

## 0. 総合判定

564#–576# の流れは、**問題の発見力は高いが、実装到達度の見積りがやや楽観的**である。

特に強く結論づけると、現状は次の4点で整理するのが妥当である。

1. **567# / 569# / 576# の「計測基盤を直す」方向性は概ね正しい**
2. **しかし 568# / 570# / 572# / 575# / 576# が言う「加法パイプライン移行」は、コード上まだ成立していない**
3. **eDRC 自体は一部 live 経路に入っているが、評価母集団が極小で、しかも 4 日集計は mixed-SHA**
4. **収益上の本丸はなお `sell` 左尾だけではなく、`buy` 平均悪化 + `preflight_insufficient` 優勢の複合問題**

したがって、現時点の最重要判断は「新数式を増やすこと」ではなく、**測れていないものを測れるようにすること**である。

---

## 1. Findings

| # | 重大度 | 対象 | 指摘 | コメント |
|---|---|---|---|---|
| 1 | CRITICAL | 568# / 570# / 572# / 575# / 576# | **`experimental_additive_pipeline` は、実際には加法パイプラインではない** | `scripts/v460/lib/fill_config.py` と `scripts/v460/lib/offset_pipeline.py` を見る限り、切り替わっているのは eDRC ベースの ceiling 解決と robust input 利用だけで、9段の乗数チェーン自体はそのまま残っている。`additive_base_bps` も未使用。現状は「加法化移行」ではなく「乗法チェーン + 動的 ceiling」である。 |
| 2 | CRITICAL | 567# / 569# / 571# / 573# / 576# | **`spread_capture_bps` / `adverse_selection_cost_bps` は依然として JSONL に落ちていない** | `scripts/v460/lib/fill_record_builder.py` では payload に積んでいるが、`ztb/metrics/fill_quality.py` の `FillRecord` スキーマにフィールドが無く、sanitize で落ちる。そのため `analyze_fill_logs.py` は `0/591 fills` と表示し続ける。564#–576# で重視されている執行品質分解の根拠が、まだ保存経路で失われている。 |
| 3 | HIGH | 566# / 568# / 569# / 570# / 574# / 575# | **eDRC 実装式が仕様と一致していない** | 仕様文書は `sigma/max(spread_bps, 1.0)` を前提にしているが、実装 `scripts/v460/lib/fill_config.py` は `exp(alpha * sigma + beta * adverse_ofi)` であり、spread 正規化が無い。したがって 569# と 574# のパラメータ論は同じ式を議論していない。 |
| 4 | HIGH | 575# | **`edrc_hard_cap` は実際には hard ではない** | `scripts/v460/lib/fill_config.py` では hard cap を先に適用した後で `hour_ceiling_mult` を掛けているため、`edrc_hard_cap=1.0` でも UTC14 の `×2.0` で `2.0` まで拡大する。確認用スニペットでも `hour14 -> 2.0` を確認した。 |
| 5 | HIGH | 576# | **`3/20–3/23` 集計は system health としては有効だが、eDRC 効果検証としては不適切** | `analyze_fill_logs.py` でも `git_sha_unique=13`。実際に `execution_additive_enabled == true` の current subset は `3 rows / 2 fills / avg -2.31bps` しかない。したがって 576# の4日集計を「eDRC有効化後の成績」と読むのは強すぎる。 |
| 6 | HIGH | 565# / 567# / 576# / `analyze_fill_logs.py` | **buy 30s / sell 90s の一次窓差は、なお主要集計ラベルに残っている** | `scripts/v460/lib/pnl_measurer.py` は sell で `post_fill_wait_sec_sell=90s` を使う一方、`analyze_fill_logs.py` は `avg_pnl30` と表示し続ける。I1 修正で E3 60/120 崩壊は直ったが、一次レポートの side 間比較はまだ apples-to-oranges である。 |
| 7 | MEDIUM | 574# / 575# / 576# | **`execution_sigma` テレメトリの単位が読みにくい** | `offset_pipeline.py` は eDRC 入力として `sigma * 10000` を渡すが、JSONL に保存している `execution_sigma` は raw ratio のまま。結果として `0.000117` のような値が出る。文書側がこれを bps と読めば 10,000 倍ずれる。 |
| 8 | MEDIUM | 564# | **`preflight_insufficient` を AS の直接証拠とみなすのは強すぎる** | 方向性としては一理あるが、現行 `configs/v460/fill_test.yaml` では `lot_sizing.enabled=false`, `kelly.enabled=false`, 固定ロット運用であり、資本制約・在庫偏り・参加率制約が混ざる。金融工学的にも inventory toxicity だけに還元し切れない。 |
| 9 | MEDIUM | 566# | **`inv_skew` 完全復活ではなく、閾値付き emergency override という整理は妥当** | ここは 566# の中で比較的良い提案。現行 `scripts/v460/lib/maker_price.py` では `inv_skew_regime_gate_enabled` により trending 時に在庫補正を切るため、`preflight_insufficient` 優勢の局面では emergency override に価値がある。ただし「常時復活」は directional alpha と衝突する。 |
| 10 | LOW | 573# / 575# | **テストは通っているが、抜けがある** | `55 passed` は確認したが、`spread_capture` の end-to-end 永続化、`hard_cap` の hour multiplier 後保証、`experimental_additive_pipeline` の名称と実装の乖離を抑えるテストが無い。 |

---

## 2. 支持できる点

### 2.1 567# I1 は正しい

`pnl_measurer.py` の E3 60s/120s 基準を `cfg.post_fill_wait_sec` から実際の `wait_sec` に変えたのは正しい。これにより、sell 側で「60s が実質 90s と同一点に崩壊する」問題は是正された。

### 2.2 569# P2 / P3 は正しい

- `maker_risk_guards.py` では favorable tighten は `side == sell` のとき無効化されている
- `pre_order_adjustments.py` では各段 multiplier に `2.0` cap が入っている

この2つは、564#–565# のレビューを具体的に実装へ落とせている点として評価してよい。

### 2.3 576# の `_build_fill_record()` TypeError 特定は正しい

`results/v460/fill_test/logs/fill_test.log` では

- `_build_fill_record() got an unexpected keyword argument 'execution_sigma'`

が多数回出ており、576# の問題提起は実ログと一致する。これは単なる仮説ではなく、稼働阻害の一次障害だった。

---

## 3. 反論・補強が必要な点

### 3.1 「加法パイプラインへ進んだ」は、まだ言い過ぎ

564#–576# を通して、議論は「乗法チェーン爆発をやめて加法/RMS へ」という方向に進んでいる。これは方向としては良い。しかし、**実コードはまだ加法化されていない**。

現状の `experimental_additive_pipeline` は、実態としては:

- 乗法チェーンはそのまま
- final clamp に入る ceiling だけ動的化
- その ceiling 入力に robust sigma / robust OFI を使う

である。よって 576# などが current run を「eDRC + additive pipeline の挙動」と書くのは、設計面では看板倒れに近い。

### 3.2 574# のパラメータ論は、式の前提が変わっている

569# は `sigma/spread` 正規化前提で `alpha=0.16, beta=0.28` を議論している。一方 574# / 575# は `sigma_bps` 直入れ前提で `alpha=0.020, beta=0.40` になっている。これは単なるチューニング差ではなく、**数式のスケールが異なる別問題**である。

したがって「569# の理論が 574# で実装された」と読むのは危険である。別名で扱うか、式を統一してから比較すべきである。

### 3.3 576# の headline は正しいが、解釈が少し先走っている

`3/20–3/23` の aggregate 自体は `analyze_fill_logs.py` でも再現した。

- `Total=1941, Filled=591, Fill rate=30.4%`
- `Avg PnL=-0.07bps`
- `buy=-0.28bps, sell=+0.21bps`
- `AS=27.1%`
- `preflight_insufficient=34.7%`

ここまでは良い。ただし、これを「575# eDRC 有効化後の system health」と読むのは強すぎる。真の eDRC-on subset は `git_sha=c164d21d367b` で、

- `6 rows / 2 fills`
- `avg_pnl=-2.31bps`
- `execution_additive_enabled == true` は 3 rows のみ

である。統計的にはまだ感想戦すら難しい水準である。

---

## 4. 金融工学・市場理論からの整理

### 4.1 現在の問題は「sell tail だけ」ではない

576# の current 4-day aggregate では

- `buy avg = -0.28bps`
- `sell avg = +0.21bps`
- ただし `sell AS率 = 36.0%`, `p10=-8.26`, `p05=-11.67`

である。つまり構図は:

- **buy**: 平均で負ける
- **sell**: 左尾が深い

であり、以前の 560# 系とは少し姿が変わっている。金融工学的には、これは

- buy は participation / quote quality の問題
- sell は tail hedge / toxicity control の問題

と分けて考えるのが自然である。

### 4.2 `preflight_insufficient` 優勢は inventory management の失敗も示すが、それだけではない

`preflight_insufficient` が 34.7% は重い。ただし、これをそのまま「AS の結果」と断じるのは危険である。現行設定は固定ロットで、lot sizing や Kelly は止まっている。したがって、

- toxic flow に巻き込まれた inventory 偏り
- 固定ロットゆえの資金効率の悪さ
- 口座残高/在庫に対する発注サイズ不整合

が同時に混じる。

よって 566# のように **emergency override 付き inventory skew** を入れるのは候補だが、まずは「どのサイドで、どの残高制約で止まっているか」を分解するのが先である。

### 4.3 現在の収益改善で一番重要なのは `spread_capture` を測れるようにすること

マーケットメイクの品質は

- `spread_capture`
- `adverse_selection_cost`

の2つに分けないと、どこが改善したのか分からない。eDRC は本来「毒を避ける」ための制御だが、これだけでは

- 毒を避けたのか
- ただ約定率を落としただけか
- spread を十分に取れているのか

が判定できない。したがって執行品質分解の保存バグは、単なるテレメトリ不備ではなく、**収益改善ループそのものの欠落**である。

---

## 5. ドキュメント別コメント

### 564#

- 乗算チェーン爆発への危機感は支持
- ただし `preflight_insufficient = ASの絶対証拠` は強すぎる
- `inv_skew` を常時復活させるより、566# の threshold override の方が実装現実に合う

### 565#

- ファクト掘り起こし力は高い
- `sell_dynamic_kill` や `90s窓` 問題を正しく当てている
- ただしその後に進んだ 567#–575# の何が本当に治ったかは、もう一段厳密に分ける必要がある

### 566# / 568# / 570#

- 方向性は良い
- ただし「数式設計」と「現コード」が乖離しており、仕様書の完成度に対して実装到達度が低い
- 今は仕様の追加より、式とコードの一致を優先すべき

### 567# / 569#

- ここは比較的良い
- I1 修正、sell favorable tighten 無効化、stage cap 導入はいずれも意味がある
- ただし 569# の `spread_capture` 修正完了という説明は不完全で、保存経路はまだ最後まで通っていない

### 571# / 573#

- `RobustStats` 自体は妥当で、テストも通る
- ただし効果測定の心臓部である `spread_capture` 永続化が未達なので、分析セクションがまだ空回りする

### 574# / 575#

- パラメータを決めて前へ進めようとする姿勢は良い
- しかし、式の単位系と hard cap の意味が揺れているため、数理仕様としてはまだ固いとは言えない

### 576#

- バグ発見力は高い
- ただし 4 日集計を eDRC 効果と読むのは早い
- より正確には「mixed-SHA の system health」と「c164 の tiny live sample」は切り分けるべき

---

## 6. 優先度付き提言

### P0

1. **`FillRecord` に `spread_capture_bps` / `adverse_selection_cost_bps` を追加し、JSONL round-trip テストを足す**
2. **eDRC 実装式を仕様と一致させるか、逆に仕様書側を現実装へ合わせて一本化する**
3. **`hard_cap` を本当に hard にする**
   - `hour_ceiling_mult` 後に cap するか、cap を最終段へ移す
4. **`experimental_additive_pipeline` の名称を見直すか、本当に M2 を実装する**
5. **`analyze_fill_logs.py` に測定窓の明示を入れる**
   - 少なくとも `buy=30s / sell=90s` をヘッダに出す

### P1

1. **current eDRC subset (`git_sha=c164d21d367b`) を独立に追う**
   - 4日 aggregate とは混ぜない
2. **inventory emergency override を小さく試す**
   - 常時復活ではなく、偏重閾値超え時のみ
3. **`preflight_insufficient` を JPY不足 / BTC不足 / 固定ロット過大 に分解する**
4. **テスト追加**
   - `spread_capture` 永続化
   - `execution_sigma` 単位
   - `edrc_hard_cap` after `hour_ceiling_mult`
   - `experimental_additive_pipeline=True` でも乗法チェーンが残ることを明示する回帰テスト

---

## 7. 結論

564#–576# は、発見フェーズとしてはかなり良い。特に 565# と 576# は、問題の臭いを嗅ぎ分ける力がある。

ただし、現時点で一番危ないのは「加法化へ進んだ」「eDRC で system health を見た」と **言葉だけが先に進んでいること**である。コードの実態はそこまで到達していない。ここを曖昧にしたまま次の機構を足すと、また attribution が壊れる。

したがって、次の一手は明確である。

1. 測定を直す
2. 式と実装を一致させる
3. same-SHA の tiny sample を増やす

この順番なら、金融工学的にも設計的にも、今よりかなり筋の良い改善ループに戻せる。
