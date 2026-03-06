# 307# 303#-306# 再レビュー: システム工学 + 市場理論

> **対象**: [303#](303_ph2_resp_301_302_review_response.md), [304#](304_ph2_refactor_bps_ssot_dry_helpers.md), [305#](305_ph2_analysis_systems_market_theory_p0_improvements.md), [306#](306_ph2_impl_six_proposals_observational_redesign.md), `analysis_results/306_deep_dive.json`  
> **観点**: システム工学 / 市場微視構造 / 統計解釈 / 運用優先度  
> **立場**: 実装者ではなくレビュワーとして、意思決定を誤らせる論点を先に潰す

---

## 1. Findings

### F1. 306# の新可観測性は「実装済み」だが「検証済み」ではない【HIGH】

`306_deep_dive.json` では、以下がすべて実質未観測になっている。

- `regime_matched[*].sell_offset.n == 0`
- `regime_matched[*].buy_offset.n == 0`
- `as_deep_dive.*.as_offset.n == 0`
- `offset_pnl_corr.sell.n == 0`, `offset_pnl_corr.buy.n == 0`
- `fill_speed.sell.n == 0`, `fill_speed.buy.n == 0`

これは単に「今回たまたま値が入らなかった」というより、分析スクリプト側の参照フィールドが新実装と噛み合っていない可能性が高い。

- `analysis/306_deep_dive.py` は offset 分析で `effective_offset_ratio` を読んでいる
- しかし FillRecord 書込側 `scripts/v460/lib/fill_cycle_executor.py` は `effective_offset_used` / `spread_offset_ratio` を保存している
- `analysis/306_deep_dive.py` の約定速度分析は `fill_timestamp` を見ている
- しかし FillRecord スキーマ `ztb/metrics/fill_quality.py` にあるのは `queue_wait_sec` で、`fill_timestamp` はない

さらに、306# の価値の中核だった `offset_stages`, `queue_depth_ahead`, `queue_fill_prob_est` は、今回の deep dive では分析に使われていない。

結論として、306# の O1/E1 は方向として正しいが、今回の出力だけでは「新観測が有効に取れている」とはまだ言えない。

**推奨**:

1. `analysis/306_deep_dive.py` を現行 FillRecord スキーマに合わせて修正する
2. `offset_stages` を JSON 展開して stage 別寄与分析を追加する
3. 修正後、同一期間で deep dive を再実行し、306# の結論を再確定する

### F2. 306# の「sell/buy 差は非有意」は維持できるが、運用判断の主軸にはしてはいけない【HIGH】

306# は 299# よりかなり良くなっている。Block Bootstrap, Matched Pair, BH FDR への拡張は妥当で、`sell vs buy` を疑似 A/B と誤表記しなくなった点も正しい。

ただし、これは依然として **観察比較** である。

303# 自身が認めている通り、side はランダム割当ではなく、`SideSelector` と `balance_forced` に強く依存する。にもかかわらず、306# deep dive はまだ以下を分離していない。

- `balance_forced_switch`
- forced buy / repair trade
- `decision_path`
- alpha trade と inventory repair trade

したがって、今回の「差なし」は

> side 全体を止める根拠は弱い

とは言えても、

> side 非対称は本質でない

とまでは言えない。

**推奨**:

1. `sell vs buy` は Gate 根拠ではなく診断レポート扱いに固定する
2. 次回分析では `alpha` / `repair` / `balance_forced_switch` を最上位で分離する
3. buy 不振や EV 逆転は、まず交絡除去後に再判定する

### F3. 今回の本丸は side 差ではなく AS / non-AS と時間帯である【HIGH】

今回の JSON で最も強いシグナルはここだった。

| 論点 | sell | buy |
|---|---:|---:|
| AS avg_pnl30 | -6.6819bps | -5.7614bps |
| non-AS avg_pnl30 | +2.3892bps | +1.8275bps |
| AS vs non-AS diff | -9.0712bps | -7.5889bps |
| p 値 | 0.0 | 0.0 |

つまり、問題は「buy が悪い / sell が悪い」という粗い話ではなく、

> 毒性フローに当たる fill をどれだけ避けられるか

である。

時間帯でも sell の危険帯がかなり明瞭である。

- UTC 08h (JST 17時): sell pnl30 `-3.546`, AS `0.630`
- UTC 13h (JST 22時): sell pnl30 `-1.943`, AS `0.407`
- UTC 14h (JST 23時): sell pnl30 `-3.164`, AS `0.429`

市場理論的にも、この種の悪化は「side 固有の欠陥」より、セッション転換点での informed flow / order-flow imbalance / 流動性薄化の影響と解釈する方が自然である。

306# の中核結論は、

> global side suppression より、AS 回避と時間帯制御の方が効く

であり、この点は妥当である。

**推奨**:

1. `side gate` より先に `AS gate` と `time-of-day gate` を優先する
2. 特に sell は UTC 08h, 13-14h を別ルール化する
3. 評価軸も `sell vs buy` ではなく `AS/non-AS`, `session`, `regime`, `repair/alpha` に組み替える

### F4. buy 側 EV 逆転は面白いが、現時点ではそのまま意思決定に使うには弱い【MEDIUM】

306# は buy で

- AS 群 EV mean `+0.254583` (`n=7`)
- non-AS 群 EV mean `-0.920775` (`n=48`)

という逆転を示している。

ただし `n=7` は小さく、しかも F2 の交絡がまだ残っている。特に buy は inventory repair が混ざりやすく、`balance_forced_switch` や `decision_path` を分離しない限り、素直に「EV が逆に働いている」と断定するのは危険である。

**推奨**:

1. buy の EV は `regime × hour × balance_forced_switch × decision_path` で再分解する
2. `repair buy` と `alpha buy` を混ぜたまま学習・Gate 設計しない
3. EV 単独ではなく `microprice_bias`, `queue_fill_prob_est`, `offset_stages` を組み合わせる

### F5. `none` レジーム問題は完全には閉じていない【MEDIUM】

303# で `none` passive MM バイパスを入れた方向性は正しい。  
ただし `analysis/306_deep_dive.py` の `matched_regime_analysis()` は依然として `none` を除外している。

そのため、306# は

- 全体比較では `none` 含有を見ている
- しかし regime 深堀りでは `none` を見ていない

という半端な状態にある。

`none` が本当に「危険だから passive 化で改善した」のか、「単に全体を悪化させる母集団として残っている」のかを今の deep dive だけでは言い切れない。

**推奨**:

1. `none` を regime テーブルにも明示的に出す
2. `none_regime_passive_mm_enabled` の hit 率と PnL を別集計する
3. `none` を warmup / missing-signal / detector-failure に再分類する

### F6. 306# A1 は一部でロジック説明が逆方向になっている可能性がある【MEDIUM】

`scripts/v460/lib/param_adapter.py` では、

- `AS 超過のみ` の場合は `offset 縮小` で AS 回避
- これは `maker_price.py` の価格式と整合している

一方で、

- `AS 超過 + fill_rate 低下 + EV 負` の場合は `offset 増加`
- 理由文は `offset 拡大で AS 回避`

となっている。

しかし `maker_price.py` の実装では、

- buy: `price = best_bid + offset`
- sell: `price = best_ask - offset`

であり、offset ratio を大きくすると基本的には **より内側・より攻撃的** になる。したがって、`offset 増加 = AS 回避` という説明は少なくともそのままでは整合しない。

もし意図が

> 収益改善ではなく deadlock break のための liveness override

なら、それは AS 防御とは分離して扱うべきである。

**推奨**:

1. A1 を `AS defense` と `deadlock break` に分離する
2. `offset increase` を許す branch には「AS 回避」ではなく「liveness 優先」と明記する
3. 実験上も `A1 triggered` レコードを別 KPI で追う

### F7. 305# の PnL 分解が、306# の実験解釈にまだ十分使われていない【MEDIUM】

305# は良い問題設定をしている。

- spread capture
- adverse selection cost
- Parkinson sigma

この分解が入ると、「offset が広すぎる」のか「毒性フローに食われている」のかを分離できる。

しかし 306# の意思決定はなお aggregate な `post_fill_30s_pnl` 中心で、`EV gate` の根拠もやや粗い。今のままだと、

- fill 不足のせいで悪い
- AS のせいで悪い
- その両方

が混ざったままになる。

**推奨**:

1. 306# deep dive に spread capture / AS cost 分解を追加する
2. EV 調整は aggregate pnl ではなく分解後指標に基づいて行う
3. `sell悪化 = spread不足` なのか `AS過多` なのかを毎回分けて判断する

---

## 2. 妥当だった点

303#-306# で評価できるのは以下である。

1. `sell vs buy` を疑似 A/B から観察比較へ明示修正したこと
2. `none` 問題を放置せず、passive MM バイパスに寄せたこと
3. Block Bootstrap / Matched Pair / BH FDR へ統計を前進させたこと
4. `offset_stages`, `queue_depth_ahead`, `microprice_bias_bps` を仕込み、次の分析余地を作ったこと
5. 304# で BPS SSOT と DRY を進め、以後の誤差源を減らしたこと

方向性自体は悪くない。問題は「実装した観測を、まだ十分に使い切れていない」点にある。

---

## 3. 優先順位

### P0

1. `analysis/306_deep_dive.py` のフィールド参照不整合を修正し、同一期間で再集計する
2. `offset_stages` / `queue_depth_ahead` / `queue_fill_prob_est` / `microprice_bias_bps` を deep dive に入れる
3. `balance_forced_switch`, `decision_path`, `repair/alpha` 分離を deep dive の既定にする
4. sell の UTC 08h, 13-14h を session-specific veto または offset widening の対象として検証する

### P1

5. `none` regime を warmup / detector-miss / true-none に分解する
6. A1 を `AS防御` と `deadlock break` の二系統に分離する
7. 305# の spread capture / AS cost 分解をダッシュボード標準指標に昇格する

### P2

8. `AS classifier` を `EV + microprice + queue + offset_stages + session + regime` で再設計する
9. buy 側は `repair buy` 専用の別 policy を用意し、alpha buy と混ぜない

---

## 4. 総評

303#-306# は、299# までにあった「雑な sell/buy 議論」をかなり是正できている。そこは前進である。  
ただし、今回の deep dive をそのまま最終根拠に使うのはまだ早い。

現時点で引くべき結論は次の3点である。

1. **global な side 差より AS / session / regime の方が支配的**
2. **306# の新観測はまだ分析側で十分に消費されていない**
3. **次の改善は sell/buy 論争ではなく、AS 回避と repair 交絡除去に振るべき**

---

## 5. 確認メモ

今回のレビューでは、対象文書に加えて以下を照合した。

- `analysis/306_deep_dive.py`
- `scripts/v460/lib/fill_cycle_executor.py`
- `scripts/v460/lib/maker_price.py`
- `scripts/v460/lib/param_adapter.py`
- `ztb/metrics/fill_quality.py`
- `tests/unit/v460/test_306_proposals.py`

また、仮想環境で `tests/unit/v460/test_306_proposals.py --no-cov` を実行し、**51 passed** を確認した。  
ただしこれは主に単体レベルの整合確認であり、今回指摘した「観測が本当に実データへ載り、deep dive がそれを正しく読むか」の保証までは与えない。
