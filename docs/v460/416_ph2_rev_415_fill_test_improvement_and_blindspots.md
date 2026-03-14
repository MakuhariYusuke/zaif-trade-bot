# 416# 415 fill_test改善案レビュー

**Date**: 2026-03-14  
**Phase**: ph2  
**Type**: rev  
**対象**: `docs/v460/415_ph2_rpt_fill_test_log_analysis.md`  
**参照**: `results/v460/fill_test/fill_records_20260311.jsonl` - `results/v460/fill_test/fill_records_20260314.jsonl`, `results/v460/fill_test/logs/fill_test.log`, `results/v460/fill_test/fill_test_state.json`, `scripts/v460/lib/maker_price.py`, `scripts/v460/lib/pre_order_adjustments.py`, `scripts/v460/lib/orchestrator_balance.py`, `scripts/v460/lib/cycle_gate_aggregator.py`, `ztb/ml/skip_gate.py`, `scripts/v460/lib/config_hot_reload.py`, `scripts/v460/lib/bayesian_regime_filter.py`

---

## §1 エグゼクティブサマリ

415# は、現状の fill test 不調をかなり良い角度で捉えている。  
特に

- 非AS平均が全日・全sideで正
- `sell` 側悪化が 3/13-3/14 の主犯
- `VG triggered` が飽和している
- `JST 09h` / `23h` が危険帯

の観察は有用である。

ただし、そのまま採用すると危ない論点が 5 つある。

1. **売側悪化を 405# に寄せすぎている**
2. **`sell_dynamic_kill` を「残高ノイズ」と見なしすぎている**
3. **SkipGate の提案の一部は既に実装済みで、真の問題を外している**
4. **`git_sha` 混在の原因推定がずれている**
5. **一番大きい盲点は「final ceiling を executor 後段が迂回している」こと**

結論を先に書くと、**次に最優先で直すべきは sell ceiling 値そのものより、`maker_price` の final ceiling 後に executor 側が offset を再拡大している構造**である。  
そのまま 0.35-0.40 の A/B に入っても、真の最終 offset が制御されていないため attribution が崩れる。

---

## §2 主な指摘事項

| # | 重大度 | 指摘 | 根拠 | 推奨対応 |
|---|--------|------|------|---------|
| 1 | **CRITICAL** | `405#` が悪化の主因という整理は強すぎる。`effective_offset_used > 0.5` は **pre-405 の 3/11 時点で既に存在**し、`offset_stages.final=0.3` と矛盾している。 | `results/v460/fill_test/fill_records_20260311.jsonl` に `effective_offset_used=1.3053`, `offset_stages.final=0.3` の sell fill が存在。`scripts/v460/lib/maker_price.py:1013` で ceiling 済みだが、`scripts/v460/lib/pre_order_adjustments.py:51` は後段で再クランプしない。 | **P0**: executor 後段の最終 clamp を追加し、`maker_price_final_offset` と `execution_final_offset` を別記録する。405# の是非判定はその後にやる。 |
| 2 | **HIGH** | `sell_dynamic_kill` を「残高制限下のノイズ」と切るのは不正確。実態は **buy残高不足 → sellへ切替 → sell_dynamic_killで即ブロック** の相互作用。 | `results/v460/fill_test/logs/fill_test.log` で `[balance] buy insufficient, switching to sell immediately (091#)` の直後に `Cycle gate blocked: sell_dynamic_kill` が繰り返し出ている。`scripts/v460/lib/orchestrator_balance.py:73` と `scripts/v460/lib/cycle_gate_aggregator.py:641` が対応。 | **P0**: `opposite side` へ切替える前に、その side が kill 中なら idle/backoff を選ぶ分岐を入れる。`SDK` を単なるノイズでなく「route-to-kill deadlock」として扱う。 |
| 3 | **HIGH** | 415# の SkipGate 改善案の一部は既に実装済み。現状は「side別 threshold が無い」のではなく、**side別 adaptive threshold があるのに score/threshold 分布が崩れている**。 | `ztb/ml/skip_gate.py:550`, `ztb/ml/skip_gate.py:643` に side別閾値あり。`configs/v460/fill_test.yaml:314` は `mode: pnl`。`skip_gate_as_prob` は 3/12-3/14 の filled で全件 `null`。 | **P1**: 「side別 threshold 追加」ではなく、`mode=pnl` 前提で side別 score 分布・threshold_used 推移・skip率を再点検する。 |
| 4 | **HIGH** | 415# の `git_sha` 原因推定はずれている。「毎サイクル `git rev-parse HEAD`」ではなく、**hot reload 時に `_git_sha` が更新される**構造。 | `scripts/v460/lib/config_hot_reload.py:674` で `new_sha != runner._git_sha` のとき更新。`fill_record_builder` は `self._git_sha` を書いているだけ。 | **P1**: `run_id` 生成時の SHA を固定値として保持し、record には `start_git_sha` と `current_git_sha` を両方残す。 |
| 5 | **MEDIUM** | 415# の 3/12-3/14 比較は mixed-SHA 汚染が強い。日単位だけで 405#/414# 効果を断定しにくい。 | 3/12 は SHA ごとに `avg_pnl30=-0.219 / +0.504 / +0.793`、3/13 も 3 SHA 混在、3/14 も 2 SHA 混在。 | **P0**: `git_sha` または `run_id` 固定で same-SHA 再集計する。日次比較は補助情報に格下げ。 |
| 6 | **MEDIUM** | `3/12 TypeError` は現行コード上は既に修正済み。アクティブ課題として残し続ける必要は薄い。 | `scripts/v460/lib/bayesian_regime_filter.py:507` で deque を list 化してから slice。 | 415# では「修正済み確認」に落とし、優先度を下げる。 |
| 7 | **MEDIUM** | 問題は sell だけではない。buy も 3/12=-0.54bps, 3/13=-0.75bps, 3/14=-0.23bps と継続赤字。 | `fill_records_20260312/13/14.jsonl` 集計。 | sell 防御を先にやるのは正しいが、最終目標は buy/sell 両側の正転。`sell ceiling` のみで「復旧」とは見なさない。 |
| 8 | **MEDIUM** | `VG triggered` は確かに飽和しているが、「無意味」より「常時リスクプレミアム化」と見る方が正確。 | 3/12-3/14 の filled で triggered 98-100%。一方 `vg_boost_factor` の median は `1.0 / 1.389 / 1.63` と差がある。`scripts/v460/lib/maker_risk_guards.py:115` は continuous ramp。 | `vg_triggered` bool を主指標から外し、`vg_boost_factor` 分布と PnL の関係で再評価する。 |

---

## §3 415# で妥当だった点

### §3.1 非AS平均が正なのは重要

これは再集計でも概ね確認できた。  
少なくとも 3/12-3/14 の平均では、

- buy 非AS: `+1.23 / +1.65 / +0.44`
- sell 非AS: `+3.26 / +4.66 / +1.66`

であり、**「取れた良い fill をもっと残し、toxic fill だけを減らす」**方針は妥当。

### §3.2 JST 09h / 23h の危険帯認識は残す価値がある

ただし、ここは今すぐ hard skip を増やすよりも

- SkipGate feature
- hourly threshold offset
- side別 multiplier

に使う方がよい。  
mixed-SHA 期間なので、時間帯だけでルール化するのはまだ早い。

### §3.3 405# は「技術的には正しいが、保護壁を壊した」整理自体は半分正しい

これは完全に否定しない。  
3/12 から 3/13-3/14 にかけて sell AS 率と sell PnL が悪化したのは事実であり、**405# が悪化に寄与した可能性は高い**。

ただし正確には、

- 405# が intermediate cap を広げた
- もともと存在していた executor 後段の offset leak
- 3/13-3/14 の mixed-SHA / one-sided balance / route-to-kill interaction

が合成されている。

よって **405# 単独犯** ではない。

---

## §4 盲点の拾い上げ

### §4.1 `Sell Final` という列名が誤解を招く

415# の `Sell Final` は実質 `maker_price.last_offset_stages["final"]` に近く、**実際の発注直前最終 offset ではない**。  
実発注ではその後に

- EV offset
- velocity offset
- trending sell offset
- toxicity offset
- VG supplement
- alert offset
- sidecar offset

が加わる。  
よって `Sell Final = 0.50` と書くと「真の final も 0.50」と誤読しやすい。

### §4.2 Skip レコードの可観測性が足りない

`sell_dynamic_kill` レコードは

- `order_price=0`
- `offset_stages=null`
- `decision_path=null`
- `balance_forced_switch=false`

で、**なぜそこへ到達したのかがレコードだけでは追えない**。

このため今必要なのは、新しい戦略を足す前に

- `requested_side`
- `resolved_side`
- `resolved_side_reason`
- `preflight_block_reason`
- `gate_path`
- `one_sided_balance`

を skip record に残すこと。

### §4.3 `offset_ceiling_ratio_sell=0.50` A/B をそのまま始めるのは早い

現状は true final offset が ceiling を超えうる。  
この状態で ceiling 値だけ動かすと、

- maker_price 側の ceiling 差
- executor 後段 multiplier 差

が混ざってしまう。

**順番は「final clamp 修正 → same-SHA 再観測 → ceiling A/B」** が正しい。

### §4.4 `skip_gate_as_prob` 前提の議論は現行運用とズレる

今の `skip_gate.mode` は `pnl` であり、3/12-3/14 の filled record では `skip_gate_as_prob` が全件 `null` だった。  
したがって「AS 確率帯での skip 強化」は、

- 別 AS モデルを再導入する
- あるいは dual-model にする

という設計判断を伴う。  
現行 `pnl` モードの調整と同列には置けない。

---

## §5 推奨アクション

### P0

1. **executor 後段の true final offset clamp を実装**
   - `maker_price` ceiling 後の multiplier 群にも最終上限制御を掛ける
   - record に `maker_final_offset` と `execution_final_offset` を分けて残す
2. **`buy insufficient -> switch sell -> sell_dynamic_kill` の相互作用を遮断**
   - kill 中 side への即切替をやめる
   - idle/backoff または別の recovery path を選ぶ
3. **same-SHA / same-run_id で 3/12-3/14 を再集計**
   - 405# / 414# の効果判定はその後

### P1

4. **SkipGate を `mode=pnl` 前提で再診断**
   - side別 `threshold_used`
   - score quantile
   - skip率
   - pass PnL
5. **Skip record の可観測性を増やす**
   - `requested_side`, `resolved_side_reason`, `gate_path`, `one_sided_balance`
6. **その後に sell ceiling 0.35-0.40 を A/B**
   - ただし true final clamp 修正後に限る

### P2

7. **VG を bool でなく boost factor で評価**
8. **AS 事前予測モデルは second model として検討**
   - いきなり main gate を差し替えるより、補助 veto / offset premium で試す

---

## §6 最終判定

415# は、**「どこを直せば fill test の回復に近いか」を考えるための素材としては有益**。  
ただし、実装とログを照合すると優先順位は少し変わる。

最重要なのは

1. `sell ceiling` の数値そのもの
2. `sell_dynamic_kill` の閾値

ではなく、

1. **post-ceiling offset leak**
2. **one-sided balance から kill へ流れる route interaction**
3. **mixed-SHA 汚染を除いた再観測**

である。

一文でまとめると、  
**「415# の方向性は悪くないが、いま先にやるべきは ceiling 値の手探り調整ではなく、true final offset の制御と route-to-kill デッドロックの解消である」**。
