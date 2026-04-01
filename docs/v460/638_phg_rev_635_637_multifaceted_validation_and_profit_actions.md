# 638# 635#-637# 多角的検証レビューと収益改善アクション

## 0. 結論

`637#` はかなり詰められており、主診断の方向性は概ね正しいです。
ただし、そのまま実行計画に落とすには補正が必要です。

今回の私の結論は次の通りです。

1. `635#` と `636#` の実装自体は repo 上で確認でき、関連テストも通りました
2. `637#` の中核主張である
   - `skip_rate_limit` による保護無効化
   - `buy freeze` の多発
   - tail loss 集中
   は、ログと既存分析で裏付けられます
3. ただし `635#` の前提だった「`buy/ranging` は明確に黒字」は、`600e1b2e0f9f` 時点では既に崩れています
4. したがって次の一手は、`sell/ranging` だけを叩くより、
   **`ranging` 全体の参加品質改善 + `skip_rate_limit` の局所緩和 + preflight 起因の side 歪み修正**
   に寄せるべきです

言い換えると、

- `637#` は「どこが痛いか」の診断として強い
- ただし「何を最初に変えるべきか」は、もう一段だけ慎重に切るべき

です。

---

## 1. 文書別判定

| 文書 | 判定 | コメント |
|---|---|---|
| `635#` | 条件付き支持 | 当時の 634# 前提に対する外科的対応としては妥当。ただし現在は `buy/ranging` も赤字化しており、思想をそのまま固定化しない方がよい |
| `636#` | 支持 | バグ修正と設定整備は正当。`offset_ceil=0.8` への是正も必要だった |
| `637#` | 強く支持だが一部補正 | 根本原因の掘り下げは今回範囲で最も強い。ただし P0 提案 2 件は「方向は良いが粒度がやや粗い」 |

---

## 2. 裏付けできたこと

### 2.1 `635#` / `636#` の実装は入っている

コード上で以下を確認しました。

- `scripts/v460/lib/side_selector.py:98`
  - `ranging` で `sell` 基底のとき `buy` 優先へ切替
- `scripts/v460/lib/skip_gate_evaluator.py:690`
  - `sell × ranging` に `skip_gate_sell_ranging_offset` を加算
- `scripts/v460/lib/orchestrator_balance.py:84`
  - 残高不足 side を `freeze_side(..., cycles=self.config.balance_freeze_cycles)`
- `configs/v460/fill_test.yaml:523`
  - `sell_ranging_offset: 0.5`
- `configs/v460/fill_test.yaml:526`
  - `offset_ceil: 0.8`
- `configs/v460/fill_test.yaml:985`
  - `balance_freeze_cycles: 3`

関連テストも確認しました。

- `tests/unit/v460/test_634_sell_ranging_suppression.py`
- `tests/unit/v460/test_336_yaml_code_drift_prevention.py`
- `tests/unit/v460/test_fill_quality.py`

結果:

- `33 passed`

したがって、`636#` の「修正は反映済み」という前提は支持できます。

### 2.2 `637#` の `skip_rate_limit` 問題は実在する

`ztb/ml/skip_gate.py:476` では、side 別直近履歴の skip 率が `max_skip_rate` を超えると、`should_skip=True` でも `force-pass` になります。

同日ログでも、実際に以下が大量に出ています。

- `sell/ranging: pnl=-5.811 th=0.600 regime_floor=0.600 -> skip_rate_limit(33%>30%)`
- `sell/ranging: pnl=-5.145 th=0.600 regime_floor=0.600 -> skip_rate_limit(38%>30%)`
- `sell/trending_down: pnl=-4.669 th=-0.100 regime_floor=-0.100 -> skip_rate_limit(35%>30%)`

つまり `637#` の

- 「skip_gate は損失を予測した」
- 「しかし rate limiter が通してしまった」

という整理は、概ね支持できます。

### 2.3 `buy freeze` 多発も実在する

`results/v460/fill_test/logs/fill_test.log` では、`2026-03-26` に

- `Freezing buy for 3 cycles`

が大量に出ています。`637#` の問題意識どおり、残高制約が side 配分をかなり歪めています。

さらに設計面で重要なのは、`scripts/v460/lib/fill_loop_orchestrator.py:331` です。
`preflight_insufficient` などの skip でも `update_last_side=True` の経路では、**約定していない side を last_side として記録**します。

これは「実際に成立した売買」と「失敗した試行」を同じ state に混ぜるため、side 選択の時系列依存を濁します。
ここは `637#` が明示していない、かなり重要な盲点です。

### 2.4 tail loss 集中も支持できる

既存分析で `600e1b2e0f9f` を見ると、

- 全体: `72 fills / avg_pnl30=-0.51bps`
- Worst 5 fills: `-65.72bps`
- Total loss に対する worst 5 の寄与: `47%`

であり、tail 集中は明確です。

また `tail_loss_analysis` でも、`2026-03-26` の tail は

- sell tail: `AS=100%`, `decision_path=ev_offset 100%`
- buy tail: `AS=100%`, `decision_path=primary_only 100%`

でした。

この点でも `637#` はかなり正確です。

---

## 3. `637#` に対する補正

### 3.1 `buy/ranging` はもう「安全地帯」ではない

ここが今回いちばん重要な補正です。

`635#` は 634# の時点で

- `buy/ranging` は取る
- `sell/ranging` を削る

という思想でした。これは当時は合理的でした。

しかし `600e1b2e0f9f` を既存分析で見ると、

- `buy/ranging`: `n=21, avg_pnl30=-1.03bps`
- `sell/ranging`: `n=25, avg_pnl30=-1.20bps`

です。

つまり今は

- `sell/ranging` の方が悪い
- ただし `buy/ranging` ももう明確なプラスではない

状態です。

したがって、`635#` の `ranging buy priority` は今後、

- 「positive alpha へ寄せる装置」
ではなく
- 「より悪い sell/ranging を避ける damage control」

として扱う方が正確です。

### 3.2 `max_skip_rate 0.30 -> 0.50` は方向は良いが、少し粗い

`637#` の P0 提案は理解できます。
実際、`skip_rate_limit` で損なわれている防御は明確です。

ただし `0.50` への一括引き上げは、`sell/ranging` には効いても、他 bucket まで一緒に抑え込みます。
実ログでも `buy/ranging` 側に

- `skip_rate_limit(40%>30%)`

が出ています。

したがって私の判定は、

- `0.30 -> 0.50` の方向性自体は支持
- ただし本命は **global 引き上げではなく、sell あるいは sell/ranging にだけ skip budget を多めに与える設計**

です。

もし今すぐ YAML だけで触るなら `0.40` までの ladder が先で、`0.50` は second move の方が安全です。

### 3.3 `balance_freeze_cycles 3 -> 1` は合理的だが、根治ではない

`637#` の指摘どおり、3 cycle は長いです。
特に `buy` 側が足りない場面では、`ranging_buy_priority` を実質的に潰しています。

ただし、ここも少しだけ補正が必要です。

本質は単に `3 -> 1` ではなく、

- **preflight failure を executed side と同じ state に入れていること**
- **freeze の固定 cycle 長が、資金状態ではなく時間で決まっていること**

です。

したがって、より本質的な設計案は次です。

1. `last_executed_side` と `last_attempted_side` を分離する
2. `preflight_insufficient` では `last_executed_side` を更新しない
3. `balance_freeze_cycles` は固定値ではなく、残高回復条件で解除するか、少なくとも `1` を初期値にする

`3 -> 1` はやる価値がありますが、**真の根治は state separation** です。

---

## 4. 三者が見落としていた可能性が高い点

### 4.1 `buy/trending_down` の良い alpha を clamp が止めている

`600e1b2e0f9f` の buy 側を見ると、

- `buy/trending_down`: `n=13, avg_pnl30=+1.48bps`

で、この日唯一はっきり良い bucket です。

一方、buy の cancel reason では

- `final_clamp_hard_skip: 7`
- 内 `buy/trending_down: 6`

でした。

これはかなり重要です。

いまは `sell/ranging` を止めることに目が向きがちですが、それと同じくらい、

**勝っている `buy/trending_down` を clamp で取り逃がしていないか**

を見るべきです。

この点は `635#` `636#` `637#` のいずれでも十分に前面化されていません。

### 4.2 cross-venue は buy 側でむしろ悪化要因の疑いがある

同じ `600e1b2e0f9f` では、buy 側の cross-venue は

- `CV applied: 20/35 fills`
- `widen PnL avg=-2.20bps`

でした。

したがって、現時点では cross-venue を広げるより、**buy 側 widen の負寄与をまず再点検**すべきです。

市場理論的にも、lead-lag 系の widen は「先行情報を取りに行く」より「遅れて悪い場所に参加する」危険があり、HFT では扱いを誤ると単なる adverse selection 追認になります。

### 4.3 sidecar は復旧傾向だが、まだ profit driver ではない

`600e1b2e0f9f` の sidecar 状態は

- `fresh: 24`
- `stale: 48`
- `error: 0`

で、629# 以前の `error` 連発からは改善しています。

ただし `stale 66.7%` は依然重く、しかも本日の損失主因は `skip_rate_limit`、`ranging`、`preflight` 側で説明できます。

よって sidecar は

- 重要だが P0 ではない
- 今は収益エンジンではなく、診断補助として扱う

のが妥当です。

---

## 5. 金融工学・市場理論から見た提案

### 5.1 `skip budget` を bucket 別にする

今の `max_skip_rate` は side 履歴のみで一律です。
しかし実損は

- `sell/ranging` は強く守りたい
- `buy/trending_down` はむしろ通したい

という非対称です。

したがって次段は、

- `max_skip_rate_sell`
- 可能なら `max_skip_rate_sell_ranging`

のような **bucket 別 skip budget** です。

これは市場理論的にも自然です。maker は状態依存で participation rate を変えるべきで、危険 bucket まで同じ participation constraint を課すのは非効率です。

### 5.2 timeout は side/regime 別にする

`637#` の `max_order_wait 25s` は支持します。
ただし fixed 25s より、

- `sell/ranging`: 短め
- `buy/trending_down`: やや長めでも許容

の方が理にかないます。

今回 same-SHA でも sell の `timeout` は `7` 件あり、`sell` は wait が長いほど stale fill 化しやすいです。微視的には、queue 後方で待ち続けるほど informed flow に踏み抜かれやすくなります。

### 5.3 `ranging_buy_priority` より `ranging participation quality` へ

今の ranging は片側誘導だけでは足りません。
両 side とも負けているため、次に見るべきは

- spread floor
- fill_prob
- queue age
- skip budget
- cross-venue widen

です。

言い換えると、`ranging` に対しては

- side の最適化
ではなく
- **参加品質の最適化**

に軸足を移すタイミングです。

---

## 6. 優先度付きアクション

> **686# 時点ステータス更新** (2026-04-02)

| 優先度 | 提案 | 判定 | 理由 | **686# 状態** |
|---|---|---|---|------|
| P0 | `preflight_insufficient` で `last_executed_side` を汚さない設計へ補正 | 強く推奨 | state 汚染を止めない限り、freeze 長だけ調整しても根が残る | ❌ **未着手**。687# Codex タスク (`687_codex_task_state_separation.md`) で対応予定 |
| P0 | `balance_freeze_cycles: 3 -> 1` | 推奨 | 応急処置として有効。`buy freeze` 過多を緩める | ✅ 641# で 3→1 完了 |
| P0 | `max_skip_rate` は一気に `0.50` ではなく、まず `0.40` で確認 | 推奨 | global 0.50 は粗い。勝ち bucket まで starving する恐れ | ⏭️ 一時 0.40 → 674# で 0.30 に復元（CI で PnL 改善なし） |
| P1 | bucket 別 skip budget (`sell` ないし `sell/ranging`) | 強く推奨 | 今回範囲で最も理にかなう次段階 | ❌ 未着手 |
| P1 | timeout の side/regime 別短縮 | 推奨 | stale adverse selection の抑制に直結 | ❌ 未着手 |
| P1 | `buy/trending_down` の `final_clamp_hard_skip` 再点検 | 強く推奨 | 勝ち筋を clamp で消している疑いが強い | ✅ 641# で hard_skip_mult=4.0 override |
| P2 | cross-venue widen の buy 側縮小または veto 強化 | 推奨 | same-SHA では buy の寄与が悪い | ✅ 641# offset_boost=1.0 で実質 widen 無効化 |
| P2 | sidecar stale 改善 | 据え置き | 必要だが、今の収益ボトルネックではない | ✅ 372# TTL=7800s で active 率改善 |

---

## 7. 総括

`637#` はかなり良いです。
特に

- `skip_rate_limit`
- `buy freeze`
- tail loss 集中

の三点を結びつけたのは、今回範囲では最も本質に近い整理でした。

一方で、次の補正が必要です。

1. `buy/ranging` はもう無条件に守るべき alpha ではない
2. `max_skip_rate 0.50` は方向は良いが少し粗い
3. 本当に直すべき盲点は `preflight_insufficient` 時の state 汚染
4. 利益を伸ばすなら `buy/trending_down` の clamp 側も見逃さない

したがって私の最終判定は、

- `637#` の診断: **支持**
- `637#` の P0 提案: **方向支持、ただし一段だけ細かくしてから入れる**
- 追加の独自提案としては **state separation** と **bucket 別 skip budget** が最重要

です。

---

## 8. 確認に使ったもの

- `docs/v460/635_cplt_profitability_focus_and_alpha_execution.md`
- `docs/v460/636_cplt_review_635_sell_ranging_bugfix.md`
- `docs/v460/637_cplt_post_restart_triple_loss_analysis.md`
- `configs/v460/fill_test.yaml`
- `scripts/v460/lib/side_selector.py`
- `scripts/v460/lib/skip_gate_evaluator.py`
- `scripts/v460/lib/orchestrator_balance.py`
- `scripts/v460/lib/fill_loop_orchestrator.py`
- `ztb/ml/skip_gate.py`
- `results/v460/fill_test/logs/fill_test.log`
- `python -m scripts.v460.analysis.analyze_fill_logs --date-from 2026-03-26 --date-to 2026-03-26`
- `python -m scripts.v460.analysis.analyze_fill_logs --date-from 2026-03-26 --date-to 2026-03-26 --git-sha 600e1b2e0f9f`
- `python -m scripts.v460.analysis.analyze_fill_logs --date-from 2026-03-26 --date-to 2026-03-26 --git-sha 721d7181f5d9db03d5060b3eca77e43134ea91f5`
- `python -m scripts.v460.analysis.tail_loss_analysis --date-from 2026-03-26 --date-to 2026-03-26`
- `pytest tests/unit/v460/test_634_sell_ranging_suppression.py tests/unit/v460/test_336_yaml_code_drift_prevention.py tests/unit/v460/test_fill_quality.py -k '634 or yaml or velocity_threshold' --no-cov`

テスト結果:

- `33 passed`
