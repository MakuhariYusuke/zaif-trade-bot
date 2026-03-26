# 634# 628#-633# 収益性リアリティチェックと集中改善レビュー

## 0. 結論

率直に言うと、今は「完全に駄目」ではありませんが、**まだ“儲かるシステム”とも言いにくい**です。

`2026-03-25` の実績を既存分析で見ると、

- 全体: `556 orders / 107 fills / avg_pnl30=+0.09bps`
- buy: `+0.79bps`
- sell: `-0.85bps`

でした。

つまり今起きていることは、

1. **buy がそこそこ稼ぐ**
2. **sell がそれを削る**
3. しかも `preflight_insufficient=203`, `no_feasible_quote=118` で、良い場面にも十分参加できていない

です。

したがって、「儲からない」の正体はかなり明確で、

- **主犯は `sell × ranging`**
- 次点で **`preflight_insufficient / no_feasible_quote` の多さ**
- sidecar は今のところ **利益の本丸ではない**

です。

今回の 628#-633# を踏まえた私の主判定は次です。

- `629#` の sidecar cache fix は正しい
- `631#` の BPS 10x バグ修正も正しい
- `632#` の ATR floor 緩和も正しい
- ただし `633#` の「Era-D だけが黒字」「Era-E は clamp 爆発が主犯」「630# を巻き戻すべき」という結論は、**使えるが強すぎる**

今やるべきことは、広く作り直すことではなく、

**「sell/ranging を減らし、buy/ranging と sell/trending_up を取りにいく」**

へ寄せることです。

---

## 1. 文書別の判定

| 文書 | 判定 | 支持点 | 補正点 |
|---|---|---|---|
| `628#` | 部分支持 | 閾値スケール不適合の問題意識は妥当 | Zスコア動的化は P2 でよく、今すぐの収益レバーではない |
| `629#` | 支持 | stale→error バグ修正は正当。診断の信頼性を回復した | これは診断改善であり、直接の alpha 改善ではない |
| `630#` | 条件付き支持 | sell velocity / regime / VG を詰める方向性は理解できる | 3箇所同時変更で attribution が汚れた。個別因果はまだ弱い |
| `631#` | 強く支持 | `min_spread_floor_bps: 3.8 -> 0.38` は明白な正しい修正 | なし |
| `632#` | 支持 | ATR floor の緩和と cap 導入は妥当 | ただし現在の主 blocker は `spread_too_narrow` より `no_feasible_quote` |
| `633#` | 部分支持 | Era 発想、sell/ranging 毒性、clamp 監視は有益 | Era-D only 黒字論、Era-E 全否定、630# 巻き戻し論は言い過ぎ |

---

## 2. いま実際に何が起きているか

今回は新しい分析スクリプトを足さず、既存の

- `scripts.v460.analysis.analyze_fill_logs`
- `scripts.v460.analysis.tail_loss_analysis`

で確認しました。

## 2.1 全体像: 3/25 は「薄くプラス」だが健全ではない

`analyze_fill_logs --date-from 2026-03-25 --date-to 2026-03-25`

- 全体 `+0.09bps`
- buy `+0.79bps`
- sell `-0.85bps`
- fill rate `19.2%`

この時点で、今の最優先論点ははっきりしています。

- buy をさらに磨くより
- **sell の負けを止める方が早い**

です。

## 2.2 利益が出ている状態は既にある

同じ 3/25 の side/regime 分解では、

- `buy/ranging`: `Net=+0.69bps (n=48)`
- `sell/trending_up`: `Net=+4.53bps (n=5)`
- `buy/trending_down`: `Net=+2.27bps (n=4)`

でした。

逆に悪いのは、

- `sell/ranging`: `Net=-1.93bps (n=39)`

です。

ここから読み取るべきなのは、

- 「全部壊れている」ではない
- **勝てる状態は既に観測できている**
- ただし **損する状態にも同じくらい参加している**

ということです。

つまり利益への近道は、新しい大機構の導入より先に、

**負け状態の参加抑制**

です。

## 2.3 テール損失の正体

`tail_loss_analysis --date-from 2026-03-25 --date-to 2026-03-26`

sell tail:

- `tail mean=-11.59bps`
- `AS=100%`
- `decision_path=ev_offset 100%`
- `ranging 100%`
- `orderbook_imbalance tail_mean=-0.1519 vs total=-0.0173`

buy tail:

- `tail mean=-4.14bps`
- `AS=100%`
- `decision_path=primary_only 100%`
- `hour_skip candidate: UTC 06`

つまり、

- sell は **`ev_offset` 側の ranging 参加**
- buy は **`primary_only` 側の一部時間帯/一部状態**

が本丸です。

628#-633# の議論で一番大事なのは、ここを混ぜないことです。

---

## 3. 633# の使える点と危うい点

## 3.1 使える点

`633#` が useful なのは次です。

1. Era-D の `CalibrationMap` 期に良い挙動があったこと
2. `631#`, `632#` の修正が「必要な止血」だったこと
3. Era-E の一部で sell 側の clamp / sell/ranging が悪化したこと

これらは否定しません。

## 3.2 危うい点

ただし、次は強く補正した方がよいです。

### A. 「Era-D だけが黒字」は強すぎる

既存の公式 `pnl30` 集計で SHA 単位に見ると、

- `447b2ec...`: `+1.49bps`
- `c164d21d367b`: `+0.86bps`
- `2ac4d05...`: `+2.60bps`

でした。

つまり、**Era-E 側にも黒字 SHA はあります**。

633# の

- `2ac4d05 = 全 SHA 中最悪`

という整理は、少なくとも `pnl30` の official view とは一致しません。

これは 633# が完全に誤りというより、

- `JPY/fill`
- `pnl30`
- mixed window
- sample size

が混ざっていることによる見え方の差です。

### B. clamp は主因というより「症状」です

633# は clamp をかなり重く見ていますが、これは半分正しく、半分危険です。

なぜなら、実際には

- `447b2ec...` でも clamp は観測される
- `2ac4d05...` でも sell clamp subset は `100%` だが利益は `+2.86bps`

だからです。

つまり、

- clamp が高い = 必ず悪い

ではありません。

本質は

- **どの状態で**
- **どの side が**
- **どの path で**
- **clamp に当たっているか**

です。

今の実データでは、sell/ranging のときに負けていることの方が、clamp 単独より強い説明力を持ちます。

### C. 630# の全面巻き戻しはまだ早い

633# は `trend_threshold_pct` と `VG velocity` の部分ロールバックを強く推しています。

ただし current day では、

- `sell/trending_up = +4.53bps`
- `buy/ranging = +0.69bps`

です。

このため、

- trend 感度を戻す
- VG を緩める

をまとめてやると、**勝っている状態まで薄める**リスクがあります。

私の判定は、

- 630# の全面 rollback はまだ早い
- やるなら **sell/ranging に限った条件付き抑制** を先にやるべき

です。

---

## 4. 629#-632# の実装に対するコメント

## 4.1 `629#`: 正しいが、利益改善ではない

`scripts/v460/lib/sidecar_signal_io.py:186` の stale fix は正しいです。  
テストも

- `test_sidecar_sac_integration.py`
- `test_239_feasible_quote.py`
- `test_336_yaml_code_drift_prevention.py`

で `108 passed` を確認しました。

ただし、3/25 全体で sidecar status は

- `fresh=17`
- `stale=8`
- `error=82`

です。

しかも

- profitable な `2ac4d05` でも `error=100%`
- 負けた `ce31662` でも `error=100%`

でした。

要するに、**sidecar は今の利益差を説明していません**。  
直しておく価値はありますが、profit roadmap の先頭には置かない方がいいです。

## 4.2 `631#`: 明確に正しい

`min_spread_floor_bps: 3.8 -> 0.38` は明白に正しい修正です。  
ここは議論の余地がありません。

この修正がなければ、そもそも比較可能な fill が出ませんでした。

## 4.3 `632#`: `spread_too_narrow` は改善したが、次の blocker が出た

632# の ATR floor 緩和は有効です。  
ただし 3/25 の blocked reasons は

- `preflight_insufficient: 203`
- `no_feasible_quote: 118`
- `spread_too_narrow: 36`

です。

つまり今は、

- 631/632 の止血が効いて
- `spread_too_narrow` は主犯から降りた
- 代わりに **`no_feasible_quote` と在庫不足が前に出た**

という段階です。

ここを読み違えて spread 系だけ追い続けると、儲かる方向から外れます。

---

## 5. 今いちばん効く改善案

## P0: `sell/ranging` を直接削る

最優先はこれです。

理由:

- `sell/ranging = -1.93bps`
- `sell/trending_up = +4.53bps`
- sell tail `100%` が `ranging × ev_offset`

だからです。

やり方は、全体 rollback ではなく sell/ranging 限定で十分です。

候補:

1. `sell + ranging` の `skip_gate` threshold を追加で厳格化
2. `sell + ranging` の `min_spread` だけ条件付きで引き上げ
3. `sell + ranging` の `ev_offset` 参加を弱め、`primary_only` or skip 側へ寄せる

特に 3/25 sell の broad cancel reason は

- `preflight_insufficient 84`
- `no_feasible_quote 70`
- `skip_gate 33`

なので、**“売るな” ではなく “悪い sell だけ減らす”** 方が自然です。

## P1: `no_feasible_quote` と `preflight_insufficient` を alpha 問題として扱う

これは単なる運用ノイズではありません。

3/25 単日では

- buy unfilled の `preflight_insufficient=121`
- sell unfilled の `preflight_insufficient=84`
- buy unfilled の `no_feasible_quote=48`
- sell unfilled の `no_feasible_quote=70`

でした。

ここで重要なのは、

- buy は儲かっているのに `preflight_insufficient` が最も多い
- sell は負けているのに `no_feasible_quote` も多い

ことです。

したがって次にやるべきは、

1. **buy の実行機会を増やす**
2. **sell の無理な試行を減らす**

です。

具体的には、

- `no_feasible_quote` 連続時の side cooldown
- `preflight_insufficient` が続く side の優先度引き下げ
- `ranging` では buy 優先にする side selection

が、profit に直結しやすいです。

## P2: `630#` は “戻す” より “切り分ける”

今のデータだけで言うと、

- `trend_threshold_pct=0.20` を戻す
- `VG velocity=6.0` を戻す

より先に、

- **sell/ranging だけ別ルール**

を試す方が良いです。

なぜなら、

- buy は現にプラス
- sell/trending_up もプラス

だからです。

全体 rollback は、勝っている領域まで巻き込む可能性があります。

## P3: cross-venue は buy 側で活かし、sell 側では追わない

3/25 全体では

- buy CV widen avg `+2.12bps`
- sell CV は `cap_hit=8`, `widen=0`

でした。

つまり今の cross-venue は

- buy では改善寄与の可能性がある
- sell ではほぼ no-op

です。

なので当面は、

- sell に CV をもっと足す

ではなく、

- **buy の CV 効能を維持**
- **sell の CV は保守的に据え置き**

が妥当です。

---

## 6. 私ならどう進めるか

順番をはっきりさせます。

1. `629#`, `631#`, `632#` は維持  
   これは正しい止血で、戻す理由がないです。

2. `633#` の Era narrative は参考に留める  
   rollback 判断の主根拠にはしません。

3. 先に `sell/ranging` 限定で防御を強める  
   global rollback より副作用が小さいです。

4. 次に `preflight_insufficient / no_feasible_quote` を side selection 問題として詰める  
   ここが詰まる限り、勝ち状態への参加量が増えません。

5. sidecar は後ろに置く  
   直しておくのは賛成ですが、今の利益差は sidecar が作っていません。

---

## 7. 最終判定

628#-633# の流れをまとめると、今の局面はこうです。

- 629/631/632 は正しい
- 630 は悪手と断定するにはまだ早い
- 633 は useful だが rollback 論が強い
- 本当の本丸は **sell/ranging**
- その次が **`preflight_insufficient` と `no_feasible_quote`**

つまり、「どうしたら良いか」への答えは、

**大きく作り直す前に、sell/ranging を切り、buy/ranging を取り、無理な side 試行を減らす**

です。

今のデータはそこまで言えるだけの材料が揃っています。

---

## 8. 今回確認したもの

- `docs/v460/628_cplt_second_opinion_626_627_structural_review.md`
- `docs/v460/629_cplt_review_evaluation_sidecar_cache_fix.md`
- `docs/v460/630_cplt_p1_threshold_tuning.md`
- `docs/v460/631_cplt_min_spread_bps_floor_10x_fix.md`
- `docs/v460/632_cplt_p2_atr_floor_log_improvements.md`
- `docs/v460/633_cplt_post_restart_analysis.md`
- `scripts.v460.analysis.analyze_fill_logs --date-from 2026-03-25 --date-to 2026-03-25`
- `scripts.v460.analysis.analyze_fill_logs --date-from 2026-03-25 --date-to 2026-03-25 --side sell`
- `scripts.v460.analysis.analyze_fill_logs --date-from 2026-03-25 --date-to 2026-03-25 --side buy`
- `scripts.v460.analysis.analyze_fill_logs --date-from 2026-03-16 --date-to 2026-03-25 --git-sha 447b2ec50a18489ccde042095d2889587f2813d0`
- `scripts.v460.analysis.analyze_fill_logs --date-from 2026-03-16 --date-to 2026-03-25 --git-sha c164d21d367b`
- `scripts.v460.analysis.analyze_fill_logs --date-from 2026-03-16 --date-to 2026-03-25 --git-sha ce31662dfa7b22ff5497caa73cf33349d00cd7d3`
- `scripts.v460.analysis.analyze_fill_logs --date-from 2026-03-25 --date-to 2026-03-25 --git-sha 2ac4d05ce2e35efd71d027a29b6feafecb934ce1`
- `scripts.v460.analysis.tail_loss_analysis --date-from 2026-03-25 --date-to 2026-03-26`
- `pytest tests/unit/v460/test_sidecar_sac_integration.py tests/unit/v460/test_239_feasible_quote.py tests/unit/v460/test_336_yaml_code_drift_prevention.py --no-cov`
  - `108 passed`
