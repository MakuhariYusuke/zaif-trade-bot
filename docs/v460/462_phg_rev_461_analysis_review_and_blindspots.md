# 462# 461# レビュー再整理: 考察の妥当性と残る盲点

**種別**: rev  
**対象**: 461# skip_gate_evaluator Mixin 分割 + Fill Test 10日間深堀り分析  
**日付**: 2026-03-18

---

## §0 結論

461# の考察は、症状の捉え方としてはかなり良い。特に以下は有益である。

1. **3/12 をピーク、3/13 以降を悪化局面と見たこと**
2. **3/16→3/17 を再起動境界として観察したこと**
3. **fill rate の回復と PnL の回復が一致しないことを明示したこと**
4. **deep-night 帯の tail loss を時間帯で可視化したこと**
5. **balance_switch 群や cross-venue 群を症状として分けて見たこと**

ただし、実装判断に直結させるにはまだ危うい。理由は単純で、461# の結論の一部が

- schema drift
- run drift
- config drift
- population drift

を十分に補正しないまま導かれているからである。

したがって本稿の最終判断は次の通り。

- **461# は観察レポートとして有用**
- **だが、因果レポートとしてはまだ未完成**
- **次にやるべきは機能追加ではなく、分析母集団の固定化である**

---

## §1 461# のどこが妥当か

### §1.1 収益悪化の症状把握は正しい

461# が描いた以下の構図は、方向として妥当である。

- 3/13 以降に fill rate / PnL が悪化
- 3/17 の前半は `ranging_low_vol_skip` が全面発火
- 3/17 後半は fill は戻ったが PnL が戻っていない
- sell 側の悪化が全体損益を引っ張っている
- deep-night 時間帯の tail risk が重い

これらは症状として見る限り、十分に価値がある。

### §1.2 「安全装置が増えたのに利益が出ない」は重要な問いである

461# は、

- clamp
- cross-venue
- balance switch
- ranging_low_vol_skip
- hot-reload 系変更

が重なった後に、むしろ利益が悪化している可能性を疑っている。この問い自体は正しい。むしろ今の局面では最も重要な問いのひとつである。

問題は、その問いに対する答えを **まだ強く言いすぎている** 点にある。

---

## §2 HIGH: schema drift を跨いだ比較が未補正

461# の最大の盲点はここである。

### §2.1 観測フィールドの導入時期が揃っていない

本稿で比較に使っている主要フィールドの一部は、後から追加されたものである。

- `execution_pre_clamp_offset` は 421# 系で追加
- `start_git_sha` と `resolved_side_reason` は 420# 系で追加
- `cross_venue_lead_lag_applied` は cross-venue 系の導入後でないと存在しない

したがって、古い SHA の fill_records を新しい SHA と同列に並べると、

- 未記録
- `None`
- `False`

が実質的に「ゼロ発火」「未適用」と見えてしまう。

### §2.2 461# のどの結論がこの影響を受けるか

特に危ないのは以下である。

1. **「旧SHAは ceiling clamp 0%」**
2. **「旧SHAは balance switch 0%」**
3. **「旧SHAはシンプルだから良かった」**

これらは現状だと、

> 本当にゼロだった

のではなく、

> その観測列がまだ無かったのでゼロに見える

可能性を排除できない。

### §2.3 含意

このため、461# の

> 「好成績 SHA の共通項 = ceiling clamp ゼロ」

という結論は、現段階では **仮説** に留めるべきである。ここを事実として書いてしまうと、次の実装判断を誤る。

---

## §3 HIGH: run drift を current git_sha だけで切っている

### §3.1 hot-reload 時代の分析単位として不十分

現行 FillRecord には、すでに

- `run_id`
- `git_sha`
- `start_git_sha`

がある。それにもかかわらず、461# が依拠する `temp/analyze_fill_test_deep_v2.py` は、実質的に `git_sha[:7]` ベースで集計している。

しかし hot-reload が入っている以上、同一 run 中で current `git_sha` は変わりうる。従って

- current `git_sha` で切る
- run を跨いで比較する
- restart 境界も同時に論じる

というやり方では、因果がまだ緩い。

### §3.2 461# のどの主張が影響を受けるか

特に影響が大きいのは、

- 5.2 SHA 比較
- 5.4 再起動境界分析
- 5.7 f840d0e 構造分析

である。

### §3.3 必要な是正

次版では少なくとも、

1. `run_id` 固定
2. `start_git_sha` 固定
3. 必要なら `current git_sha` を副指標扱い

とするべきである。

---

## §4 HIGH: config drift がまだ未観測

これは 461# で明示されていないが、非常に大きい盲点である。

### §4.1 同じ SHA でも同じ設定とは限らない

459# で hot-reload の到達性が広がった以上、同一 `git_sha` の中でも YAML 変更によって挙動が変わりうる。

ところが FillRecord 側には、少なくとも本分析で直接使える

- `config_hash`
- `config_version`
- `hot_reload_seq`

のような識別子が載っていない。

つまり、たとえ `run_id` と `start_git_sha` を固定しても、

> 同じ run / 同じ code だが設定だけ違う

という混線が残る。

### §4.2 含意

461# の

- d0769f2 前半 7 時間
- f840d0e 後半 11 時間

の差を code 起因だけで読むのは危険である。ここには config drift が混ざっている可能性がある。

### §4.3 結論

今後の fill analysis では、`run_id + start_git_sha` だけでは足りず、**config 識別子も必要** である。これは本稿で追加すべき重要盲点である。

---

## §5 HIGH: population drift を見ずに filled だけで語っている箇所がある

### §5.1 filled-only 分析は有用だが、それだけでは危ない

461# の後半では、

- EV bin
- cross-venue applied vs not applied
- balance_switch vs normal

などの比較をしている。これは面白い。

ただし、これらの多くは **filled population だけ** を対象としている。すると、

- 何が約定候補として残ったのか
- 何が gate で落とされたのか
- どの層で母集団が変形したのか

が見えない。

### §5.2 具体例

たとえば EV 0.5-1.0 帯が空白だとしても、それは

- EV 計算の不連続
- その帯だけ gate/cancel で落ちた
- 偶然 sample が少ない

のどれでも説明できる。filled だけでは切り分けられない。

cross-venue も同様で、`cv_applied` 群はもともと hint 条件を通過した「分かりやすいケース」に偏る。

### §5.3 含意

461# のこれらの分析は、実装示唆としては useful だが、**原因断定には未達** である。次版では

- raw
- attempted / processed
- filled

の 3 母集団を並べて出すべきである。

---

## §6 MEDIUM: 市場局面 drift を旧SHA比較で十分に補正していない

3/8-3/9 と 3/17 を比べて、

- 旧SHAは良かった
- 現行SHAは悪い

と言うのは分かりやすい。だが、その期間には

- 上昇トレンドの強さ
- spread 環境
- ボラティリティ
- 参加者の板厚

が変化している可能性が高い。

よって、旧SHA比較は参考にはなるが、**matched-market comparison** ではない。時間帯や regime や spread 帯を揃えずに「単純構成の方が強かった」と断ずるのはまだ早い。

---

## §7 MEDIUM: 461# の提案群で慎重に扱うべき点

### §7.1 Buy ceiling 緩和は筋があるが、根拠はまだ弱い

`buy ceiling 0.20 -> 0.35` は魅力的だが、現状では

- clamp fired 群の pre/post 比較
- side / regime 固定
- filled 以外を含めた母集団比較

が不足している。今のままだと、tight clamp 問題を見ているのか、市場局面悪化を見ているのかがまだ分かれていない。

### §7.2 Cross-venue 適用率引上げは飛躍がある

461# は `cv_applied` 群の改善から、適用率 42% → 70% という方向を示唆しているが、selection bias が強い。ここは「適用率を上げる」より「適用条件の効用を同一母集団で再検証する」が先である。

### §7.3 EV 0.5-1.0 帯空白は「疑い」止まり

ここも面白いが、filled-only では断定不能である。追加調査テーマとしては良いが、現時点で pipeline bug の証拠とまでは言えない。

---

## §8 それでも 461# が持つ価値

批判だけでは不公平なので、残すべき価値も明確に書く。

### §8.1 実装優先順位のヒントは十分にある

461# から引き出せる、比較的安全な示唆は以下である。

1. **deep-night 帯は別管理にすべき**
2. **balance_switch 群は少なくとも症状として悪い**
3. **cross-venue は完全 no-op ではない可能性が高い**
4. **fill rate 改善だけでは利益改善の保証にならない**
5. **ranging 系 gating は戦略の主戦場と衝突しやすい**

これらは因果断定なしでも十分に使える。

### §8.2 461# を次に繋げるなら、分析基盤の修正が先

本稿の真の次アクションは、新しい guard や boost を増やすことではない。まずは

- schema-aware 集計
- run-aware 集計
- config-aware 集計
- population-aware 集計

を入れて、同じ議論を再計測することである。

---

## §9 最終判断

461# の考察に対する私の最終評価は以下である。

- 症状観察: **高評価**
- 問題提起: **妥当**
- 因果推定: **未完成**
- 実装提案への直結度: **まだ低い**

### §9.1 追加で確認できた盲点

461# に対して、今回新たに強く確認した盲点は次の 5 つである。

1. schema drift
2. run drift
3. config drift
4. population drift
5. market regime drift

### §9.2 462# として残すべき結論

> 461# は「何が悪そうか」をかなり正しく炙り出しているが、まだ「何が犯人か」を断定する段階には達していない。次に必要なのは guard を増やすことではなく、比較単位を固定した再分析である。

### §9.3 実務上の次アクション

| 優先 | アクション | 目的 |
|---|---|---|
| P0 | `analyze_fill_test_deep_v2.py` を schema-aware 化 | late-added field 擬似ゼロ問題の除去 |
| P0 | `run_id` / `start_git_sha` 固定モード追加 | hot-reload / restart 混線の排除 |
| P0 | config 識別子の追加または分析側補助 | config drift の可視化 |
| P1 | raw / processed / filled の 3 母集団を並列出力 | population drift の可視化 |
| P1 | side / regime / hour 固定の matched comparison | market regime drift の除去 |

本稿の結論は、461# を否定することではない。**461# を、次の実装判断に耐える分析へ引き上げるための補助線**である。

---

## §10 463# 検証結果 (2026-03-18 追記)

462# の各指摘を `temp/analyze_fill_test_v3.py` (schema-aware 分析スクリプト) で実データ検証した結果を以下に記録する。

### §10.1 検証サマリー

| 462# 指摘 | 判定 | 詳細 |
|---|---|---|
| §2 Schema Drift | **✅ CONFIRMED** | 3/8-3/14: `execution_pre_clamp_offset`, `cross_venue_lead_lag_applied`, `resolved_side_reason`, `start_git_sha` が全て不在。461# の「ceiling 0%」「balance_switch 0%」は計測不能の擬似ゼロ |
| §3 Run Drift | **✅ CONFIRMED** | `run_id` は全 date に存在 (28 runs)、`start_git_sha` は 3/15+ のみ。run `1772917391_66b6e3a0` (3/8) は 2 SHA (eb24cf4, fea7911) = hot-reload 確認 |
| §4 Config Drift | **Valid (未検証)** | FillRecord に config hash/id は不在。構造上の欠損として妥当 |
| §5 Population Drift | **⚠ PARTIALLY CONFIRMED** | EV 0.5-1.0 帯: 全母集団 (filled+cancel) でも f840d0e, d0769f2 は 0 件 — **構造的空白が確定** (旧 SHA は 7-13%)。CV: selection bias は sell 側で裏付け (cv_on=-2.60 vs cv_off=-2.05) |
| §6 Market Regime | **Valid** | 異なる市場条件の比較であることは暗黙の前提として残る |
| §7 Implementation Caution | **⚠ PARTIALLY VALID** | ceiling 緩和は market 要因を排除できていない。CV 適用率引上げは buy 側のみ有効 |

### §10.2 462# が予見していなかった新発見

1. **eb24cf4 balance_forced_switch 問題**: `balance_forced_switch=True` が 21/89 fills (23.6%)。461# は `resolved_side_reason` (不在フィールド) で 0% とし、462# もこれを「擬似ゼロ」と指摘したが、**実際には True 値がフィールドには存在していた**。しかもそのPnLは+0.1bps (normal -0.06bps より良好)

2. **balance_forced_switch vs resolved_side_reason の二重計測問題**: f840d0e は `balance_forced_switch=None` (全88件) だが `resolved_side_reason="balance_switch"` が 51件。旧 SHA は `balance_forced_switch=True/None` のみ。**2つのフィールドが異なる計測系であることが判明**

3. **d0769f2 ceiling = 100%**: 461# は 83.3%/50.0%、462# は特に修正提案なし。v3 で pre_clamp 存在 fills に限定再計測した結果 100% (buy 5/5, sell 3/3)

4. **sell_dynamic_kill の非単調変動**: 3/12 bff652e=0% → 3/13 bff652e=73%、同日の 5c3238f=41%。SHA が同じでも日によって大きく異なり、「特定変更が kill を消した」という因果は成立しにくい

5. **CV 効果の sell 逆効果**: buy ではcv_on=+0.25bps vs cv_off=-1.43bps (有効)、sell では cv_on=-2.60bps vs cv_off=-2.05bps (CV 適用時のほうが悪い)。462# §7.2 の selection bias 懸念は **sell で逆効果を示す形で裏付けられた**

### §10.3 462# P0/P1 アクションの完了状況

| アクション | 状況 |
|---|---|
| P0: schema-aware 化 | **✅ 完了** — `temp/analyze_fill_test_v3.py` 作成・実行済 |
| P0: run_id / start_git_sha 固定モード | **✅ 完了** — v3 PART2 で 28 runs の run-based 分析実施 |
| P0: config 識別子追加 | **⬜ 未着手** — FillRecord への config_hash 追加が必要 |
| P1: 3母集団並列出力 | **✅ 部分完了** — v3 PART4 で filled+cancel の全母集団 EV 分析実施 |
| P1: matched comparison | **⚠ 部分実施** — v3 PART6 で side/regime 固定の CV 分析実施。hour 固定は未実施 |
