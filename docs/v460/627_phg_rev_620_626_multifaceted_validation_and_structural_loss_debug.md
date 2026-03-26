# 627# 620#-626# 多角的検証レビューと構造損失デバッグ

## 0. 結論

まず結論です。

- `620#`-`625#` で行われた修正のうち、**壊れていた挙動を正した部分は概ね妥当**です。特に `skip_gate regime_thresholds bypass`、`sidecar ceiling 適用順序`、`hard_skip_utc_hours` 廃止、`kill duration drift` 修正、`min_spread` の動的化は方向として支持できます。
- ただし、**「大損バグは除去できた」ことと「残損の主因まで特定できた」ことは別**です。`626#` の問題意識は鋭い一方、結論はやや sell 側に寄り過ぎています。
- 既存の公式分析スクリプトで `2026-03-22`-`2026-03-25` を見ると、`pnl30` ベースでは `buy=-49.23bps`, `sell=-12.44bps` であり、**broad window では buy も十分に負けています**。
- その一方で、`2026-03-25` の主力 SHA `ce31662dfa7b22ff5497caa73cf33349d00cd7d3` に絞ると、`buy=+0.40bps`, `sell=-1.24bps` で、**現行の残課題は sell / ranging / ev_offset のテール損失**と読むのが自然です。
- さらに、`621#` で revive を狙った sidecar は live 上では実質止まっています。`cache/sidecar_signal.json` は `2026-03-24T11:42:39Z` の stale 内容のままで、`2026-03-25` 主力 SHA の `59 fills` は **全件 `sidecar_signal_status=error`** でした。

要するに、今の局面は

1. 大損バグの止血は前進  
2. しかし残損の本丸は「sell 単独」ではなく  
3. **AS テール + ranging 実行品質劣化 + clamp 飽和 + sidecar 健康不良**  

という整理が最も無理のない結論です。

---

## 1. 文書別の判定

| 文書 | 判定 | 支持できる点 | 補正が必要な点 |
|---|---|---|---|
| `620#` | 支持 | `skip_gate` の regime floor 強制、sidecar ceiling 順序修正は妥当 | 収益改善の証明ではなく correctness fix |
| `621#` | 条件付き支持 | NormLoader 統合、entry_gate observe 接続は実装方向として良い | 実運用上の sidecar revive は未達 |
| `622#` | 支持 | SAD/MCB 有効化、診断ログ補強は必要な安全層整備 | 安全強化であって alpha 改善とは切り分けるべき |
| `623#` | 支持 | hard skip を動的防御へ委譲する思想は妥当 | mixed-SHA 窓には旧 hard skip 痕跡が混ざる |
| `624#` | 支持 | kill drift 修正、ATR floor 導入は理にかなう | 単独では残損の主因に届かない |
| `625#` | 条件付き支持 | BPS floor + ATR floor + absolute floor の三層化は市場理論と整合 | starvation / fill減少とのトレードオフ監視が要る |
| `626#` | 部分支持 | sell 側テール観察、AS 重視、レイヤー不発火への着眼は良い | sell 単独犯論、velocity 主犯論、時間軸混在は言い過ぎ |

---

## 2. コード上の裏付け

`620#` の regime threshold fix は、`ztb/ml/skip_gate.py:670` で `regime_floor` を `max(regime_floor, adaptive)` で強制する形になっており、文書記載と整合しています。

sidecar ceiling 順序修正も、`scripts/v460/lib/multiplicative_pipeline.py:282` で final clamp 後に sidecar を注入する構造になっており、こちらも実装は確認できました。

`625#` の三層 min spread は、`scripts/v460/lib/maker_price.py:1051` にある通り

- `min_spread_jpy`
- `min_spread_floor_bps`
- `min_spread_atr_mult`

の `max()` で効く形です。理論ラベルも

- Stoll (1978) order processing cost
- Glosten-Milgrom 的 adverse selection cost

として置かれており、実装意図は一貫しています。

---

## 3. 公式集計で見た実態

今回は新しい分析スクリプトは足さず、既存の

- `scripts.v460.analysis.analyze_fill_logs`
- `scripts.v460.analysis.tail_loss_analysis`

で再確認しました。

### 3.1 `2026-03-22`-`2026-03-25` broad window

`analyze_fill_logs --date-from 2026-03-22 --date-to 2026-03-25`

- `total_records=1553`
- `filled=446`
- `avg_pnl30_bps=-0.128`
- `git_sha_unique=22`

side 別:

- `buy`: `253 fills`, `avg_pnl30=-0.19bps`
- `sell`: `195 fills`, `avg_pnl30=-0.06bps`

ここでまず重要なのは、**626# の「sell が損失の 95%」は、少なくとも標準的な `pnl30` 集計では再現しない**ことです。  
これは 626# が誤りというより、`temp/analyze_625_deep.py` + `PnL_120s` で見た像と、運用側の主集計軸が異なるという意味です。

したがって broad window では

- 「sell が痛い」
- しかし「buy も普通に負けている」

の両方を保持すべきです。

### 3.2 `2026-03-25` 主力 SHA `ce31662dfa7b22ff5497caa73cf33349d00cd7d3`

`analyze_fill_logs --date-from 2026-03-25 --date-to 2026-03-25 --git-sha ce31662dfa7b22ff5497caa73cf33349d00cd7d3`

- `138 orders / 59 fills / avg_pnl30=-0.27bps`
- `buy`: `35 fills / +0.40bps / AS 20.0%`
- `sell`: `24 fills / -1.24bps / AS 33.3%`

この same-SHA の見方ではじめて、**「今の残課題は sell 側」**という 626# の問題意識が強く支持されます。

ただし、その中身は 626# より少し違います。

- `sell/ranging`: `Net=-1.86bps`
- `sell/trending_up`: `Net=+3.70bps` だが `n=1`
- `sell tail worst 5` は全て sell
- `decision_path=ev_offset` が sell tail を独占

つまり現行本丸は、

**sell 全般**

ではなく

**sell × ranging × ev_offset**

です。

---

## 4. 626# のどこが強く、どこが言い過ぎか

## 4.1 強い点

`626#` の良いところは次の3点です。

1. 表層的な時間帯論ではなく、**レイヤー不発火**へ論点を下ろしたこと  
2. 売り損失を **AS 起点**で読もうとしたこと  
3. `velocity`、`regime`、`skip_gate` を個別ではなく連鎖として見たこと  

この方向性は正しいです。

## 4.2 言い過ぎな点

一方で、次は補正した方がよいです。

### A. sell 単独犯論

broad window の公式集計では buy も負けています。  
特に `buy/ranging` は `Net=-0.68bps`、tail loss も `-19.51bps`, `-14.96bps`, `-14.35bps` と十分に重いです。

したがって、627時点の整理は

- broad history: **buy/sell 両方に問題**
- current same-SHA: **sell の残課題がより前面**

です。

### B. velocity 主犯論

`626#` は velocity threshold `6.0 -> 2.5-3.0` を強く推していますが、これは **候補ではあっても主犯断定は強い**です。

理由は2つあります。

1. `tail_loss_analysis --date-from 2026-03-25 --date-to 2026-03-25` の sell tail 3件では  
   `mid_price_trend_5s tail_mean=-0.0117 vs total_mean=+0.9986`  
   で、**現行主力SHAの最悪売りは必ずしも正の velocity ではありません**。
2. raw fill を見ると、worst sell には  
   - `vel=+3.59`  
   - `vel=+3.11`  
   もありますが、同時に  
   - `vel=-0.57`  
   - `vel=-0.51`  
   - `vel=-1.12`  
   もあります。  
   つまり **売り損は「上昇中にだけ出ている」わけではない**です。

このため、velocity は

- broad history では有力特徴量
- current same-SHA では単独防御として不十分

くらいの扱いが妥当です。

### C. 時間軸の混在

`626#` は `PnL_120s` で深掘りしていますが、運用判断で最も頻繁に見ているのは `post_fill_30s_pnl` です。  
この2つを混ぜると、

- `skip_gate` は 30s でそこそこ合理的
- 120s でだけ悪化

のようなケースを「誤判定」と過剰に読んでしまいます。

レビューとしては、

- `30s = 実運用の即時品質`
- `120s = 残留 toxicity / drift`

で分けて議論する方が安全です。

---

## 5. 真の残損構造

今回の確認で、残損は大きく4層に分かれます。

## 5.1 sell のテールは `ev_offset` に集中

`2026-03-25` 主力 SHA では、sell fill `24件` が **全件 `decision_path=ev_offset`** でした。  
さらに AS が乗った `9件` の平均は `-6.96bps`、Non-AS `20件` は `+2.36bps` です。

つまり sell 側は

- side そのものが悪いのではなく
- **ev_offset で取りに行った約定が AS 化すると大きく死ぬ**

構造です。

これは 626# の「AS さえ防げれば sell は利益サイド」という指摘をかなり支持します。

## 5.2 しかし buy も `primary_only` で負ける

同日の buy fill `42件` は **全件 `decision_path=primary_only`** でした。  
AS buy `8件` の平均は `-3.67bps`、Non-AS `34件` は `+1.77bps` です。

つまり buy は

- sell ほど壊れてはいない
- しかし primary model 単独での ranging 参加にはまだ甘さがある

という状態です。

ここを無視して sell だけを直すと、次の相場で再び buy 側が表に出ます。

## 5.3 clamp 飽和は「見た目」ではなく構造問題

broad window では

- buy: `116/117` fills clamped (`99%`)
- sell: `99/99` fills clamped (`100%`)

でした。

same-SHA の `2026-03-25` でも

- buy: `6/6` clamped
- sell: `9/9` clamped

です。

ここで大事なのは、clamp 飽和は単なる可観測性問題ではなく、**上流 multiplier 群が最終的に ceiling に張り付き、状態差を価格に十分反映できていない**ことを意味する点です。

金融工学的に言えば、今は risk pricing が連続的に効いておらず、

- 危険でない状態でも ceiling
- 危険な状態でも ceiling

になりやすい。  
これでは offset を通じた risk discrimination が弱くなります。

## 5.4 sell tail は「低 spread + 負側 imbalance」で悪化

`tail_loss_analysis` の `2026-03-25` sell tail では

- `spread_at_order tail_mean=1877.7` vs total `2643.9`
- `orderbook_imbalance tail_mean=-0.2721` vs total `-0.0545`

でした。

ここから読めるのは、現行 same-SHA の sell tail は

- 広すぎる spread での passive miss ではなく
- **むしろ spread compensation が薄い状態**
- かつ **不利な板偏り**

で刺さっていることです。

したがって 625# の `min_spread` 動的化は方向として合っています。  
ただし、**「もっと広げればよい」でもない**点には注意が必要です。buy 側では wide spread でも tail が出ているためです。

---

## 6. 見落とされていたデバッグ所見

今回一番大きい技術的発見はこれです。

## 6.1 stale sidecar が 2 回目以降 `error` に化ける

`scripts/v460/lib/sidecar_signal_io.py:181` では、TTL 超過時に

- cache に `(mtime, None)` を保存
- 返り値は `"stale"`

になっています。

ところが次回同じ mtime を読むと、`scripts/v460/lib/sidecar_signal_io.py:146` で `cached_signal is None` を **`"error"`** として返します。

実際に簡易再現すると、

```text
(None, 'stale')
(None, 'error')
```

になります。

これは実運用ログの

- 初回だけ `stale`
- 以後ほぼ全部 `error`

という 3/25 の挙動と一致します。

したがって、`621#`-`626#` で sidecar が「error なので inference/配線が壊れている」と読んでいた部分の一部は、**実は stale signal が error に誤分類されているだけ**です。

これは収益論点に加えて、

- attribution 汚染
- sidecar revive 成否の誤診
- 事後レビューの誤誘導

を起こします。かなり重要です。

## 6.2 既存テストがこのケースを落としている

`tests/unit/v460/test_sidecar_sac_integration.py:783` には stale 単発テストがありますが、  
**同じ stale ファイルを 2 回読むケース**がありません。

実際、同テストファイルは `77 passed` でしたが、この stale→error 変質は未捕捉でした。  
ここは回帰テストの穴です。

## 6.3 sidecar revive は「未証明」ではなく「現状ほぼ死んでいる」

`cache/sidecar_signal.json` の実ファイルは

- timestamp: `2026-03-24T11:42:39.669519+00:00`
- mtime: `2026-03-25 08:12` 頃

で止まっています。

また `results/v460/fill_test/logs/fill_test.log` には

- `[487# sidecar] fresh=0, stale=1, missing=90`
- `[551# sidecar_nonzero] 0/91`

が記録されています。

つまり、621# の revive 施策自体は実装されていても、**運用上の signal freshness は現時点で成立していません**。

---

## 7. 金融工学・市場理論から見た補足

## 7.1 sell は「上昇相場に逆らっている」だけではない

626# は上昇バイアスを強く見ていますが、same-SHA の worst sell には negative / flat velocity もあります。  
したがって今の sell loss は単純な momentum miss ではなく、

- 板の偏り
- spread compensation 不足
- ev_offset path の参加条件

の複合です。

maker の視点では、これは

**“flow toxicity を十分に価格転嫁できていないまま、約定だけ取っている”**

状態です。

## 7.2 buy は「儲かる方向に近づいた」が、まだ alpha ではない

3/25 主力 SHA の buy は `+0.40bps` で、ここは素直に前進です。  
ただし broad window では `-0.19bps` なので、まだ安定優位ではありません。

したがって buy は

- 完成した

ではなく

- current SHA では改善した
- 別 regime / 別日ではまだ崩れる

と表現するのが適切です。

## 7.3 625# の dynamic floor は理論上正しいが、万能ではない

`min_spread = max(abs floor, bps floor, ATR floor)` は、microstructure 的には自然です。  
ただし broad window では

- `preflight_insufficient`
- `skip_gate`
- `timeout`
- `spread_too_narrow`

が既に多いので、floor だけをさらに厳格化すると starvation へ寄る危険があります。

つまり次の一手は

- blanket に spread を広げる

ではなく

- **sell × ranging × adverse book state にだけ条件付きで floor / ceiling / veto を強める**

方がよいです。

---

## 8. 優先アクション

## P0

1. `scripts/v460/lib/sidecar_signal_io.py` の stale cache semantics を直す  
   `stale` を `error` に化けさせないこと。  
   これは収益改善そのものではないが、診断を歪めるので先に止血すべきです。

2. `tests/unit/v460/test_sidecar_sac_integration.py` に  
   「同じ stale signal を 2 回読むと 2 回とも stale」  
   の回帰テストを足す

3. `2026-03-25` 主力 SHA の sell tail を主対象に据える  
   broad mixed-SHA ではなく `--git-sha ce31662dfa7b22ff5497caa73cf33349d00cd7d3` を基準にする

## P1

1. `sell × ranging × ev_offset` 条件での参加条件を詰める  
   velocity 単独ではなく
   - spread
   - imbalance
   - SG margin
   - effective_offset_used
   の joint condition で見るべきです。

2. `626#` の velocity threshold 案は ladder で試す  
   `6.0 -> 4.0 -> 3.0` の順で検証し、`2.5` へ一気に飛ばない方が安全です。

3. `buy × primary_only × ranging` の tail も捨てない  
   3/25 では buy が改善しても、broad window では依然マイナスです。

## P2

1. `analyze_fill_logs.py` と `tail_loss_analysis.py` の既存出力だけで十分回せるよう、same-SHA / same-run の比較運用を徹底する  
2. cross-venue は sell で主犯ではないが、`cap_hit` 多発なので no-op 化の監視は継続  
3. `preflight_insufficient` を単なる雑音ではなく opportunity loss として追う  

---

## 9. 最終評価

今回の流れは、悲観するよりむしろ整理が進んだと見ています。

- `620#`-`625#` で correctness はかなり改善
- `626#` で sell 側の痛点に光は当たった

一方で、まだ次の補正が要ります。

- sell 単独犯論は強すぎる
- velocity 主犯論も強すぎる
- sidecar revive は未達
- stale が error に見える実装バグで診断が濁っている

したがって 627 時点の最終判定は、

**「大損バグは止血済み。ただし残損の本質は sell/ranging/ev_offset テールを中心とする AS 問題であり、buy 側の基礎負けと sidecar 診断不良もまだ残っている」**

です。

---

## 10. 付記: 今回確認したもの

- `scripts.v460.analysis.analyze_fill_logs --date-from 2026-03-22 --date-to 2026-03-25`
- `scripts.v460.analysis.analyze_fill_logs --date-from 2026-03-22 --date-to 2026-03-25 --side sell`
- `scripts.v460.analysis.analyze_fill_logs --date-from 2026-03-22 --date-to 2026-03-25 --side buy`
- `scripts.v460.analysis.analyze_fill_logs --date-from 2026-03-25 --date-to 2026-03-25 --git-sha ce31662dfa7b22ff5497caa73cf33349d00cd7d3`
- `scripts.v460.analysis.tail_loss_analysis --date-from 2026-03-25 --date-to 2026-03-25`
- `pytest tests/unit/v460/test_sidecar_sac_integration.py --no-cov`
  - `77 passed`
- `pytest tests/unit/features/test_norm_loader.py tests/unit/v460/test_skip_gate_v3.py tests/unit/v460/test_sidecar_sac_integration.py tests/unit/v460/test_211_spread_anomaly_detector.py tests/unit/v460/test_211_micro_circuit_breaker.py --no-cov`
  - `146 passed`
