# 651# 639#-650# 利益保全型レビュー: 勝ち筋を壊さず負け筋だけを削る

## 0. 結論

率直に言うと、`641#` 以降でシステムはかなりまともになっています。
少なくとも今回の確認範囲では、もう「全面的に負けるシステム」ではありません。

既存分析で確認できた current に近い状態は次です。

- `5832c87fee` (`2026-03-29 09:05` - `2026-03-30 00:20`)
  - `avg_pnl30=+0.428bps`
  - `sum_pnl30=+13.3bps`
  - Roundtrip: `15 RT / Avg +0.53bps / PF 1.20`
- `ranging` fills は両 side ともプラス
  - `buy/ranging = +0.62bps`
  - `sell/ranging = +0.73bps`
- 一方で cancel はまだ重い
  - `preflight_insufficient=157`
  - `no_feasible_quote=68`
  - `spread_too_narrow=54`

したがって、今の主眼は

1. 既に出ている **`ranging` の薄い正の期待値** を潰さない
2. **`sell-entry` の大損 RT** と **在庫制約由来の機会損失** だけを切る
3. 大改造より **inventory / toxic sell / hold time** の3点に絞る

です。

今回の私の総判定は次です。

- `641#` `645#` `646#` `648#` は概ね正しい
- `647#` の「σ stale / ATR 固定が fill stop の主因」は支持
- `650#` の roundtrip 視点は非常に良い。今後は `pnl30` 単独より **RT PnL を主KPI** にすべき
- ただし今やるべきは、さらに guard を積むことではなく、**勝ち筋を守る局所条件付きの負け筋遮断**

です。

---

## 1. 何が改善したか

### 1.1 `641#` - `648#` の止血は効いている

今回確認した current config / code は以下です。

- `configs/v460/fill_test.yaml:348`
  - `cross_venue_lead_lag.offset_boost: 1.0`
- `configs/v460/fill_test.yaml:457`
  - `skip_gate.max_skip_rate: 0.4`
- `configs/v460/fill_test.yaml:993`
  - `balance_freeze_cycles: 1`
- `configs/v460/fill_test.yaml:729`
  - `execution_final_clamp_hard_skip_mult_overrides.buy/trending_down: 4.0`
- `configs/v460/fill_test.yaml:424`
  - `model_path_sell: null`
- `configs/v460/fill_test.yaml:1017`
  - `side_min_samples: 200`

3日レポートでも改善の流れは見えます。

| SHA | fills | avg_pnl30 | 所見 |
|---|---:|---:|---|
| `8f728d9232` | 30 | `-0.845bps` | まだ悪い |
| `5832c87fee` | 31 | `+0.428bps` | プラス復帰 |

この変化を見る限り、`645#` / `646#` の

- 退化 sell モデル停止
- 再学習過学習ガード

は方向として支持できます。

### 1.2 `648#` の σ stale fix も支持

`sha_performance_report --days 3` では、`f7faac4f12` が

- fills `0`
- `no_feasible_quote=87`
- `spread_reject gap=+1284JPY`
- `σ AT CAP`

で完全停止していました。

その後の SHA で fills 自体は回復しているため、`648#` の修正は有効と見てよいです。

---

## 2. いま守るべき勝ち筋

### 2.1 `ranging` はもう「捨てる相場」ではない

`2026-03-29` の `5832c87fee` では

- `ranging avg_pnl30 = +0.67bps`
- `buy/ranging = +0.62bps`
- `sell/ranging = +0.73bps`
- Roundtrip `ranging = +11.18bps`

でした。

これは重要です。
かつての 635-638 期では `sell/ranging` が病巣でしたが、今はもう

- `ranging` を雑に殺す
- `sell/ranging` を一律に殺す

のは危険です。

現在は **`ranging` 全体が薄く勝てる状態へ戻りつつある** ため、
負け筋の条件付けが必要です。

### 2.2 Roundtrip では `buy-entry` を守るべき

`650#` と実出力の両方で、`2026-03-29` の RT は

- `buy-entry: +8.99bps`
- `sell-entry: -1.06bps`

でした。

つまり今の収益構造は

- buy-entry RT が収益源
- sell-entry RT が尾を引っ張る

です。

このため、次の改修は

- buy を増やす
ではなく
- **sell-entry の中でも悪いものだけを減らす**

べきです。

### 2.3 sweet spot は `2.7-3.1bps`

`Spread vs Fill Quality` はかなり価値がありました。

`2026-03-29` では

- Q1 `<2.3bps`: `-0.35bps`
- Q2 `2.3-2.7bps`: `+0.02bps`
- Q3 `2.7-3.1bps`: `+2.66bps`
- Q4 `>=3.1bps`: `-0.67bps`

でした。

これは示唆が明快です。

- 狭すぎる板は toxic
- 広すぎる板は stale / overdefensive
- **中上位帯の spread が最もおいしい**

したがって profit-first の観点では、
次に守るべきなのは「RT」だけでなく **spread sweet spot** です。

---

## 3. いま削るべき負け筋

### 3.1 主犯は `sell-entry` の一部 tail

`650#` で最悪 RT は次でした。

- `sell-entry -14.27bps` (`trending_up -> ranging`, hold `69.7m`)
- `sell-entry -9.14bps` (`ranging -> ranging`, spread `1.6/1.1bps`)

ここから見るべきことは、sell 全体を悪者にしないことです。
問題は sell 全般ではなく、主に次です。

1. `trending_up` 起点の sell-entry
2. 低 spread の toxic sell-entry
3. 長時間 hold に陥る sell-entry

つまり **sell-entry tail の条件付き遮断** が本丸です。

### 3.2 在庫制約は依然として最重級の機会損失

`2026-03-29` では

- `preflight_insufficient = 157/363`
- `buy-side blocked = 134`
- JPY ratio = `0.9%`
- BTC ratio = `99.1%`

でした。

しかも `balance_freeze_cycles=1` に短縮した後でも、まだこの偏りです。

ここから言えるのは、

- `641#` の freeze=1 は正しかった
- しかし **それだけでは在庫の詰まりを解消できない**

ということです。

現在の profit bottleneck は、guard 過多よりむしろ **JPY 再調達能力の弱さ** にあります。

### 3.3 `no_feasible_quote` / `spread_too_narrow` は「悪い guard」ではなく「雑な試行」の痕跡

`2026-03-29` はプラスでしたが、同時に

- `no_feasible_quote = 68`
- `spread_too_narrow = 54`

も多いです。

ここでやってはいけないのは、単純に spread guard を緩めることです。
Q1 と Q4 が負けているため、guard を広く緩めると勝ち筋まで壊します。

むしろ必要なのは

- 低 spread toxic sell を減らす
- 在庫制約で buy を出せない時に同じ無理筋 sell を繰り返さない

ことです。

---

## 4. 文書別レビュー

| 文書 | 判定 | コメント |
|---|---|---|
| `639#` | 部分支持 | `EV hard floor` や `inventory skew` の方向は良いが、既存実装の把握不足がある |
| `640#` | 支持 | 638/639 を code/log に戻しており、今回範囲の交通整理として優秀 |
| `641#` | 支持 | P0-P1 の打ち手は妥当。とくに `max_skip_rate 0.4` と `freeze=1` は支持 |
| `642#` | 強く支持 | forced pass / balance snapshot / CV action 可視化は今後の必需品 |
| `645#` | 強く支持 | degenerate sell model 無効化は止血として正しい |
| `646#` | 強く支持 | 過学習ガードは妥当。side model 再導入のハードルとして機能する |
| `647#` | 支持 | sigma stale / ATR floor 固定の分析は整合的 |
| `648#` | 強く支持 | 648 の修正がなければ今のプラス復帰は難しかった |
| `650#` | 強く支持だが一部補正 | roundtrip 分析は価値が高い。一方で MCB 対策は今すぐの P0 ではない |

---

## 5. 三者が見落としていた点

### 5.1 `skip_rate_limit` はもう主戦場ではない

以前は forced pass が大きな問題でしたが、`2026-03-29` では

- skip total `22/363 = 6.1%`

まで下がっています。

したがって、今の主戦場は

- `skip_rate_limit` 追加緩和
ではなく
- inventory / toxic sell / hold time

です。

`max_skip_rate` をさらに緩めるより先に、別のレバーを使う方が利益に近いです。

### 5.2 sidecar stale は P0 ではない

`2026-03-29` は

- fresh `3`
- stale `28`
- error `0`

で 90% stale でしたが、`avg_pnl30` と RT はプラスでした。

つまり sidecar の健全化は大事でも、今は **収益改善の第一レバーではありません**。
ここに工数を寄せすぎると、いま効いている execution/safety 改善の手が鈍ります。

### 5.3 `sell/ranging` 一律防御はもう古い

`2026-03-29` の `sell/ranging = +0.73bps` を見ると、過去の反省だけで sell/ranging を機械的に削るのはもう危険です。

いま必要なのは

- `sell/ranging` を止めること
ではなく
- **`toxic sell-entry` だけ止めること**

です。

---

## 6. 利益を壊さずに負け筋だけを切る提案

### P0-1. inventory skew を「効く水準」まで弱く再調整する

`650#` の最大の有効提案はこれです。
現状、`inventory_skewing.enabled=true` なのに、実質的に効いていません。

ただし `neutral_band: 0.1 -> 0.03` は少し急です。
今の `ranging` 収益を壊さないため、初手は次が安全です。

- `neutral_band: 0.10 -> 0.05`
- `decay_tau_sec: 1800 -> 3600`

これで

- buy 不足時にだけ skew を効かせる
- 既存の `ranging` 収益を急に壊しにくい

状態を狙えます。

### P0-2. `sell-entry` の低 spread toxic 条件だけを veto する

一律の low-spread guard は危険です。
`650#` もここは慎重で、私も同意します。

本命は次のような **三点条件** です。

- `side = sell`
- `spread_bps < 2.3`
- `ob_imbalance > 0.25` and `vpin > 0.65`

理由:

- Q1 は負けている
- RT#7 のような toxic sell は捕まえたい
- しかし RT#5 のような低 spread 勝ち trade までは潰したくない

つまり「狭い spread を全部切る」のではなく、**狭い上に toxic な sell だけ切る** のが正解です。

### P0-3. `hold time` の長い sell-entry だけ先に逃がす

最悪 RT は hold `69.7m` でした。
これは単なる entry 品質だけでなく、在庫制約で exit が遅れた結果でもあります。

したがって、`micro-timeout` を全面有効化する前に、まずは

- `sell-entry` かつ
- hold が一定時間を超えた open inventory

に対して、より積極的な close 優先モードを入れるのがよいです。

市場理論的にも、maker の損失 tail はエントリーより **在庫を抱えたまま regime が変わること** で肥大化します。

### P1-1. `buy/trending_down` の保護は維持、これ以上いじりすぎない

`641#` の hard skip override は支持です。
ただし `2026-03-29` では `trending_down` 全体が `+0.03bps` と薄く、圧倒的勝ち筋というほどではありません。

よって現時点では

- これ以上の緩和を足す
より
- 今の override を維持して観察

が妥当です。

### P1-2. MCB の open-position 対応は P2 扱いでよい

`650#` は MCB pre-close を提案していますが、`2026-03-29` の分析では

- During/Post-MCB fills: `+0.55bps`
- Outside MCB window: `-0.04bps`

でした。

つまり、MCB は少なくとも current snapshot では一律悪ではありません。

大損 RT の背景要因ではあるものの、いま最初に工数を投じるべき対象ではない、というのが私の判定です。

---

## 7. 実行順序

| 優先度 | 提案 | 理由 |
|---|---|---|
| P0 | `inventory_skewing` の軽調整 (`neutral_band 0.05`, `tau 3600`) | buy 機会損失を減らしつつ、今の ranging 収益を壊しにくい |
| P0 | toxic low-spread sell 条件 veto | `sell-entry` tail を狙い撃ちできる |
| P0 | 長時間 hold の sell-entry 逃がし | RT tail を直接削れる |
| P1 | `buy/trending_down` override 維持観察 | 勝ち筋を壊さない |
| P1 | roundtrip を主KPI化 | `pnl30` 単独より実収益に近い |
| P2 | MCB open-position 特例 | 有効だが current 最優先ではない |
| P2 | sidecar stale 改善 | 必要だが現状の profit driver ではない |

---

## 8. 最終判定

今回の 639#-650# で得るべき一番大きい教訓はこれです。

- もう「全部悪い」フェーズではない
- いま勝っているのは `ranging` と `buy-entry RT`
- いま切るべきなのは `sell-entry tail` と `inventory starvation`

です。

したがって、次の改善は

- 防御を全体へさらに厚く積む
ではなく
- **在庫を少し動かしやすくする**
- **toxic な sell-entry だけ落とす**
- **長時間 hold を切る**

の3本に寄せるのが最も profit-first です。

---

## 9. 確認に使ったもの

- `docs/v460/639_cplt_deep_review_beyond_638_and_strategic_proposals.md`
- `docs/v460/640_cplt_synthesis_638_639_verification_and_action_plan.md`
- `docs/v460/641_cplt_p0_p1_implementation_cv_widen_hardskip_regime.md`
- `docs/v460/642_cplt_observability_fill_record_6_fields.md`
- `docs/v460/645_cplt_p0_degenerate_sell_model_fix.md`
- `docs/v460/646_cplt_overfitting_guards.md`
- `docs/v460/647_cplt_post_deploy_analysis_sigma_fill_rate.md`
- `docs/v460/648_cplt_sigma_stale_feedback_loop_fix.md`
- `docs/v460/650_cplt_roundtrip_analysis_sections.md`
- `configs/v460/fill_test.yaml`
- `scripts/v460/lib/fill_config.py`
- `scripts/v460/lib/fill_config_parser.py`
- `scripts/v460/lib/skip_gate_model_loader.py`
- `scripts/v460/lib/maker_price.py`
- `scripts/v460/lib/maker_microstructure.py`
- `scripts/v460/lib/multiplicative_pipeline.py`
- `python -m scripts.v460.analysis.sha_performance_report --days 3`
- `python -m scripts.v460.analysis.analyze_fill_logs --date-from 2026-03-29 --date-to 2026-03-29`
- `pytest tests/unit/v460/test_645_degeneracy_check.py tests/unit/v460/test_646_overfitting_guards.py tests/unit/v460/test_648_inventory_deadlock.py tests/unit/v460/test_634_sell_ranging_suppression.py tests/unit/v460/test_336_yaml_code_drift_prevention.py --no-cov`

テスト結果:

- `55 passed`
