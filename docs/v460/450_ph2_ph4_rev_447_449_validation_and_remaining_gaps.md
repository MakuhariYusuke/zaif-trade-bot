# 450# 447#-449# 検証レビュー: Cross-Venue 修正の妥当性と未解消ギャップ

**種別**: rev  
**対象**: 447# / 448# / 449#  
**日付**: 2026-03-16

---

## §0 結論

447# の問題提起は概ね正しく、448# / 449# のコード修正方針も妥当です。  
ただし、**「修正がコードに存在する」ことと「修正が live fill test に反映され、観測できている」ことはまだ一致していません。**

今回の結論は次の 4 点です。

1. **448# / 449# の repo 上の修正自体は実在する**  
   `scripts/v460/lib/cross_venue_lead_lag.py:330`、`scripts/v460/lib/maker_risk_guards.py:245`、`scripts/v460/lib/fill_record_builder.py:201`

2. **しかし 2026-03-16 の live ログ / fill_records には旧挙動の痕跡が残っている**  
   `results/v460/fill_test/logs/fill_test.log` では  
   `spread=-5.23bps ... ema_spread=-3.89bps`  
   のように、448# 前の「EMA と point spread の混線」が見える。

3. **仮に新コードが live で動いても、FillRecord スキーマが未更新のため新観測値が JSONL に落ちない**  
   `ztb/metrics/fill_quality.py:174` 以降の `FillRecord` には  
   `cross_venue_lead_lag_point_spread_bps`  
   `cross_venue_lead_lag_pre_offset`  
   `cross_venue_lead_lag_post_offset`  
   `cross_venue_lead_lag_cap_hit`  
   が存在せず、`ztb/metrics/fill_quality.py:235` の sanitize で捨てられる。

4. **447# が強調した mixed-SHA 問題は未解決**  
   `scripts/v460/analysis/ab_offset_comparison.py:318` は依然として date split のみで、`git_sha` / `run_id` 固定比較ができない。

要するに、**448# / 449# は「設計上の正しい前進」だが、収益寄与を判定する段階にはまだ入っていない**、という整理が妥当です。

---

## §1 447# の指摘に対する検証

### §1.1 F1: mixed-SHA A/B 汚染

これは引き続き **正しい指摘** です。

`scripts/v460/analysis/ab_offset_comparison.py --compare --split-date 2026-03-16` の出力は:

- Before: `11713 records`
- After: `183 records`
- Overall Δ: `+1.397bps`

ですが、`fill_records_20260316.jsonl` には少なくとも 5 SHA が混在しています。

- `f34467b5c2f8`: 79 rows, cv_rows=0
- `a9714ad9af85`: 65 rows, cv_rows=6
- `e23a063923ee`: 23 rows, cv_rows=3
- `1d64e64db506`: 10 rows
- `c38c15ec943c`: 6 rows

最大母集団の `f34467...` は cross-venue 観測を持たないため、`3/16 改善 = 448#/449# 効果` とはまだ言えません。

### §1.2 F2/F3: EMA spread 混線と no-op 可視化

repo 上のコード修正は確認できました。

- `scripts/v460/lib/cross_venue_lead_lag.py:330`
  - EMA モードで `spread_bps=ema_spread_bps`
  - `point_spread_bps` を別保持
- `scripts/v460/lib/maker_risk_guards.py:295`
  - `pt_spread` / `CAP_HIT=NO-OP` ログ対応
- `scripts/v460/lib/fill_record_builder.py:207`
  - pre/post/cap_hit を追加

ここまでは良いです。

ただし、**実ログはまだ旧挙動** です。  
`results/v460/fill_test/logs/fill_test.log` には:

```text
[cross_venue] hint direction=down adverse_side=buy spread=-5.23bps ... ema_spread=-3.89bps
```

が残っており、これは 448# 後の挙動と一致しません。  
448# 後なら `spread` は `ema_spread` と同系統の値になるはずです。

さらに `fill_records_20260316.jsonl` 側でも、cross-venue 行に

- `cross_venue_lead_lag_point_spread_bps = None`
- `cross_venue_lead_lag_cap_hit` キー自体が未出力

となっており、448# で言う「可視化完了」は live 記録上まだ確認できません。

### §1.3 F4: buy+ranging 過剰 skip

依然として重いです。

`fill_records_20260316.jsonl` では top cancel reason が:

- `ranging_low_vol_skip = 81`
- `route_to_kill_deadlock = 24`
- `sell_dynamic_kill = 15`
- `skip_gate = 10`

です。

cross-venue より前に participation が削られており、447# の「Micro-Timeout で stale exposure を切る」議論は有望ですが、現状はまず **過剰 suppression と kill 系支配** を分離しないと解釈がぶれます。

### §1.4 F5/F6: heuristic toxicity / deadlock / dynamic_kill

447# はここも正しいです。

`fill_test.log` の restored counters では:

- `toxic_veto_set = 367`
- `toxic_veto_block = 53`
- `gate_sell_dynamic_kill = 896`
- `gate_buy_dynamic_kill = 489`
- `route_to_kill_deadlock = 60`
- `gate_ranging_low_vol_skip = 255`

が残っています。

したがって、現段階で「新 edge が弱い」のか「既存 blocking が強すぎて edge 評価不能」なのかを分けると、後者の比重がかなり大きいです。

---

## §2 448# / 449# 実装に対するレビュー

## §2.1 良い点

- 448# の **EMA / point spread 分離** は論理的に正しい
- 448# の **no-op 可視化** は観測設計として正しい
- 449# の **spread 計算 DRY** は保守性改善として妥当
- 449# の **config 化** は backward compatible で安全

対象コード:

- `scripts/v460/lib/cross_venue_lead_lag.py:195`
- `scripts/v460/lib/fill_cycle_executor.py:197`
- `scripts/v460/lib/fill_config.py:363`
- `scripts/v460/lib/fill_config_parser.py:192`

## §2.2 ただし、現時点では「効いた」とは判定しない

理由は 3 つです。

### 1. live runtime 反映が確認できていない

3/16 ログの `spread != ema_spread` は、少なくともその時点の runtime が 448# ロジックどおりではなかったことを示します。

### 2. FillRecord 側のスキーマ穴がある

`ztb/metrics/fill_quality.py:174` 以降の `FillRecord` に新 field が無いため、  
`fill_record_builder.py` が field を足しても `build_fill_record()` で落ちます。

該当箇所:

- `ztb/metrics/fill_quality.py:174`
- `ztb/metrics/fill_quality.py:235`
- `ztb/metrics/fill_quality.py:281`

これは **HIGH** です。  
「ログには出るが JSONL には残らない」では、事後分析も A/B 判定もできません。

### 3. 448# / 449# 向けテストは unit 境界で止まっている

`tests/unit/v460/test_439_cross_venue_lead_lag.py` は、

- hint 計算
- event/fill field 生成
- 既存 cross-venue field の round-trip

は見ていますが、**新しく追加した point_spread / pre_offset / post_offset / cap_hit が FillRecord まで到達するか** は見ていません。

そのため今回のように

- repo コード上は修正済
- live JSONL では消失

という穴を unit test が捕まえられていません。

---

## §3 447# 新パラダイム提案の評価

## §3.1 提案B: Micro-Timeout

これは 3 案の中で **最も実装優先度が高い** です。

理由:

- `buy + ranging` の本質的な弱点は stale passive exposure
- offset ceiling に当たりにくい
- cross-venue より小さい改修で試せる
- 既存 `timeout` 機構を短くする方向なので導入面積が小さい

ただし、`cycle_interval_sec=120` の系でやるなら、

- cancel 後の待機時間設計
- cancel race
- status_unknown_fast

との整合を先に見るべきです。

## §3.2 提案A: Asymmetric Inventory Sponging

発想は面白いですが、**今は早い** です。

現状はすでに

- `route_to_kill_deadlock`
- `buy_dynamic_kill`
- `sell_dynamic_kill`
- one-sided 系制約

が強く、ここに inventory をさらに非対称化すると、alpha ではなく liveness 問題を増幅するリスクが高いです。

## §3.3 提案C: Global Spread Shadowing

方向性は悪くありませんが、**今の cross-venue path の観測品質が整ってから** が順番です。

いま reference venue シグナル自体の

- runtime 反映
- JSONL 観測
- same-SHA 効果測定

が未整備なので、その上にさらに global spread shadowing を足すと、解釈不能な層が増えます。

---

## §4 見落としとして強く補うべき点

### §4.1 `git_sha` は「そのコードが live に載っていた」証拠になっていない

3/16 の `a9714ad9af85` 行には confidence 系 field がある一方で、`spread=-5.23 / ema=-3.89` の旧意味論が残っています。  
したがって、現行の `git_sha` は **少なくとも runtime code semantics の完全証明には使えません。**

原因候補は:

- プロセス未再起動
- hot-reload による attribution ずれ
- 同日 mixed runtime

のいずれかです。  
この点は 448#/449# の評価以前の **運用観測問題** です。

### §4.2 `cap_hit` は実装済みでも live ではまだ見えていない

ログ上は

```text
offset 0.3000->0.3000
```

の no-op があるのに、`CAP_HIT=NO-OP` は出ていません。  
これも「コードがある」ことと「そのコードで走った」ことが別である証拠です。

### §4.3 449# の理論補強は悪くないが、収益レバーとしては後順位

`confidence_floor` や `depth_imbalance_threshold` の config 化は保守上よいです。  
ただ、現在の支配項は

- mixed-SHA
- live 未反映
- FillRecord スキーマ穴
- kill / deadlock / over-skip

なので、profit-first の優先順位では P0 ではありません。

---

## §5 優先順位

### P0

1. `ztb/metrics/fill_quality.py` の `FillRecord` に 448# 新 field を追加  
2. `test_439_cross_venue_lead_lag.py` か `test_fill_quality.py` に、builder→FillRecord→JSONL の統合テストを追加  
3. fill test を **プロセス再起動前提** で 1 SHA に固定し、448#/449# 反映後の clean run を取り直す  
4. `ab_offset_comparison.py` に `git_sha` / `run_id` filter を追加

### P1

1. 448#/449# の same-SHA window で
   - `cv_rows`
   - `applied`
   - `veto`
   - `cap_hit`
   - `avg pnl conditional on applied/veto`
   を再測定
2. `ranging_low_vol_skip` と `buy_dynamic_kill` の同時多発を分離
3. 既存 heuristic toxicity の再校正を再開

### P2

1. 447# 提案の中では `Micro-Timeout` を先行試験
2. `Inventory Sponging` は liveness 安定化後
3. `Spread Shadowing` は cross-venue path の clean attribution 完了後

---

## §6 総評

447# は、方向修正の起点として十分に価値があります。  
448# / 449# も、コードとしては良い修正です。

ただし現状は、

- **修正の live 反映が未確認**
- **反映されても FillRecord に残らない**
- **A/B が still mixed-SHA**

という 3 重の穴があり、ここを飛ばして次の高度な edge へ行くのは早いです。

したがって次の判断は、

- **新機能追加より先に、448#/449# を「観測可能な状態」で再実走させる**

が正解です。  
その上で 447# の新提案を選ぶなら、最有力は **Micro-Timeout** です。
