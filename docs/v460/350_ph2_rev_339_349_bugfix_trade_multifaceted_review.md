# 350# 339–349 総合レビュー: バグ修正後トレードの技術・市場理論監査

**日付**: 2026-03-09  
**対象**: 339#–349#, `prompts/349_review_prompt.md`, 現行実装  
**観点**: 収益性最優先 + システム工学 + 市場理論 + 評価設計

---

## 総括

339#–349# の流れは、主に以下 3 系統の是正としては概ね正しい。

1. **壊れていた kill 制御の正常化**  
   340# の符号修正、341# の閾値復元、344# の EWMA 化、349# の EWMA 永続化は妥当。
2. **構造的な複雑性・デッドロック要因の切除**  
   348# の `balance_forced` 撤廃は方向として正しい。
3. **分析基盤の再整理**  
   349# の分析ツール一元化と重複削減は保守性向上に効く。

ただし、**349# で確認できたのは「liveness と観測可能性の回復」であって、「収益優位の確立」ではない**。  
349# 自身の 7 日集計でも `buy=-97.3bps`, `sell=+2.7bps`, worst segment=`buy_ranging -102.9bps` であり、本丸は依然として **buy 側の ranging 参加品質** にある。

---

## Findings

| # | 重大度 | 対象 | 指摘 | 推奨 |
|---|---|---|---|---|
| 1 | HIGH | `ztb/risk/sell_dynamic_kill.py:283` | `_rebuild_ewma_from_history()` は「再構築」と呼ぶには数学的に不正確。平均 seed 後に全履歴 replay しており、元の EWMA と一致しない。 | `ewma = history[0]` から `history[1:]` を replay する形に修正し、旧 state フォールバック時も元の `track()` と同一式にする。 |
| 2 | HIGH | `tests/unit/v460/test_349_ewma_fixes.py:43` | 旧 state フォールバックのテストが「`None` ではない」しか見ておらず、再構築値の正しさを検証していない。 | `export_state()` から `ewma_value` を意図的に落とした state で、再構築後 EWMA が元値と一致するテストを追加する。 |
| 3 | HIGH | `docs/v460/349_phg_refactor_analysis_dedup.md:196` | 349# の「修正前 vs 修正後」は before/after 比較としては読めるが、因果的に「収益改善を証明」してはいない。市場状態も窓長も違う。 | 349# の結論は「kill 無限ループ修復で fill と可観測性が回復」に留める。収益性判断は same-SHA / same-session / regime 控え目比較で行う。 |
| 4 | MEDIUM | `ztb/risk/sell_dynamic_kill.py:480` | TIME LIMIT 後の `threshold * 0.8` は方向は正しいが、固定係数のハードコードで regime/side/alpha 非依存。 | `time_limit_reset_ratio` を config 化するか、`old_ewma` と `threshold` の補間で reset 幅を制御する。 |
| 5 | MEDIUM | `docs/v460/347_ph2_rpt_min_lot_constraint_analysis.md:18`, `configs/v460/fill_test.yaml:76` | 348# の satoshi 精度化は正しいが、`min_lot=0.001` のまま・資本約 2mBTC のままでは、縮小系ロット制御の多くは依然 floor に張り付く。 | 348# の効果は「将来の粒度改善」と「コード負債解消」と位置付ける。短期収益レバーとして過大評価しない。 |
| 6 | MEDIUM | `docs/v460/348_ph2_impl_satoshi_and_balance_forced_removal.md:22` | `balance_forced` 撤廃は正しいが、代替の `inventory_escape` / `quiescence` / `no_feasible` 系が十分に観測されないと、問題が「強制トレード」から「長時間不参加」へ名前を変えるだけになりうる。 | 348# 後は `inventory_escape`, `one_sided_freeze`, `no_feasible`, `quiescence` の発火率と PnL を side 別に常設集計する。 |
| 7 | MEDIUM | `docs/v460/349_phg_refactor_analysis_dedup.md:22` | 直近の損失構造はもはや sell kill ではなく `buy_ranging`。sell は 7 日集計でほぼ中立まで戻っている。 | 次の主戦場を `buy_ranging` に固定し、VPIN・短期 drift・待ち時間・EV 矛盾を使った bid 参加選別に寄せる。 |

---

## 1. 何が正しく直ったか

以下は評価できる。

- **340# の符号修正**  
  `threshold_offset_bps > 0` が本来の意味通り「緩和」に戻った。これは必須修正だった。
- **341# の閾値復元**  
  336#/337# の過緩和が「符号バグを前提にした調整」だった以上、pre-bug 水準へ戻した判断は正しい。
- **343# の skip_gate/kill release grace**  
  kill 解除直後の過剰抑制を少し和らげる方向で合理的。
- **344# の EWMA 導入**  
  count-based rolling より市場変化への応答が速い。方向性は正しい。
- **348# の `balance_forced` 撤廃**  
  収益性・保守性・デッドロック耐性のいずれから見ても、無理筋の構造を切った判断は妥当。
- **349# の分析整理**  
  分析コードの一元化は今後のレビュー精度に効く。

結論として、339#–349# は「迷走」よりも、**壊れた制御系を評価可能な状態へ戻すための正常化フェーズ** と見るのが適切。

---

## 2. 349# EWMA 修正で最も重要な補正点

### 2.1 `_rebuild_ewma_from_history()` は正確な再構築ではない

現実装は以下。

- `ztb/risk/sell_dynamic_kill.py:293` 平均で seed
- `ztb/risk/sell_dynamic_kill.py:294` 全履歴を replay

これは `track()` の通常更新式とは一致しない。  
通常の `track()` は初回値を seed にし、その後の値だけを EWMA 更新する。

実測でも差が出る。  
`[1.0, -1.0, 0.5, -0.5, 0.0, -10.0], alpha=0.1` で確認すると:

- 通常 `track()` 後 EWMA: `-0.47917`
- 349# fallback rebuild 後 EWMA: `-1.896346`
- 差分: `-1.417176`

この差は小さくない。  
つまり、349# の fallback path は「旧 state でもだいたい復元」ではなく、**条件次第で kill 判定を有意に変える近似** になっている。

### 2.2 テストがこのズレを見逃している

`tests/unit/v460/test_349_ewma_fixes.py:43` の旧 state テストは、

- `import_state()` して
- `_ewma_value is not None`

しか見ていない。  
これでは「値が正しいか」は分からない。

### 2.3 判断

349# P0 の主旨自体は正しい。  
ただし現状の fallback 実装は **HIGH** と評価する。理由は、これは upgrade 直後や旧 state 読み込み時の挙動を左右するから。

---

## 3. 349# の改善は「利益改善」より「停止不具合修復」

349# の before/after:

- 修正前: 5 fills / `-18.54bps`
- 修正後: 41 fills / `-7.08bps`

これは確かに改善だが、ここから強く言えるのは以下まで。

1. kill 無限ループが止まった
2. fill が戻った
3. 観測可能性が戻った

一方で、349# 自身の 7 日集計では:

- Total PnL: `-94.6bps`
- Buy PnL: `-97.3bps`
- Sell PnL: `+2.7bps`
- Worst segment: `buy_ranging -102.9bps`

したがって、349# を「収益化の突破口」と読むのは早い。  
**正確には、profitability 以前の liveness bug を潰した** と読むべき。

これはシステム工学的にも重要で、**評価不能な状態から評価可能な状態に戻した** という意味で価値がある。

---

## 4. 市場理論から見た現在の本丸

### 4.1 sell は「守り過ぎ」より「ほぼ正常化」に近づいた

349# の 7 日集計では sell が `+2.7bps`。  
まだ十分ではないが、少なくとも「真っ先に sell kill をさらに緩める」局面ではない。

市場理論的にも、sell 側は通常、

- BTC 過剰在庫の解消
- 上昇局面での逆選択

の綱引きになる。  
340#–344# の修正で、ここはかなり素直な制御に戻っている。

### 4.2 本丸は `buy_ranging`

`buy_ranging -102.9bps` は、受動的な bid 提供が

- 反発前の安値拾いではなく
- 情報優位者に売りつけられている

ことを示唆する。

言い換えると、今の buy は「安く買えている」のではなく、**下げの途中で古い bid が踏まれている** 可能性が高い。

### 4.3 何を強めるべきか

次の改善軸は kill ではなく、buy entry quality。

- `VPIN` 高時の bid 参加抑制
- 短期 drift / velocity が下向きのときの buy 厳格化
- 長待ち buy の stale quote 検出
- `ev_score > 0` でも microstructure が逆行なら不参加
- ranging と見えても実際は transition 中の局面を弾く

要するに、**buy の受動参加条件を情報優位者基準で再設計する段階** に入っている。

---

## 5. 347#/348# の読み方の補正

347# の分析は正しい。  
ただし 348# の satoshi 精度化については、短期効果を冷静に見る必要がある。

現状設定:

- `configs/v460/fill_test.yaml:76` `min_lot: 0.001`
- `configs/v460/fill_test.yaml:78` `lot_step: 0.00000001`

ここで効くのは主に:

- 将来の finer-grained increase/decrease 余地
- dust / 端数処理
- コード上の二重管理除去

一方で、**縮小方向の制御** は依然として `min_lot` に張り付く場面が多い。  
したがって、

- `confidence_lot`
- `balance_shrink`
- `per_side_recovery`
- Kelly 下方調整

の大半は、資本が薄いままだとまだ本格復活しない。

つまり 348# は「大きな土台改善」ではあるが、**すぐ利益が伸びるレバーではない**。

---

## 6. 348# balance_forced 撤廃の評価

これは概ね賛成。

理由:

1. 強制トレードは PnL 汚染源だった
2. kill / halt / forced の組み合わせは複雑性が高過ぎた
3. 市場理論的にも、負の期待値局面で無理に参加するより `No Trade` の方が合理的なことがある

ただし、副作用の監視は必要。

もし 348# 後に

- `inventory_escape` 多発
- `no_feasible` 多発
- `quiescence` 長時間化
- side 偏り固定

が起きるなら、それは「強制トレード問題を別名に置き換えた」だけになる。

よって 348# 以後は、**強制 fill 件数の代わりに、不参加・逃避・片側凍結の観測を厚くする** のが次の筋。

---

## 7. 次にやるべきこと

### P0: 次の実験前に必須

1. `ztb/risk/sell_dynamic_kill.py` の fallback rebuild を厳密式に修正する  
   `ewma = history[0]` から始め、`history[1:]` のみ replay。
2. `tests/unit/v460/test_349_ewma_fixes.py` に exact rebuild test を追加する  
   旧 format state 復元後 EWMA が元値と一致することを確認。
3. 349# のドキュメント上の結論を「収益改善」ではなく「liveness bug 修復」に言い換える  
   誤った楽観を避ける。

### P1: 収益に直結

1. `buy_ranging` を主戦場にして deep dive する  
   `side × regime × wait bucket × VPIN × velocity × ev_score符号` で分解。
2. `ev_score` と実損益が逆転した buy を抽出する  
   「モデルは買いと言うが microstructure は危険」のパターンを固定化してルール化。
3. 長待ち buy と即約定 buy を分ける  
   stale quote 問題と fast adverse selection は別問題なので混ぜない。

### P2: その後

1. TIME LIMIT 後 reset の config 化  
   `reset_ratio` か `reset_to_threshold_margin_bps` を side/regime 別に。
2. 348# 後の observability 追加  
   `inventory_escape`, `no_feasible`, `quiescence`, `one_sided_freeze` を side 別に定点観測。
3. lot/sizing は positive edge が数日単位で安定してから再評価  
   今は sizing より edge の純化が先。

---

## 8. 最終判断

339#–349# は、利益を直接生む本丸施策というより、

- 壊れていた kill 制御の修復
- 無駄に複雑だった forced 系の切除
- 分析基盤の整備

としては有意義だった。

一方で、現時点の利益面の本丸は明確で、

- **sell kill 追加調整ではない**
- **lot_step 変更でもない**
- **`buy_ranging` の情報劣位参加をどう止めるか**

である。

よって次の優先順位は:

1. 349# fallback EWMA の厳密修正
2. 348# 後の不参加系 observability 強化
3. `buy_ranging` 集中分析

この順が妥当。

---

## 検証メモ

- `./.venv/Scripts/python.exe -m pytest tests/unit/v460/test_349_ewma_fixes.py --no-cov`
  - `13 passed`
- 手元再現:
  - 入力 `[1.0, -1.0, 0.5, -0.5, 0.0, -10.0]`, `alpha=0.1`
  - 通常 EWMA `-0.47917`
  - 349# fallback rebuild `-1.896346`
  - 差分 `-1.417176`
