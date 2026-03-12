# 375# ph3 レビュー: 369#-374# Profit-First 再点検と SAC スコープ是正

| 項目 | 値 |
|---|---|
| 文書番号 | 375# |
| フェーズ | ph2/ph3 境界レビュー |
| 対象 | 369#-374# |
| 作成 | Codex |
| ステータス | Review |

---

## §1 総評

結論を先に述べる。

- **369#-373# の流れは概ね正しい。** 特に「まず ph2 の実運用ボトルネックを抑える」「SAC は sidecar として限定注入する」「安全性監査を先に潰す」という順序は妥当である。
- **374# は設計メモとしては有益だが、そのまま実装計画に昇格させるのは危険**である。理由は、設計の中核が `profit-first` ではなく `SAC の表現力拡張` に引っ張られており、現状の live ボトルネックとズレ始めているためである。
- 現時点の推奨は次の通り。
  - **Phase 3.1 は縮小版でのみ GO**
  - **Phase 3.2 は HOLD**
  - **Phase 3.3 / 3.4 は NO-GO**

要するに、今やるべきは「SAC を賢くすること」ではなく、**SAC が live で本当に 1bps でも改善するのかを極小変更で証明すること**である。

---

## §2 主要指摘

### 1. CRITICAL: 374# §3.1 の `max_boost_bps=3.0` は「微調整」ではなく価格形成の主役になり得る

374# は Proportional Boost を最小変更としているが、数値感が危険である。

実測確認:

- `results/v460/fill_test/fill_records_20260310.jsonl` の retained records では `mid_at_order` 中央値は **10,916,213.5 JPY**
- 同ファイルの `spread_at_order` 中央値は **2,282 JPY**
- 現行 `configs/v460/fill_test.yaml` の `spread_offset_ratio=0.05` なので、基礎オフセット量は概算 **114.1 JPY**

このとき:

- `0.3bps ≈ 327.5 JPY` で **基礎オフセットの約 2.87 倍**
- `1.0bps ≈ 1,091.6 JPY` で **約 9.57 倍**
- `3.0bps ≈ 3,274.9 JPY` で **約 28.7 倍**、かつ **中央値スプレッド 2,282JPY を上回る**

つまり 374# §3.1 は「既存パイプラインの末尾に軽く足す」設計ではない。`scripts/v460/lib/fill_cycle_executor.py` の sidecar 注入は最終段にあり、3.0bps は **offset pipeline の補助量ではなく支配量** になり得る。

このため 3.1 の正しい初期案は以下である。

- `max_boost_bps` は **0.1-0.3bps** の ladder から開始
- same-SHA / same-window で `fill_rate`, `post_fill_30s_pnl`, `adverse_selected`, `postonly_crossing_skip` を比較
- 3.0bps は最後の上限候補であり、初期値ではない

### 2. HIGH: 374# §3.2-§3.3 は実装スコープを過小評価している

374# は 3.2 を「M2-M5 obs 注入」、3.3 を「LiteTradingEnv v2 の multi-dim action」としているが、現HEADとの距離が大きい。

確認結果:

- `scripts/v460/ml/sac_retrain_scheduler.py` の現行学習環境は **`HeavyTradingEnv`** であり、374# が前提とする `LiteTradingEnv v2` ではない
- `data/btc_jpy_1m_full_registry_features.parquet` は **77列**だが、以下の 374# 想定列は現時点で存在しない
  - `vpin_vol_sync`
  - `microprice_bias_bps`
  - `bayesian_trending_up`
  - `bayesian_trending_down`
  - `bayesian_ranging`
  - `bayesian_volatile`
  - `vol_cluster`
  - `glft_fill_prob`

つまり 3.2 は「特徴量を数列追加するだけ」ではない。実際には少なくとも次が必要である。

- オフライン前処理仕様の策定
- path-dependent な M2 Bayesian posterior の再現計算
- M3/M4/M5 を parquet 化する ETL
- 学習 env と live sidecar 入力契約の再統一

従って、374# の工数見積もり `4-8h` / `1-2d` は楽観的すぎる。**3.2 以降は小工事ではなくデータ契約の改版**である。

### 3. HIGH: 374# §11 の α 検証は、maker 実運用の評価軸として不十分

374# は `corr(bias_t, return_{t+k})` を主指標にしているが、これは direction 予測器の評価であって、**maker sidecar の収益評価**ではない。

現システムでは:

- SAC は注文そのものを出さず、`scripts/v460/lib/cycle_gate_aggregator.py` で offset modifier として注入される
- hard gate に blocked されたサイクルには影響できない
- 実際の収益は fill probability, queue position, adverse selection, timeout に強く依存する

したがって、bias と将来リターンの相関が正でも、maker 収益が改善するとは限らない。正しい一次判定は次である。

- `sidecar_offset_bps != 0` 群の **fill_rate 変化**
- `post_fill_30s_pnl` / `ev_weighted_pnl` の **uplift**
- `adverse_selected` 比率の変化
- `postonly_crossing_skip` / `spread_too_narrow` / `timeout` の悪化有無

要するに、**SAC α は return 相関ではなく maker uplift で検証すべき**である。

### 4. HIGH: 374# §6 Closed-Loop Reward は因果が崩れやすく、現段階では危険

`fill_records` を hindsight reward に使う発想自体は理解できるが、現段階で live 学習に入れるのは危険である。

理由:

- `actual_pnl` は sidecar 以外にも `buy_dynamic_kill`, `sell_dynamic_kill`, `daily_drawdown`, `toxicity`, `regime gate`, `inventory escape` 等の影響を受ける
- そのため `sidecar_bias * actual_pnl` は **policy の純効果**を表さない
- deploy 中 policy が生成した結果で policy を再学習すると、観測バイアスと自己強化が入る

374# 自身も自己批判しているが、ここはより強く扱うべきである。Closed-Loop Reward は現時点では **offline attribution study の題材**であって、ph3 の実装優先事項ではない。

### 5. HIGH: retained log では ph3 本命より ph2 の参加抑制がまだ支配的

`results/v460/fill_test/logs/fill_test.log` を確認すると、保持されているログでは:

- `[372# sidecar]` 出現回数: **0**
- `Sidecar signal updated` 出現回数: **0**
- `buy_dynamic_kill` 出現回数: **1628**
- `sell_dynamic_kill` 出現回数: **1195**

また、現ワークスペースには以下が存在しない。

- `cache/sidecar_signal.json`
- `logs/sac_retrain_history.jsonl`

一方で retained log には `2026-03-10` 付近で:

- `[M2] Bayesian regime filter enabled`
- `[M3] σ-clustering enabled for adaptation`
- `[M4] GLFT dynamic k enabled for AS δ*`

の初期化痕跡はある。

この意味は明確である。**市場理論システム M2-M4 の配線は確認できるが、SAC sidecar は retained artifact 上ではまだ live 効果を持った証跡がない。** よって 374# は「SAC をどう高度化するか」より先に「SAC を本当に live 経路へ載せて観測できているか」を詰めるべきである。

### 6. MEDIUM: 371# は「配線完了」というより「初期化・基礎配線完了」と表現すべき

371# の方向性自体は概ね妥当である。実コード上も:

- M2 Bayesian Regime は `scripts/v460/run_fill_test.py` で `regime_detector` に注入
- M3 σ-Clustering は `AdaptationEngine` に注入
- M4 GLFT dynamic k は `MakerPriceCalculator` / `AdaptationEngine` に注入
- M5 volume-sync VPIN は `skip_gate_evaluator.py` と `maker_price.py` 系で runtime に反映

ただし、これらは **「profit contribution が検証済み」** を意味しない。したがって 371# の表現は、より正確には「市場理論システムの基礎配線・有効化確認」である。

### 7. MEDIUM: 374# §12 の既存 SAC 資産棚卸しは有益だが、実装計画に混ぜると焦点がぼける

374# §12 は情報量が多く価値がある。ただし現局面では、これを「今すぐ再利用すべき実装候補」と「将来参照用アーカイブ」に分けるべきである。

今すぐ再利用してよいのは主に以下である。

- 解析系 (`sac_analyzer`, callback 群)
- 品質監視系
- 既存テスト観点

逆に、今の段階で取り込まない方がよいのは以下である。

- 旧 trainer stack の大規模再統合
- LSTM / Transformer への早期拡張
- curriculum 学習の再持込
- UnifiedTrainer 依存の再拡大

理由は単純で、**sidecar の α 未証明な段階で trainer 生態系まで広げると、デバッグ対象だけが増える**ためである。

### 8. MEDIUM: 文書運用のトレーサビリティに乱れがある

確認した範囲だけでも:

- `docs/v460/372_ph2_dust_sweep_refinement_and_sac_wiring_plan.md`
- `docs/v460/372_ph3_audit_report.md`

と **372 が二重採番**になっている。また `docs/v460/index.md` では `361` / `363` が重複掲載されている。

これは内容の正否とは別に、後から「どの判断に対するレビューか」を追えなくする。ph3 以降は文書量がさらに増えるため、ここは放置しない方がよい。

### 9. LOW: sidecar signal の診断メトリクス名に軽微なズレがある

`scripts/v460/ml/sac_retrain_scheduler.py` の `_update_sidecar_signal()` では、`training_metrics` に

- `gross_roi`
- `total_timesteps`

を書いているが、`total_timesteps` には実際には `trade_count` が入っている。軽微だが、将来の運用診断で誤読を招く。

---

## §3 374# の各 Phase 判定

| Phase | 374# の主張 | 判定 | コメント |
|---|---|---|---|
| 3.1 | Proportional Boost | **条件付き GO** | ただし `0.1-0.3bps` の縮小版で始めるべき。3.0bps は初期値として大きすぎる |
| 3.2 | Regime-Aware Observation | **HOLD** | parquet に必要列が無く、`HeavyTradingEnv` 前提も崩れる。前処理契約の策定が先 |
| 3.3 | Parameter Modulation | **NO-GO** | α 未証明の段階で action space を増やすのは時期尚早 |
| 3.4 | Closed-Loop Reward | **NO-GO** | 因果が崩れやすく、自己強化バイアスが強い。まず offline attribution に留めるべき |

---

## §4 369#-373# で維持すべき良い判断

以下は維持すべきである。

1. **369#/370# の profit-first への回帰**
   - まず ph2 の liveness / kill / DD / participation を抑えるという順序は正しい。

2. **372#/373# の「監査を先に潰す」方針**
   - `sidecar_bias` 記録、`SACRetrainConfig` バリデーション、TOCTOU 修正、`balance_checker` 例外安全化は、SAC 高度化以前の必須土台である。

3. **市場理論システムを sidecar と独立に保持する方針**
   - M2-M5 を ph2 執行系の改善として持ち、SAC は後段の modifier とする整理は妥当である。

つまり、374# は 369#-373# を踏まえた「次の設計メモ」であるべきで、**369#-373# で確立した優先順位を上書きしてはいけない**。

---

## §5 著者が見落としやすい盲点

### 5.1 「SAC を活かす」以前に「SAC が live に存在しているか」の確認が未完了

現 retained artifacts では sidecar の runtime 痕跡が薄い。したがって次の最小 KPI が必要である。

- `cache/sidecar_signal.json` が一定周期で更新される
- `logs/sac_retrain_history.jsonl` に retrain 履歴が残る
- `fill_records` に `sidecar_offset_bps` / `sidecar_bias` が non-null で入る
- log に `[372# sidecar]` が出る

これが揃って初めて「sidecar が利益に効いたか」を論じられる。

### 5.2 M2-M5 を obs に入れる前に、「オンライン計算」と「オフライン再現計算」の差を潰す必要がある

特に Bayesian posterior は path-dependent なので、単純な列追加では済まない。ここを曖昧にしたまま 3.2 に入ると、**学習時と live 時の特徴量定義ズレ**が再発する。

### 5.3 ph3 の学習器改善と ph2 の実収益改善を混ぜない方がよい

ph2 の本丸は live participation と quote quality である。ph3 はその上に乗る追加改善であり、主戦場を入れ替えてはいけない。

---

## §6 推奨優先順位

### P0: 374# をそのまま実装しない

374# は下記のように読み替えるべきである。

- 3.1 = **縮小版 sidecar offset 実験計画**
- 3.2+ = **保留中の構想メモ**

### P1: 3.1-lite を same-SHA で小さく検証する

推奨設定:

- `max_boost_bps`: `0.1`, `0.2`, `0.3` の ladder
- `dead_zone`: 0.10 で開始してよいが固定せず config 化
- 比較指標:
  - `fill_rate`
  - `post_fill_30s_pnl`
  - `adverse_selected`
  - `timeout`
  - `postonly_crossing_skip`

### P2: sidecar の live 可観測性を完成させる

最低限やるべきこと:

- scheduler 実行履歴の永続化確認
- sidecar signal artifact の常時存在確認
- fill_records で non-null 比率を集計する分析スクリプト追加
- `training_metrics.total_timesteps` の誤ラベル修正

### P3: 3.2 の前に feature contract 文書を切る

内容:

- どの列を parquet に持つか
- どの列が online-only / offline reproducible か
- path-dependent feature の再計算手順
- train/live 一致性の保証方法

これなしに 3.2 に入るべきではない。

---

## §7 確認結果

本レビューでは以下を確認した。

- 文書読込: `docs/v460/369_gemini_verification_and_next_steps.md`, `docs/v460/370_ph2_pivot_368_369_review_acceptance.md`, `docs/v460/371_ph2_366_market_theory_wiring.md`, `docs/v460/372_ph2_dust_sweep_refinement_and_sac_wiring_plan.md`, `docs/v460/372_ph3_audit_report.md`, `docs/v460/373_phg_fix_post_sac_audit_and_safety_hardening.md`, `docs/v460/374_ph3_design_sac_continuous_value_and_market_theory_integration.md`
- コード確認: `scripts/v460/ml/sac_retrain_scheduler.py`, `scripts/v460/lib/cycle_gate_aggregator.py`, `scripts/v460/lib/fill_cycle_executor.py`, `scripts/v460/lib/fill_record_builder.py`, `scripts/v460/run_fill_test.py`, `scripts/v460/lib/regime_detector.py`, `scripts/v460/lib/fill_probability_model.py`, `scripts/v460/lib/maker_microstructure.py`, `configs/v460/fill_test.yaml`, `configs/v460/experiments/g2_sac_train.yaml`
- retained data/log:
  - `results/v460/fill_test/fill_records_20260310.jsonl`
  - `results/v460/fill_test/logs/fill_test.log`
- テスト:
  - `tests/unit/v460/test_sidecar_sac_integration.py`
  - `tests/unit/v460/test_vpin_volume_sync.py`
  - `tests/unit/v460/test_bayesian_regime_filter.py`
  - `tests/unit/v460/test_sigma_clustering.py`
  - `tests/unit/v460/test_fill_probability_model.py`
  - `tests/unit/v460/test_356_g2_sac_blockers.py`
  - `tests/unit/v460/test_373_critical_fixes.py`

実行結果:

- `80 passed`
- `156 passed`

したがって、**現HEADは「安全性と基礎配線」は前進しているが、「SAC が利益寄与している」とまではまだ言えない**、というのが妥当な結論である。

---

## §8 最終判断

374# は破棄すべきではない。ただし、以下のように扱うべきである。

- **採用する部分**: 3.1 の発想、3.2 の feature contract 問題提起、既存資産棚卸し
- **保留する部分**: 3.2 全実装、3.3 multi-dim action、3.4 closed-loop reward
- **修正すべき部分**: 3.1 の数値感、α検証の指標、工数見積もり、live readiness の前提

Profit-first に徹するなら、次の一手は「SAC を大きくする」ことではない。

**SAC を 0.1-0.3bps の極小 modifier として live に確実に存在させ、その増分価値を same-SHA で証明すること**である。
