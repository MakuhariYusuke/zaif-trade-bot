# 368# ph2 362#-367# レビュー: SAC接続・市場理論・緊急ブロッカー再整理

| 項目 | 値 |
|---|---|
| 文書番号 | 368# |
| フェーズ | ph2 / ph3 境界レビュー |
| 対象 | 362#-367# + 現行コード + 2026-03-10 時点ログ |
| レビュー日時 | 2026-03-10 |
| 観点 | 収益最優先 / システム工学 / 市場理論 |

---

## §1 結論

362#-367# で整理された「SAC は direct trader ではなく sidecar として使うべき」という方向性自体は正しい。  
ただし、**2026-03-10 時点では sidecar は live 実行系に未接続**であり、さらに **sidecar signal 生成経路にも current market を見ていない設計欠陥**が残っている。

したがって、直近の優先順位は以下:

1. **ph2 live の本丸ブロッカー** (`buy_dynamic_kill` / `daily_drawdown` 再停止 / ranging sell の逆選択) を先に詰める
2. **sidecar を end-to-end で正しく配線**する
3. **SAC の live deploy gate を G2 相当に引き上げる**

逆に、367# の `FIX-0 post_fill PnL coverage 0%` は**現時点では最優先ではない**。現ログでは 30s coverage は復旧している。

---

## §2 Findings

| # | 重大度 | 対象 | 問題 | 根拠 / 推奨対応 |
|---|---|---|---|---|
| F1 | **CRITICAL** | `scripts/v460/lib/orchestrator_mid_cycle.py:135` `scripts/v460/lib/orchestrator_mid_cycle.py:383` | **SAC sidecar が live 実行系に未接続。** 365#/366# で設計・テストはあるが、実運用経路では `sidecar_signal` を読まず、`sidecar_offset_bps` も使っていない。 | `CycleGateAggregator.evaluate()` 呼出しに `sidecar_signal` が渡されていない。`run_single_cycle()` へも sidecar 由来の offset が渡っていない。`read_sidecar_signal()` は live 経路から未参照。`tests/unit/v460/test_sidecar_sac_integration.py` が green でも live PnL には未寄与。**「設計完了」と「通電完了」を分けて扱うこと。** |
| F2 | **CRITICAL** | `scripts/v460/ml/sac_retrain_scheduler.py:648` `ztb/trading/environment/heavy_env/core.py:685` `ztb/trading/environment/utils/config.py:312` | **sidecar signal 生成が current market を見ていない。** `env.reset()` 後の観測から bias を書き出しており、これは train window の先頭バー。 | `random_start=False` がデフォルトなので `reset()` は `current_step=0`。このため `_update_sidecar_signal()` が書く bias は「現在」の相場観ではなく、学習窓の先頭状態。**signal は env からではなく最新 feature row から直接推論すること。** |
| F3 | **HIGH** | `scripts/v460/ml/sac_retrain_scheduler.py:316` `scripts/v460/run_experiment.py:321` | **sidecar の live deploy gate が G2 より弱すぎる。** Scheduler 側は単一 seed・単一 split・`gross_roi > 0` だけで deploy する。 | `run_experiment.py` の G2 判定は seed 間安定性・convergence・worst seed を見る一方、scheduler は OOS `gross_roi` 単独。**SAC を live に繋ぐなら scheduler でも G2 相当の gate を要求すべき。最低でも multi-window / worst-window floor / stability を追加。** |
| F4 | **HIGH** | `docs/v460/367_ph2_deep_analysis_360_ai_review.md` `results/v460/fill_test/fill_records_20260308.jsonl` `results/v460/fill_test/fill_records_20260309.jsonl` `results/v460/fill_test/fill_records_20260310.jsonl` | **367# の `post_fill PnL coverage 0%` は現時点では stale。** これを前提に優先順位を組むと作業を外す。 | 実ログ再集計では 30s coverage は `2026-03-08: 146/146`, `2026-03-09: 188/188`, `2026-03-10: 27/27`。`results/v460/fill_test/logs/fill_test.log` にも `Waiting ... for PnL measurement` と `pnl=...bps` が継続出力。**FIX-0 は「再発監視」へ降格し、最優先は別件へ移すべき。** |
| F5 | **HIGH** | `results/v460/fill_test/fill_records_20260310.jsonl` `results/v460/fill_test/logs/fill_test.log` `results/v460/fill_test/fill_test_state.json` | **現時点の緊急ブロッカーは `buy_dynamic_kill` と `daily_drawdown` 再停止。** ph2 はここで実質停止しており、SAC を繋いでも利益に届かない。 | 2026-03-10 日次 records では `buy_dynamic_kill=21`, `skip_gate=14`, `spread_too_narrow=9`, `stale_adverse_drift=8`。`fill_test.log` では `08:35`, `09:49` に `DEADLOCK WARNING: 10 consecutive gate blocks (reason=buy_dynamic_kill)`、`11:31` に sell fill `-41.76bps` を受け `daily_drawdown` が再停止。`fill_test_state.json` でも `halted=true`, `daily_pnl_bps=-82.07`。**P0 は BDK と DD re-arm の分離観測・緩和設計。** |
| F6 | **HIGH** | `results/v460/fill_test/logs/fill_test.log` | **今の利益源は microstructure edge より inventory MTM に寄っている。** SAC を offset 微調整器としてだけ繋ぐと、在庫リスクが暗黙化する。 | `2026-03-10 11:17:39` 時点で `spreadPnL=-760.7JPY` に対し `btcMTM=+26600.1JPY`, `totalEquityΔ=+25839.3JPY`。これは「実質 directional inventory」で勝っている局面。**SAC 出力は quote offset より `target inventory bias` に寄せた方が理にかなう。** |
| F7 | **MEDIUM** | `docs/v460/365_ph3_design_sac_sidecar_and_env_obstacles.md` `results/v460/fill_test/logs/fill_test.log` | **「市場理論システムを外せるようにした」は training 用には正しいが、live 用には危険。** 外してよいものと外してはならないものを分ける必要がある。 | SAC 訓練で reward shaping / hybrid override を切るのは妥当。一方、live では `dynamic_kill`, `daily_drawdown`, `toxicity veto`, `post_only`, `spread floor`, `phantom guard` が大損失を抑えている。**外してよいのは train-time の歪み要因であり、live hard guard ではない。** |
| F8 | **MEDIUM** | `results/v460/fill_test/judgment_168h.json` `results/v460/fill_test/logs/watchdog.log` | **運用判断に使う成果物の時点が混在している。** stale artifact を current truth と混ぜると判断が濁る。 | `judgment_168h.json` は 2026-02-20 時点成果物で、2026-03-10 の current run ではない。一方 watchdog は 2026-03-10 に 13h+ 連続 RUNNING。**run_id / date / git_sha を固定した current snapshot を SSOT にするべき。** |
| F9 | **MEDIUM** | `results/v460/fill_test/fill_records_20260310.jsonl` | **2026-03-10 は `ranging` 優勢で、sell の本丸は方向予測ではなく逆選択。** | 日次 records は `ranging=69/85`。filled の 30s PnL は `buy/ranging=-0.199bps (n=12)`, `sell/ranging=-3.667bps (n=12)`。この局面で OHLCV 由来の directional SAC を強く効かせても効率は低い。**ranging では sidecar は neutral 優勢、必要なら participation / inventory だけ触る方が良い。** |

---

## §3 362#-367# のうち「今も正しいもの」と「更新すべきもの」

### 3.1 今も正しい

1. **SAC は direct execution ではなく sidecar**  
2. **G2 は out-of-sample / seed stability を見るべき**  
3. **train-time の市場理論 override は SAC 学習を歪めうる**  
4. **ph2 の microstructure 問題を放置したまま ph3 へ期待しすぎるのは危険**

### 3.2 更新すべき

1. **367# の `FIX-0 post_fill coverage 0%` は current truth ではない**  
2. **OPS-5 は repo 側 XML 問題ではなく、本番 drift 確認問題**  
3. **sell_dynamic_kill は「完全終了」ではなく「主役から降りた」程度**  
4. **今の主役は `buy_dynamic_kill` と `daily_drawdown` 再停止**

---

## §4 SAC 接続にあたり「外してよいもの / 外してはならないもの」

| 区分 | 項目 | 判断 |
|---|---|---|
| 外してよい | `HeavyTradingEnv` 側の reward shaping / curriculum / hybrid entry-exit override / signal guidance | **可**。SAC 学習の credit assignment を歪めるため |
| 外してよい | train-time の regime multiplier や強制多様性 | **可**。まず pure PnL / pure OOS を見る方がよい |
| 外してはならない | live `post_only`, spread floor, crossing guard | **不可**。maker 優位の最低条件 |
| 外してはならない | live `buy/sell_dynamic_kill`, `daily_drawdown`, `toxicity veto` | **不可**。Glosten-Milgrom / 在庫リスクに対する最後の防波堤 |
| 外してはならない | `phantom_position_guard`, watchdog, state save | **不可**。liveness と整合性の基盤 |
| 触るなら慎重 | `skip_gate` の side 別しきい値 | **要注意**。`sg < -2` の sell fill に勝ち負けが混在しており、単純締め付けは winners も消す |

**整理**:  
SAC を活かすために外すべきなのは「訓練を歪める market-theory wrapper」であり、  
外してはならないのは「live の損失制御・執行品質制御」である。

---

## §5 市場理論から見た SAC の正しい役割

### 5.1 direct quote policy にしない理由

現在の負け方は、主に以下:

1. `ranging` での sell 逆選択
2. `buy_dynamic_kill` による長時間ブロック
3. `daily_drawdown` 再停止による duty cycle 崩壊

これは 1 分足 OHLCV policy が直接 quote を出して解決する類ではない。  
Glosten-Milgrom 的には、live quote は informed flow に対して常に不利であり、  
OHLCV SAC にそこを直接任せるのは情報粒度が足りない。

### 5.2 sidecar としてなら活かせる役割

1. **target inventory bias**  
2. **buy/sell participation bias**  
3. **reservation price の微小バイアス**  

この順で使うのがよい。  
特に現ログでは MTM が利益源になっているため、SAC は「在庫目標」を明示化する方が自然。

### 5.3 ranging での扱い

2026-03-10 は `ranging` が大半で、`sell/ranging` が特に悪い。  
この局面では「方向を当てる」より「無理に参加しない」「在庫を増やし過ぎない」の方が価値がある。

したがって ranging では:

1. `neutral` を基本
2. 高 confidence のときだけ bias を有効化
3. bias も offset 直結ではなく inventory / participation に優先配分

が妥当。

---

## §6 Profit-First 次アクション

### P0: SAC 接続前に必須

1. **F1 修正**: `read_sidecar_signal()` → orchestrator → gate_result → executor / maker_price の end-to-end 配線  
2. **F2 修正**: sidecar signal を `env.reset()` ではなく「最新 feature row」から生成  
3. **F5 対応**: `buy_dynamic_kill` の deadlock と `daily_drawdown` re-arm を分離観測して再設計

### P1: 収益直結

1. `buy_dynamic_kill` を hard kill 一本ではなく staged response 化  
   例: participation 縮小 → offset 拡大 → hard kill  
2. `daily_drawdown` 再停止条件を「単発 fill 30s PnL」依存から見直す  
3. ranging sell の逆選択を、SAC ではなく ph2 microstructure 側で抑える

### P2: SAC 有効化後

1. sidecar の効果は **微小** (`±0.1-0.3bps` 程度) から始める  
2. hard guard を bypass させない  
3. deploy gate は scheduler 単独基準ではなく G2 相当に寄せる  

---

## §7 実測確認

今回のレビューでは以下を確認した。

1. `tests/unit/v460/test_356_g2_sac_blockers.py`  
2. `tests/unit/v460/test_sidecar_sac_integration.py`  
3. `tests/unit/v460/test_fill_test_cli_diagnostics.py`  
4. `tests/unit/v460/test_fill_test_watchdog_ops.py`  

結果: **94 passed**

加えて、`data/btc_jpy_1m_full_registry_features.parquet` は存在し、  
G2 用 12 特徴量は **1,216,930 rows / 12 cols / null 0** を確認した。

---

## §8 最終判断

**SAC を繋ぐ方向性そのものは正しい。**  
ただし現時点では、

1. **配線が未完**
2. **signal 生成が current market を見ていない**
3. **live の主ブロッカーが ph2 側に残っている**

ため、**今のまま繋いでも儲けには直結しない**。

直近の勝ち筋は、

1. `buy_dynamic_kill` / `daily_drawdown` を先に詰める  
2. SAC は `inventory bias sidecar` として限定接続する  
3. live hard guard は絶対に外さない

この順で進めること。
