# 389# 383-388レビュー: Profit-First 再整理と G3 接続前監査

## 概要

383#-388# の流れは、`382#` で露呈した SAC/G2 計測破綻を是正し、
`385#` で設定監査、`387#` で reward tuning、`388#` で G3-pnl 準備へ進める構図になっている。

方向性そのものは概ね正しい。ただし、現時点では次の 3 つを混同している。

1. **パイプライン修正の達成**
2. **G2 Gate 上の通過**
3. **儲かるモデルの獲得**

この 3 つは同義ではない。特に `387#` と `388#` は、G2 PASS をもって
「次の高収益候補が固まった」と読める書き方になっているが、profit-first の観点ではまだ早い。

---

## 総合判定

- `384#` と `385#` の修正は実質的で、ここは前進
- `387#` の reward-tuned は **Gate 適合性の改善** としては意味がある
- しかし **利益最大化の観点では、reward-tuned を第一候補と断定できない**
- `388#` の G3 方向は正しいが、**新規実装を増やし過ぎると再び二重化する**

結論として、次にやるべきことは
**「reward-tuned を称揚すること」ではなく、
「G2 PASS の再現性・reward と PnL の整合・G3 計測の一本化」を先に固めること**
である。

---

## P0: CRITICAL

### P0-1. `387#` / `388#` の「G2 PASS」は再現可能だが、保存 artifact はまだ `FAIL`

reward-tuned の実体とみられる `results/v460/v460_g2train_seed42_20260312_073155.json` は、
保存済み `g2_judgment_cache` ではまだ `FAIL` である。

同 JSON の seed 結果:

| Seed | gross_roi |
|---|---:|
| 42 | +0.01% |
| 123 | +0.39% |
| 456 | -3.21% |
| 789 | +4.09% |

保存済み cache は旧 E4 閾値 `-0.02` を使っており `FAIL`。
一方、現行 `configs/v460/gate_thresholds.yaml` は `worst_seed_min_roi: -0.035` なので、
現行 gate で再判定すると **PASS** になることを確認した。

確認結果:

```bash
./.venv/Scripts/python.exe scripts/v460/run_gate_check.py \
  --gate G2 \
  --results-path results/v460/v460_g2train_seed42_20260312_073155.json
```

出力: `gate_result = PASS`

つまり `387#` / `388#` の主張は **現行 gate policy 上は成立** する。
しかし、保存済み実験 artifact はまだ `FAIL` のままであり、
後続工程が `g2_judgment_cache` を見ると判定が食い違う。

**問題の本質**:
- モデル改善
- gate policy 変更
- 保存済み artifact の整合

が 1 本化されていない。

**推奨対応**:
- reward-tuned の G2 判定を current threshold で再保存した JSON を別名で出す
- `387#` / `388#` に「PASS は E4 緩和後の再判定結果」と明記する
- 今後は gate policy 変更時に既存 artifact を再審査する運用を決める

### P0-2. reward-tuned は Gate を通したが、profit-first では最良候補と断定できない

`387#` の 4 実験比較では平均 ROI は以下だった。

| 実験 | Mean ROI | G2 |
|---|---:|---|
| baseline | **+1.39%** | FAIL |
| gamma=0.95 | +0.74% | FAIL |
| reward-tuned | **+0.32%** | PASS |
| warm-start | -2.65% | FAIL |

つまり reward-tuned は **最も儲かった構成ではない**。
通したのは主に

- E2 seed 分散の収まり
- E4 閾値緩和後の許容

であり、平均収益の優位ではない。

profit-first に読むなら、
`387#` の本当の意味は
「reward-tuned が儲かるモデルを見つけた」
ではなく
「reward-tuned が current G2 governance に最も適合した」
である。

**推奨対応**:
- `388#` の前提を「G2 PASS 確定モデル」ではなく「G2 通過候補」に弱める
- G3 候補は reward-tuned 1 本に固定せず、baseline / gamma095 も同じ計測軸で比較する

### P0-3. reward と PnL の整合が壊れており、reward-tuned はむしろ悪化している

ここが一番重要である。

reward-tuned の `raw_results[*].eval_metrics` を見ると、
`mean_reward` と `gross_pnl` / `gross_roi` の符号が一致しない seed が複数ある。

reward-tuned (`20260312_073155`) の例:

| Seed | gross_roi | gross_pnl | mean_reward |
|---|---:|---:|---:|
| 42 | +0.01% | +1,226 JPY | **+335,640** |
| 123 | +0.39% | +39,345 JPY | +81,719 |
| 456 | -3.21% | -320,273 JPY | **+103,157** |
| 789 | +4.09% | +409,345 JPY | **-9,243** |

4 seed ベースでの `mean_reward` と `gross_pnl` の相関は:

| 実験 | corr(mean_reward, gross_pnl) |
|---|---:|
| baseline | +0.89 |
| gamma=0.95 | +0.98 |
| reward-tuned | **-0.38** |

sample は 4 seed と少ないが、reward-tuned だけ負相関に反転しているのは見過ごせない。

これは市場理論以前に、**学習目標が利益最大化からズレている** ことを示す。
profit-first である以上、G3 へ進む前に
「reward を最大化したとき本当に PnL が伸びるか」
を最低限再点検すべきである。

**推奨対応**:
- G2/G3 前に `reward_profit_alignment` 指標を追加する
- 具体的には seed ごとに `gross_pnl`, `gross_roi`, `mean_reward`, `trade_count` を必須出力にし、
  符号一致率と相関を監視する
- reward-tuned を live 候補にする前に、少なくとも `corr(mean_reward, gross_pnl) > 0` を要求する

---

## P1: HIGH

### P1-1. `388#` の G3 計画は方向は正しいが、`g3_gate_check.py` 新設は二重化になる

`388#` は `scripts/v460/lib/tasks/g3_gate_check.py` の新設を提案している。
しかし現行 repo には既に `scripts/v460/run_gate_check.py` の `run_g3_judgment()` がある。

G3 に必要なのは gate ロジックの新設ではなく、
`sac_train.py` 側が `seed_metrics` を出せるようにすることだと整理すべきである。

既存 G3 判定が期待する入力:

- `pf`
- `sharpe_annual`
- `max_drawdown`
- `avg_gross_per_trade`
- `avg_fee_per_trade`

**推奨対応**:
- 新規 `g3_gate_check.py` は作らず、`run_gate_check.py --gate G3` に統一する
- `evaluate_model_oos()` かそのラッパーを拡張して `seed_metrics` を出力する
- 判定責務は既存 gate checker、計測責務は evaluator に分離する

### P1-2. G3 に進んでも、今の cost model では実利益を過大評価しやすい

現行 G2/G3 候補 YAML は `transaction_cost: 0.0` で、`slippage` も明示されていない。
`EnvironmentConfig` のデフォルト上 `slippage=0.0` である。

Coincheck maker 0% 前提自体は文脈上理解できるが、
市場理論上は **maker 0% = 実コスト 0** ではない。

残るコスト:

- queue miss
- adverse selection
- 約定遅延
- 部分約定後の在庫持ち越し
- 実運用時の taker 化

特に reward-tuned は平均 trade 数が `44,254 → 184,315` に急増しており、
この turnover で friction を 0 扱いすると G3 は楽観化しやすい。

**推奨対応**:
- G3 では main 判定の他に stress 判定を入れる
- 例: `fee=0`, `fee=0 + slippage 1tick`, `fee=0 + maker miss penalty` の 3 条件
- reward-tuned のような高回転モデルは、stress 条件で崩れるなら live 候補から外す

### P1-3. `383#` の `gamma=0.99` / `500K` / curriculum は一般論としては正しいが、今の最優先ではない

Gemini 383# の提案:

- `gamma=0.99` 以上
- `curriculum_learning`
- 500K steps 以上

は RL 一般論として妥当な部分がある。
ただし今の局面でこれを先頭に置くと、再び「長時間回したが評価軸がズレていた」を繰り返す危険がある。

現時点で先に固めるべきは:

1. reward と profit の整合
2. G3 計測出力の一本化
3. artifact 判定の整合

である。

その後に、

- `gamma=0.95 → 0.99`
- curriculum
- timesteps 拡大

を same-metric で比較すべきで、いきなり 500K を既定路線にするのは早い。

---

## P2: MEDIUM

### P2-1. SB3 cleanup は正しい方向だが、完了扱いはまだ早い

`384#` の整理は妥当で、現状もその判断を支持する。

現物確認:

- `_sb3_test_stub/` は残存
- `ztb/support/sb3_compat.py` は残存
- `tests/conftest.py` は依然 `ensure_sb3_compat()` を import
- `tests/conftest.py` には SB3 の global fallback 注入コードが大量に残存
- `sitecustomize.py` は実質 docstring のみで害は小さい

従って 383# の「即全削除」は理想論としては理解できるが、
実務上は 384# の
「conftest detox を先にやる」
の方が正しい。

**推奨対応**:
- test cleanup 系は別 Codex 作業と整合を取りつつ、まず `tests/conftest.py` 依存を落とす
- その後 `_sb3_test_stub/` と `sb3_compat.py` を消す
- `sitecustomize.py` は最後でよい

### P2-2. 文書と設定の追跡性に drift がある

確認できた drift:

1. `configs/v460/experiments/g2_sac_gamma095_reward_tuned.yaml`
   の根拠コメントが `docs/v460/386_reward_analysis.md` を指している
2. `configs/v460/experiments/g2_sac_train.yaml`
   のコメントは「12特徴量全載」と読めるが、現行 selected は 17 特徴量
3. `387#` は「旧番号 386# → 387#」を明記しているが、周辺 YAML/コメントの追従が不完全

軽微に見えるが、phase が長引くほど
「どの設定がどの判断根拠か」
を失いやすい。

**推奨対応**:
- reward-tuned YAML の根拠コメントを 387# に更新
- G2 YAML コメントの特徴量数を現行値へ修正
- gate policy 変更時は docs と artifact の両方を同期させる

### P2-3. `388#` の既存実装活用は selective reuse が正しい

`388#` が挙げた既存資産のうち、使い方に強弱がある。

妥当:

- `ztb.types.evaluation_types.EvaluationMetrics`
- `ztb.metrics.metrics`
- `ztb.trading.comprehensive_backtest`
- `ztb.analysis.backtest.analyze_backtest`

注意:

- `ztb.training.reward_function_evaluator` は存在するが **deprecated**

したがって G3 では
**型・指標関数は再利用し、古い evaluator orchestration には寄せ過ぎない**
のが妥当である。

---

## 既存実装の活用提案

### 1. G3 指標計算

再利用候補:

- `scripts/v460/run_gate_check.py` の `run_g3_judgment()`
- `ztb.metrics.metrics`
- `ztb.types.evaluation_types.EvaluationMetrics`

方針:
- evaluator 側で `seed_metrics` を組み立てる
- gate 判定は既存 G3 に渡す

### 2. trade log / equity 出力

再利用候補:

- `ztb.trading.live.simulation.sim_broker.py`
- `ztb.trading.live.simulation.paper_trader.py`

これらは既に

- `trade_log`
- `pnl_series`
- `stats.json`

を出す流れを持っている。
G3 用に新規フォーマットを設計するより、既存の出力粒度に揃える方が保守しやすい。

### 3. 市場理論の反映

G3 前に最低限欲しいのは:

- turnover
- average holding time
- positive trade ratio
- profit factor
- fee/slippage stress

reward-tuned は trade 数が急増しており、これは
「penalty を下げたので儲けやすくなった」
というより、
「行動量が増え、friction 非考慮下で only ROI を押し上げた可能性」
を疑うべきである。

---

## 次アクション優先順位

### P0

1. reward-tuned の G2 判定を current threshold で再保存し、artifact と docs を一致させる
2. `reward_profit_alignment` を seed 単位で出力し、reward と PnL の整合を可視化する
3. G3 は新規 checker を作らず、既存 `run_gate_check.py` に入力を合わせる

### P1

1. reward-tuned / baseline / gamma095 を同一 G3 指標で横並び比較する
2. `fee=0`, `slippage>0`, `maker miss` の stress 条件を入れる
3. その後に初めて `gamma=0.99` / curriculum / 500K を比較する

### P2

1. `tests/conftest.py` 依存を整理した後、`_sb3_test_stub/` と `sb3_compat.py` を撤去する
2. YAML コメントと docs 番号 drift を掃除する

---

## 検証メモ

今回確認した主な事項:

- `tests/unit/v460/test_384_pipeline_fixes.py`
- `tests/unit/v460/test_385_config_audit.py`
- `tests/unit/v460/test_385_transaction_cost.py`
- `tests/unit/v460/test_356_g2_sac_blockers.py`

結果:

- `36 passed`
- `47 passed`

加えて以下を確認:

- `results/v460/v460_g2train_seed42_20260312_073155.json`
  は保存 cache では `FAIL`
- 同一ファイルを `run_gate_check.py --gate G2` で current thresholds で再判定すると `PASS`

---

## 最終結論

383#-388# で最も価値があったのは、
**SAC の計測パイプライン修正と設定監査** である。

一方、最も危険なのは、
**reward-tuned の G2 PASS を「儲かるモデル発見」と読み替えてしまうこと**
である。

現段階での正しい整理は次の通り。

- `384#` / `385#`: 前進
- `387#`: gate 適合性の改善。ただし reward-profit alignment は悪化
- `388#`: G3 へ進む方向は正しいが、新規実装の増殖は避けるべき

profit-first で進めるなら、
次に固めるべきは **G3 接続そのもの** ではなく、
**「reward が profit を向いているか」の再検証**
である。
