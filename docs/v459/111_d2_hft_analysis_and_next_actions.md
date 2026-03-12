# 111# D2結果再点検（v459整合版）: HFT志向と次アクション

**Date**: 2026-02-11  
**対象**: `110_d2_swing_experiments_and_109_validation.md`（再点検）  
**追加観点**: `00/66/91` との整合、論理誤りの検証、既存実装・過去vXXX資産の再利用

---

## 0. 改訂結論（先に要点）

1. `d2_cost05@10K` は改善方向として有効だが、現時点は「HFTで勝てる状態」ではない。  
2. 収益未達の主因は依然としてコスト吸収不足（`avg_gross/avg_fee = 0.51`）。  
3. 旧111の方向性は概ね妥当だが、`0/66/91` 基準で見ると「評価信頼性」と「既存資産優先」の記述が不足。  
4. 次ステップは機能追加より、`Fix First + Validate`（0番方針）で評価経路の信頼性を先に担保すべき。  
5. ステップ数拡張は後回しで正しい。10Kで悪条件を刈り、通過条件を満たした候補のみ50Kへ進める。

---

## 1. `0/66/91` との整合性点検

| 観点 | 基準（0/66/91） | 旧111の状態 | 修正方針 |
|---|---|---|---|
| 0番: No New Features | 新規機能追加より統合・品質担保優先 | 新規計測提案がやや先行 | 既存計測・既存実装の再利用を優先 |
| 0番: Fix First | 既知の不整合修正が先 | 再現性FAILを前提に評価を進めていた | eval経路の状態汚染を先に解消 |
| 66番: Validate | ROI以外の多指標・因果分離・統計妥当性 | ROI中心で、確証不足の推定が混在 | 10Kは「棄却判定」に限定し、確証は50K/OOSで取得 |
| 91番: Tiny Edge対策 | Gross/Net差分を常時計測し、コスト負けを主課題化 | 方向性は一致 | `edge_after_cost` を採択ゲートに昇格 |
| 91番: v451再利用 | `gamma=0.80`, 非対称損失の再検証 | 一部言及のみ | 既存実験定義（`c1_*`, `d2_asymm12`）を優先活用 |
| 91番: v456教訓 | ペナルティ積み増し回避 | 新規フィルタ案が多め | まず閾値・保持・コストなど既存ノブで改善 |

---

## 2. D2結果の再評価（事実ベース）

### 2.1 有効だった点（`d2_cost05@10K`）

- eval ROI: `-2.247%`  
- eval trades: `449`（closed trades: `225`）  
- trade win rate: `41.3%`  
- avg gross/trade: `+10.06 JPY`  
- avg fee/trade: `19.73 JPY`  
- avg net/trade: `-9.67 JPY`  

方向性（粗利正値化）は出ているが、収益化は未達。

### 2.2 不確実性・弱点（見落とし補正）

1. `d2_cost10@10K` は学習後の評価途中で中断（`KeyboardInterrupt`）し、最終JSONがない。  
2. 再現性チェックはFAIL継続（`roi_diff_pt=0.393186`, `trades_match=false`）。  
3. `d2_cost05_v2`（50K別実行）では eval ROI `-16.95%` と大幅悪化。実行安定性が低い。  
4. `train_end_index not provided` 警告が出ており、将来OOS時のリークリスクが残る。  

---

## 3. 「高値で売り・安値で買う」理想への接近度

### 3.1 定量評価

- fee coverage ratio = `10.06 / 19.73 = 0.51`（損益分岐1.00未満）  
- 損益分岐に必要な改善 = `+9.67 JPY/trade`（現状比 約+96%）  
- ROI +5%到達に必要な改善 = `+31.90 JPY/trade`（closed trade基準）  
- 取引頻度は既に高い（closedベース約`32.4` roundtrip/day）  

### 3.2 判定

- 頻度不足よりも、**1取引あたり期待値不足**が主因。  
- 現状は「理想に近づき始めたが、まだ半分未満」。  
- 実務判定としては、接近度は **30-40%程度** が妥当。

---

## 4. 論理点検で見つかった重要修正点

### 4.1 eval経路の信頼性リスク（優先修正）

`scripts/v459/run_phase_c.py` の `_deterministic_eval_gate2()` は、同一 `raw_env` を
`evalA -> reproducibility_check -> evalB -> multi_dd` で逐次再利用している。  
`dd100`再利用の修正は入っているが、他の評価間で状態がブリードする余地が残る。

### 4.2 動的閾値の実験経路

`ztb/trading/environment/components/threshold_manager.py` には  
`dynamic_threshold_mode`, `z_score_window`, `z_score_threshold` が存在する。  
一方、`run_phase_c.py` の実験設定経路ではこれらを明示設定しておらず、  
`EnvironmentConfig` 側でも明示フィールドが不足しているため、現状は固定閾値運用が中心。

### 4.3 66番基準での主張強度

10K結果で「良い条件の候補」は選べるが、「有意な勝ち筋確定」はできない。  
よって10Kは棄却判定専用、確証は50K + walk-forwardで取るべき。

---

## 5. 既存実装の再利用マップ（v459方針準拠）

### 5.1 即時再利用（優先）

| 用途 | 実装 |
|---|---|
| ベースライン比較（Random/BuyHold/Momentum） | `scripts/v459/run_baselines.py` |
| Walk-Forward分割 | `ztb/evaluation/walk_forward/splitter.py` |
| Walk-Forward評価 | `ztb/evaluation/walk_forward/evaluator.py` |
| 統合評価器 | `ztb/evaluation/unified_evaluation.py` |
| 閾値制御ロジック | `ztb/trading/environment/components/threshold_manager.py` |
| 実行モデル（現実約定） | `ztb/trading/execution/realistic.py` |
| 実行モデル（疑似HFT） | `ztb/trading/execution/pseudo_hft.py` |
| 特徴量前処理（メモリ安全） | `scripts/v459/precompute_optimized_features_memory_safe.py` |
| 報酬純度ガード | `tests/unit/trading/components/test_gate05_reward_purity.py` |

### 5.2 条件付き再利用

| 用途 | 実装 | 条件 |
|---|---|---|
| v451設定再現 | `config/v451/sac_v451_optimized.json` | 現行環境でA/B因果分離が前提 |
| 動的閾値運用 | `ztb/trading/environment/components/threshold_manager.py` | `EnvironmentConfig` 経路の明示配線を先に実施 |
| 報酬差分検証 | `d2_asymm12`（`run_phase_c.py`） | PnL-only系と分離して比較 |

### 5.3 非推奨（現時点）

- まずは `ztb/evaluation/unified_evaluation.py` を主経路にし、  
  `ztb/analysis/evaluation/unified_evaluation.py`（deprecated shim）への依存を増やさない。  
- 大規模新機能（新アルゴリズム/新環境）より、既存経路の品質担保を優先。

---

## 6. 過去vXXXシリーズの再利用（91番の再具体化）

| 由来 | 教訓 | D2での具体適用 |
|---|---|---|
| v451 | `gamma=0.80` は短期志向に有利 | `c1_gamma_080` 系をD2候補比較に再投入 |
| v451 | 損失1.2倍の非対称性 | `d2_asymm12` を10Kスクリーニングで継続 |
| v457.2 | Tiny Edgeはあるがコストで負ける | `avg_gross`, `avg_fee`, `avg_net` を採択ゲート化 |
| v457.3 | チャーン抑制は有効 | `threshold` と `min_holding` を軽度（3-5）で最適化 |
| v456 | ペナルティ積み増しは失敗しやすい | 新ペナルティ追加は後回し、まず既存ノブ調整 |
| v454 | 強シグナル過信は危険 | action強度bin別の損益診断を実施 |

---

## 7. 次アクション（ステップ拡大なし版）

### 7.1 Phase D2.1（最優先: 評価信頼性）

1. `run_phase_c.py` の evalを「fresh env単位」に分離。  
2. 再現性判定を先に通し、FAIL条件はランキング対象外にする。  
3. `train_end_index` を明示し、将来OOS時のリーク警告を消す。

### 7.2 Phase D2.2（10Kスクリーニング継続）

1. 高頻度維持で `threshold` を `0.70/0.75/0.80` で比較。  
2. `min_holding_period` は `3/5` で比較（10/30は後段）。  
3. `cost=0.0005` を主軸、`gamma=0.80` と `loss×1.2` を独立要因で比較。  

### 7.3 10K採択ゲート（厳格化）

- `avg_gross_per_trade / avg_fee_per_trade >= 0.70`  
- `avg_net_pnl_per_trade >= -3.0 JPY`  
- `trade_win_rate >= 45%`  
- 再現性PASS（`roi_diff_pt < 0.2` かつ `trades_match=true`）

### 7.4 50K移行条件

10K採択ゲートを満たした上位候補のみ50Kへ進む。  
50K完走後に `walk_forward` で最終判定（66番要求に整合）。

---

## 8. 最終判定（改訂版）

- 旧111の「スイング寄せより選別型HFT」は方向性として妥当。  
- ただし、`0/66/91` 基準では「評価信頼性の担保」が先。  
- 現在は「高値売り・安値買いの芽」はあるが、利益化には未達。  
- 最短ルートは **既存実装の徹底再利用 + 10K棄却判定の厳格化 + 通過条件のみ50K/OOS**。

---

## 9. `run_phase_c.py` 整理計画（重複削減 / Godオブジェクト回避）

### 9.1 現状の問題（構造）

1. `scripts/v459/run_phase_c.py` は **1,317行**で、責務が過密。  
2. 実験定義、設定ビルド、学習実行、評価、集計、保存、CLIが1ファイルに同居。  
3. 重複ロジックが散在し、変更点の波及範囲が広い。  
4. `TOTAL_TIMESTEPS` のグローバル上書き等、実行時副作用が追跡しにくい。  

### 9.2 重複の主要ポイント（優先削減）

| 重複/密結合 | 現在 | 改善方針 |
|---|---|---|
| Gate2計算 | `compute_gate2_metrics_from_balances` と `compute_gate2_metrics` が実質同型 | 1つの純関数へ統合し、env→balances抽出は別関数化 |
| 評価フロー | Eval-A / 再現性 / Eval-B / Multi-DD が手続き重複 | `evaluate_modes()` に集約し、モードをデータ駆動化 |
| 実験定義 | `get_experiment_configs()` + `BATCHES` が巨大ハードコード | YAML定義へ外出し、Loaderで検証して読み込み |
| 結果出力 | summary/saveで辞書アクセスが散在 | `ExperimentResult`（`ztb/experiments/base.py`）を活用し、整形責務を分離 |

### 9.3 目標アーキテクチャ（`ztb`優先）

新規ロジックは `scripts` 側に増やさず、`ztb` 側へ寄せる。

1. `ztb/experiments/phase_c/registry.py`  
   - 実験定義・バッチ定義の管理（読み込み/検証）  
2. `ztb/experiments/phase_c/config_builder.py`  
   - `build_config()` を移管し、副作用を排除  
3. `ztb/experiments/phase_c/evaluator.py`  
   - Eval-A/B・再現性・Multi-DD・risk reset を集約  
4. `ztb/experiments/phase_c/metrics.py`  
   - Gate2計算を純関数として一元化  
5. `ztb/experiments/phase_c/runner.py`  
   - 単発/バッチ実行のオーケストレーション  
6. `ztb/experiments/phase_c/reporting.py`  
   - summary表示と結果保存の整形責務を分離  
7. `ztb/experiments/phase_c/cli.py`  
   - 引数解釈と実行入口  

`scripts/v459/run_phase_c.py` は **薄い互換ラッパー** のみを残す。

### 9.4 移行フェーズ（段階的・後方互換）

| フェーズ | 目的 | 主要作業 | 完了条件 |
|---|---|---|---|
| R0 | 安全網構築 | 現行出力JSONのスナップショット化 | 既存CLI結果の比較基準が確立 |
| R1 | メトリクス重複除去 | Gate2計算を1実装に統合 | 旧実装との差分が許容範囲内 |
| R2 | 評価責務分離 | evaluatorモジュール抽出 | Eval-A/B/Multi-DDの挙動一致 |
| R3 | 実験定義外部化 | YAML化 + `ztb/config/loaders` 活用 | `--single-run/--batch` が互換動作 |
| R4 | 実行器分割 | runner/reporting/cliへ責務分離 | `run_phase_c.py` が薄い入口になる |
| R5 | 仕上げ | 不要関数整理と命名統一 | 主要ファイルの責務が明確化 |

### 9.5 Godオブジェクト回避ルール（実装規約）

1. 1モジュール1責務（設定/評価/実行/出力を分離）。  
2. `Any`辞書の受け渡しを減らし、型付きスキーマで接続。  
3. グローバル可変状態を廃止（timesteps等は引数経由）。  
4. 副作用は境界層（runner/evaluator）に限定し、計算系は純関数化。  
5. 新規機能追加時は既存モジュール拡張を優先し、`run_phase_c.py` へ直書きしない。  

### 9.6 検証計画（回帰防止）

1. `--single-run` で代表3実験（`c0_baseline_p1`, `d2_cost05`, `d2_cost10`）の出力比較。  
2. `--batch d2_cost` の結果構造（必須キー）互換性確認。  
3. Gate2の主要値（ROI/PF/Sharpe/Trades）が旧実装と同等であることを確認。  
4. 実行時間の劣化が許容範囲（目安: +3%以内）であることを確認。  

### 9.7 まず着手すべき最小セット（短期）

1. R1: Gate2計算統合（最も効果が高く低リスク）。  
2. R2: evaluator抽出（再現性問題の修正基盤になる）。  
3. R3: 実験定義YAML化（`ztb/config/loaders` 経由、保守コストを即時低減）。  

この3点を完了すれば、重複削減とGodオブジェクト回避の土台が整い、以降の機能改善を安全に進められる。

### 9.8 `/ztb` 既存実装の最大活用マッピング

| 目的 | 既存資産（`ztb`） | `run_phase_c.py` 整理での使い方 |
|---|---|---|
| 実験実行の骨格 | `ztb/experiments/base.py` | `ExperimentBase` / `ExperimentResult` を流用し、結果スキーマ乱立を防止 |
| 環境メトリクス抽出 | `ztb/utils/env_metrics.py` | 既存の `resolve_env/unwrap_env/extract_*` を維持し、独自抽出を増やさない |
| Gate評価枠組み | `ztb/training/evaluation/eval_gates.py` | Gate2判定を `EvalGates` スタイルに統合（閾値管理を集約） |
| KPI計算 | `ztb/metrics/metrics.py` | Sharpe/PF/MaxDD/WinRateを既存関数に統一 |
| 結果I/O | `ztb/io/json_io.py` | 保存ロジックを標準I/Oへ寄せ、JSON出力の実装重複を削減 |
| Walk-Forward/OOS | `ztb/evaluation/walk_forward/*` | 50K通過後の最終判定を既存評価器へ接続 |
| 実行モデル | `ztb/trading/execution/realistic.py`, `ztb/trading/execution/pseudo_hft.py` | コスト/約定感度評価を追加する際の基盤として流用 |
| 閾値管理 | `ztb/trading/environment/components/threshold_manager.py` | 固定閾値実験から動的閾値実験へ段階移行する際の中核 |

### 9.9 `ztb`活用時の注意点（先に明示）

1. `ztb/analysis/evaluation/unified_evaluation.py` は deprecated shim のため、新規依存を増やさない。  
2. `ztb/ops/pipelines/training.py` / `ztb/ops/pipelines/evaluation_pipeline.py` は現状TODOであり、直ちに主経路へは採用しない。  
3. 既存 `ztb/experiments/*` は粒度差が大きいため、Phase C向けの薄いサブパッケージを追加して統合する。  
4. `scripts/v459/run_phase_c.py` 直書きで機能追加しない運用ルールを徹底する。  
