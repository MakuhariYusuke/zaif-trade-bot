# SAC v448 Layer 6 — MTF Weight Optimizer（設計仕様）

## 概要
Layer 6 は Multi-Timeframe (MTF) の重み付け最適化を自動化する機能群です。Layer 5 の基礎実装 (MTFWeightManager, curriculum integration) をもとに、候補生成・オフライン評価・選定・CI gating を通じて、実際に運用可能な MTF 重みを探索するためのエンジンを提供します。

## 目的
- MTF 重みを最適化し、1分足学習におけるバイアスや過剰トレードを抑制しつつ収益と安定性を向上させる。
- 自動化された評価により、再現性と持続性を確保する。
- CI とナイトリーのワークフローで自動選抜し、良好な候補を `best_model/` に保存する。

## スコープ（MVP）
1. `MTFOptimizer` クラス（候補生成・スコアリング）
2. `CandidateEvaluator`（ab_test_runner をラップして候補を評価）
3. CI ワークフローの拡張：`training-eval` の nightly ジョブで optimizer を実行して結果を集計
4. 成果物： `reports/ab_summary.json`、`best_model/` の候補保存、`experiments/` に結果記録

## 成功基準
- ローカル：MTFOptimizer のユニットテスト（候補生成・スコア集計）に合格。
- CI：training-eval ジョブにて 3 seeds × 4k steps の AB-run 実行と評価が成功し、少なくとも 1 つの candidate が所定のゲート（たとえば Sharpe>=0.5 かつ total_return>=0.05）を満たす。
- 実験：Nightly の optimizer run が 3 night 連続で同様の候補を返す（再現性チェック）。

## コンポーネント詳細
### 1) MTFOptimizer
  - propose_candidates(n: int) -> List[CandidateConfig]
  - evaluate_candidate(cfg: CandidateConfig) -> CandidateScore
  - run_optimization(iterations: int, population: int)
 - `MTFOptimizer.run()` returns a tuple `(CandidateConfig, CandidateScore)` (best candidate & its score). Consumers such as `MTFScheduler` expect a tuple and will apply the best candidate from the first element.

### 2) CandidateEvaluator
- 使用: `tools/ab_test_runner.py` を透過的に呼び出し、各 candidate について multi-seed 結果を収集
- 出力: `CandidateScore`:
  - mean_sharpe, mean_total_return, max_drawdown, balance_stability
  - composite_score (default: sustainable_profitability_score)

  - Implementation Notes & Requirements:
  - CandidateEvaluator should:
    - Validate `training.model_name` exists in the candidate config and fail fast with a helpful error if missing.
    - Support retries with exponential backoff (configurable) and cleanup of partial report artifacts produced by failed runs.
    - Parse `reports/training_report_*.json` for the candidate's `model_name` to compute multi-seed statistics.
    - Return clear, typed metrics (mean_sharpe, mean_total_return, composite_score) and diagnostic values such as `report_count` and optionally store `run_artifacts` (report paths) for debugging.
    - `report_count` semantics & usage:
      - `report_count` is the number of reports found for the candidate's `training.model_name` in `reports/` (typically should match `seeds`).
      - `report_count < seeds` indicates partial failures/timeouts; prefer to re-run the candidate or treat it as a non-pass unless below a configured tolerance.
      - Gating: for short-listing, require `report_count >= seeds` or a specified threshold (e.g., `report_count >= seeds - 1`). Only accept metrics for candidates meeting the `report_count` threshold.
      - `report_count` = 0 implies no reports matched the candidate; returnable metrics in this case will be zeros and the candidate should typically be discarded or retried.
    - For reliability, CandidateEvaluator should be robust to partial reports, missing fields, and postmortem cleanup.


### 3) Integration Points
- `MTFWeightManager` に apply() メソッドを提供して Candidate を実行時に反映
 - `MTFWeightManager.set_weights(weights: Dict[str,float]) -> bool` を用意し、`weights` に `_candidate_id` の optional key を含めて呼ぶことで、適用結果（成功/失敗）と適用された candidate のIDを telemetry に記録します。
- `RewardCalculator` / `BehavioralPenaltyCalculator` に候補関連 telemetry を出力
 - `BalanceCurriculumManager` の stage-change listener を利用して、MTFScheduler の最適化/適用を実行できるようにする
   - 例: `bcm.add_stage_change_listener(mtf_scheduler.create_stage_change_callback(stage_filter=["balanced_transition"]))`
  - 実装補足: `HeavyTradingEnv` は `behavior.mtf.weight_optimizer.enabled` / `mtf_optimizer.enabled` が `true` の場合、自動的に `MTFScheduler` を初期化して `RewardCalculator.curriculum_manager` に callback を登録します (設定に `base_config`, `dry_run`, `stage_filter` を含めると上書きできます)。

**Note**: For MVP the optimizer writes candidate config files and `ab_test_runner` runs them for evaluation. A runtime `apply_candidate_to_manager` helper is provided that can be used to apply the winning candidate's weights into an `MTFWeightManager` instance at runtime. This permits online A/B testing in the environment.

### 4) CI や Gating
- Nightly job: MTF optimizer を実行して上位の候補を評価・保存
- Gate Conditions: candidate の composite_score >= threshold
- Artifacts: `reports/`、`logs/`、`best_model/` を保管
  - Gate script: `tools/ci/check_optimizer_gates.py` evaluates `reports/ab_summary.json` or `reports/mtf_optimizer_summary.json` against configured thresholds (`--sharpe`, `--return`) and fails the job if no candidate meets gates.

  Two-Stage Candidate Adoption Pattern (recommended):
  1. Quick prefilter: Run `MTFOptimizer` with `dry_run=True` or low-timesteps (short seeds), evaluate candidate viability by `CandidateEvaluator` quick mode and a conservative gate (sharpe >= X, return >= Y). Keep top-N.
    2. Long-run verification: For the top-N shortlisted candidates, run a longer AB-run (higher timesteps) using `ab_test_runner` and re-evaluate metrics using `CandidateEvaluator`.
      - During the long-run verification, consider requiring `report_count >= seeds` (or a slightly relaxed threshold) before accepting the candidate's metrics. This prevents falsely high mean metrics due to a single successful seed while others failed/time-out.
  3. If a candidate passes the long-run gate (AND condition on `sharpe` and `total_return`), apply it using `apply_candidate_to_manager()` and log candidate id/time in telemetry. Avoid auto-apply without a long-run check in production.
     - When applying a candidate via `apply_candidate_to_manager()` verify that `report_count` and `composite_score` match expectations for the candidate. If a candidate has low `report_count`, prefer to re-run the verification or prevent apply.
  - Tooling suggestion: use `tools/training/confirm_candidate.py` to orchestrate prefilter -> longer-run verification -> apply flows in CI/nightly use cases.

  This reduces risk of overfitting to short-run noise and avoids deploying unstable candidates to running environments.

## 実装メモ & 改善提案 (発見事項)

- `MTFOptimizer.propose_candidates()` は `base_config` の `multi_timeframe.feature_weights` を直接読み書きします。実行時に `multi_timeframe` が欠落していると `KeyError` になります。運用安全のため、`MTFOptimizer._load_base_config()` 後に `multi_timeframe` と `feature_weights` の存在を検証し、適切なエラーメッセージを返すか、フォールバックデフォルトを用意してください。

- `HeavyTradingEnv` で `MTFScheduler` を自動生成するパスは `mtf_optimizer` と `behavior.mtf.weight_optimizer` の両方をサポートします。設定が dict か attribute オブジェクトかの両方に対応するための `_read_val()` ヘルパーを使う実装が入っています。

- `BalanceCurriculumManager._revert_to_forced_balance()` 内の `previous_stage` 参照は未定義の可能性があり、イベント通知の `previous_stage` 値を正しく渡すための修正/確認が必要です（現在は `previous_stage` という変数はこのメソッド内で定義されていません）。これを修正することで、ステージ復帰時のリスナー通知に正しい `previous_stage` を付与できるようになります。

- テスト中に、環境用の DataFrame を軽量モック (`FakeDataFrame`) することで pandas を要求せずに `HeavyTradingEnv` のユニットテストが走ることが確認されました。CI や unit-tests では heavy imports を回避するため、同様のスタブ/モック・パターンを推奨します。

- 並列性の注意: `MTFScheduler` が `BalanceCurriculumManager` の callback として動くと、トレーニング中に `MTFWeightManager.set_weights()` を実行する可能性があります。`MTFWeightManager` で `set_weights()` を実行する際は、atomic update（ロックや copy-on-write）を考慮してください。`set_weights()` は成功/失敗のブールを返すと良いです。

- Telemetry 提案: `MTFScheduler`、`MTFOptimizer`、`CandidateEvaluator` が candidate を選び適用するたび、`RewardCalculator.last_reward_components` または `MTFWeightManager` telemetry に次の情報を出力することを推奨します:
  - `applied_candidate_id`
  - `applied_weights`
  - `applied_at` (timestamp)
  - `composite_score` (if available)
 これにより CI レポートと runtime の調査が簡単になります。

- Candidate ファイル名と `training.model_name` を `candidate_id` 固有の命名規則に従わせるべきです。テストでは `sac_v448_mtf_candidate_candidate_0` のように一意にすることで、`ab_test_runner` の生成レポートと candidate ファイルを確実に紐づけられることを確認しました。

- `CandidateEvaluator` の堅牢性: `ab_test_runner` の失敗や部分的なレポートに備えて、`CandidateEvaluator` 側でリトライ（最大N 回）と、レポート名の一貫性チェック、部分的なレポートのクリーンアップ（失敗時の artifact removal）を行うべきです。

- CI の gating を行う `tools/ci/check_optimizer_gates.py` では、`composite_score` の導入により gate 閾値を評価していますが、より保守的に `sharpe >= X` と `total_return >= Y` の AND 条件にしておくことを推奨します（shortlist を複数で絞り、その後再評価する flows を作る）。

- 自動適用ポリシー: 最良候補を即時に適用するのではなく、まず dry-run での解析（3 seeds × short timesteps）→ 合格者を LONGER AB-run (より長い timesteps) で再評価→ それでも合格した場合のみ `mtf_scheduler.apply` を実行する保険的フローを推奨します。


## 設計方針
- 保守性: 初期実装は単純かつ説明可能であること（ランダム/グリッド）。
- 再現性: すべての run は seed と timesteps を明示してログに残す。
- CI: CPU-only Torch を使用して、Windows DLL 依存性に左右されない環境で実行する。

## 実装ステップ（MVP）
1. 文書・テンプレート作成（本書） ✅
2. `ztb/training/reward_function_optimizer/mtf_optimizer.py` の骨格を作る
3. `ztb/training/reward_function_optimizer/candidate_evaluator.py` を作る
4. CI job を `training-eval` に追加して nightly 実行（現在の job を拡張）
5. `tools/ci/evaluate_training_runs.py` を拡張して候補ごとのランキングを作る
6. 単体テスト/統合テストを作成してパイプラインを検証

### 優先実行シーケンス（推奨）
優先度と進め方を考慮した実行順序を下記に示します。実装は段階的に行い、各段階でユニットテスト・統合テスト・CI dry-run を通して安定性を確認してください。

1. CandidateEvaluator の安定化 (1-2日)
  - Retries, report_dir, cleanup, per-candidate model_name 確保
  - Unit tests (dry-run & partial report aggregation) を追加
2. MTFOptimizer MVP の統合 (1-2日)
  - Candidate generation の改善（現在はランダム）
  - Evaluate step で CandidateEvaluator を利用
  - CandidateConfig の `training.model_name` を `candidate_id` に組み込む
  - Unit tests (run dry-run & propose candidate normalization)
3. Scheduler の作成（1日）
  - `MTFScheduler` を作成し、最良 candidate の定期適用を実装
  - Trainer/RewardCalculator 内にフックするための簡易インターフェースを提供
4. CI の dry-run と gating（1-2日）
  - Nightly で `mtf_optimizer` dry-run を動作確認 (CI smoke)
  - Gate 条件として composite_score, sharpe, total_return を導入
5. Curriculum および Runtime Integration（2-4日）
  - `BalanceCurriculumManager` と `MTFWeightManager` を連携させるフックを追加
  - 学習段階（stage）に応じた自動切替や試験的な auto-promote を実装
6. 最適化アルゴリズムの拡張（研究/改善）
  - Bayesian 最適化・CMA-ES・PBT などの導入と効果検証


## 推奨実行コマンド（テスト用）
- Quick test (ローカル):
```bash
python -m ztb.training.reward_function_optimizer.mtf_optimizer \
  --config config/v448/mtf_optimizer_template.json \
  --seeds 3 --timesteps 2000 --candidates 10 --iterations 3
```

- Nightly CI (例):
  - `training-eval` ジョブ内で:
```bash
python tools/ab_test_runner.py --configs config/v448/mtf_candidate_*.json --seeds 3 --timesteps 4000 --jobs 3
python tools/ci/evaluate_training_runs.py --out reports/mtf_optimizer_summary.json
```

## 設定テンプレート
- 参考として `config/v448/templates/mtf_optimizer_template.json` を作成し、候補構成を明示する。

## テスト戦略
1. 単体: CandidateConfig 検証、score aggregation
2. 統合: AB-run で candidate を評価し、`tools/ci/evaluate_training_runs.py` が正しい JSON を作成
3. CI: Nightly run で gating に基づく合格/不合格判断を自動化

## 将来的な拡張
- Bayesian Optimizer、PBT（Population Based Training）、AutoML の導入
- オフラインデータの活用（バックテスト）により候補を prefilter
- ランタイム（オンライン）での軽量最適化（少数回の更新）

---

ドキュメント作成: `SAC_v448_LAYER6_DESIGN_SPEC.md` を更新していきます。
