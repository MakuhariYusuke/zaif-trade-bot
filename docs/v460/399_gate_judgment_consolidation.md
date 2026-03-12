# 399# G2/G3 judgment 統合 + 本番実験セッション

## 概要

G2/G3 ゲート判定ロジックの重複を統合し、コード品質改善を実施。
統合後のパイプラインでドライラン検証 → 本番 SAC 実験を実行。

## 変更一覧

### 1. G2/G3 判定ロジック統合 (gate_judgment_core.py)

**問題**: `run_experiment.py` と `run_gate_check.py` に ~320行の G2/G3 判定ロジックが重複。
さらに `run_gate_check.py` には 384# HIGH-1 の `error_seeds` チェックが欠落していた。

**解決**:
- `scripts/v460/lib/gate_judgment_core.py` を新規作成（~210行）
  - `evaluate_g2_checks(seed_results, convergence, thresholds)` → 4チェック (E1-E4) + error_seeds
  - `evaluate_g3_checks(seed_metrics, thresholds)` → 5チェック (E1-E5)
- `run_experiment.py`: ~160行削除、core からの import に置換
- `run_gate_check.py`: G2/G3 関数を thin wrapper に縮小

**バグ修正**: `run_gate_check.py` の error_seeds 未検出が修正された。

### 2. sac_train.py 改善

- `SACTrainModelProtocol` エイリアス削除（5箇所 → `SACModelProtocol` 直接使用）
- `_save_model_schema`: 特徴量解決の重複を `_resolve_feature_columns(cfg)` 呼出に統合
- `_build_environment_config`: `data.train_end_index` を `EnvironmentConfig` に注入
  （scaler リーク防止、将来の安全性確保）

### 3. テスト更新

- `test_356_g2_sac_blockers.py`: import 先を `gate_judgment_core` に変更
  - `test_empty_seed_results`: 期待値を "FAIL" → "NO_DATA" に修正（空=データ不足≠失敗）
- `test_396_g3_pipeline.py`: import 先を `gate_judgment_core` に変更

### 4. 本番実験

- **構成**: `g2_sac_gamma095_reward_tuned_fast.yaml`
  - γ=0.95, 20K steps × 4 seeds, val_ratio=0.02
  - 報酬チューニング: balance_penalty=0.1, hold_penalty=0.001, consistency=0.01
  - 所要時間: ~60分 (seed毎に ~15分)

#### 結果

| Seed | ROI | PF | Sharpe (年率) | MaxDD | reward-PnL相関 |
|------|-----|-----|-------|-------|---------|
| 42 | -0.31% | 0.955 | -2.85 | 0.50% | -0.14 |
| 123 | +0.29% | 1.050 | +2.82 | 0.21% | +0.26 |
| 456 | +0.03% | 1.012 | +0.61 | 0.37% | +0.13 |
| 789 | +0.001% | 1.000 | -0.18 | 0.33% | -0.38 |

- **G2: PASS** (4/4 checks)
  - positive_ratio=0.75 (3/4 seeds profitable)
  - roi_std=0.0025, convergence=0.0 (20K→30K以降データなし)
  - worst_roi=-0.31% > -3.5%
- **G3: FAIL** (3/5 checks)
  - PF median=1.006 < 1.05 (FAIL)
  - Sharpe median=0.216 < 0.8 (FAIL)
  - PF worst=0.955 > 0.95, MaxDD=0.5% < 15%, gross>fee (PASS)

#### 考察

1. 20K steps では学習量が不足。PF=1.006 は break-even レベル
2. reward-PnL 相関が seed 789 で -0.38 — 報酬関数がまだ不十分
3. MaxDD < 1% は全 seed で良好 — リスク制御は機能している
4. G2 PASS は確認: 訓練の安定性・再現性は問題なし
5. 次ステップ: 50K-100K steps で本格実験、または報酬関数の再検討

## ドライラン結果 (1K steps × 2 seeds, val_ratio=0.02)

| Seed | ROI | PF | Sharpe | MaxDD |
|------|-----|----|--------|-------|
| 42 | +0.32% | 1.074 | 2.44 | 0.63% |
| 123 | +0.57% | 1.102 | 4.71 | 0.53% |

- G2: **PASS** (E1 positive_ratio=1.0≥0.75, E2 std=0.0018≤0.03, E3 conv=0.0≤5.0, E4 worst=0.0032>-0.035)
- G3: **PASS** (E1 PF_med=1.088>1.05, E2 PF_worst=1.074>0.95, E3 gross>fee, E4 maxDD=0.63%<15%, E5 Sharpe=3.58>0.8)

## 調査した改善点

| 優先度 | 問題 | 影響 | 状態 |
|--------|------|------|------|
| P1 | `train_end_index` が env に未伝達 | 現行無害（split済みDF）、将来リスク | ✅ 修正済み |
| P2 | `_build_fast_access_buffers` 重複呼出 | パフォーマンスのみ | 後回し |
| P3 | Non-finite values (1行) | 既に対処済み | パイプライン側で対応 |
| P2 | YAML 6ファイルの共通設定重複 | 保守性 | 後回し |

## コミット

- `2c75aced1`: 399# G2/G3 judgment logic consolidation + sac_train improvements
