# 359# ph3 G2 gate メトリクス修正 + セルフレビュー品質改善

> **358# セルフレビュー指摘対応 + L-3/L-5 G2 gate パイプライン完成**
> 作成日: 2026-03-09

---

## §1 概要

358# (B1/B3/B4 ブロッカー解消) のセルフレビューを実施し、5 件の CRITICAL・2 件の SIGNIFICANT・2 件の LOW を修正。
加えて、G2 gate E1/E3 判定を実質的に無効化していたメトリクス欠損 (L-3/L-5) を解消し、
G2 SAC 訓練 → gate 判定パイプラインを完全に機能する状態にした。

---

## §2 セルフレビュー修正 (commit `0f5c24dc1`)

### 2.1 CRITICAL (5 件)

| ID | 内容 | 修正 |
|---|---|---|
| **C-1** | `run_experiment.py` に dead import `run_g2_judgment` | 削除 (ImportError リスク排除) |
| **C-2** | `_run_multi_seed` — 1 seed 例外で全結果消失 | try/except + partial result 保持 |
| **C-3** | `g2_sac_train.yaml` に `356a#` 参照 3 箇所残存 | `356#` に統一 |
| **C-4** | `test_356_g2_sac_blockers.py` に `356a#` 参照 2 箇所残存 | `356#` に統一 |
| **C-5** | `358_ph3_impl_g2_sac_blockers.md` に `356a#` 参照 1 箇所残存 | `356#` に統一 |

### 2.2 SIGNIFICANT (4 件)

| ID | 内容 | 修正 |
|---|---|---|
| **S-1** | `task_fn: object` → `Callable[[dict], dict]` | 型安全向上、`cast` 削除 |
| **S-2** | `env_info["obs_dim"]` int\|str 型で `range()` に渡す | `env_info` 型を `dict[str, int\|str\|bool]` に修正、`int(env_info["obs_dim"])` で明示的変換 |
| **S-3** | `sac_cfg` shadowing (dict コピーでパラメータ名上書き) | `sac_params` にリネームで shadowing 解消 |
| **S-4** | `_compute_convergence` 返り値型 `dict[str, int]` | `dict[str, int \| float]` に修正 |

### 2.3 LOW (2 件)

| ID | 内容 | 修正 |
|---|---|---|
| **L-1** | `_run_multi_seed` 内 dead import `stdev` | 削除 |
| **L-2** | `_run_multi_seed` 内 dead import `Callable` | 削除 |

### 2.4 テスト追加

- `test_seed_failure_captured_not_propagated`: 1 seed 失敗時に他 seed 結果が保持されることを検証

---

## §3 L-3: チェックポイント ROI 記録 (E3 convergence 有効化)

### 3.1 問題

`_train_with_checkpoints()` のチェックポイントメトリクスに `timesteps` しか記録されておらず、
`_compute_convergence()` が参照する `roi` フィールドが常に欠損していた。
結果として E3 convergence 判定 (30K 以降 ROI 変動 ≤ 5%) が常に 0.0 = PASS となり、実質無効化。

### 3.2 修正

| 関数 | 変更 |
|---|---|
| `_train_with_checkpoints()` | `env` パラメータ追加。各チェックポイントで `_checkpoint_eval_roi()` を呼び出し ROI を記録 |
| `_checkpoint_eval_roi()` | **新規**。1-episode deterministic eval を実行し環境から ROI を算出 |
| `_extract_roi_from_env()` | **新規**。`portfolio_value / initial_portfolio_value` から ROI を duck-typing で取得 |
| `task_sac_train()` | `_train_with_checkpoints` 呼び出しに `env` を追加 |

### 3.3 設計判断

- **別環境ではなく訓練環境を直接使用**: チェックポイント eval で env.reset() → rollout を行う。
  SB3 の DummyVecEnv は同一環境オブジェクトを参照しており、次の `model.learn()` で自動リセットされるため安全。
- **duck-typing 採用**: `TrainingEnvProtocol` には portfolio 属性を含めず、
  `getattr` で HeavyTradingEnv 固有属性を安全に取得。後方互換性を維持。

---

## §4 L-5: 評価メトリクス gross_roi 追加 (E1/E4 有効化)

### 4.1 問題

`_evaluate_trained_model()` の返り値に `gross_roi` が含まれず、
`_run_multi_seed()` でのフォールバック値 `mean_reward` が使用されていた。
`mean_reward` は報酬合計の平均であり ROI (リターン率) とは異なるため、
E1 (positive_seed_ratio) と E4 (worst_seed_roi) の判定が不正確だった。

### 4.2 修正

`_evaluate_trained_model()` の返り値に以下を追加:

| フィールド | 値 | 用途 |
|---|---|---|
| `gross_roi` | `(portfolio_value - initial) / initial` | E1 positive_seed_ratio, E4 worst_seed_roi |
| `trade_count` | `env.trades_count` | 取引回数 (可観測性) |
| `gross_pnl` | `env.total_pnl` | PnL 絶対値 (ログ・分析用) |

旧 `get_metrics()` フォールバックパターンを `_extract_roi_from_env()` + 直接属性取得に置換。

---

## §5 影響範囲

| E チェック | 修正前 | 修正後 |
|---|---|---|
| **E1** positive_seed_ratio | `mean_reward` ベースで判定 (不正確) | `gross_roi` ベースで判定 (正確) |
| **E2** ic_seed_std | 変更なし (IC は RL では 0.0 — 将来対応) | 同左 |
| **E3** convergence | ROI 値なし → 常に 0.0 = PASS (無効) | チェックポイント ROI で正しく判定 |
| **E4** worst_seed_roi | `mean_reward` ベースで判定 (不正確) | `gross_roi` ベースで判定 (正確) |

---

## §6 テスト

### 6.1 新規テスト

| テスト | 検証内容 |
|---|---|
| `test_extract_roi_from_env_with_portfolio` | `_extract_roi_from_env` が ROI を正しく算出 |
| `test_extract_roi_from_env_missing_attrs` | 属性欠損時に 0.0 フォールバック |
| `test_extract_roi_from_env_zero_initial` | initial=0 で ZeroDivision 防止 |
| `test_extract_roi_negative` | 損失ケースで負の ROI |
| `test_checkpoint_metrics_contain_roi` | チェックポイントに `roi` フィールドが存在 |
| `test_eval_metrics_contain_gross_roi` | 評価メトリクスに `gross_roi` フィールドが存在 |
| `test_yaml_data_file_exists_and_valid` | P3A-1: YAML 参照データが有効な Parquet |
| `test_yaml_features_present_in_data` | P3A-1: 12 特徴量がデータに存在 |
| `test_data_has_close_column` | P3A-1: HeavyTradingEnv 必須の close カラム |

### 6.2 既存テスト

- `TestConvergenceComputation` (3 件): ROI 値を使用するテストが既に存在 → そのまま PASS
- `TestB4G2GateEvaluation` (7 件): seed_results に `gross_roi` を設定済み → そのまま PASS

---

## §7 残課題

| ID | 内容 | 優先度 | 次ステップ |
|---|---|---|---|
| E2-IC | RL での IC 定義未確定 (現在は 0.0) | MED | SAC 訓練本番前に要検討 |
| 065a/065b | index.md の既存アルファベット枝番 | LOW | 空き番号への再割当 |
| 336# | index.md に 6 行同番号 | LOW | 次回整理 |
| P3A-2 | HeavyTradingEnv 統合テスト | MED | 実データが確保できたので実施可能 |

---

## §8 P3A-1: 実データパイプライン整備

### 8.1 問題

`g2_sac_train.yaml` が参照する `data/btc_jpy_1m_v451_optimized_features.parquet` は
133 バイトの stub ファイル (Parquet magic bytes なし) であり、訓練実行不能だった。

### 8.2 修正

YAML `data.ohlcv_path` を有効な実データに変更:

| 項目 | 旧 | 新 |
|---|---|---|
| ohlcv_path | `btc_jpy_1m_v451_optimized_features.parquet` (133B stub) | `btc_jpy_1m_full_registry_features.parquet` (143MB, 77列) |
| 行数 | N/A | 1,216,930 |
| NaN 率 | N/A | 0.0000 (12 特徴量に対して) |
| 12 特徴量 | 未検証 | ✅ 全存在確認済 |
| close カラム | 未検証 | ✅ 存在確認済 |

### 8.3 テスト

`TestTrainingDataIntegrity` (3 件) でデータファイルの整合性を自動検証:
- Parquet ファイル有効性
- 12 特徴量がデータに存在
- `close` カラム存在

---

## §9 変更ファイル一覧

| ファイル | 種別 | 内容 |
|---|---|---|
| `scripts/v460/lib/tasks/sac_train.py` | MOD | L-3: checkpoint ROI, L-5: eval gross_roi, S-2: 型修正, S-3: sac_params リネーム |
| `scripts/v460/run_experiment.py` | MOD | (前回 commit) C-1~C-2, S-1, S-4, L-1/L-2 |
| `configs/v460/experiments/g2_sac_train.yaml` | MOD | P3A-1: ohlcv_path を有効な full_registry に変更 |
| `tests/unit/v460/test_356_g2_sac_blockers.py` | MOD | 新規テスト 10 件追加 (ROI 4 + eval 2 + data 3 + seed 1) |
| `docs/v460/359_ph3_impl_g2_gate_metrics_fix.md` | NEW | 本ドキュメント |
| `docs/v460/index.md` | MOD | 359# エントリ追加 |
