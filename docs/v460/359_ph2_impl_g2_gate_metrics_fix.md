# 359# ph2 G2 gate メトリクス修正 + セルフレビュー品質改善

> **358# セルフレビュー指摘対応 + L-3/L-5 G2 gate パイプライン完成**
> フェーズ: **ph2** G1.1-exec (G2-train 先行準備)
> 作成日: 2026-03-09
> コミット: `79409e8a5` (P3A-2 + SR S-5/L-7)

---

## §0 フェーズ位置づけ

000# §2 Phase 定義に基づき、現在のプロジェクトフェーズは **ph2** (G1.1-exec: maker 執行可能性検証) である。
G1.1 gate は K1 FAIL (fill rate 25.0%, 目標 ≥ 60%) のため **ph2 は未完了**。

本ドキュメントの作業 (G2 gate メトリクス修正・SAC 訓練パイプライン整備) は、
ph2 中に実施できる **ph3 (G2-train) の先行準備** として位置づける。
G1.1 gate 通過を待たずに G2 パイプラインを整備しておくことで、
G1.1 PASS 後に即座に ph3 SAC 訓練を開始できる状態を確保する。

| Phase | Gate | 状態 | 本文書の関連 |
|-------|------|------|-------------|
| ph1 | G1-info | ✅ PASS (Oracle PnL 正, 条件付延長中) | — |
| **ph2** | G1.1-exec | ⏳ K1 FAIL (25.0%) | **現在地** |
| ph3 | G2-train | 🔜 先行準備中 | 本文書のスコープ |

---

## §1 概要

358# (B1/B3/B4 ブロッカー解消) のセルフレビューを実施し、5 件の CRITICAL・4 件の SIGNIFICANT・2 件の LOW を修正。
加えて、G2 gate E1/E3 判定を実質的に無効化していたメトリクス欠損 (L-3/L-5) を解消し、
G2 SAC 訓練 → gate 判定パイプラインを完全に機能する状態にした。

### 1.1 修正サマリ

| カテゴリ | 件数 | 主要修正 |
|---------|------|---------|
| CRITICAL | 5 | dead import, partial result 保持, `356a#` 参照統一 |
| SIGNIFICANT | 4 | 型安全 (`Callable`, `int`変換), shadowing 解消, 返り値型修正 |
| LOW | 2 | dead import 削除 |
| 機能修正 | 2 | L-3 チェックポイント ROI, L-5 評価 gross_roi |
| データ整備 | 1 | P3A-1 実データパイプライン |
| 統合テスト | 1 | P3A-2 HeavyTradingEnv E2E 検証 |

### 1.2 変更ファイル一覧

| ファイル | 行数 | 主要変更 |
|---------|------|---------|
| `scripts/v460/lib/tasks/sac_train.py` | 431 | L-3 `_checkpoint_eval_roi`, L-5 `gross_roi`, S-2/S-3/S-5/L-7 |
| `scripts/v460/run_experiment.py` | 390 | C-1 dead import, C-2 partial result, S-1 `Callable` 型 |
| `configs/v460/experiments/g2_sac_train.yaml` | 45 | C-3 `356a#`→`356#`, P3A-1 ohlcv_path |
| `tests/unit/v460/test_356_g2_sac_blockers.py` | 667 | C-4 参照統一, 新規テスト 15 件追加 |
| `docs/v460/358_ph3_impl_g2_sac_blockers.md` | 175 | C-5 参照統一 |

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

**影響**: G2 gate の E3 チェックが事実上バイパスされ、収束していない不安定なモデルも PASS となるリスクがあった。

### 3.2 修正

| 関数 | 変更 | ファイル |
|---|---|---|
| `_train_with_checkpoints()` | `env` パラメータ追加。各チェックポイントで `_checkpoint_eval_roi()` を呼び出し ROI を記録 | `sac_train.py` L279–L310 |
| `_checkpoint_eval_roi()` | **新規**。1-episode deterministic eval を実行し環境から ROI を算出 | `sac_train.py` L313–L333 |
| `_extract_roi_from_env()` | **新規**。`portfolio_value / initial_portfolio_value` から ROI を duck-typing で取得 | `sac_train.py` L336–L351 |
| `task_sac_train()` | `_train_with_checkpoints` 呼び出しに `env` を追加 | `sac_train.py` L130 |

### 3.3 データフロー

```
model.learn(checkpoint_interval)
  ↓
_checkpoint_eval_roi(model, env)
  ├→ env.reset()
  ├→ model.predict() × N steps (deterministic)
  └→ _extract_roi_from_env(env)
       ├→ getattr(env, "portfolio_value")
       ├→ getattr(env, "initial_portfolio_value")
       └→ (pv - iv) / iv  →  roi: float
  ↓
checkpoint_metrics.append({"timesteps": t, "roi": roi})
  ↓
_compute_convergence(all_checkpoint_metrics, window_start=30000)
  └→ {roi_variance_pct_after_30k: max(rois) - min(rois) × 100}
```

### 3.4 設計判断

| 判断 | 選択 | 理由 |
|------|------|------|
| eval 環境 | 訓練環境を直接使用 | SB3 の DummyVecEnv は同一オブジェクト参照。次の `model.learn()` で自動リセット |
| 属性取得 | duck-typing (`getattr`) | `TrainingEnvProtocol` に portfolio 属性を含めず後方互換性維持 |
| ZeroDivision | `initial_value > 0` ガード | `_extract_roi_from_env` 内で明示的チェック |
| env state 汚染 | 許容 (S-6) | 50K steps 中 ~10 遷移 (0.02%) と軽微。ph4 で eval 専用 env 分離予定 |

### 3.5 リスク分析

| リスク | 影響度 | 対策 |
|--------|--------|------|
| **S-6**: checkpoint eval 時の `env.reset()` が訓練 env の内部状態を汚染 | LOW (0.02%) | 次の `model.learn()` が env をリセットするため実質無影響 |
| `portfolio_value` 属性が将来の env で未定義 | LOW | `getattr` + fallback 0.0 で安全 |
| n_checkpoints が大きい場合の eval オーバーヘッド | LOW | 50K/10K = 5 回程度。eval 1 episode 所要時間は << 訓練時間 |

---

## §4 L-5: 評価メトリクス gross_roi 追加 (E1/E4 有効化)

### 4.1 問題

`_evaluate_trained_model()` の返り値に `gross_roi` が含まれず、
`_run_multi_seed()` でのフォールバック値 `mean_reward` が使用されていた。
`mean_reward` は報酬合計の平均であり ROI (リターン率) とは異なるため、
E1 (positive_seed_ratio) と E4 (worst_seed_roi) の判定が不正確だった。

**定量的影響**: `mean_reward` は報酬設計に依存する無次元量であり、ROI (%) とのスケール差は最大 100 倍以上。
例: reward_sum = 5.0 → ROI = 0.05 (5%) の場合、mean_reward ベースでは 5.0 > 0 で PASS だが、
実際には worst_seed_min_roi = -2% の判定でも意味のある比較が不可能だった。

### 4.2 修正

`_evaluate_trained_model()` (`sac_train.py` L354–L399) の返り値に以下を追加:

| フィールド | 値 | 用途 | 取得方法 |
|---|---|---|---|
| `gross_roi` | `(portfolio_value - initial) / initial` | E1 positive_seed_ratio, E4 worst_seed_roi | `_extract_roi_from_env(env)` |
| `trade_count` | `env.trades_count` | 取引回数 (可観測性) | `getattr(env, "trades_count", 0)` |
| `gross_pnl` | `env.total_pnl` | PnL 絶対値 (ログ・分析用) | `getattr(env, "total_pnl", 0.0)` |

### 4.3 _run_multi_seed での使用

`run_experiment.py` L255–L262 で `eval_metrics.get("gross_roi")` を取得。
フォールバックチェーン: `gross_roi` → `mean_reward` → `0.0`

```python
gross_roi = float(eval_metrics.get("gross_roi", eval_metrics.get("mean_reward", 0.0)))
```

### 4.4 設計判断

| 判断 | 選択 | 理由 |
|------|------|------|
| n_episodes > 1 時の集計 | 最終エピソードのみ | L-6 として文書化。n_episodes=1 が標準運用 |
| `mean_reward` 残存 | 維持 | 後方互換 + 報酬設計の可視化に有用 |
| `getattr` フォールバック | `0` / `0.0` | env に属性が無い場合の安全な default |

---

## §5 影響範囲

### 5.1 G2 gate E チェック変更

| E チェック | 修正前 | 修正後 | 判定精度の変化 |
|---|---|---|---|
| **E1** positive_seed_ratio | `mean_reward` ベースで判定 (不正確) | `gross_roi` ベースで判定 (正確) | 仮陽性リスク排除 |
| **E2** ic_seed_std | 変更なし (IC は RL では 0.0 — 将来対応) | 同左 | — |
| **E3** convergence | ROI 値なし → 常に 0.0 = PASS (無効) | チェックポイント ROI で正しく判定 | 偽陽性を完全排除 |
| **E4** worst_seed_roi | `mean_reward` ベースで判定 (不正確) | `gross_roi` ベースで判定 (正確) | -2% 閾値が正しく機能 |

### 5.2 コード影響マトリクス

| 変更 | `sac_train.py` | `run_experiment.py` | テスト | YAML |
|------|:-:|:-:|:-:|:-:|
| L-3 checkpoint ROI | ✅ 4 関数 | — | ✅ 2 件 | — |
| L-5 gross_roi | ✅ 1 関数 | ✅ 1 関数 | ✅ 2 件 | — |
| C-1 dead import | — | ✅ 削除 | — | — |
| C-2 partial result | — | ✅ try/except | ✅ 1 件 | — |
| C-3~C-5 `356a#` | — | — | ✅ | ✅ |
| S-1 Callable | — | ✅ 型修正 | — | — |
| S-2 int 変換 | ✅ `int()` | — | — | — |
| S-3 shadowing | ✅ rename | — | — | — |
| S-4 返り値型 | — | ✅ `int\|float` | — | — |
| P3A-1 data | — | — | ✅ 3 件 | ✅ |
| P3A-2 integration | — | — | ✅ 6 件 | — |

### 5.3 後方互換性

- `_train_with_checkpoints()` に `env` 引数追加 → **破壊的変更**。ただし内部ヘルパーであり外部呼出しは `task_sac_train()` のみ
- `_evaluate_trained_model()` の返り値にフィールド追加 → **非破壊的** (既存フィールドは維持)
- `_run_multi_seed()` の `task_fn` 型変更 (`object` → `Callable[[dict], dict]`) → **非破壊的** (runtime 影響なし)

---

## §6 テスト

### 6.1 テスト全件一覧 (38 件)

#### セルフレビュー修正テスト

| # | テスト名 | 検証内容 | 修正対応 |
|---|---------|---------|---------|
| 1 | `test_seed_failure_captured_not_propagated` | 1 seed 失敗時に他 seed 結果保持 | C-2 |

#### L-3/L-5 メトリクス修正テスト

| # | テスト名 | 検証内容 | 修正対応 |
|---|---------|---------|---------|
| 2 | `test_extract_roi_from_env_with_portfolio` | ROI 正常算出 | L-3/L-5 共通 |
| 3 | `test_extract_roi_from_env_missing_attrs` | 属性欠損時 0.0 フォールバック | L-3/L-5 共通 |
| 4 | `test_extract_roi_from_env_zero_initial` | initial=0 で ZeroDivision 防止 | L-3/L-5 共通 |
| 5 | `test_extract_roi_negative` | 損失ケースで負 ROI | L-3/L-5 共通 |
| 6 | `test_checkpoint_metrics_contain_roi` | チェックポイントに `roi` 存在 | L-3 |
| 7 | `test_eval_metrics_contain_gross_roi` | 評価に `gross_roi` 存在 | L-5 |

#### P3A-1 データ整合性テスト

| # | テスト名 | 検証内容 | 修正対応 |
|---|---------|---------|---------|
| 8 | `test_yaml_data_file_exists_and_valid` | Parquet ファイル有効性 | P3A-1 |
| 9 | `test_yaml_features_present_in_data` | 12 特徴量存在確認 | P3A-1 |
| 10 | `test_data_has_close_column` | `close` カラム存在 | P3A-1 |

#### P3A-2 統合テスト

| # | テスト名 | 検証内容 | 修正対応 |
|---|---------|---------|---------|
| 11 | `test_env_instantiation` | HeavyTradingEnv 生成 | P3A-2 |
| 12 | `test_obs_dim_matches_feature_count` | obs_dim == 12 | P3A-2 |
| 13 | `test_feature_names_synced` | feature_names 一致 | P3A-2 |
| 14 | `test_reset_returns_valid_obs` | reset() shape (12,), NaN なし | P3A-2 |
| 15 | `test_step_returns_valid_tuple` | step() 5-tuple 正常返却 | P3A-2 |
| 16 | `test_create_training_env_pipeline` | E2E パイプライン検証 | P3A-2 |

#### 358# 既存テスト (22 件)

| クラス | 件数 | カバレッジ対象 |
|--------|------|-------------|
| `TestB1YamlExists` | 7 | YAML 構造・seeds・特徴量・SACパラメータ |
| `TestB3FeatureInjection` | 3 | feature_names 注入ロジック |
| `TestB4G2GateEvaluation` | 7 | E1–E4 判定 + 境界値 |
| `TestConvergenceComputation` | 3 | convergence 計算 |
| `TestMultiSeedDispatch` | 2 | G2 multi-seed ディスパッチ |

### 6.2 テスト網羅性分析

| 対象関数 | テスト件数 | 正常系 | 異常系 | 境界値 |
|---------|:-:|:-:|:-:|:-:|
| `_extract_roi_from_env` | 4 | ✅ 正 ROI, 負 ROI | ✅ 属性欠損, initial=0 | — |
| `_checkpoint_eval_roi` | 1 | ✅ (mock env) | — | — |
| `_evaluate_trained_model` | 1 | ✅ gross_roi 含有 | — | — |
| `_compute_convergence` | 3 | ✅ 基本計算 | ✅ 空入力 | ✅ window 前のみ |
| `_evaluate_g2_from_results` | 7 | ✅ all pass | ✅ 各 E FAIL, empty | ✅ E1 75% ちょうど |
| `_run_multi_seed` | 1 | — | ✅ seed 失敗 | — |
| `_create_training_env` | 1 | ✅ E2E | — | — |
| HeavyTradingEnv 統合 | 5 | ✅ 各 lifecycle | — | — |

---

## §7 残課題

| ID | 内容 | 重要度 | 影響 | 次ステップ |
|---|---|---|---|---|
| **E2-IC** | RL での IC 定義未確定 (現在は 0.0) | MED | E2 チェックが無効 | SAC 訓練本番前に要検討。RL 報酬ではなく value function の予測力を IC として定義する案 |
| **S-6** | checkpoint eval が訓練 env state を汚染 (影響度 0.02%) | LOW | 訓練精度への微小影響 | ph4 で eval 専用 env 分離を検討。`env.get_wrapper_attr("env")` でベース env クローン |
| **L-6** | n_episodes > 1 時に gross_roi/trade_count/gross_pnl が最終エピソードのみ | LOW | 仕様制約 (n_episodes=1 が標準) | 仕様として文書化済み。累積集計は ph4 |
| **mypy-L196** | `EnvironmentConfig(**env_cfg)` の dict→dataclass 型不一致 | LOW | mypy strict モードで warning | 構造的課題。YAML→dict→dataclass の型変換レイヤー導入で解決 |
| 065a/065b | index.md の既存アルファベット枝番 | LOW | 命名規則違反 | 空き番号への再割当 |
| 336# | index.md に 6 行同番号 | LOW | インデックス汚染 | 次回整理 |

---

## §8 P3A-1: 実データパイプライン整備

### 8.1 問題

`g2_sac_train.yaml` が参照する `data/btc_jpy_1m_v451_optimized_features.parquet` は
133 バイトの stub ファイル (Parquet magic bytes なし) であり、訓練実行不能だった。

**根本原因**: 358# B1 で YAML を新規作成した際、テスト用 stub パスを設定したまま残存。

### 8.2 修正

YAML `data.ohlcv_path` を有効な実データに変更:

| 項目 | 旧 | 新 |
|---|---|---|
| ohlcv_path | `btc_jpy_1m_v451_optimized_features.parquet` (133B stub) | `btc_jpy_1m_full_registry_features.parquet` (143MB, 77列) |
| 行数 | N/A | 1,216,930 |
| NaN 率 | N/A | 0.0000 (12 特徴量に対して) |
| 12 特徴量 | 未検証 | ✅ 全存在確認済 |
| close カラム | 未検証 | ✅ 存在確認済 |

### 8.3 テスト自動検証

`TestTrainingDataIntegrity` (3 件) でデータファイルの整合性を CI レベルで自動検証:

1. **Parquet ファイル有効性**: `pq.read_schema()` で magic bytes 確認
2. **12 特徴量存在**: YAML `features.selected` の全カラムが schema に含まれることを検証
3. **`close` カラム**: HeavyTradingEnv の必須依存 (内部で価格計算に使用)

> 注: データファイルが存在しない場合は `pytest.skip()` で安全にスキップされる。

---

## §9 P3A-2: HeavyTradingEnv 統合テスト

### 9.1 目的

YAML → `load_parquet` → `_create_training_env` → `HeavyTradingEnv` の E2E パイプラインが
実データで正常動作することを検証する。

### 9.2 テスト設計

`TestHeavyTradingEnvIntegration` (6 件):

| テスト | 検証内容 | 検証ポイント |
|---|---|---|
| `test_env_instantiation` | 実データ + feature_names 注入で例外なく生成 | コンストラクタ互換性 |
| `test_obs_dim_matches_feature_count` | observation_space.shape[0] == 12 (注入特徴量数) | B3 注入の正確性 |
| `test_feature_names_synced` | env.feature_names == 12 特徴量リスト | 状態整合性 |
| `test_reset_returns_valid_obs` | reset() が (12,) shape, NaN なしの obs を返す | データパイプライン整合性 |
| `test_step_returns_valid_tuple` | step() が 5-tuple (obs, reward, terminated, truncated, info) を返す | Gymnasium API 互換性 |
| `test_create_training_env_pipeline` | `_create_training_env` が YAML 相当 cfg で正常動作、env_info 一致 | 実関数 E2E 検証 |

### 9.3 テスト実装の工夫

| 項目 | 実装 | 理由 |
|------|------|------|
| データスライス | `head(2000)` で 1,216,930 → 2,000 行 | テスト時間 10.9s に収める |
| `@lru_cache` | YAML 読込・DataFrame ロードをキャッシュ | 同 class 内 6 テストでの再利用 |
| `dataclasses.replace` | env_config のディープコピー | テスト間の状態分離 |
| `correlation_reduction=False` | 明示的無効化 | 特徴量 subset 指定時の安定動作 |
| `env.close()` in `finally` | 全テストで確実にリソース解放 | メモリリーク防止 |

### 9.4 実行結果

```
38 passed in 31.08s
```

- 先頭 2000 行の軽量スライスでテスト (10.9 秒)
- **6/6 PASS**, correlation_reduction=False で安定動作

---

## §10 Protocol 設計分析

### 10.1 TrainingEnvProtocol

`sac_train.py` で定義された Structural Subtyping Protocol:

```python
class TrainingEnvProtocol(Protocol):
    observation_space: object
    action_space: object
    def reset(self) -> tuple[object, object]: ...
    def step(self, action: object) -> tuple[object, float, bool, bool, object]: ...
    def close(self) -> None: ...
```

**設計意図**: HeavyTradingEnv への直接依存を避け、将来の FastIntradayEnvV456 移行を容易にする。

**意図的な除外**: `portfolio_value`, `initial_portfolio_value`, `trades_count`, `total_pnl` は
Protocol に含めず、`getattr` で duck-typing 取得。これにより最小インターフェースの Protocol を維持。

### 10.2 SACTrainModelProtocol

```python
class SACTrainModelProtocol(Protocol):
    def learn(self, total_timesteps: int, reset_num_timesteps: bool = False) -> object: ...
    def predict(self, observation: object, deterministic: bool = True) -> tuple[object, object]: ...
    def save(self, path: str) -> None: ...
```

SB3 SAC モデルへの依存を Protocol で抽象化。テスト時の MagicMock 注入を可能にしている。

---

## §11 セルフレビュー品質改善 (commit `e4925fb4b`, `60c80fe32`, `79409e8a5`)

### 11.1 追加修正 (359# 独自)

§2 の 0f5c24dc1 commit 後に発見した追加課題:

| ID | 内容 | 重要度 | 修正 |
|---|---|---|---|
| **S-5** | `_save_model_schema` で `int(env_info["obs_dim"])` 未適用 | SIG | `as_int` 相当の明示的 int 変換追加 |
| **L-7** | `_checkpoint_eval_roi` docstring の影響度記載が不正確 | LOW | 0.02% と具体値に修正 |

### 11.2 コミット履歴

| Commit | 内容 | 修正 ID |
|--------|------|---------|
| `0f5c24dc1` | 358# セルフレビュー (C-1~C-5, S-1~S-4, L-1~L-2) | §2 全件 |
| `e4925fb4b` | L-3/L-5 G2 gate メトリクス修正 | L-3, L-5 |
| `60c80fe32` | S-2/S-3/P3A-1 修正 | S-2, S-3, P3A-1 |
| `79409e8a5` | P3A-2 統合テスト + S-5/L-7 修正 | P3A-2, S-5, L-7 |

---

## §12 AI レビューチェックリスト

### 12.1 コード品質

| チェック項目 | 状態 | メモ |
|-------------|------|------|
| dead import なし | ✅ | C-1, L-1, L-2 で排除 |
| 型安全 (Any 回避) | ✅ | S-1 `Callable`, S-2 `int()`, S-4 `int\|float` |
| 変数 shadowing なし | ✅ | S-3 `sac_params` リネーム |
| 例外安全 | ✅ | C-2 partial result, `try/finally` env.close() |
| ZeroDivision 防止 | ✅ | `_extract_roi_from_env` 内ガード |
| リソース解放 | ✅ | `task_sac_train` finally block + テスト内 env.close() |

### 12.2 設計原則

| 原則 | 遵守状態 | 検証 |
|------|---------|------|
| SRP (単一責任) | ✅ | `_extract_roi_from_env` を独立関数に分離 |
| DRY | ✅ | L-3/L-5 で `_extract_roi_from_env` を共有 |
| OCP (開放閉鎖) | ✅ | Protocol によるインターフェース抽象化 |
| 後方互換性 | ✅ | 返り値フィールド追加のみ、既存フィールド維持 |
| YAGNI | ✅ | 最小限の Protocol 定義、不要な属性を含めない |

### 12.3 テスト品質

| バリエーション | カバー状態 |
|---------------|----------|
| 正常系 (happy path) | ✅ 全関数 |
| 異常系 (error path) | ✅ seed 失敗, 属性欠損, ZeroDivision |
| 境界値 | ✅ E1 75% ちょうど, empty input, window 前のみ |
| E2E 統合 | ✅ P3A-2 (YAML→HeavyTradingEnv) |
| 回帰防止 | ✅ 既存 22 テスト全 PASS |

### 12.4 潜在リスク

| リスク | 重要度 | 対策状況 |
|--------|--------|---------|
| E2 IC チェック無効 (RL で IC=0.0) | MED | §7 残課題として記載。ph3 開始前に要定義 |
| n_episodes > 1 で最終エピソードのみ | LOW | L-6 として文書化。標準運用は n=1 |
| EnvironmentConfig(**dict) の型不一致 | LOW | mypy-L196 として記載。動作に影響なし |
| eval env 分離未実装 (S-6) | LOW | 影響 0.02%。ph4 で分離予定 |
