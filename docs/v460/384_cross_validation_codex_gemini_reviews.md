# 384# 検証: Codex (382#) / Gemini (383#) レビュー クロスバリデーション

**Date**: 2026-03-12
**Scope**: `docs/v460/382_ph3_rev_379_sb3_stub_and_g2_pipeline_review.md`, `docs/v460/383_gemini_sb3_pipeline_and_codex_review.md`

---

## 0. 検証方針

Codex・Gemini 両レビューの全指摘事項をコード実証で検証し、判定（確認/否認/部分的）と対応方針を決定する。
既存の v4XX シリーズ実装・ドキュメントからの参照可能資産の拾い上げも行う。

---

## 1. CRITICAL-1: Checkpoint 評価が training env を直接進めている

### 検証結果: **確認 (Confirmed)** — ただし影響度は Codex/Gemini の主張より限定的

### 実証

```python
# scripts/v460/lib/tasks/sac_train.py L287-315
def _train_with_checkpoints(model, env, total_timesteps, cfg):
    while remaining > 0:
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
        roi = _checkpoint_eval_roi(model, env)  # ← 同一 env!

# _checkpoint_eval_roi (L349-358)
def _checkpoint_eval_roi(model, env, max_steps=5000):
    obs, _ = env.reset()       # ← env state をリセット
    while not done and steps < max_steps:
        obs, ..., = env.step(action)  # ← env.current_step を進める
```

SB3 2.7.0 の `OffPolicyAlgorithm._setup_learn()` (base_class.py L354):
```python
if reset_num_timesteps or self._last_obs is None:
    self._last_obs = self.env.reset()
```
`reset_num_timesteps=False` の場合、`_last_obs` は更新されない。
しかし `_checkpoint_eval_roi()` が `env.reset()` + `env.step()` でenvの内部状態を変更済み。

**SB3の `self.env` は `DummyVecEnv` でラップされるが、内部は同一envインスタンスへの参照。**

### 影響の精密評価

Codex/Gemini は「replay buffer や rollout state と env 状態の致命的同期破壊」と主張するが:

1. **`_last_obs` のズレ**: checkpoint eval 後、`_last_obs` は eval 前の obs のまま。次の `collect_rollouts()` で 1 transition だけ `(stale_obs, action, reward, new_obs)` が生成される。50K steps 中 10 checkpoints で **10 corrupt transitions** (0.02%) — SAC off-policy buffer としては無視可能。

2. **env position state のリセット**: より深刻。Checkpoint eval が `env.reset()` → `current_step=0` → 5K steps 進める。訓練はデータの途中から再開されるべきだが、eval 後は step=5000 の位置から再開される。**データの前半に偏った学習**が発生する。

3. **replay buffer 自体は影響なし**: off-policy なので buffer 内の古い transition は state-independent に利用される。buffer 内容の汚染はない。

**結論**: 「致命的破壊」は過大評価だが、**データ位置のリセットによる学習バイアスは実在**する。修正必須。

### 対応方針

- `_train_with_checkpoints()` 内で別途 `checkpoint_eval_env` を作成し、`_checkpoint_eval_roi()` に渡す
- 訓練 env は一切触らない
- **365# §4.2** にも warm-start 時の env 状態保存について言及あり → 同一設計思想

---

## 2. CRITICAL-2: OOS 評価が同じ先頭 10K step を 3 回繰り返す

### 検証結果: **確認 (Confirmed)**

### 実証

```python
# scripts/v460/lib/sac_common.py L116-139 (evaluate_model_oos)
for _ in range(max(n_episodes, 1)):    # n_episodes=3
    obs, _ = env.reset()               # random_start=False → current_step=0
    while not done and steps < max_steps_per_episode:  # max_steps=10_000
        ...
```

`HeavyTradingEnv.reset()` (core.py L697-700):
```python
else:  # random_start=False
    self.current_step = 0
```

**val_df は ~243K 行。3 episode × 10K steps = 先頭 10K の 3 回再生。残り 233K 行は未評価。**

### 追加問題: 集約の不整合

```python
# sac_common.py L133-139
episode_rois.append(extract_roi_from_env(env))
total_trades += int(getattr(env, "trades_count", 0))
...
avg_roi = sum(episode_rois) / len(episode_rois)  # ROI は平均
"trade_count": total_trades,                       # trades は合計
"gross_pnl": float(getattr(env, "total_pnl", 0.0)),  # PnL は最終 episode のみ
```

同じ 10K を 3 回走るので `trade_count` が 3 倍計上されるが、ROI は同区間の平均（≈同じ値）、PnL は最後の 1 回分のみ。

### 対応方針

- G2 OOS: n_episodes=1、max_steps 制約なし、holdout 全区間 1-pass
- `gross_pnl` も episode 分の合計に統一
- 速度対策が必要なら、データの非重複ウィンドウ分割を明示的に実装

---

## 3. HIGH-1: 1 seed がクラッシュしても G2 が PASS しうる

### 検証結果: **確認 (Confirmed)**

### 実証

```python
# scripts/v460/run_experiment.py L245-253
except Exception as e:
    seed_results.append({
        "seed": seed,
        "gross_roi": 0.0,  # ← error でも ROI=0.0
        "error": str(e),
    })
    continue

# _evaluate_g2_from_results L327-350
positive_seeds = sum(1 for s in seed_results if float(s.get("gross_roi", 0)) > 0)
```

- 3 seed 正 (ROI > 0) + 1 seed crash (ROI = 0.0)
- `float(0.0) > 0` → **False** → positive_seeds = 3
- `ratio = 3/4 = 0.75 >= 0.75` → **PASS**
- `worst_seed_roi = 0.0 > -0.02` → **PASS**

**ただし**: 現実には positive_seed_ratio の判定自体は正しく 0.0 を非正として扱う。
本当の問題は **error フラグが G2 判定ロジックで無視される** こと。

### 対応方針

- `_evaluate_g2_from_results()` の冒頭で `error` キーを持つ seed が存在する場合は即 `gate_result="ERROR"` を返す
- 代替: `n_successful_seeds == len(seeds)` を必須条件に追加

---

## 4. HIGH-2: Val env が OOS データで scaler を再計算

### 検証結果: **確認 (Confirmed)**

### 実証

```python
# sac_train.py L120,156
env, env_info = _create_training_env(train_df, cfg)     # train scaler
eval_env, _ = _create_training_env(val_df, cfg)          # val scaler (別)

# _create_training_env → HeavyTradingEnv.__init__ → _compute_scaler_from_data
# config.train_end_index = None (未設定)
```

```python
# initialization.py L556-566
if train_end_index is not None and train_end_index < self._feature_matrix.shape[0]:
    scaler_features = self._feature_matrix[:train_end_index]
else:
    scaler_features = self._feature_matrix  # ← 全データ (train/val それぞれ)
    if train_end_index is None:
        logger.warning("train_end_index not provided. Computing scaler on the entire dataset.")
```

**train env: train_df (973K行) の統計で正規化**
**val env: val_df (243K行) の統計で正規化**
→ 同じモデルが異なるスケールの observation を受け取る

### 対応方針

`EnvironmentConfig` は `scaler_mean` / `scaler_std` フィールドを既に持っている (config.py L203-204):
```python
scaler_mean: list[float] | None = None  # Optional schema-provided observation scaler
scaler_std: list[float] | None = None   # Optional schema-provided observation scaler
```
→ train env の scaler を val env の EnvironmentConfig に注入すればよい。既存の `_setup_scaler()` がこれを読む:
```python
# PHASE3_IMPLEMENTATION_INSTRUCTIONS.md L243-246
def _setup_scaler(self):
    if "scaler_mean" in self.config and "scaler_std" in self.config:
```

---

## 5. HIGH-3: テスト基盤の SB3 グローバル mock

### 検証結果: **確認 (Confirmed)** — ただし現状は pip 版 SB3 が正常にロードされるため即座の問題はない

### 実証

`tests/conftest.py` の SB3 関連コード:
- **L95-130**: `project_root / "stable_baselines3" / "__init__.py"` を探す → **不存在** (リネーム済み)
- **L235-260**: 同上、`stable_baselines3/` ディレクトリを探す → **不存在**
- **L308-468**: SB3 import 失敗時に `_DummyModel` を注入 → pip 版が存在するため発火しない

**現時点では害はないが、デッドコードが 200 行以上残存。** pip版 SB3 が何らかの理由で import できない場合にダミーが注入される経路が残る。

### 対応方針 (後続タスク)

- `_sb3_test_stub/` の削除は conftest.py 整理の後
- conftest.py の global stub 注入を削除し、pip 版 SB3 の import 失敗はテスト skip (`@pytest.mark.skipif`) に
- **384# では着手しない** — テスト影響範囲が広く、単体での検証が必要

---

## 6. MEDIUM-1: 仕様ドリフト (000 vs 現行)

### 検証結果: **確認 (Confirmed)**

| 項目 | 000 提案書 | 現行実装 | 乖離 |
|---|---|---|---|
| 手数料 | maker 0% (§0, §1 教訓 #4) | `transaction_cost: 0.001` | ✗ 不一致 |
| G2 E2 | IC seed std ≤ 0.03 (§3.4) | ROI seed std ≤ 0.03 (gate_thresholds.yaml) | ✗ 不一致 |
| G2 測定 | 4 seed × 50K steps | 同一 | ✓ 一致 |

### 判断

「ROI が負なのは 0.1% コストのため」という 379# の解釈は 000 の前提と矛盾する。
000 は Coincheck maker 0% を明示的に前提としている。

**しかし**: Coincheck の maker fee は実際は **0%** (公式)。`transaction_cost: 0.001` は 000 の YAML コメントでは「Coincheck Maker」と記載しているが、これは誤記。Coincheck の taker fee が 0% (2024年以降)。

→ **手数料の正確な値を取引所 API / 公式ドキュメントから再確認し、000 または YAML を修正する必要がある。**

### 対応方針

- 384# では `transaction_cost` の値変更は行わない (取引所手数料の実態確認が必要)
- 000 に「E2: IC → ROI に変更済み (363# A4)」の改訂注記を追加
- 本件は別チケットで取引所手数料の正確な値を確定させる

---

## 7. LOW-1: sitecustomize.py と _sb3_test_stub 整理

### 検証結果: **確認 (Confirmed)**

- `sitecustomize.py`: `_prefer_local_package()` は `return False`、`_replace_stub_with_filebacked()` は定義のみ残存
- `_sb3_test_stub/`: テスト用途のみ。pip 版 SB3 が利用可能な現状では不要

### Codex/Gemini の相違点

| 項目 | Codex | Gemini | 検証判断 |
|---|---|---|---|
| `_sb3_test_stub/` 削除 | テスト局所mock化が先 | 即座に削除 | **Codex寄り** — conftest.py整理が先 |
| `sitecustomize.py` 削除 | torch bootstrap のため残す余地あり | 即座に削除 | **Gemini寄り** — torch bootstrap は conftest.py に移動可能 |
| `sb3_compat.py` 削除 | テスト整理後 | 即座に削除 | **Codex寄り** — conftest.py が依存 |
| `import_real_sb3()` 削除 | 賛成 | 賛成 | **両者合意** — スタブ削除後は不要 |

### 対応方針 (384#)

- `sitecustomize.py`: SB3 関連デッドコードを削除、最小化
- `import_real_sb3()`: 削除 (sac_common.py)
- `_sb3_test_stub/` / `sb3_compat.py` / conftest.py 整理: 後続タスク

---

## 8. Gemini ROI 改善戦略の検証

### 仮説 A (transaction_cost の圧殺): 部分的に妥当

Gemini: 「cost=0.0 からスタートし段階的に引き上げる curriculum が必要」

**検証**: 000 提案書 §0 は「maker 0%」を前提とする。YAML の `transaction_cost: 0.001` が正しいか自体が不確定。
もし本当に maker 0% なら、現行の cost は **誤設定** であり、curriculum 以前の問題。

**既存実装**: `BalanceCurriculumManager` (balance_curriculum.py) は action balance の curriculum を管理するが、**transaction_cost の段階的引き上げ機能は未実装**。EnvironmentConfig に `transaction_cost` フィールドがあるため、エピソード毎に動的変更は技術的に可能だが、SB3 の replay buffer との整合性問題がある (365# §3.5 参照)。

### 仮説 B (50K ステップ不足): 妥当

Gemini: 「最低 500K 必要」

**検証**: 一般的な SAC 訓練で 50K は確かに短い。ただし Codex が指摘する通り、**計測系 (CRITICAL-1,2) を先に修正しないと、長時間訓練の結果も信頼できない**。

### 仮説 D (gamma=0.80 短期すぎ): 要検討

Gemini: 「gamma=0.99 以上が必須」

**検証**: gamma=0.80 は v451 由来で「短期スキャルピング」用。
- γ=0.80: 実効ホライズン = 1/(1-0.80) = 5 steps = 5 分
- γ=0.99: 実効ホライズン = 1/(1-0.99) = 100 steps = 100 分 ≈ 1.7 時間
- γ=0.95: 実効ホライズン = 20 steps = 20 分

1 分足で 10bps 超のスイングを捉えるには 5 分は短い。**ただし gamma を上げると Q-value の variance も増大し、学習不安定性のリスクがある**。段階的に 0.95 → 0.99 での比較実験が妥当。

### Gemini の優先順位への見解

Gemini 383#:
1. `curriculum_learning` (最優先)
2. `adaptive_threshold_mode` (次点)
3. `domain_randomization` (後回し)

**Codex 382# の優先順位** (検証同意):
1. 計測系修正 (CRITICAL-1,2) **← これが最優先**
2. seed fail-fast (HIGH-1)
3. scaler 引き渡し (HIGH-2)
4. パラメータチューニング (gamma, curriculum)

**結論**: Codex の「計測系修正が先」は正しい。計測が壊れた状態でパラメータを変えても Gate 判定が信頼できない。

---

## 9. 見落とし事項 / 追加発見

### 9.1 `_create_training_env()` が EnvironmentConfig に train_end_index を未注入

`data.train_end_index` は YAML で 973544 が設定されているが、`_create_training_env()` は:
```python
env_config = EnvironmentConfig(**env_cfg)  # env_cfg は environment: セクションのみ
```
`env_cfg` は `configs.environment` であり `data.train_end_index` は含まない。
→ scaler は常に `train_end_index=None` で計算される

### 9.2 `_checkpoint_eval_roi()` の戻り値が `_sig_cache` キャッシュ修正前に定義

`_sig_cache` の遅延初期化 (`if not hasattr(self, '_sig_cache')`) は毎ステップ `hasattr` を呼ぶ。
**Gemini の指摘通り `__init__` 初期化が正攻法だが、`hasattr` のコストは ~50ns でありステップ単位では無視可能**。
→ 修正は望ましいが、384# の scope 外 (reward_calculator.py の大幅変更を伴う)

### 9.3 `EnvironmentConfig` の `scaler_mean/scaler_std` は `list[float]` 型

HIGH-2 修正時に `train_env.scaler_mean` (ndarray) を `list[float]` に変換して注入する必要がある。

---

## 10. 既存実装・vXXX シリーズ参照

| 参照元 | 関連内容 | 活用方法 |
|---|---|---|
| **365# §3.5** | CurriculumManager 有効化時の注意事項 | curriculum_learning 導入時に replay buffer flush を要検討 |
| **365# §4** | Warm-start incremental training 設計 | checkpoint eval env 分離と同一思想 |
| **363# A3** | 時系列 train/val split (shuffle 禁止) | OOS 1-pass 評価の設計根拠 |
| **363# A4** | IC seed std → ROI seed std 変更 | 000 改訂の根拠 |
| **361# F1, 362# G1** | in-sample 評価の過学習リスク | eval env 分離の設計根拠 |
| **372# audit** | evaluate_model_oos の集約修正 | 集約規約の整理根拠 |
| **BalanceCurriculumManager** | balance_curriculum.py (506行) | cost curriculum は未実装だが stage 管理は流用可能 |
| **EnvironmentConfig.scaler_mean/std** | config.py L203-204 | train→val scaler 引き渡しに既存フィールドを活用 |

---

## 11. 384# 即時対応アクションプラン

### 即時修正 (実装完了 ✅)

| # | 対応 | ファイル | 重要度 | 状態 |
|---|---|---|---|---|
| A1 | checkpoint eval を別 env で実行 | sac_train.py | CRITICAL | ✅ |
| A2 | OOS 評価を 1-pass (max_steps 制約なし) に変更 | sac_common.py, g2_sac_train.yaml | CRITICAL | ✅ |
| A3 | seed 例外時に G2 を ERROR にする | run_experiment.py | HIGH | ✅ |
| A4 | train scaler を val env に注入 (`_build_val_env_config`) | sac_train.py | HIGH | ✅ |
| A5 | sitecustomize.py の SB3 デッドコード削除 | sitecustomize.py | LOW | ✅ |
| A6 | `import_real_sb3()` の削除 | sac_common.py, sac_retrain_scheduler.py | LOW | ✅ |

#### 変更サマリ

- **sac_train.py**: `checkpoint_eval_env` の新規作成、`_build_val_env_config()` 追加、`_train_with_checkpoints()` に `checkpoint_eval_env` 引数追加
- **sac_common.py**: `evaluate_model_oos()` の `max_steps_per_episode` デフォルトを `None` (全走査) に変更、`gross_pnl` を全エピソード平均に修正、`import_real_sb3()` 削除
- **run_experiment.py**: `_evaluate_g2_from_results()` に error seed 検出ロジック追加 (`gate_result="ERROR"`)
- **g2_sac_train.yaml**: `n_episodes: 3` → `n_episodes: 1`
- **sitecustomize.py**: SB3 スタブ関連 dead code 全削除 (docstring のみ残存)
- **sac_retrain_scheduler.py**: `import_real_sb3` → 直接 `from stable_baselines3 import SAC`
- **test_sac_retrain_scheduler.py**: `_mock_sb3_import` を `patch.dict("sys.modules")` 方式に移行

#### テスト

- 新規テスト: `test_384_pipeline_fixes.py` (7 tests) — OOS 全走査、scaler 注入、cfg 非変更
- 既存テスト拡張: `test_356_g2_sac_blockers.py` に error seed テスト 2 件追加
- 全 v460 テスト: **4614 passed, 0 failed**

#### セルフレビュー所見

1. **OOS 全走査の所要時間**: `max_steps` 制約撤廃により OOS 243K steps を全走査する。379# で `inspect.signature()` キャッシュ化済みのため ~0.3ms/step の overhead は解消されているが、**243K steps × ~0.1ms/step ≈ 24 秒/seed**。4 seed × 24s ≈ 2 分で許容範囲。ただし将来データ量増加時は要注意。

2. **`_build_val_env_config` の scaler 注入パス**: `EnvironmentConfig(**env_cfg)` が dataclass コンストラクタで `scaler_mean`/`scaler_std` を受け取り、`_setup_scaler()` が config から読んで `_compute_scaler_from_data()` をスキップする流れを **コードレベルで確認済み** (initialization.py L520-536, core.py L548-554)。

3. **`checkpoint_eval_env` が train_df 全体から scaler を再計算する重複負荷**: `_create_training_env(train_df, cfg)` を 2 回呼ぶため、scaler 計算が 2 回行われる。train env の scaler を checkpoint_eval_env にも注入すれば回避可能だが、**973K 行の mean/std 計算は ~数 ms で無視可能**。現時点では最適化不要。

4. **`n_episodes: 1` への変更と `trade_count` の意味変化**: 旧 n_episodes=3 では trade_count が 3 倍計上されていた。n_episodes=1 に変更したため、**既存の訓練結果 JSON (trade_count: 1001〜1455) との比較は不整合**。ドキュメントに明記済み。

### 後続タスク (別セッション)

| # | 対応 | 依存 |
|---|---|---|
| B1 | conftest.py の SB3 global stub 整理 | テスト影響調査 |
| B2 | `_sb3_test_stub/` 削除 | B1 完了後 |
| B3 | `sb3_compat.py` 削除 | B1 完了後 |
| B4 | gamma 比較実験 (0.80 vs 0.95 vs 0.99) | A1-A4 完了後 |
| B5 | transaction_cost の正確な値確定 | 取引所 API 確認 |
| B6 | curriculum_learning (cost progression) 設計 | B5 完了後 |
| B7 | 000 提案書の改訂 (E2, 手数料) | B5 完了後 |
