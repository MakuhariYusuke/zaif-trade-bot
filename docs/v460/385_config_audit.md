# 385# 設定監査レポート

## 概要

385# における `transaction_cost` の矛盾修正をトリガーとして、
g2_sac_train.yaml と関連設定の包括的な監査を実施。
複数の不整合を特定。

## P0: CRITICAL — 運用に直接影響

### P0-1: `continuous_to_discrete_threshold` 訓練/ライブ乖離 — **386# 修正済み**

| 場所 | 旧値 | 新値 (386#) | 備考 |
|------|------|------------|------|
| **g2_sac_train.yaml** | **0.10** | 0.10 (変更なし) | 379# で設定済み |
| SAC_CONTINUOUS_THRESHOLD (constants.py) | ~~0.3333~~ | **0.10** | 386# で統一 |
| **live_trader/config.py** | ~~0.33~~ | **0.10** | 386# で統一 |
| backtest/adapters.py | 0.01 | 0.01 (未変更) | バックテスト用ハードコード |

**386# 修正**: `SAC_CONTINUOUS_THRESHOLD` と `live_trader/config.py` のデフォルトを
0.10 に統一。`ZTB_CONTINUOUS_TO_DISCRETE_THRESHOLD` 環境変数による上書きも引き続き可能。

### P0-2: `reward_scaling = 6.0` の暗黙適用 → デッドコード — **386# 修正済み**

- ~~EnvironmentConfig デフォルト: `reward_scaling = 6.0` (PPO由来)~~ → **1.0 に変更**
- ~~`_calculate_default_reward()` に `reward_scaling` パラメータなし~~ → **追加済み**
- ~~`inspect.signature()` フィルタで除外~~ → **正常に流れるようになった**
- SAC YAML に `reward_scaling: 1.0` を明示設定済み
- `DEFAULT_TRADING_ENV_CONFIG["reward_scaling"]` も 6.0 → 1.0 に変更
- `EnvironmentConfig.reward_scaling` デフォルトを `DEFAULT_REWARD_SCALING` (PPO) → 直接 `1.0` に変更

## P1: HIGH — 潜在的な結果影響

### P1-1: `action_space_type: "continuous_1d"` の無声書き換え

- YAML: `action_space_type: "continuous_1d"`
- `EnvironmentConfig.from_dict()` で `use_continuous_actions: true` を検出
- `action_space_type = "continuous"` に上書き (L639-646)
- sac_trainer.py は `startswith("cont")` チェックなので訓練は正常
- ただし `"continuous_1d"` 固有ロジックがあればサイレント破損

**評価**: 現時点では実害なし。YAML からの意図の表現力が損なわれる程度。

### P1-2: base.yaml `sac.gamma: 0.99` vs g2 YAML `sac_hyperparameters.gamma: 0.80`

- sac_trainer は `sac_hyperparameters` セクションを参照 → 0.80 が使われる
- base.yaml の `sac:` セクション経由のコードパスがあれば 0.99 が適用

**評価**: 現在の訓練パイプラインでは問題なし (0.80 を使用)。
ただし base.yaml の `sac.gamma: 0.99` は misleading。

### P1-3: 旧 EnvironmentConfig の共存 (`ztb/training/environments/environment_config.py`)

- メイン: `ztb/trading/environment/utils/config.py` (HeavyTradingEnv が使用)
- 旧版: `ztb/training/environments/environment_config.py`
  - `commission: 0.001` (coincheck 0% と矛盾)
  - `signal_guidance_enabled: True` (G2 で無効化意図)
- PPO trainer が旧版をインポート

**評価**: SAC 訓練には影響しない。PPO コード流用時にリスク。

### P1-4: `DEFAULT_FEE_RATE = 0.001` フォールバック汚染

- `ExchangeFeeModel` 初期化失敗時 → `FixedFeeModel` にフォールバック
- `FixedFeeModel` デフォルト = `DEFAULT_FEE_RATE = 0.001` (0.1%)
- **385# fix** の `transaction_cost: 0.0` は YAML 経由で直接設定されるため回避
- ただしコードパス上の防御が不完全

## P2: MEDIUM — 設計負債

### P2-1: HeavyTradingEnv.__init__ シグネチャのデフォルト値

```python
# core.py L355-358 — これらは config 経由で上書きされるため使われないが誤解を招く
initial_balance: float = 100_000.0      # 10万 ≠ YAML 10M
transaction_cost: float = 0.00075       # 0.075% ≠ 0% or 0.1%
max_position_size: float = 1.0          # 1 BTC ≠ YAML 0.01
```

### P2-2: `DEFAULT_TRANSACTION_COST` 二重定義

- `env_config.py`: `DEFAULT_TRANSACTION_COST = 0.001`
- `constants.py`: `DEFAULT_TRANSACTION_COST = 1e-5`
- 同名で 100 倍の差。使用箇所によって異なる値が適用される。

### P2-3: `bankruptcy_threshold = 2000` の実質無効

- 10M ポートフォリオの 0.02% → 99.98% 損失しないと発動しない。
- 意図: 安全装置。実態: 機能しない閾値。

### P2-4: `initial_portfolio_value` デフォルト (200K) と YAML (10M) の 50 倍差

- YAML 明示設定で問題なし。設定漏れ時に 50 倍の乖離リスク。

## 対応計画

| ID | 対策 | 対応時期 | 影響度 |
|----|------|---------|--------|
| P0-1 | threshold 統一 or ライブ env var 設定 | G4 (ライブ投入前) | CRITICAL |
| P0-2 | reward_scaling 修正 | 386# | HIGH |
| P1-2 | base.yaml の sac.gamma 削除/整理 | 386# | MEDIUM |
| P1-3 | 旧 EnvironmentConfig 廃止計画 | v461 | MEDIUM |
| P1-4 | FeeModel フォールバック防御強化 | 386# | MEDIUM |
| P2-* | 段階的なデフォルト値整理 | v461 | LOW |
