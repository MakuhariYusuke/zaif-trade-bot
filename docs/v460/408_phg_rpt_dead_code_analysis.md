# 408# デッドコード調査・重複解消 (`ztb/trading/environment/`)

**作成日**: 2026-03-13  
**調査対象**: `ztb/trading/environment/` 配下の全 `.py`（`archived/` は除外）  
**実スキャン結果**: 57 files / 17,764 LOC  
**除外方針**: `tests/` からの参照だけでは「生コード使用」とみなさない。`scripts/testing/` も同様に参考扱いとした。

---

## 1. 結論サマリ

### 1.1 現行 v460 訓練で実際に使われる報酬経路
現行 v460 訓練パイプラインでは、`scripts/v460/lib/tasks/sac_train.py::_create_training_env(...)` が `HeavyTradingEnv` を生成し、`ztb/trading/environment/heavy_env/mixins/initialization.py` で報酬計算器を選ぶ。

現行 config `configs/v460/experiments/g2_sac_reward_clean.yaml` には以下が存在しない。

- `reward_settings.custom_reward_params.type`
- `use_simple_reward`
- `curriculum_stage`
- `curriculum_learning`

そのため、現行 v460 の live path は次で固定される。

```text
HeavyTradingEnv.step()
  -> RewardCalculator.calculate_reward(...)
     -> _calculate_default_reward(...)
     -> _post_process_reward(...)
```

### 1.2 現行 v460 で使われない主な報酬経路
- `V457RewardCalculator`:
  - `custom_reward_params.type == "pnl_centered"` のときだけ使用
  - 現行 v460 config では未使用
- `calculate_reward_simple()`:
  - `use_simple_reward == true` のときだけ使用
  - 現行 v460 config では未使用
- `curriculum_stage` ベースの `_calculate_*_reward()` 群:
  - `curriculum_stage` または curriculum manager が有効なときだけ使用
  - 現行 v460 config では未使用

### 1.3 調査結果の要旨
- **本当に危ない God Object は `RewardCalculator`**。
- `components/calculators/`・`components/reward/`・`components/rewards/` の 3 分裂は実在し、命名と責務が崩れている。
- 一方で、`dynamic_reward_shaper.py`、`asymmetric_reward_scaler.py`、`signal_integrator.py` は**死コードではない**。`RewardCalculator` の専属 collaborator である。
- `bridge.py`、`reward/metrics.py`、`simplified_reward_calculator.py`、`components/reward_utils.py` は**死コードまたは準死コード候補**。
- `factory_v456.py`、`fast_intraday_env*.py`、`schema_env_factory.py` は**現行 v460 では未使用だが、旧版スクリプトからは参照されている legacy-live**。

---

## 2. 調査手順

- `find ztb/trading/environment -path '*/archived/*' -prune -o -name '*.py'`
- `wc -l` で行数を集計
- `rg` で `ztb/` と `scripts/` 配下の非テスト参照を追跡
- `configs/v460/experiments/g2_sac_reward_clean.yaml` と関連初期化コードを読んで、現行訓練パスを特定
- `RewardCalculator` 本体、`heavy_env/mixins/initialization.py`、`environment.py`、`components/reward_calculator.py`、`factory_v456.py`、`schema_env_factory.py`、`bridge.py`、`reward/` と `rewards/` のモジュールを読んで責務と到達性を整理

---

## 3. RewardCalculator 到達性調査

### 3.1 選択ロジック
`ztb/trading/environment/heavy_env/mixins/initialization.py:651-676` の報酬計算器選択は次。

| 条件 | 使用される計算器 |
|---|---|
| `reward_settings.custom_reward_params.type == "pnl_centered"` | `V457RewardCalculator` |
| それ以外 | `RewardCalculator` |

現行 `g2_sac_reward_clean.yaml` には `custom_reward_params.type` がないため、**`RewardCalculator` が使われる**。

### 3.2 RewardCalculator 内の分岐
`ztb/trading/environment/components/calculators/reward_calculator.py:866-1258`

| 分岐 | 条件 | 現行 v460 での到達 |
|---|---|---|
| `calculate_reward_simple()` | `use_simple_reward == true` | なし |
| `_calculate_action_discovery_reward()` | `curriculum_stage == "action_discovery"` | なし |
| `_calculate_forced_balance_reward()` | `curriculum_stage == "forced_balance"` | なし |
| `_calculate_smart_incentive_reward()` | `curriculum_stage == "smart_incentive"` | なし |
| `_calculate_balanced_transition_reward()` | `curriculum_stage == "balanced_transition"` | なし |
| `_calculate_pnl_focused_reward()` | `curriculum_stage == "pnl_focused"` | なし |
| `_calculate_trading_focused_reward()` | `curriculum_stage == "trading_focused"` | なし |
| `_calculate_profit_optimized_reward()` | `curriculum_stage == "profit_optimized"` | なし |
| `_calculate_risk_management_reward()` | `curriculum_stage == "risk_management"` | なし |
| `_calculate_opportunity_cost_reward()` | `curriculum_stage == "opportunity_cost"` | なし |
| `_calculate_ultra_profit_reward()` | `curriculum_stage == "ultra_profit"` | なし |
| `_calculate_stability_optimized_reward()` | `curriculum_stage == "stability_optimized"` | なし |
| `_calculate_backtest_optimization_reward()` | `curriculum_stage == "backtest_optimization"` | なし |
| `_calculate_default_reward()` | stage 未設定または map fallback | **あり** |

### 3.3 現行 v460 で実際に通る経路

```text
scripts/v460/lib/tasks/sac_train.py::_create_training_env(...)
  -> HeavyTradingEnv(...)
  -> heavy_env/mixins/initialization.py
     -> RewardCalculator(...)
  -> RewardCalculator.calculate_reward(...)
     -> _calculate_default_reward(...)
     -> _post_process_reward(...)
```

### 3.4 RewardCalculator 内の死メソッド候補
非テスト参照を追った結果、以下は `RewardCalculator` 外から呼ばれていない。

| シンボル | 位置 | 判定 | 理由 |
|---|---:|---|---|
| `get_current_regime()` | `reward_calculator.py:196` | likely dead | 非テスト参照なし |
| `reset_episode_state()` | `reward_calculator.py:707` | likely dead | 非テスト参照なし |
| `test_reward_calculation()` | `reward_calculator.py:1947` | dead in prod | 本番クラス内 self-test。非テスト参照なし |

`test_reward_calculation()` は本番コードから切り離して `tests/` へ移すべき対象である。

---

## 4. デッドコードリスト

### 4.1 強い dead / isolated 候補

| パス | 行数 | 判定 | 理由 |
|---|---:|---|---|
| `ztb/trading/environment/components/calculators/simplified_reward_calculator.py` | 37 | dead / tests-only | `scripts/testing/test_simplified_reward_calculator.py` 以外の非テスト参照なし。現行 v460 でも未使用 |
| `ztb/trading/environment/components/reward/metrics.py` | 301 | likely dead | `LongTermMetrics` の非テスト参照なし |
| `ztb/trading/environment/components/reward_utils.py` | 59 | likely dead | `RewardUtils` 相当の accessor/helper が他所に重複しており、直接の非テスト参照なし |
| `ztb/trading/environment/bridge.py` | 867 | likely dead | `VirtualTradingBridge` / `LiveTradingBridge` / `BridgeReplay` の外部 import を確認できず、README mention のみ |
| `ztb/trading/environment/components/calculators/reward_calculator.py::test_reward_calculation` | method | dead in prod | 本番クラス内テストメソッド。外部 caller なし |
| `ztb/trading/environment/components/calculators/reward_calculator.py::get_current_regime` | method | likely dead | 外部 caller なし |
| `ztb/trading/environment/components/calculators/reward_calculator.py::reset_episode_state` | method | likely dead | 外部 caller なし |

### 4.2 proxy-only / compatibility only
これは dead ではないが、**責務が proxy に限られる**。

| パス | 行数 | 判定 | 理由 |
|---|---:|---|---|
| `ztb/trading/environment/components/reward_calculator.py` | 17 | proxy-live | `RewardCalculator` / `SimplifiedRewardCalculator` / `V457RewardCalculator` の re-export shim。非テスト caller は旧 `scripts/v456/...` だけ |
| `ztb/trading/environment/environment.py` | 21 | proxy-live | `HeavyTradingEnv` / `EnvironmentConfig` export + `gym.register`。現行でも import 多数 |

### 4.3 legacy-live（現行 v460 では未使用）

| パス | 行数 | 判定 | 理由 |
|---|---:|---|---|
| `ztb/trading/environment/factory_v456.py` | 556 | legacy-live | `v456` / `v457` 系スクリプトから参照。現行 v460 からは未使用 |
| `ztb/trading/environment/fast_intraday_env.py` | 353 | legacy-live | `scripts/v455/*` から参照 |
| `ztb/trading/environment/fast_intraday_env_v456.py` | 1061 | legacy-live | `scripts/v456/*`, `scripts/v457/*` から参照 |
| `ztb/trading/environment/schema_env_factory.py` | 125 | legacy-live | `train_sac_v436.py` / `backtest_sac_v43x.py` など旧 schema-based script から参照 |
| `ztb/trading/environment/components/calculators/v457_reward_calculator.py` | 81 | legacy-live | `custom_reward_params.type == "pnl_centered"` 分岐専用。現行 v460 config では未使用 |

---

## 5. `components/calculators/` / `reward/` / `rewards/` の整理結果

### 5.1 `components/calculators/`

| ファイル | 行数 | 現状 |
|---|---:|---|
| `reward_calculator.py` | 2252 | 現行本流 |
| `simplified_reward_calculator.py` | 37 | tests-only / dead candidate |
| `v457_reward_calculator.py` | 81 | legacy conditional path |

### 5.2 `components/reward/`

| ファイル | 行数 | 現状 |
|---|---:|---|
| `balance_curriculum.py` | 505 | `RewardCalculator` から live |
| `mtf_weight_manager.py` | 205 | `RewardCalculator` から live |
| `trend_detector.py` | 216 | `RewardCalculator` から live |
| `unrealized_loss_penalty_calculator.py` | 68 | `RewardCalculator` から live |
| `opportunity_cost_penalty_calculator.py` | 51 | `RewardCalculator` から live |
| `metrics.py` | 301 | caller なし、dead candidate |

### 5.3 `components/rewards/`

| ファイル | 行数 | 現状 |
|---|---:|---|
| `forced_balance.py` | 389 | `RewardCalculator` から live |
| `pnl_focused.py` | 171 | `RewardCalculator` から live |
| `profit_optimized.py` | 112 | `RewardCalculator` から live |
| `smart_incentive.py` | 111 | `RewardCalculator` から live |
| `trading_focused.py` | 48 | `RewardCalculator` から live |
| `ultra_profit.py` | 65 | `RewardCalculator` から live |
| `confidence_penalty.py` | 96 | `RewardCalculator` から live |
| `base.py` | 126 | reward component 基底 |
| `utils.py` | 155 | `RewardCalculator` と一部 training/analysis script から live |
| `__init__.py` | 0 | 空 |

### 5.4 判断
- `components/rewards/` は **RewardCalculator のステージ別 reward policy 群** で、現行でも生きている。
- `components/reward/` は **補助 manager / detector / penalty calculator 群** と dead candidate `metrics.py` が混在している。
- `components/calculators/` は **router/orchestrator (`RewardCalculator`) と legacy calculator 群** が混在している。

現在の 3 分裂は、責務分割ではなく**履歴分裂**に近い。

---

## 6. `components/` 直下の重点ファイル判定

### 6.1 `dynamic_reward_shaper.py` / `asymmetric_reward_scaler.py` / `signal_integrator.py`
**死コードではない。**

いずれも `RewardCalculator` の collaborator として live。
外部 caller が少ないのは「使われていない」のではなく、`RewardCalculator` に凝集しているためである。

### 6.2 `data_manager.py` vs `data_processor.py`
**重複よりも密結合が問題。**

- `DataManager`: runtime accessor / step-level data access / buffer ownership
- `DataProcessor`: preprocessing / dtype optimization / streaming preparation

役割は異なるが、`heavy_env/mixins/initialization.py` で内部 array を直結しており、責務境界が弱い。

結論:
- dead code ではない
- ただし `DataProcessor -> DataManager` への内部バッファ注入は保守性リスク

### 6.3 `components/reward_calculator.py`
**統合残骸ではあるが、まだ消せない。**

理由:
- 旧スクリプトが `ztb.trading.environment.components.reward_calculator` を import している
- `heavy_env/core.py` でも TYPE_CHECKING import に使っている

整理方針は「即削除」ではなく、**caller を `components.calculators.*` に寄せた後で proxy 廃止**。

---

## 7. `environment/` 直下ファイルの判定

| ファイル | 行数 | 判定 | 理由 |
|---|---:|---|---|
| `environment.py` | 21 | proxy-live | `HeavyTradingEnv` export + `gym.register`。現行 caller 多数 |
| `factory_v456.py` | 556 | legacy-live | v456/v457 script 群が使用 |
| `fast_intraday_env.py` | 353 | legacy-live | v455 script 群が使用 |
| `fast_intraday_env_v456.py` | 1061 | legacy-live | v456/v457 script 群が使用 |
| `bridge.py` | 867 | likely dead | README 以外の外部 caller を確認できない |
| `schema_env_factory.py` | 125 | legacy-live | v43x training/backtest schema path で使用 |

---

## 8. 重複ロジックリスト

### 8.1 `calculate_reward()` と `calculate_reward_simple()` の後処理重複

| ペア | 行範囲 | 重複内容 | 統合提案 |
|---|---|---|---|
| `RewardCalculator.calculate_reward()` + `_post_process_reward()` | `reward_calculator.py:866-1258` | asymmetric scaling / clipping / signal integration / `_last_reward_components` 更新 | stage/base reward の計算だけを各 policy に残し、後処理は `RewardPostProcessor` に一本化 |
| `RewardCalculator.calculate_reward_simple()` | `reward_calculator.py:1261-1459` | dynamic shaping / signal integration / asymmetric scaling / reward components 更新を独自実装 | `calculate_reward_simple()` も同じ post-processor を使う。simple/complex の違いは base reward 生成までに限定 |

**所見**: `calculate_reward_simple()` は「簡易版」なのに後処理パイプラインを別実装しており、ドリフト源になっている。

### 8.2 設定 accessor の多重実装

| ペア | 行範囲 | 重複内容 | 統合提案 |
|---|---|---|---|
| `RewardCalculator.get_setting_* + _get_nested_setting` | `reward_calculator.py:711-763` | nested config 読み出し・型変換 | `RewardSettingsAccessor` 1 クラスに集約 |
| `RewardComponent._get_setting*` | `components/rewards/base.py:65-120` | `RewardContext` 経由の fallback 読み出し | 同上。context-aware adapter を薄く残すだけにする |
| `ForcedBalanceReward._get_nested_setting/_get_setting` | `components/rewards/forced_balance.py:28-83` | component 専用 nested key 解決 | dot-notation と component prefix ルールを共通 accessor に吸収 |
| `_get_reward_setting_*` | `heavy_env/mixins/reward.py:7-34` | int/float/bool getter | env 側も同じ accessor を使う |
| `RewardUtils` | `components/reward_utils.py:12-59` | int/float/bool getter + safe math | accessor 部は共通化、math helper だけ残すか `rewards/utils.py` に統合 |

### 8.3 forced-balance mapping の二重化

| ペア | 行範囲 | 重複内容 | 統合提案 |
|---|---|---|---|
| `RewardCalculator._map_forced_balance_penalty/_bonus` | `reward_calculator.py:600-676` | deviation->penalty/bonus mapping | `ForcedBalanceReward` 側に一本化し、`RewardCalculator` から旧 helper を削除 |
| `ForcedBalanceReward._map_forced_balance_penalty/_bonus` | `components/rewards/forced_balance.py:85-152` | 同等ロジック | 同上 |

### 8.4 proxy/export の二重窓口

| ペア | 行範囲 | 重複内容 | 統合提案 |
|---|---|---|---|
| `environment.py` vs `heavy_env/core.py` | `environment.py:1-21` | `HeavyTradingEnv` の別 import 入口 | caller を `heavy_env.core` へ寄せ、最後に proxy を縮退 |
| `components/reward_calculator.py` vs `components/calculators/*` | `components/reward_calculator.py:1-17` | reward calculator 別 import 入口 | v456/v457 caller を移行後に proxy 廃止 |

---

## 9. `archived/` 移動候補リスト

### 9.1 即候補

| ファイル | 移動先候補 | 理由 |
|---|---|---|
| `ztb/trading/environment/components/calculators/simplified_reward_calculator.py` | `ztb/trading/environment/archived/reward/simplified_reward_calculator.py` | script-level tests 専用で現行本流に無関係 |
| `ztb/trading/environment/components/reward/metrics.py` | `ztb/trading/environment/archived/reward/metrics.py` | 非テスト caller なし |
| `ztb/trading/environment/bridge.py` | `ztb/trading/environment/archived/bridge/bridge.py` | 外部 caller を確認できない |

### 9.2 条件付き候補（caller 移行後）

| ファイル | 移動先候補 | 前提 |
|---|---|---|
| `ztb/trading/environment/components/reward_calculator.py` | `ztb/trading/environment/archived/compat/components_reward_calculator.py` | v456/v457 caller を `components.calculators.*` へ移行後 |
| `ztb/trading/environment/factory_v456.py` | `ztb/trading/environment/archived/v456/factory_v456.py` | v456/v457 script を legacy lane へ隔離後 |
| `ztb/trading/environment/fast_intraday_env.py` | `ztb/trading/environment/archived/v455/fast_intraday_env.py` | v455 script 群の退避後 |
| `ztb/trading/environment/fast_intraday_env_v456.py` | `ztb/trading/environment/archived/v456/fast_intraday_env_v456.py` | v456/v457 script 群の退避後 |
| `ztb/trading/environment/schema_env_factory.py` | `ztb/trading/environment/archived/v43x/schema_env_factory.py` | v43x backtest/train scripts の legacy 化後 |
| `ztb/trading/environment/components/reward_utils.py` | `ztb/trading/environment/archived/reward/reward_utils.py` または統合削除 | caller 再確認後。共通 accessor へ吸収可能 |

---

## 10. 構造整理提案

### 10.1 現状の問題
- `reward/` と `rewards/` が単数/複数で分裂しており、意味がわからない
- `calculators/` に router (`RewardCalculator`) と legacy implementation (`V457`, `Simplified`) が同居している
- `components/` 直下にも reward 系 helper が散在している

### 10.2 推奨統合案

```text
ztb/trading/environment/components/reward/
  __init__.py
  router.py                 # 旧 RewardCalculator
  accessor.py               # 設定 accessor
  post_processor.py         # asymmetric scaling / clipping / signal integration
  context.py                # RewardContext
  policies/
    default.py
    action_discovery.py
    forced_balance.py
    smart_incentive.py
    pnl_focused.py
    trading_focused.py
    profit_optimized.py
    ultra_profit.py
    risk_management.py
    opportunity_cost.py
    stability_optimized.py
    backtest_optimization.py
  support/
    balance_curriculum.py
    mtf_weight_manager.py
    trend_detector.py
    dynamic_reward_shaper.py
    signal_integrator.py
    asymmetric_reward_scaler.py
    confidence_penalty.py
    unrealized_loss_penalty_calculator.py
    opportunity_cost_penalty_calculator.py
    utils.py
  legacy/
    v457_reward_calculator.py
    simplified_reward_calculator.py
```

### 10.3 具体的な統合方針
- `components/rewards/` と `components/reward/` は **`components/reward/` に統一**するのが自然
- `components/calculators/reward_calculator.py` は `components/reward/router.py` 相当へ改名・移動候補
- legacy 実装 (`simplified`, `v457`) は `legacy/` へ隔離
- `components/reward_calculator.py` proxy は caller 移行後に削除

---

## 11. RewardCalculator 分割提案

### 11.1 現状の問題
`RewardCalculator` は 2252 行 / 約 50 メソッドで、以下を 1 クラスで抱えている。

- 設定解決
- internal state tracking
- curriculum stage routing
- 各 reward policy 実装
- forced-balance math
- dynamic shaping
- signal integration
- asymmetric scaling
- logging / diagnostics
- self-test (`test_reward_calculation`)

これは単一責任原則に反している。

### 11.2 分割案（概略）

#### A. `RewardSettingsAccessor`
責務:
- `get_setting_float/int/bool/str`
- dot-notation nested key 解決
- `custom_reward_params` fallback
- component prefix ルール

#### B. `RewardStageRouter`
責務:
- `curriculum_stage` の解決
- stage -> policy mapping
- `use_simple_reward` / `pnl_centered` の分岐整理

#### C. `RewardPostProcessor`
責務:
- asymmetric scaling
- clipping
- signal integration
- telemetry component recording

#### D. `RewardPolicy` 群
責務:
- stage-specific base reward 計算のみ
- `default`, `forced_balance`, `pnl_focused`, `profit_optimized` など

#### E. `RewardRuntimeState`
責務:
- `_action_counts`
- `_win_count` / `_loss_count`
- `_previous_portfolio_value`
- `_last_reward_components`

#### F. `RewardDiagnostics`
責務:
- logging / structured logging / optional self-check
- `test_reward_calculation()` はここからも切り離し、実際には `tests/` へ移す

### 11.3 先にやるべき低リスク分割順
1. `test_reward_calculation()` を `tests/` へ移動
2. accessor 群を `RewardSettingsAccessor` に抽出
3. `_post_process_reward()` を独立クラスへ抽出
4. `forced_balance` helper を `ForcedBalanceReward` に一本化
5. stage-specific `_calculate_*_reward()` を policy class へ切り出す

---

## 12. 実務上の優先順位

### P0
- `RewardCalculator.test_reward_calculation()` を本番コードから除去
- `components/reward/metrics.py` の到達性を再確認し、なければ archive
- `bridge.py` を archive 候補として隔離

### P1
- `RewardCalculator` の accessor 群を共通 accessor へ抽出
- `calculate_reward()` と `calculate_reward_simple()` の後処理を一本化
- `forced_balance` mapping helper を二重定義から一本化

### P2
- `components/reward/` + `components/rewards/` の統合
- `components/reward_calculator.py` proxy の caller 移行
- v455/v456/v457 legacy 環境ファイルの archive 計画化

---

## 13. 最終判断

- **現行 v460 の本流は `RewardCalculator` + `_calculate_default_reward()` だけ**であり、報酬コードの大半は「現在は到達しない分岐」か「legacy 互換 path」である。
- 死コードそのものもあるが、より大きい問題は **proxy・legacy・policy・support が同一 package に混在している構造崩壊**。
- まずは `RewardCalculator` の self-test / accessor / post-process / forced-balance helper の切り出しを行い、その後に `reward/` と `rewards/` の統合へ進むのが最も低リスク。

## Codex 修正済み項目

| タスク | 修正内容 | 主対象 |
|---|---|---|
| T11 | dead file を archive へ移動し live export から `SimplifiedRewardCalculator` を除去 | `components/calculators/simplified_reward_calculator.py`, `components/reward/metrics.py`, `bridge.py`, `components/__init__.py`, `components/calculators/__init__.py` |
| T12 | `RewardCalculator.test_reward_calculation()` を本番コードから削除し、外部 regression test へ移動 | `components/calculators/reward_calculator.py`, `tests/unit/v460/test_codex_408_409_fixes.py` |
| T13 | `get_current_regime()` / `reset_episode_state()` に `DeprecationWarning` を追加 | `components/calculators/reward_calculator.py` |
| T14 | forced-balance penalty/bonus mapping を `ForcedBalanceReward` static helper に一本化 | `components/rewards/forced_balance.py`, `components/calculators/reward_calculator.py` |

補足:
- `scripts/testing/test_simplified_reward_calculator.py` も cascade で dead reference と判定し `archived/scripts/testing/` へ退避した。
- 回帰テストは `tests/unit/v460/test_codex_408_409_fixes.py` に集約した。
