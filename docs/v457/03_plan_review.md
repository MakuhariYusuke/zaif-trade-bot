# v457 逆風打開レビュー（戦略・資産発掘・改善提案）

## 0. 先に結論（最重要だけ）
- v457の「PnLリセット」は**現状コード上で発動していない可能性が高い**ため、まず“意図どおりの報酬が適用されているか”を最優先で確認してください。  
  根拠: `ztb/trading/environment/heavy_env/mixins/initialization.py` は常に `RewardCalculator` を生成し、`V457RewardCalculator` が参照されていません。`config/v457/base/config.yaml` の `reward_settings` 内キーの多くは `RewardSettings.from_dict` で破棄されます。
- v451の勝因は `Gamma=0.8` / Hold Penalty だけでなく、**ExecutionModel（スリッページ＋レイテンシ）やレジーム適応**が寄与していた可能性が高いです。  
  根拠: `config/v451/sac_v451_optimized.json` に `execution_model`・`advanced_market_regime` が存在。
- v450系の「Dynamic Thresholds」は**既にコードが残っており再活用が容易**です。  
  根拠: `config/v450/base/config.yaml` と `ztb/trading/environment/components/threshold_manager.py`。

---

## 1. 戦略的クリティーク

### A. `Gamma=0.8` は妥当か？
- **短期スキャル向けには合理的**です。信用割引が低いほど直近報酬を重視し、HFT的に動きやすくなります。
- ただし**過度に短期化すると“微小ノイズ反応 → 取引過多”**になりがちで、手数料・スリッページ前提では逆効果になる可能性があります。
- 推奨: `0.8 / 0.9 / 0.95` の**3点比較**で、手数料オンの一貫性を評価（取引回数と総PnLの整合性を必ずチェック）。

### B. `V457RewardCalculator` で旧Rewardを完全バイパスする判断
- **方針としては正しい**（“God Object”回避）ですが、**現状はバイパスに失敗しています**。  
  `RewardCalculator` が常時使われるため、**Hold/Position/Consistency Penalty が復活**しています。
- さらに `config/v457/base/config.yaml` の `reward_settings.type=pnl_centered` は **`RewardSettings` に存在しないため無視**されます。  
  つまり「v451風の報酬復元」は未適用の可能性が高いです。
- “最低限のPnL化”をやるなら、**V457RewardCalculatorの明示配線**か、**RewardCalculatorに `pnl_centered` を新設**し、不要なシェーピングを無効化する導線が必要です。

### C. Survivorship Bias への注意
- v451の成功は**市場レジームや期間依存**の可能性が高く、**単純な復元だけでは再現しません**。
- さらに v451 には `execution_model` / `advanced_market_regime` が含まれており、**“勝因の一部が別機能”だった可能性**があります。
- 推奨: **「v451パラメータ×別期間×手数料/スリッページ有無」**の交差検証で再現性を確認。

---

## 2. Artifact Hunting（Lost Technology）

### 1) Dynamic Thresholds（v450）
- v450で **`dynamic_threshold_mode: z_score`** が導入され、Z-Scoreで行動閾値を動的に調整していました。  
  参照: `config/v450/base/config.yaml`, `config/v450/experiments/sac_v450_zscore.json`
- 現在も `ThresholdManager` が **固定/ボラ/Z-Score** を実装し、`HeavyTradingEnv.step` で使用されています。  
  参照: `ztb/trading/environment/components/threshold_manager.py`, `ztb/trading/environment/heavy_env/core.py`
- **再導入は設定のみで可能**。v457の“取引が止まる問題”に対し、行動閾値を市場ボラ・モデル出力分布で調整できるため、効果的な再発掘候補です。

### 2) Execution Model（v451）
- v451は `execution_model` で **ATR連動スリッページ＋レイテンシ**を導入していました。  
  参照: `config/v451/sac_v451_optimized.json`, `ztb/trading/execution/realistic.py`
- v457のconfigには `execution_model` が無く、**スリッページが事実上ゼロになりやすい**。  
  これは “過大評価PnL → 本番崩壊” の主要原因になり得ます。
- **v451の `execution_model` は「復活候補の筆頭」**です。

### 3) Feature Engineering（v444）
- v444は `feature_set: v443_enhanced` ＋ **feature selection (MI top_k=120)** や **robust_scaler** を用いた「重装備型」でした。  
  参照: `config/v444/sac_v444_6_optimized_config.json`
- v457は **MFI/Lags/VWAP/Donchian/Kalman/RSI/RollingMean14/RollingStd14/ATR_simplified/OBV** の最小構成です。  
  参照: `config/v457/features.yaml`
- **Kalmanは残っています**。  
  一方で **ZScore, ROC, ReturnStdDev, Kalman_Residual_Norm** は v445〜v446で使われており、v457では外れています。  
  参照: `config/v445/sac_v445.2_aggressive_performance_optimized.json`, `config/v446/sac_v446_multitimeframe_shortterm_optimized.json`, `config/feature_sets.yaml`
- **Hurstはv444のfeatureでは確認できず**、v452のドキュメント上の提案に留まっています。  
  参照: `docs/v452/changes_v452.md`

---

## 3. タスク外の「意外な発見」（ただし重要）

1) **報酬設定が反映されていない可能性**
- `reward_settings` の多くが `RewardSettings.from_dict` で無視されます（`type`, `loss_multiplier`, `profit_multiplier`, `hold_penalty` など）。  
  参照: `ztb/trading/environment/utils/config.py`, `config/v457/base/config.yaml`
- `RewardCalculator` のデフォルト経路では Hold/Position/Consistency のペナルティが有効。  
  参照: `ztb/trading/environment/components/reward_calculator.py`

2) **v457の features.yaml が読み込まれていない可能性**
- `feature_config_path` は **トップレベル**から読み込まれますが、v457では `training.environment` に置かれています。  
  参照: `ztb/training/core/config_manager.py`, `config/v457/base/config.yaml`
- さらに `FeatureSetManager` は `feature_sets` 形式を期待するため、`config/v457/features.yaml` の `features:` 形式ではロードに失敗します。  
  参照: `ztb/features/feature_set_manager.py`, `ztb/training/core/ppo_trainer.py`

3) **`initial_balance` が `initial_portfolio_value` に変換されていない可能性**
- `EnvironmentConfig` の実体は `initial_portfolio_value` を使用します。  
  参照: `ztb/trading/environment/utils/config.py`, `ztb/trading/environment/heavy_env/core.py`
- v457では `initial_balance` のみ指定されており、**実際はデフォルト値（200,000）で動いている可能性**があります。

---

## 4. v456から「残すべき1つ」
- **マルチスケール検証＋再現性ログ（config hash含む）**  
  理由: これは“報酬を汚さずにバグ・過学習を止める”唯一の安全装置です。  
  v456の失敗要因は「良すぎるバックテストを信じたこと」だったため、**性能検証の整備だけは保持すべき**です。

---

## 5. 次のミニアクション（優先順）
1. **報酬導線の固定**  
   - v457専用の報酬が実際に使われるよう、`V457RewardCalculator` を環境に接続、または `RewardCalculator` に `pnl_centered` 経路を追加。  
   - Hold/Position/Consistency のペナルティが“本当に無効”かをログで確認。
2. **ExecutionModel の再導入**  
   - `execution_model` を v451準拠で復活させ、PnLが過大評価されないようにする。
3. **Dynamic Thresholds の再投入**  
   - `dynamic_threshold_mode=z_score` を少数ステップの検証でA/B比較。
4. **Feature 読み込み経路の修正**  
   - `feature_config_path` をトップレベルに移すか、`custom_features` で明示指定。  
   - 追加するなら **ZScore/ReturnStdDev/Kalman_Residual_Norm** を “最小追加” として検討。

