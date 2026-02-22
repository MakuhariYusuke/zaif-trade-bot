# v457 初期検証ログへの多角的レビューと提言

## 1. まず厳しめの指摘（根本的な評価の前提）
- **学習と評価が「実データ特徴量」前提になっていない（改善はしたが未完）**  
  `scripts/v457/train.py` はベース特徴量を計算するよう修正済みですが、ADX/DI がランダム値のため実運用に近い学習とは言えません。  
  参照: `scripts/v457/train.py`
- **Config反映は一部改善したが、重要項目が無効化されている**  
  SACパラメータは YAML から読み込まれるが、FastIntradayEnvは `execution_model` / `dynamic_threshold_mode` / `reward_settings` を実際には利用しません。  
  参照: `scripts/v457/train.py`, `config/v457/base/config.yaml`, `ztb/trading/environment/fast_intraday_env_v456.py`
- **バックテストの特徴量が一部ランダム**  
  `scripts/v457/backtest.py` の `calculate_base_features` で ADX/DI をランダム生成しており、再現性はあっても正当性がありません。  
  参照: `scripts/v457/backtest.py`
- **backtest.py が `env.df` を参照しているが、FastIntradayEnv は `df` を破棄する設計**  
  実行時に `NoneType` 参照となる可能性があり、ログ取得自体が不安定。  
  参照: `scripts/v457/backtest.py`, `ztb/trading/environment/fast_intraday_env_v456.py`
- **ダミーデータの時間粒度が5分で、configの1分前提とズレる**  
  取引頻度や特徴量スケールがずれるため、ダミー検証でも歪みが出る。  
  参照: `scripts/v457/train.py`, `config/v457/base/config.yaml`

## 2. 初期検証ログの評価（良い点 / 注意点）
### 良い点
- 10kステップの**パイプライン耐久性**は確認できています（クラッシュ回避）。
- バックテストで **PnLリセットバグが出ていない** という安定性は前進。

### 注意点（評価の根拠が弱い）
- **ダミーデータ学習の収益は意味が薄い**（擬似ノイズで学習したモデルが偶然当たっただけの可能性が高い）。
- **行動分布の解釈が難しい**  
  85% SELLは相場崩壊期間の結果として妥当だが、「ロングオンリー bug」の影響が大きく、行動バイアス評価としては不成立。

## 3. 過去資料から取り込むべき成果（具体）
### 3.1 Dynamic Thresholds（v450）
取引停止/過剰取引の両方に効く資産。v457 configで再導入済だが、**FastIntradayEnvでは使われない**。  
参照: `config/v450/base/config.yaml`, `ztb/trading/environment/components/threshold_manager.py`

### 3.2 v453 Hybrid Filters（Vol/Regime Gate）
`v453_hybrid_v3` のリターンは高く、**「学習モデル＋ルールゲート」**の実用性が高い。  
参照: `config/v453/hybrid_config_v3.json`, `backtest_results/v453_hybrid_v3/backtest_results.json`

### 3.3 Execution Model（v451, v455）
実運用の過大評価を防ぐ重要資産。FastIntradayEnvでは反映されないため、**HeavyTradingEnvを使う構成へ戻すべき**。  
参照: `config/v451/sac_v451_optimized.json`, `ztb/trading/execution/realistic.py`, `ztb/trading/execution/pseudo_hft.py`

### 3.4 特徴量の“最小追加”候補
v457の最小セットに **ZScore / ReturnStdDev / Kalman_Residual_Norm** を追加するだけでもエッジが戻る可能性。  
参照: `config/v445/sac_v445.2_aggressive_performance_optimized.json`, `config/feature_sets.yaml`

## 4. 多角的な批判（視点別）

### Data/Feature 視点
- `data/datasets/btc_jpy_real_dataset.csv` は **101行しかなく訓練不能**。  
  参照: `data/datasets/btc_jpy_real_dataset.csv`
- `scripts/v457/backtest.py` のADx/DIがランダム値で、**再現性はあるが意味がない**。
- FastIntradayEnv の cyclical/global 特徴量は **常に0で実質無効**。  
  参照: `ztb/trading/environment/fast_intraday_env_v456.py`
- `read_csv(..., index_col=0)` により `timestamp` が列として消えるため、HeavyTradingEnv系で MTF 特徴量が無効化されやすい。  
  参照: `scripts/v457/train.py`, `scripts/v457/backtest.py`

### Reward/Cost 視点
- FastIntradayEnv の報酬は `reward/100` で **[-0.1, 0.1] にクリップ**されるため、PnL変化が学習に反映されにくい。  
  参照: `ztb/trading/environment/fast_intraday_env_v456.py`
- ExecutionModel未使用の訓練/評価は **「紙上の勝ち」**に陥る。

### Evaluation 視点
- 2,000ステップのバックテストは統計的に弱く、特に **手数料影響が不安定**。
- `analyze_potential.py` は理想上限として有用だが、**コスト・制約・スリッページがゼロ**で上限が過大。

### Process/Debug 視点（タスク外改善）
- `scripts/v456/verify_fixes_v2.py` の **config/報酬/特徴量検証**を再利用すべき。
- `scripts/v456/diagnose_env.py` の環境診断もv457に転用すると初期トラブルが減る。
- `scripts/v457/dry_run.py` では V457RewardCalculator の稼働判定をしているが、現状の wiring では失敗する可能性が高い。  
  参照: `scripts/v457/dry_run.py`, `ztb/trading/environment/heavy_env/mixins/initialization.py`

## 5. 改善提案（優先順）
1. **v457の訓練環境を HeavyTradingEnv に寄せる**  
   execution_model / reward_settings / dynamic_threshold を「実際に動く」状態に。
2. **特徴量の正規化と計算の一本化**  
   `scripts/v456/feature_calculator_v456.py` を使い、学習/評価で同一の特徴量計算に統一。
3. **SACハイパーパラメータを YAML から読み込み**  
   10k実験の再現性と検証の正当性を確保。
4. **v453 Hybrid Filters をA/Bで復活**  
   既存バックテストで高いリターンが出ているため、**学習モデルの癖を補正する短期解**として有効。
5. **評価の「費用込み上限」を再計算**  
   `analyze_potential.py` に手数料・スリッページを加味し、現実的な上限を把握。

## 6. まとめ
現状は「パイプラインの安定性」確認に成功していますが、**学習・評価の前提が実データの特徴量/報酬設計と一致していない**ため、収益性の議論にはまだ早い段階です。  
「v457の思想（PnL中心・ExecutionModel・Dynamic Threshold）」を**実際に使う環境に戻すことが最優先**です。

## 7. 具体的な修正案（追記）
1. **`scripts/v457/train.py` の特徴量算出を“実データ準拠”へ置換**  
   - ADX/DI のランダム生成を廃止し、`scripts/v456/feature_calculator_v456.py` の計算ロジックを流用。  
   - 目的: “ダミーでも動く”から “実データで妥当” へ移行。
2. **`scripts/v457/backtest.py` のログ取得を安定化**  
   - `env.df` 参照をやめ、`df`（前処理済み）か `env.close_prices` から価格を取得。  
   - 目的: `NoneType`参照事故を防止し、ログ整合性を確保。
3. **ダミーデータ粒度をconfigに合わせる**  
   - `training.environment.timeframe` から頻度を算出して `date_range(freq=...)` を生成。  
   - 目的: 学習スケールの歪みを削減。
4. **YAMLの環境設定を実環境に反映させる導線を明確化**  
   - FastIntradayEnv経由で v457の `execution_model` / `dynamic_threshold_mode` / `reward_settings` が反映されないので、  
     HeavyTradingEnv経由に切り替えるか、FastIntradayEnv側でサポートする。  
   - 目的: v457の本質（PnL reset）を学習経路に乗せる。
5. **`scripts/v457/dry_run.py` を v457仕様の真偽判定に使えるよう修正**  
   - `reward_settings.custom_reward_params["type"] == "pnl_centered"` をトリガに V457RewardCalculator が使われるように wiring。  
   - 目的: “PnL reset が有効か”を一発で確認できるテストにする。

## 8. 追加で見つかった改善点（もう一歩）
- **v456の「ランダム特徴量撤廃」を完全復活**  
  v456では「ランダム特徴量撤廃」がP0修正事項。v457も同等の厳しさで適用すべき。  
  参照: `docs/v456/59_V456_FINAL_RETROSPECTIVE.md`
- **“良すぎる結果は疑う”チェックを強制化**  
  v456で「勝率63%→0.3%の実勝率」問題が発生。v457では **取引件数とPnLの整合性チェック**を必須にする。  
  参照: `docs/v456/59_V456_FINAL_RETROSPECTIVE.md`, `docs/v456/58_BACKTEST_FINAL_SUMMARY_20260115.md`
- **timestamp列の保持とタイムゾーン統一**  
  HeavyTradingEnv/MTF系の機能を使う場合、`timestamp`を列として保持し、tz-aware化を統一する。  
  参照: `ztb/features/time/cyclical_v456.py`
- **学習の再現性強化（seed固定）**  
  `np.random.seed` / `torch.manual_seed` / `SAC(seed=...)` で再現性を担保。  
  目的: “挙動の良し悪し”を正しく比較できる状態にする。
