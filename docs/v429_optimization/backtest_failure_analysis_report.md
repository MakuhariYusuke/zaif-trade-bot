# バックテスト異常結果調査レポート

## 概要
2025年10月19日に実行されたバックテストで異常な結果が得られたため、原因調査と今までの作業内容をまとめたレポート。

**実行コマンド:**
```bash
python -m ztb.trading.backtest.runner --policy rl --dataset data/btc_jpy_extended_dataset.csv --initial-capital 200000
```

**異常結果:**
- Sharpe Ratio: -5413002941.819 (異常値)
- Max Drawdown: -0.00% (マイナス値)
- Total Return: 0.00%
- Win Rate: 0.0%
- Total Trades: 1

## 今までの作業履歴

### 1. SAC v428 ハイパーパラメータ最適化 (2025年10月15日頃)
- **目的**: SAC (Soft Actor-Critic) モデルのハイパーパラメータ最適化
- **手法**: Optunaを使用したベイズ最適化
- **結果**: 最適化されたパラメータでモデル訓練
- **成果物**:
  - `models/sac_v428_position_optimized.zip`
  - `models/sac_v428_ultra_profit_optimized.zip`

### 2. 報酬関数構造最適化 (2025年10月18-19日)
- **目的**: 報酬関数のパラメータをデータ駆動で最適化
- **手法**: カリキュラム学習アプローチ (4つのステージ)
- **ステージ**:
  1. **balanced_transition**: 安定性重視
  2. **trading_focused**: 取引効率重視
  3. **profit_optimized**: 利益最適化
  4. **ultra_profit**: 超利益最大化

- **最適化結果**:
  | ステージ | ベストスコア | 利益 | Win Rate | Profit Factor |
  |----------|-------------|------|----------|---------------|
  | balanced_transition | 0.0347 | -0.0025 | 56.78% | 2.9999 |
  | trading_focused | 0.0058 | 0.0055 | 46.48% | 3.5236 |
  | profit_optimized | -0.0016 | -0.0071 | 39.48% | 0.8309 |
  | ultra_profit | -0.0013 | 0.0232 | 39.02% | 0.8417 |

- **成果物**:
  - `optimization_results/reward_optimization_*_result.json`
  - `optimization_results/reward_optimization_*_report.md`
  - `optimization_results/reward_optimization_comparison_report.md`

### 3. バックテスト実行 (2025年10月19日)
- **目的**: 最適化されたRLモデルを使用したバックテスト
- **使用モデル**: `models/sac_v428_ultra_profit_optimized.zip`
- **データセット**: `data/btc_jpy_extended_dataset.csv`
- **結果**: 上記の異常結果

## 異常結果の原因分析

### 1. ログ分析
**最新ログ**: `logs/backtest_trading_20251015_202305_508132.log`
- データセット: `data/btc_jpy_featured_dataset.csv` (コマンド指定と異なる)
- データポイント: 4981
- 期待特徴量: 5
- 完了メッセージ: "Backtest completed"

**デバッグログ**: `debug_actions.log`
- モデル出力: 常に `[-1.]` (SELL)
- 離散化アクション: 常に `-1` (SELL)

### 2. コード分析

#### RLPolicyAdapterの問題点
**ファイル**: `ztb/trading/backtest/adapters.py`

1. **特徴量数の不一致**:
   ```python
   selected_features = ['close', 'returns', 'sma_5', 'sma_10', 'rsi_14', 'sma_20']  # 6 features
   expected_features = 13  # Hard-coded
   ```
   - 選択特徴量: 6個
   - 期待特徴量: 13個 (ハードコード)
   - パディングで補完されるが、不適切

2. **モデル出力の異常**:
   - 常に `[-1.]` (SELL) を出力
   - 観測値がモデルにとって意味のない値になっている可能性

3. **特徴量処理の問題**:
   ```python
   # NaN処理が不十分
   if np.isnan(obs).any():
       obs = np.nan_to_num(obs, nan=0.0)  # 単純な0埋め
   ```

#### BacktestEngineの問題点
**ファイル**: `ztb/trading/backtest/runner.py`

1. **ポジションサイジングのバグ**:
   ```python
   elif signal["action"] == "sell" and position >= 0:
       # SELLの場合のshares計算
       shares = capital / price  # capitalが0の場合、division by zero
       shares = shares if "shares" in locals() else 0  # 未定義の場合0
   ```

2. **エクイティ計算の誤り**:
   ```python
   current_equity = (
       capital if position == 0
       else capital + (position * row["close"] * (shares if "shares" in locals() else 0))
   )
   ```
   - sharesが未定義の場合0になる
   - SELL後のエクイティが常に0

3. **キャピタル管理の誤り**:
   ```python
   capital = 0  # All in (simplified) - BUY後capital=0
   ```
   - 全額投資モデルだが、SELL時のキャピタル計算が不正確

#### MetricsCalculatorの問題点
**ファイル**: `ztb/trading/backtest/metrics.py`

1. **Sharpe Ratio計算の異常**:
   ```python
   sharpe = (
       excess_returns.mean() / excess_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
   )
   ```
   - エクイティが変化しない場合、`excess_returns.std() ≈ 0`
   - Division by near-zero → 異常な値 (-5413002941.819)

### 3. データセットの問題
**指定データセット**: `data/btc_jpy_extended_dataset.csv`
**ログ読み込み**: `data/btc_jpy_featured_dataset.csv`

- データセットの不一致
- 特徴量のNaN値が多い (最初の数行全てNaN)

### 4. モデルと特徴量の不整合
- SACモデルは13次元の観測空間で訓練された可能性
- アダプタは6次元の特徴量を使用
- モデル-特徴量間の不整合

## 放置されていない問題点の完全リスト

### クリティカル問題
1. **Division by Zero**: SELL時のshares計算でcapital=0の場合
2. **未定義変数**: `shares`がSELL時に未定義
3. **エクイティ計算誤り**: position後のエクイティが0固定
4. **特徴量次元不一致**: モデル期待13 vs アダプタ提供6
5. **データセット不一致**: 指定 vs 実際の読み込みファイル

### メジャー問題
6. **NaN処理不足**: 特徴量のNaNを0で埋めるだけ
7. **モデル出力固定**: 常にSELLを出力
8. **ログ設定不備**: 最新実行のログが出力されていない
9. **ポジション管理**: ショートポジションの処理が不完全
10. **リスク管理**: 強制全額投資が現実的でない

### マイナー問題
11. **エラーハンドリング**: モデルロード失敗時のフォールバック不備
12. **デバッグ情報**: 観測値の内容がログに出ない
13. **パフォーマンス**: 毎回特徴量を再計算
14. **設定ハードコード**: expected_features = 13 がハードコード
15. **テスト不足**: 統合テストが不十分

## 推奨修正

### 即時修正
1. **ポジションサイジング修正**:
   ```python
   # SELL時のshares計算を修正
   if signal["action"] == "sell":
       shares = abs(capital) / price if capital != 0 else 0
   ```

2. **特徴量次元統一**:
   ```python
   # モデルが期待する次元を確認し統一
   expected_features = self.model.observation_space.shape[0]
   ```

3. **エクイティ計算修正**:
   ```python
   # 適切なエクイティ計算
   if position == 0:
       current_equity = capital
   else:
       current_equity = capital + (position * row["close"] * shares)
   ```

### 中期修正
4. **モデル-特徴量整合性確認**
5. **NaN処理改善**
6. **ログ出力改善**
7. **テスト追加**

### 長期修正
8. **アーキテクチャ見直し**
9. **リスク管理実装**
10. **複数モデル対応**

## 結論
バックテストの異常結果は、主に以下の根本原因による:
1. コードのバグ (ポジションサイジング、エクイティ計算)
2. モデルと特徴量の不整合
3. データセットの問題

今までの作業 (SAC最適化 + 報酬関数最適化) は成功したが、バックテスト統合部分に重大な欠陥がある。

**次のステップ**: 修正を実装し、再テストを行う。</content>
<filePath">c:\Users\Admin\dev\zaif-trade-bot\backtest_failure_analysis_report.md