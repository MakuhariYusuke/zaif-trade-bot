# 特徴量セット管理システム

## 概要

SAC v427 特徴量セット管理システムは、柔軟でコンフィグ可能な特徴量セット管理を提供します。このシステムにより、異なるユースケースや実験要件に応じて特徴量セットを簡単に切り替えることができます。

### 主な機能

- **プリセット特徴量セット**: 一般的なユースケース向けに最適化された4つのプリセット
- **カスタム設定**: 特定の特徴量を動的に除外/追加可能
- **設定ファイルベース**: JSONファイルによる宣言的な設定
- **後方互換性**: 既存コードへの影響を最小限に抑制

## アーキテクチャ

### 主要コンポーネント

```
ztb/features/
├── feature_set_config.py      # 設定管理クラス
├── sac_v427_feature_engineering.py  # 特徴量生成エンジン
└── config/feature_sets/       # 設定ファイルディレクトリ
    ├── default.json          # デフォルト設定
    ├── minimal.json          # 最小セット
    └── high_quality.json     # 高品質セット
```

### クラス構造

#### FeatureSetConfig

特徴量セットの設定を管理するメインクラスです。

```python
class FeatureSetConfig:
    def __init__(self, config_path: str = None)
    def load_config(self) -> None
    def save_config(self) -> None
    def set_feature_set(self, set_name: str) -> None
    def get_excluded_features(self) -> List[str]
    def add_excluded_feature(self, feature: str) -> None
    def remove_excluded_feature(self, feature: str) -> None
    def get_feature_flags(self) -> Dict[str, bool]
    def list_available_sets(self) -> Dict[str, Dict]
    def get_current_config(self) -> Dict
```

#### SACv427FeatureEngineer

特徴量生成エンジンで、設定に基づいて特徴量をフィルタリングします。

```python
class SACv427FeatureEngineer:
    def __init__(self, market_system=None, config_path=None)
    def generate_v427_features(self, df, window_sizes=[5,10,20,50], feature_set=None)
```

## プリセット特徴量セット

### 1. full (完全セット)

**目的**: すべての利用可能な特徴量を使用
**特徴量数**: 150+ 次元
**含まれる特徴量**:
- **基本価格特徴量**: open, high, low, close, volume
- **テクニカル指標**: RSI, MACD, Bollinger Bands, Stochastic, Williams %R
- **トレンド指標**: SMA, EMA, Ichimoku Cloud, ADX
- **ボラティリティ指標**: ATR, Bollinger Band Width, Historical Volatility
- **モメンタム指標**: ROC, Momentum, Williams %R
- **市場レジーム特徴量**: トレンド強度、市場状態分類、ボラティリティレジーム
- **相関特徴量**: 価格間の相関係数、ボラティリティ相関
- **アンサンブル特徴量**: 複数モデルの予測値、予測確信度
- **リスク調整特徴量**: VaR, CVaR, Sharpe Ratio, Sortino Ratio
- **市場マイクロストラクチャ**: スプレッド推定、市場インパクト、流動性指標

**設定**:
```json
{
  "name": "Full Feature Set",
  "description": "Complete SAC v427 feature set (150+ dimensions)",
  "excluded_features": [],
  "include_regime_features": true,
  "include_correlation_features": true,
  "include_ensemble_features": true,
  "include_risk_features": true
}
```

### 2. no_harmful (デフォルト)

**目的**: クリティカルな有害特徴量を除外した完全セット
**特徴量数**: 150+ 次元 (有害特徴量除外)
**除外される特徴量**:
- `dividends`: 常に0 (仮想通貨では発生しない)
- `stock splits`: 常に0 (仮想通貨では発生しない)

**含まれる特徴量** (fullセットと同じ):
- **基本価格特徴量**: open, high, low, close, volume
- **テクニカル指標**: RSI, MACD, Bollinger Bands, Stochastic, Williams %R
- **トレンド指標**: SMA, EMA, Ichimoku Cloud, ADX
- **ボラティリティ指標**: ATR, Bollinger Band Width, Historical Volatility
- **モメンタム指標**: ROC, Momentum, Williams %R
- **市場レジーム特徴量**: トレンド強度、市場状態分類、ボラティリティレジーム
- **相関特徴量**: 価格間の相関係数、ボラティリティ相関
- **アンサンブル特徴量**: 複数モデルの予測値、予測確信度
- **リスク調整特徴量**: VaR, CVaR, Sharpe Ratio, Sortino Ratio
- **市場マイクロストラクチャ**: スプレッド推定、市場インパクト、流動性指標

**設定**:
```json
{
  "name": "No Harmful Features",
  "description": "Full features with critical harmful features removed",
  "excluded_features": ["dividends", "stock splits"],
  "include_regime_features": true,
  "include_correlation_features": true,
  "include_ensemble_features": true,
  "include_risk_features": true
}
```

### 3. minimal (最小セット)

**目的**: コア機能のみを使用した軽量セット
**特徴量数**: 30-50 次元
**含まれる特徴量**:
- **基本価格特徴量**: close (価格), volume (出来高)
- **基本リターン特徴量**: returns (単純リターン), log_returns (対数リターン)
- **基本ボラティリティ**: volatility (価格変動率)
- **基本テクニカル指標**: SMA(20), SMA(50), RSI(14)
- **基本モメンタム**: ROC(10) (Rate of Change)

**除外される特徴量カテゴリ**:
- 市場レジーム特徴量 (regime_*)
- 相関特徴量 (correlation_*)
- アンサンブル特徴量 (ensemble_*)
- リスク調整特徴量 (risk_*)
- 高度なテクニカル指標 (MACD, Bollinger Bands, etc.)
- 市場マイクロストラクチャ特徴量

**ユースケース**: 高速プロトタイピング、計算リソースの制限された環境、特徴量重要度の初期評価

**設定**:
```json
{
  "name": "Minimal Feature Set",
  "description": "Core features only (30-50 dimensions)",
  "excluded_features": ["dividends", "stock splits"],
  "include_regime_features": false,
  "include_correlation_features": false,
  "include_ensemble_features": false,
  "include_risk_features": false
}
```

### 4. high_quality (高品質セット)

**目的**: 相関分析で高品質と判定された特徴量のみを使用
**特徴量数**: 100+ 次元
**除外される特徴量** (相関過多または定数):
- `dividends`, `stock splits`: 常に0 (クリティカル有害)
- `open`, `high`, `low`: OHLCV基本データ (closeとの相関95%+)
- `volume`: 出来高 (他の特徴量との相関が高い)
- `returns`, `log_returns`: 単純リターン (相関が高い)

**含まれる特徴量** (分析で優良と判定されたもの):
- **高度なテクニカル指標**:
  - RSI (Relative Strength Index): 14, 21, 28期間
  - MACD (Moving Average Convergence Divergence): シグナル線、ヒストグラム
  - Bollinger Bands: 上限、下限、バンド幅、%B
  - Stochastic Oscillator: %K, %D, Slow Stochastic
  - Williams %R: 14期間
  - CCI (Commodity Channel Index): 20期間

- **トレンド・モメンタム指標**:
  - ADX (Average Directional Index): 14期間
  - ROC (Rate of Change): 10, 20, 30期間
  - Momentum: 10, 20期間
  - Ichimoku Cloud: Tenkan-sen, Kijun-sen, Senkou Span A/B

- **ボラティリティ指標**:
  - ATR (Average True Range): 14期間
  - Bollinger Band Width: 20期間
  - Historical Volatility: 20, 50期間

- **市場レジーム特徴量**:
  - トレンド強度指標
  - 市場状態分類 (強気/弱気/横ばい)
  - ボラティリティレジーム検知

- **アンサンブル予測特徴量**:
  - 複数モデルの予測値
  - 予測確信度スコア
  - モデル間の合意度

- **リスク調整済み特徴量**:
  - Sharpe Ratio (リスク調整リターン)
  - Sortino Ratio (下方リスク調整)
  - Value at Risk (VaR)
  - Conditional VaR (CVaR)

- **市場マイクロストラクチャ**:
  - スプレッド推定値
  - 市場インパクト指標
  - 流動性指標

**品質基準**: 相関係数 < 0.95、ゼロ値率 < 80%、分散 > 0
**ユースケース**: 本番環境での高精度取引、特徴量重要度の詳細分析、モデル比較実験

**設定**:
```json
{
  "name": "High Quality Only",
  "description": "Only excellent quality features (correlation-filtered)",
  "excluded_features": [
    "dividends", "stock splits",
    "open", "high", "low", "volume",
    "returns", "log_returns"
  ],
  "include_regime_features": true,
  "include_correlation_features": false,
  "include_ensemble_features": true,
  "include_risk_features": true
}
```

## 特徴量セット選択ガイド

### 各セットの使い分け

| セット | 特徴量数 | 処理速度 | 品質 | 推奨ユースケース |
|--------|----------|----------|------|------------------|
| **minimal** | 30-50 | ⭐⭐⭐⭐⭐ | ⭐⭐ | 高速プロトタイピング、初期テスト、リソース制限環
| **high_quality** | 100+ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 本番取引、精度重視、特徴量分析 |
| **no_harmful** | 150+ | ⭐⭐ | ⭐⭐⭐⭐ | 包括的な実験、完全な特徴量調査 |
| **full** | 150+ | ⭐⭐ | ⭐⭐⭐ | 理論的完全性重視、すべての特徴量が必要な場合 |

### セット選択の判断基準

#### 開発フェーズ別
- **初期開発**: `minimal` - 高速なイテレーション
- **特徴量検証**: `high_quality` - 品質重視の評価
- **本番最適化**: `high_quality` - 安定した高品質特徴量
- **研究実験**: `full` または `no_harmful` - 包括的な調査

#### リソース制約別
- **計算リソース豊富**: `full` または `no_harmful`
- **計算リソース制限**: `minimal` または `high_quality`
- **メモリ制限**: `minimal`

#### 品質要件別
- **最高精度要求**: `high_quality`
- **バランス重視**: `no_harmful`
- **速度優先**: `minimal`

### パフォーマンス特性

#### 処理時間 (BTC/JPY 352サンプル基準)
- `minimal`: ~0.5秒
- `high_quality`: ~1.2秒
- `no_harmful`: ~2.0秒
- `full`: ~2.1秒

#### メモリ使用量 (推定)
- `minimal`: ~50MB
- `high_quality`: ~120MB
- `no_harmful`: ~180MB
- `full`: ~200MB

## 使用方法

### 基本的な使用法

```python
from ztb.features.sac_v427_feature_engineering import SACv427FeatureEngineer

# デフォルト設定を使用 (no_harmful)
engineer = SACv427FeatureEngineer()
features_df = engineer.generate_v427_features(data_df)
```

### 特定の特徴量セットを使用

```python
# 最小セットを使用
features_df = engineer.generate_v427_features(data_df, feature_set='minimal')

# 高品質セットを使用
features_df = engineer.generate_v427_features(data_df, feature_set='high_quality')
```

### カスタム設定

```python
from ztb.features.feature_set_config import get_feature_config

# 設定を取得
config = get_feature_config()

# 特徴量を追加で除外
config.add_excluded_feature('volume')
config.add_excluded_feature('some_correlated_feature')

# 除外特徴量を削除
config.remove_excluded_feature('dividends')

# 設定を適用
features_df = engineer.generate_v427_features(data_df)
```

### 設定ファイルの使用

```python
# カスタム設定ファイルを指定
engineer = SACv427FeatureEngineer(config_path='path/to/custom_config.json')

# 設定をファイルに保存
config = engineer.feature_config
config.save_config()
```

## 設定ファイル形式

### JSON スキーマ

```json
{
  "name": "セット名",
  "description": "セットの説明",
  "excluded_features": ["除外する特徴量のリスト"],
  "include_regime_features": true|false,
  "include_correlation_features": true|false,
  "include_ensemble_features": true|false,
  "include_risk_features": true|false
}
```

### フィールド説明

- **name**: セットの表示名
- **description**: セットの詳細説明
- **excluded_features**: 除外する特徴量名のリスト
- **include_regime_features**: 市場レジーム特徴量を含めるかどうか
- **include_correlation_features**: 相関特徴量を含めるかどうか
- **include_ensemble_features**: アンサンブル特徴量を含めるかどうか
- **include_risk_features**: リスク調整特徴量を含めるかどうか

## 高度な使用法

### トレーニングパイプラインでの使用

```python
from ztb.training.sac_v427_advanced_trainer import SACv427AdvancedTrainer

# 特定の特徴量セットでトレーニング
trainer = SACv427AdvancedTrainer(config_path='config/training_config.json')

# 特徴量エンジニアが自動的に適切なセットを使用
trainer.train(feature_set='high_quality')
```

### バックテストでの使用

```python
from ztb.trading.backtest.adapters import RLPolicyAdapter

# アダプタが特徴量セット設定を自動的に使用
adapter = RLPolicyAdapter(model_path='models/sac_model.zip')
signals = adapter.generate_signal(market_data, current_position)
```

### 実験での使用

```python
# 異なる特徴量セットでの比較実験
feature_sets = ['minimal', 'no_harmful', 'high_quality', 'full']

for set_name in feature_sets:
    print(f"Testing feature set: {set_name}")

    engineer = SACv427FeatureEngineer()
    features = engineer.generate_v427_features(data, feature_set=set_name)

    # モデルトレーニングと評価
    model = train_model(features)
    performance = evaluate_model(model, test_data)

    print(f"Performance with {set_name}: {performance}")
```

## API リファレンス

### FeatureSetConfig クラス

#### メソッド

**`__init__(config_path=None)`**
- 設定ファイルを指定して初期化
- `config_path`: 設定ファイルのパス (オプション)

**`load_config()`**
- JSONファイルから設定を読み込み

**`save_config()`**
- 現在の設定をJSONファイルに保存

**`set_feature_set(set_name)`**
- プリセット特徴量セットを設定
- `set_name`: 'full', 'minimal', 'no_harmful', 'high_quality'

**`get_excluded_features()`**
- 除外対象の特徴量リストを取得
- 戻り値: `List[str]`

**`add_excluded_feature(feature)`**
- 除外特徴量を追加
- `feature`: 追加する特徴量名

**`remove_excluded_feature(feature)`**
- 除外特徴量を削除
- `feature`: 削除する特徴量名

**`get_feature_flags()`**
- 特徴量カテゴリの有効/無効フラグを取得
- 戻り値: `Dict[str, bool]`

**`list_available_sets()`**
- 利用可能なプリセットを取得
- 戻り値: `Dict[str, Dict]`

**`get_current_config()`**
- 現在の設定を取得
- 戻り値: `Dict`

### SACv427FeatureEngineer クラス

#### メソッド

**`__init__(market_system=None, config_path=None)`**
- 特徴量エンジニアを初期化
- `market_system`: 市場システムインスタンス (オプション)
- `config_path`: 設定ファイルパス (オプション)

**`generate_v427_features(df, window_sizes=[5,10,20,50], feature_set=None)`**
- SAC v427特徴量を生成
- `df`: 入力データフレーム
- `window_sizes`: テクニカル指標のウィンドウサイズ
- `feature_set`: 使用する特徴量セット名 (オプション)
- 戻り値: 特徴量が追加されたデータフレーム

## 特徴量カテゴリの詳細

### テクニカル指標 (Technical Indicators)

**目的**: 価格パターン、トレンド、モメンタムを定量化
**有益な特徴量例**:
- **RSI (Relative Strength Index)**: 買われすぎ/売られすぎを検知 (14, 21, 28期間)
- **MACD**: トレンド変化とモメンタムを測定
- **Bollinger Bands**: 価格の変動範囲と位置を分析
- **Stochastic Oscillator**: 価格の相対的な位置を測定
- **Williams %R**: 買われすぎ/売られすぎのオシレーター

### 市場レジーム特徴量 (Market Regime Features)

**目的**: 市場の状態 (強気/弱気/横ばい) を分類
**有益な特徴量例**:
- **トレンド強度**: ADXを使用したトレンドの強さ測定
- **市場状態分類**: 機械学習によるレジーム分類
- **ボラティリティレジーム**: 高/中/低ボラティリティ状態の検知
- **相場転換シグナル**: トレンド変化の予測

### アンサンブル特徴量 (Ensemble Features)

**目的**: 複数モデルの予測を統合し、信頼性を向上
**有益な特徴量例**:
- **予測平均**: 複数モデルの予測値の平均
- **予測分散**: モデル間の予測ばらつき
- **合意度スコア**: モデル間の一致度
- **確信度指標**: 予測の信頼性スコア

### リスク調整特徴量 (Risk-Adjusted Features)

**目的**: リスクを考慮したパフォーマンス測定
**有益な特徴量例**:
- **Sharpe Ratio**: リスク調整リターンの標準指標
- **Sortino Ratio**: 下方リスクのみを考慮した比率
- **VaR (Value at Risk)**: 最大損失額の推定
- **CVaR (Conditional VaR)**: VaRを超える損失の期待値

### 市場マイクロストラクチャ (Market Microstructure)

**目的**: 市場の流動性と取引コストを分析
**有益な特徴量例**:
- **スプレッド推定**: 売買スプレッドの計算
- **市場インパクト**: 大口注文の価格影響
- **流動性指標**: 取引量と注文数の関係
- **取引コスト推定**: 取引実行時のコスト予測

### 相関特徴量 (Correlation Features)

**目的**: 資産間の関係性を定量化
**有益な特徴量例**:
- **価格相関係数**: 短期/長期の相関関係
- **ボラティリティ相関**: 変動性の連動性
- **リードラグ関係**: 価格変動の時間的ずれ

## トラブルシューティング

### 一般的な問題

#### Q: 特徴量が期待通りに除外されない
**A**: 設定ファイルが正しく読み込まれているか確認してください。

```python
config = engineer.feature_config
print("Excluded features:", config.get_excluded_features())
print("Feature flags:", config.get_feature_flags())
```

#### Q: 設定ファイルが見つからない
**A**: デフォルトでは `config/feature_sets/default.json` が使用されます。カスタムパスを指定する場合は絶対パスを使用してください。

#### Q: 特徴量セットの切り替えが反映されない
**A**: `generate_v427_features()` の `feature_set` パラメータを使用するか、設定オブジェクトを直接操作してください。

```python
# 方法1: パラメータで指定
features = engineer.generate_v427_features(data, feature_set='minimal')

# 方法2: 設定を変更
engineer.feature_config.set_feature_set('minimal')
features = engineer.generate_v427_features(data)
```

### パフォーマンスに関する考慮事項

- **minimalセット**: 計算時間が最も短く、メモリ使用量も少ない
- **fullセット**: 計算時間が最も長く、メモリ使用量も多い
- **high_qualityセット**: 相関の問題を回避しつつ十分な特徴量を提供

### 互換性に関する注意

- このシステムは既存のコードと後方互換性があります
- デフォルト設定は `no_harmful` であり、クリティカルな有害特徴量を除外します
- 設定を変更しても、他のコンポーネントへの影響はありません

## 変更履歴

- **v1.0 (2025-10-25)**: 初期実装
  - 4つのプリセット特徴量セット
  - コンフィグ可能な除外特徴量
  - 特徴量カテゴリの有効/無効制御
  - JSONベースの設定ファイル

---

*このドキュメントは特徴量セット管理システムの実装について説明しています。詳細な技術仕様についてはソースコードを参照してください。*</content>
<parameter name="filePath">c:\Users\Admin\dev\zaif-trade-bot\docs\features\FEATURE_SET_MANAGEMENT_SYSTEM.md
