# Market Regime Detection Package

このパッケージは、適応型取引戦略のための様々な市場レジーム検出機能を提供します。

## 概要

市場レジーム検出は、現在の市場状況を分類し、取引戦略を適応させるために重要です。このパッケージは、基本的なレジーム検出から高度な12レジーム分類までを提供します。

## コンポーネント

### 1. BasicRegimeDetector (基本レジーム検出器)
- **ファイル**: `basic_regime_detector.py`
- **機能**: 4つの基本レジーム（bull, bear, sideways, volatile）を検出
- **使用ケース**: シンプルなレジーム適応が必要な場合
- **特徴**:
  - 価格変動パターンベースの検出
  - トレンド強度とボラティリティの評価
  - 軽量で高速

### 2. AdvancedRegimeDetector (高度レジーム検出器)
- **ファイル**: `advanced_regime_detector.py`
- **機能**: 12種類の市場レジームを検出
- **使用ケース**: SAC v445のような高度な適応戦略
- **特徴**:
  - RSI, ADX, MACDなどの技術指標を使用
  - 13種類のレジーム分類（強気/弱気トレンド、様々なボラティリティレンジ、統合局面、ブレイクセットアップ）
  - 信頼度スコアリング
  - 詳細な統計情報

### 3. TechnicalIndicators (技術指標ユーティリティ)
- **機能**: 各種技術指標の計算
- **提供指標**:
  - RSI (Relative Strength Index)
  - ADX (Average Directional Index)
  - MACD (Moving Average Convergence Divergence)
  - ボラティリティ
  - モメンタム

## レジーム分類

### AdvancedRegimeDetectorの12レジーム

| レジーム | 説明 | 特徴 |
|----------|------|------|
| `strong_bull_trend` | 強い上昇トレンド | 高モメンタム、高確信度 |
| `moderate_bull_trend` | 中程度の上昇トレンド | 安定した上昇 |
| `weak_bull_trend` | 弱い上昇トレンド | 低いモメンタム |
| `strong_bear_trend` | 強い下降トレンド | 高モメンタム、高確信度 |
| `moderate_bear_trend` | 中程度の下降トレンド | 安定した下降 |
| `weak_bear_trend` | 弱い下降トレンド | 低いモメンタム |
| `high_volatility_ranging` | 高ボラティリティ保ち合い | 激しい値動き |
| `moderate_volatility_ranging` | 中ボラティリティ保ち合い | 適度な変動 |
| `low_volatility_ranging` | 低ボラティリティ保ち合い | 安定した狭いレンジ |
| `extreme_volatility` | 極端なボラティリティ | 異常な変動 |
| `consolidation` | 統合局面 | 均衡状態 |
| `breakout_setup` | ブレイクアウト形成 | 準備段階 |
| `breakdown_setup` | ブレークダウン形成 | 準備段階 |

## 使用方法

### BasicRegimeDetector

```python
from ztb.analysis.regime import MarketRegimeDetector

detector = MarketRegimeDetector()
regime = detector.detect_regime(current_price=100.0, step=1)
print(f"Current regime: {regime}")  # 'bull', 'bear', 'sideways', 'volatile'
```

### AdvancedRegimeDetector

```python
from ztb.analysis.regime import AdvancedRegimeDetector, MarketRegime

detector = AdvancedRegimeDetector()

# 価格データを更新
for i in range(60):
    price = 100.0 + i * 0.5
    detector.update_price_data(price, price + 1, price - 1)

# レジーム検出
result = detector.detect_regime()
print(f"Regime: {result.regime.value}")
print(f"Confidence: {result.confidence}")
print(f"Indicators: {result.indicators}")

# 統計情報取得
stats = detector.get_regime_statistics()
print(f"Total detections: {stats['total_detections']}")
```

## テスト

テストは `tests/unit/analysis/regime/` ディレクトリに配置されています。

```bash
# すべてのレジーム関連テストを実行
pytest tests/unit/analysis/regime/

# 特定のテストを実行
pytest tests/unit/analysis/regime/test_advanced_regime_detector.py
pytest tests/unit/analysis/regime/test_basic_regime_detector.py
```

## 設定

### BasicRegimeDetector設定
- `regime_detection_window`: 検出ウィンドウサイズ（デフォルト: 20）
- `adaptation_frequency`: 適応頻度（デフォルト: 10）
- `high_volatility_threshold`: 高ボラティリティ閾値（デフォルト: 0.02）
- `low_volatility_threshold`: 低ボラティリティ閾値（デフォルト: 0.005）
- `trend_strength_threshold`: トレンド強度閾値（デフォルト: 0.001）

### AdvancedRegimeDetector設定
- `detection_window`: 検出ウィンドウサイズ（デフォルト: 50）
- `adaptation_frequency`: 適応頻度（デフォルト: 10）

## 統合

このパッケージは以下のコンポーネントと統合されています：

- **RewardCalculator**: 報酬計算におけるレジーム適応
- **SAC v445**: 高度なレジーム適応機能
- **Training Environment**: 環境設定でのレジームパラメータ

## 開発ノート

- **パフォーマンス**: AdvancedRegimeDetectorは計算コストが高いため、適度な使用を推奨
- **互換性**: BasicRegimeDetectorは既存の環境との後方互換性を維持
- **拡張性**: 新しいレジームタイプや指標の追加が容易

## 関連ドキュメント

- `SAC_v445_ADVANCED_REGIME_ADAPTATION.md`: 詳細な技術仕様
- `docs/features/regime_adaptation/`: 機能別ドキュメント