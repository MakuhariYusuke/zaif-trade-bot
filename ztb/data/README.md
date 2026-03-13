# Data Management Developer Guide

This guide covers data handling, streaming pipelines, and external data sources in the `ztb/data/` module.

## Configuration Guide

## Core Configuration Classes

### ZTBConfig
Central configuration management for all system components.

```python
from ztb.utils.config import ZTBConfig

config = ZTBConfig()
mem_profile = config.get('ZTB_MEM_PROFILE', False)
cuda_warn = config.get('ZTB_CUDA_WARN_GB', 0.0)
```

### Environment Variable Categories

1. **Observability**: `ZTB_MEM_PROFILE`, `ZTB_CUDA_WARN_GB`, `ZTB_LOG_LEVEL`
2. **Training**: `ZTB_CHECKPOINT_INTERVAL`, `ZTB_MAX_MEMORY_GB`
3. **Testing**: `ZTB_TEST_ISOLATION`, `ZTB_FLOAT_TOLERANCE`

## Configuration File Examples

### YAML Configuration File

```yaml
# config/trade-config.yaml
observability:
  mem_profile: true
  cuda_warn_gb: 8.0
  log_level: "INFO"

training:
  checkpoint_interval: 1000
  max_memory_gb: 16.0

testing:
  isolation: true
  float_tolerance: 0.01
```

### JSON Configuration File

```json
{
  "observability": {
    "mem_profile": true,
    "cuda_warn_gb": 8.0,
    "log_level": "INFO"
  },
  "training": {
    "checkpoint_interval": 1000,
    "max_memory_gb": 16.0
  },
  "testing": {
    "isolation": true,
    "float_tolerance": 0.01
  }
}
```

## Environment Variables vs Configuration Files

| Aspect | Environment Variables | Configuration Files |
|--------|----------------------|-------------------|
| **Priority** | High (overrides files) | Low (fallback) |
| **Use Case** | Secrets, runtime overrides | Default settings, complex configs |
| **Format** | String only | YAML/JSON (structured) |
| **Validation** | Type conversion with fallbacks | Schema validation |
| **Examples** | `ZTB_MEM_PROFILE=1` | `mem_profile: true` |

### Usage Patterns

**Development (Environment Variables):**

```bash
export ZTB_MEM_PROFILE=1
export ZTB_CUDA_WARN_GB=4.0
export ZTB_LOG_LEVEL=DEBUG
```

**Production (Configuration File):**

```yaml
# Load via ZTB_CONFIG_FILE=/path/to/config.yaml
observability:
  mem_profile: true
  cuda_warn_gb: 8.0
  log_level: "WARN"
```

**Hybrid Approach:**

```bash
# Base config from file
export ZTB_CONFIG_FILE=config/prod.yaml
# Runtime overrides
export ZTB_LOG_LEVEL=DEBUG
```

## データ源の切替

### MarketDataSourceRegistry

`MarketDataSourceRegistry` は異なるデータソース（キャッシュ、ストリーミング、リプレイ）を統一的に管理するためのレジストリです。

#### 利用可能なデータソース

- **`cached`**: キャッシュされた価格データをファイルシステムから読み込み
- **`streaming`**: CoinGecko APIからのリアルタイムストリーミングデータ
- **`replay`**: 履歴データの再生（バックテスト用）

#### 使用例

```python
from ztb.data.marketdata_registry import create_market_data_source

# キャッシュデータソースの作成
cached_source = create_market_data_source('cached', cache_path='/path/to/cache')

# ストリーミングデータソースの作成
streaming_source = create_market_data_source('streaming',
                                           symbols=['bitcoin', 'ethereum'],
                                           buffer_capacity=500000)

# リプレイデータソースの作成
replay_source = create_market_data_source('replay',
                                        data_path='/path/to/historical_data.csv',
                                        speed_multiplier=2.0)
```

#### カスタムデータソースの登録

```python
from ztb.data.marketdata_registry import get_market_data_registry

registry = get_market_data_registry()
registry.register_factory('custom', CustomDataSourceFactory())
```

#### CLI統合

既存のCLIは挙動を変更せずに使用可能。新オプション `--source` でデータソースを指定：

```bash
# デフォルト（streaming）
python -m ztb.live.paper_trader --policy sma_fast_slow

# 明示的にstreamingを指定
python -m ztb.live.paper_trader --policy sma_fast_slow --source streaming

# replayを使用
python -m ztb.live.paper_trader --policy sma_fast_slow --source replay --data-path /path/to/data.csv
```

---

# データ処理機能 (Data Processing Features)

このセクションでは、新しく実装された金融時系列データに対する包括的なデータ処理機能について説明します。

## 主な機能

### 1. データ拡張 (Data Augmentation)
金融時系列データに対する様々な拡張手法を実装：
- **ガウスノイズ追加**: 価格変動のシミュレーション
- **Salt-and-pepperノイズ**: 極端な値のシミュレーション
- **時間軸ワーピング**: 時間的歪みのシミュレーション
- **特徴量ミキシング**: 特徴量間の相関変化のシミュレーション
- **スケーリング変換**: 値域の変化シミュレーション
- **欠損値シミュレーション**: データ欠損のシミュレーション

### 2. 異常値検出・処理 (Outlier Detection & Handling)
複数の手法による異常値検出と処理：
- **統計的手法**: Z-score, IQR, Modified Z-score
- **機械学習手法**: Isolation Forest, Local Outlier Factor
- **時系列特化手法**: STL分解, ARIMA残差分析
- **処理手法**: 除去, 補間, クリッピング, 置換

### 3. データバリデーション (Data Validation)
包括的なデータ品質チェック：
- **スキーマ検証**: データ型、範囲、必須フィールドのチェック
- **整合性チェック**: インデックス、列間関係、時系列順序の検証
- **品質メトリクス**: 完全性、正確性、一貫性、有効性の評価
- **異常検知**: 統計的特性の変化、分布シフトの検出

### 4. データ処理パイプライン (Data Processing Pipeline)
エンドツーエンドのデータ処理ワークフロー：
- 設定ファイルによる柔軟な構成
- ステップバイステップの処理実行
- 品質チェックとレポート生成
- 金融データ向け最適化設定

## 使用例

### 基本的な使用方法

```python
import pandas as pd
import numpy as np
from ztb.data import (
    DataAugmentation,
    OutlierDetector,
    OutlierHandler,
    DataValidator,
    DataProcessingPipeline
)

# サンプルデータの作成
dates = pd.date_range('2023-01-01', periods=100, freq='H')
data = pd.DataFrame({
    'price': np.random.normal(100, 5, 100),
    'volume': np.random.normal(1000, 100, 100),
    'timestamp': dates
})

# 1. データ拡張
augmenter = DataAugmentation(random_seed=42)
augmentations = [
    {"type": "gaussian_noise", "std": 0.01},
    {"type": "time_warping", "sigma": 0.1}
]
augmented_data = augmenter.apply_augmentations(data, augmentations)

# 2. 異常値検出
detector = OutlierDetector(random_seed=42)
methods = [
    {"type": "z_score", "threshold": 3.0},
    {"type": "iqr", "multiplier": 1.5}
]
data_with_outliers = detector.detect_outliers(data, methods)

# 3. 異常値処理
handler = OutlierHandler()
cleaned_data = handler.handle_outliers(
    data_with_outliers,
    method="interpolate",
    outlier_columns=["price_is_outlier", "volume_is_outlier"]
)

# 4. データバリデーション
validator = DataValidator()
schema = {
    'price': {'type': 'float', 'range': [0, 200], 'not_null': True},
    'volume': {'type': 'float', 'range': [0, 2000], 'not_null': True},
    'timestamp': {'type': 'datetime', 'not_null': True}
}
validation_result = validator.validate_data(cleaned_data, schema)

print(f"Validation passed: {validation_result.is_valid}")
print(f"Quality metrics: {validation_result.metrics}")
```

### パイプラインを使用した処理

```python
from ztb.data import create_financial_data_pipeline

# 金融データ向けパイプラインの作成
pipeline = create_financial_data_pipeline(
    augmentation_techniques=[
        {"type": "gaussian_noise", "std": 0.005},
        {"type": "time_warping", "sigma": 0.05}
    ],
    outlier_methods=[
        {"type": "z_score", "threshold": 2.5},
        {"type": "isolation_forest", "contamination": 0.05}
    ]
)

# パイプライン実行
result = pipeline.process_data(
    data,
    steps=["validation", "outlier_detection", "outlier_handling", "augmentation"]
)

print(f"Processing completed in {result.processing_stats['duration_seconds']:.2f} seconds")
print(f"Final quality: {result.quality_metrics}")
```

## 拡張手法の詳細

### データ拡張手法

| 手法 | 説明 | パラメータ |
|------|------|-----------|
| `gaussian_noise` | 正規分布ノイズの追加 | `std`: 標準偏差 |
| `salt_pepper_noise` | 極端な値のランダム置換 | `prob`: 適用確率 |
| `time_warping` | 時間軸の非線形変換 | `sigma`: 歪みの強度 |
| `feature_mixing` | 特徴量間の線形補間 | `mix_ratio`: 混合比率 |
| `scaling` | 値のスケーリング | `scale_factor`: スケール係数 |
| `missing_values` | 欠損値のシミュレーション | `missing_prob`: 欠損確率 |

### 異常値検出手法

| 手法 | 説明 | パラメータ |
|------|------|-----------|
| `z_score` | 標準偏差からの偏差 | `threshold`: 閾値 |
| `iqr` | 四分位範囲からの偏差 | `multiplier`: IQR倍率 |
| `modified_z_score` | 修正Z-score法 | `threshold`: 閾値 |
| `isolation_forest` | 孤立森アルゴリズム | `contamination`: 異常値割合 |
| `lof` | 局所異常因子 | `n_neighbors`, `contamination` |
| `stl_decomposition` | STL時系列分解 | `seasonal`, `threshold` |
| `arima_residual` | ARIMA残差分析 | `order`, `threshold` |

## テスト実行

```bash
# テストの実行
python -m pytest tests/unit/data/test_data_processing.py -v

# 特定のテストクラスの実行
python -m pytest tests/unit/data/test_data_processing.py::TestDataAugmentation -v
```

## 依存関係

- numpy
- pandas
- scipy
- scikit-learn
- statsmodels

## 注意事項

- 金融データの特性を考慮した設計（価格の非負性、時系列順序など）
- 再現性確保のための乱数シード設定
- メモリ効率を考慮した処理設計
- 包括的なエラーハンドリングとログ出力
 
 
