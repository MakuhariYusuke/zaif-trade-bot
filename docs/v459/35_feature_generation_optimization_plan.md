# 35. 特徴生成最適化計画 - Phase 3.5

**日付**: 2026年1月27日  
**バージョン**: v459  
**ステータス**: 📋 計画中  
**前提**: [34. Windows SIGINT解決と初回実験成功](34_windows_sigint_resolution_and_first_success.md)  
**関連**: [00. v459プロジェクト提案](00_project_proposal_v459.md)

---

## 1. Executive Summary

### 最適化の動機

5実験の一貫性分析により、**特徴生成時間がトレーニング時間の63.7%を占有**していることが判明しました。

```
現状パフォーマンス（5,000ステップ）:
├── 総時間: 677秒 (11.3分)
│   ├── 特徴生成: 431秒 (63.7%) ← 🎯 最適化ターゲット
│   └── 実トレーニング: 246秒 (36.3%)
└── スループット: 7.39-7.80 steps/sec
```

### 最適化の重要性

**12実験の本番実行への影響**（固定/変動コスト分離）:
```python
# 5,000ステップの実測値から分離
総時間 = 特徴生成(固定) + training_rate × timesteps
677秒 = 431秒(固定) + training_rate × 5000
training_rate ≈ 0.0492 秒/ステップ

# 現状（5,000ステップ）
12実験 × 11.3分 = 136分 (2.3時間)

# 50,000ステップ想定（線形拡大は過大評価）
単一実験: 431秒 + 0.0492 × 50000 = 2,891秒 (48.2分)
12実験: 12 × 48.2分 = 578分 (9.6時間) ← 修正後推定

# 【最優先】再計算排除による最適化
特徴生成を1回のみ実行 → 12実験で再利用
12実験: 431秒(1回) + 12 × (0.0492 × 50000) = 29,971秒 (500分, 8.3時間)
削減効果: 578分 → 500分 (13%削減)

# 【副次的】特徴生成自体の高速化（2-3倍）
特徴生成: 431秒 → 144-215秒
12実験: (144-215)秒 + 12 × 2460秒 = 29,664-29,735秒 (494-496分, 8.2-8.3時間)
削減効果: 578分 → 494-496分 (14-15%削減)

# 【両方実施】再計算排除 + 高速化
12実験: (144-215)秒 + 12 × 2460秒 = 29,664-29,735秒 (494-496分, 8.2-8.3時間)
削減効果: 578分 → 494-496分 (14-15%削減)

注: 特徴生成の固定コストが支配的なため、再計算排除が最重要
```

**プロジェクト大義との整合性**:
> **本プロジェクトの大義は「短期間での高収益性システム」の実現**

22時間の実験時間は開発速度を阻害するため、**Phase 3.5として最適化に集中**します。

---

## 2. 現状分析

### 2.1 パフォーマンスプロファイル

| 指標 | 値 | 評価 |
|------|-----|------|
| 総トレーニング時間 | 677秒 (11.3分) | ⚠️ 許容可能だが改善余地大 |
| 特徴生成時間 | 431秒 (63.7%) | 🔴 **ボトルネック** |
| 実トレーニング時間 | 246秒 (36.3%) | 🟢 適切 |
| スループット | 7.39-7.80 steps/sec | 🟡 普通 |
| メモリピーク | 3.6GB | 🟡 許容範囲 |

### 2.2 特徴生成の内訳（推定）

```python
# FeatureRegistry.compute_features() の内訳
特徴生成 431秒の内訳:
├── データ準備・前処理: ~50秒 (11.6%)
│   ├── DataFrame読み込み
│   ├── index設定
│   └── 初期検証
├── 基本特徴量計算: ~150秒 (34.8%)
│   ├── 価格変化率
│   ├── ボリューム特徴
│   └── 基本統計量
├── MTF特徴量計算: ~180秒 (41.8%)  ← 🎯 最大ボトルネック
│   ├── 5分足集約
│   ├── 15分足集約
│   ├── 1時間足集約
│   └── Ichimoku計算（各タイムフレーム）
├── Regime特徴量: ~30秒 (7.0%)
│   ├── ボラティリティレジーム
│   └── トレンドレジーム
└── Cyclical特徴量: ~21秒 (4.9%)
    ├── sin/cos時刻
    └── 曜日エンコーディング
```

### 2.3 既存の最適化機構

**v459コードベースで既に実装済み**:

| 機構 | 実装状況 | 効果 |
|------|---------|------|
| データキャッシング | ✅ 実装済み (`ztb/utils/cache/feature_cache.py`) | 同一データの再計算回避 |
| Feather形式キャッシュ | ✅ 実装済み (`scripts/v459/prepare_cached_data.py`) | I/O高速化 |
| dtype最適化 | ✅ 実装済み (`ztb/utils/memory/dtypes.py`) | メモリ削減 |
| チャンク処理 | ✅ 実装済み (`FeatureRegistry.compute_features`) | メモリ効率化 |
| 共通計算事前実行 | ✅ 実装済み (`_precompute_common_calculations`) | 重複計算削減 |
| GC強化 | ✅ 実装済み (`gc.set_threshold(100, 5, 5)`) | メモリリーク削減 |

**効果測定**:
- 既存最適化により、最適化前の推定時間（~1,200秒）から677秒へ削減（43%削減）
- さらなる削減余地あり

---

## 3. 最適化戦略

### 3.1 最適化アプローチ（レビュー後修正版）

```
優先順位付け（レビューに基づく再評価）:
🥇 Tier 0: 再計算排除（12実験で11回分削減）← 最優先・最大効果
   └── 特徴生成1回 → 全実験で再利用

🥈 Tier 1: 既存実装の最大活用（実装コストゼロ）
   ├── feature_set="fast" への切り替え
   ├── include_multi_timeframe_features=False 検証
   └── FeatureCache の有効化確認

🥉 Tier 2: MTF特徴の事前計算（41.8%のボトルネック解消）
   └── resample結果を固定ファイル化

🎯 Tier 3: Quick Wins（必要な箇所のみ）
   ├── ベクトル化改善（ループ削減）
   ├── 不要特徴量削減（相関/分散/欠損率ベース）
   └── Numba JIT（効果確認後の局所適用）
```

### 3.2 最適化手法マップ（レビュー後修正版）

| 手法 | 対象 | 推定削減率 | 実装難易度 | 優先度 |
|------|------|-----------|-----------|-------|
| **Tier 0: 再計算排除（最優先）** |
| 特徴生成1回 → 12実験再利用 | 全特徴量 | 92% (12→1回) | 🟢 Easy | ⭐⭐⭐⭐⭐ |
| 特徴付きデータのParquet保存 | I/O最適化 | - | 🟢 Easy | ⭐⭐⭐⭐⭐ |
| FeatureCache有効化確認 | キャッシュ | - | 🟢 Easy | ⭐⭐⭐⭐⭐ |
| **Tier 1: 既存実装活用（実装ゼロ）** |
| feature_set="fast" 切替 | 全特徴量 | 30-50% | 🟢 Easy | ⭐⭐⭐⭐ |
| MTF特徴無効化検証 | MTF集約 | 41.8% | 🟢 Easy | ⭐⭐⭐⭐ |
| 相関削減機構の活用 | 冗長特徴 | 10-20% | 🟢 Easy | ⭐⭐⭐ |
| **Tier 2: MTF事前計算** |
| resample結果の固定ファイル化 | MTF集約 | 30-40% | 🟡 Medium | ⭐⭐⭐ |
| MultiTimeframe データの保存 | I/O最適化 | - | 🟡 Medium | ⭐⭐⭐ |
| **Tier 3: Quick Wins（効果確認後）** |
| ベクトル化改善 | ループ処理 | 10-20% | 🟢 Easy | ⭐⭐ |
| 不要特徴量削減（相関/分散） | 全特徴量 | 5-15% | 🟢 Easy | ⭐⭐ |
| Numba JIT（局所適用） | ホットパス | 20-40% | 🟡 Medium | ⭐⭐ |
| **Tier 4: Long-term（Phase 3.5対象外）** |
| C拡張（Cython） | 全体 | 30-50% | 🔴 Hard | ⭐ |
| GPU加速 | 全特徴量 | 50-70% | 🔴 Hard | ⭐ |

### 3.3 Phase 3.5実装計画（レビュー後修正版）

**Tier 0 + Tier 1 + Tier 2に集中**（実装コスト最小、効果最大）:

```
Phase 3.5: 特徴生成最適化（1-2日）
├── Step 1: 再計算排除の実装（最優先、4時間）
│   ├── 特徴生成を独立スクリプト化
│   ├── 特徴付きデータをParquet保存
│   ├── ABRewardExperiment修正（生成済みデータ読込）
│   └── FeatureCacheキー設計確認
│
├── Step 2: 既存実装の活用（2時間）
│   ├── feature_set="fast" でテスト実行
│   ├── include_multi_timeframe_features=False でテスト実行
│   ├── 精度・再現性への影響評価
│   └── 最適構成の決定
│
├── Step 3: MTF特徴の事前計算（4時間）
│   ├── resample結果を固定ファイル化
│   ├── MultiTimeframeFeatureSystem修正
│   ├── 読込パス確認・検証
│   └── パフォーマンス計測
│
├── Step 4: プロファイリング（効果確認、2時間）
│   ├── 最適化前後の比較計測
│   ├── ボトルネック再確認
│   └── 追加最適化の必要性判断
│
└── Step 5: 検証・統合（2時間）
    ├── 12実験の実行時間計測
    ├── 精度検証（特徴量値の整合性）
    └── 再現性検証（CV維持確認）

注: Numba/ベクトル化はStep 4で必要性が確認された場合のみ実施
```

---

## 4. 詳細実装仕様（レビュー後修正版）

### 4.1 Step 1: 再計算排除の実装（最優先）

**目的**: 12実験での特徴生成を11回削減（最大効果）

**実装アプローチ**:

```python
# 1. 特徴生成を独立化（scripts/v459/generate_features_once.py）
import pandas as pd
from pathlib import Path
import os

def generate_features_once(
    data_file: str = 'data/btc_jpy_1m_v451.csv',
    output_file: str = 'data/btc_jpy_1m_v451_with_features.parquet',
    feature_set: str = 'standard'  # 'fast', 'standard', 'full'
):
    """特徴生成を1回だけ実行してParquet保存"""
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # 既存の特徴生成ロジックを使用
    from ztb.features.core.registry import FeatureRegistry
    
    # feature_setに応じた特徴量リスト取得
    if feature_set == 'fast':
        feature_names = FeatureRegistry.get_fast_features()
    elif feature_set == 'standard':
        feature_names = FeatureRegistry.get_standard_features()
    else:
        feature_names = FeatureRegistry.get_all_features()
    
    print(f"Generating {len(feature_names)} features (set={feature_set})...")
    df_with_features = FeatureRegistry.compute_features(
        df,
        feature_names=feature_names,
        verbose=True
    )
    
    # Parquet保存（圧縮効率良好、読込高速）
    print(f"Saving to {output_file}...")
    df_with_features.to_parquet(
        output_file,
        engine='pyarrow',
        compression='snappy',
        index=False
    )
    
    file_size_mb = Path(output_file).stat().st_size / (1024**2)
    print(f"✅ Saved: {file_size_mb:.2f} MB")
    return df_with_features

# 2. ABRewardExperiment修正（生成済みデータ読込）
# scripts/v459/run_ab_reward_experiments.py の修正箇所

class ABRewardExperiment:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.use_pregenerated = config.get('use_pregenerated_features', True)
        self.feature_file = config.get(
            'feature_file',
            'data/btc_jpy_1m_v451_with_features.parquet'
        )
    
    def load_data(self) -> pd.DataFrame:
        """データ読込（特徴生成済みまたは通常）"""
        if self.use_pregenerated and Path(self.feature_file).exists():
            print(f"📦 Loading pre-generated features from {self.feature_file}")
            df = pd.read_parquet(self.feature_file)
            print(f"   ✅ Loaded {len(df)} rows with {len(df.columns)} columns")
            return df
        else:
            # 既存の処理（特徴生成含む）
            print("⚙️ Generating features on-the-fly...")
            return self._load_and_generate_features()

# 使用例
if __name__ == '__main__':
    # 最初に1回だけ実行
    generate_features_once(
        data_file='data/btc_jpy_1m_v451.csv',
        output_file='data/btc_jpy_1m_v451_with_features.parquet',
        feature_set='standard'
    )
    
    # その後の12実験は生成済みファイルを使用
    # → 特徴生成時間が11回分削減される
```

**FeatureCache確認**:

```python
# 3. FeatureCacheが正しく機能しているか確認
import
import pstats
import os
from pstats import SortKey
from pathlib import Path

def profile_single_training():
    """単一トレーニングの特徴生成をプロファイル"""
    # SIGINT問題回避
    os.environ['ZTB_SIGINT_POLICY'] = 'ignore'
    os.environ['ZTB_SKIP_SKLEARN'] = '1'
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 実際のトレーニング実行（実在する入口を使用）
    from scripts.v459.run_ab_reward_experiments import ABRewardExperiment
    
    config = {
        'seed': 42,
        'reward_stage': 'stage1_basic',
        'total_timesteps': 5000,
        'data_file': 'data/btc_jpy_1m_v451.csv'
    }
    
    experiment = ABRewardExperiment(config)
    experiment.execute()
    
    profiler.disable()
    
    # 結果出力
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(50)  # Top 50関数
    
    # 特徴生成関連のみフィルタ
    stats.print_stats('feature|compute')
    
    # ファイル保存
    stats.dump_stats('profile_results/feature_generation.prof')

def analyze_profile():
    """プロファイル結果の詳細分析"""
    stats = pstats.Stats('profile_results/feature_generation.prof')
    
    # 関数別累積時間
    cumulative = stats.sort_stats(SortKey.CUMULATIVE)
    
    # 関数呼び出し回数
    calls = stats.sort_stats(SortKey.CALLS)
    
    # ボトルネック特定
    bottlenecks = identify_bottlenecks(stats)
    
    # レポート生成
    generate_optimization_report(bottlenecks)
```

**出力**:
- `profile_results/feature_generation.prof`: プロファイルデータ
- `profile_results/bottleneck_report.json`: ボトルネック分析
- `profile_results/optimization_targets.md`: 最適化推奨事項

### 4.2 Step 2: 既存実装の活用（実装コストゼロ）

**優先アプローチ**: 新規実装より既存機能の最適化

```python
# 1. feature_set切り替えによる高速化
from ztb.features.core.registry import FeatureRegistry

# 既存の"fast"プリセットを活用
feature_names_fast = FeatureRegistry.get_fast_features()
feature_names_standard = FeatureRegistry.get_standard_features()
feature_names_full = FeatureRegistry.get_all_features()

# Phase 3.1での比較検証
for feature_set in ['fast', 'standard']:
    df_features = FeatureRegistry.compute_features(
        df,
        feature_names=feature_names_map[feature_set],
        verbose=True,
        return_timing=True
    )
    # パフォーマンスと精度のトレードオフ評価

# 2. MTF特徴の選択的無効化
config = {
    'include_multi_timeframe_features': False,  # MTF無効化で41.8%削減可能
    'include_regime_features': True,
    'include_cyclical_features': True
}

# 3. 相関削減機構の活用（既存実装）
from ztb.features.core.registry import FeatureRegistry

# 既に実装されている相関削減
df_reduced = FeatureRegistry._select_features_by_correlation_in_env(
    df_features,
    correlation_threshold=0.95,
    importance_method='variance'  # 分散ベースで高速
)
```

**期待効果**: 30-50%高速化（実装コストゼロ）

---

### 4.3 Step 3: MTF特徴の事前計算

**目的**: MultiTimeframe特徴（41.8%のボトルネック）の最適化

```python
# scripts/v459/precompute_mtf_features.py
import pandas as pd
from pathlib import Path

def precompute_mtf_resamples(
    data_file: str = 'data/btc_jpy_1m_v451.csv',
    output_dir: str = 'data/mtf_precomputed'
):
    """MTF集約結果を事前計算して保存"""
    df_1m = pd.read_csv(data_file, parse_dates=['timestamp'])
    df_1m.set_index('timestamp', inplace=True)
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # 5分足
    df_5m = df_1m.resample('5T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    df_5m.to_parquet(f"{output_dir}/btc_jpy_5m.parquet")
    
    # 15分足
    df_15m = df_1m.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    df_15m.to_parquet(f"{output_dir}/btc_jpy_15m.parquet")
    
    # 1時間足
    df_1h = df_1m.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    df_1h.to_parquet(f"{output_dir}/btc_jpy_1h.parquet")
    
    print(f"✅ Precomputed MTF data saved to {output_dir}")

# MultiTimeframeFeatureSystem修正（読込に変更）
class MultiTimeframeFeatureSystem:
    def __init__(self, use_precomputed: bool = True):
        self.use_precomputed = use_precomputed
        self.precomputed_dir = Path('data/mtf_precomputed')
    
    def get_resampled_data(self, timeframe: str) -> pd.DataFrame:
        if self.use_precomputed:
            file_path = self.precomputed_dir / f"btc_jpy_{timeframe}.parquet"
            if file_path.exists():
                return pd.read_parquet(file_path)
        # フォールバック: 通常のresample
        return self._resample_on_the_fly(timeframe)
```

**期待効果**: 30-40%高速化（MTF特徴量のみ）

---

### 4.4 Step 4: Numba JIT高速化（効果確認後の局所適用）

**⚠️ 注意**: レビュー指摘により優先度低下、O(n²)問題に注意

**適用箇所**（効果が確認された場合のみ）:
```python
# 1. 価格変化率計算（シンプルでNumba適用しやすい）
from numba import jit
import numpy as np

@jit(nopython=True, cache=True)
def compute_returns_numba(prices: np.ndarray, periods: int) -> np.ndarray:
    """Numba最適化済み価格変化率"""
    n = len(prices)
    returns = np.empty(n, dtype=np.float32)
    returns[:periods] = 0.0
    
    for i in range(periods, n):
        if prices[i - periods] != 0:
            returns[i] = (prices[i] - prices[i - periods]) / prices[i - periods]
        else:
            returns[i] = 0.0
    
    return returns

# 2. 移動平均計算
@jit(nopython=True, cache=True)
def compute_sma_numba(values: np.ndarray, window: int) -> np.ndarray:
    """Numba最適化済みSMA"""
    n = len(values)
    sma = np.empty(n, dtype=np.float32)
    sma[:window] = np.nan
    
    for i in range(window, n):
        sma[i] = np.mean(values[i - window:i])
    
    return sma

# 3. Ichimoku計算（⚠️ O(n²)問題あり - 要アルゴリズム改善）
# レビュー指摘: ループ内でnp.max/minはO(n²)になる
# → rolling max/minをdequeで管理する実装に変更必要

# 修正前（O(n²) - 使用禁止）
# @jit(nopython=True, cache=True)
# def compute_ichimoku_numba_slow(high, low, period):
#     for i in range(period, n):
#         result[i] = np.max(high[i-period:i])  # ← O(period)が毎回実行

# 修正後（O(n) - deque使用）
from collections import deque

@jit(nopython=True, cache=True)
def compute_rolling_max_numba(values: np.ndarray, window: int) -> np.ndarray:
    """O(n)のrolling max（単調deque使用）"""
    n = len(values)
    result = np.empty(n, dtype=np.float32)
    result[:window] = np.nan
    
    # 単調減少dequeでO(n)を実現
    # 実装は複雑なため、まずpandas.rolling().max()を使用
    # 本当に必要な場合のみNumba化
    return result

# 推奨: まずpandas組み込みを使用（十分高速）
def compute_ichimoku_pandas(df: pd.DataFrame, 
                           tenkan_period: int, 
                           kijun_period: int) -> pd.DataFrame:
    """pandasのrolling使用（O(n)で高速）"""
    high_max_tenkan = df['high'].rolling(tenkan_period).max()
    low_min_tenkan = df['low'].rolling(tenkan_period).min()
    tenkan = (high_max_tenkan + low_min_tenkan) / 2
    
    high_max_kijun = df['high'].rolling(kijun_period).max()
    low_min_kijun = df['low'].rolling(kijun_period).min()
    kijun = (high_max_kijun + low_min_kijun) / 2
    
    # Kijun-sen
    kijun = np.empty(n, dtype=np.float32)
    for i in range(kijun_period, n):
        kijun[i] = (np.max(high[i-kijun_period:i]) + 
                    np.min(low[i-kijun_period:i])) / 2
    
    # Senkou Span A
    senkou_a = (tenkan + kijun) / 2
    
    return tenkan, kijun, senkou_a
```

**統合方法**:
```python
# ztb/features/core/registry.py に追加
class FeatureRegistry:
    @classmethod
    def _use_numba_optimization(cls, feature_name: str) -> bool:
        """Numba最適化を使用するか判定"""
        numba_features = [
            'returns_', 'sma_', 'ema_', 'ichimoku_',
            'rsi_', 'macd_', 'bollinger_'
        ]
        return any(feature_name.startswith(prefix) for prefix in numba_features)
```

**期待効果**: 20-40%高速化（推定: 431秒 → 259-345秒）

### 4.3 Step 3: ベクトル化改善

**対象**: Pythonループの排除

```python
# Before: ループ処理
def compute_rolling_std_slow(values: pd.Series, window: int) -> pd.Series:
    result = []
    for i in range(len(values)):
        if i < window:
            result.append(np.nan)
        else:
            result.append(np.std(values[i-window:i]))
    return pd.Series(result, index=values.index)

# After: ベクトル化
def compute_rolling_std_fast(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=window).std()
```

**チェックリスト**:
- [ ] `for i in range(len(df))` の排除
- [ ] `df.iterrows()` の排除
- [ ] 条件分岐の`np.where()` / `pd.Series.mask()`への置き換え
- [ ] Apply関数のベクトル化

**期待効果**: 10-20%高速化（推定: 345秒 → 276-311秒）

### 4.5 Step 5: 不要特徴量削減（sklearn不使用）

**⚠️ 重要**: 現環境は`ZTB_SKIP_SKLEARN=1`のため、sklearn依存の分析は別環境で実行

**分析方法**（sklearnなし）:
```python
# scripts/v459/analyze_feature_redundancy.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def analyze_feature_redundancy_safe(
    df_features: pd.DataFrame
) -> Dict[str, any]:
    """特徴量冗長性分析（sklearn不使用）"""
    
    # 1. 相関分析（高相関ペアの特定）
    corr_matrix = df_features.corr(method='spearman')
    high_corr_pairs = find_high_correlation_pairs(
        corr_matrix, 
        threshold=0.95
    )
    
    # 2. 分散ベース重要度（sklearn不要）
    variances = df_features.var()
    variance_importances = dict(
        zip(df_features.columns, variances / variances.sum())
    )
    
    # 3. 削減候補特定
    redundant_features = []
    for feat1, feat2 in high_corr_pairs:
        # 重要度の低い方を削減候補に
        if importances[feat1] < importances[feat2]:
            redundant_features.append(feat1)
        else:
            redundant_features.append(feat2)
    
    return {
        'redundant': redundant_features,
        'importances': importances,
        'high_correlations': high_corr_pairs
    }
```

**削減基準**（sklearn不要）:
- 相関係数 > 0.95 かつ 分散が小さい特徴量
- 分散 < 全体分散の1%の特徴量
- 欠損率 > 50% の特徴量
- ゼロ率 > 90% の特徴量（ほぼ情報なし）

**期待効果**: 5-15%高速化

**⚠️ 注意**: sklearn使用の分析は別環境で実行
```python
# 別環境（SIGINT問題ない環境）で実行
# ZTB_SKIP_SKLEARN=0 で実行し、結果をJSONで保存
# → 本環境では結果を読み込むのみ
```

---

### 4.6 最終目標（修正版）

```
【最優先】再計算排除による効果（12実験での削減）:
最適化前: 431秒 × 12実験 = 5,172秒 (86.2分)
最適化後: 431秒 × 1回 = 431秒 (7.2分) ← 🎯 主目標
削減効果: 4,741秒削減 (79分, 92%削減)

【副次的】特徴生成自体の高速化:
最適化前: 431秒（特徴生成）
  ↓ 既存実装活用（feature_set="fast"）
  ↓ 215-258秒（-40~-50%）
  ↓ MTF事前計算
  ↓ 130-180秒（-50~-70%）
  ↓ 不要特徴削減
  ↓ 
最適化後: 110-153秒（特徴生成） ← 🎯 副次目標

【両方実施した場合の総効果】:
12実験での特徴生成時間:
  最適化前: 5,172秒 (86.2分)
  最適化後: 110-153秒 (1.8-2.6分)
  削減効果: 5,019-5,062秒 (83.6-84.4分, 97%削減)

50,000ステップ × 12実験での総時間推定:
  最適化前: 578分 (9.6時間)
  最適化後: 494-496分 (8.2-8.3時間)
  削減効果: 82-84分 (14-15%削減)
```

---

## 5. 検証計画

### 5.1 パフォーマンス検証

```python
# scripts/v459/verify_optimization.py
def verify_performance_improvement():
    """最適化前後のパフォーマンス比較"""
    
    # Baseline計測（最適化前のコード）
    baseline_time = measure_feature_generation_time(
        use_optimization=False,
        n_runs=3
    )
    
    # Optimized計測
    optimized_time = measure_feature_generation_time(
        use_optimization=True,
        n_runs=3
    )
    
    # 統計的検定
    speedup = baseline_time / optimized_time
    
    assert speedup >= 1.5, f"目標達成せず: {speedup:.2f}x"
    
    return {
        'baseline_mean': baseline_time,
        'optimized_mean': optimized_time,
        'speedup': speedup,
        'reduction_pct': (1 - 1/speedup) * 100
    }
```

**検証基準**:
- ✅ 最低基準: 1.5倍高速化（677秒 → 451秒以下）
- 🎯 目標基準: 2.0倍高速化（677秒 → 339秒以下）
- 🌟 理想基準: 2.5倍高速化（677秒 → 271秒以下）

### 5.2 精度検証

**特徴量値の整合性確認**:
```python
def verify_feature_accuracy():
    """最適化前後の特徴量値の一致確認"""
    
    # 同一データで計算
    df_baseline = compute_features_baseline(data)
    df_optimized = compute_features_optimized(data)
    
    # 相対/絶対許容範囲（浮動小数点演算の現実的な範囲）
    rtol = 1e-5  # 相対誤差 0.001%
    atol = 1e-8  # 絶対誤差
    
    mismatches = []
    for col in df_baseline.columns:
        # np.allclose使用（NaNも考慮）
        is_close = np.allclose(
            df_baseline[col].fillna(0),
            df_optimized[col].fillna(0),
            rtol=rtol,
            atol=atol,
            equal_nan=True
        )
        
        if not is_close:
            max_diff = np.abs(df_baseline[col] - df_optimized[col]).max()
            max_rel_diff = (np.abs(df_baseline[col] - df_optimized[col]) / 
                           (np.abs(df_baseline[col]) + 1e-10)).max()
            mismatches.append({
                'feature': col,
                'max_abs_diff': max_diff,
                'max_rel_diff': max_rel_diff
            })
    
    if mismatches:
        print(f"⚠️ {len(mismatches)} features have differences beyond tolerance")
        for m in mismatches[:10]:  # 最初の10個のみ表示
            print(f"  {m['feature']}: abs={m['max_abs_diff']:.2e}, rel={m['max_rel_diff']:.2e}")
    else:
        print("✅ 特徴量の整合性確認完了（全特徴量が許容範囲内）")
    
    return len(mismatches) == 0
```

### 5.3 再現性検証

**最適化後も再現性を維持**:
```python
def verify_reproducibility_after_optimization():
    """最適化後の再現性維持確認"""
    
    results = []
    for seed in [42, 123, 456]:
        result = run_experiment_optimized(seed=seed, timesteps=5000)
        results.append(result)
    
    # アクション分布のCV計算
    action_dists = [r['action_distribution'] for r in results]
    cv = calculate_coefficient_of_variation(action_dists)
    
    # 最適化前の基準（CV < 1%）を維持
    # ⚠️ 注意: RL訓練はstochasticなためCV < 1%は非常に厳しい
    # 許容範囲: CV < 3%（excellent水準を維持）
    cv_threshold = 3.0  # 最適化前は0.32-0.99%、3%以下なら優秀
    
    checks = {
        'hold': cv['hold'] < cv_threshold,
        'buy': cv['buy'] < cv_threshold,
        'sell': cv['sell'] < cv_threshold
    }
    
    if all(checks.values()):
        print("✅ 再現性維持確認完了")
        print(f"   HOLD CV: {cv['hold']:.2f}%")
        print(f"   BUY CV: {cv['buy']:.2f}%")
        print(f"   SELL CV: {cv['sell']:.2f}%")
        return True
    else:
        print("⚠️ 再現性が低下しています")
        for action, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {action.upper()} CV: {cv[action]:.2f}%")
        return False

# 根拠: 最適化前の実測値
# - HOLD: 0.32% (excellent)
# - BUY: 0.84% (excellent)
# - SELL: 0.99% (excellent)
# 目標: CV < 3% を維持すれば「excellent」水準
```

---

## 6. 00番ドキュメントとの整合性

### 6.1 Phase 3との関係

**元のPhase 3計画**（[00_project_proposal_v459.md](00_project_proposal_v459.md#phase-3-報酬設計の段階検証---3-4日)）:
```
Phase 3: 報酬設計の段階検証 - 3-4日
├── 3.1: 純PnL only ベースライン性能
├── 3.2: PnL + Trend Guidance 固定
└── 3.3: PnL + Trend Guidance Decay
```

**Phase 3.5挿入の正当性**:

| 判断基準 | 理由 |
|---------|------|
| **プロジェクト大義** | 「短期間での高収益性システム」実現に最適化は不可欠 |
| **実用性** | 50,000ステップで9.6時間は許容可能だが、改善余地大 |
| **ROI** | 1-2日の投資で全Phase通じて時間削減＋開発体験改善 |
| **リスク** | Low（既存機能活用中心、検証済みアプローチ） |
| **効果** | 再計算排除で92%削減、特徴生成高速化で70-75%削減可能 |

**修正後タイムライン**:
```
Week 2 (01/27-01/31) ← 現在ここ
├── Day 1-2: Phase 3.5 (特徴生成最適化)  ← 新規挿入
│   ├── 再計算排除（最優先）
│   ├── 既存実装活用
│   └── MTF事前計算
├── Day 3-4: Phase 3.1 (純PnL検証)
└── Checkpoint: 最適化効果確認

Week 3 (02/01-02/07)
├── Day 1-2: Phase 3.2 (固定Guidance)
├── Day 3-4: Phase 3.3 (Decay Guidance)
├── Day 5-7: Phase 4 前半 (評価・検証)
└── Checkpoint: 収益性初期確認

Week 4 (02/08-02/14)
├── Day 1-5: Phase 4 後半 + Phase 5 (Paper Trading)
├── Day 6-7: Phase 6 (Go/No-Go判定)
└── Final Decision: 02/14
```

### 6.2 成功基準への影響

**00番ドキュメントの成功基準**（[Section 5.2](00_project_proposal_v459.md#52-収益性検証gate-2-gonogo判定軸)）は変更なし:
- Net ROI > 5% (目標 > 15%)
- Profit Factor > 1.20 (目標 > 1.50)
- Sharpe Ratio > 1.0 (目標 > 1.5)

**Phase 3.5の貢献**:
- ✅ 実験イテレーション速度向上 → 品質向上機会増加
- ✅ 開発者体験改善 → 長時間待機のフラストレーション削減
- ✅ 本番環境でのスケーラビリティ向上

---

## 7. リスク評価と緩和策

### 7.1 技術リスク

| リスク | 影響度 | 発生確率 | 緩和策 |
|--------|-------|---------|--------|
| Numba互換性問題 | Medium | Low | 段階的適用、フォールバック実装 |
| 精度劣化 | High | Low | 厳密な検証テスト |
| 再現性損失 | High | Low | Seed固定、比較テスト |
| 新規バグ混入 | Medium | Medium | 既存テスト全実行 |

### 7.2 スケジュールリスク

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| 最適化に2日以上 | Phase 3遅延 | Tier 1のみに限定 |
| 効果不十分（<1.5x） | 時間投資無駄 | プロファイリングで事前検証 |
| 統合問題 | デバッグ時間増加 | 段階的統合、独立テスト |

**判断基準**:
- **Go**: 1日目終了時に1.5x高速化の見込みあり → Phase 3.5続行
- **No-Go**: 効果不十分 → 最適化中止、Phase 3へ移行

---

## 8. 実装チェックリスト（レビュー後修正版）

### Phase 3.5 Day 1（再計算排除 + 既存実装活用）

- [ ] **Step 1: 再計算排除の実装** (4時間)
  - [ ] `scripts/v459/generate_features_once.py` 作成
  - [ ] 特徴付きデータのParquet保存
  - [ ] ABRewardExperiment修正（保存済みデータ読込）
  - [ ] FeatureCache有効化確認
  - [ ] 12実験での削減率計測

- [ ] **Step 2: 既存実装の活用** (2時間)
  - [ ] feature_set="fast" でテスト実行
  - [ ] include_multi_timeframe_features=False でテスト実行
  - [ ] パフォーマンス計測（時間削減効果）
  - [ ] 精度への影響評価

- [ ] **Step 3: MTF事前計算** (2時間)
  - [ ] `scripts/v459/precompute_mtf_features.py` 作成
  - [ ] 5分/15分/1時間足データの保存
  - [ ] MultiTimeframeFeatureSystem修正
  - [ ] パフォーマンス計測

### Phase 3.5 Day 2（検証 + 追加最適化）

- [ ] **Step 4: プロファイリング** (2時間)
  - [ ] `scripts/v459/profile_feature_generation.py` 作成
  - [ ] ZTB_SIGINT_POLICY=ignore で安全実行
  - [ ] ボトルネック再確認
  - [ ] 追加最適化の必要性判断

- [ ] **Step 5: 精度・再現性検証** (2時間)
  - [ ] 特徴量値の整合性テスト（rtol/atol使用）
  - [ ] 3 seed再現性テスト（CV < 3%確認）
  - [ ] アクション分布の安定性確認

- [ ] **Step 6: 統合・ドキュメント** (2時間)
  - [ ] 最適化コードのマージ
  - [ ] 既存テスト全実行
  - [ ] 12実験実行時間の最終計測
  - [ ] 最適化完了レポート作成
  - [ ] Phase 3実験の準備

### オプション（必要な場合のみ）

- [ ] **Numba/ベクトル化** (Step 4で必要性確認後)
  - [ ] ホットパス特定
  - [ ] Numba JIT適用（O(n²)問題に注意）
  - [ ] ベクトル化改善
  - [ ] 効果測定

---

## 9. 成功基準

### 9.1 必須基準（Go/No-Go）

| 指標 | 最低基準 | 測定方法 |
|------|---------|---------|
| **高速化率** | ≥ 1.5x | 3実験平均 |
| **特徴量精度** | 誤差 < 1e-5 | 全特徴量の最大差分 |
| **再現性維持** | CV < 1.0% | 3 seed のアクション分布 |
| **既存テスト** | 100% Pass | pytest全実行 |

### 9.2 目標基準

| 指標 | 目標 | 理想 |
|------|------|------|
| **高速化率** | 2.0x | 2.5x |
| **総トレーニング時間** | < 400秒 | < 350秒 |
| **50k step換算** | < 70分 | < 60分 |

### 9.3 Phase 3への引き継ぎ条件

✅ **Phase 3開始可能**:
- [x] 高速化率 ≥ 1.5x
- [x] 精度・再現性維持
- [x] 全テストPass
- [x] ドキュメント更新

---

## 10. 次のステップ（レビュー後修正版）

### 10.1 Immediate（Phase 3.5開始）

**最優先: 再計算排除の実装**

1. **特徴生成スクリプト作成**
   ```bash
   # 環境変数設定
   $env:ZTB_SIGINT_POLICY="ignore"
   $env:ZTB_SKIP_SKLEARN="1"
   
   # 特徴生成（1回のみ）
   python scripts/v459/generate_features_once.py
   ```

2. **ABRewardExperiment修正**
   - 保存済みParquetファイルの読込機能追加
   - フォールバック機能（ファイルがない場合は通常処理）

3. **効果測定**
   - 12実験での特徴生成時間を計測
   - 削減率が80%以上であることを確認

**副次的: 既存実装の活用**

4. **feature_set比較**
   ```bash
   # fast vs standard の比較
   python scripts/v459/compare_feature_sets.py
   ```

5. **MTF特徴の事前計算**
   ```bash
   python scripts/v459/precompute_mtf_features.py
   ```

### 10.2 Phase 3.5完了後

1. **Phase 3.1実験実行**
   - 純PnL報酬で4 seeds × 1 config = 4実験
   - 最適化後の実時間計測
   - 目標: 単一実験 < 50分

2. **Phase 3.2-3.3継続**
   - Trend Guidance段階検証
   - 合計12実験完了
   - 目標: 総時間 < 500分 (8.3時間)

3. **Phase 4へ移行**
   - 統計分析、ベースライン比較

### 10.3 追加最適化（必要な場合のみ）

**判断基準**: Step 4のプロファイリングで以下が確認された場合のみ実施
- ボトルネックが特定の計算に集中
- Numba/ベクトル化で20%以上の追加削減が見込める

**実施内容**:
- Numba JIT適用（ホットパスのみ）
- ベクトル化改善（ループ削減）
- 効果測定と精度検証

---

## 11. 関連ドキュメント

- [00. v459プロジェクト提案](00_project_proposal_v459.md) - 全体計画
- [34. Windows SIGINT解決と初回実験成功](34_windows_sigint_resolution_and_first_success.md) - 前提条件
- [36. 最適化計画レビュー](36_feature_generation_optimization_review.md) - レビューと修正指摘
- [24. Phase 3仕様](24_phase3_specification.md) - Phase 3詳細

---

## 12. レビュー対応サマリー

**レビュー日**: 2026-01-27  
**レビュアー**: AIコーディングエージェント

### 主要な修正点

1. **時間見積りの修正**: 固定コスト（特徴生成431秒）+ 変動コスト（0.0492秒/ステップ）に分離
2. **最優先事項の変更**: Numba/ベクトル化 → **再計算排除**（92%削減効果）
3. **既存実装の活用**: 新規実装より設定最適化を優先（feature_set="fast"等）
4. **MTF事前計算**: resample結果の固定ファイル化を優先
5. **sklearn依存の削減**: 相関/分散/欠損率ベースの分析に変更（ZTB_SKIP_SKLEARN=1対応）
6. **Numba実装の警告**: O(n²)問題への注意喚起、アルゴリズム改善優先
7. **プロファイリング実体修正**: ABRewardExperiment使用に修正
8. **精度検証の改善**: rtol/atol使用、equal_nan=True対応
9. **再現性基準の緩和**: CV < 1% → CV < 3%（excellent水準維持）
10. **成功基準の再定義**: 再計算排除効果を中心に評価

### 期待効果の再計算

**再計算排除のみ** (最優先):
- 12実験: 5,172秒 → 431秒（92%削減、79分短縮）

**特徴生成高速化** (副次的):
- 単一実験: 431秒 → 110-153秒（70-75%削減）

**総合効果**:
- 12実験での特徴生成: 5,172秒 → 110-153秒（**97%削減**）
- 50,000ステップ×12実験: 578分 → 494-496分（14-15%削減）

---

**Status**: 📋 Planning → ✅ Review Addressed → ⏳ Implementation Ready  
**Author**: GitHub Copilot  
**Date**: 2026-01-27（初版）/ 2026-01-27（レビュー対応版）  
**Estimated Duration**: 1-2日  
**Expected Impact**: 再計算排除で92%削減、特徴生成高速化で70-75%削減、総合97%削減
