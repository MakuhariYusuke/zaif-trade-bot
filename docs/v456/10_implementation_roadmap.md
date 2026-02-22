# v456 実装ロードマップ

> **Version**: v456.2  
> **Date**: 2026-01-13  
> **Status**: Active Development

---

## 概要

第2次外部レビュー対応完了。実装フェーズに移行。  
既存v455実装の最大限再利用 + v456仕様への段階的拡張。

---

## Week 1: データ整合性確保 (Critical Path)

### タスク 1-1: MTFリーク検出テスト実装

**ファイル**: `tests/unit/features/test_mtf_no_future_leak.py`

```python
# 実装予定
def test_mtf_closed_bar_boundary():
    """MTF特徴量がクローズドバーのみを使用することを検証"""
    # 10:07時点での特徴量取得
    # 期待: 10:05バーのデータのみ
    # 確認: 10:10以降のデータが含まれていないこと

def test_mtf_asof_missing_data():
    """asof()による欠損データ処理が正しいことを確認"""
    # 欠損バーがある場合の挙動
    # 期待: 最新の有効バーの値を使用

def test_timezone_aware_timestamps():
    """Naive timestampが拒否されることを確認"""
    # Naive timestamp入力 → ValueError
    # UTC aware timestamp → 正常処理
```

**既存リソース**:
- `ztb/features/multi_timeframe.py` - MTF実装
- `ztb/features/generators/multi_timeframe/` - MTFエンジン
- `ztb/trading/environment/fast_intraday_env.py` - 環境統合

### タスク 1-2: 正規化パイプライン分離実装

**ファイル**: `ztb/features/grouping/grouped_scaler.py`

```python
# 実装予定
class GroupedFeatureScaler:
    """
    グループベースの正規化スケーラー
    - online_zscore: base + global_continuous のみ
    - no_scaling: mtf, cyclical_time, regime_onehot, account
    """
    def __init__(self, feature_groups: Dict[str, List[str]]):
        # グループごとにスケーラーを分離
        self.online_scaler = OnlineZScoreScaler(...)
        self.no_scaling_groups = [...]
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        # グループごとに処理
        pass
```

**既存リソース**:
- `ztb/processing/online_scaler.py` - OnlineScaler実装
- `ztb/utils/normalization.py` - 正規化統計管理

### タスク 1-3: タイムゾーン検証ユーティリティ

**ファイル**: `ztb/features/utils/timestamp_validator.py`

```python
# 実装予定
def validate_and_convert_timestamp(
    timestamp: pd.Timestamp,
    require_tz: bool = True,
    target_tz: str = "UTC"
) -> pd.Timestamp:
    """Naive timestampを拒否、tz-aware化"""
    if timestamp.tzinfo is None and require_tz:
        raise ValueError(f"Naive timestamp not allowed: {timestamp}")
    return timestamp.tz_convert(target_tz)
```

**既存リソース**:
- `ztb/data/` - データ処理ユーティリティ
- バックテストスクリプト - タイムゾーン処理例

---

## Week 2: 特徴量追加 + MLPベースライン確立

### タスク 2-1: Cyclical Time Features

**ファイル**: `ztb/features/time/cyclical_v456.py`

```python
# 実装予定
def calc_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    6特徴量: hour_sin, hour_cos, minute_sin, minute_cos, dow_sin, dow_cos
    """
    df = df.copy()
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    # ...
    return df[['hour_sin', 'hour_cos', ...]]
```

**既存リソース**:
- `ztb/features/time/time_features.py` - 既存時間特徴量実装

### タスク 2-2: Global Market Features (拡張)

**ファイル**: `ztb/features/global_market_v456.py`

```python
# 実装予定
class GlobalMarketFeatureEngineerV456:
    """
    9特徴量:
    - 6連続: spread, return_1m, return_5m, vol_1m, vol_ratio, usdt_premium
    - 3フラグ: spread_flag, return_flag, stale_flag
    """
    def generate_features(self) -> np.ndarray:
        # size 9
        pass
    
    def handle_stale_global_features(self) -> None:
        """データ鮮度フラグ処理"""
        pass
```

**既存リソース**:
- `ztb/features/global_market.py` - Lead-Lag特徴量実装済み

### タスク 2-3: MLP SAC学習

**ファイル**: `scripts/v456/train_mlp_baseline.py`

```python
# 既存スクリプトの拡張
# scripts/v455/train_hft.py から fork
# 変更点:
# - 特徴量数: 30 → 88
# - Reward calibration追加
# - フィルタリング統合
```

**既存リソース**:
- `scripts/v455/train_hft.py` - v455学習スクリプト
- `config/v454/sac_v454_*.json` - 設定ファイルテンプレート

---

## Week 3: フィルタリング統合 (Train-Live Parity)

### タスク 3-1: 環境内フィルタリング統合

**ファイル**: `ztb/trading/environment/fast_intraday_env_v456.py`

```python
# 既存環境 (fast_intraday_env.py) の拡張
class FastIntradayEnvV456(FastIntradayEnv):
    """
    step()内部でフィルタリング:
    1. Soft Filter (時間/レジーム制約)
    2. Calibration Gate (EV判定)
    3. 報酬計算（フィルタリングコスト含む）
    """
```

**既存リソース**:
- `ztb/trading/environment/fast_intraday_env.py` - 基本環境
- `ztb/trading/rewards/` - 報酬関数群

---

## ディレクトリ構成計画

```
config/v456/
├── v456_base.json             # ベース設定
├── v456_mlp_baseline.json     # MLP学習用
└── v456_gru_optional.json     # GRU導入時用

ztb/features/
├── grouping/
│   ├── __init__.py
│   └── grouped_scaler.py       # GroupedFeatureScaler (NEW)
├── time/
│   ├── __init__.py
│   ├── time_features.py        # 既存
│   └── cyclical_v456.py        # NEW
├── global_market_v456.py       # NEW (拡張版)
└── utils/
    ├── timestamp_validator.py  # NEW
    └── ...

ztb/trading/environment/
├── fast_intraday_env.py        # 既存
├── fast_intraday_env_v456.py   # NEW
└── ...

scripts/v456/
├── train_mlp_baseline.py       # NEW
├── backtest_mlp_baseline.py    # NEW
├── config_generator.py         # 設定生成
└── ...

tests/unit/features/
├── test_mtf_no_future_leak.py  # NEW
├── test_grouping_scaler.py     # NEW
└── test_timestamp_validator.py # NEW
```

---

## 既存リソース活用マトリクス

| 機能 | 既存ファイル | 状態 | アクション |
|-----|-----------|------|---------|
| OnlineScaler | `ztb/processing/online_scaler.py` | ✅ 完成 | 再利用 |
| MTF実装 | `ztb/features/multi_timeframe.py` | ✅ 完成 | 検証、クローズドバー確認 |
| 環境基本 | `ztb/trading/environment/fast_intraday_env.py` | ✅ 完成 | v456拡張作成 |
| 報酬関数 | `ztb/trading/rewards/fast_intraday.py` | ✅ 完成 | キャリブレーション追加 |
| 学習スクリプト | `scripts/v455/train_hft.py` | ✅ 完成 | Fork+修正 |
| 設定テンプレート | `config/v454/sac_v454_*.json` | ✅ 完成 | Fork+拡張 |
| 時間特徴量 | `ztb/features/time/time_features.py` | ✅ 完成 | 検証 |
| Global特徴量 | `ztb/features/global_market.py` | ✅ 完成 | 拡張 |

---

## マイルストーン

| 週 | タスク | 完了条件 | ドキュメント |
|----|-------|--------|-----------|
| **Week 1** | Data Integrity | テスト100%パス | `11_week1_completion.md` |
| **Week 2** | Feature Eng + MLP | Sharpe > 0.3 | `12_week2_completion.md` |
| **Week 3** | Filtering Integration | Train-Live Parity | `13_week3_completion.md` |
| **Week 4-5** | Signal Fusion + Dynamic TP/SL | End-to-End流 | `14_week45_completion.md` |
| **Week 6+** | GRU (Optional) | 条件付き | `15_gru_optional.md` |

---

## 注記

- **リソース枯渇対策**: 既存実装を最大限再利用し、新規ファイルは必要最小限
- **品質管理**: 各タスク完了時にテスト実施、ドキュメント更新
- **レビュー対応**: 09_second_review_response.mdで指摘された全項目をチェックリスト化

