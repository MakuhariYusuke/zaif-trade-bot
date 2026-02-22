# v456 Feature Engineering Specification: 特徴量設計書

> **Version**: v456.2 (Second Review Revision)  
> **Date**: 2026-01-13  
> **Status**: Draft (Second Revision)

---

## ⚠️ Critical Design Notes (External Review Feedback)

### データリーク防止に関する重要事項
1. **MTFリサンプリング**: 必ず「クローズドバー」のみ使用、タイムフレーム表記は"5min"に統一
2. **正規化パイプライン**: カテゴリカル/時間特徴量は正規化対象外とすること
3. **タイムゾーン処理**: naive timestampをUTCとして明示的に処理すること

### 再レビュー指摘事項（2026-01-13追加）
4. **特徴量数の統一**: 全ドキュメントで**88特徴量**に統一
5. **正規化実装例**: グループ別スケーラーを明示、全体スケーリング記載は削除
6. **Global特徴量**: 連続6 + フラグ3 = 9 に統一

詳細は各セクションの「⚠️ CRITICAL」マークを参照。

---

## 1. 特徴量アーキテクチャ概要

### 1.1. 特徴量カテゴリ（⚠️ 統一版: 88特徴量）
```
v456 Feature Vector (Total: 88 features)
├── 1. Base Features (1分足)           30 features  [Online Z-Score]
├── 2. MTF Features (5min/15min/1h)    27 features  [NO scaling - pre-normalized]
├── 3. Cyclical Time Features           6 features  [NO scaling - sin/cos [-1,1]]
├── 4. Global Market Features           9 features  [Mixed: 6 Z-Score + 3 Flags]
├── 5. Regime Features                  13 features [NO scaling - categorical]
└── 6. Account State Features           3 features  [NO scaling - pre-normalized]
                                       ─────────────
                                 Total: 88 features ← 全ドキュメントでこの値に統一
```

### 1.2. 正規化グループ分離（⚠️ CRITICAL - 再レビュー修正）
**外部レビュー指摘**: 全特徴量に`OnlineScaler`を適用するとOne-Hot/Sin-Cos特徴量が歪む
**再レビュー指摘**: 実装例が矛盾していた → グループ別スケーラーに修正

```python
# ⚠️ 再レビュー修正: 正規化グループの明確化と実装例の整合
NORMALIZATION_GROUPS = {
    "online_zscore": {
        "features": ["base_features", "global_continuous"],
        "count": 30 + 6,  # = 36 features
        "method": "OnlineZScoreScaler",
    },
    "no_scaling": {
        "features": [
            "mtf_features",      # 27 (pre-normalized to [-1, 1] or categorical)
            "cyclical_time",     # 6 (sin/cos, already [-1, 1])
            "regime_onehot",     # 10 (0/1)
            "regime_continuous", # 3 (already [0, 1])
            "global_flags",      # 3 (0/1 flags)
            "account_state",     # 3 (pre-normalized)
        ],
        "count": 27 + 6 + 10 + 3 + 3 + 3,  # = 52 features
        "method": None,  # NO SCALING
    },
}
# Total: 36 + 52 = 88 features ✓


class GroupedFeatureScaler:
    """
    ⚠️ 再レビュー修正: グループ別スケーラー実装例
    
    全体に単一スケーラーを適用するのではなく、グループごとに処理を分離
    """
    
    def __init__(self):
        # OnlineScaler対象のみ
        self.online_scaler = OnlineZScoreScaler(
            window_size=100,
            feature_indices=list(range(36)),  # base(30) + global_continuous(6)
        )
    
    def transform(self, obs: np.ndarray) -> np.ndarray:
        """
        観測ベクトルを正規化
        
        Args:
            obs: [88] の生観測ベクトル
                [0:30]   - base_features (要スケーリング)
                [30:57]  - mtf_features (スキップ)
                [57:63]  - cyclical_time (スキップ)
                [63:72]  - global_market (6連続をスケーリング + 3フラグはスキップ)
                [72:85]  - regime_features (スキップ)
                [85:88]  - account_state (スキップ)
        
        Returns:
            scaled_obs: [88] スケーリング済みベクトル
        """
        scaled = obs.copy()
        
        # ⚠️ base_features (0:30) をオンラインZスコア
        scaled[0:30] = self.online_scaler.transform_partial(obs[0:30])
        
        # ⚠️ global_continuous (63:69) をオンラインZスコア
        # global_flags (69:72) はスキップ
        scaled[63:69] = self.online_scaler.transform_partial(obs[63:69])
        
        # 他のグループはそのまま（NO SCALING）
        # mtf_features, cyclical_time, regime, account_state
        
        return scaled
```

### 1.3. 実装ファイルマッピング
| カテゴリ | 実装ファイル | ステータス |
|---------|------------|----------|
| Base Features | `ztb/features/generators/technical/` | ✅ 既存 |
| MTF Features | `ztb/features/generators/multi_timeframe/engine.py` | ✅ 既存（要統合） |
| Cyclical Time | 新規作成 | 🆕 新規 |
| Global Market | `ztb/features/global_market.py` | ✅ 既存（要拡張） |
| Regime Features | `ztb/features/regime/` | ✅ 既存 |
| Account State | `FastIntradayEnv`内 | ✅ 既存 |

---

## 2. Base Features（1分足特徴量）

### 2.1. 価格系特徴量
| Feature Name | Formula | Description | Range |
|-------------|---------|-------------|-------|
| `log_return` | `log(close_t / close_{t-1})` | 対数リターン | (-∞, +∞) |
| `log_return_5` | `log(close_t / close_{t-5})` | 5期間リターン | (-∞, +∞) |
| `log_return_20` | `log(close_t / close_{t-20})` | 20期間リターン | (-∞, +∞) |
| `high_low_ratio` | `(high - low) / close` | レンジ比率 | [0, +∞) |
| `close_position` | `(close - low) / (high - low)` | レンジ内位置 | [0, 1] |

### 2.2. ボラティリティ系特徴量
| Feature Name | Formula | Description | Range |
|-------------|---------|-------------|-------|
| `atr_14` | `ATR(14)` | 14期間ATR | [0, +∞) |
| `atr_ratio` | `atr_14 / close` | ATR比率（正規化） | [0, +∞) |
| `vol_zscore` | `(atr - atr_ma) / atr_std` | ボラティリティZ-Score | (-∞, +∞) |
| `bb_width` | `(upper - lower) / middle` | ボリンジャーバンド幅 | [0, +∞) |
| `bb_position` | `(close - lower) / (upper - lower)` | BB内位置 | [0, 1] |

### 2.3. モメンタム系特徴量
| Feature Name | Formula | Description | Range |
|-------------|---------|-------------|-------|
| `rsi_14` | `RSI(14)` | RSI | [0, 100] |
| `rsi_zscore` | `(rsi - 50) / rsi_std` | RSI Z-Score | (-∞, +∞) |
| `macd_signal` | `MACD - Signal` | MACDシグナル差 | (-∞, +∞) |
| `macd_hist_slope` | `macd_hist_t - macd_hist_{t-1}` | ヒストグラム傾き | (-∞, +∞) |
| `adx_14` | `ADX(14)` | トレンド強度 | [0, 100] |
| `di_diff` | `(+DI - -DI) / 100` | DI差分 | [-1, 1] |

### 2.4. 出来高系特徴量
| Feature Name | Formula | Description | Range |
|-------------|---------|-------------|-------|
| `volume_ratio` | `volume / volume_ma_20` | 出来高比率 | [0, +∞) |
| `volume_zscore` | `(vol - vol_ma) / vol_std` | 出来高Z-Score | (-∞, +∞) |
| `obv_slope` | `OBV_t - OBV_{t-5}` | OBV傾き | (-∞, +∞) |
| `vwap_diff` | `(close - vwap) / close` | VWAP乖離 | (-∞, +∞) |

### 2.5. 正規化ルール
```python
BASE_FEATURE_NORMALIZATION = {
    "log_return*": "none",          # Already normalized
    "atr_ratio": "none",            # Already ratio
    "vol_zscore": "none",           # Already z-score
    "rsi_*": "minmax_0_100",        # Scale to [0, 1]
    "bb_*": "none",                 # Already [0, 1]
    "adx_*": "minmax_0_100",        # Scale to [0, 1]
    "volume_*": "clip_zscore_3",    # Clip at 3σ, then z-score
    "default": "online_zscore",     # Rolling z-score
}
```

---

## 3. MTF Features（マルチタイムフレーム特徴量）

### 3.1. 各タイムフレームで計算する特徴量
```python
MTF_FEATURES_PER_TIMEFRAME = [
    # Trend Direction
    "ema_cross_direction",    # EMA9 vs EMA21 クロス方向 {-1, 0, +1}
    "ema_slope",              # EMA21の傾き（正規化）
    "price_vs_ema",           # 価格とEMA21の位置関係
    
    # Volatility Context
    "atr_ratio_to_1m",        # ATR比率（対1分足）
    "bb_width_ratio",         # BB幅比率
    
    # Support/Resistance
    "distance_to_high_20",    # 20期間高値までの距離
    "distance_to_low_20",     # 20期間安値までの距離
    
    # Momentum
    "rsi_level",              # RSI（カテゴリ: oversold/neutral/overbought）
    "adx_strength",           # ADX強度（カテゴリ: weak/moderate/strong）
]
```

### 3.2. タイムフレーム別設定
| Timeframe | Window Short | Window Mid | Window Long | 用途 |
|-----------|-------------|-----------|-------------|------|
| 5m | 5 | 10 | 20 | 短期トレンド確認 |
| 15m | 4 | 8 | 16 | 中期トレンド確認 |
| 1h | 4 | 12 | 24 | 長期トレンド方向 |

### ⚠️ 3.3. CRITICAL: MTFバーアライメント（未来データリーク防止）

> **外部レビュー指摘**: 5分足が閉じる前に5分足のローソク情報を使うと未来データリークになる
> **再レビュー指摘**: 境界判定が恒等で常に1本前に戻る問題、"5min"と"5m"の表記混在

#### 3.3.0. クローズドバーのみ使用ルール
```python
# ⚠️ タイムフレーム表記は必ず "5min", "15min", "1h" に統一（pd.Timedelta互換）
TIMEFRAME_MAPPING = {
    "5m": "5min",    # 使用禁止 → "5min" に変換
    "15m": "15min",  # 使用禁止 → "15min" に変換
    "1h": "1h",      # OK
}

def normalize_timeframe(tf: str) -> str:
    """タイムフレーム表記を統一"""
    return TIMEFRAME_MAPPING.get(tf, tf)


def get_mtf_closed_bar(
    current_1m_timestamp: pd.Timestamp,
    mtf_timeframe: str,
    mtf_data: pd.DataFrame
) -> pd.Series:
    """
    ⚠️ CRITICAL: 必ず確定済みのバーのみを使用する
    
    Args:
        current_1m_timestamp: 現在の1分足タイムスタンプ（例: 2024-01-15 10:07:00）
        mtf_timeframe: "5min", "15min", "1h" ※"5m"等は使用禁止
        mtf_data: リサンプル済みMTFデータ（インデックスはバー開始時刻）
    
    Returns:
        最新の確定済みバー（現在時刻より前のバー）
    
    Example:
        - current_1m = 10:07:00
        - 5分足: 10:05:00のバーを使用（10:05-10:10の期間だが、10:05開始=10:00-10:05確定後）
        - 実際は「10:00-10:05」のデータなので10:00のバーを使用
    
    ⚠️ 再レビュー修正: 
        - 境界判定の恒等条件を修正
        - asof()で直近確定バーを安全に取得
        - 表記を"5min"に統一
    """
    # タイムフレーム表記の統一
    mtf_timeframe = normalize_timeframe(mtf_timeframe)
    
    # 現在時刻をMTFタイムフレームでフロア
    floored_time = current_1m_timestamp.floor(mtf_timeframe)
    
    # ⚠️ 修正: 現在時刻がバー開始時刻と一致する場合のみ1本前を使用
    # 例: 10:05:00ちょうどの場合、10:05バーはまだ未確定なので10:00バーを使用
    if current_1m_timestamp == floored_time:
        # バー開始時刻ちょうど → 1本前のバーを使用
        target_bar_time = floored_time - pd.Timedelta(mtf_timeframe)
    else:
        # バー途中 → 現在のフロア時刻（=最新確定バー）を使用
        # 例: 10:07:00 → floor → 10:05:00 → 10:00-10:05が確定済み
        # ⚠️ 注意: リサンプルの慣例によりインデックスがバー開始時刻か終了時刻か確認必要
        # ここでは開始時刻（left-labeled）を想定
        target_bar_time = floored_time - pd.Timedelta(mtf_timeframe)
    
    # ⚠️ 再レビュー修正: asof()で直近の有効バーを取得（欠損対応）
    if target_bar_time in mtf_data.index:
        return mtf_data.loc[target_bar_time]
    else:
        # データ欠損時はasof()で直近の確定バーを取得
        nearest_bar = mtf_data.index.asof(target_bar_time)
        if pd.isna(nearest_bar):
            logger.warning(f"MTF bar not found and no prior data: {target_bar_time}")
            return pd.Series({col: np.nan for col in mtf_data.columns})
        
        logger.warning(f"MTF bar {target_bar_time} not found, using {nearest_bar}")
        return mtf_data.loc[nearest_bar]


def get_mtf_features_safe(
    current_timestamp: pd.Timestamp,
    mtf_5min_data: pd.DataFrame,
    mtf_15min_data: pd.DataFrame,
    mtf_1h_data: pd.DataFrame,
) -> dict[str, float]:
    """
    ⚠️ CRITICAL: リークなしでMTF特徴量を取得
    
    バックテスト時にもライブ時にも同じロジックを使用することで
    train-liveの乖離を防ぐ
    
    ⚠️ 再レビュー修正: 引数名も"5min"表記に統一
    """
    features = {}
    
    # ⚠️ タイムフレーム表記は"5min"に統一
    for tf, data in [("5min", mtf_5min_data), ("15min", mtf_15min_data), ("1h", mtf_1h_data)]:
        bar = get_mtf_closed_bar(current_timestamp, tf, data)
        
        # 特徴量名は短縮形を維持（互換性のため）
        tf_short = tf.replace("min", "m")  # "5min" → "5m" for feature names only
        for col in bar.index:
            features[f"mtf_{tf_short}_{col}"] = bar[col]
    
    return features
```

#### 3.3.0.1. リーク検出テスト
```python
def test_mtf_no_future_leak():
    """MTF特徴量が未来データを使用していないことを検証"""
    
    # ⚠️ 再レビュー修正: 複数のエッジケースをテスト
    
    # 5分足データ（バー開始時刻でインデックス）
    mtf_5min = pd.DataFrame({
        "close": [100, 101, 102, 103],
    }, index=pd.to_datetime([
        "2024-01-15 09:50:00",  # 09:50-09:55の確定データ
        "2024-01-15 09:55:00",  # 09:55-10:00の確定データ
        "2024-01-15 10:00:00",  # 10:00-10:05の確定データ
        "2024-01-15 10:05:00",  # 10:05-10:10の確定データ（未確定の可能性）
    ]).tz_localize("UTC"))
    
    # Case 1: バー途中（10:07:00）
    ts1 = pd.Timestamp("2024-01-15 10:07:00", tz="UTC")
    bar1 = get_mtf_closed_bar(ts1, "5min", mtf_5min)
    assert bar1.name == pd.Timestamp("2024-01-15 10:00:00", tz="UTC"), "10:07 should use 10:00 bar"
    assert bar1["close"] == 102
    
    # Case 2: バー開始時刻ちょうど（10:05:00）
    ts2 = pd.Timestamp("2024-01-15 10:05:00", tz="UTC")
    bar2 = get_mtf_closed_bar(ts2, "5min", mtf_5min)
    assert bar2.name == pd.Timestamp("2024-01-15 10:00:00", tz="UTC"), "10:05 exactly should use 10:00 bar"
    assert bar2["close"] == 102
    
    # Case 3: 00:00境界（日跨ぎ）
    ts3 = pd.Timestamp("2024-01-16 00:02:00", tz="UTC")
    mtf_midnight = pd.DataFrame({
        "close": [200, 201],
    }, index=pd.to_datetime([
        "2024-01-15 23:55:00",
        "2024-01-16 00:00:00",  # 未確定の可能性
    ]).tz_localize("UTC"))
    bar3 = get_mtf_closed_bar(ts3, "5min", mtf_midnight)
    assert bar3.name == pd.Timestamp("2024-01-15 23:55:00", tz="UTC"), "00:02 should use 23:55 bar"
    
    # Case 4: データ欠損時（asof使用）
    ts4 = pd.Timestamp("2024-01-15 10:12:00", tz="UTC")
    # 10:05バーが欠損している場合
    mtf_gap = pd.DataFrame({
        "close": [100, 102],
    }, index=pd.to_datetime([
        "2024-01-15 09:55:00",
        "2024-01-15 10:00:00",
        # 10:05バー欠損
    ]).tz_localize("UTC"))
    bar4 = get_mtf_closed_bar(ts4, "5min", mtf_gap)
    assert bar4.name == pd.Timestamp("2024-01-15 10:00:00", tz="UTC"), "Should fallback to 10:00 via asof"
```

### 3.3.1. EMAクロス方向
```python
def calc_ema_cross_direction(df: pd.DataFrame, short: int = 9, long: int = 21) -> int:
    """
    EMAクロス方向を計算
    
    Returns:
        +1: Golden Cross (Short > Long, 上向き)
        -1: Dead Cross (Short < Long, 下向き)
         0: 不明確（クロス直後）
    """
    ema_short = df["close"].ewm(span=short).mean()
    ema_long = df["close"].ewm(span=long).mean()
    
    diff = ema_short.iloc[-1] - ema_long.iloc[-1]
    diff_prev = ema_short.iloc[-2] - ema_long.iloc[-2]
    
    if diff > 0 and diff_prev > 0:
        return 1  # Established uptrend
    elif diff < 0 and diff_prev < 0:
        return -1  # Established downtrend
    else:
        return 0  # Recent cross, unclear
```

#### 3.3.2. EMAスロープ（正規化）
```python
def calc_ema_slope_normalized(df: pd.DataFrame, period: int = 21, lookback: int = 5) -> float:
    """
    EMAの傾きを正規化して計算
    
    Returns:
        slope: [-1, +1] の範囲に正規化された傾き
    """
    ema = df["close"].ewm(span=period).mean()
    
    # 5期間での変化率
    slope = (ema.iloc[-1] - ema.iloc[-lookback]) / ema.iloc[-lookback]
    
    # ATRで正規化
    atr = df["atr"].iloc[-1] if "atr" in df.columns else df["close"].iloc[-1] * 0.01
    normalized_slope = slope / (atr / df["close"].iloc[-1] + 1e-8)
    
    # [-1, +1]にクリップ
    return np.clip(normalized_slope, -1.0, 1.0)
```

#### 3.3.3. 上位足との乖離
```python
def calc_price_vs_ema(close: float, ema: float) -> float:
    """
    価格とEMAの位置関係
    
    Returns:
        position: [-1, +1]
            +1: 価格がEMAを大きく上回る
            -1: 価格がEMAを大きく下回る
             0: EMA付近
    """
    diff_pct = (close - ema) / ema
    
    # 2%以上の乖離で±1
    return np.clip(diff_pct / 0.02, -1.0, 1.0)
```

### 3.4. MTF特徴量リスト（完全版）
```python
MTF_FEATURE_NAMES = [
    # 5分足
    "mtf_5m_ema_cross_dir",       # {-1, 0, +1}
    "mtf_5m_ema_slope",           # [-1, +1]
    "mtf_5m_price_vs_ema",        # [-1, +1]
    "mtf_5m_atr_ratio",           # [0, +∞) typically [0.5, 2.0]
    "mtf_5m_bb_width_ratio",      # [0, +∞)
    "mtf_5m_dist_high_20",        # [-1, 0] negative distance
    "mtf_5m_dist_low_20",         # [0, +1] positive distance
    "mtf_5m_rsi_cat",             # {-1, 0, +1} (oversold/neutral/overbought)
    "mtf_5m_adx_cat",             # {0, 1, 2} (weak/moderate/strong)
    
    # 15分足
    "mtf_15m_ema_cross_dir",
    "mtf_15m_ema_slope",
    "mtf_15m_price_vs_ema",
    "mtf_15m_atr_ratio",
    "mtf_15m_bb_width_ratio",
    "mtf_15m_dist_high_20",
    "mtf_15m_dist_low_20",
    "mtf_15m_rsi_cat",
    "mtf_15m_adx_cat",
    
    # 1時間足
    "mtf_1h_ema_cross_dir",
    "mtf_1h_ema_slope",
    "mtf_1h_price_vs_ema",
    "mtf_1h_atr_ratio",
    "mtf_1h_bb_width_ratio",
    "mtf_1h_dist_high_20",
    "mtf_1h_dist_low_20",
    "mtf_1h_rsi_cat",
    "mtf_1h_adx_cat",
]
# Total: 27 features
```

---

## 4. Cyclical Time Features（周期的時刻特徴量）

### 4.1. 設計理由（v451教訓）
1分足での取引において、特定の時間帯（14:00, 17:00, 01:00 JST）で損失が集中することが判明。
時刻情報を周期関数でエンコードすることで、エージェントに「時間の概念」を与える。

### ⚠️ 4.2. CRITICAL: タイムゾーン処理ルール

> **外部レビュー指摘**: naive timestampの扱いが曖昧で、ライブとバックテストで挙動が異なる可能性

```python
def validate_and_convert_timestamp(timestamp: pd.Timestamp, require_tz: bool = True) -> pd.Timestamp:
    """
    ⚠️ CRITICAL: タイムゾーン処理の一貫性を保証
    
    Args:
        timestamp: 入力タイムスタンプ
        require_tz: Trueの場合、naive timestampでエラー
    
    Returns:
        JST変換済みタイムスタンプ
    
    Raises:
        ValueError: require_tz=True かつ naive timestamp の場合
    """
    if timestamp.tzinfo is None:
        if require_tz:
            raise ValueError(
                f"Naive timestamp detected: {timestamp}. "
                "All timestamps must be timezone-aware. "
                "Use pd.Timestamp(..., tz='UTC') or .tz_localize('UTC')"
            )
        # Fallback: naive timestampをUTCとして扱う（警告ログ出力）
        logger.warning(f"Naive timestamp {timestamp} treated as UTC")
        timestamp = timestamp.tz_localize("UTC")
    
    return timestamp.tz_convert("Asia/Tokyo")


def calc_cyclical_time_features(timestamp: pd.Timestamp) -> Dict[str, float]:
    """
    周期的時刻特徴量を計算
    
    Args:
        timestamp: timezone-aware timestamp (UTC recommended)
    
    Returns:
        dict with 6 features (all in [-1, 1] range, NO normalization needed)
    """
    # ⚠️ CRITICAL: タイムゾーン検証を必ず実施
    jst = validate_and_convert_timestamp(timestamp, require_tz=True)
    
    hour = jst.hour
    minute = jst.minute
    day_of_week = jst.dayofweek  # 0=Monday, 6=Sunday
    
    features = {
        # Hour encoding (24時間周期) - 出力: [-1, 1]
        "time_hour_sin": np.sin(2 * np.pi * hour / 24),
        "time_hour_cos": np.cos(2 * np.pi * hour / 24),
        
        # Minute encoding (60分周期) - 出力: [-1, 1]
        "time_minute_sin": np.sin(2 * np.pi * minute / 60),
        "time_minute_cos": np.cos(2 * np.pi * minute / 60),
        
        # Day of week encoding (7日周期) - 出力: [-1, 1]
        "time_dow_sin": np.sin(2 * np.pi * day_of_week / 7),
        "time_dow_cos": np.cos(2 * np.pi * day_of_week / 7),
    }
    
    return features
    # ⚠️ NOTE: これらの特徴量は OnlineScaler の対象外とすること
```

### 4.3. 危険時間帯マッピング
```python
# これらの時間帯は時刻特徴量を通じてエージェントが学習すべき
DANGER_HOURS_JST = {
    14: "Pre-European Trap（流動性低下→ブレイクアウト）",
    17: "London Open（ストップハント、フェイクアウト）",
    1: "London Fix（大口フローによる乱高下）",
    7: "Tokyo Open（方向感なし、ホイップソー）",
}

# Soft Filterで補完的に制限
TIME_RESTRICTION_CONFIG = {
    14: {"position_mult": 0.5, "threshold_mod": +0.2},
    17: {"position_mult": 0.3, "threshold_mod": +0.3},
    1: {"position_mult": 0.3, "threshold_mod": +0.3},
    7: {"position_mult": 0.7, "threshold_mod": +0.1},
}
```

---

## 5. Global Market Features（グローバル市場特徴量）

### 5.1. 設計理由（v449教訓）
小規模取引所（Zaif等）の価格は、主要取引所（Binance等）に追随するLead-Lag効果が存在。
この先行情報を特徴量として取り入れることで、数秒〜数分の予測優位性を獲得。

### ⚠️ 5.2. CRITICAL: FX/ベーシスリスク考慮（外部レビュー指摘）

> **外部レビュー指摘**: BTC/USDTとBTC/JPYの比較では為替変動を無視している

```python
GLOBAL_MARKET_FEATURES = [
    # Lead-Lag Features (FX調整済み)
    "global_btc_return_1m",     # Binance BTC/USDT 1分リターン
    "global_btc_return_5m",     # Binance BTC/USDT 5分リターン
    "global_return_spread",     # Local - Global リターン差 (FX調整後)
    
    # ⚠️ NEW: FX/Basis Features
    "global_usdjpy_return_1m",  # USD/JPY 1分リターン（為替ヘッジ情報）
    "global_usdt_premium",      # USDT/USD プレミアム（Tether乖離）
    "global_fx_adjusted_spread",# FX調整済みのLocal-Global乖離
    
    # Correlation Features
    "global_corr_5m",           # 5分ローリング相関
    "global_corr_15m",          # 15分ローリング相関
    
    # Market Sentiment
    "global_funding_rate",      # Binance Futures Funding Rate (8h更新)
    
    # ⚠️ NEW: Data Quality Flags (外部レビュー指摘)
    "global_data_stale_flag",   # 0/1: データが古い場合1
    "global_usdjpy_stale_flag", # 0/1: FXデータが古い場合1
    "global_api_error_flag",    # 0/1: API取得失敗時1
]
# ⚠️ 再レビュー修正: Total: 9 features (6 continuous + 3 flags) に統一
# 連続値: btc_return_1m, btc_return_5m, fx_adjusted_spread, usdjpy_return_1m, corr_5m, funding_rate
# フラグ: data_stale, usdjpy_stale, api_error
# 注: global_return_spread と global_fx_adjusted_spread の重複は削除（fx_adjusted_spreadのみ使用）
# 注: global_usdt_premium と global_corr_15m は削除（情報冗長）
```

### ⚠️ 5.2.1. 再レビュー修正: 特徴量数の整理
```python
# ⚠️ 再レビュー指摘: 定義と生成の不整合を解消
GLOBAL_MARKET_FEATURES_FINAL = {
    # 連続値（6 features） → OnlineScaler対象
    "continuous": [
        "global_btc_return_1m",      # Binance BTC/USDT 1分リターン
        "global_btc_return_5m",      # Binance BTC/USDT 5分リターン
        "global_fx_adjusted_spread", # FX調整済みLocal-Global乖離
        "global_usdjpy_return_1m",   # USD/JPY 1分リターン
        "global_corr_5m",            # 5分ローリング相関
        "global_funding_rate",       # Funding Rate
    ],
    # フラグ値（3 features） → スケーリング対象外
    "flags": [
        "global_data_stale_flag",    # 0/1
        "global_usdjpy_stale_flag",  # 0/1
        "global_api_error_flag",     # 0/1
    ],
    # Total: 9 features
}

# 削除した特徴量と理由
REMOVED_FEATURES = {
    "global_return_spread": "global_fx_adjusted_spreadと重複",
    "global_usdt_premium": "情報としてfx_adjusted_spreadに内包",
    "global_corr_15m": "global_corr_5mで十分、次元削減",
}
```

### 5.3. FX調整ロジック
```python
def calc_fx_adjusted_spread(
    local_btcjpy_close: float,
    global_btcusdt_close: float,
    usdjpy_rate: float,
    usdt_premium: float = 1.0,
) -> float:
    """
    FX調整済みのLocal-Globalスプレッドを計算
    
    Args:
        local_btcjpy_close: Zaif BTC/JPY価格
        global_btcusdt_close: Binance BTC/USDT価格
        usdjpy_rate: USD/JPY為替レート
        usdt_premium: USDT/USDプレミアム（通常1.0前後）
    
    Returns:
        fx_adjusted_spread: 為替調整後の乖離率
    
    ⚠️ 再レビュー指摘: 取引所固有プレミアム、手数料差も考慮すべき
    → 将来拡張としてexchange_premiumパラメータ追加を検討
    """
    # BTC/USDTをJPYに換算
    global_btcjpy_equivalent = global_btcusdt_close * usdjpy_rate * usdt_premium
    
    # 乖離率を計算
    spread = (local_btcjpy_close - global_btcjpy_equivalent) / global_btcjpy_equivalent
    
    return spread
```

### 5.4. データ鮮度フラグ実装と活用（⚠️ 再レビュー修正）
```python
def check_data_staleness(
    data_timestamp: pd.Timestamp,
    current_timestamp: pd.Timestamp,
    max_age_seconds: int = 60,
) -> int:
    """
    データの鮮度をチェック
    
    Args:
        data_timestamp: データのタイムスタンプ
        current_timestamp: 現在時刻
        max_age_seconds: 許容遅延秒数
    
    Returns:
        0: Fresh data
        1: Stale data (フォールバック使用中)
    """
    age = (current_timestamp - data_timestamp).total_seconds()
    return 1 if age > max_age_seconds else 0


# ⚠️ 再レビュー指摘: staleフラグの活用方法を明確化
def handle_stale_global_features(
    features: dict[str, float],
    stale_flags: dict[str, int],
) -> dict[str, float]:
    """
    ⚠️ 再レビュー回答: stale時のGlobal特徴量処理
    
    stale=1の場合、連続値特徴量を0化して重みを低減
    Calibration Gate側で閾値を強化（別途実装）
    """
    processed = features.copy()
    
    if stale_flags.get("global_data_stale_flag", 0) == 1:
        # Global BTC関連を0化
        processed["global_btc_return_1m"] = 0.0
        processed["global_btc_return_5m"] = 0.0
        processed["global_fx_adjusted_spread"] = 0.0
        processed["global_corr_5m"] = 0.0
    
    if stale_flags.get("global_usdjpy_stale_flag", 0) == 1:
        # FX関連を0化
        processed["global_usdjpy_return_1m"] = 0.0
        # fx_adjusted_spreadはglobal_return_spreadにフォールバック
    
    return processed
```

### 5.5. 実装詳細（⚠️ 再レビュー修正: 生成フロー整合）
```python
class GlobalMarketFeatureEngineerV456(GlobalMarketFeatureEngineer):
    """
    v456拡張版グローバル市場特徴量エンジニア
    
    ⚠️ 再レビュー修正: 定義された9特徴量すべてを生成するよう整合
    """
    
    def generate_features(
        self,
        local_df: pd.DataFrame,
        global_df: pd.DataFrame,
        usdjpy_df: pd.DataFrame = None,
        current_timestamp: pd.Timestamp = None,
    ) -> pd.DataFrame:
        """
        グローバル市場特徴量を生成
        
        Args:
            local_df: ローカル取引所データ（Zaif等）
            global_df: グローバル取引所データ（Binance）
            usdjpy_df: USD/JPY為替データ（オプション）
            current_timestamp: 現在時刻（鮮度チェック用）
        
        Returns:
            features: 9特徴量を含むDataFrame
        """
        result = pd.DataFrame(index=local_df.index)
        
        # 1. データマージ
        merged = self.merge_external_data(
            local_df, global_df, suffix="_global", fill_method="ffill"
        )
        
        # 2. ⚠️ 連続値特徴量（6 features）
        # 2-1. BTC returns
        result["global_btc_return_1m"] = global_df["close"].pct_change(1)
        result["global_btc_return_5m"] = global_df["close"].pct_change(5)
        
        # 2-2. FX調整スプレッド
        if usdjpy_df is not None:
            merged = merged.join(usdjpy_df[["close"]].rename(columns={"close": "usdjpy"}))
            result["global_fx_adjusted_spread"] = merged.apply(
                lambda row: calc_fx_adjusted_spread(
                    row["close"], row["close_global"], row.get("usdjpy", 150.0)
                ),
                axis=1
            )
            result["global_usdjpy_return_1m"] = usdjpy_df["close"].pct_change(1)
        else:
            # FXデータなし: 単純スプレッド + 0埋め
            result["global_fx_adjusted_spread"] = (
                (local_df["close"] - global_df["close"] * 150) / (global_df["close"] * 150)
            )  # 仮の為替レート150円
            result["global_usdjpy_return_1m"] = 0.0
        
        # 2-3. 相関
        result["global_corr_5m"] = (
            local_df["close"].pct_change()
            .rolling(5).corr(global_df["close"].pct_change())
        )
        
        # 2-4. Funding Rate
        if "funding_rate" in merged.columns:
            result["global_funding_rate"] = merged["funding_rate"]
        else:
            result["global_funding_rate"] = 0.0
        
        # 3. ⚠️ フラグ特徴量（3 features）- ライブ時に動的更新
        result["global_data_stale_flag"] = 0
        result["global_usdjpy_stale_flag"] = 0 if usdjpy_df is not None else 1
        result["global_api_error_flag"] = 0
        
        # 4. NaN処理
        result = result.fillna(0.0)
        
        return result
```

### 5.6. データ取得フロー
```
[Data Pipeline]
                                    
Binance API ─────┐                  
(BTC/USDT 1m)    │                  
                 │
USD/JPY API ─────┼──→ Data Merger ──→ Feature Engineer ──→ Observation
(外為ドットコム等)│                   ↓
                 │              [Data Quality Check]
Zaif API ────────┘                  ↓
(BTC/JPY 1m)               global_data_stale_flag = 0/1
                                    
[Latency Consideration]             
- Binance data: ~500ms delay        
- Zaif data: ~1000ms delay
- FX data: ~1000ms delay (週末注意)
- Net lead: ~500ms advantage        
```

---

## 6. Regime Features（レジーム特徴量）

### 6.1. レジーム分類
```python
MARKET_REGIMES = {
    0: "strong_bull_trend",      # 強い上昇トレンド
    1: "weak_bull_trend",        # 弱い上昇トレンド
    2: "strong_bear_trend",      # 強い下降トレンド
    3: "weak_bear_trend",        # 弱い下降トレンド
    4: "ranging",                # レンジ相場
    5: "sideways",               # 横ばい
    6: "high_volatility_ranging",# 高ボラレンジ（危険）
    7: "low_volatility",         # 低ボラ
    8: "breakout_bull",          # 上方ブレイクアウト
    9: "breakout_bear",          # 下方ブレイクアウト
}
```

### 6.2. レジーム特徴量
```python
REGIME_FEATURES = [
    # One-Hot Encoding (10 regimes)
    "regime_0", "regime_1", "regime_2", "regime_3", "regime_4",
    "regime_5", "regime_6", "regime_7", "regime_8", "regime_9",
    
    # Continuous Regime Indicators
    "vol_rank",        # [0, 1] ボラティリティ順位
    "vol_ratio",       # 短期/長期ボラティリティ比率
    "trend_strength",  # [0, 1] ADXベースのトレンド強度
]
# Total: 13 features (10 one-hot + 3 continuous)
```

### 6.3. レジーム判定ロジック
```python
def classify_regime(
    adx: float,
    di_plus: float,
    di_minus: float,
    vol_ratio: float,
    price_vs_bb: float,
) -> int:
    """
    市場レジームを分類
    
    Args:
        adx: ADX値 [0, 100]
        di_plus: +DI値 [0, 100]
        di_minus: -DI値 [0, 100]
        vol_ratio: 短期/長期ボラ比率
        price_vs_bb: BB内位置 [0, 1]
    
    Returns:
        regime_id: 0-9
    """
    trend_dir = di_plus - di_minus
    is_trending = adx > 25
    is_strong_trend = adx > 40
    is_high_vol = vol_ratio > 1.5
    is_low_vol = vol_ratio < 0.5
    
    if is_strong_trend:
        if trend_dir > 0:
            return 0  # strong_bull_trend
        else:
            return 2  # strong_bear_trend
    elif is_trending:
        if trend_dir > 0:
            return 1  # weak_bull_trend
        else:
            return 3  # weak_bear_trend
    elif is_high_vol:
        return 6  # high_volatility_ranging (DANGER)
    elif is_low_vol:
        return 7  # low_volatility
    elif price_vs_bb > 0.9 and trend_dir > 0:
        return 8  # breakout_bull
    elif price_vs_bb < 0.1 and trend_dir < 0:
        return 9  # breakout_bear
    elif abs(trend_dir) < 5:
        return 5  # sideways
    else:
        return 4  # ranging
```

---

## 7. Account State Features（アカウント状態特徴量）

### 7.1. 特徴量定義
```python
ACCOUNT_STATE_FEATURES = [
    "position_norm",       # 現在ポジション / max_position [-1, +1]
    "remaining_ttl_norm",  # 残りTTL / max_ttl [0, 1]
    "last_cost_norm",      # 直近コスト / denom [0, +∞)
]
```

### 7.2. 計算ロジック（FastIntradayEnv内）
```python
def _get_account_state(self) -> np.ndarray:
    """アカウント状態特徴量を取得"""
    
    # Position normalized to [-1, +1]
    position_norm = self.position / self.max_position
    
    # TTL normalized to [0, 1]
    remaining_ttl_norm = max(0, self.position_ttl) / self.max_ttl_steps
    
    # Cost normalized by denominator
    denom = max(self.atr_data[self.current_step], 1.0) * self.max_position
    last_cost_norm = self.last_step_cost / denom
    
    return np.array([position_norm, remaining_ttl_norm, last_cost_norm], dtype=np.float32)
```

---

## 8. 特徴量パイプライン

### 8.1. 処理フロー
```
[Raw Data]
    │
    ▼
[1. Resampling]
    │  - 1m → 5m, 15m, 1h aggregation
    │
    ▼
[2. Technical Indicator Calculation]
    │  - EMA, RSI, ADX, ATR, BB for each timeframe
    │
    ▼
[3. Feature Generation]
    │  - Base features
    │  - MTF features
    │  - Global features
    │  - Regime features
    │  - Time features
    │
    ▼
[4. Normalization]
    │  - Online Z-Score
    │  - Min-Max
    │  - Clipping
    │
    ▼
[5. Feature Vector Assembly]
    │  - Concatenate all features
    │  - Add account state
    │
    ▼
[Observation]
```

### 8.2. 実装クラス設計
```python
class V456FeatureEngineer:
    """v456統合特徴量エンジニア"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Sub-engineers
        self.base_engineer = SACv427FeatureEngineer()
        self.mtf_engineer = MultiTimeframeFeatureEngineer()
        self.global_engineer = GlobalMarketFeatureEngineerV456()
        self.regime_classifier = MarketRegimeClassifier()
        
        # Normalization
        self.scaler = OnlineScaler(shape=(self.get_feature_dim(),))
    
    def generate_features(
        self,
        data_1m: pd.DataFrame,
        data_5m: pd.DataFrame,
        data_15m: pd.DataFrame,
        data_1h: pd.DataFrame,
        global_data: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> np.ndarray:
        """
        全特徴量を生成
        
        Returns:
            feature_vector: shape (feature_dim,)
        """
        # 1. Base features
        base = self.base_engineer.generate_features(data_1m)
        
        # 2. MTF features
        mtf = self.mtf_engineer.generate_features({
            "5m": data_5m,
            "15m": data_15m,
            "1h": data_1h,
        })
        
        # 3. Time features
        time_feat = calc_cyclical_time_features(timestamp)
        
        # 4. Global features
        global_feat = self.global_engineer.generate_features(data_1m, global_data)
        
        # 5. Regime features
        regime_feat = self.regime_classifier.classify_and_encode(data_1m)
        
        # 6. Concatenate
        feature_vector = np.concatenate([
            base.values[-1],
            mtf.values[-1],
            np.array(list(time_feat.values())),
            global_feat.values[-1],
            regime_feat,
        ])
        
        # 7. Normalize
        normalized = self.scaler.transform(feature_vector)
        
        return normalized
    
    def get_feature_dim(self) -> int:
        """特徴量次元数を取得"""
        return (
            30 +  # Base
            27 +  # MTF
            6 +   # Time
            6 +   # Global
            13 +  # Regime
            3     # Account (added separately in env)
        )  # = 85
```

---

## 9. 特徴量の重要度ガイダンス

### 9.1. 優先度分類
| Tier | Features | 理由 |
|------|----------|------|
| **Tier 1** (必須) | MTF EMA方向, 時刻エンコード, Vol Rank | 過去バージョンで効果実証済み |
| **Tier 2** (推奨) | Global Return, Regime One-Hot, Z-Score | 実装済み・低コスト |
| **Tier 3** (実験的) | Ichimoku, Elliott Wave, Harmonic | 計算コスト高・効果未検証 |

### 9.2. 特徴量選択基準
```python
FEATURE_SELECTION_CRITERIA = {
    "min_importance": 0.01,        # 最小重要度閾値
    "max_correlation": 0.95,       # 最大相関閾値（冗長排除）
    "min_variance": 0.001,         # 最小分散閾値
    "max_nan_ratio": 0.05,         # 最大欠損率
}
```

---

## 10. テスト要件

### 10.1. 特徴量単体テスト
```python
# tests/unit/features/test_v456_features.py

def test_mtf_features_shape():
    """MTF特徴量の次元数確認"""
    pass

def test_cyclical_time_range():
    """時刻特徴量が[-1, +1]の範囲内か確認"""
    pass

def test_regime_one_hot_sum():
    """One-Hotの合計が1であることを確認"""
    pass

def test_feature_no_nan():
    """特徴量にNaNがないことを確認"""
    pass

def test_feature_no_inf():
    """特徴量にInfがないことを確認"""
    pass
```

### 10.2. 特徴量品質チェック
```python
def validate_feature_quality(features: np.ndarray) -> Dict[str, bool]:
    """特徴量品質バリデーション"""
    return {
        "no_nan": not np.isnan(features).any(),
        "no_inf": not np.isinf(features).any(),
        "reasonable_range": np.abs(features).max() < 100,
        "sufficient_variance": np.var(features) > 1e-6,
    }
```

---

## Appendix: 特徴量一覧表（完全版）

| # | Feature Name | Category | Type | Range | Normalization |
|---|-------------|----------|------|-------|---------------|
| 1 | log_return | Base | float | (-∞, +∞) | none |
| 2 | log_return_5 | Base | float | (-∞, +∞) | none |
| 3 | ... | ... | ... | ... | ... |
| 30 | vwap_diff | Base | float | (-∞, +∞) | clip_zscore |
| 31 | mtf_5m_ema_cross_dir | MTF | int | {-1, 0, +1} | none |
| 32 | mtf_5m_ema_slope | MTF | float | [-1, +1] | none |
| ... | ... | ... | ... | ... | ... |
| 57 | mtf_1h_adx_cat | MTF | int | {0, 1, 2} | none |
| 58 | time_hour_sin | Time | float | [-1, +1] | none |
| 59 | time_hour_cos | Time | float | [-1, +1] | none |
| 60 | time_minute_sin | Time | float | [-1, +1] | none |
| 61 | time_minute_cos | Time | float | [-1, +1] | none |
| 62 | time_dow_sin | Time | float | [-1, +1] | none |
| 63 | time_dow_cos | Time | float | [-1, +1] | none |
| 64 | global_btc_return_1m | Global | float | (-∞, +∞) | clip_zscore |
| 65 | global_btc_return_5m | Global | float | (-∞, +∞) | clip_zscore |
| 66 | global_return_spread | Global | float | (-∞, +∞) | clip_zscore |
| 67 | global_corr_5m | Global | float | [-1, +1] | none |
| 68 | global_corr_15m | Global | float | [-1, +1] | none |
| 69 | global_funding_rate | Global | float | (-∞, +∞) | clip_zscore |
| 70-79 | regime_0 to regime_9 | Regime | int | {0, 1} | none |
| 80 | vol_rank | Regime | float | [0, 1] | none |
| 81 | vol_ratio | Regime | float | [0, +∞) | clip |
| 82 | trend_strength | Regime | float | [0, 1] | none |
| 83 | position_norm | Account | float | [-1, +1] | none |
| 84 | remaining_ttl_norm | Account | float | [0, 1] | none |
| 85 | last_cost_norm | Account | float | [0, +∞) | none |
