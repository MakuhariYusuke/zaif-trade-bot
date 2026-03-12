# Implementation Plan: Online Scaling & Multi-Timeframe System

## 1. 概要
本ドキュメントは、`docs/v455/05_backtest_analysis_and_creative_solutions.md` で提起された課題（データリーク、Alpha不足）に対する具体的解決策の実装計画である。
**「全期間統計によるスケーリング（リーク）」を廃止し、「Online Scaler」へ移行する**とともに、**「Multi-Timeframe (MTF) System」**を導入してモデルの予測精度（Alpha）を強化する。

---

## 2. Online Scaler Design (データリーク防止)

従来の「全期間の統計量を使ったスケーリング」は、未来の情報を現在に持ち込むデータリークの原因となる。これを防ぐため、**Welfordのアルゴリズム**を用いたオンラインスケーラーを導入する。

### 2.1 Python Class Structure: `OnlineScaler`

```python
import numpy as np
from typing import Optional, Union

class OnlineScaler:
    """
    Welford's algorithm implementation for online standardization.
    Computes running mean and variance without storing the full history.
    """
    def __init__(self, shape: tuple, epsilon: float = 1e-5, clip: float = 10.0):
        self.shape = shape
        self.epsilon = epsilon
        self.clip = clip

        # Statistics
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float32)
        self.M2 = np.zeros(shape, dtype=np.float32)  # Sum of squares of differences from the current mean

    def update(self, x: np.ndarray) -> None:
        """Update statistics with a new sample x."""
        if x.shape != self.shape:
            raise ValueError(f"Shape mismatch: expected {self.shape}, got {x.shape}")

        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Standardize x using current statistics."""
        if self.n < 2:
            return np.zeros_like(x)  # Not enough data to scale

        var = self.M2 / (self.n - 1)
        std = np.sqrt(var)

        # Z-score normalization
        scaled = (x - self.mean) / (std + self.epsilon)

        # Clip to prevent extreme outliers from destabilizing the model
        if self.clip > 0:
            scaled = np.clip(scaled, -self.clip, self.clip)

        return scaled

    def partial_fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Update stats and then transform (convenience method)."""
        self.update(x)
        return self.transform(x)
```

### 2.2 Warm-up Handling
*   **Warm-up期間**: 環境の `warmup_steps` (例: 1000ステップ) の間は、`update` のみを実行し、モデルの推論結果は使用しない（またはランダムアクション）。
*   **事前ロード**: 可能であれば、過去の確定データ（直近1週間分など）を使ってスケーラーを初期化 (`pre_fit`) してからスタートする。

---

## 3. Multi-Timeframe Architecture (MTF)

1分足のステップごとに、上位足（5m, 15m, 1h）の特徴量をリアルタイムで生成するアーキテクチャ。

### 3.1 Data Buffer Management & Feature Generation
*   **確定足バッファ**: 過去の確定した上位足を保持。
*   **形成中足 (Forming Bar)**: 現在進行形で更新されている上位足。1分ごとにHigh/Low/Close/Volumeを更新。

### 3.2 Python Class Structure: `MultiTimeframeManager`

```python
import pandas as pd
import numpy as np
from collections import deque
from typing import Dict, List

class MultiTimeframeManager:
    def __init__(self, timeframes: List[str] = ['5m', '15m', '1h'], window_size: int = 100):
        self.timeframes = timeframes
        self.window_size = window_size

        # Timeframe definitions in minutes
        self.tf_minutes = {
            '5m': 5,
            '15m': 15,
            '1h': 60
        }

        # Buffers for completed bars (DataFrame or list of dicts)
        self.buffers: Dict[str, deque] = {
            tf: deque(maxlen=window_size) for tf in timeframes
        }

        # Currently forming bars
        self.forming_bars: Dict[str, dict] = {
            tf: None for tf in timeframes
        }

    def update(self, timestamp, open_, high, low, close, volume):
        """
        Called every 1-minute step with the latest 1m bar data.
        """
        current_time = pd.to_datetime(timestamp)

        for tf in self.timeframes:
            minutes = self.tf_minutes[tf]

            # Initialize forming bar if None
            if self.forming_bars[tf] is None:
                self.forming_bars[tf] = {
                    'timestamp': current_time,
                    'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume
                }
            else:
                # Update forming bar
                fb = self.forming_bars[tf]
                fb['high'] = max(fb['high'], high)
                fb['low'] = min(fb['low'], low)
                fb['close'] = close
                fb['volume'] += volume

            # Check if bar is completed (simplified check based on time)
            # Assuming timestamp is the close time of the 1m bar
            if current_time.minute % minutes == 0: # E.g., 10:05, 10:10
                # Commit forming bar to buffer
                self.buffers[tf].append(self.forming_bars[tf].copy())
                # Reset forming bar for next cycle
                self.forming_bars[tf] = None

    def get_features(self) -> np.ndarray:
        """
        Generate technical indicators for all timeframes based on current buffers.
        Returns a flattened numpy array of features.
        """
        features = []

        for tf in self.timeframes:
            if len(self.buffers[tf]) < 20: # Need minimum history for indicators
                # Return zeros or padding if not enough history
                features.extend([0.0] * 3) # Example: RSI, MACD, BB_Width
                continue

            # Convert buffer to DataFrame for calculation
            df = pd.DataFrame(list(self.buffers[tf]))

            # If we want to include the forming bar for "latest" view:
            if self.forming_bars[tf]:
                df = pd.concat([df, pd.DataFrame([self.forming_bars[tf]])], ignore_index=True)

            # Calculate Indicators (Example using pandas/numpy)
            # 1. RSI (14)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # 2. MACD
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26

            # 3. Bollinger Bands Width
            sma20 = df['close'].rolling(window=20).mean()
            std20 = df['close'].rolling(window=20).std()
            bb_width = (4 * std20) / sma20

            # Append latest values
            features.append(rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0)
            features.append(macd.iloc[-1] if not np.isnan(macd.iloc[-1]) else 0.0)
            features.append(bb_width.iloc[-1] if not np.isnan(bb_width.iloc[-1]) else 0.0)

        return np.array(features, dtype=np.float32)
```

### 3.3 Feature Fusion
生成されたMTF特徴量は、ベースとなる1分足の特徴量ベクトルと**結合（Concatenation）**する。
*   **Observation Space**: `[Base_Features (1m)] + [MTF_Features (5m, 15m, 1h)]`

---

## 4. Integration Plan

`HeavyTradingEnv` を修正し、これらのコンポーネントを統合する。

### 4.1 Modifications

1.  **初期化 (`__init__` / `_initialize_components`)**:
    *   `OnlineScaler` を初期化。
    *   `MultiTimeframeManager` を初期化。
    *   `observation_space` の次元数を、MTF特徴量の分だけ拡張する。

2.  **ステップ処理 (`step`)**:
    *   現在の1分足データ（OHLCV）を取得。
    *   `mtf_manager.update(...)` を呼び出し。
    *   `_get_observation` 内で `mtf_manager.get_features()` を呼び出し。

3.  **観測取得 (`_get_observation`)**:

```python
    def _get_observation(self) -> np.ndarray:
        # 1. Get Base Features (Existing logic, unscaled)
        base_features = self.data_manager.get_features_at_step(self.current_step)

        # 2. Get MTF Features
        mtf_features = self.mtf_manager.get_features()

        # 3. Concatenate
        raw_obs = np.concatenate([base_features, mtf_features])

        # 4. Online Scaling (Update & Transform)
        # Only update scaler during training or warm-up
        if self.training:
            self.online_scaler.update(raw_obs)

        scaled_obs = self.online_scaler.transform(raw_obs)

        return scaled_obs
```

### 4.2 Architectural Diagram (Mermaid)

```mermaid
graph TD
    subgraph Environment [HeavyTradingEnv]
        Step[Step(action)] --> Data[Get 1m OHLCV]
        Data --> MTF[MultiTimeframeManager]

        subgraph MTF_Logic
            MTF -->|Update| Buffers[Buffers: 5m, 15m, 1h]
            Buffers -->|Resample & Calc| Indicators[Calc RSI, MACD, BB]
        end

        Data --> BaseFeat[Base Feature Extractor]

        Indicators --> Concat[Concatenate]
        BaseFeat --> Concat

        Concat --> RawObs[Raw Observation Vector]
        RawObs --> Scaler[OnlineScaler (Welford)]
        Scaler --> Obs[Scaled Observation]
    end

    Obs --> Agent[SAC Agent / Gate]
    Agent --> Step
```
