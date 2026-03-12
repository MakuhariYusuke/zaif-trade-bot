# Codex Prompt: 366# T4+T9 scalping.py 計算高速化 (numpy vectorize)

## タスク概要

`ztb/features/scalping.py` の以下3関数を **for ループ → numpy vectorize** に書き換え、計算量を改善する。

## 対象ファイル

- `ztb/features/scalping.py` — 3関数を修正 (API・引数・戻り値は変更しない)
- `tests/unit/core/features/test_scalping_features.py` — 必要に応じてテスト追加

## 修正対象とアルゴリズム

### 1. `realized_volatility()` (L210-234) — O(n²) → O(n)

**現状**: 二重ループ (外ループ n 回 × 内ループ window 回で returns を毎回再計算)

**改善案**:
```python
@register("realized_volatility")
def realized_volatility(df: pd.DataFrame, window: int = 10) -> pd.Series:
    close = df["close"].values.astype(np.float64)
    rv = np.zeros(len(close), dtype=np.float64)
    # log returns (close[i-1]==0 のケースは 0.0 扱い)
    safe_close = np.where(close > 0, close, 1.0)
    log_returns = np.diff(np.log(safe_close))
    squared_returns = log_returns ** 2
    # rolling sum via cumsum
    cumsum = np.cumsum(squared_returns)
    # cumsum[window-2] = sum of first (window-1) squared returns
    rv[window] = np.sqrt(cumsum[window - 2])  # 修正: window-1 個の returns
    for i in range(window + 1, len(close)):
        rv[i] = np.sqrt(cumsum[i - 2] - cumsum[i - window - 1])
    # ↑ もしくは完全 vectorize:
    # rv[window:] = np.sqrt(cumsum[window-2:] - np.concatenate([[0], cumsum[:len(close)-window-1]]))
    return pd.Series(rv, index=df.index, name="realized_volatility")
```

**重要**: `close[idx - 1] == 0` の場合に returns を 0.0 にする既存の挙動を保つこと。

### 2. `order_flow_imbalance()` (L125-143) — for ループ → numpy vectorize

**現状**: 1重ループだが numpy 化可能

**改善案**:
```python
@register("order_flow_imbalance")
def order_flow_imbalance(df: pd.DataFrame) -> pd.Series:
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    prev_close = np.roll(close, 1)
    body_size = np.abs(close - prev_close)
    max_c = np.maximum(close, prev_close)
    min_c = np.minimum(close, prev_close)
    upper_wick = high - max_c
    lower_wick = min_c - low
    imbalance = np.where(body_size > 0, (upper_wick - lower_wick) / body_size, 0.0)
    imbalance[0] = 0.0  # 先頭は前値がないので 0
    return pd.Series(imbalance, index=df.index, name="order_flow_imbalance")
```

### 3. `micro_volatility()` (L145-163) — 二重ループ → numpy vectorize

**現状**: 外ループ + 内ループで returns を計算し `np.std()` を適用

**改善案**:
```python
@register("micro_volatility")
def micro_volatility(df: pd.DataFrame, window: int = 5) -> pd.Series:
    close = df["close"].values.astype(np.float64)
    volatility = np.zeros(len(close), dtype=np.float64)
    safe_close = np.where(close > 0, close, 1.0)
    pct_returns = np.diff(safe_close) / safe_close[:-1]  # (n-1,)
    # rolling std of pct_returns with window (window-1)
    # pandas rolling を使う or cumsum ベースの rolling variance
    ret_series = pd.Series(pct_returns)
    rolling_std = ret_series.rolling(window - 1, min_periods=1).std(ddof=0)
    volatility[window:] = rolling_std.values[window - 1:]
    return pd.Series(volatility, index=df.index, name="micro_volatility")
```

## 制約

1. **関数シグネチャ (引数・戻り値) は一切変更しない** — `pd.Series` 返却、`name` 属性も維持
2. **`@register()` デコレータはそのまま維持**
3. **数値的等価性**: 既存テスト (`tests/unit/core/features/test_scalping_features.py`) が全て pass すること
4. `import` の追加は `numpy` / `pandas` のみ許可 (既に import 済み)
5. **`close == 0` のエッジケース**: 既存の挙動 (0 除算で 0.0 を返す) を維持すること
6. **型**: `np.float64` を維持

## テスト計画

```bash
pytest tests/unit/core/features/test_scalping_features.py -v
```

全テストが pass し、性能的に O(n²) → O(n) に改善されていることを確認する。

追加テスト案:
- 空 DataFrame, 1行 DataFrame, close=0 を含む DataFrame
- window > len(df) の場合

## 参考: 既存テストの構成

- `TestScalpingFeatures` クラスに `test_realized_volatility_basic`, `test_realized_volatility_window_parameter` 等が存在
- fixture `sample_dataframe` は n=100 の OHLCV データを生成

## コミット

```
feat(366#): T4+T9 scalping.py numpy vectorize

- realized_volatility: O(n²) → O(n) via cumsum
- order_flow_imbalance: for loop → numpy vectorize
- micro_volatility: nested loop → rolling std
```
