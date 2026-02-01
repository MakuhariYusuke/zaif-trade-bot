# 外部AIエージェントへの緊急依頼 v4 - Python組み込み関数でもSIGINT発生

## 🚨 深刻な問題: Python組み込み関数でもSIGINT発生

### 最新の発生箇所
```
File: ztb/features/generators/multi_timeframe/datetime_utils.py, Line 98
text = str(value).strip()  ← ここでSIGINT!
```

### 経緯
1. **最初**: `pd.to_datetime()` のC拡張 (`array_strptime`) でSIGINT
2. **対策**: `safe_to_datetime_series()` でPythonレベルのパーサーを実装
3. **結果**: **`str()`組み込み関数でもSIGINT発生**

### 技術的分析

**これはもはやC拡張の問題ではありません。Windows環境で何らかのシグナルが繰り返し送信されています。**

考えられる原因：
1. **Windowsプロセス管理の問題**: Python実行が別プロセスから SIGINT を受信
2. **ウイルス対策ソフト**: セキュリティソフトがプロセスをスキャン中に干渉
3. **システムサービス**: Windowsの何らかのサービスが干渉
4. **メモリ/リソース競合**: 他のプロセスとのリソース競合
5. **Pythonインタープリタの問題**: Python 3.11.9 特有のバグ

## 根本的な解決策の提案

### アプローチ1: シグナルを完全に無効化

```python
import signal

# Windows環境でSIGINTを一時的に無視
original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
try:
    # 危険な処理（タイムスタンプ変換等）
    result = process_data()
finally:
    # 元に戻す
    signal.signal(signal.SIGINT, original_sigint)
```

### アプローチ2: データの事前処理とキャッシュ

```python
# データ読み込み時に一度だけ変換し、pickle/featherで保存
def load_and_cache_data(csv_path):
    cache_path = csv_path.replace(".csv", "_cached.feather")
    
    if os.path.exists(cache_path):
        # キャッシュから高速読み込み（タイムスタンプ既に変換済み）
        return pd.read_feather(cache_path)
    
    # 初回のみ変換（リスクあるが1回だけ）
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.to_feather(cache_path)
    return df
```

### アプローチ3: 別プロセスでの事前変換

```python
# 完全に別のPythonプロセスで事前変換
subprocess.run([
    "python", "-c",
    "import pandas as pd; "
    "df = pd.read_csv('data.csv'); "
    "df['timestamp'] = pd.to_datetime(df['timestamp']); "
    "df.to_feather('data_cached.feather')"
])
```

### アプローチ4: Windows特有のシグナルハンドリング

```python
import ctypes

# Windows APIで割り込みを無効化
kernel32 = ctypes.windll.kernel32
kernel32.SetConsoleCtrlHandler(None, 1)  # Ctrl+Cを無効化

try:
    # 処理実行
    result = sensitive_operation()
finally:
    kernel32.SetConsoleCtrlHandler(None, 0)  # 元に戻す
```

## 依頼内容

### 最優先: 即座に実装可能な回避策

以下のいずれかを実装してください：

1. **データキャッシング戦略** (推奨)
   - `btc_jpy_1m_v451.csv` を事前に `.feather` または `.parquet` に変換
   - タイムスタンプを事前変換済みで保存
   - トレーニング時はキャッシュから直接読み込み

2. **シグナル無視戦略**
   - 環境初期化中のみ SIGINT を一時的に無視
   - `HeavyTradingEnv.__init__` の最初と最後でシグナル制御

3. **別プロセス分離**
   - データ変換を別プロセスで実行
   - main processは変換済みデータのみ使用

### 具体的な実装箇所

**ファイル**: `ztb/trading/environment/heavy_env/core.py` (Line 512付近)
**メソッド**: `HeavyTradingEnv.__init__`
**目的**: `_initialize_features_and_spaces()` 呼び出し前にデータを安全に準備

**ファイル**: `ztb/utils/data_utils.py`
**関数**: `load_csv_data()`
**目的**: キャッシング機能を追加

## プロジェクト情報

- **データファイル**: `data/btc_jpy_1m_v451.csv` (149,487行)
- **問題**: 2回目以降の実行で 環境初期化段階で SIGINT
- **環境**: Windows 11, Python 3.11.9, pandas 2.x
- **制約**: 連続48実験を1プロセスで実行する必要がある

## 期待する回答

### 1. データキャッシュ実装コード

```python
# ztb/utils/data_utils.py に追加

def load_csv_data_cached(csv_path: str, force_refresh: bool = False) -> pd.DataFrame:
    """Load CSV with timestamp caching to avoid repeated parsing."""
    # ... あなたの実装 ...
```

### 2. 環境初期化での適用

```python
# ztb/trading/environment/heavy_env/core.py の __init__ 修正

def __init__(self, ...):
    # ... 既存コード ...
    
    # SIGINT回避策を適用
    # ... あなたの実装 ...
    
    self._initialize_features_and_spaces(max_features)
```

### 3. 緊急回避スクリプト

```python
# scripts/prepare_cached_data.py (新規作成)

"""事前にデータを変換してキャッシュ"""
import pandas as pd

df = pd.read_csv("data/btc_jpy_1m_v451.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.to_feather("data/btc_jpy_1m_v451_cached.feather")
print("✅ Cache created")
```

## 補足情報

### システム環境調査結果
- `str(value).strip()` でSIGINT → Python組み込み関数でも発生
- 1回目は成功、2回目以降は必ず失敗 → 状態依存性あり
- 環境初期化の同じ箇所で毎回発生 → 再現性100%

### 試した対策（全て失敗）
- ✗ scipy/sklearn lazy imports
- ✗ torch thread limiting
- ✗ `safe_to_datetime_series()` (Python実装)
- ✗ 環境変数 `ZTB_SAFE_DATETIME=1`
- ✗ signal handler の詳細ロギング

---

**緊急度: 最高**

このままでは実験実行が不可能です。データキャッシング戦略の実装コードを最優先でお願いします。

よろしくお願いいたします。
