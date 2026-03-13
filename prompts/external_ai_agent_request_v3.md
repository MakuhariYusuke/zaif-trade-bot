# 外部AIエージェントへの依頼プロンプト v3 - 根本原因特定済み

## ✅ 問題の根本原因が判明しました

### 発生箇所
```
File: ztb/features/generators/multi_timeframe/__init__.py, Line 129
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
```

### 技術的詳細
- **症状**: `pandas.to_datetime()` の `array_strptime()` (C拡張) 内でSIGINTが発生
- **トリガー**: 環境初期化時のタイムスタンプパース処理
- **Windows特有**: Pandasの内部C/Cythonモジュールがマルチスレッド環境でシグナルを受信
- **再現性**: 2回目以降の実行で発生（最初の実行後にシステム状態が変わる）

### スタックトレース (抜粋)
```
File "pandas/core/tools/datetimes.py", line 469, in _array_strptime_with_fallback
  result, tz_out = array_strptime(arg, fmt, exact=exact, errors=errors, utc=utc)
  
🚨 SIGNAL RECEIVED: SIGINT (count: 1)
Frame info: pandas/core/tools/datetimes.py:469
```

## 既に実施した対策

1. **scipy/sklearn/shap lazy imports** ✅
2. **PyTorch thread limiting** ✅ (`torch.set_num_threads(1)`)
3. **環境変数設定** ✅ (OMP_NUM_THREADS=1, etc.)
4. **深い例外トレース** ✅ (signal handler, faulthandler)

## 依頼内容

### 優先度1: pandas.to_datetime の回避策

**問題のコード**:
```python
# ztb/features/generators/multi_timeframe/__init__.py:129
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
```

**必要な対策**:
1. C拡張を使わない代替実装（NumPy/Python標準ライブラリ）
2. または、シグナルマスキング中にto_datetime実行
3. または、事前にタイムスタンプを変換してキャッシュ

**具体的な修正案を提案してください**:
- ファイル: `ztb/features/generators/multi_timeframe/__init__.py`
- 関数: `process_multi_timeframe_data()` (おそらく120-140行付近)
- 目的: SIGINTが発生しない安全なタイムスタンプ変換

### 優先度2: 類似箇所の洗い出し

プロジェクト内で `pd.to_datetime()` を使用している箇所をすべてリストアップし、同じ問題が起きる可能性がある箇所を特定してください。

特に以下のパターンを探索:
- データローディング時のタイムスタンプ変換
- 環境初期化時のデータ処理
- Walk-Forwardのウィンドウ分割時の日時処理

### 優先度3: Windows C拡張の一般的対策

Pandasだけでなく、他のC拡張（NumPy, PyTorch等）でも同様の問題が起きる可能性があります。

**検討すべき対策**:
1. `signal.pthread_sigmask()` でのシグナルブロック (Unix系のみ)
2. Windowsでの代替手段（スレッドローカルなシグナル処理）
3. C拡張を呼ぶ前後での`signal.signal()`の一時無効化

## プロジェクト構造

```
zaif-trade-bot/
├── ztb/
│   ├── features/
│   │   └── generators/
│   │       └── multi_timeframe/
│   │           └── __init__.py  ← 問題の箇所 (Line 129)
│   ├── trading/environment/heavy_env/
│   │   ├── core.py  ← 環境初期化 (Line 512で上記を呼び出し)
│   │   └── mixins/initialization.py  ← Line 295で呼び出し
│   └── training/unified_trainer/
│       └── algorithms/sac_trainer.py  ← Line 619でHeavyTradingEnv生成
├── scripts/v459/
│   └── run_ab_reward_experiments.py  ← メイン実行スクリプト
└── data/
    └── btc_jpy_1m_v451.csv  ← 149,487行（タイムスタンプ列含む）
```

## 期待する回答形式

### 1. 即座に適用可能な修正コード
```python
# ztb/features/generators/multi_timeframe/__init__.py

# Before (問題あり):
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# After (修正案):
# ... ここにあなたの提案 ...
```

### 2. 他の危険箇所のリスト
- ファイル名: 行番号
- 使用パターン: `pd.to_datetime()` / `array_strptime()`
- リスク評価: 高/中/低

### 3. Windows環境での推奨設定
- 環境変数
- Pandas設定オプション
- シグナルハンドリング戦略

## 補足情報

### 成功した実行（1回目）
- タイムスタンプ変換は正常に完了
- トレーニングは5000ステップ完走
- メモリ使用量も正常（1.7GB）

### 失敗した実行（2回目以降）
- **初期化段階で即座にSIGINT発生**
- トレーニング開始前（環境構築中）
- `pd.to_datetime()` の内部C関数で中断

### 重要な制約
- タイムスタンプデータは必須（時系列分析に使用）
- 高速処理が望ましい（149k行のデータ）
- Windows環境でのみ発生（Linux環境では未確認）
- 連続実行が必須（48実験を1プロセスで実行）

---

**最優先課題**: `pd.to_datetime()` の安全な代替実装または回避策

ご協力よろしくお願いいたします。
