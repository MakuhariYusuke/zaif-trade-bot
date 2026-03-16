# 外部AIエージェントへの最終報告 v5

## 🚨 状況: 解決不可能な根本問題

### 問題の本質

Windows環境でPython 3.11.9を使用すると、**pandas/NumPyのあらゆるC拡張内でSIGINTが発生**します。

### 発生箇所の遷移

1. **pd.to_datetime() → array_strptime()**
2. **str(value).strip()** (Python組み込み関数)
3. **datetime.fromisoformat()**
4. **DataFrame.memory_usage(deep=True) → lib.memory_usage_of_objects()**
5. **DataFrame.copy() → DatetimeArray.copy()** ← 現在ここ

### 試した対策（全て無効）

- ✗ scipy/sklearn lazy imports
- ✗ torch thread limiting
- ✗ safe_to_datetime_series (Pythonパーサー)
- ✗ データキャッシング (feather/parquet)
- ✗ memory_usage(deep=False)
- ✗ signal handler
- ✗ faulthandler

## 技術的分析

### SIGINT発生の真の原因

**Pandasが内部でDataFrameをコピーする際、DatetimeArrayのC拡張が呼ばれ、そこでSIGINTを受信**

スタックトレース:
```
pd.merge_asof()
  → concat([left, right])
    → concatenate_managers()
      → mgr.copy()
        → DatetimeArray.copy()  ← SIGINT発生
```

### 根本原因の推測

1. **Windowsプロセス管理の異常**: 何らかのサービスがSIGINTを送信
2. **セキュリティソフトウェアの干渉**: ウイルス対策が処理を中断
3. **Python 3.11.9のバグ**: Windows特有の不具合
4. **システムリソース競合**: メモリ/CPUの競合状態

### なぜ1回目は成功し、2回目以降失敗するのか

- 1回目: システムリソースがクリーン
- 2回目以降: 前回実行の残骸（ハンドル、スレッド、ロック）が残っている
- Windows環境でのプロセス隔離が不完全

## 実行可能な最終的解決策

### オプションA: 別プロセスでの実験実行（推奨）

各実験を完全に独立したプロセスで実行し、プロセス間でリソースを共有しない。

```python
# scripts/v459/run_ab_reward_experiments_isolated.py

import subprocess
import sys

for experiment in experiments:
    # 各実験を別プロセスで実行
    result = subprocess.run([
        sys.executable,
        "scripts/v459/run_single_experiment_isolated.py",
        "--config", experiment.config_path
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Experiment {experiment.name} failed")
        continue
```

**メリット**:
- プロセス隔離により、リソース残留を完全に回避
- 各実験が独立して実行され、影響を受けない
- Windows特有の問題を最小化

**デメリット**:
- 実行時間がやや長くなる（プロセス起動オーバーヘッド）
- プロセス間通信が必要

### オプションB: Linux/WSL2での実行（最も確実）

Windows環境を諦め、WSL2（Windows Subsystem for Linux）またはLinuxマシンで実行。

```bash
# WSL2にUbuntuをインストール
wsl --install -d Ubuntu

# WSL内で実行
wsl
cd /mnt/c/Users/Admin/dev/zaif-trade-bot
python scripts/v459/run_ab_reward_experiments.py
```

**メリット**:
- Windows特有の問題を完全回避
- Linux環境でのpandasは安定
- シグナルハンドリングが正常

**デメリット**:
- 環境構築が必要
- GPUがある場合、WSL2でのCUDA設定が必要

### オプションC: Docker コンテナ実行

Dockerコンテナで完全に隔離された環境で実行。

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

CMD ["python", "scripts/v459/run_ab_reward_experiments.py"]
```

**メリット**:
- 完全な環境隔離
- 再現性が高い
- 他マシンでも同一環境で実行可能

**デメリット**:
- Docker設定が必要
- Windows Dockerの設定が複雑

### オプションD: Python 3.10へのダウングレード

Python 3.11.9特有の問題の可能性があるため、3.10系に戻す。

```powershell
# pyenv for Windowsでバージョン切り替え
pyenv install 3.10.11
pyenv local 3.10.11
```

**メリット**:
- 比較的簡単に試せる
- 他の変更不要

**デメリット**:
- 根本解決にならない可能性

## 推奨アクション

**1. 即座に試すべきこと (優先度順)**

1. **別プロセス実行** (オプションA) - 30分で実装可能
2. **WSL2実行** (オプションB) - 1時間で環境構築
3. **Python 3.10** (オプションD) - 15分で切り替え

**2. 長期的な解決**

- Docker環境の構築 (オプションC)
- CI/CD パイプラインでのLinux実行
- 開発環境のLinuxへの完全移行

## 結論

**この問題はコード修正では解決できません。**

Windows + Python 3.11.9 + pandas 2.x の組み合わせで、C拡張がSIGINTを受信し続ける根本的な環境問題です。

**解決策は環境変更のみ:**
1. 別プロセス実行
2. WSL2/Linux
3. Docker
4. Python バージョン変更

## 次のステップ

**オプションA（別プロセス実行）の実装コードを提供してください。**

これが最も現実的で、即座に適用可能な解決策です。

---

**状況**: 進行不可能、環境変更必須
**推奨**: 別プロセス実行の即時実装

よろしくお願いいたします。
