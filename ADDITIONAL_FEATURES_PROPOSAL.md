# 追加機能提案リスト

**日付**: 2025年10月7日  
**背景**: 100kテスト実行中、並行して追加機能を導入  
**目標**: 型安全性・保守性向上、運用効率化

---

## ✅ 実装済み機能

### 1. ログレベル制御（2025-10-07実装）

- `--log-level` 引数追加（DEBUG/INFO/WARNING/ERROR/CRITICAL）
- デフォルトはINFO（DEBUGログを抑制）
- ルートロガーも制御し、サードパーティログも抑制
- ドキュメント: LOG_LEVEL_CONTROL.md、LOG_LEVEL_INTEGRATION_SUMMARY.md

**効果**: 視認性大幅向上、並列実行時の混乱解消

---

## 🔴 優先度: 高（1-2日で実装推奨）

### 2. 設定ファイル検証ツール

**目的**: 設定ファイルの不整合を自動検出・修正

**実装内容**:

#### 2.1 check_config_consistency.py

```python
#!/usr/bin/env python3
"""
設定ファイル不整合検出ツール

Usage:
    python check_config_consistency.py configs/train/*.json
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Set

def check_config_consistency(config_files: List[Path]) -> Dict[str, List[str]]:
    """設定ファイルの不整合を検出"""
    issues = {
        "naming_inconsistencies": [],
        "missing_required_params": [],
        "type_mismatches": [],
        "default_value_differences": [],
    }
    
    # 全設定ファイル読み込み
    configs = {}
    for path in config_files:
        with open(path) as f:
            configs[path.name] = json.load(f)
    
    # 1. パラメータ命名の一貫性チェック
    all_keys: Set[str] = set()
    for config in configs.values():
        all_keys.update(flatten_keys(config))
    
    # スネークケース以外を検出
    for key in all_keys:
        if not is_snake_case(key):
            issues["naming_inconsistencies"].append(
                f"Non-snake_case key: {key}"
            )
    
    # 2. 必須パラメータ存在チェック
    required_params = [
        "algorithm",
        "data_path",
        "total_timesteps",
        "checkpoint_interval",
    ]
    
    for filename, config in configs.items():
        for param in required_params:
            if param not in config:
                issues["missing_required_params"].append(
                    f"{filename}: Missing required parameter '{param}'"
                )
    
    # 3. 型不一致チェック
    for filename, config in configs.items():
        if "total_timesteps" in config:
            if not isinstance(config["total_timesteps"], int):
                issues["type_mismatches"].append(
                    f"{filename}: total_timesteps should be int, got {type(config['total_timesteps'])}"
                )
    
    # 4. デフォルト値の差異チェック
    # 同じキーで異なるデフォルト値を使っている場合に警告
    key_values: Dict[str, List[Any]] = {}
    for filename, config in configs.items():
        for key, value in flatten_dict(config).items():
            if key not in key_values:
                key_values[key] = []
            key_values[key].append((filename, value))
    
    for key, values in key_values.items():
        unique_values = set(v for _, v in values)
        if len(unique_values) > 1:
            # 意図的な差異（ent_coef等）は除外
            if key not in ["ent_coef", "seed", "session_id", "checkpoint_dir", "log_dir", "model_dir"]:
                issues["default_value_differences"].append(
                    f"{key}: Different values across configs: {unique_values}"
                )
    
    return issues

def flatten_keys(d: Dict, parent_key: str = "") -> Set[str]:
    """ネストされた辞書のキーをフラット化"""
    keys = set()
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        keys.add(k)  # トップレベルのキーも追加
        if isinstance(v, dict):
            keys.update(flatten_keys(v, new_key))
    return keys

def flatten_dict(d: Dict, parent_key: str = "") -> Dict[str, Any]:
    """ネストされた辞書をフラット化"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)

def is_snake_case(s: str) -> bool:
    """スネークケースかチェック"""
    return s.islower() or s.replace("_", "").islower()

if __name__ == "__main__":
    import sys
    
    config_files = [Path(p) for p in sys.argv[1:]]
    issues = check_config_consistency(config_files)
    
    print("=" * 60)
    print("設定ファイル不整合検出結果")
    print("=" * 60)
    print()
    
    total_issues = sum(len(v) for v in issues.values())
    
    if total_issues == 0:
        print("✅ 不整合は検出されませんでした")
    else:
        for category, items in issues.items():
            if items:
                print(f"\n❌ {category}:")
                for item in items:
                    print(f"  - {item}")
        
        print()
        print("=" * 60)
        print(f"合計 {total_issues} 件の問題が検出されました")
```

**実行例**:
```bash
# 全設定ファイルをチェック
python check_config_consistency.py configs/train/*.json

# 100kテスト設定のみチェック
python check_config_consistency.py configs/train/ensemble_*_100k_test.json
```

**期待効果**:
- 設定ファイルの不整合を自動検出
- 命名規則違反の早期発見
- 型ミスマッチの検出

---

### 3. 学習進捗監視ツール

**目的**: TensorBoardなしでコマンドラインから学習進捗を確認

**実装内容**:

#### 3.1 watch_training.py

```python
#!/usr/bin/env python3
"""
学習進捗リアルタイム監視ツール

Usage:
    python watch_training.py --log-dir logs/ensemble_B_100k_test
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Optional

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("❌ TensorBoard not installed. Run: pip install tensorboard")
    exit(1)

def watch_training(log_dir: Path, interval: int = 10):
    """学習進捗をリアルタイム監視"""
    print(f"🔍 Monitoring: {log_dir}")
    print(f"📊 Refresh interval: {interval}s")
    print("=" * 80)
    
    ea = event_accumulator.EventAccumulator(str(log_dir))
    
    while True:
        ea.Reload()
        
        # スカラー値取得
        scalars = ea.Tags().get('scalars', [])
        
        # 重要な指標を表示
        important_metrics = [
            "train/legal_sell_rate",
            "train/entropy",
            "eval/sharpe_proxy",
            "train/grad_norm(SELL)",
            "rollout/ep_rew_mean",
        ]
        
        print("\n" + "=" * 80)
        print(f"⏱️  {time.strftime('%H:%M:%S')}")
        print("=" * 80)
        
        for metric in important_metrics:
            if metric in scalars:
                values = ea.Scalars(metric)
                if values:
                    latest = values[-1]
                    print(f"{metric:30s}: {latest.value:10.4f} (step {latest.step})")
        
        time.sleep(interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()
    
    watch_training(Path(args.log_dir), args.interval)
```

**実行例**:
```bash
# モデルBの進捗を10秒ごとに更新
python watch_training.py --log-dir logs/ensemble_B_100k_test --interval 10
```

**期待効果**:
- TensorBoardを開かずに進捗確認
- コマンドラインで複数モデルを並列監視
- 重要指標だけ表示（ノイズ削減）

---

### 4. チェックポイント比較ツール

**目的**: 複数チェックポイントの性能を一覧比較

**実装内容**:

#### 4.1 compare_checkpoints.py

```python
#!/usr/bin/env python3
"""
チェックポイント性能比較ツール

Usage:
    python compare_checkpoints.py --checkpoint-dir checkpoints/ensemble_B_100k_test
"""

import argparse
from pathlib import Path
from typing import List, Dict

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("❌ TensorBoard not installed")
    exit(1)

def compare_checkpoints(checkpoint_dir: Path):
    """チェックポイントの性能を比較"""
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*"))
    
    print("=" * 100)
    print(f"📊 Checkpoint Comparison: {checkpoint_dir.name}")
    print("=" * 100)
    print()
    
    # ヘッダー
    print(f"{'Checkpoint':20s} {'SELL Rate':12s} {'Sharpe':12s} {'Entropy':12s} {'Reward':12s}")
    print("-" * 100)
    
    for ckpt in checkpoints:
        step = ckpt.name.split("_")[-1]
        
        # 対応するログディレクトリから指標を取得
        log_dir = checkpoint_dir.parent.parent / "logs" / checkpoint_dir.name
        
        if log_dir.exists():
            ea = event_accumulator.EventAccumulator(str(log_dir))
            ea.Reload()
            
            sell_rate = get_metric_at_step(ea, "train/legal_sell_rate", int(step))
            sharpe = get_metric_at_step(ea, "eval/sharpe_proxy", int(step))
            entropy = get_metric_at_step(ea, "train/entropy", int(step))
            reward = get_metric_at_step(ea, "rollout/ep_rew_mean", int(step))
            
            print(f"{ckpt.name:20s} {sell_rate:12.4f} {sharpe:12.4f} {entropy:12.4f} {reward:12.4f}")

def get_metric_at_step(ea, metric: str, step: int) -> float:
    """指定ステップでの指標値を取得"""
    try:
        values = ea.Scalars(metric)
        # stepに最も近い値を取得
        closest = min(values, key=lambda x: abs(x.step - step))
        return closest.value
    except:
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()
    
    compare_checkpoints(Path(args.checkpoint_dir))
```

**実行例**:
```bash
# モデルBの全チェックポイントを比較
python compare_checkpoints.py --checkpoint-dir checkpoints/ensemble_B_100k_test
```

**出力例**:
```
====================================================================================================
📊 Checkpoint Comparison: ensemble_B_100k_test
====================================================================================================

Checkpoint           SELL Rate    Sharpe       Entropy      Reward      
----------------------------------------------------------------------------------------------------
checkpoint_10000        0.0300      -0.5000       0.8500      100.0000
checkpoint_20000        0.0450       0.2000       0.7800      150.0000
checkpoint_30000        0.0600       0.5000       0.7200      180.0000  ← 最良
checkpoint_40000        0.0550       0.4500       0.6800      170.0000
```

**期待効果**:
- 最良チェックポイントを一目で特定
- 過学習検出（後半で性能悪化）
- 学習停止タイミングの判断

---

## 🟠 優先度: 中（1週間以内に実装）

### 5. 型スタブ生成ツール

**目的**: 既存コードから型スタブ(.pyi)を自動生成

**実装**: stubgenやmypyの機能を活用

```bash
# 型スタブ生成
stubgen -p ztb.training -o stubs/

# 生成された.pyiファイルを確認・修正
```

---

### 6. 設定ファイルスキーマ定義

**目的**: TypedDictで設定ファイルのスキーマを明示

```python
from typing import TypedDict, Literal

class TrainingConfig(TypedDict):
    learning_rate: float
    n_steps: int
    batch_size: int
    # ...

class UnifiedTrainerConfig(TypedDict):
    algorithm: Literal["ppo", "iterative", "ensemble", "curriculum"]
    data_path: str
    total_timesteps: int
    checkpoint_interval: int
    training: TrainingConfig
    # ...
```

---

## 🟡 優先度: 低（2週間以内に実装）

### 7. ユニットテスト強化

**目的**: テストカバレッジ向上

**対象**:
- unified_trainer.py（各アルゴリズムの実行テスト）
- 設定ファイル読み込みテスト
- ログレベル制御テスト

---

### 8. ドキュメント自動生成

**目的**: docstringからAPIドキュメント自動生成

**実装**: Sphinx + autodoc

```bash
# Sphinx初期化
sphinx-quickstart docs

# autodoc設定
# conf.py に extensions = ['sphinx.ext.autodoc'] 追加

# ドキュメント生成
sphinx-build -b html docs docs/_build
```

---

## 🚀 即座に実装可能な改善

### 9. 設定ファイルテンプレート

**目的**: 新規設定ファイル作成を容易化

**実装**: configs/train/template.json作成

```json
{
  "comment": "設定ファイルテンプレート - コピーして使用",
  "algorithm": "ppo",
  "data_path": "ml-dataset-enhanced.csv",
  "session_id": "YOUR_SESSION_ID",
  "total_timesteps": 100000,
  
  "training": {
    "learning_rate": 3.0e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "ent_coef": 0.7,
    "seed": 42
  },
  
  "checkpoint_interval": 10000,
  "checkpoint_dir": "checkpoints/YOUR_SESSION_ID",
  "log_dir": "logs/YOUR_SESSION_ID",
  "model_dir": "models/YOUR_SESSION_ID"
}
```

---

## 📋 実装優先順位まとめ

| 優先度 | 機能 | 所要時間 | 効果 |
|--------|------|----------|------|
| ✅ 完了 | ログレベル制御 | - | 視認性向上 |
| 🔴 高 | 設定ファイル検証ツール | 2-3時間 | 不整合早期検出 |
| 🔴 高 | 学習進捗監視ツール | 1-2時間 | 運用効率化 |
| 🔴 高 | チェックポイント比較ツール | 1-2時間 | 最良モデル特定 |
| 🟠 中 | 型スタブ生成 | 2-4時間 | 型安全性向上 |
| 🟠 中 | 設定ファイルスキーマ | 3-5時間 | 型チェック強化 |
| 🟡 低 | ユニットテスト強化 | 1週間 | 品質保証 |
| 🟡 低 | ドキュメント自動生成 | 3-5時間 | 保守性向上 |
| 🟢 即座 | 設定ファイルテンプレート | 10分 | 利便性向上 |

---

## 🎯 推奨実装順序

### Week 1（現在）
1. ✅ ログレベル制御（完了）
2. 🟢 設定ファイルテンプレート（10分）
3. 🔴 設定ファイル検証ツール（2-3時間）

### Week 2
4. 🔴 学習進捗監視ツール（1-2時間）
5. 🔴 チェックポイント比較ツール（1-2時間）

### Week 3
6. 🟠 型スタブ生成（2-4時間）
7. 🟠 設定ファイルスキーマ（3-5時間）

### Week 4
8. 🟡 ユニットテスト強化（継続）
9. 🟡 ドキュメント自動生成（3-5時間）

---

## 💡 次のアクション

### 今すぐ実装（10分）
```bash
# 設定ファイルテンプレート作成
# → configs/train/template.json
```

### 今日中に実装（2-3時間）
```bash
# 設定ファイル検証ツール作成
# → check_config_consistency.py
```

### 明日実装（3-4時間）
```bash
# 学習進捗監視ツール + チェックポイント比較ツール
# → watch_training.py
# → compare_checkpoints.py
```

---

## ✅ まとめ

1. **ログレベル制御**: ✅ 実装完了
2. **次の優先機能**: 設定ファイル検証ツール（不整合検出）
3. **運用改善**: 学習進捗監視・チェックポイント比較ツール
4. **長期改善**: 型安全性・テスト・ドキュメント

**100kテスト実行中に、並行して設定ファイル検証ツールを実装することを推奨します！**
