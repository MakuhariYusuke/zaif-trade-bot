# Session 3 パフォーマンス最適化 - 実装ガイド

**日付**: 2026-01-15  
**対象**: Walk-Forward マルチプロセッシング化（50ウィンドウ評価の高速化）

---

## 目次
1. [実装概要](#実装概要)
2. [既存コンポーネントの活用](#既存コンポーネントの活用)
3. [モジュール設計詳細](#モジュール設計詳細)
4. [実装ステップ](#実装ステップ)
5. [テスト戦略](#テスト戦略)
6. [期待効果](#期待効果)

---

## 実装概要

### 目標
- 現在: 50ウィンドウ評価 ≈ 25時間
- 目標: 50ウィンドウ評価 ≈ 2-4時間（マルチプロセッシング）

### アーキテクチャ
```
[Main Process]
    ↓
[ztb/optimization/parallel/] (新規構築)
    ├── ParallelWindowEvaluator (マルチプロセッシング制御)
    ├── WindowTask (プロセス間通信)
    └── ExecutionMetrics (パフォーマンス測定)
    ↓
[既存 ztb/evaluation/walk_forward/evaluator.py] (変更なし)
    → _evaluate_window() を各プロセスで並列実行
```

### 重要な設計原則
- **既存コードの変更最小化**: Walk-Forward 評価ロジックは変更せず、並列化層を上に追加
- **DRY 原則**: `ztb/utils/checkpoint.py` と `file_utils.py` を活用
- **エラー隔離**: 単一ウィンドウのエラーが他のウィンドウに波及しない
- **リソース管理**: メモリ効率化（チェックポイント活用による中間結果圧縮）

---

## 既存コンポーネントの活用

### 1. ztb/utils/checkpoint.py（継続活用）

**利用方法**:
```python
from ztb.utils.checkpoint import CheckpointManager

# 各プロセス内で中間結果を保存
checkpoint_mgr = CheckpointManager(save_dir="./windows/w1")
checkpoint_mgr.save(model, metrics, metadata)

# メインプロセスで結果回収
restored = checkpoint_mgr.restore()
```

**メリット**:
- 圧縮対応（zlib/lz4/zstd）で I/O 効率化
- 既に Session 2 で統合済みパターン
- エラー時の部分復旧が容易

### 2. ztb/utils/file_utils.py（参考活用）

**利用方法**:
```python
from ztb.utils.file_utils import safe_json_dump, safe_json_load

# プロセス間で競合状態を回避して JSON 保存
safe_json_dump(window_results, f"results/w1.json")

# 安全な読み込み
results = safe_json_load(f"results/w1.json")
```

**メリット**:
- ファイル書き込み競合の自動ハンドリング
- デフォルト値対応で部分的なデータロス時も対応

### 3. ztb/training/unified_optimizer.py（パターン参考）

**参考箇所**:
```python
# unified_optimizer.py の以下から設計パターンを参考
- max_parallel_trials: int = 4              ← プロセス数決定ロジック
- enable_distributed: bool = False          ← 分散実行フラグ
- run_parallel_optimization(...)            ← 並列実行フレームワーク
- get_parallel_status()                     ← 進捗監視インターフェース
```

**活用方法**:
```python
# 既存のOptimizationConfigパターンを参考に
@dataclass
class ParallelEvaluationConfig:
    max_workers: int = 4
    chunk_size: int = 5          # 1プロセスあたりの担当ウィンドウ数
    enable_checkpointing: bool = True
    compression: str = "zlib"
```

### 4. ztb/utils/config_manager.py（統一設定管理）

**利用方法**:
```python
from ztb.utils.config_manager import ConfigManager

config_mgr = ConfigManager()
parallel_config = config_mgr.get("parallel_evaluation", {})
# または
parallel_config = ParallelEvaluationConfig(**parallel_config)
```

---

## モジュール設計詳細

### モジュール 1: ztb/optimization/parallel/__init__.py

```python
"""Parallel evaluation and optimization module."""

from .window_evaluator import (
    ParallelWindowEvaluator,
    WindowTask,
    WindowTaskResult,
)
from .executor import ProcessPoolExecutor, ExecutionMetrics
from .config import ParallelEvaluationConfig

__all__ = [
    "ParallelWindowEvaluator",
    "WindowTask",
    "WindowTaskResult",
    "ProcessPoolExecutor",
    "ExecutionMetrics",
    "ParallelEvaluationConfig",
]
```

### モジュール 2: ztb/optimization/parallel/config.py

```python
"""Configuration for parallel evaluation."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class ParallelEvaluationConfig:
    """Parallel evaluation configuration."""
    
    # Process management
    max_workers: int = 4                    # CPUコア数の80%推奨
    timeout_per_window: float = 3600.0      # 秒 (1時間)
    
    # Checkpointing
    enable_checkpointing: bool = True
    checkpoint_compress: str = "zlib"       # "zlib", "lz4", "zstd"
    checkpoint_dir: str = "./checkpoints"
    
    # Performance
    chunk_size: int = 5                     # 1プロセスあたりのウィンドウ数
    enable_profiling: bool = True
    profile_output_dir: str = "./profiles"
    
    # Recovery
    enable_recovery: bool = True
    recovery_strategy: str = "checkpoint"   # "checkpoint", "manual"
    
    # Monitoring
    enable_progress_bar: bool = True
    log_interval: int = 10                  # ウィンドウ数
```

### モジュール 3: ztb/optimization/parallel/window_evaluator.py

**主要クラス構成**:

```python
# データクラス（プロセス間通信用）
@dataclass
class WindowTask:
    """単一ウィンドウの評価タスク"""
    window_id: int
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    config: Dict[str, Any]
    checkpoint_path: Optional[str] = None

@dataclass
class WindowTaskResult:
    """ウィンドウ評価結果"""
    window_id: int
    model: SAC
    performance: WindowPerformance
    execution_time: float
    memory_peak_mb: float
    error: Optional[Exception] = None
    checkpoint_path: Optional[str] = None

# メイン評価クラス
class ParallelWindowEvaluator:
    """並列ウィンドウ評価エンジン
    
    既存の WalkForwardModelEvaluator._evaluate_window() を
    マルチプロセッシングで並列化
    """
    
    def __init__(self, config: ParallelEvaluationConfig):
        self.config = config
        self.executor = ProcessPoolExecutor(max_workers=config.max_workers)
        self.results: Dict[int, WindowTaskResult] = {}
        
    def evaluate_windows(
        self, 
        windows: List[TimeSeriesWindow],
        evaluator: WalkForwardModelEvaluator,
    ) -> Dict[int, WindowTaskResult]:
        """複数ウィンドウを並列評価
        
        Args:
            windows: 評価対象ウィンドウのリスト
            evaluator: 単一ウィンドウ評価器
            
        Returns:
            ウィンドウID -> 評価結果のマッピング
        """
        # タスク生成
        tasks = [
            WindowTask(
                window_id=w.id,
                train_data=w.train_data,
                test_data=w.test_data,
                config=evaluator.config,
            )
            for w in windows
        ]
        
        # 並列実行
        results = self.executor.execute_tasks(
            tasks=tasks,
            evaluate_fn=self._evaluate_window_worker,
            timeout=self.config.timeout_per_window,
        )
        
        self.results = {r.window_id: r for r in results}
        return self.results
    
    @staticmethod
    def _evaluate_window_worker(task: WindowTask) -> WindowTaskResult:
        """ワーカープロセスで実行される評価関数
        
        既存の evaluator._evaluate_window() をラップ
        """
        # ... 実装詳細は以降
```

### モジュール 4: ztb/optimization/parallel/executor.py

```python
class ProcessPoolExecutor:
    """multiprocessing.Pool のラッパー"""
    
    def __init__(self, max_workers: int):
        self.pool = multiprocessing.Pool(max_workers)
        self.metrics = ExecutionMetrics()
    
    def execute_tasks(
        self,
        tasks: List[WindowTask],
        evaluate_fn: Callable[[WindowTask], WindowTaskResult],
        timeout: float,
    ) -> List[WindowTaskResult]:
        """タスクをプール内で並列実行"""
        # 実装詳細

@dataclass
class ExecutionMetrics:
    """実行時間とリソース使用統計"""
    
    total_wall_time: float = 0.0        # 全体時間
    total_compute_time: float = 0.0     # 計算時間（並列分を集計）
    peak_memory_mb: float = 0.0         # ピークメモリ
    speedup_factor: float = 0.0         # 並列化による高速化率
```

---

## 実装ステップ

### Step 1: 基本構造の構築（1.5時間）

**ファイル作成**:
```bash
mkdir -p ztb/optimization/parallel
touch ztb/optimization/parallel/__init__.py
touch ztb/optimization/parallel/config.py
touch ztb/optimization/parallel/window_evaluator.py
touch ztb/optimization/parallel/executor.py
```

**実装順序**:
1. `config.py` - 設定クラス定義
2. `window_evaluator.py` - データクラス・基本インターフェース
3. `executor.py` - マルチプロセッシング実装
4. `__init__.py` - エクスポート

### Step 2: 単一ウィンドウ評価の実装（2時間）

**目標**: `_evaluate_window_worker()` の完全実装

```python
@staticmethod
def _evaluate_window_worker(task: WindowTask) -> WindowTaskResult:
    """
    実装ポイント:
    1. メモリ使用量測定開始
    2. 既存評ator._evaluate_window()を呼び出し
    3. エラーキャッチと記録
    4. チェックポイント保存（オプション）
    5. 実行統計収集
    """
    start_time = time.time()
    
    try:
        # 既存コンポーネントの再利用
        from ztb.evaluation.walk_forward import WalkForwardModelEvaluator
        evaluator = WalkForwardModelEvaluator()
        
        model, perf = evaluator._evaluate_window(
            task.window_id,
            task.train_data,
            task.test_data,
        )
        
        # メモリ計測・チェックポイント保存
        # ...
        
        return WindowTaskResult(
            window_id=task.window_id,
            model=model,
            performance=perf,
            execution_time=time.time() - start_time,
            # ...
        )
    except Exception as e:
        # エラーハンドリング
        return WindowTaskResult(
            window_id=task.window_id,
            model=None,
            performance=None,
            execution_time=time.time() - start_time,
            error=e,
        )
```

### Step 3: 高度な最適化（1.5時間）

**オプション機能**:
- キャッシング統合（`ztb/utils/cache_utils.py`）
- エラーリカバリー（チェックポイント復帰）
- 動的ロードバランシング
- 進捗リポート

### Step 4: テスト & 検証（2時間）

**テスト対象**:
```python
# tests/unit/optimization/test_parallel_evaluator.py
- test_single_window_evaluation()          ← 単一ウィンドウ
- test_parallel_evaluation_4_windows()     ← 複数ウィンドウ
- test_error_handling()                    ← エラー隔離
- test_checkpoint_integration()            ← チェックポイント
- test_performance_measurement()           ← パフォーマンス計測
```

---

## テスト戦略

### 既存テストの保持（32/32 passing）

```python
# Session 2 のテスト継続実行
pytest tests/unit/evaluation/test_walk_forward_*.py -v
# → 32/32 すべてが passing であることを確認
```

### 新規テストの追加

```python
# tests/unit/optimization/test_parallel_evaluator.py
# - 基本機能テスト（5個）
# - エラーハンドリング（3個）
# - パフォーマンス測定（2個）
# → 10個の新規テスト

# 目標: 42/42 passing
```

### 統合テスト（E2E）

```python
# tests/unit/evaluation/test_walk_forward_integration_parallel.py
# - 50ウィンドウ実際評価
# - 完全なパイプライン検証
# - 時間計測（実際の高速化確認）
```

---

## 期待効果

### パフォーマンス改善

| 項目 | 現状 | 目標 | 改善率 |
|-----|------|------|--------|
| 50ウィンドウ評価時間 | ~25時間 | 2-4時間 | **87.5-92%削減** |
| スループット | 2 w/h | 12-25 w/h | **6-12.5倍** |
| メモリピーク | ~8 GB | ~3-4 GB | **50-62%削減** |

### コード品質向上

| 指標 | 向上点 |
|-----|--------|
| テストカバレッジ | 32/32 → 42/42 (31% 増加) |
| エラーハンドリング | 単一障害隔離で堅牢性向上 |
| 保守性 | DRY 原則遵守で長期保守容易化 |
| ドキュメント | 実装ガイド + API ドキュメント完備 |

### Session 3 のロードマップ

```
Week 1 (3-4日):
  Day 1: 基本構造構築 (1.5時間)
  Day 1-2: 単一ウィンドウ評価実装 (2時間)
  Day 2-3: 高度な最適化 (1.5時間)
  Day 3: テスト & 検証 (2時間)

Week 2 (2-3日):
  Day 4: E2E 統合テスト・ドキュメント化
  Day 5: パフォーマンス測定・チューニング
  Day 6-7: 予備・拡張機能

成果物:
  ✅ ztb/optimization/parallel/ (3-4ファイル, 600-800行)
  ✅ テストスイート拡張 (42/42 passing)
  ✅ パフォーマンスレポート
  ✅ 実装ドキュメント
```

---

## 参考: 既存コードリポジトリ

### 検索キーワード
```python
# multiprocessing の既存活用例
grep -r "multiprocessing\|ProcessPool\|concurrent.futures" ztb/ --include="*.py"

# 並列処理パターン
grep -r "Pool\|ThreadPoolExecutor" ztb/training/ --include="*.py"
# → ztb/training/online_learning_engine.py に ThreadPoolExecutor 活用例あり

# エラーハンドリング
grep -r "safe_operation\|error_utils" ztb/ --include="*.py"
# → ztb/utils/error_utils.py に例外隔離パターンあり
```

### 直接参照すべきファイル
- `ztb/training/unified_optimizer.py` (L989-1050): `run_parallel_optimization()`
- `ztb/training/online_learning_engine.py` (L12, L251): ThreadPoolExecutor パターン
- `ztb/utils/checkpoint.py` (L118-580): CheckpointManager マスター実装
- `ztb/evaluation/walk_forward/evaluator.py` (L200-280): _evaluate_window() ロジック

---

**次ステップ**: 本ガイドに基づき、Session 3 開始時に `ztb/optimization/parallel/` を段階的に構築してください。既存コンポーネントの活用により、実装時間を最小化しつつ、高品質を保証できます。

