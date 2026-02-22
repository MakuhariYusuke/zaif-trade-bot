"""
Walk-Forward Checkpoint/Resume 機能

長時間実行の評価タスクをスキップしたり再開したりするための
チェックポイント管理機能を提供します。

既存の ztb.utils.checkpoint の実装パターンを統合・再利用し、
圧縮・エラーハンドリング・ランダム状態復元の機能を提供します。

## 使用例

```python
from ztb.evaluation.walk_forward.checkpoint import CheckpointManager
from ztb.evaluation.unified_evaluation import UnifiedEvaluator

# チェックポイント作成（圧縮機能付き）
checkpoint_mgr = CheckpointManager(
    checkpoint_dir="./checkpoints",
    compress="zstd"  # 既存utilと共通の圧縮方式
)

# 既存チェックポイントから復元
evaluator = UnifiedEvaluator(
    config={"walk_forward_windows": [], "walk_forward_timesteps": 10000}
)
checkpoint_mgr.restore(evaluator, run_id="run_001")

# 評価実行
evaluation, performances, errors = evaluator.evaluate_walk_forward_details_from_df(
    df=df,
    windows=windows,
    model_name="walk_forward",
)

# チェックポイント保存
checkpoint_mgr.save(evaluator, run_id="run_001")

# 進捗確認
status = checkpoint_mgr.get_run_status("run_001")
print(f"Progress: {status['completed_windows']}/{status['total_windows']}")
```
"""

import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
try:
    from stable_baselines3 import SAC  # type: ignore
except Exception:  # pragma: no cover - defensive fallback for test environments
    SAC = None  # type: ignore

from ztb.utils.checkpoint import CheckpointManager as CoreCheckpointManager
from ztb.utils.error_utils import safe_operation
from ztb.utils.file_utils import safe_json_dump, safe_json_load
from ztb.utils.path_utils import ensure_dir
from .types import WindowPerformance

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Walk-Forward 評価のチェックポイント管理（ztb.utils の軽量アダプタ）

    ztb.utils.checkpoint.CheckpointManager をラップした、Walk-Forward特化版。
    核となる圧縮・メタデータ管理ロジックはマスター実装に委譲。

    ## 設計方針

    - **マスター実装**: ztb.utils.checkpoint.CheckpointManager
      → 圧縮（zlib/lz4/zstd）、メタデータ管理、差分チェックポイント
    
    - **アダプタ実装**: walk_forward/checkpoint.CheckpointManager (本クラス)
      → ウィンドウ管理、評価結果集約、ランメタデータ化
    
    重複コード排除により、保守性向上・バグ修正効率化を実現。

    ## 利用例

    ```python
    from ztb.evaluation.walk_forward.checkpoint import CheckpointManager
    from ztb.evaluation.unified_evaluation import EvaluationType, UnifiedEvaluator

    # Walk-Forward特化のチェックポイント作成
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir="./checkpoints",
        compress="zstd"  # ztb.utils.checkpoint との共通圧縮方式
    )

    # チェックポイント保存（圧縮実施）
    evaluator = UnifiedEvaluator(
        config={
            "walk_forward_windows": [],
            "walk_forward_timesteps": 10000,
            "walk_forward_checkpoint_dir": "./checkpoints",
        }
    )
    evaluator.evaluate_walk_forward_details_from_df(
        df=df,
        windows=windows,
        model_name="walk_forward",
    )
    # 内部で checkpoint_mgr.save() が呼ばれ、runtime_data が圧縮される

    # チェックポイント復元（自動解凍）
    checkpoint_mgr.restore(evaluator, run_id="run_001")

    # 進捗確認
    status = checkpoint_mgr.get_run_status("run_001")
    results = checkpoint_mgr.get_results_summary("run_001")
    ```

    Attributes:
        checkpoint_dir: チェックポイント保存ディレクトリ
        compress: 圧縮方式（'zlib'/'lz4'/'zstd'）。デフォルト 'zlib'
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        compress: str = "zlib",
    ) -> None:
        """初期化

        Args:
            checkpoint_dir: チェックポイント保存ディレクトリ
            compress: 圧縮方式（'zlib'/'lz4'/'zstd'）
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        ensure_dir(self.checkpoint_dir)
        self.compress = compress
        # コア実装インスタンスを保有（圧縮機能委譲用）
        self._core_manager = CoreCheckpointManager(
            save_dir=str(checkpoint_dir),
            compress=compress
        )

    def _compress_data(self, data: Dict[str, Any]) -> bytes:
        """データを圧縮（ztb.utils.checkpoint 委譲）

        ztb.utils.checkpoint.CheckpointManager の圧縮ロジックに委譲。
        DRY原則に基づき、マスター実装の圧縮機能を再利用。

        Args:
            data: 圧縮対象のDict

        Returns:
            bytes: 圧縮済みバイト列
        """
        pickled = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        # コア実装の圧縮機能を活用
        return self._core_manager._compress_data(pickled)

    def _resolve_evaluator(self, evaluator: Any) -> Any:
        if hasattr(evaluator, "results") and hasattr(evaluator, "models"):
            return evaluator

        inner = getattr(evaluator, "_last_walk_forward_evaluator", None)
        if inner is not None:
            return inner

        config = getattr(evaluator, "config", {}) or {}
        try:
            from ztb.evaluation.walk_forward.evaluator import WalkForwardModelEvaluator
        except Exception as e:
            raise ValueError("Cannot resolve walk-forward evaluator") from e

        wf_evaluator = WalkForwardModelEvaluator(
            env_factory=config.get("walk_forward_env_factory"),
            algorithm_factory=config.get("walk_forward_algorithm_factory"),
            checkpoint_dir=config.get("walk_forward_checkpoint_dir"),
        )
        setattr(evaluator, "_last_walk_forward_evaluator", wf_evaluator)
        return wf_evaluator

    def _decompress_data(self, compressed: bytes) -> Dict[str, Any]:
        """データを解凍（ztb.utils.checkpoint 委譲）

        ztb.utils.checkpoint.CheckpointManager の解凍ロジックに委譲。
        複数圧縮形式の自動検出も含む。

        Args:
            compressed: 圧縮済みバイト列

        Returns:
            Dict[str, Any]: 解凍後のデータ
        """
        try:
            # コア実装の解凍機能を活用
            decompressed = self._core_manager._decompress_data(compressed)
            return pickle.loads(decompressed)
        except Exception as e:
            logger.error(f"Failed to decompress data: {e}")
            raise ValueError(f"Failed to decompress checkpoint data: {e}") from e

    def save(
        self,
        evaluator: Any,
        run_id: str,
        window_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """評価状態をチェックポイント保存

        Args:
            evaluator: UnifiedEvaluator (推奨) または WalkForwardModelEvaluator (旧API)
            run_id: 実行ID（一意）
            window_ids: 保存対象ウィンドウID。デフォルトは全ウィンドウ

        Returns:
            Dict[str, Any]: 保存情報（ウィンドウ数、タイムスタンプなど）
        """
        run_dir = self.checkpoint_dir / run_id
        ensure_dir(run_dir)

        evaluator = self._resolve_evaluator(evaluator)

        # 対象ウィンドウを決定
        target_window_ids = (
            window_ids if window_ids is not None else list(evaluator.results.keys())
        )

        saved_windows = 0

        # 各ウィンドウごとにチェックポイント保存
        for window_id in target_window_ids:
            if window_id not in evaluator.results:
                logger.warning(
                    f"Window {window_id} not in results, skipping checkpoint"
                )
                continue

            def save_window_checkpoint():
                window_dir = run_dir / f"window_{window_id}"
                ensure_dir(window_dir)

                # 1. 窓メタデータ
                metadata = {
                    "window_id": window_id,
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed",
                }
                safe_json_dump(
                    metadata,
                    window_dir / "checkpoint_metadata.json",
                    indent=2
                )

                # 2. SAC モデル
                if window_id in evaluator.models:
                    model_path = window_dir / "model"
                    try:
                        evaluator.models[window_id].save(str(model_path))
                        logger.debug(f"Saved model for window {window_id}")
                    except Exception as e:
                        logger.warning(f"Failed to save model for window {window_id}: {e}")

                # 3. WindowPerformance 結果
                performance = evaluator.results[window_id]
                perf_data = {
                    "window_id": performance.window_id,
                    "val_roi": float(performance.val_roi),
                    "test_roi": float(performance.test_roi),
                    "val_final_balance": float(performance.val_final_balance),
                    "test_final_balance": float(performance.test_final_balance),
                    "sharpe_ratio": float(performance.sharpe_ratio),
                    "max_drawdown": float(performance.max_drawdown),
                    "win_rate": float(performance.win_rate),
                    "trades": int(performance.trades),
                }
                safe_json_dump(
                    perf_data,
                    window_dir / "window_results.json",
                    indent=2
                )

            # safe_operation を使用してエラー隔離
            safe_operation(
                save_window_checkpoint,
                operation_name=f"Save checkpoint for window {window_id}",
                default_result=None
            )
            saved_windows += 1
            logger.debug(f"Saved checkpoint for window {window_id}")

        # ランレベルのメタデータ（全体進捗）を圧縮保存
        def save_run_metadata():
            run_metadata = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "total_windows_completed": len(evaluator.results),
                "total_windows_failed": len(evaluator.errors),
                "saved_windows": saved_windows,
                "compress_method": self.compress,
            }
            safe_json_dump(
                run_metadata,
                run_dir / "run_metadata.json",
                indent=2
            )

            # ランタイムデータ（全体状態）を圧縮保存
            runtime_data = {
                "results": evaluator.results,
                "errors": {k: str(v) for k, v in evaluator.errors.items()},
                "models_count": len(evaluator.models),
                "timestamp": time.time(),
            }
            compressed = self._compress_data(runtime_data)
            runtime_path = run_dir / f"runtime_data.pkl"
            with open(runtime_path, "wb") as f:
                f.write(compressed)

        safe_operation(
            save_run_metadata,
            operation_name=f"Save run metadata for {run_id}",
            default_result=None
        )

        run_metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "total_windows_completed": len(evaluator.results),
            "total_windows_failed": len(evaluator.errors),
            "saved_windows": saved_windows,
        }

        logger.info(
            f"✓ Checkpoint saved: run_id={run_id}, "
            f"windows={saved_windows}, errors={len(evaluator.errors)}"
        )

        return run_metadata

    def restore(
        self,
        evaluator: Any,
        run_id: str,
        restore_models: bool = True,
    ) -> Dict[str, Any]:
        """チェックポイントから評価状態を復元

        Args:
            evaluator: UnifiedEvaluator (推奨) または WalkForwardModelEvaluator (旧API)
            run_id: 実行ID
            restore_models: Trueの場合、SACモデルも復元

        Returns:
            Dict[str, Any]: 復元情報（復元ウィンドウ数、エラーなど）

        Raises:
            FileNotFoundError: チェックポイントが見つからない場合
        """
        evaluator = self._resolve_evaluator(evaluator)
        run_dir = self.checkpoint_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {run_dir}")

        restored_windows = 0
        restored_errors = 0

        # ランタイムデータから全体状態を復元（あれば）
        def restore_runtime_data():
            runtime_path = run_dir / "runtime_data.pkl"
            if not runtime_path.exists():
                return

            try:
                with open(runtime_path, "rb") as f:
                    compressed = f.read()

                runtime_data = self._decompress_data(compressed)
                evaluator.results = runtime_data.get("results", {})
                # エラーは文字列として保存されているので、Exceptionオブジェクトに復元
                error_strs = runtime_data.get("errors", {})
                evaluator.errors = {
                    k: Exception(v) for k, v in error_strs.items()
                }
                logger.debug(
                    f"Restored {len(evaluator.results)} results "
                    f"and {len(evaluator.errors)} errors from runtime data"
                )
            except Exception as e:
                logger.warning(f"Failed to load runtime data: {e}")

        safe_operation(
            restore_runtime_data,
            operation_name=f"Restore runtime data for {run_id}",
            default_result=None
        )

        # 各ウィンドウごとにチェックポイント復元
        window_dirs = sorted(run_dir.glob("window_*"))
        for window_dir in window_dirs:
            def restore_window():
                # メタデータ読込
                metadata_path = window_dir / "checkpoint_metadata.json"
                if not metadata_path.exists():
                    logger.warning(f"Metadata not found: {window_dir}")
                    return

                metadata = safe_json_load(metadata_path)
                if metadata is None:
                    logger.warning(f"Failed to load metadata from {metadata_path}")
                    return

                window_id = metadata.get("window_id")

                # WindowPerformance 読込
                perf_path = window_dir / "window_results.json"
                if perf_path.exists():
                    perf_data = safe_json_load(perf_path)
                    if perf_data:
                        performance = WindowPerformance(
                            window_id=perf_data["window_id"],
                            val_roi=perf_data["val_roi"],
                            test_roi=perf_data["test_roi"],
                            val_final_balance=perf_data["val_final_balance"],
                            test_final_balance=perf_data["test_final_balance"],
                            sharpe_ratio=perf_data["sharpe_ratio"],
                            max_drawdown=perf_data["max_drawdown"],
                            win_rate=perf_data["win_rate"],
                            trades=perf_data["trades"],
                        )
                        evaluator.results[window_id] = performance

                # SAC モデル読込（オプション）
                if restore_models:
                    model_path = window_dir / "model"
                    if model_path.with_suffix(".zip").exists():
                        try:
                            model = SAC.load(str(model_path))
                            evaluator.models[window_id] = model
                            logger.debug(f"Loaded model for window {window_id}")
                        except Exception as e:
                            logger.warning(f"Failed to load model for window {window_id}: {e}")

            safe_operation(
                restore_window,
                operation_name=f"Restore checkpoint from {window_dir}",
                default_result=None
            )
            restored_windows += 1

        logger.info(
            f"✓ Checkpoint restored: run_id={run_id}, "
            f"windows={restored_windows}, errors={restored_errors}"
        )

        return {
            "run_id": run_id,
            "restored_windows": restored_windows,
            "restore_errors": restored_errors,
        }

    def get_run_status(self, run_id: str) -> Dict[str, Any]:
        """実行ステータスを取得

        Args:
            run_id: 実行ID

        Returns:
            Dict[str, Any]: ステータス情報

        Raises:
            FileNotFoundError: チェックポイントが見つからない場合
        """
        run_dir = self.checkpoint_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {run_dir}")

        # メタデータ読込
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.exists():
            return {
                "run_id": run_id,
                "status": "no_metadata",
                "completed_windows": 0,
                "failed_windows": 0,
                "total_windows": 0,
            }

        metadata = safe_json_load(metadata_path)
        if metadata is None:
            return {
                "run_id": run_id,
                "status": "metadata_error",
                "completed_windows": 0,
                "failed_windows": 0,
                "total_windows": 0,
            }

        total = (
            metadata["total_windows_completed"] + metadata["total_windows_failed"]
        )
        return {
            "run_id": run_id,
            "status": "in_progress" if total > 0 else "empty",
            "completed_windows": metadata["total_windows_completed"],
            "failed_windows": metadata["total_windows_failed"],
            "total_windows": total,
            "progress_pct": (
                100.0 * metadata["total_windows_completed"] / total
                if total > 0
                else 0.0
            ),
            "timestamp": metadata.get("timestamp", "unknown"),
        }

    def list_runs(self) -> List[str]:
        """保存されたすべての実行IDを列挙

        Returns:
            List[str]: 実行IDのリスト
        """
        if not self.checkpoint_dir.exists():
            return []

        run_ids = []

        def collect_run_ids():
            for d in self.checkpoint_dir.iterdir():
                if d.is_dir():
                    run_ids.append(d.name)

        safe_operation(
            collect_run_ids,
            operation_name="Collect run IDs",
            default_result=None
        )

        return sorted(run_ids)

    def delete_run(self, run_id: str) -> bool:
        """実行チェックポイントを削除

        Args:
            run_id: 実行ID

        Returns:
            bool: 削除成功時 True
        """
        run_dir = self.checkpoint_dir / run_id
        if not run_dir.exists():
            logger.warning(f"Checkpoint not found: {run_dir}")
            return False

        def delete_run_dir():
            import shutil
            shutil.rmtree(run_dir)

        result = safe_operation(
            delete_run_dir,
            operation_name=f"Delete checkpoint for {run_id}",
            default_result=False
        )

        if result is not False:
            logger.info(f"✓ Deleted checkpoint: {run_id}")

        return result is not False

    def get_completed_windows(self, run_id: str) -> List[int]:
        """完了済みウィンドウIDを取得

        Args:
            run_id: 実行ID

        Returns:
            List[int]: 完了ウィンドウ ID リスト

        Raises:
            FileNotFoundError: チェックポイントが見つからない場合
        """
        run_dir = self.checkpoint_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {run_dir}")

        window_ids = []

        def collect_window_ids():
            for window_dir in run_dir.glob("window_*"):
                metadata_path = window_dir / "checkpoint_metadata.json"
                if metadata_path.exists():
                    metadata = safe_json_load(metadata_path)
                    if metadata:
                        window_ids.append(metadata["window_id"])

        safe_operation(
            collect_window_ids,
            operation_name=f"Collect completed window IDs from {run_id}",
            default_result=None
        )

        return sorted(window_ids)

    def get_results_summary(self, run_id: str) -> Dict[str, Any]:
        """チェックポイントから結果サマリーを取得

        Args:
            run_id: 実行ID

        Returns:
            Dict[str, Any]: 結果統計

        Raises:
            FileNotFoundError: チェックポイントが見つない場合
        """
        run_dir = self.checkpoint_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {run_dir}")

        val_rois = []
        test_rois = []
        sharpes = []

        def collect_results():
            for window_dir in run_dir.glob("window_*"):
                perf_path = window_dir / "window_results.json"
                if perf_path.exists():
                    perf_data = safe_json_load(perf_path)
                    if perf_data:
                        val_rois.append(perf_data["val_roi"])
                        test_rois.append(perf_data["test_roi"])
                        sharpes.append(perf_data["sharpe_ratio"])

        safe_operation(
            collect_results,
            operation_name=f"Collect results from {run_id}",
            default_result=None
        )

        return {
            "total_windows": len(val_rois),
            "avg_val_roi": float(np.mean(val_rois)) if val_rois else 0.0,
            "avg_test_roi": float(np.mean(test_rois)) if test_rois else 0.0,
            "std_test_roi": float(np.std(test_rois)) if test_rois else 0.0,
            "avg_sharpe": float(np.mean(sharpes)) if sharpes else 0.0,
            "std_sharpe": float(np.std(sharpes)) if sharpes else 0.0,
        }
