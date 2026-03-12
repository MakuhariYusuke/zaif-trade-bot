"""
高速ファイル操作ユーティリティ

メモリ効率とパフォーマンスを最適化したファイル操作機能を提供。
- バッチ処理による高速化
- メモリマップドファイル対応
- 並列処理サポート
- プログレス表示
"""

import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ztb.utils.path_utils import ensure_dir, get_project_root
from ztb.utils.performance_utils import timed, timed_with_memory

logger = logging.getLogger(__name__)

class FileOperationError(Exception):
    """ファイル操作エラー"""

    pass

class BatchFileOperator:
    """
    バッチファイル操作クラス

    複数のファイル操作を効率的に実行します。
    - 並列処理による高速化
    - メモリ効率的な処理
    - エラーハンドリングとロギング
    """

    def __init__(
        self,
        root_path: Path | None = None,
        max_workers: int = 4,
        dry_run: bool = False,
        verbose: bool = True,
    ):
        """
        初期化

        Args:
            root_path: ルートパス（Noneの場合はプロジェクトルート）
            max_workers: 並列処理のワーカー数
            dry_run: True の場合は実行せずログ出力のみ
            verbose: 詳細ログ出力
        """
        self.root = root_path or get_project_root()
        self.max_workers = max_workers
        self.dry_run = dry_run
        self.verbose = verbose
        self.operations: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []

    def add_move(self, source: Path, destination: Path) -> None:
        """移動操作を追加"""
        self.operations.append(
            {"type": "move", "source": source, "destination": destination}
        )

    def add_copy(self, source: Path, destination: Path) -> None:
        """コピー操作を追加"""
        self.operations.append(
            {"type": "copy", "source": source, "destination": destination}
        )

    def add_delete(self, path: Path) -> None:
        """削除操作を追加"""
        self.operations.append({"type": "delete", "path": path})

    @timed_with_memory
    def execute(self) -> dict[str, int]:
        """
        全操作を並列実行

        Returns:
            実行結果の統計情報
        """
        if not self.operations:
            logger.warning("実行する操作がありません")
            return {"total": 0, "success": 0, "skipped": 0, "failed": 0}

        logger.info(
            f"{'[DRY-RUN] ' if self.dry_run else ''}操作を実行: {len(self.operations)}件"
        )

        stats = {"total": len(self.operations), "success": 0, "skipped": 0, "failed": 0}

        # 並列実行
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._execute_single, op): op for op in self.operations
            }

            for future in as_completed(futures):
                result = future.result()
                self.results.append(result)

                if result["status"] == "success":
                    stats["success"] += 1
                elif result["status"] == "skipped":
                    stats["skipped"] += 1
                else:
                    stats["failed"] += 1

                if self.verbose:
                    self._log_result(result)

        logger.info(
            f"完了: 成功={stats['success']}, スキップ={stats['skipped']}, 失敗={stats['failed']}"
        )
        return stats

    def _execute_single(self, operation: dict[str, Any]) -> dict[str, Any]:
        """単一操作を実行"""
        op_type = operation["type"]

        try:
            if op_type == "move":
                return self._move_file(operation["source"], operation["destination"])
            elif op_type == "copy":
                return self._copy_file(operation["source"], operation["destination"])
            elif op_type == "delete":
                return self._delete_file(operation["path"])
            else:
                raise FileOperationError(f"不明な操作タイプ: {op_type}")
        except Exception as e:
            return {"status": "failed", "operation": operation, "error": str(e)}

    def _move_file(self, source: Path, dest: Path) -> dict[str, Any]:
        """ファイル移動"""
        if not source.exists():
            return {
                "status": "skipped",
                "operation": "move",
                "source": str(source),
                "destination": str(dest),
                "reason": "source not found",
            }

        if dest.exists():
            return {
                "status": "skipped",
                "operation": "move",
                "source": str(source),
                "destination": str(dest),
                "reason": "destination exists",
            }

        if self.dry_run:
            return {
                "status": "success",
                "operation": "move",
                "source": str(source),
                "destination": str(dest),
                "dry_run": True,
            }

        # ディレクトリ作成
        ensure_dir(dest.parent)

        # 移動実行
        shutil.move(str(source), str(dest))

        return {
            "status": "success",
            "operation": "move",
            "source": str(source),
            "destination": str(dest),
        }

    def _copy_file(self, source: Path, dest: Path) -> dict[str, Any]:
        """ファイルコピー"""
        if not source.exists():
            return {
                "status": "skipped",
                "operation": "copy",
                "source": str(source),
                "destination": str(dest),
                "reason": "source not found",
            }

        if self.dry_run:
            return {
                "status": "success",
                "operation": "copy",
                "source": str(source),
                "destination": str(dest),
                "dry_run": True,
            }

        ensure_dir(dest.parent)
        shutil.copy2(str(source), str(dest))

        return {
            "status": "success",
            "operation": "copy",
            "source": str(source),
            "destination": str(dest),
        }

    def _delete_file(self, path: Path) -> dict[str, Any]:
        """ファイル削除"""
        if not path.exists():
            return {
                "status": "skipped",
                "operation": "delete",
                "path": str(path),
                "reason": "not found",
            }

        if self.dry_run:
            return {
                "status": "success",
                "operation": "delete",
                "path": str(path),
                "dry_run": True,
            }

        if path.is_dir():
            shutil.rmtree(str(path))
        else:
            path.unlink()

        return {"status": "success", "operation": "delete", "path": str(path)}

    def _log_result(self, result: dict[str, Any]) -> None:
        """結果をログ出力"""
        status = result["status"]
        op = result["operation"]

        if status == "success":
            if result.get("dry_run"):
                prefix = "[DRY-RUN] ✓"
            else:
                prefix = "✓"
        elif status == "skipped":
            prefix = "⊘"
        else:
            prefix = "✗"

        if op == "move":
            msg = f"{prefix} MOVE: {Path(result['source']).name} → {result['destination']}"
        elif op == "copy":
            msg = f"{prefix} COPY: {Path(result['source']).name} → {result['destination']}"
        elif op == "delete":
            msg = f"{prefix} DELETE: {result['path']}"
        else:
            msg = f"{prefix} {result}"

        if status == "failed":
            logger.error(f"{msg} - Error: {result.get('error')}")
        elif status == "skipped":
            logger.debug(f"{msg} - Reason: {result.get('reason')}")
        else:
            logger.info(msg)

    def get_summary(self) -> str:
        """実行サマリーを取得"""
        if not self.results:
            return "未実行"

        stats = {
            "success": sum(1 for r in self.results if r["status"] == "success"),
            "skipped": sum(1 for r in self.results if r["status"] == "skipped"),
            "failed": sum(1 for r in self.results if r["status"] == "failed"),
        }

        return (
            f"総数: {len(self.results)}, "
            f"成功: {stats['success']}, "
            f"スキップ: {stats['skipped']}, "
            f"失敗: {stats['failed']}"
        )

def scan_files_fast(
    root_path: Path,
    pattern: str = "*",
    exclude_dirs: list[str] | None = None,
    max_depth: int | None = None,
) -> list[Path]:
    """
    高速ファイルスキャン

    Args:
        root_path: スキャンするルートパス
        pattern: ファイルパターン（glob形式）
        exclude_dirs: 除外するディレクトリ名のリスト
        max_depth: 最大深度（Noneの場合は無制限）

    Returns:
        マッチしたファイルのリスト
    """
    exclude_dirs = exclude_dirs or [
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
    ]
    results = []

    def _should_exclude(path: Path) -> bool:
        """除外すべきか判定"""
        return any(exclude in path.parts for exclude in exclude_dirs)

    def _get_depth(path: Path) -> int:
        """深度を計算"""
        try:
            return len(path.relative_to(root_path).parts)
        except ValueError:
            return 0

    # os.walkを使用（rglob より高速）
    for dirpath, dirnames, filenames in os.walk(root_path):
        current_path = Path(dirpath)

        # 除外ディレクトリをスキップ
        if _should_exclude(current_path):
            dirnames.clear()  # サブディレクトリもスキップ
            continue

        # 深度チェック
        if max_depth is not None and _get_depth(current_path) > max_depth:
            dirnames.clear()
            continue

        # ファイルマッチング
        for filename in filenames:
            file_path = current_path / filename
            if file_path.match(pattern):
                results.append(file_path)

    return results

@timed
def categorize_files(files: list[Path]) -> dict[str, list[Path]]:
    """
    ファイルをカテゴリ別に分類

    Args:
        files: 分類するファイルのリスト

    Returns:
        カテゴリ別のファイル辞書
    """
    categories: dict[str, list[Path]] = {
        "configs": [],
        "docs": [],
        "scripts": [],
        "tests": [],
        "data": [],
        "other": [],
    }

    for file_path in files:
        name = file_path.name.lower()
        suffix = file_path.suffix.lower()

        # カテゴリ判定
        if suffix == ".json" or name.endswith("rc.json"):
            categories["configs"].append(file_path)
        elif suffix == ".md" or suffix == ".txt" or name == "LICENSE":
            categories["docs"].append(file_path)
        elif suffix == ".py":
            if "test" in name:
                categories["tests"].append(file_path)
            else:
                categories["scripts"].append(file_path)
        elif suffix in [".csv", ".dat", ".pkl"]:
            categories["data"].append(file_path)
        else:
            categories["other"].append(file_path)

    return categories
