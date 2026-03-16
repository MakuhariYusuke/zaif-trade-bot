#!/usr/bin/env python3
"""SAC 実装依存グラフ生成.

017# P1: SAC 関連モジュール間の import 依存を自動解析し、
安全な deprecation 計画の土台を提供する.

Usage:
  python scripts/v460/sac_dependency_graph.py
  python scripts/v460/sac_dependency_graph.py --output results/v460/sac_deps.md
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# SAC 関連モジュール — 015# / 017# で特定した主要ファイル
SAC_MODULES: dict[str, str] = {
    "ztb/training/unified_trainer/algorithms/sac_trainer.py": "#1 統一 SACTrainer",
    "ztb/training/trainers/sac_trainer.py": "#2 旧 SACAlgorithmTrainer",
    "ztb/training/sac_trainer.py": "#3 ファサード SACTrainer",
    "ztb/training/sac.py": "#4 SACSuite CLI",
    "ztb/training/adaptive_sac_core.py": "#5 AdaptiveSACCore",
    "ztb/training/v435/train_sac_v435.py": "#6 v435 スタブ",
    "ztb/training/algorithms/sac/sac_algorithm.py": "#7 SACAlgorithm",
    "ztb/training/online_learning_engine.py": "OnlineLearningEngine",
    "ztb/training/sac_v430_training_optimizations.py": "v430 最適化ユーティリティ",
    "ztb/trading/live_trader/action_prediction.py": "ActionPrediction (推論)",
    "ztb/trading/live_trader/model_loading.py": "ModelLoading",
}


def _extract_imports(filepath: Path) -> list[str]:
    """AST を使って import 先モジュールを抽出."""
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
    return modules


def _find_importers(target_module: str, search_dirs: list[Path]) -> list[str]:
    """target_module を import しているファイルを検索."""
    importers: list[str] = []
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for py_file in search_dir.rglob("*.py"):
            rel = py_file.relative_to(_PROJECT_ROOT).as_posix()
            imports = _extract_imports(py_file)
            for imp in imports:
                if target_module in imp:
                    importers.append(rel)
                    break
    return importers


def _path_to_module(path: str) -> str:
    """ファイルパスを Python モジュール名に変換."""
    mod = path.replace("/", ".").replace("\\", ".")
    if mod.endswith(".py"):
        mod = mod[:-3]
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    return mod


def generate_dependency_report() -> str:
    """SAC 依存グラフレポートを生成."""
    search_dirs = [
        _PROJECT_ROOT / "ztb",
        _PROJECT_ROOT / "scripts",
        _PROJECT_ROOT / "tests",
    ]

    # .venv, __pycache__, node_modules 等を除外
    _SKIP_DIRS = {"__pycache__", ".venv", "node_modules", ".git", "cache", ".mypy_cache"}

    def _rglob_py(base: Path) -> list[Path]:
        """__pycache__ 等を除外した .py ファイル列挙."""
        result: list[Path] = []
        if not base.exists():
            return result
        for item in base.iterdir():
            if item.is_dir():
                if item.name in _SKIP_DIRS:
                    continue
                result.extend(_rglob_py(item))
            elif item.suffix == ".py":
                result.append(item)
        return result

    lines: list[str] = []
    lines.append("# SAC 実装依存グラフ")
    lines.append("")
    lines.append(f"自動生成 | 対象: {len(SAC_MODULES)} モジュール")
    lines.append("")

    # 各 SAC モジュールについて依存元を探索
    dep_graph: dict[str, list[str]] = {}
    all_importers: dict[str, list[str]] = defaultdict(list)

    for rel_path, label in SAC_MODULES.items():
        module_name = _path_to_module(rel_path)
        # 短い名前でも検索 (e.g. "sac_trainer" in "from ztb.training.sac_trainer import ...")
        short_names = set()
        short_names.add(module_name)
        parts = module_name.split(".")
        if len(parts) >= 2:
            short_names.add(".".join(parts[-2:]))

        importers: list[str] = []
        for search_dir in search_dirs:
            for py_file in _rglob_py(search_dir):
                file_rel = py_file.relative_to(_PROJECT_ROOT).as_posix()
                # 自分自身はスキップ
                if file_rel == rel_path:
                    continue
                file_imports = _extract_imports(py_file)
                for imp in file_imports:
                    if any(sn in imp for sn in short_names):
                        importers.append(file_rel)
                        all_importers[file_rel].append(rel_path)
                        break

        dep_graph[rel_path] = sorted(set(importers))

    # ── レポート出力 ──
    lines.append("## 依存元マトリクス")
    lines.append("")
    lines.append("| SAC モジュール | ラベル | 依存元数 | 依存元ファイル |")
    lines.append("|---------------|--------|---------|--------------|")

    for rel_path, label in SAC_MODULES.items():
        importers = dep_graph.get(rel_path, [])
        count = len(importers)
        if importers:
            files = ", ".join(f"`{f}`" for f in importers[:5])
            if count > 5:
                files += f" ... (+{count - 5})"
        else:
            files = "(なし)"
        lines.append(f"| `{rel_path}` | {label} | {count} | {files} |")

    lines.append("")

    # ── Deprecation リスク評価 ──
    lines.append("## Deprecation リスク評価")
    lines.append("")
    lines.append("| モジュール | ラベル | 依存元数 | 安全に削除可能か |")
    lines.append("|-----------|--------|---------|----------------|")

    for rel_path, label in SAC_MODULES.items():
        importers = dep_graph.get(rel_path, [])
        count = len(importers)
        if count == 0:
            safety = "✅ 即削除可能"
        elif count <= 2:
            safety = "⚠️ 参照置換後に削除"
        else:
            safety = "❌ 段階的移行が必要"
        lines.append(f"| `{rel_path}` | {label} | {count} | {safety} |")

    lines.append("")

    # ── 相互依存検出 ──
    lines.append("## SAC モジュール間の相互依存")
    lines.append("")
    sac_paths = set(SAC_MODULES.keys())
    cross_deps: list[str] = []
    for rel_path in SAC_MODULES:
        importers = dep_graph.get(rel_path, [])
        for imp in importers:
            if imp in sac_paths:
                cross_deps.append(f"  `{imp}` → `{rel_path}`")

    if cross_deps:
        lines.append("```")
        for dep in sorted(set(cross_deps)):
            lines.append(dep)
        lines.append("```")
    else:
        lines.append("(相互依存なし)")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAC Dependency Graph Generator")
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: stdout)",
    )
    args = parser.parse_args()

    report = generate_dependency_report()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Report saved: {out_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
