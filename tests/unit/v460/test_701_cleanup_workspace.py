from __future__ import annotations

from pathlib import Path

from scripts.v460.tools.cleanup_workspace import (
    CleanupSummary,
    discover_cleanup_candidates,
    execute_cleanup,
)


def _make_workspace(root: Path) -> None:
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "data/temp").mkdir(parents=True, exist_ok=True)


class Test701CleanupWorkspace:
    def test_dry_run_does_not_delete(self, tmp_path: Path, capsys) -> None:
        _make_workspace(tmp_path)
        target = tmp_path / "config" / "ab_search_temp_deadbeef.json"
        target.write_text("{}", encoding="utf-8")

        summary = discover_cleanup_candidates(repo_root=tmp_path, tracked_paths=frozenset())
        execute_cleanup(summary, execute=False)

        assert target.exists()
        captured = capsys.readouterr().out
        assert "[DRY-RUN]" in captured

    def test_execute_removes_ab_search_temp(self, tmp_path: Path) -> None:
        _make_workspace(tmp_path)
        target = tmp_path / "config" / "ab_search_temp_deadbeef.json"
        target.write_text("{}", encoding="utf-8")

        summary = discover_cleanup_candidates(repo_root=tmp_path, tracked_paths=frozenset())
        execute_cleanup(summary, execute=True, sleep_fn=lambda _sec: None)

        assert not target.exists()

    def test_tracked_files_never_deleted(self, tmp_path: Path) -> None:
        _make_workspace(tmp_path)
        target = tmp_path / "config" / "ab_search_temp_keep.json"
        target.write_text("{}", encoding="utf-8")

        summary = discover_cleanup_candidates(
            repo_root=tmp_path,
            tracked_paths=frozenset({"config/ab_search_temp_keep.json"}),
        )

        assert summary == CleanupSummary(candidates=tuple())
        assert target.exists()

    def test_empty_workspace_no_error(self, tmp_path: Path, capsys) -> None:
        _make_workspace(tmp_path)
        summary = discover_cleanup_candidates(repo_root=tmp_path, tracked_paths=frozenset())
        execute_cleanup(summary, execute=False)

        captured = capsys.readouterr().out
        assert "No cleanup targets found" in captured

    def test_summary_output_format(self, tmp_path: Path, capsys) -> None:
        _make_workspace(tmp_path)
        for idx in range(2):
            (tmp_path / "config" / f"ab_search_temp_{idx:08x}.json").write_text("{}", encoding="utf-8")

        cache_dir = tmp_path / "data/temp/.mypy_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "x.json").write_text("{}", encoding="utf-8")

        summary = discover_cleanup_candidates(repo_root=tmp_path, tracked_paths=frozenset())
        execute_cleanup(summary, execute=False)

        captured = capsys.readouterr().out
        assert "config/ab_search_temp_*.json" in captured
        assert "data/temp/.mypy_cache/" in captured
        assert "Total:" in captured
