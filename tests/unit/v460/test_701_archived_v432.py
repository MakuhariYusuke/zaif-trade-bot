from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


class Test701ArchivedV432:
    def test_v432_json_not_in_ztb_analysis(self) -> None:
        assert list((REPO_ROOT / "ztb/analysis").glob("sac_v432_*.json")) == []

    def test_v432_json_in_archived(self) -> None:
        archived_paths = sorted((REPO_ROOT / "archived/analysis").glob("sac_v432_*.json"))
        assert len(archived_paths) == 7

    def test_no_code_references_old_ztb_analysis_location(self) -> None:
        roots = (
            REPO_ROOT / "ztb",
            REPO_ROOT / "scripts",
        )
        matches: list[Path] = []
        for root in roots:
            for file_path in root.rglob("*.py"):
                contents = file_path.read_text(encoding="utf-8")
                if "ztb/analysis/sac_v432_" in contents:
                    matches.append(file_path.relative_to(REPO_ROOT))
        assert matches == []
